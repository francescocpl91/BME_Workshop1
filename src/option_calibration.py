"""Offline adapters and auditable output for OIS/forward/individual-IV calculation.

This is deliberately separate from data_cleaning, which inventories observations
without interpreting option prices. It consumes the existing processed schema,
verifies input hashes, and never constructs a downloader or contacts a provider.
Live and historical observations share the same pricing functions, but retain
separate timing, underlying-price and dividend-availability rules.
"""
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone, timedelta
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
import re

import exchange_calendars as xcals
import hashlib
import io
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .ois_bootstrap import bootstrap_ois
from .option_math import black_price, black_vega, equity_forward, implied_volatility
from .storage import write_json

ENGINE_VERSION = 3
SECONDS_PER_YEAR = 365.0 * 86400


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_json_key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite_json_value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique, parse_constant=invalid)


def _number(value):
    # CSV fields are strings; blank, bool and provider NaN markers are missing.
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _true(value):
    return value is True or value == 'True'


def _stamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError("missing_observation_timestamp")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('timestamp_without_timezone')
    return result


def _inside(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('input_path_outside_declared_directory')
    return target


def load_calibration_config(path):
    """Validate user policies before reading data or starting calculations.

    Configuration mistakes are expected input errors: always expose them as
    ValueError so run.py can report unavailable calibration and continue the
    independent statistics/MC stages. Do not broadly catch implementation
    exceptions in run.py; genuine programming errors must remain visible.
    """
    cfg = _read_json(path)
    allowed = {'schema_version', 'valuation_timezone',
               'curve_conventions', 'live', 'historical', 'expiry', 'solver',
               'manual_carry_inputs', 'instructions'}
    if not isinstance(cfg, dict) or set(cfg) != allowed or type(cfg['schema_version']) is not int or cfg['schema_version'] != 1:
        raise ValueError('invalid_calibration_schema_or_keys')
    schemas = {
        'live': {'maximum_model_age_seconds','maximum_relative_spread','capture_span_warning_minutes','allow_broker_dividend_pv'},
        'historical': {'price_priority','cutoff_local_time','maximum_bar_age_minutes','allow_daily_close_approximation','underlying_provider_priority'},
        'expiry': {'timezone','times_by_settlement','instrument_overrides'},
        'solver': {'maximum_volatility','minimum_time_value_fraction_of_spot','minimum_vega_fraction_of_spot'}
    }
    for section, keys in schemas.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) != keys:
            raise ValueError('invalid_policy_keys: ' + section)
    # Validate nested shapes before .values(), key lookup, or hashing. JSON
    # can otherwise turn a harmless user typo into AttributeError/KeyError.
    def text(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('invalid_text_policy: ' + field)

    def zone(value, field):
        text(value, field)
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError('invalid_timezone_policy: ' + field) from exc

    def local_time(value, field):
        text(value, field)
        try:
            parsed = time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError('invalid_local_time_policy: ' + field) from exc
        # The separate IANA timezone determines UTC/DST; an embedded UTC
        # offset would introduce a second, potentially conflicting convention.
        if parsed.tzinfo is not None:
            raise ValueError('local_time_must_not_have_offset: ' + field)

    def expiry_policy(policy, field):
        if 'timezone' in policy:
            zone(policy['timezone'], field + '.timezone')
        if 'times_by_settlement' in policy:
            times = policy['times_by_settlement']
            if not isinstance(times, dict) or not times:
                raise ValueError('invalid_settlement_times: ' + field)
            for name, value in times.items():
                text(name, field + '.settlement')
                local_time(value, field + '.times_by_settlement.' + name)

    overrides = cfg['expiry']['instrument_overrides']
    if not isinstance(overrides, dict):
        raise ValueError('invalid_expiry_overrides')
    expiry_policy(cfg['expiry'], 'expiry')
    for underlying, override in overrides.items():
        text(underlying, 'expiry.instrument_overrides.underlying')
        if not isinstance(override, dict) or set(override) - {'timezone', 'times_by_settlement'}:
            raise ValueError('invalid_expiry_override: ' + underlying)
        expiry_policy(override, 'expiry.instrument_overrides.' + underlying)
    zone(cfg['valuation_timezone'], 'valuation_timezone')
    local_time(cfg['historical']['cutoff_local_time'], 'historical.cutoff_local_time')
    if not isinstance(cfg['curve_conventions'], dict) or not isinstance(cfg['instructions'], dict):
        raise ValueError('invalid_curve_conventions_or_instructions')
    for currency, convention in cfg['curve_conventions'].items():
        text(currency, 'curve_conventions.currency')
        if not isinstance(convention, dict):
            raise ValueError('invalid_currency_conventions: ' + currency)
    # Currency-specific convention support is still checked by bootstrap_ois;
    # an unavailable currency curve should not prevent independent currencies.
    for obj, key in [(cfg['live'], 'maximum_model_age_seconds'),
                     (cfg['live'], 'maximum_relative_spread'),
                     (cfg['live'], 'capture_span_warning_minutes'),
                     (cfg['historical'], 'maximum_bar_age_minutes'),
                     *[(cfg['solver'], key) for key in cfg['solver']]]:
        if type(obj[key]) not in (float, int) or not math.isfinite(obj[key]) or obj[key] <= 0:
            raise ValueError('invalid_positive_policy: ' + key)
    if type(cfg['live']['allow_broker_dividend_pv']) is not bool or type(cfg['historical']['allow_daily_close_approximation']) is not bool:
        raise ValueError('invalid_boolean_policy')
    if not isinstance(cfg['historical']['price_priority'], list) or not cfg['historical']['price_priority'] or any(x not in ('MIDPOINT','TRADES') for x in cfg['historical']['price_priority']):
        raise ValueError('unsupported_historical_price_priority')
    providers = cfg['historical']['underlying_provider_priority']
    if not isinstance(providers, list) or not providers:
        raise ValueError('invalid_underlying_provider_priority')
    for provider in providers:
        text(provider, 'historical.underlying_provider_priority')
    if not isinstance(cfg['manual_carry_inputs'], list):
        raise ValueError('invalid_manual_carry_inputs')
    seen = set()
    identity_keys = ('underlying','currency','valuation_date','expiration')
    for entry in cfg['manual_carry_inputs']:
        if not isinstance(entry, dict) or not set(identity_keys).issubset(entry):
            raise ValueError('missing_manual_carry_identity_fields')
        for key in identity_keys:
            text(entry[key], 'manual_carry_inputs.' + key)
        identity = tuple(entry[k] for k in identity_keys)
        date.fromisoformat(identity[2]); date.fromisoformat(identity[3])
        if identity in seen or not isinstance(entry.get('source'), str) or not entry['source'].strip() or (('forward' in entry) == ('dividend_pv' in entry)):
            raise ValueError('ambiguous_or_unsourced_manual_carry')
        seen.add(identity)
        field = 'forward' if 'forward' in entry else 'dividend_pv'
        value = entry[field]
        if type(value) not in (float,int) or not math.isfinite(value) or value < 0 or (field == 'forward' and value == 0):
            raise ValueError('invalid_manual_carry_value')
    return cfg


def load_processed_inputs(folder):
    """Verify every consumed table against its existing processing manifest.

    We deliberately read selected quotes rather than option_iv_candidates: the
    latter requires broker IV, which must NOT be required for our own inversion.
    Contracts for historical bars are recovered from the original resolved JSON.
    """
    folder = Path(folder).resolve()
    manifest_path = folder / 'manifest.json'
    manifest = _read_json(manifest_path)
    if not manifest.get('complete'):
        raise ValueError('processed_input_not_complete')
    hashes = {'processed_manifest': _hash(manifest_path.read_bytes())}
    tables = {}
    for name in ('selected_option_quotes','historical_option_bars','provider_prices','ois_quotes'):
        record = manifest['files'][name]
        raw = _inside(folder, record['file']).read_bytes()
        if _hash(raw) != record['sha256']:
            raise ValueError('processed_checksum_mismatch: ' + name)
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        hashes[name] = _hash(raw)
    archive = Path(manifest['archive']).resolve()
    raw = (archive / 'resolved_inputs.json').read_bytes()
    if _hash(raw) != manifest['archive_input_hashes']['resolved_inputs.json']:
        raise ValueError('archive_contract_checksum_mismatch')
    hashes['resolved_inputs'] = _hash(raw)
    resolved = json.loads(raw.decode('utf-8-sig'))
    contracts = {}
    for request in resolved['requests']:
        if request.get('contract'):
            if request['id'] in contracts and contracts[request['id']] != request['contract']:
                raise ValueError('ambiguous_archive_contract')
            contracts[request['id']] = request['contract']
    return manifest, tables, contracts, hashes


def build_curves(rows, cfg):
    """Date mismatches stop the affected currency before any bootstrap occurs."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['currency']].append(row)
    curves, nodes, repricing, statuses = {}, [], [], []
    for currency, group in sorted(groups.items()):
        try:
            if {r['date'] for r in group} != {cfg['valuation_date']}:
                raise ValueError('curve_date_mismatch')
            if any(not _true(r['eligible']) for r in group):
                raise ValueError('ineligible_ois_observation')
            identities = {(r['curve_id'],r['overnight_index'],r['quote_type'],r['rate_unit']) for r in group}
            if len(identities) != 1:
                raise ValueError('ambiguous_curve_identity')
            _, index, quote_type, unit = identities.pop()
            curve_input = dict(as_of_date=cfg['valuation_date'], currency=currency,
                overnight_index=index, quote_type=quote_type, rate_unit=unit,
                points=[dict(tenor=r['tenor'],rate_percent=_number(r['rate_percent'])) for r in group])
            if currency not in cfg['curve_conventions']:
                raise ValueError('missing_currency_conventions')
            curve, ns, rs = bootstrap_ois(curve_input, cfg['curve_conventions'][currency])
            curves[currency] = curve
            nodes.extend(dict(currency=currency,as_of_date=cfg['valuation_date'],**r) for r in ns)
            repricing.extend(dict(currency=currency,as_of_date=cfg['valuation_date'],**r) for r in rs)
            statuses.append(dict(currency=currency,status='bootstrapped',
                                 convention_status=curve.conventions['status']))
        except (ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
            statuses.append(dict(currency=currency,status='insufficient_data',reason=str(exc)))
    return curves, nodes, repricing, statuses


def _mapping(value):
    """Decode an optional JSON object from a processed CSV without guessing."""
    value = json.loads(value) if isinstance(value, str) and value else value
    if value in (None, ''):
        return {}
    if not isinstance(value, dict):
        raise ValueError('invalid_contract_metadata')
    return value


def _contract(row, contracts):
    # Configured metadata defines the intended contract; resolved IBKR metadata
    # independently verifies it in calculate_option, including direct callers.
    result = dict(_mapping(row.get('contract')) or contracts.get(row['instrument_id'], {}))
    for key in ('underlying','expiration','strike','option_type','exercise_style','settlement','settlement_time','multiplier'):
        if row.get(key) not in (None,''):
            if key in result and str(result[key]) != str(row[key]):
                numeric_match = key in ('strike', 'multiplier') and _number(result[key]) == _number(row[key])
                if not numeric_match:
                    raise ValueError('conflicting_configured_contract: ' + key)
            result[key] = row[key]
    return result


def _validate_contract_identity(row, contract):
    """Reject conflicting broker identity, without confusing last trade and expiry.

    IBKR's lastTradeDateOrContractMonth is NOT a legal expiry check for AM
    contracts. Prefer realExpirationDate and independently check OCC symbols
    when supplied. Missing broker fields remain explicitly unverified.
    """
    resolved = _mapping(row.get('resolved_contract'))
    details = _mapping(row.get('contract_dates'))
    checks = []
    for source, target in (('symbol','underlying'), ('right','option_type'),
                           ('strike','strike'), ('multiplier','multiplier'), ('currency','currency')):
        actual = resolved.get(source)
        if actual in (None, ''):
            continue
        expected = row.get('currency') if target == 'currency' else contract.get(target)
        if source in ('strike', 'multiplier'):
            a, b = _number(actual), _number(expected)
            equal = a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
        else:
            if source == 'right':
                actual = {'C':'call', 'P':'put'}.get(str(actual).upper(), str(actual).lower())
            equal = actual == expected
        if not equal:
            raise ValueError('resolved_contract_mismatch: ' + target)
        checks.append(target)
    expiry = details.get('realExpirationDate') or resolved.get('realExpirationDate')
    if expiry:
        if str(expiry).replace('-', '') != contract['expiration'].replace('-', ''):
            raise ValueError('resolved_contract_mismatch: expiration')
        checks.append('expiration')
    # OCC symbols additionally encode expiry, right and strike. This check is
    # conditional: non-US symbols must not be parsed as OCC identifiers.
    local = re.fullmatch(r'([A-Z0-9.]+)\s*(\d{6})([CP])(\d{8})', str(resolved.get('localSymbol', '')))
    if local:
        root, expiry, right, strike = local.groups()
        expected_root = contract.get('root')
        if ((expected_root and root != expected_root)
                or expiry != date.fromisoformat(contract['expiration']).strftime('%y%m%d')
                or right != {'call':'C','put':'P'}.get(contract['option_type'])
                or not math.isclose(int(strike)/1000, _number(contract['strike']) or 0, abs_tol=1e-10)):
            raise ValueError('resolved_contract_mismatch: localSymbol')
        checks.append('OCC_symbol')
    return checks


def _expiry(contract, cfg):
    policy = {**cfg['expiry'], **cfg['expiry']['instrument_overrides'].get(contract['underlying'], {})}
    local_time = policy['times_by_settlement'].get(contract['settlement_time'])
    if local_time is None:
        raise ValueError('missing_expiry_time_convention')
    return datetime.combine(date.fromisoformat(contract['expiration']), time.fromisoformat(local_time), ZoneInfo(policy['timezone']))


def _curve_time(stamp, curve, valuation_zone):
    # Curve nodes are civil dates on ACT/365F. Intraday fractions are measured
    # on the same local civil-day axis; the option tau itself uses UTC elapsed
    # seconds, so DST transitions do not lose or duplicate an hour of maturity.
    local = stamp.astimezone(ZoneInfo(valuation_zone))
    days = (local.date() - date.fromisoformat(curve.as_of_date)).days
    fraction = (local.hour*3600 + local.minute*60 + local.second + local.microsecond/1e6) / 86400
    return (days + fraction) / 365


@lru_cache(maxsize=256)
def _session_close(calendar, observed):
    """The underlying calendar supplies the actual close, including half-days.

    Daily provider timestamps often mark the session OPEN. Never treat those
    timestamps as the time at which the daily closing price became available.
    """
    day = date.fromisoformat(observed)
    cal = xcals.get_calendar(calendar, start=(day-timedelta(days=7)).isoformat(),
                             end=(day+timedelta(days=7)).isoformat())
    if not cal.is_session(observed):
        raise ValueError('underlying_date_not_trading_session')
    return cal.session_close(observed).to_pydatetime()


def _historical_spot(prices, underlying, currency, observed, cfg):
    # Only a raw daily close/index level is comparable with the option strike.
    matches = [r for r in prices if r['instrument_id'] == underlying and r['date'] == observed
               and r['currency'] == currency and _true(r['eligible'])
               and r.get('frequency') in (None, '', '1d')
               and not (r.get('source_currency') and r['source_currency'] != currency)
               and r.get('price_basis') in ('index_level','unadjusted_close')
               and (_number(r.get('close')) or 0) > 0]
    for provider in cfg['historical']['underlying_provider_priority']:
        chosen = [r for r in matches if r['provider'] == provider]
        if chosen:
            values = {_number(r['close']) for r in chosen}
            calendars = {r.get('calendar') for r in chosen}
            if len(values) != 1:
                raise ValueError('conflicting_historical_underlying_closes')
            if len(calendars) != 1 or not next(iter(calendars)):
                raise ValueError('missing_or_conflicting_underlying_calendar')
            return values.pop(), provider, _session_close(calendars.pop(), observed)
    raise ValueError('missing_same_date_unadjusted_underlying')


def _historical_context(row, contract, prices, cfg):
    """This prototype is a closing-session model, not an intraday spot model.

    Clamp a later configured cutoff to the session close (e.g. 13:00 on a US
    half-day). Reject an earlier cutoff: a daily close would be future data.
    """
    if not cfg['historical']['allow_daily_close_approximation']:
        raise ValueError('daily_underlying_close_approximation_disabled')
    observed = row['date']
    spot, provider, close = _historical_spot(prices, contract['underlying'], row['currency'], observed, cfg)
    requested = datetime.combine(date.fromisoformat(observed),
        time.fromisoformat(cfg['historical']['cutoff_local_time']), ZoneInfo(cfg['valuation_timezone']))
    if requested.astimezone(timezone.utc) < close:
        raise ValueError('historical_cutoff_precedes_underlying_close')
    return spot, provider, close


def _historical_bar_end(row, cutoff, cfg):
    """Validate availability of a bar's CLOSE rather than its starting label."""
    size = str(row.get('bar_size') or '').strip()
    if size == '1 day' and not row.get('timestamp'):
        if not cfg['historical']['allow_daily_close_approximation']:
            raise ValueError('daily_option_close_approximation_disabled')
        return cutoff  # Explicit same-session daily-close approximation.
    units = {'sec':1,'secs':1,'second':1,'seconds':1,'min':60,'mins':60,
             'minute':60,'minutes':60,'hour':3600,'hours':3600}
    match = re.fullmatch(r'([1-9]\d*)\s+(\w+)', size)
    if not match or match[2] not in units:
        raise ValueError('missing_or_unsupported_option_bar_size')
    start = _stamp(row.get('timestamp')).astimezone(timezone.utc)
    end = start + timedelta(seconds=int(match[1])*units[match[2]])
    age = (cutoff.astimezone(timezone.utc)-end).total_seconds()/60
    zone = ZoneInfo(cfg['valuation_timezone'])
    if (start.astimezone(zone).date().isoformat() != row['date']
            or not 0 <= age <= cfg['historical']['maximum_bar_age_minutes']):
        raise ValueError('historical_bar_outside_close_window')
    return end


def _historical_selection(rows, cfg, prices, contracts):
    """One completed closing observation per contract/provider/day.

    MIDPOINT/TRADES remain separate. We never build a fictitious bid/ask pair
    from independently aggregated bars. Missing metadata produces a diagnostic.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row['instrument_id'],row['provider'],row['date'])].append(row)
    for _, group in sorted(groups.items()):
        representative = group[0]
        if representative['date'] != cfg['valuation_date']:
            yield representative, 'option_date_mismatch'; continue
        try:
            contract = _contract(representative, contracts)
            _, _, cutoff = _historical_context(representative, contract, prices, cfg)
        except (ValueError, KeyError, TypeError) as exc:
            yield representative, str(exc); continue
        selected = None
        failure = 'no_usable_historical_closing_bar'
        for dataset in cfg['historical']['price_priority']:
            candidates = []
            for row in group:
                if row.get('option_dataset') != dataset or not _true(row['eligible']) or (_number(row.get('close')) or 0) <= 0:
                    continue
                try:
                    _validate_contract_identity(row, _contract(row, contracts))
                    candidates.append((_historical_bar_end(row,cutoff,cfg), row))
                except (ValueError, KeyError, TypeError) as exc:
                    failure = str(exc)
            if candidates:
                latest = max(x[0] for x in candidates)
                winners = [row for stamp,row in candidates if stamp == latest]
                if len({_number(row['close']) for row in winners}) != 1:
                    yield representative, 'conflicting_historical_option_closes'
                    selected = False; break
                selected = winners[0]; break
        if selected is None:
            yield representative, failure
        elif selected is not False:
            yield selected, None


def calculate_option(row, route, contract, curves, prices, cfg):
    """Calculate one IV using explicit timing and carry assumptions.

    Returns diagnostics rather than fabricating missing dates, dividends, rates,
    spot prices or volatility. Provider IV and Greeks are not prerequisites.
    """
    result = dict(instrument_id=row.get('instrument_id'),provider=row.get('provider'),
                  route=route,status='insufficient_data', source_file=row.get('source_file'),
                  assumptions=[], reason=None)
    try:
        if route not in ('live', 'historical'):
            raise ValueError('unsupported_option_route')
        result['contract_identity_checks'] = _validate_contract_identity(row, contract)
        currency = row.get('resolved_currency') or row['currency']
        if row.get('source_currency') and row['source_currency'] != currency:
            raise ValueError('source_currency_mismatch')
        result.update(currency=currency,underlying=contract['underlying'],
                      expiration=contract['expiration'],strike=_number(contract['strike']),
                      option_type=contract['option_type'],exercise_style=contract.get('exercise_style'),
                      settlement=contract.get('settlement'),settlement_time=contract.get('settlement_time'))
        if contract['exercise_style'] != 'european':
            raise ValueError('non_european_option_not_supported')
        if contract.get('settlement') not in ('cash','physical'):
            raise ValueError('unknown_settlement')
        expiry = _expiry(contract,cfg)
        result['expiry_timestamp'] = expiry.isoformat()
        result['assumptions'].append('configured_expiry_time_approximation; payoff_discounted_to_expiry')
        if route == 'live':
            # Fresh receipt timestamps do not turn a frozen quote into today's
            # market observation. Delayed/unknown types are also blocked until
            # an independently verified observation-time policy is available.
            result['market_data_type'] = _number(row.get('market_data_type'))
            if result['market_data_type'] != 1:
                raise ValueError('non_live_or_unknown_market_data_type')
            stamp = _stamp(row.get('capture_timestamp_utc') or row['timestamp'])
        else:
            if row['date'] != cfg['valuation_date']:
                raise ValueError('option_date_mismatch')
            spot, provider, stamp = _historical_context(row,contract,prices,cfg)
            bar_end = _historical_bar_end(row,stamp,cfg)
            result.update(bar_end_timestamp=bar_end.isoformat(),bar_size=row.get('bar_size'),
                          bar_age_minutes=(stamp-bar_end).total_seconds()/60)
        observed = stamp.astimezone(ZoneInfo(cfg['valuation_timezone'])).date().isoformat()
        result['valuation_timestamp'] = stamp.isoformat()
        result['valuation_date'] = observed
        if observed != cfg['valuation_date']:
            raise ValueError('option_date_mismatch')
        if currency not in curves:
            raise ValueError('missing_valid_date_matched_curve')
        curve = curves[currency]
        if curve.as_of_date != observed:
            raise ValueError('curve_date_mismatch')
        if not _true(row.get('quote_eligible',row.get('eligible'))):
            raise ValueError('upstream_quote_ineligible')
        tau = (expiry.astimezone(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()/SECONDS_PER_YEAR
        if tau <= 0:
            raise ValueError('expired_option')
        discount = curve.interval_discount(_curve_time(stamp,curve,cfg['valuation_timezone']),
                                           _curve_time(expiry,curve,cfg['valuation_timezone']))
        result.update(time_to_expiry=tau,discount_factor=discount,curve_date=curve.as_of_date)
        if curve.conventions['status'] == 'assumed':
            result['assumptions'].append('unverified_ois_conventions')
        if route == 'live':
            bid,ask = _number(row.get('bid')),_number(row.get('ask'))
            if bid is None or ask is None or not 0 < bid <= ask:
                raise ValueError('missing_or_crossed_bid_ask')
            price = (bid+ask)/2
            if (ask-bid)/price > cfg['live']['maximum_relative_spread']:
                raise ValueError('wide_bid_ask_spread')
            model_stamp = _stamp(row['model_received_at_utc'])
            model_age = (stamp-model_stamp).total_seconds()
            if model_stamp.astimezone(ZoneInfo(cfg['valuation_timezone'])).date().isoformat() != observed or not 0 <= model_age <= cfg['live']['maximum_model_age_seconds']:
                raise ValueError('stale_or_future_paired_model')
            spot = _number(row.get('modelGreeks.undPrice'))
            dividend_pv = _number(row.get('modelGreeks.pvDividend')) if cfg['live']['allow_broker_dividend_pv'] else None
            result.update(price_type='bid_ask_midpoint',bid=bid,ask=ask,model_age_seconds=model_age)
            result['assumptions'].append('SDK_receipt_time_not_verified_exchange_time')
        else:
            price = _number(row['close'])
            # The spot and actual exchange close were already checked above.
            dividend_pv = None
            result.update(price_type=row['option_dataset']+'_bar_close',underlying_provider=provider,
                          bar_timestamp=row.get('timestamp'))
            result['assumptions'].append('historical_bar_and_daily_underlying_close_approximation')
        if spot is None or spot <= 0:
            raise ValueError('missing_positive_underlying')
        result.update(spot=spot,option_price=price)
        # Carry entries are expiry-specific and dated. We never select the
        # nearest date, interpolate across an absent date, or assume q=0.
        carry = next((x for x in cfg['manual_carry_inputs'] if
            (x['underlying'],x['currency'],x['valuation_date'],x['expiration']) ==
            (contract['underlying'],currency,observed,contract['expiration'])),None)
        if carry and 'forward' in carry:
            forward = carry['forward']
            dividend_pv = None
            result['carry_source'] = carry['source']
        else:
            if carry:
                dividend_pv = carry['dividend_pv']
            if dividend_pv is None:
                raise ValueError('missing_date_matched_dividend_or_forward')
            forward = equity_forward(spot,dividend_pv,discount)
            result['carry_source'] = carry['source'] if carry else 'IBKR modelGreeks.pvDividend'
            if carry is None:
                result['assumptions'].append('broker_dividend_PV_with_our_OIS_discounting')
        result.update(forward=forward,dividend_pv=dividend_pv,
                      effective_dividend_yield=-math.log(forward*discount/spot)/tau)
        strike,right = result['strike'],contract['option_type']
        iv = implied_volatility(price,forward,strike,tau,discount,right,maximum=cfg['solver']['maximum_volatility'])
        intrinsic = black_price(forward,strike,tau,discount,0,right)
        vega = black_vega(forward,strike,tau,discount,iv)
        result.update(iv=iv,iv_unit='annualized_decimal',vega_per_unit_volatility=vega,
                      time_value=price-intrinsic,log_forward_moneyness=math.log(strike/forward),
                      total_implied_variance=iv*iv*tau,
                      repriced_option=black_price(forward,strike,tau,discount,iv,right))
        # Successful inversion alone does not imply reliable IV identification.
        if price-intrinsic < cfg['solver']['minimum_time_value_fraction_of_spot']*spot or vega < cfg['solver']['minimum_vega_fraction_of_spot']*spot:
            raise ValueError('ill_conditioned_iv')
        for side in ('bid','ask'):
            if result.get(side) is not None:
                try:
                    result[side+'_iv'] = implied_volatility(result[side],forward,strike,tau,discount,right,maximum=cfg['solver']['maximum_volatility'])
                except ValueError as exc:
                    result[side+'_iv_reason'] = str(exc)
        provider_iv = _number(row.get('modelGreeks.impliedVol'))
        if provider_iv is not None and provider_iv > 0:
            result.update(provider_model_iv=provider_iv,iv_minus_provider_model_iv=iv-provider_iv,
                          price_using_provider_iv=black_price(forward,strike,tau,discount,provider_iv,right))
        result['bid_ask_iv_complete'] = (result.get('bid_iv') is not None and result.get('ask_iv') is not None) if route == 'live' else None
        result['status'] = 'iv_calculated_with_assumptions'
    except (ValueError,KeyError,TypeError,OverflowError,ZeroDivisionError) as exc:
        result['reason'] = str(exc)
    return result


@dataclass
class CalibrationResult:
    """In-memory derived result; save is explicit and never replaces inputs."""
    folder: Path
    run_id: str
    manifest: dict
    tables: dict

    def save(self):
        target = self.folder / 'option_calibration' / self.run_id
        if target.exists():
            saved = _read_json(target/'manifest.json')
            if saved.get('identity') != self.manifest['identity'] or not saved.get('complete'):
                raise ValueError('existing_calibration_result_inconsistent')
            # A truncated file index must not make a missing table invisible
            # to verification. The manifest must enumerate the entire export.
            expected = {name:name+'.csv' for name in self.tables}
            expected['report'] = 'report.md'
            records = saved.get('files')
            if (not isinstance(records, dict) or set(records) != set(expected)
                    or any(not isinstance(records[name], dict)
                           or records[name].get('file') != filename
                           or not isinstance(records[name].get('sha256'), str)
                           for name,filename in expected.items())):
                raise ValueError('existing_calibration_output_index_incomplete')
            for record in records.values():
                if _hash(_inside(target,record['file']).read_bytes()) != record['sha256']:
                    raise ValueError('existing_calibration_output_checksum_mismatch')
            return target
        # mkdir is exclusive: concurrent callers cannot overwrite each other's
        # partial results. A failed export remains incomplete and is not reused.
        target.mkdir(parents=True,exist_ok=False)
        files = {}
        for name,rows in self.tables.items():
            columns = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
            output = io.StringIO(newline='')
            writer = csv.DictWriter(output,fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
            raw = output.getvalue().encode('utf-8')
            path = target/(name+'.csv'); path.write_bytes(raw)
            files[name] = dict(file=path.name,rows=len(rows),sha256=_hash(raw))
        summary = self.manifest['summary']
        report = ('Offline option-input calibration\n\n'
                  + 'Archive COB: '+self.manifest['archive_cob']+'\n'
                  + 'Option observation dates: '+', '.join(summary['valuation_dates'])+'\n'
                  + json.dumps(summary,indent=2)+'\n\n'
                  + '\n'.join('- '+x for x in self.manifest['limitations'])+'\n')
        (target/'report.md').write_text(report,encoding='utf-8')
        files['report'] = dict(file='report.md',sha256=_hash((target/'report.md').read_bytes()))
        write_json(target/'manifest.json',{**self.manifest,'files':files,'complete':True})
        return target


def calibrate_processed(folder, config_file):
    """Read local inputs, build dated curves, and invert live/historical prices."""
    import scipy
    import numpy
    import sys
    cfg = load_calibration_config(config_file)
    manifest,tables,contracts,hashes = load_processed_inputs(folder)
    results, nodes, repricing, statuses, definitions = [], [], [], [], {}

    def observation_date(row, route):
        if route == 'historical':
            return date.fromisoformat(row['date']).isoformat()
        return _stamp(row.get('capture_timestamp_utc') or row.get('timestamp')).astimezone(
            ZoneInfo(cfg['valuation_timezone'])).date().isoformat()

    groups = defaultdict(lambda: {'live': [], 'historical': []})
    for route, name in [('live','selected_option_quotes'),('historical','historical_option_bars')]:
        for row in tables[name]:
            try:
                day = observation_date(row,route)
                groups[day][route].append(row)
            except (ValueError,TypeError,KeyError) as exc:
                results.append(dict(instrument_id=row.get('instrument_id'),route=route,
                                    status='insufficient_data',reason=str(exc)))
    # Keep each date isolated. A curve dated 20 October never calibrates an
    # option dated 19 October, even when both rows live in the same COB archive.
    for day, group in sorted(groups.items()):
        policy = {**cfg, 'valuation_date':day}
        curves,ns,rs,ss = build_curves(tables['ois_quotes'],policy)
        nodes.extend(ns)
        repricing.extend(rs)
        statuses.extend(dict(valuation_date=day,**s) for s in ss)
        definitions[day] = {k:asdict(v) for k,v in curves.items()}
        for row in group['live']:
            try:
                results.append(calculate_option(row,'live',_contract(row,contracts),curves,tables['provider_prices'],policy))
            except (ValueError,TypeError,KeyError) as exc:
                results.append(dict(instrument_id=row.get('instrument_id'),route='live',status='insufficient_data',reason=str(exc)))
        for row,reason in _historical_selection(group['historical'],policy,tables['provider_prices'],contracts):
            if reason:
                results.append(dict(instrument_id=row['instrument_id'],route='historical',status='insufficient_data',reason=reason))
                continue
            try:
                results.append(calculate_option(row,'historical',_contract(row,contracts),curves,tables['provider_prices'],policy))
            except (ValueError,TypeError,KeyError) as exc:
                results.append(dict(instrument_id=row['instrument_id'],route='historical',status='insufficient_data',reason=str(exc)))
    try:
        quantlib_version = version('QuantLib')
    except PackageNotFoundError:
        quantlib_version = 'unavailable'
    code_files = [Path(__file__),Path(__file__).with_name('ois_bootstrap.py'),Path(__file__).with_name('option_math.py'),Path(__file__).with_name('storage.py')]
    identity = dict(engine_version=ENGINE_VERSION,config=cfg,source_folder=str(Path(folder).resolve()),input_hashes=hashes,
                    code_hashes={p.name:_hash(p.read_bytes()) for p in code_files},
                    dependencies={'QuantLib':quantlib_version,'exchange_calendars':version('exchange-calendars'),'scipy':scipy.__version__,'numpy':numpy.__version__,'python':sys.version.split()[0]})
    run_id = _hash(_json_bytes(identity))[:16]
    limitations = [
        'Individual IVs only: no fitted implied-volatility surface, Dupire or simulation.',
        'OIS convention assumptions and short-end log-linear discount interpolation are recorded; quote-source conventions remain unverified.',
        'Provider dividend PV with our discount factors is an approximation; no forecast cashflow schedule is inferred.',
        'Expiry times and discounting payoff to expiry are explicit approximations; actual settlement publication/payment timing may differ.',
        'Curve and option dates must match exactly. Matching dates do not establish simultaneous quotes.',
        'Historical closes need same-date carry inputs and unadjusted underlying levels; missing fields block IV calculation.'
    ]
    # A surface belongs to one underlying and one observation date. Combining
    # different dates/instruments would manufacture a misleading capture span.
    captures = defaultdict(list)
    for row in results:
        if row.get('route') == 'live' and row.get('valuation_timestamp'):
            captures[(row['underlying'],row['valuation_date'])].append(_stamp(row['valuation_timestamp']))
    windows = []
    for (underlying,day), stamps in sorted(captures.items()):
        minutes = (max(stamps)-min(stamps)).total_seconds()/60
        windows.append(dict(underlying=underlying,valuation_date=day,span_minutes=minutes))
        if minutes > cfg['live']['capture_span_warning_minutes']:
            limitations.append('%s on %s: live observations span %.3f minutes; these IVs are NOT a synchronized surface.' % (underlying,day,minutes))
    span = max((w['span_minutes'] for w in windows), default=None)
    summary = dict(option_records=len(results),iv_calculated=sum(r['status']=='iv_calculated_with_assumptions' for r in results),
                   unavailable_reasons=dict(Counter(r['reason'] for r in results if r.get('reason'))),
                   historical_records=sum(r['route']=='historical' for r in results),live_capture_span_minutes=span,
                   live_capture_windows=windows,
                   curve_status=statuses)
    summary['valuation_dates'] = sorted(groups)
    summary['status'] = 'completed' if summary['iv_calculated'] else 'skipped'
    summary['skip_reason'] = None if summary['iv_calculated'] else ('no_option_data' if not results else 'insufficient_eligible_inputs')
    forward_rows = [{k:r.get(k) for k in ('instrument_id','route','currency','underlying','valuation_timestamp','expiration','spot','dividend_pv','discount_factor','forward','effective_dividend_yield','carry_source','assumptions')}
                    for r in results if r.get('forward') is not None]
    result_manifest = dict(identity=identity,archive_cob=manifest['cob'],summary=summary,
                           curve_definitions=definitions,limitations=limitations)
    return CalibrationResult(Path(folder).resolve(),run_id,result_manifest,
                             dict(curve_nodes=nodes,ois_repricing=repricing,curve_status=statuses,
                                  equity_forwards=forward_rows,option_iv=results))
