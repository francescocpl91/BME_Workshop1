"""Requested-window retrieval followed by provider-specific maximum-history fallback.

Raw responses remain untouched. Fallbacks can include rows outside the requested
start; consumers must use the saved requested range and actual observation dates.
"""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import quote

from .models import DateWindow, RetrievalError
from .providers.base import check_request_support

# Only date/data/plan-range failures warrant changing the requested window.
# Missing keys, transport failures, pacing and unsupported products do not.
RANGE_ERRORS = {'no_data', 'invalid_request', 'provider_error', 'entitlement_error', 'access_denied'}


def fetch_history(provider, job):
    info = provider.history_info = {'history_fallback_used': False, 'history_attempts': []}
    symbol = job.instrument.symbols[provider.name]
    check_request_support(provider.name, job.data_type)
    info['history_attempts'].append({'mode': 'requested_window', 'start': str(job.window.start), 'end': str(job.window.end)})
    try:
        provider.fetch_raw(job.instrument, symbol, job.window)
        return 'response_saved', 'Requested history saved; available observations can start later than the requested date.'
    except RetrievalError as exc:
        info['history_attempts'][-1]['error'] = exc.code
        if exc.code not in RANGE_ERRORS:
            raise
        info.update(history_fallback_used=True, history_fallback_reason=exc.code,
                    history_fallback_policy='maximum_accessible_through_cob')
    try:
        status, message = _maximum(provider, job, symbol, info)
        info['history_fallback_status'] = status
        return status, message
    except RetrievalError as exc:
        info['history_fallback_status'] = exc.code
        raise


def _maximum(provider, job, symbol, info):
    end = job.window.end
    attempt = {'mode': 'maximum_accessible', 'end': str(end)}
    info['history_attempts'].append(attempt)
    name = provider.name
    if name in {'fred', 'ecb'}:
        # These APIs accept an omitted lower bound. Keep the COB upper bound.
        attempt['method'] = 'omit_start_parameter'
        provider.fetch_raw(job.instrument, symbol, SimpleNamespace(start=None, end=end))
    elif name == 'alpha_vantage':
        attempt['method'] = 'full_outputsize_then_compact_if_denied'
        # Alpha's daily endpoint has no date parameters. Preserve the full raw
        # response and make the need to select the requested dates explicit.
        info['raw_dates_may_exceed_requested_window'] = True
        provider.settings['ALPHA_VANTAGE_OUTPUTSIZE'] = 'full'
        try:
            provider.fetch_raw(job.instrument, symbol, job.window)
        except RetrievalError as exc:
            if exc.code != 'entitlement_error': raise
            provider.settings['ALPHA_VANTAGE_OUTPUTSIZE'] = 'compact'
            info['history_attempts'].append({'mode': 'compact_access_fallback', 'end': str(end)})
            provider.fetch_raw(job.instrument, symbol, job.window)
            return 'limited_history', 'Maximum/full history denied; latest compact history saved. It may not cover COB.'
    elif name == 'yahoo':
        attempt['method'] = 'first_trade_timestamp_then_bounded_chart'
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol, safe='')
        metadata = provider.client.get_json(url, params={'range':'5d', 'interval':'1d'})
        rows = metadata.get('chart', {}).get('result') or []
        first = rows[0].get('meta', {}).get('firstTradeDate') if rows else None
        if not isinstance(first, (int, float)):
            raise RetrievalError('no_data', 'Yahoo did not provide its first available trade timestamp.')
        start = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=first)).date()
        attempt['start'] = str(start)
        if start > end:
            raise RetrievalError('no_data', 'Yahoo listing starts after COB; no future prices substituted.')
        provider.fetch_raw(job.instrument, symbol, DateWindow(start, end))
    elif name == 'marketdata':
        return _marketdata_max(provider, symbol, end, info)
    elif name == 'massive':
        return _massive_max(provider, job, symbol, info)
    else:
        raise RetrievalError('unsupported_operation', 'No maximum-history fallback implemented for this provider.')
    return 'response_saved', 'Provider maximum-history fallback saved; inspect actual dates before selecting the requested window.'


def _marketdata_max(provider, symbol, end, info):
    provider.check_access()
    size = 10000
    cursor = str(end)
    got = False
    for _ in range(200):
        attempt = {'mode':'countback', 'end':cursor, 'countback':size}
        info['history_attempts'].append(attempt)
        try:
            payload = provider.client.get_json(
                f"{provider.api_url}/stocks/candles/D/{quote(symbol, safe='')}/",
                params={'to':cursor, 'countback':size, 'adjustsplits':'true',
                        'adjustdividends':str(provider.cache_settings()['adjustment']=='split_and_dividend').lower()},
                headers={'Authorization':f'Bearer {provider.key}'})
            if payload.get('s') != 'ok' or not payload.get('t'):
                raise RetrievalError('no_data' if payload.get('s') == 'no_data' else 'provider_error', 'No successful countback candle response.')
        except RetrievalError as exc:
            attempt['error'] = exc.code
            if exc.code not in RANGE_ERRORS: raise
            if size > 1:
                size = max(1, size // 10)
                continue
            if not got: raise
            info['history_stop_reason'] = 'first_unavailable_countback'
            return 'limited_history', 'Accessible history saved back to a provider boundary; maximum beyond that boundary is unverified.'
        stamps = payload['t']
        earliest = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=min(stamps))).date()
        latest = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=max(stamps))).date()
        if latest > date.fromisoformat(cursor):
            raise RetrievalError('invalid_response', 'Countback returned dates after its upper bound.')
        attempt.update(first_observation=str(earliest), observations=len(stamps))
        got = True
        cursor = str(earliest - timedelta(days=1))
        # Probe beyond short pages too: a short page can be a service/account cap.
        size = 10000
    raise RetrievalError('partial_history', 'Maximum-history pagination safety limit reached; retained every received page.')


def _massive_max(provider, job, symbol, info):
    # Aggregate URLs reject negative Unix dates. This is an API search boundary,
    # not a replacement for the user's chosen start or proof of dataset inception.
    floor = date(1970, 1, 1)
    end = job.window.end
    got = False
    span = 365
    for _ in range(400):
        if end < floor:
            info['history_stop_reason'] = 'nonnegative_timestamp_boundary'
            if not got: raise RetrievalError('no_data', 'COB precedes the supported aggregate search boundary.')
            return 'limited_history', 'History saved to the API timestamp search boundary; earlier coverage is unverified.'
        start = max(floor, end - timedelta(days=span-1))
        attempt = {'mode':'backward_window', 'start':str(start), 'end':str(end)}
        info['history_attempts'].append(attempt)
        try:
            provider.fetch_raw(job.instrument, symbol, DateWindow(start, end))
        except RetrievalError as exc:
            attempt['error'] = exc.code
            if exc.code not in RANGE_ERRORS: raise
            if exc.code != 'no_data' and span > 1:
                span = max(1, span // 2)
                continue
            if not got: raise
            info['history_stop_reason'] = 'first_unavailable_window'
            return 'limited_history', 'History saved back to the first unavailable window; earlier coverage/gaps are unverified.'
        got = True
        end = start - timedelta(days=1)
        span = 365
    raise RetrievalError('partial_history', 'Maximum-history search safety limit reached; retained every received response.')
