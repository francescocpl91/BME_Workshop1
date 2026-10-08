"""Tutorial-style ib_insync snapshots on one connection, with bounded concurrency.

Qualification and reqTickersAsync use the same calls as qualifyContracts and
reqTickers in the tutorial. A request is one contract; the shared socket stays
open across batches. No paid regulatory snapshots are requested.
"""
import asyncio
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import time

from ..models import RetrievalError
from .ibkr import _resolved_contract, _model_summary


def plain(value):
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, datetime): return value.isoformat()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if hasattr(value, '_asdict'): return plain(value._asdict())
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return str(value)


def sdk():
    try:
        from ib_insync import IB, Contract
    except ImportError:
        raise RetrievalError('missing_dependency', 'Install the project dependencies, including ib_insync.') from None
    return IB, Contract


def snapshot_model_summary(events, requested_mode):
    """Keep supplied IV/price visible without pretending a missing spot is present."""
    try:
        return _model_summary(events, requested_mode), None
    except RetrievalError:
        def valid(value, positive=False):
            return (type(value) in (int, float) and math.isfinite(value)
                    and (value > 0 if positive else value >= 0) and value < 1e100)
        models = [e for e in events if e.get('event') == 'tickOptionComputation'
                  and e.get('tick_type') in {13, 83}
                  and valid(e.get('implied_volatility'), True)
                  and valid(e.get('option_price'))]
        if not models:
            raise
        latest = models[-1]
        modes = [e['market_data_type'] for e in events if e.get('event') == 'marketDataType']
        return {
            'model_observations': len(models), 'model_tick_type': latest['tick_type'],
            'model_data_timing': 'delayed' if latest['tick_type'] == 83 else 'live_or_frozen',
            'requested_market_data_type': requested_mode,
            'actual_market_data_type': modes[-1] if modes else None,
            'model_received_at_utc': latest.get('received_at_utc'),
            'implied_volatility': latest['implied_volatility'],
            'model_option_price': latest['option_price'],
            'underlying_price': None, 'missing_fields': ['underlying_price'],
        }, RetrievalError('partial_model_data',
            'Option model IV and price received; underlying spot price is missing. '
            'Historical underlying data is retrieved separately; no historical close was substituted.')


def iter_snapshots(provider, requests, progress):
    requests = list(requests)
    if not requests: return
    try: IB, Contract = sdk()
    except RetrievalError as error:
        for job in requests: yield dict(request=job, body=None, info={}, error=error)
        return
    try:
        limit = int(provider.settings.get('IBKR_OPTION_MAX_CONCURRENT', '5'))
        if not 1 <= limit <= 50: raise ValueError
    except (ValueError, TypeError):
        raise RetrievalError('configuration_error', 'IBKR option concurrency must be 1-50.') from None
    ib = IB()
    host = provider.settings.get('IBKR_HOST', '127.0.0.1')
    port = int(provider.settings.get('IBKR_PORT', '7497'))
    timeout = float(provider.settings.get('IBKR_TIMEOUT', '20'))
    events, details, request_ids, common = {}, {}, {}, []
    delivered = set()

    def error(req_id, code, message, contract):
        event = dict(event='error', request_id=req_id, code=code, message=message,
                     received_at_utc=datetime.now(timezone.utc).isoformat())
        key = request_ids.get(req_id)
        if key is None: common.append(event)
        else: events[key].append(event)
    ib.errorEvent += error
    original_details = ib.reqContractDetailsAsync
    async def lookup(contract):
        result = await original_details(contract)
        details[id(contract)] = result
        return result
    ib.reqContractDetailsAsync = lookup
    original_request = ib.client.reqMktData
    def request(req_id, contract, *args):
        request_ids[req_id] = id(contract)
        return original_request(req_id, contract, *args)
    ib.client.reqMktData = request
    original_error = getattr(ib.wrapper, 'error', None)
    if original_error is not None:
        def receive_error(req_id, code, message, advanced_order_reject_json=''):
            # ib_insync treats 10090 as fatal and resolves snapshot futures early.
            # IB explicitly keeps the subscribed parts active: preserve the warning
            # and wait for tickSnapshotEnd or our timeout to capture those fields.
            if code == 10090 and req_id in request_ids:
                contract = getattr(ib.wrapper, '_reqId2Contract', {}).get(req_id)
                ib.errorEvent.emit(req_id, code, message, contract)
                return
            return original_error(req_id, code, message, advanced_order_reject_json)
        ib.wrapper.error = receive_error
    for name in ('marketDataType', 'tickOptionComputation', 'tickPrice', 'priceSizeTick',
                 'tickSize', 'tickString', 'tickSnapshotEnd'):
        original = getattr(ib.wrapper, name, None)
        if original is None: continue
        def wrap(label, callback):
            def receive(*args):
                key = request_ids.get(args[0])
                if key is not None:
                    event = dict(event=label, request_id=args[0], args=plain(args),
                                 received_at_utc=datetime.now(timezone.utc).isoformat())
                    if label == 'marketDataType': event['market_data_type'] = args[1]
                    if label == 'tickOptionComputation':
                        event.update(zip(('tick_type','tick_attribute','implied_volatility','delta',
                                          'option_price','pv_dividend','gamma','vega','theta','underlying_price'), args[1:]))
                    events[key].append(event)
                return callback(*args)
            return receive
        setattr(ib.wrapper, name, wrap(name, original))

    async def capture(job):
        started = time.monotonic()
        native = Contract(**job.instrument.provider_inputs['ibkr'])
        key = id(native); events[key] = []
        result = dict(request=job, body=None, info={}, error=None,
                      url=f'ibkr://{host}:{port}/reqTickers')
        ticker = None
        completed = False
        seconds = float(job.provider_options.get('IBKR_OPTION_CAPTURE_SECONDS',
                        provider.settings.get('IBKR_OPTION_CAPTURE_SECONDS', '60')))
        mode = int(job.provider_options.get('IBKR_OPTION_MARKET_DATA_TYPE',
                   provider.settings.get('IBKR_OPTION_MARKET_DATA_TYPE', '1')))
        try:
            qualified = await asyncio.wait_for(ib.qualifyContractsAsync(native), timeout)
            if len(qualified) != 1:
                raise RetrievalError('contract_not_found', 'TWS did not qualify exactly one contract.')
            _resolved_contract(details.get(key, []), job.instrument,
                               job.instrument.provider_inputs['ibkr'], job.contract)
            # This is the tutorial's reqTickers call, using its async equivalent.
            # The mode is set for each batch before these tasks are scheduled.
            tickers = await asyncio.wait_for(ib.reqTickersAsync(native, regulatorySnapshot=False), seconds)
            ticker = tickers[0]
            completed = True
            result['info'], result['error'] = snapshot_model_summary(events[key], mode)
        except asyncio.TimeoutError:
            # Preserve partial quotes/models even if tickSnapshotEnd never arrives.
            ticker = ib.ticker(native)
            try: result['info'], result['error'] = snapshot_model_summary(events[key], mode)
            except RetrievalError:
                result['error'] = RetrievalError('snapshot_timeout', 'Snapshot timed out; available quotes and warnings were retained.')
        except RetrievalError as exc: result['error'] = exc
        except (ConnectionError, OSError):
            result['error'] = RetrievalError('network_error', 'TWS connection failed during snapshot retrieval.')
        except Exception as exc:
            result['error'] = RetrievalError('invalid_response', 'Snapshot failed (' + type(exc).__name__ + '); partial response retained.')
        finally:
            if ticker is None: ticker = ib.ticker(native)
            for req_id, identity in list(request_ids.items()):
                if identity == key:
                    if not completed:
                        ib.client.cancelMktData(req_id)
                        ib.wrapper._endReq(req_id)
                    request_ids.pop(req_id, None)
            if ticker is not None: ib.wrapper.endTicker(ticker, 'snapshot')
            elapsed = round(time.monotonic()-started, 3)
            result['info'].update(elapsed_seconds=elapsed, snapshot_completed=completed)
            selected = {name: plain(getattr(ticker, name, None)) for name in
                        ('time','marketDataType','bid','ask','last','close','bidSize','askSize',
                         'lastSize','volume','bidGreeks','askGreeks','lastGreeks','modelGreeks')}
            body = dict(capture_format='serialized_sdk_callbacks', adapter='ib_insync',
                        transport='shared_tws_connection', operation='option_model_snapshot',
                        method='reqTickersAsync', snapshot=True, regulatory_snapshot=False,
                        requested_market_data_type=mode, snapshot_completed=completed,
                        contract_input=job.instrument.provider_inputs['ibkr'],
                        resolved_contract=plain(native), contract_details=plain(details.pop(key, [])),
                        ticker=selected, events=plain(events.pop(key, [])), session_events=common,
                        elapsed_seconds=elapsed)
            result['body'] = json.dumps(body, indent=2, allow_nan=False).encode()
        return result

    try:
        # Exactly one authenticated TWS connection for all option requests.
        progress('IBKR: connecting to TWS for option snapshots ...')
        ib.connect(host, port, clientId=int(provider.settings.get('IBKR_CLIENT_ID','81')),
                   timeout=timeout, readonly=True)
        progress(f'IBKR ib_insync connection ready; snapshot batches of at most {limit}.')
        # Group by data mode because reqMarketDataType applies to the connection.
        modes = {}
        for job in requests:
            mode = int(job.provider_options.get('IBKR_OPTION_MARKET_DATA_TYPE',
                       provider.settings.get('IBKR_OPTION_MARKET_DATA_TYPE','1')))
            modes.setdefault(mode, []).append(job)
        for mode, jobs in modes.items():
            ib.reqMarketDataType(mode)
            for start in range(0, len(jobs), limit):
                expected_date = provider.settings.get('IBKR_LIVE_EXPECTED_DATE')
                if expected_date:
                    from zoneinfo import ZoneInfo
                    if datetime.now(ZoneInfo('Europe/London')).date().isoformat() != expected_date:
                        for job in jobs[start:]:
                            yield dict(request=job,body=None,info={},error=RetrievalError('live_date_unavailable','Live API cannot retrieve a past or future capture date.'))
                        break
                batch = jobs[start:start+limit]
                if not ib.isConnected():
                    for job in jobs[start:]:
                        yield dict(request=job,body=None,info={},error=RetrievalError('network_error','TWS disconnected.'))
                    break
                async def gather(): return await asyncio.gather(*(capture(job) for job in batch))
                for result in ib.run(gather()):
                    progress('  IBKR ['+{1:'LIVE',2:'FROZEN',3:'DELAYED',4:'DELAYED FROZEN'}.get(mode,'UNKNOWN')+' SNAPSHOT] '+result['request'].instrument.ticker+': '+(result['error'].code if result['error'] else 'model_received'))
                    delivered.add(result['request'].request_id)
                    yield result
    except (ConnectionError, OSError, asyncio.TimeoutError):
        for job in requests:
            if job.request_id not in delivered:
                yield dict(request=job,body=None,info={},error=RetrievalError('network_error','TWS unavailable; check login, port and API settings.'))
    finally:
        ib.disconnect()
