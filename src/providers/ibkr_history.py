"""Historical-only TWS session. Never calls reqMktData or requests live Greeks."""
import asyncio
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import json
import time as clock
from zoneinfo import ZoneInfo

from .ibkr import _resolved_contract
from .ibkr_snapshots import plain
from ..models import RetrievalError


def error_code(exc):
    text = str(exc).lower()
    if isinstance(exc, asyncio.TimeoutError): return 'timeout'
    if 'pacing' in text or 'too many' in text: return 'rate_limited'
    if 'permission' in text or 'subscrib' in text: return 'entitlement_error'
    if 'no data' in text or 'no historical' in text: return 'no_data'
    if isinstance(exc, (ConnectionError, OSError)): return 'network_error'
    if isinstance(exc, RetrievalError): return exc.code
    return 'provider_error'


def iter_history(jobs, settings, progress, earliest_available=False, *, fallback_max=False):
    from ib_insync import IB, Contract
    jobs = list(jobs)
    if not jobs: return
    limit = int(settings.get('IBKR_HISTORY_MAX_CONCURRENT', '5'))
    interval = float(settings.get('IBKR_HISTORY_REQUEST_INTERVAL', '.25'))
    timeout = float(settings.get('IBKR_HISTORY_TIMEOUT', '60'))
    if not 1 <= limit <= 10 or not .1 <= interval <= 10 or not 1 <= timeout <= 180:
        raise ValueError('History settings: concurrency 1-10, interval .1-10 seconds, timeout 1-180 seconds.')
    ib = IB(); ib.RaiseRequestErrors = True
    errors = {}; common = []; requests = {}; head_requests = {}; failures = Counter(); delivered = set()
    gate = asyncio.Lock(); next_send = 0.
    def receive_error(req_id, code, message, contract):
        item = dict(code=code, message=message, received_at_utc=datetime.now(timezone.utc).isoformat())
        errors.setdefault(req_id, []).append(item)
        if req_id < 0: common.append(item)
    ib.errorEvent += receive_error
    original_history = ib.client.reqHistoricalData
    def send(req_id, contract, *args):
        requests[id(contract)] = req_id
        return original_history(req_id, contract, *args)
    ib.client.reqHistoricalData = send
    if hasattr(ib.client, 'reqHeadTimeStamp'):
        original_head = ib.client.reqHeadTimeStamp
        def send_head(req_id, contract, *args):
            head_requests[id(contract)] = req_id
            return original_head(req_id, contract, *args)
        ib.client.reqHeadTimeStamp = send_head

    async def pace():
        nonlocal next_send
        async with gate:
            await asyncio.sleep(max(0., next_send-clock.monotonic()))
            next_send = clock.monotonic()+interval

    async def history(contract, end, duration, size, kind):
        await pace()
        payload = {'capture_format':'serialized_sdk_historical_bars', 'method':'reqHistoricalData',
                   'request':dict(end_utc=end,duration=duration,bar_size=size,what_to_show=kind,
                                  use_rth=True,format_date=2,keep_up_to_date=False),
                   'bars':[], 'errors':[], 'completed':False}
        try:
            bars = await asyncio.wait_for(ib.reqHistoricalDataAsync(contract,
                endDateTime=end,durationStr=duration,barSizeSetting=size,whatToShow=kind,
                useRTH=True,formatDate=2,keepUpToDate=False,timeout=0),timeout)
            payload.update(bars=plain(bars),completed=True)
            return payload, None
        except Exception as exc:
            code = error_code(exc)
            req_id = requests.get(id(contract))
            if code == 'timeout' and req_id is not None:
                partial = ib.wrapper._results.get(req_id, [])
                payload['bars'] = plain(partial)
                ib.client.cancelHistoricalData(req_id)
                ib.wrapper._endReq(req_id)
            return payload, code
        finally:
            req_id = requests.pop(id(contract), None)
            payload['request_id'] = req_id
            payload['errors'] = errors.pop(req_id, [])
            payload['received_at_utc'] = datetime.now(timezone.utc).isoformat()

    async def capture(job):
        result = dict(request=job, payloads=[], info={}, status='response_saved', message='Historical bars saved.')
        contract = Contract(**job.instrument.provider_inputs['ibkr'])
        try:
            if job.contract and date.fromisoformat(job.contract['expiration']) < date.today():
                raise RetrievalError('historical_unavailable','IBKR does not supply history for expired options.')
            await pace()
            details = await asyncio.wait_for(ib.reqContractDetailsAsync(contract),timeout)
            resolved = _resolved_contract(details,job.instrument,job.instrument.provider_inputs['ibkr'],job.contract)
            contract.update(**{k:v for k,v in vars(resolved).items() if v is not None})
            result['payloads'].append({'capture_format':'serialized_sdk_contract_details',
                'method':'reqContractDetails','retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                'contract_details':plain(details),'note':'Contract metadata retrieved now; not a historical quote or historical Greeks.'})
            if job.instrument.kind == 'option':
                opts = {**settings, **job.provider_options}
                zone = ZoneInfo(opts.get('IBKR_HISTORY_TIMEZONE','America/New_York'))
                cutoff = datetime.combine(job.window.end,time.fromisoformat(opts.get('IBKR_HISTORY_CUTOFF','16:00:00')),zone)
                end = cutoff.astimezone(timezone.utc).strftime('%Y%m%d-%H:%M:%S')
                channels = {}
                for kind in json.loads(opts.get('IBKR_HISTORY_DATA_TYPES','["TRADES","BID","ASK","MIDPOINT"]')):
                    if failures[kind] >= 3:
                        channels[kind] = {'status':'channel_paused','reason':'Three consecutive timeout/rate/network failures; no further requests for this channel in this run.'}
                        continue
                    payload, code = await history(contract,end,'1 D','1 min',kind)
                    result['payloads'].append(payload)
                    def within(bar):
                        try:
                            stamp=datetime.fromisoformat(bar['date']).astimezone(zone)
                            return stamp.date()==job.window.end and stamp < cutoff
                        except (ValueError,TypeError,KeyError):return False
                    count = sum(within(b) for b in payload['bars'])
                    if code in {'timeout','rate_limited','network_error'}: failures[kind]+=1
                    else: failures[kind]=0
                    channels[kind] = {'status':code or ('response_saved' if count else 'no_data'),
                                      'bars_on_cob_before_cutoff':count,'bars_returned':len(payload['bars'])}
                statuses = [v['status'] for v in channels.values()]
                result['info'] = {'channels':channels,'cutoff_utc':end,
                    'historical_model_iv_available':False,'historical_greeks_available':False,
                    'bar_timestamp_meaning':'Start of one-minute interval; not an official option settlement price.'}
                result['status'] = 'response_saved' if all(s=='response_saved' for s in statuses) else 'partial_history' if any(s=='response_saved' for s in statuses) else 'no_data' if all(s=='no_data' for s in statuses) else 'history_error'
                result['message'] = 'Historical option bars only. IV/Greeks not supplied by this endpoint. Missing channels are recorded separately.'
            else:
                await pace()
                # A failed earliest-date lookup does not mean price bars are unavailable.
                head_payload = {'method':'reqHeadTimeStamp','completed':False,
                                'received_at_utc':datetime.now(timezone.utc).isoformat()}
                head_known = False
                try:
                    head = await asyncio.wait_for(ib.reqHeadTimeStampAsync(contract,'TRADES',True,formatDate=2),timeout)
                    head_day = head.date() if isinstance(head,datetime) else date.fromisoformat(str(head)[:10])
                    head_payload.update(completed=True,earliest_timestamp=plain(head))
                    result['info']['earliest_available_date'] = head_day.isoformat()
                    head_known = True
                except Exception as exc:
                    head_payload.update(status=error_code(exc),message=str(exc),
                                        error_code=getattr(exc,'code',None))
                    req_id=head_requests.pop(id(contract),None)
                    if req_id is not None:
                        ib.client.cancelHeadTimeStamp(req_id)
                        ib.wrapper._endReq(req_id)
                        head_payload['errors']=errors.pop(req_id,[])
                    head_day = job.window.start
                    result['info']['earliest_date_lookup_failed'] = True
                    progress(f"IBKR [HISTORICAL] {job.instrument.ticker}: earliest-date lookup failed; requesting daily bars backwards from COB.")
                head_requests.pop(id(contract),None)
                result['payloads'].append(head_payload)
                lower_bound = head_day if head_known and earliest_available else max(head_day, job.window.start)
                result['info'].update(effective_history_start=lower_bound.isoformat(), history_fallback_used=False,
                                      requested_history_start=job.window.start.isoformat(), history_attempts=[])
                modes = [(lower_bound, False)]
                total = 0
                for lower_bound, maximum in modes:
                    end_day = job.window.end
                    stop_reason = 'requested_lower_bound'
                    while end_day >= lower_bound:
                        start = max(lower_bound, end_day-timedelta(days=364))
                        end = datetime.combine(end_day+timedelta(days=1),time.min,timezone.utc).strftime('%Y%m%d-%H:%M:%S')
                        days = (end_day-start).days+1
                        payload, code = await history(contract,end,'1 Y' if days==365 else f'{days} D','1 day','TRADES')
                        payload.update(history_mode='maximum_accessible' if maximum else 'requested_window',
                                       requested_start=str(start), requested_end=str(end_day))
                        result['payloads'].append(payload)
                        total += len(payload['bars'])
                        result['info']['history_attempts'].append(dict(mode=payload['history_mode'],start=str(start),end=str(end_day),error=code))
                        empty = code == 'no_data' or (code is None and not payload['bars'])
                        if fallback_max and not maximum and (empty or code in {'invalid_request','provider_error','entitlement_error'}):
                            # A date/data error changes the policy once. Never use
                            # live data, or widen history for a transport error.
                            maximum_start = head_day if head_known else date(1900,1,1)
                            result['info'].update(history_fallback_used=True,history_fallback_reason=code or 'no_data',
                                history_fallback_start=str(maximum_start), maximum_start_from_head_timestamp=head_known)
                            modes.append((maximum_start, True))
                            stop_reason = 'retry_maximum_accessible'
                            break
                        if empty and (maximum or not head_known):
                            stop_reason = 'first_empty_year_without_head_timestamp' if not head_known else 'first_unavailable_window'
                            break
                        if code:
                            raise RetrievalError(code,'Historical price chunk failed; earlier received chunks are retained.')
                        end_day = start-timedelta(days=1)
                    if maximum and not head_known and stop_reason == 'requested_lower_bound':
                        stop_reason = 'search_safety_boundary'
                    result['info'].update(bars_returned=total,history_lower_bound_reached=(stop_reason=='requested_lower_bound'),history_stop_reason=stop_reason)
                    if maximum:
                        result['info']['history_fallback_status'] = 'response_saved' if stop_reason=='requested_lower_bound' else 'limited_history'
                    if stop_reason not in {'requested_lower_bound','retry_maximum_accessible'}:
                        result['message']='Historical bars saved to the first unavailable interval. Coverage before this boundary is unverified.'
                        if maximum: result['status']='limited_history'
                if not total:
                    result.update(status='no_data',message='No historical prices on or before this COB.')
                    if result['info'].get('history_fallback_used'):
                        result['info']['history_fallback_status'] = 'no_data'

        except Exception as exc:
            if result['info'].get('history_fallback_used'):
                result['info']['history_fallback_status'] = error_code(exc)
            result.update(status=error_code(exc),message=str(exc) if isinstance(exc,RetrievalError) else 'IBKR historical request failed ('+type(exc).__name__+'); inspect retained responses.')
        return result

    try:
        ib.connect(settings.get('IBKR_HOST','127.0.0.1'),int(settings.get('IBKR_PORT','7496')),
                   clientId=int(settings.get('IBKR_CLIENT_ID','81')),timeout=timeout,readonly=True)
        progress(f'IBKR [HISTORICAL] connection ready; at most {limit} concurrent contracts. Live options, if selected, use a separate snapshot connection.')
        # All history uses this one session. Price histories precede the option list.
        ordered=sorted(jobs,key=lambda j:j.instrument.kind=='option')
        for start in range(0,len(ordered),limit):
            async def gather():return await asyncio.gather(*(capture(j) for j in ordered[start:start+limit]))
            for result in ib.run(gather()):
                delivered.add(result['request'].request_id)
                yield result
    except Exception as exc:
        for job in jobs:
            if job.request_id not in delivered:
                yield dict(request=job,payloads=[],info={},status=error_code(exc),message='IBKR historical session unavailable; other providers can continue.')
    finally:ib.disconnect()
