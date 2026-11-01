"""Persistent COB archives with explicit historical and live-option provenance."""
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pandas as pd
from .config import parse_requests, PROVIDERS
from .models import RetrievalError
from .raw import ResponseRecorder
from .storage import safe_name, write_json


def london_today():
    return datetime.now(ZoneInfo('Europe/London')).date()


def forecast_start_for_cob(value):
    """Return exactly archive COB + one calendar day, without reading a clock.

    Terminal input is validated separately by parse_cob. Modeling uses the
    selected archive's ISO date, so offline reruns never depend on today's date
    or a stale configuration date. A weekend/holiday is not rolled forward.
    """
    try:
        if type(value) is date:
            day = value
        elif isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            day = date.fromisoformat(value)
        else:
            raise ValueError('invalid date')
        return (day + timedelta(days=1)).isoformat()
    except (ValueError, OverflowError):
        raise ValueError('simulation_requires_valid_archive_cob') from None


def parse_cob(value, *, today=None):
    today = today or london_today()
    if type(value) is date: cob = value
    else:
        if not isinstance(value,str) or not re.fullmatch(r'\d{2}/\d{2}/\d{4}',value.strip()):
            raise ValueError('Enter the COB date as DD/MM/YYYY.')
        try:cob=datetime.strptime(value.strip(),'%d/%m/%Y').date()
        except ValueError:raise ValueError('Enter a valid calendar date as DD/MM/YYYY.') from None
    if cob >= today:raise ValueError('COB must be yesterday or earlier; today and future dates are not allowed.')
    if cob < date(1900,1,1):raise ValueError('COB must be 01/01/1900 or later for these configured products.')
    return cob


def parse_history_start(value, cob):
    """A caller-selected start, never silently replaced with a default."""
    if type(value) is date:
        start = value
    else:
        if not isinstance(value, str) or not re.fullmatch(r'\d{2}/\d{2}/\d{4}', value.strip()):
            raise ValueError('Enter the history start as DD/MM/YYYY.')
        try:
            start = datetime.strptime(value.strip(), '%d/%m/%Y').date()
        except ValueError:
            raise ValueError('Enter a valid history start date as DD/MM/YYYY.') from None
    if start > cob:
        raise ValueError('History start must be on or before COB.')
    if start < date(1900, 1, 1):
        raise ValueError('History start must be 01/01/1900 or later.')
    return start


@dataclass
class CobResult:
    folder: Path
    summary: pd.DataFrame
    reused: bool = False


def option_route(cob, today=None):
    """Date alone selects the API; an older COB never requests live options."""
    today = today or london_today()
    cob = parse_cob(cob, today=today)
    return 'live' if cob == today - timedelta(days=1) else 'historical'


def _http_history(provider, job):
    from .history import fetch_history
    return fetch_history(provider, job)


def _compatible_archive(saved, requested):
    """Allow broader saved history, while preserving instrument/settings checks.

    Schema 2 did not record effective runtime settings. Reuse it as originally
    captured, without claiming those missing settings have been verified.
    """
    prior, current = deepcopy(saved), deepcopy(requested)
    schema = prior.pop('schema', None)
    current.pop('schema', None)
    if schema not in {2, 3}:
        return False
    if schema == 2:
        current.pop('data_settings', None)
    old_rows = prior.get('requests', [])
    new_rows = current.get('requests', [])
    if len(old_rows) != len(new_rows):
        return False
    for old, new in zip(old_rows, new_rows):
        if old.get('kind') != 'option' and new.get('kind') != 'option':
            old_window, new_window = old.get('timeframe', {}), new.get('timeframe', {})
            if (old_window.get('end') == new_window.get('end')
                    and old_window.get('start', '9999') <= new_window.get('start', '0000')):
                new['timeframe'] = deepcopy(old_window)
    return prior == current


def _saved_range_covers(row, job, folder, original_requests):
    """Check executed attempt bounds, not just the immutable archive identity.

    This verifies the range attempted, not completeness of provider observations.
    Old first attempts can inherit original identity bounds; later attempts must
    have their own recorded bounds. Live option captures remain date-stable.
    """
    if not row or not row.get('finished'):
        return False
    if job.instrument.kind == 'option':
        return True
    start, end = row.get('requested_start'), row.get('requested_end')
    if start is None and end is None:
        attempt = str(row.get('attempt', '')).replace('\\', '/').rsplit('/', 1)[-1]
        if not re.fullmatch(r'attempt_[0-9]+', attempt):
            return False
        request_file = folder / row['provider'] / safe_name(job.request_id) / attempt / 'request.json'
        if request_file.exists():
            metadata = json.loads(request_file.read_text(encoding='utf-8-sig'))
            start, end = metadata.get('requested_start'), metadata.get('requested_end')
        if start is None and end is None and attempt == 'attempt_001':
            original = original_requests.get(job.request_id, {})
            if row['provider'] not in original.get('symbols', {}):
                return False
            window = original.get('timeframe', {})
            start, end = window.get('start'), window.get('end')
    try:
        return date.fromisoformat(start) <= job.window.start and date.fromisoformat(end) >= job.window.end
    except (ValueError, TypeError):
        return False


def download_cob(downloader,cob,output,*,portfolio_file=None,retry_failed=False,history_start=None,ois_file=None):
    execution_date=london_today()
    cob=parse_cob(cob,today=execution_date)
    config=deepcopy(downloader._retrieval_config)
    if config is None:raise ValueError('COB download requires explicit JSON/configured requests.')
    if history_start is not None: history_start = parse_history_start(history_start, cob)
    original_windows = {job.request_id: job.window for job in parse_requests(config)}
    config['requests'] = [row for row in config['requests'] if row.get('enabled', True)]
    # Explicit caller/JSON start dates take precedence; options remain COB-only.
    for row in config['requests']:
        if row['kind']=='option':
            if set(row['symbols'])!={'ibkr'}:raise ValueError('COB options must use IBKR only.')
            if row.get('operation')!='option_history':raise ValueError('COB option inputs require option_history inputs; the COB date selects the actual API.')
            row['timeframe']={'start':str(cob),'end':str(cob)}
        else:
            start = history_start or original_windows[row['id']].start
            if start > cob: raise ValueError('History start must be on or before COB.')
            row['timeframe']={'start':start.isoformat(),'end':str(cob)}
    jobs=parse_requests(config)
    # Only data-affecting choices enter the fingerprint, never credentials or
    # transport settings. Include both option routes so reuse is date-stable.
    option_keys = {
        'alpha_vantage': ('ALPHA_VANTAGE_OUTPUTSIZE', 'ALPHA_VANTAGE_ADJUSTED'),
        'marketdata': ('MARKETDATA_ADJUSTMENT',),
    }
    data_settings = {}
    for job in jobs:
        for name in job.providers:
            keys = option_keys.get(name, ())
            if name == 'ibkr' and job.instrument.kind == 'option':
                keys = ('IBKR_HISTORY_DATA_TYPES', 'IBKR_HISTORY_CUTOFF',
                        'IBKR_HISTORY_TIMEZONE', 'IBKR_OPTION_CAPTURE_SECONDS')
            effective = {**downloader._settings, **job.provider_options}
            data_settings.setdefault(name, {})[job.request_id] = {key: effective[key] for key in keys}
    identity={'schema':3,'option_routing_policy':'yesterday_live_older_historical','cob':str(cob),'requests':config['requests'],
              'data_settings':data_settings}
    if portfolio_file:
        from .portfolio import load_portfolio
        load_portfolio(portfolio_file,config)
        identity['portfolio']=json.loads(Path(portfolio_file).read_text(encoding='utf-8-sig'))
    ois_document = None
    if ois_file is not None:
        from .ois import load_ois_config, ois_fingerprint, write_ois_dump, verify_ois_dump
        ois_document = load_ois_config(ois_file)
        if ois_document is not None: identity['manual_ois_fingerprint'] = ois_fingerprint(ois_document)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=Path(output).resolve()/str(cob)
    manifest_file=folder/'manifest.json';summary_file=folder/'summary.json'
    existing=json.loads(manifest_file.read_text(encoding='utf-8-sig')) if manifest_file.exists() else {}
    if existing and existing.get('config_hash') != digest:
        inputs_file = folder / 'resolved_inputs.json'
        if not inputs_file.exists():
            raise ValueError('COB archive inputs are missing; cannot verify saved instruments. No API calls made.')
        saved_identity = json.loads(inputs_file.read_text(encoding='utf-8-sig'))
        saved_digest = hashlib.sha256(json.dumps(saved_identity,sort_keys=True).encode()).hexdigest()
        if saved_digest != existing.get('config_hash') or not _compatible_archive(saved_identity, identity):
            raise ValueError('A COB archive exists with different inputs. Preserve/rename that archive before running changed inputs; no API calls made.')
        if existing.get('schema_version') == 2:
            downloader.progress('Reusing legacy archive inputs as captured; schema 2 did not record effective runtime settings.')
        # Keep the original identity/provenance on resume or explicit retry.
        # A change to the fallback date must not rewrite older captured history.
        identity, digest = saved_identity, saved_digest
    original_requests = {row['id']: row for row in identity['requests'] if row.get('enabled', True)}
    if existing.get('complete') and not retry_failed:
        if not summary_file.exists():raise ValueError('COB archive summary is missing; archive requires inspection, no APIs called.')
        saved_rows = json.loads(summary_file.read_text(encoding='utf-8-sig'))
        by_key = {(row['request_id'], row['provider']): row for row in saved_rows}
        for job in jobs:
            for name in job.providers:
                if not _saved_range_covers(by_key.get((job.request_id, name)), job, folder, original_requests):
                    raise ValueError('The saved attempt does not cover the requested range for '+job.request_id+'/'+name+'. Use retry_failed=True to retrieve the required range in a new attempt; no API calls or archive writes made.')
        if ois_document is not None: verify_ois_dump(ois_document, cob, folder, saved_rows)
        downloader.progress('Reusing completed COB archive: '+str(folder)+'; no API requests.')
        return CobResult(folder,pd.DataFrame(saved_rows),True)
    folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'.running'
    try:handle=lock.open('x',encoding='utf-8')
    except FileExistsError:raise ValueError('A run lock exists for this COB. Check the active run before removing a stale .running file.') from None
    records=[];pending=[];settings=downloader._settings;progress=downloader.progress
    base={'schema_version':identity['schema'],'execution_date':str(execution_date),'option_route':option_route(cob,execution_date),'cob':str(cob),'config_hash':digest,'complete':False,
          'meaning_of_complete':'Every configured request was attempted; see all_requests_succeeded and individual statuses.',
          'history_start_policy':'User/JSON start through COB; range failure triggers provider-specific maximum accessible history. See per-request fallback metadata.',
          'notes':['Prices/rates requested through COB; raw payloads may contain provider padding or later metadata. Alpha Vantage has no historical end-date parameter; filter its dated rows to COB before analysis.',
                   'Macro series are current vintages with observation dates bounded by COB, not point-in-time revision archives.',
                   'Yesterday COB selects current live option snapshots; older COB selects historical option bars only. Historical bars have no contract IV/Greeks.',
                   'Live option capture is separate from COB historical prices. Callback receipt time is not a guaranteed exchange quote timestamp.',
                   'No cleansing, filling, implied-volatility calculation or surface calibration is performed.']}
    try:
        handle.write('COB download in progress');handle.close()
        write_json(manifest_file,base)
        write_json(folder/'resolved_inputs.json',identity)
        if portfolio_file:write_json(folder/'portfolio.json',identity['portfolio'])
        if ois_document is not None: records.extend(write_ois_dump(ois_document, cob, folder, progress))
        for job in jobs:
            for name in job.providers:
                dest=folder/name/safe_name(job.request_id)
                checkpoint=dest/'status.json'
                old=json.loads(checkpoint.read_text(encoding='utf-8-sig')) if checkpoint.exists() else None
                if old and old.get('finished') and _saved_range_covers(old, job, folder, original_requests) and (not retry_failed or old['status']=='response_saved'):
                    records.append(old)
                else:pending.append((job,name,dest))
        secrets=[p.key for p in downloader.providers if p.key]
        def prepare(job,name,dest,method,executed_job=None):
            executed_job=executed_job or job
            dest.mkdir(parents=True,exist_ok=True)
            attempt=dest/('attempt_'+str(len(list(dest.glob('attempt_*')))+1).zfill(3))
            recorder=ResponseRecorder(attempt,secrets)
            window = executed_job.window
            requested_range = {'requested_start':str(window.start) if window else None,
                               'requested_end':str(window.end) if window else None}
            metadata = {**executed_job.metadata(),**requested_range,
                'configured_operation':job.operation,'cob':str(cob),'provider':name,'retrieval_method':method,
                'provider_input':executed_job.instrument.provider_inputs.get(name),'provider_options':executed_job.provider_options}
            # Each attempt keeps its own execution inputs. resolved_inputs.json
            # remains the original archive identity, even after a compatible retry.
            write_json(attempt/'request.json',metadata)
            write_json(dest/'request.json',metadata)
            return recorder,{'request_id':job.request_id,'ticker':job.instrument.ticker,'provider':name,'operation':executed_job.operation,'configured_operation':job.operation,
                **requested_range,'cob':str(cob),'retrieval_method':method,'folder':str(dest.relative_to(folder)),'attempt':str(attempt.relative_to(folder))}
        def finish(dest,row,recorder):
            row.update(finished=True,responses_saved=len(recorder.responses))
            write_json(dest/'status.json',row)
            records[:]=[r for r in records if (r['provider'],r['request_id'])!=(row['provider'],row['request_id'])]
            records.append(row)
            label='LIVE SNAPSHOT' if row['retrieval_method']=='reqMktData_snapshot_live' else 'HISTORICAL'
            progress(row['provider']+' ['+label+'] '+row['ticker']+': '+row['status'])
        # Independent HTTP providers first; each failure is recorded without
        # turning an unavailable provider into a successful cached dataset.
        blocked={}
        clients={p.name:p.client for p in downloader.providers}
        for job,name,dest in pending:
            if name=='ibkr':continue
            recorder,row=prepare(job,name,dest,'historical_http')
            provider=PROVIDERS[name]({**settings,**job.provider_options},client=clients[name])
            previous_sink=provider.client.response_sink
            provider.client.response_sink=recorder
            try:
                if name in blocked:raise RetrievalError('provider_paused',blocked[name])
                status,message=_http_history(provider,job)
                row.update(status=status,message=message)
            except RetrievalError as exc:
                row.update(status=exc.code,message=recorder.scrub(str(exc)))
                if exc.code=='unsupported_operation':row['api_request_sent']=False
                if exc.code in {'network_error','rate_limited','missing_key','authentication_error'}:blocked[name]=str(exc)
            except Exception as exc:row.update(status='provider_error',message=type(exc).__name__+'; inspect saved responses.')
            finally:
                row.update(getattr(provider, 'history_info', {}))
                provider.client.response_sink=previous_sink
            finish(dest,row,recorder)
        ib_jobs={job.request_id:(job,dest) for job,name,dest in pending if name=='ibkr'}
        historical=[];live=[]
        for job,dest in ib_jobs.values():
            route=option_route(cob,execution_date) if job.instrument.kind=='option' else 'historical'
            if route=='live':live.append(job)
            else:historical.append(job)
        if historical:
            from .providers.ibkr_history import iter_history
            for result in iter_history(historical,settings,progress, fallback_max=True):
                job=result['request'];dest=ib_jobs[job.request_id][1]
                recorder,row=prepare(job,'ibkr',dest,'reqHistoricalData')
                for payload in result['payloads']:
                    recorder(json.dumps(payload,allow_nan=False).encode(),None,'application/json','ibkr://historical/'+payload.get('method','response'))
                row.update(result['info'],status=result['status'],message=result['message'])
                if job.instrument.kind=='option':
                    row['historical_model_iv_available']=False
                    row['historical_greeks_available']=False
                    row['option_route']='historical'
                finish(dest,row,recorder)
        if live:
            from .providers.ibkr import IBKRProvider
            capture_date=execution_date
            live_settings={**settings,'IBKR_LIVE_EXPECTED_DATE':str(capture_date)}
            provider=IBKRProvider(live_settings)
            converted=[replace(j,operation='option_model_snapshot',window=None,frequency='snapshot',provider_options={
                'IBKR_OPTION_MARKET_DATA_TYPE':'1','IBKR_OPTION_CAPTURE_SECONDS':settings.get('IBKR_OPTION_CAPTURE_SECONDS','60')}) for j in live]
            progress('COB is yesterday: capturing current live options dated '+str(capture_date)+'.')
            for result in provider.iter_option_models(converted,progress):
                original,dest=ib_jobs[result['request'].request_id]
                recorder,row=prepare(original,'ibkr',dest,'reqMktData_snapshot_live',executed_job=result['request'])
                if result['body'] is not None:recorder(result['body'],None,'application/json',result['url'])
                error=result['error']
                row.update(result['info'],status=error.code if error else 'response_saved',
                    message=str(error) if error else 'Live option model captured; separate from historical COB prices.',
                    live_capture_date=str(capture_date),option_route='live',
                    historical_option_query_skipped='COB is yesterday; date policy prioritizes current live options.')
                if result['body'] is not None:
                    import math
                    payload=json.loads(result['body']);ticker=payload.get('ticker',{});greeks=ticker.get('modelGreeks') or {}
                    required=('impliedVol','optPrice','undPrice','delta','gamma','vega','theta','pvDividend')
                    missing=[k for k in required if isinstance(greeks.get(k),bool) or not isinstance(greeks.get(k),(int,float)) or not math.isfinite(greeks[k]) or abs(greeks[k])>=1e100]
                    for k in ('impliedVol','optPrice','undPrice','gamma','vega','pvDividend'):
                        if k not in missing and greeks[k]<0:missing.append(k)
                    if 'delta' not in missing and not -1<=greeks['delta']<=1:missing.append('delta')
                    row['missing_model_fields']=missing
                    row['required_model_fields_complete']=not missing
                    if missing and row['status']=='response_saved':row['status']='partial_model_data'
                if capture_date!=london_today():row.update(status='live_date_mismatch',message='Capture crossed the requested live-date boundary; inspect individual timestamps.')
                finish(dest,row,recorder)
        write_json(summary_file,records)
        counts=pd.Series([r['status'] for r in records]).value_counts().to_dict()
        write_json(manifest_file,{**base,'complete':True,'all_requests_succeeded':all(r['status']=='response_saved' for r in records),
            'completed_at':datetime.now(ZoneInfo('Europe/London')).isoformat(),'request_count':len(records),'status_counts':{k:int(v) for k,v in counts.items()}})
        return CobResult(folder,pd.DataFrame(records))
    finally:
        handle.close()
        lock.unlink(missing_ok=True)
