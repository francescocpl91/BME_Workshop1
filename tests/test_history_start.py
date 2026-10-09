"""User dates and provider fallbacks, using only recorded-shape fake responses."""
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import MarketDataDownloader
from src.cob import parse_history_start
from src.config import parse_requests, PROVIDERS
from src.history import fetch_history
from src.json_config import load_provider_configs
from src.models import DateWindow, Instrument, RetrievalError
from src.providers.ibkr_history import iter_history
from test_ibkr_history import HistoricalIB
from test_cob_integrity import capture
from test_raw_dump import config

ROOT = Path(__file__).resolve().parents[1]
COB = date(2026,10,8)
START = date(2015,1,2)


def job(name):
    return SimpleNamespace(instrument=Instrument('SPY','price',{name:'SPY'},'USD'),
                           data_type='etf_price',window=DateWindow(START,COB))


class HistoryStartTests(unittest.TestCase):
    def test_validation_and_required_placeholder(self):
        self.assertEqual(parse_history_start('02/01/2015', COB), START)
        for value in ('2015-01-02','2/1/2015','31/02/2015','09/10/2026','31/12/1899'):
            with self.subTest(value=value), self.assertRaises(ValueError):parse_history_start(value, COB)
        with self.assertRaisesRegex(ValueError,'history_start'):
            load_provider_configs(ROOT/'config/yahoo.json',cob_date=COB)

    def test_start_applies_to_all_providers_but_not_options(self):
        jobs=parse_requests(load_provider_configs(ROOT/'config',cob_date=COB,history_start=START))
        providers=set()
        for j in jobs:
            self.assertEqual(j.window.end,COB)
            self.assertEqual(j.window.start,COB if j.instrument.kind=='option' else START)
            providers.update(j.providers)
        self.assertEqual(providers,set(PROVIDERS))

    def test_main_prompts_twice_and_rejects_start_after_cob(self):
        import run
        import pandas as pd
        result=SimpleNamespace(folder=Path('fixture'),reused=False,summary=pd.DataFrame(columns=['ticker','provider','status','retrieval_method']))
        with patch.object(run,'london_today',return_value=date(2026,10,9)), patch('builtins.print'), \
             patch('builtins.input',side_effect=['08/10/2026','09/10/2026','02/01/2015']) as prompt, \
             patch.object(run.MarketDataDownloader,'from_json') as factory:
            factory.return_value.download_cob.return_value=result
            run.main()
            self.assertEqual(prompt.call_count,3)
            self.assertEqual(factory.call_args.kwargs['history_start'],START)
            self.assertEqual(factory.call_args.kwargs['cob_date'],COB)

    def test_custom_json_start_survives_cob_and_wider_request_rejects_archive(self):
        initial=config({'yahoo':'SAMPLE'})
        initial['requests'][0]['timeframe']={'start':str(START),'end':str(COB)}
        dl=MarketDataDownloader.from_config(initial)
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=date(2026,10,9)), \
             patch('src.cob._http_history',side_effect=capture) as api:
            result=dl.download_cob(COB,tmp)
            self.assertEqual(api.call_args.args[1].window,DateWindow(START,COB))
            before={p:p.read_bytes() for p in result.folder.rglob('*') if p.is_file()}
            self.assertTrue(dl.download_cob(COB,tmp).reused)
            with self.assertRaisesRegex(ValueError,'different inputs'):
                dl.download_cob(COB,tmp,history_start=date(2010,1,1))
            self.assertEqual(before,{p:p.read_bytes() for p in result.folder.rglob('*') if p.is_file()})
            api.assert_called_once()

    def test_success_does_not_expand_window_and_transport_errors_do_not_retry(self):
        for name in ('yahoo','alpha_vantage','marketdata','massive','fred','ecb'):
            for error in (None,'network_error','authentication_error','missing_key','rate_limited'):
                p=PROVIDERS[name]({},client=Mock())
                with self.subTest(name=name,error=error), patch.object(p,'fetch_raw') as api, patch('src.history.check_request_support'):
                    if error:
                        api.side_effect=RetrievalError(error,'fixture')
                        with self.assertRaises(RetrievalError):fetch_history(p,job(name))
                    else:self.assertEqual(fetch_history(p,job(name))[0],'response_saved')
                    api.assert_called_once()
                    self.assertEqual(api.call_args.args[2],DateWindow(START,COB))
                    self.assertFalse(p.history_info['history_fallback_used'])

    def test_fred_ecb_fallback_omits_start_but_preserves_cob(self):
        for name in ('fred','ecb'):
            p=PROVIDERS[name]({},client=Mock())
            with self.subTest(name=name), patch.object(p,'fetch_raw',side_effect=[RetrievalError('no_data','fixture'),None]) as api, patch('src.history.check_request_support'):
                self.assertEqual(fetch_history(p,job(name))[0],'response_saved')
                self.assertIsNone(api.call_args.args[2].start)
                self.assertEqual(api.call_args.args[2].end,COB)
                self.assertTrue(p.history_info['history_fallback_used'])

    def test_native_reference_requests_really_omit_start(self):
        for name,symbol in [('fred','DGS10'),('ecb','EXR/D.USD.EUR.SP00.A')]:
            client=Mock()
            client.get_json.side_effect=[{'seriess':[{}]},{'observations':[{'date':'2026-10-08','value':'4'}],'count':1}]
            client.get.return_value='TIME_PERIOD,OBS_VALUE\n2026-10-08,1.1\n'
            p=PROVIDERS[name]({},client=client)
            with self.subTest(name=name), patch.object(p,'check_access'):
                p.fetch_raw(None,symbol,SimpleNamespace(start=None,end=COB))
                args=(client.get_json if name=='fred' else client.get).call_args.kwargs['params']
                self.assertNotIn('observation_start' if name=='fred' else 'startPeriod',args)
                self.assertEqual(args['observation_end' if name=='fred' else 'endPeriod'],str(COB))

    def test_yahoo_fallback_uses_first_trade_and_never_future_listing(self):
        for first in (date(1993,1,29),date(2027,1,1)):
            client=Mock();client.get_json.return_value={'chart':{'result':[{'meta':{'firstTradeDate':int(datetime.combine(first,datetime.min.time(),timezone.utc).timestamp())}}]}}
            p=PROVIDERS['yahoo']({},client=client)
            with self.subTest(first=first), patch.object(p,'fetch_raw',side_effect=[RetrievalError('no_data','fixture'),None]) as api:
                if first>COB:
                    with self.assertRaises(RetrievalError):fetch_history(p,job('yahoo'))
                    self.assertEqual(api.call_count,1)
                else:
                    fetch_history(p,job('yahoo'))
                    self.assertEqual(api.call_args.args[2],DateWindow(first,COB))

    def test_alpha_full_then_compact_is_labelled_limited(self):
        p=PROVIDERS['alpha_vantage']({},client=Mock());sizes=[]
        def fetch(*args):
            sizes.append(p.settings.get('ALPHA_VANTAGE_OUTPUTSIZE'))
            if len(sizes)<3:raise RetrievalError('entitlement_error','fixture')
        with patch.object(p,'fetch_raw',side_effect=fetch):
            self.assertEqual(fetch_history(p,job('alpha_vantage'))[0],'limited_history')
        self.assertEqual(sizes[-2:],['full','compact'])

    def test_marketdata_pages_back_without_from_and_records_boundary(self):
        client=Mock()
        stamp=int(datetime(2025,1,2,tzinfo=timezone.utc).timestamp())
        client.get_json.side_effect=[{'s':'ok','t':[stamp]}]+[{'s':'no_data'}]*5
        p=PROVIDERS['marketdata']({},client=client)
        with patch.object(p,'check_access'),patch.object(p,'fetch_raw',side_effect=RetrievalError('invalid_request','range')):
            self.assertEqual(fetch_history(p,job('marketdata'))[0],'limited_history')
        calls=client.get_json.call_args_list
        self.assertEqual(calls[0].kwargs['params']['to'],str(COB))
        self.assertEqual(calls[1].kwargs['params']['to'],'2025-01-01')
        self.assertTrue(all('from' not in c.kwargs['params'] for c in calls))
        self.assertEqual(p.history_info['history_stop_reason'],'first_unavailable_countback')

    def test_massive_rejected_window_then_backward_history_boundary(self):
        p=PROVIDERS['massive']({},client=Mock())
        with patch.object(p,'fetch_raw',side_effect=[RetrievalError('invalid_request','range'),None,RetrievalError('no_data','boundary')]) as api:
            self.assertEqual(fetch_history(p,job('massive'))[0],'limited_history')
        windows=[c.args[2] for c in api.call_args_list]
        self.assertEqual(windows[0],DateWindow(START,COB))
        self.assertEqual(windows[1].end,COB)
        self.assertLess(windows[2].end,windows[1].start)
        self.assertTrue(p.history_info['history_fallback_used'])

    def test_ibkr_retries_range_failure_from_head_but_not_network_failure(self):
        inputs=dict(symbol='SPY',secType='STK',exchange='SMART',currency='USD',primaryExchange='ARCA')
        instrument=Instrument('SPY','price',{'ibkr':'SPY'},'USD',provider_inputs={'ibkr':inputs})
        j=SimpleNamespace(request_id='SPY',instrument=instrument,contract=None,window=DateWindow(date(2026,10,1),COB),provider_options={})
        class PriceIB(HistoricalIB):
            async def reqContractDetailsAsync(self,contract):
                contract.conId=123
                return [SimpleNamespace(contract=deepcopy(contract))]
            async def reqHeadTimeStampAsync(self,*args,**kwargs):return datetime(2025,1,2,tzinfo=timezone.utc)
            async def reqHistoricalDataAsync(self,contract,**kwargs):
                if not self.calls:
                    self.calls.append(kwargs)
                    raise RetrievalError(self.failure,'fixture')
                return await super().reqHistoricalDataAsync(contract,**kwargs)
        for failure in ('no_data','network_error'):
            ib=PriceIB();ib.failure=failure
            with self.subTest(failure=failure),patch('ib_insync.IB',return_value=ib):
                result=list(iter_history([j],{'IBKR_HISTORY_REQUEST_INTERVAL':'.1'},lambda _:None,fallback_max=True))[0]
            self.assertEqual(ib.calls[0]['durationStr'],'8 D')
            self.assertEqual(result['info']['history_fallback_used'],failure=='no_data')
            if failure=='no_data':
                self.assertEqual(result['info']['history_fallback_start'],'2025-01-02')
                self.assertEqual(result['status'],'response_saved')
            else:self.assertEqual(len(ib.calls),1)

    def test_fallback_details_are_saved_alongside_original_request(self):
        import json
        from src.providers.yahoo import YahooProvider
        initial=config({'yahoo':'SAMPLE'})
        initial['requests'][0]['timeframe']={'start':str(START),'end':str(COB)}
        dl=MarketDataDownloader.from_config(initial)
        first=int(datetime(1993,1,29,tzinfo=timezone.utc).timestamp())
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=date(2026,10,9)), \
             patch.object(YahooProvider,'fetch_raw',side_effect=[RetrievalError('no_data','range'),None]), \
             patch.object(dl.providers[0].client,'get_json',return_value={'chart':{'result':[{'meta':{'firstTradeDate':first}}]}}):
            result=dl.download_cob(COB,tmp)
            row=result.summary.iloc[0]
            saved=json.loads((result.folder/row['folder']/'status.json').read_text())
            request=json.loads((result.folder/row['attempt']/'request.json').read_text())
            self.assertEqual(request['requested_start'],str(START))
            self.assertTrue(saved['history_fallback_used'])
            self.assertEqual(saved['history_attempts'][-1]['start'],'1993-01-29')
            self.assertEqual(saved['history_attempts'][-1]['end'],str(COB))

    def test_empty_success_payload_triggers_maximum_fallback(self):
        for name, payload in [('yahoo', {'chart':{'result':[{'timestamp':None}], 'error':None}}),
                              ('marketdata', {'s':'ok','t':[]})]:
            client=Mock();client.get_json.return_value=payload
            p=PROVIDERS[name]({},client=client)
            with self.subTest(name=name), patch.object(p,'check_access'), \
                 patch('src.history._maximum',return_value=('response_saved','maximum fixture')) as maximum:
                self.assertEqual(fetch_history(p,job(name))[0],'response_saved')
                maximum.assert_called_once()
                self.assertEqual(p.history_info['history_fallback_reason'],'no_data')

    def test_failed_maximum_retains_failure_and_does_not_try_a_fixed_date(self):
        for name in ('fred','ecb','yahoo','alpha_vantage','marketdata','massive'):
            p=PROVIDERS[name]({},client=Mock())
            with self.subTest(name=name), patch('src.history.check_request_support'), \
                 patch.object(p,'fetch_raw',side_effect=RetrievalError('no_data','initial')) as requested, \
                 patch('src.history._maximum',side_effect=RetrievalError('entitlement_error','maximum denied')) as maximum:
                with self.assertRaises(RetrievalError):fetch_history(p,job(name))
                requested.assert_called_once()
                maximum.assert_called_once()
                self.assertEqual(p.history_info['history_fallback_status'],'entitlement_error')
                self.assertEqual(requested.call_args.args[2],DateWindow(START,COB))

    def test_marketdata_rejects_a_page_containing_future_observations(self):
        stamps=[int(datetime(2025,1,2,tzinfo=timezone.utc).timestamp()),
                int(datetime(2027,1,2,tzinfo=timezone.utc).timestamp())]
        client=Mock();client.get_json.return_value={'s':'ok','t':stamps}
        p=PROVIDERS['marketdata']({},client=client)
        with patch.object(p,'check_access'), patch.object(p,'fetch_raw',side_effect=RetrievalError('no_data','initial')):
            with self.assertRaises(RetrievalError) as error:fetch_history(p,job('marketdata'))
        self.assertEqual(error.exception.code,'invalid_response')
