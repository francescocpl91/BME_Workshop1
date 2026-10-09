"""COB routing and persistent archive tests. No external API calls."""
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import MarketDataDownloader
from src.cob import parse_cob, option_route
from src.json_config import load_provider_configs

ROOT=Path(__file__).resolve().parents[1]
TODAY=date(2026,10,8)


def fixture(count=1):
    doc=json.loads((ROOT/'config/ibkr.json').read_text())
    doc['requests']=[r for r in doc['requests'] if r.get('contract',{}).get('expiration')=='2026-12-18'][:count]
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'ibkr.json';path.write_text(json.dumps(doc))
        return load_provider_configs(path,cob_date=date(2026,10,7),history_start=date(2000,1,3))


def history(jobs,*args,**kwargs):
    for job in jobs:
        yield dict(request=job,payloads=[{'method':'reqHistoricalData','bars':[]}],info={},status='response_saved',message='Fixture bars.')


def live(jobs,*args):
    for job in jobs:
        body={'ticker':{'modelGreeks':dict(impliedVol=.2,optPrice=30.,undPrice=7800.,delta=.5,gamma=.01,vega=2.,theta=-1.,pvDividend=0.)}}
        yield dict(request=job,body=json.dumps(body).encode(),url='ibkr://snapshot',info={},error=None)


class CobTests(unittest.TestCase):
    def test_strict_date_and_calendar_boundaries(self):
        self.assertEqual(parse_cob('07/10/2026',today=TODAY),date(2026,10,7))
        for text in ('2026-10-07','7/10/2026','31/02/2026','08/10/2026','09/10/2026','31/12/1899'):
            with self.subTest(text=text), self.assertRaises(ValueError):parse_cob(text,today=TODAY)
        self.assertEqual(option_route(date(2026,12,31),date(2027,1,1)),'live')
        self.assertEqual(option_route(date(2026,10,6),TODAY),'historical')

    def test_option_route_accepts_equivalent_date_objects_and_strings(self):
        today=date(2026,10,9)
        for value,expected in ((date(2026,10,8),'live'),('08/10/2026','live'),
                               (' 08/10/2026 ','live'),(date(2026,10,7),'historical'),
                               ('07/10/2026','historical')):
            with self.subTest(value=value):self.assertEqual(option_route(value,today),expected)
        self.assertEqual(option_route('31/12/2026',date(2027,1,1)),'live')
        for value in ('09/10/2026','10/10/2026','31/02/2026','2026-10-08'):
            with self.subTest(value=value),self.assertRaises(ValueError):option_route(value,today)

    def test_yesterday_live_and_archive_reused_even_on_later_day(self):
        dl=MarketDataDownloader.from_config(fixture())
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY), patch('src.providers.ibkr_history.iter_history') as old, patch('src.providers.ibkr.IBKRProvider.iter_option_models',side_effect=live) as current:
            result=dl.download_cob('07/10/2026',tmp)
            old.assert_not_called();current.assert_called_once()
            row=result.summary.iloc[0]
            self.assertEqual(row['cob'],'2026-10-07');self.assertEqual(row['live_capture_date'],'2026-10-08')
            self.assertEqual(row['retrieval_method'],'reqMktData_snapshot_live')
            self.assertEqual(row['operation'],'option_model_snapshot')
            self.assertEqual(row['configured_operation'],'option_history')
            saved=json.loads((result.folder/row['folder']/'request.json').read_text())
            self.assertEqual(saved['operation'],'option_model_snapshot')
            self.assertIsNone(saved['requested_start'])
            self.assertIsNone(saved['requested_end'])
            self.assertEqual(saved,json.loads((result.folder/row['attempt']/'request.json').read_text()))
            self.assertEqual(saved['provider_options']['IBKR_OPTION_MARKET_DATA_TYPE'],'1')
            self.assertTrue(row['required_model_fields_complete'])
            with patch('src.cob.london_today',return_value=date(2026,10,9)):
                reused=dl.download_cob('07/10/2026',tmp)
            self.assertTrue(reused.reused);current.assert_called_once();old.assert_not_called()

    def test_older_date_only_history_and_iv_absence_is_explicit(self):
        dl=MarketDataDownloader.from_config(fixture())
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY), patch('src.providers.ibkr_history.iter_history',side_effect=history) as old, patch('src.providers.ibkr.IBKRProvider.iter_option_models') as current:
            result=dl.download_cob('06/10/2026',tmp)
            old.assert_called_once();current.assert_not_called()
            self.assertFalse(result.summary.iloc[0]['historical_model_iv_available'])
            self.assertEqual(old.call_args.args[0][0].window.end,date(2026,10,6))

    def test_history_failure_never_switches_to_live_and_is_cached(self):
        def failed(jobs,*args,**kwargs):
            for r in history(jobs):
                r.update(status='entitlement_error');yield r
        dl=MarketDataDownloader.from_config(fixture())
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY), patch('src.providers.ibkr_history.iter_history',side_effect=failed) as old, patch('src.providers.ibkr.IBKRProvider.iter_option_models') as current:
            result=dl.download_cob('06/10/2026',tmp)
            manifest=json.loads((result.folder/'manifest.json').read_text())
            self.assertTrue(manifest['complete']);self.assertFalse(manifest['all_requests_succeeded'])
            self.assertTrue(dl.download_cob('06/10/2026',tmp).reused)
            current.assert_not_called();old.assert_called_once()

    def test_config_mismatch_never_calls_apis(self):
        config=fixture();dl=MarketDataDownloader.from_config(config)
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY), patch('src.providers.ibkr_history.iter_history',side_effect=history) as old:
            dl.download_cob('06/10/2026',tmp)
            config['requests'][0]['description']='Changed input'
            changed=MarketDataDownloader.from_config(config)
            with self.assertRaisesRegex(ValueError,'different inputs'):changed.download_cob('06/10/2026',tmp)
            old.assert_called_once()

    def test_interrupted_archive_resumes_without_repeating_saved_request(self):
        def interrupted(jobs,*args,**kwargs):
            yield next(history(jobs));raise KeyboardInterrupt()
        dl=MarketDataDownloader.from_config(fixture(2))
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY):
            with patch('src.providers.ibkr_history.iter_history',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):dl.download_cob('06/10/2026',tmp)
            self.assertFalse((Path(tmp)/'2026-10-06'/'.running').exists())
            with patch('src.providers.ibkr_history.iter_history',side_effect=history) as remaining:
                result=dl.download_cob('06/10/2026',tmp)
                self.assertEqual(len(remaining.call_args.args[0]),1)
            self.assertEqual(len(result.summary),2)

    def test_missing_live_fields_saved_as_partial_not_success(self):
        def partial(jobs,*args):
            for r in live(jobs):
                r['body']=b'{"ticker":{"modelGreeks":{"impliedVol":0.2,"undPrice":-1}}}';yield r
        dl=MarketDataDownloader.from_config(fixture())
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY), patch('src.providers.ibkr.IBKRProvider.iter_option_models',side_effect=partial):
            result=dl.download_cob('07/10/2026',tmp)
            self.assertEqual(result.summary.iloc[0]['status'],'partial_model_data')
            self.assertIn('undPrice',result.summary.iloc[0]['missing_model_fields'])

    def test_market_inputs_include_spy_and_user_start_placeholder(self):
        for name in ('yahoo','alpha_vantage','marketdata','massive','ibkr'):
            doc=json.loads((ROOT/'config'/f'{name}.json').read_text())
            prices=[r for r in doc['requests'] if r['product'] in {'index','etf','stock'}]
            self.assertEqual({r['ticker'] for r in prices},{'SPX','IEF','SPY'})
            self.assertTrue(all(r['timeframe']=={'start':'HISTORY_START','end':'COB'} for r in prices))
            self.assertTrue(next(r for r in prices if r['ticker']=='IEF')['enabled'])
            if name!='ibkr':self.assertFalse(any(r.get('contract') for r in doc['requests']))

    def test_cob_driver_passes_selected_window_to_every_price_provider(self):
        from contextlib import ExitStack
        from src.config import PROVIDERS
        config=load_provider_configs(ROOT/'config',cob_date=date(2026,10,7),history_start=date(2000,1,3))
        config['requests']=[r for r in config['requests'] if r['kind']=='price']
        dl=MarketDataDownloader.from_config(config)
        seen=[]
        def http(provider,job):
            seen.append((provider.name,job.instrument.ticker,job.window));return 'response_saved','fixture'
        def ib(jobs,*args,**kwargs):
            for job in jobs:seen.append(('ibkr',job.instrument.ticker,job.window))
            yield from history(jobs)
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=TODAY), patch('src.cob._http_history',side_effect=http),patch('src.providers.ibkr_history.iter_history',side_effect=ib):
            dl.download_cob('06/10/2026',tmp)
        self.assertEqual({p for p,t,w in seen},{'yahoo','alpha_vantage','marketdata','massive','ibkr'})
        self.assertTrue(all(w.start==date(2000,1,3) and w.end==date(2026,10,6) for p,t,w in seen))
        self.assertEqual(sum(t=='IEF' for p,t,w in seen),5)

    def test_enabled_unsupported_indexes_save_json_and_do_not_block_etfs(self):
        from src.config import PROVIDERS
        for name in ('alpha_vantage','marketdata'):
            doc=json.loads((ROOT/'config'/f'{name}.json').read_text())
            self.assertTrue(all(r['enabled'] for r in doc['requests']))
            with self.subTest(provider=name),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/f'{name}.json';path.write_text(json.dumps(doc))
                dl=MarketDataDownloader.from_json(path,cob_date=date(2026,10,6),history_start=date(2000,1,3))
                def fetch(provider,instrument,symbol,window):
                    self.assertIn(instrument.ticker,{'IEF','SPY'})
                    provider.client.capture_response(b'{"fixture":"ETF history"}',200,'application/json','https://example.invalid/history')
                with patch('src.cob.london_today',return_value=TODAY),patch.object(PROVIDERS[name],'fetch_raw',autospec=True,side_effect=fetch) as api:
                    result=dl.download_cob('06/10/2026',Path(tmp)/'dump')
                self.assertEqual(api.call_count,2)
                rows=result.summary.set_index('ticker')
                self.assertEqual(rows.loc['IEF','status'],'response_saved')
                self.assertEqual(rows.loc['SPX','status'],'unsupported_operation')
                status_path=result.folder/rows.loc['SPX','folder']/'status.json'
                status=json.loads(status_path.read_text())
                self.assertIs(status['api_request_sent'],False)
                self.assertEqual(status['responses_saved'],0)
                self.assertTrue((status_path.parent/'request.json').exists())
