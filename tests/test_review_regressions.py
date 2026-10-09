"""Regressions from the independent pre-run review; no live provider calls."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import MarketDataDownloader
from src.history import fetch_history
from src.models import RetrievalError, DateWindow
from src.providers.yahoo import YahooProvider
from test_history_start import job
from test_raw_dump import config
from test_cob_integrity import capture


class ReviewRegressions(unittest.TestCase):
    def test_narrow_retry_cannot_satisfy_original_broader_request(self):
        dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=date(2026,10,9)):
            with patch('src.cob._http_history',side_effect=RetrievalError('network_error','fixture')):
                first=dl.download_cob('07/10/2026',tmp,history_start=date(2000,1,3))
            identity=(first.folder/'resolved_inputs.json').read_bytes()
            with patch('src.cob._http_history',side_effect=capture):
                narrow=dl.download_cob('07/10/2026',tmp,history_start=date(2010,1,4),retry_failed=True)
            before={p:p.read_bytes() for p in first.folder.rglob('*') if p.is_file()}
            with patch('src.cob._http_history',side_effect=AssertionError('No API allowed')):
                with self.assertRaisesRegex(ValueError,'saved attempt.*range'):
                    dl.download_cob('07/10/2026',tmp,history_start=date(2000,1,3))
                self.assertTrue(dl.download_cob('07/10/2026',tmp,history_start=date(2010,1,4)).reused)
            self.assertEqual(before,{p:p.read_bytes() for p in first.folder.rglob('*') if p.is_file()})
            with patch('src.cob._http_history',side_effect=capture) as api:
                repaired=dl.download_cob('07/10/2026',tmp,history_start=date(2000,1,3),retry_failed=True)
            api.assert_called_once()
            self.assertEqual(api.call_args.args[1].window.start,date(2000,1,3))
            self.assertEqual((first.folder/'resolved_inputs.json').read_bytes(),identity)
            self.assertNotEqual(repaired.summary.iloc[0]['attempt'],narrow.summary.iloc[0]['attempt'])
            for p,body in before.items():
                if 'attempt_' in str(p):self.assertEqual(p.read_bytes(),body)

    def test_disabled_direct_config_entries_are_ignored(self):
        cfg=config({'yahoo':'SAMPLE'})
        disabled=deepcopy(cfg['requests'][0]);disabled.update(id='DISABLED',enabled=False)
        cfg['requests'].append(disabled)
        # Disabled incomplete entries must be ignored just as parse_requests ignores them.
        cfg['requests'].append({'enabled':False})
        dl=MarketDataDownloader.from_config(cfg)
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today',return_value=date(2026,10,9)),patch('src.cob._http_history',side_effect=capture) as api:
            result=dl.download_cob('07/10/2026',tmp)
            self.assertEqual(len(result.summary),1)
            api.assert_called_once()
            saved=json.loads((result.folder/'resolved_inputs.json').read_text())
            self.assertEqual(len(saved['requests']),1)

    def test_yahoo_pre_epoch_fallback_is_portable(self):
        expected=date(1927,12,30)
        seconds=int((datetime(1927,12,30,tzinfo=timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())
        client=Mock();client.get_json.return_value={'chart':{'result':[{'meta':{'firstTradeDate':seconds}}]}}
        provider=YahooProvider({},client=client)
        class WindowsDatetime(datetime):
            @classmethod
            def fromtimestamp(cls, *args, **kwargs):
                raise OSError(22,'Simulated Windows negative timestamp failure')
        with patch('src.history.datetime',WindowsDatetime),patch.object(provider,'fetch_raw',side_effect=[RetrievalError('no_data','fixture'),None]) as api:
            self.assertEqual(fetch_history(provider,job('yahoo'))[0],'response_saved')
        self.assertEqual(api.call_args.args[2].start,expected)

    def test_yahoo_native_request_can_send_pre_epoch_start(self):
        client=Mock();client.get_json.return_value={'chart':{'result':[{'timestamp':[-1325462400]}]}}
        provider=YahooProvider({},client=client)
        provider.fetch_raw(None,'^GSPC',DateWindow(date(1927,12,30),date(2026,10,7)))
        self.assertLess(client.get_json.call_args.kwargs['params']['period1'],0)

    def test_readme_direct_use_configuration_loads_offline(self):
        from src.json_config import load_provider_configs
        root=Path(__file__).resolve().parents[1]
        text=(root/'README.md').read_text()
        snippet=text.split('Direct library use:')[1].split('```python')[1].split('```')[0]
        namespace={}
        original=MarketDataDownloader.from_json
        def load(paths,**kwargs):return original(root/paths,**kwargs)
        with patch.object(MarketDataDownloader,'from_json',side_effect=load), \
             patch.object(MarketDataDownloader,'download_cob',return_value='offline') as download:
            exec(snippet,namespace)
        self.assertEqual(namespace['result'],'offline')
        download.assert_called_once()
