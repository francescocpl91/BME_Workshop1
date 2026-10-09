"""Archive identity and recorder-lifetime regressions; no network calls."""
from copy import deepcopy
from datetime import date
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import MarketDataDownloader
from src.models import RetrievalError
from test_cob import fixture, history
from test_raw_dump import config

TODAY = date(2026, 10, 8)


def capture(provider, job):
    provider.client.capture_response(b'{"fixture":true}', 200, 'application/json',
                                     'https://example.invalid/history')
    return 'response_saved', 'Fixture response.'


class CobIntegrityTests(unittest.TestCase):
    def test_changed_effective_adjustment_rejects_reuse_and_retry(self):
        initial = config({'marketdata': 'SAMPLE'})
        initial['settings']['MARKETDATA_ADJUSTMENT'] = 'split_only'
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY), patch('src.cob._http_history', side_effect=capture) as api:
            result = MarketDataDownloader.from_config(initial).download_cob('06/10/2026', tmp)
            before = (result.folder / 'manifest.json').read_bytes()
            changed = deepcopy(initial)
            changed['settings']['MARKETDATA_ADJUSTMENT'] = 'split_and_dividend'
            for retry in (False, True):
                with self.subTest(retry=retry), self.assertRaisesRegex(ValueError, 'different inputs'):
                    MarketDataDownloader.from_config(changed).download_cob('06/10/2026', tmp, retry_failed=retry)
            api.assert_called_once()
            self.assertEqual((result.folder / 'manifest.json').read_bytes(), before)

    def test_credentials_and_overridden_defaults_do_not_invalidate_archive(self):
        initial = config({'marketdata': 'SAMPLE'})
        initial['settings'].update(MARKETDATA_ADJUSTMENT='split_only', MARKETDATA_API_TOKEN='fixture-old-key')
        initial['requests'][0]['provider_options'] = {'MARKETDATA_ADJUSTMENT': 'split_only'}
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY), patch('src.cob._http_history', side_effect=capture) as api:
            result = MarketDataDownloader.from_config(initial).download_cob('06/10/2026', tmp)
            changed = deepcopy(initial)
            changed['settings'].update(MARKETDATA_ADJUSTMENT='split_and_dividend', MARKETDATA_API_TOKEN='fixture-new-key', MARKETDATA_MIN_INTERVAL='2')
            self.assertTrue(MarketDataDownloader.from_config(changed).download_cob('06/10/2026', tmp).reused)
            api.assert_called_once()
            inputs = (result.folder / 'resolved_inputs.json').read_text(encoding='utf-8')
            self.assertNotIn('fixture-old-key', inputs)
            self.assertNotIn('MARKETDATA_API_TOKEN', inputs)
            self.assertEqual(json.loads(inputs)['data_settings']['marketdata']['SAMPLE'],
                             {'MARKETDATA_ADJUSTMENT': 'split_only'})

    def test_option_capture_settings_are_part_of_identity(self):
        initial = fixture()
        initial['settings'] = {'IBKR_OPTION_CAPTURE_SECONDS': '30'}
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY), patch('src.providers.ibkr_history.iter_history', side_effect=history) as api:
            MarketDataDownloader.from_config(initial).download_cob('06/10/2026', tmp)
            initial['settings']['IBKR_OPTION_CAPTURE_SECONDS'] = '60'
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                MarketDataDownloader.from_config(initial).download_cob('06/10/2026', tmp)
            api.assert_called_once()

    def test_legacy_archive_is_preserved_without_api_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob._http_history') as api, patch('src.cob.london_today', return_value=TODAY):
            folder = Path(tmp) / '2026-10-06'
            folder.mkdir()
            manifest = folder / 'manifest.json'
            original = b'{"schema_version":2,"complete":true,"config_hash":"legacy"}'
            manifest.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'archive inputs are missing'):
                MarketDataDownloader.from_config(config({'yahoo': 'SAMPLE'})).download_cob('06/10/2026', tmp)
            api.assert_not_called()
            self.assertEqual(manifest.read_bytes(), original)

    def test_compatible_legacy_archive_reused_without_changing_any_files(self):
        initial = config({'yahoo': 'SAMPLE'})
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY), patch('src.cob._http_history', side_effect=capture) as api:
            dl = MarketDataDownloader.from_config(initial)
            result = dl.download_cob('06/10/2026', tmp)
            inputs_file = result.folder / 'resolved_inputs.json'
            saved = json.loads(inputs_file.read_text())
            saved['schema'] = 2
            saved.pop('data_settings')
            saved['requests'][0]['timeframe']['start'] = '1900-01-01'
            inputs_file.write_text(json.dumps(saved))
            manifest_file = result.folder / 'manifest.json'
            manifest = json.loads(manifest_file.read_text())
            manifest.update(schema_version=2, config_hash=hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest())
            manifest_file.write_text(json.dumps(manifest))
            before = {f: f.read_bytes() for f in result.folder.rglob('*') if f.is_file()}
            with patch('src.cob.london_today', return_value=date(2026,10,10)):
                reused = dl.download_cob('06/10/2026', tmp)
            self.assertTrue(reused.reused)
            api.assert_called_once()
            self.assertEqual(before, {f: f.read_bytes() for f in result.folder.rglob('*') if f.is_file()})
            changed = deepcopy(initial)
            changed['requests'][0]['symbols']['yahoo'] = 'DIFFERENT'
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                MarketDataDownloader.from_config(changed).download_cob('06/10/2026', tmp)
            api.assert_called_once()

    def test_schema_three_broader_history_is_reused(self):
        initial = config({'yahoo': 'SAMPLE'})
        with tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY), patch('src.cob._http_history', side_effect=capture) as api:
            dl = MarketDataDownloader.from_config(initial)
            result = dl.download_cob('06/10/2026', tmp, history_start=date(1900,1,1))
            before = (result.folder / 'manifest.json').read_bytes()
            self.assertTrue(dl.download_cob('06/10/2026', tmp).reused)
            self.assertEqual(before, (result.folder / 'manifest.json').read_bytes())
            api.assert_called_once()

    def test_massive_max_history_sends_supported_fallback_date(self):
        from src.cob import _http_history
        from src.config import parse_requests
        from src.json_config import load_provider_configs
        from src.providers.massive import MassiveProvider
        root = Path(__file__).resolve().parents[1]
        jobs = parse_requests(load_provider_configs(root/'config/massive.json', cob_date=date(2026,10,7),history_start=date(2000,1,3)))
        job = next(j for j in jobs if j.instrument.ticker == 'IEF')
        client = Mock()
        client.get_json.return_value = {'status':'OK', 'results':[{'t':1791326400000,'c':89.11}]}
        provider = MassiveProvider({}, client=client)
        with patch.object(provider, 'check_access'):
            self.assertEqual(_http_history(provider, job)[0], 'response_saved')
        self.assertIn('/2000-01-03/2026-10-07', client.get_json.call_args.args[0])

    def test_retry_records_new_range_without_overwriting_original_attempt(self):
        for schema in (2, 3):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY):
                dl = MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
                seen = []
                def failed(provider, job):
                    seen.append(job.window)
                    capture(provider, job)
                    raise RetrievalError('network_error', 'Fixture interruption after a response')
                with patch('src.cob._http_history', side_effect=failed):
                    first = dl.download_cob('06/10/2026', tmp, history_start=date(1900,1,1))
                inputs_file = first.folder / 'resolved_inputs.json'
                identity = json.loads(inputs_file.read_text())
                if schema == 2:
                    identity['schema'] = 2
                    identity.pop('data_settings')
                    from src.storage import write_json
                    write_json(inputs_file, identity)
                    manifest_file = first.folder / 'manifest.json'
                    manifest = json.loads(manifest_file.read_text())
                    manifest.update(schema_version=2, config_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
                    write_json(manifest_file, manifest)
                original_inputs = inputs_file.read_bytes()
                original_attempt = first.folder / first.summary.iloc[0]['attempt']
                original_files = {f:f.read_bytes() for f in original_attempt.iterdir()}
                def success(provider, job):
                    seen.append(job.window)
                    return capture(provider, job)
                with patch('src.cob._http_history', side_effect=success) as api:
                    result = dl.download_cob('06/10/2026', tmp, retry_failed=True, history_start=date(2000,1,3))
                api.assert_called_once()
                self.assertEqual([w.start for w in seen],[date(1900,1,1),date(2000,1,3)])
                row = result.summary.iloc[0]
                self.assertEqual(row['requested_start'],'2000-01-03')
                self.assertEqual(row['requested_end'],'2026-10-06')
                self.assertNotEqual(row['attempt'],first.summary.iloc[0]['attempt'])
                for path in [result.folder/row['attempt']/'request.json',result.folder/row['folder']/'request.json',result.folder/row['folder']/'status.json']:
                    saved = json.loads(path.read_text())
                    self.assertEqual(saved['requested_start'],'2000-01-03')
                    self.assertEqual(saved['requested_end'],'2026-10-06')
                self.assertEqual(original_inputs,inputs_file.read_bytes())
                self.assertEqual(original_files,{f:f.read_bytes() for f in original_attempt.iterdir()})

    def test_http_recorder_is_restored_on_success_failure_and_interruption(self):
        for failure in (None, RetrievalError('network_error', 'Fixture failure'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp, patch('src.cob.london_today', return_value=TODAY):
                dl = MarketDataDownloader.from_config(config({'yahoo': 'SAMPLE'}))
                client = dl.providers[0].client
                original_sink = Mock()
                client.response_sink = original_sink
                def fetch(provider, job):
                    result = capture(provider, job)
                    if failure is not None:
                        raise failure
                    return result
                with patch('src.cob._http_history', side_effect=fetch):
                    if isinstance(failure, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            dl.download_cob('06/10/2026', tmp)
                    else:
                        dl.download_cob('06/10/2026', tmp)
                self.assertIs(client.response_sink, original_sink)
                archive = {p: p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
                original_sink.assert_not_called()
                client.capture_response(b'{"later":true}', 200, 'application/json', 'https://example.invalid/later')
                original_sink.assert_called_once()
                self.assertEqual(archive, {p: p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()})
