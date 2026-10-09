"""Manual OIS inputs preserve source quotes, observation dates and archive identity."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import MarketDataDownloader
from src.ois import load_ois_config, ois_fingerprint, plan_ois_dump
from test_raw_dump import config
from test_cob_integrity import capture

ROOT=Path(__file__).resolve().parents[1]
COB=date(2026,10,8)


class OisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.document=load_ois_config(ROOT/'config/ois.json')

    def test_fixed_curves_are_included_for_every_cob_without_relabelling(self):
        cases=((date(2025,1,1),'after_cob'),(date(2026,10,7),'after_cob'),
               (COB,'after_cob'),(date(2026,10,9),'same_as_cob'),
               (date(2026,10,10),'before_cob'),(date(2027,12,31),'before_cob'))
        for cob,relation in cases:
            with self.subTest(cob=cob):
                planned=plan_ois_dump(self.document,cob)
                self.assertEqual([r[2]['quote_count'] for r in planned],[86,44])
                for region,(_,payload,row) in zip(('us','euro'),planned):
                    self.assertEqual(row['status'],'response_saved')
                    self.assertEqual(row['curve_as_of_date'],'2026-10-09')
                    self.assertEqual(row['observation_date_relation'],relation)
                    self.assertEqual(row['usage_policy'],'fixed_for_all_cobs')
                    self.assertEqual(row['suitable_for_strict_cob'],relation=='same_as_cob')
                    self.assertEqual(payload['curve'],self.document['curves'][region][0])
                    self.assertFalse(payload['discount_factors_calculated'])

    def test_fixed_curves_save_and_reuse_in_archives_before_and_after_source_date(self):
        dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
        class ExecutionDate(date):
            @classmethod
            def today(cls):return date(2026,10,12)
        with tempfile.TemporaryDirectory() as tmp,patch('src.cob.london_today',return_value=date(2026,10,12)),patch('src.models.date',ExecutionDate):
            for cob in (date(2026,10,1),date(2026,10,9),date(2026,10,11)):
                with self.subTest(cob=cob),patch('src.cob._http_history',side_effect=capture) as api:
                    result=dl.download_cob(cob,tmp,ois_file=ROOT/'config/ois.json')
                    api.assert_called_once()
                    for region in ('us','euro'):
                        saved=json.loads((result.folder/'ois'/region/'curve.json').read_text(encoding='utf-8'))
                        self.assertEqual(saved['curve'],self.document['curves'][region][0])
                    with patch('src.cob._http_history',side_effect=AssertionError('API forbidden')),patch('src.cob.write_json',side_effect=AssertionError('Write forbidden')),patch('src.ois.write_json',side_effect=AssertionError('OIS write forbidden')):
                        self.assertTrue(dl.download_cob(cob,tmp,ois_file=ROOT/'config/ois.json').reused)

    def test_fixed_policy_requires_exactly_one_curve_per_region(self):
        for snapshots in ([],self.document['curves']['us']*2):
            doc=deepcopy(self.document);doc['curves']['us']=snapshots
            with self.subTest(count=len(snapshots)),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'ois.json';path.write_text(json.dumps(doc),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'exactly one'):load_ois_config(path)

    def test_unicode_archive_reuse_and_retry_with_windows_default_encoding(self):
        doc=deepcopy(self.document);doc['source']['label']='\u20acSTR - taux europ\u00e9ens'
        original_read=Path.read_text
        def windows_read(path,encoding=None,errors=None):
            return original_read(path,encoding=encoding or 'cp1252',errors=errors)
        def unicode_capture(provider,job):
            status,_=capture(provider,job)
            return status,'Saved \u20acSTR fixture.'
        with tempfile.TemporaryDirectory() as tmp,patch('src.cob.london_today',return_value=date(2026,10,9)):
            path=Path(tmp)/'ois.json';path.write_text(json.dumps(doc,ensure_ascii=False),encoding='utf-8')
            dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
            with patch('src.cob._http_history',side_effect=unicode_capture):
                result=dl.download_cob(COB,Path(tmp)/'dump',ois_file=path)
            before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in result.folder.rglob('*') if p.is_file()}
            with patch.object(Path,'read_text',windows_read),patch('src.cob._http_history',side_effect=AssertionError('API forbidden')):
                with patch('src.cob.write_json',side_effect=AssertionError('Write forbidden')),patch('src.ois.write_json',side_effect=AssertionError('OIS write forbidden')):
                    reused=dl.download_cob(COB,Path(tmp)/'dump',ois_file=path)
                    self.assertTrue(reused.reused)
                    self.assertEqual(reused.summary.set_index('ticker').loc['SAMPLE','message'],'Saved \u20acSTR fixture.')
                self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in result.folder.rglob('*') if p.is_file()})
                retried=dl.download_cob(COB,Path(tmp)/'dump',ois_file=path,retry_failed=True)
                self.assertEqual(retried.summary.set_index('ticker').loc['SAMPLE','message'],'Saved \u20acSTR fixture.')
            saved=json.loads((result.folder/'ois/euro/curve.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['source']['label'],doc['source']['label'])

    def test_invalid_quotes_fail_before_provider_calls_or_dump_writes(self):
        for field,value in [('rate_percent',True),('rate_percent',float('nan')),('tenor','1X')]:
            doc=deepcopy(self.document);doc['curves']['us'][0]['points'][0][field]=value
            with self.subTest(field=field,value=value),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'ois.json';path.write_text(json.dumps(doc))
                dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
                with patch('src.cob.london_today',return_value=date(2026,10,9)),patch('src.cob._http_history') as api:
                    with self.assertRaises(ValueError):dl.download_cob(COB,Path(tmp)/'dump',ois_file=path)
                api.assert_not_called();self.assertFalse((Path(tmp)/'dump').exists())

    def test_negative_rates_valid_duplicate_dates_and_wrong_units_rejected(self):
        doc=deepcopy(self.document);doc['curves']['us'][0]['points'][0]['rate_percent']=-0.1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ois.json';path.write_text(json.dumps(doc))
            self.assertEqual(load_ois_config(path)['curves']['us'][0]['points'][0]['rate_percent'],-0.1)
            doc['curves']['us'].append(deepcopy(doc['curves']['us'][0]));path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):load_ois_config(path)
            doc['curves']['us'].pop();doc['curves']['us'][0]['rate_unit']='decimal';path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):load_ois_config(path)

    def test_combined_dump_reuse_and_changed_ois_protection(self):
        with tempfile.TemporaryDirectory() as tmp,patch('src.cob.london_today',return_value=date(2026,10,9)):
            path=Path(tmp)/'ois.json';path.write_text(json.dumps(self.document))
            dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
            with patch('src.cob._http_history',side_effect=capture) as api:
                result=dl.download_cob(COB,Path(tmp)/'dump',ois_file=path)
            api.assert_called_once();self.assertEqual(len(result.summary),3)
            self.assertTrue((result.folder/'ois/us/curve.json').exists())
            self.assertTrue((result.folder/'ois/euro/curve.json').exists())
            before={p:p.read_bytes() for p in result.folder.rglob('*') if p.is_file()}
            with patch('src.cob._http_history',side_effect=AssertionError('API forbidden')),patch('src.cob.write_json',side_effect=AssertionError('Write forbidden')),patch('src.ois.write_json',side_effect=AssertionError('OIS write forbidden')):
                self.assertTrue(dl.download_cob(COB,Path(tmp)/'dump',ois_file=path).reused)
                doc=deepcopy(self.document);doc['curves']['us'][0]['points'][0]['rate_percent']+=0.001
                path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError,'different inputs'):dl.download_cob(COB,Path(tmp)/'dump',ois_file=path)
            self.assertEqual(before,{p:p.read_bytes() for p in result.folder.rglob('*') if p.is_file()})

    def test_corrupted_manual_archive_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp,patch('src.cob.london_today',return_value=date(2026,10,9)),patch('src.cob._http_history',side_effect=capture):
            dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
            result=dl.download_cob(COB,tmp,ois_file=ROOT/'config/ois.json')
            curve=result.folder/'ois/us/curve.json'
            curve.write_text('{}')
            with patch('src.cob._http_history',side_effect=AssertionError('API forbidden')),patch('src.cob.write_json',side_effect=AssertionError('Write forbidden')):
                with self.assertRaisesRegex(ValueError,'Saved OIS input'):dl.download_cob(COB,tmp,ois_file=ROOT/'config/ois.json')

    def test_manual_quotes_saved_even_if_api_unavailable(self):
        from src.models import RetrievalError
        with tempfile.TemporaryDirectory() as tmp,patch('src.cob.london_today',return_value=date(2026,10,9)),patch('src.cob._http_history',side_effect=RetrievalError('network_error','fixture')):
            dl=MarketDataDownloader.from_config(config({'yahoo':'SAMPLE'}))
            result=dl.download_cob(COB,tmp,ois_file=ROOT/'config/ois.json')
            rows=result.summary.set_index('ticker')
            self.assertEqual(rows.loc['USD_SOFR_OIS','status'],'response_saved')
            self.assertEqual(rows.loc['EUR_ESTR_OIS','status'],'response_saved')
            self.assertEqual(rows.loc['SAMPLE','status'],'network_error')

    def test_instructions_do_not_change_quote_fingerprint(self):
        doc=deepcopy(self.document);doc['instructions']['editing']='Updated wording'
        self.assertEqual(ois_fingerprint(doc),ois_fingerprint(self.document))

    def test_disabled_manual_input_and_malformed_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ois.json'
            disabled=deepcopy(self.document);disabled['enabled']=False
            path.write_text(json.dumps(disabled));self.assertIsNone(load_ois_config(path))
            bad=deepcopy(self.document);bad['curves']['us'][0]['points'][0]=None
            for value in ([],bad):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):load_ois_config(path)
