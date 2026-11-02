"""Offline reporting tests: no current portfolio settings or provider requests."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest
import numpy as np
from test_local_vol_monte_carlo import settings,flat_adapter,quotes
from src.local_vol_monte_carlo import simulate_cohort,validate_local_vol,load_config
from src.local_vol_mc_reporting import PathRecorder,assets,settings as report_settings,token


def config():
    return {**settings(),'reporting':dict(plots=True,sample_paths=12,time_points=7,save_all_paths=True)}


def fixtures(folder):
    a=flat_adapter();q=quotes(a)
    local=SimpleNamespace(run_id='local',source_folder=folder,
        manifest=dict(identity=dict(source_surface_id='surface'),archive_cob='2031-02-02'),
        tables=dict(state_mapping=[dict(group_id='group',underlying='ANY',currency='USD',route='historical')],option_repricing=q),state_evaluators={'group':a})
    return local,SimpleNamespace(run_id='surface',tables={'repricing':q},
        manifest=dict(archive_cob='2031-02-02',identity=dict(
            calibration_identity=dict(config=dict(valuation_timezone='UTC')))))


class ReportingTests(unittest.TestCase):
    def test_malformed_existing_manifest_is_a_controlled_integrity_error(self):
        # A corrupt saved manifest must not leak AttributeError through run.py
        # or be overwritten. Use a tiny isolated result, not a user's archive.
        from src.local_vol_monte_carlo import ValidationResult
        with TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'local_vol_mc'/'fixture';target.mkdir(parents=True)
            result=ValidationResult(root,'fixture',dict(identity={'config':{}},summary={}),{})
            for invalid in ([],None,{'complete':True,'identity':{'config':{}},'files':[]},
                            {'complete':True,'identity':{'config':{}},'files':{'report':[]}}):
                (target/'manifest.json').write_text(json.dumps(invalid),encoding='utf-8')
                before=(target/'manifest.json').read_bytes()
                with self.subTest(invalid=invalid),self.assertRaisesRegex(ValueError,'mc_existing_output'):
                    result.save()
                self.assertEqual(before,(target/'manifest.json').read_bytes())

    def test_capture_does_not_change_rng_prices_or_acceptance(self):
        a=flat_adapter();cfg=config();q=quotes(a)
        expected=simulate_cohort(a,q,cfg,101)
        recorder=PathRecorder(a,cfg)
        actual=simulate_cohort(a,q,cfg,101,recorder=recorder)
        self.assertEqual(expected,actual)
        tables,data=recorder.finish()
        self.assertEqual(data['log_normalized_paths'].shape[1],cfg['paths'])
        self.assertEqual(len(tables['terminal_paths']),cfg['paths'])
        self.assertTrue(np.all(np.diff(data['years'])>0))
        for t in [0,*a.boundary.times]:self.assertTrue(np.any(np.isclose(data['years'],t)))
        terminal=np.exp(data['log_normalized_paths'][-1])*data['forward'][-1]
        np.testing.assert_allclose(terminal,[r['underlying_price'] for r in tables['terminal_paths']])
        self.assertAlmostEqual(tables['path_summary'][-1]['mean'],terminal.mean())
        self.assertAlmostEqual(tables['path_summary'][-1]['p01'],np.quantile(terminal,.01))
        for r in tables['sample_paths']:
            j=int(np.argmin(abs(data['years']-r['years'])))
            self.assertAlmostEqual(r['underlying_price'],np.exp(data['log_normalized_paths'][j,r['path_id']])*data['forward'][j])

    def test_expiries_preserved_even_with_two_requested_observations(self):
        cfg=config();cfg['reporting']['time_points']=2
        a=flat_adapter();rec=PathRecorder(a,cfg);simulate_cohort(a,quotes(a),cfg,2,recorder=rec)
        tables,data=rec.finish();np.testing.assert_allclose(data['years'],[0,.25,.5])
        self.assertEqual(len(tables['path_summary']),3)

    def test_older_config_and_disabled_reporting_preserve_small_outputs(self):
        a=flat_adapter();rec=PathRecorder(a,settings());rec.record(0,np.zeros(2))
        self.assertEqual(rec.finish(),({},None))
        cfg=config();cfg['reporting'].update(plots=False,sample_paths=0,save_all_paths=False)
        rec=PathRecorder(a,cfg);self.assertEqual(rec.finish(),({},None))

    def test_validation_rejects_invalid_reporting_and_excessive_storage(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.json'
            for value in (None,{},'yes',dict(plots=1,sample_paths=2,time_points=3,save_all_paths=True),
                dict(plots=True,sample_paths=True,time_points=3,save_all_paths=True),
                dict(plots=True,sample_paths=2,time_points=1,save_all_paths=True)):
                p.write_text(json.dumps({**settings(),'reporting':value}),encoding='utf-8')
                with self.assertRaises(ValueError):load_config(p)
        cfg=config();cfg['paths']=2_000_000;cfg['reporting']['time_points']=250
        with self.assertRaisesRegex(ValueError,'storage_too_large'):PathRecorder(flat_adapter(),cfg)

    def test_completed_review_required_run_has_plots_npz_and_immutable_checksums(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);local,surface=fixtures(folder);cfg=config()
            cfg.update(maximum_se_forward_fraction=1e-12,maximum_se_relative_price=1e-12)
            p=folder/'cfg.json';p.write_text(json.dumps(cfg),encoding='utf-8')
            with patch('socket.socket.connect',side_effect=AssertionError('network')):
                result=validate_local_vol(local,surface,p);target=result.save()
            self.assertEqual(result.tables['cohorts'][0]['status'],'review_required')
            self.assertEqual(len(list(target.glob('*.png'))),3)
            for image in target.glob('*.png'):self.assertEqual(image.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            with np.load(target/('paths_'+token('group')+'.npz'),allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['log_normalized_paths'],result.path_data['group']['log_normalized_paths'])
            text=(target/'report.html').read_text(encoding='utf-8')
            self.assertIn('review_required',text);self.assertIn('Every launch runs Monte Carlo',text)
            self.assertIn('terminal_paths.csv',text)
            old={f:f.read_bytes() for f in target.iterdir()}
            self.assertEqual(result.save(),target);self.assertTrue(all(f.read_bytes()==b for f,b in old.items()))
            npz=target/('paths_'+token('group')+'.npz');npz.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):result.save()

    def test_new_validation_recalculates_even_when_output_already_exists(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);local,surface=fixtures(folder);cfg=config();cfg['reporting']['plots']=False
            p=folder/'cfg.json';p.write_text(json.dumps(cfg),encoding='utf-8')
            with patch('src.local_vol_monte_carlo.simulate_cohort',wraps=simulate_cohort) as simulate:
                first=validate_local_vol(local,surface,p);target=first.save()
                second=validate_local_vol(local,surface,p)
                self.assertEqual(second.save(),target);self.assertEqual(simulate.call_count,2)
            self.assertEqual(first.run_id,second.run_id)

    def test_failed_cohort_does_not_export_partial_paths_as_complete(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);local,surface=fixtures(folder)
            local.state_evaluators['group'].bounds=(-1e-10,1e-10)
            p=folder/'cfg.json';p.write_text(json.dumps(config()),encoding='utf-8')
            result=validate_local_vol(local,surface,p)
            self.assertEqual(result.tables['cohorts'][0]['status'],'unavailable')
            self.assertEqual(result.path_data,{})
            self.assertNotIn('path_summary',result.tables);self.assertEqual(assets(result),{})

    def test_save_all_paths_can_be_disabled_without_disabling_bands(self):
        cfg=config();cfg['reporting']['save_all_paths']=False
        a=flat_adapter();r=PathRecorder(a,cfg);simulate_cohort(a,quotes(a),cfg,12,recorder=r)
        tables,data=r.finish();self.assertIsNone(data);self.assertTrue(tables['path_summary'])
        self.assertEqual(len(tables['terminal_paths']),cfg['paths'])
