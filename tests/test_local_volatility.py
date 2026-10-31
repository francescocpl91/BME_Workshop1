"""Offline economic/numerical invariants; independent of user JSON and archives."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch, Mock
import hashlib
import json
import unittest

import numpy as np

from src.local_volatility import LocalVolEvaluator, convert_surface, load_local_vol_config, _sample_interval
from src.surface_model import SurfaceModel, SviSurface


def settings():
    return dict(schema_version=1, enabled=True, plots=False,
        grid=dict(maturities_per_interval=3,moneyness_points=7),
        validation=dict(step_fraction=.001,relative_tolerance=.001,absolute_variance_tolerance=1e-6))


def flat_model():
    return SurfaceModel([.25,.5,.75],[.01,.02,.03],0.,0.,.5,[-.2]*3,[.2]*3)


def permission(model):
    return dict(allowed=True,purpose='bounded_prototype_dupire_input',maturity_knots='excluded',extrapolation='forbidden',
        warnings=['Intraday approximation; €STR conventions assumed'],reference_timestamps=['2031-02-03T12:00:00Z'],
        domain=[dict(start_years=a,end_years=b,k_min=-.15,k_max=.15,start_expiry='start',end_expiry='end')
                for a,b in zip(model.times[:-1],model.times[1:])])


def source(folder, model=None, route='live'):
    model=model or flat_model()
    return SimpleNamespace(folder=folder,run_id='surface-fixture',
        manifest=dict(identity=dict(config=dict(derivative_diagnostics=dict(minimum_density_factor=.05,negative_tolerance=1e-9))),archive_cob='2031-02-02'),
        models={'group':model},tables=dict(cohort_status=[dict(group_id='group',underlying='ANY_ASSET',currency='USD',
            route=route,valuation_date='2031-02-03',local_vol_eligibility=permission(model))]))


@dataclass
class QuadraticVariance:
    times: list=field(default_factory=lambda:[.25,.5,.75])
    def interval_domain(self,a,b):return -.2,.2
    def evaluate(self,k,t):
        k,t=np.broadcast_arrays(np.asarray(k,float),np.asarray(t,float))
        w=.04*t+.02*t*t
        return dict(w=w,w_t=.04+.04*t,w_k=np.zeros_like(w),w_kk=np.zeros_like(w))


class LocalVolTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cfg=settings();self.config=self.root/'local.json'
        self.surface=source(self.root)

    def convert(self):
        self.config.write_text(json.dumps(self.cfg),encoding='utf-8')
        return convert_surface(self.surface,self.config)

    def test_constant_black_volatility_recovers_twenty_percent(self):
        result=self.convert()
        self.assertEqual(result.tables['cohorts'][0]['status'],'validated_on_supported_grid')
        np.testing.assert_allclose([r['local_volatility'] for r in result.tables['local_vol_grid']],.2,atol=1e-12)
        self.assertTrue(all(r['status']=='passed' for r in result.tables['validation']))
        self.assertAlmostEqual(result.evaluators['group'].evaluate(.4,.1)['local_volatility'],.2)
        self.assertFalse(result.manifest['summary']['full_simulation_ready'])

    def test_time_varying_flat_smile_recovers_variance_derivative(self):
        self.surface=source(self.root,QuadraticVariance())
        result=self.convert()
        self.assertEqual(result.tables['cohorts'][0]['status'],'validated_on_supported_grid')
        for r in result.tables['local_vol_grid']:
            self.assertAlmostEqual(r['local_variance'],.04+.04*r['years'])

    def test_raw_svi_and_ssvi_both_convert(self):
        self.surface=source(self.root,SviSurface([.25,.5,.75],
            [[.005,.04,-.5,.02,.15],[.012,.07,-.6,.04,.17],[.02,.11,-.55,.06,.2]],[-.2]*3,[.2]*3))
        result=self.convert()
        self.assertEqual(result.tables['cohorts'][0]['status'],'validated_on_supported_grid')
        self.assertEqual(result.evaluators['group'].evaluate(.4,0.)['status'],'available')

    def test_point_evaluator_blocks_knots_wings_and_invalid_numbers(self):
        result=self.convert();ev=result.evaluators['group']
        for t,k in [(0.,0.),(.25,0.),(.5,0.),(.75,0.),(1.,0.),(.4,.151),(.4,-.151),
                    (True,0.),('0.4',0.),(.4,float('nan')),(float('inf'),0.)]:
            with self.subTest(t=t,k=k):self.assertEqual(ev.evaluate(t,k)['status'],'unavailable')
        self.assertEqual(ev.evaluate(.4,.15)['status'],'available')

    def test_blocked_permission_does_not_evaluate_the_model(self):
        self.surface.tables['cohort_status'][0]['local_vol_eligibility'].update(allowed=False,blockers=['missing_data'])
        with patch.object(self.surface.models['group'],'evaluate',side_effect=AssertionError('must not evaluate')):
            result=self.convert()
        self.assertEqual(result.tables['cohorts'][0]['status'],'skipped')
        self.assertEqual(result.tables['local_vol_grid'],[])
        self.assertEqual(result.evaluators,{})

    def test_disabled_and_no_option_groups_are_safe(self):
        self.cfg['enabled']=False
        self.assertEqual(self.convert().tables['cohorts'][0]['reason'],'local_vol_disabled')
        self.surface.tables['cohort_status']=[];self.surface.models={}
        result=self.convert();self.assertEqual(result.manifest['summary']['validated_cohorts'],0)
        self.assertTrue((result.save()/'report.html').is_file())

    def test_historical_and_live_follow_same_math_without_inventing_spreads(self):
        live=self.convert();self.surface=source(self.root,route='historical');historical=self.convert()
        self.assertNotEqual(live.run_id,historical.run_id)
        np.testing.assert_equal([r['local_variance'] for r in live.tables['local_vol_grid']],
                                [r['local_variance'] for r in historical.tables['local_vol_grid']])
        self.assertTrue(all(r['route']=='historical' for r in historical.tables['local_vol_grid']))

    def test_invalid_derivatives_are_not_clipped(self):
        ev=LocalVolEvaluator(self.surface.models['group'],permission(self.surface.models['group']),.05,1e-9)
        base=dict(w=.016,w_k=0.,w_kk=0.,w_t=.04)
        for field,value,reason in [('w',0.,'nonpositive'),('w_t',-.01,'negative_time'),
                                  ('w_kk',-1.98,'density_below'),('w_k',float('nan'),'nonfinite')]:
            with patch.object(ev.model,'evaluate',return_value={**base,field:value}):
                r=ev.evaluate(.4,0.);self.assertEqual(r['status'],'unavailable');self.assertIn(reason,r['reason'])
        with patch.object(ev.model,'evaluate',return_value={**base,'w_t':-5e-10}):
            r=ev.evaluate(.4,0.);self.assertEqual(r['local_variance'],0.);self.assertTrue(r['roundoff_adjusted'])
        with patch.object(ev.model,'evaluate',return_value={**base,'w_t':10.}):
            self.assertGreater(ev.evaluate(.4,0.)['local_volatility'],3.)

    def test_price_based_validation_detects_wrong_analytic_derivative(self):
        model=self.surface.models['group'];original=model.evaluate
        def corrupted(k,t):
            values=original(k,t);values['w_t']=values['w_t']*1.2;return values
        with patch.object(model,'evaluate',side_effect=corrupted):result=self.convert()
        self.assertEqual(result.tables['cohorts'][0]['status'],'review_required')
        self.assertEqual(result.evaluators,{})
        self.assertTrue(any(r['status']=='failed' for r in result.tables['validation']))

    def test_narrow_domain_stencils_remain_inside_each_interval(self):
        model=flat_model();perm=permission(model)
        for d in perm['domain']:d.update(k_min=-.001,k_max=.001)
        ev=LocalVolEvaluator(model,perm,.05,1e-9)
        for d in perm['domain']:
            rows,checks=_sample_interval(ev,d,settings())
            self.assertTrue(all(d['start_years']<r['years']-r['time_step']<r['years']+r['time_step']<d['end_years'] for r in checks))
            self.assertTrue(all(d['k_min']<r['k']-r['k_step']<r['k']+r['k_step']<d['k_max'] for r in checks))

    def test_overbroad_or_cross_knot_permission_is_rejected(self):
        for change in (dict(k_min=-.3),dict(end_years=.7)):
            self.surface=source(self.root)
            self.surface.tables['cohort_status'][0]['local_vol_eligibility']['domain'][0].update(change)
            result=self.convert();self.assertEqual(result.tables['cohorts'][0]['status'],'unavailable')
            self.assertEqual(result.tables['local_vol_grid'],[])

    def test_config_validation_is_strict_and_not_portfolio_dependent(self):
        for change in [('enabled',1),('schema_version',True),('grid',{}),('validation',{}),('plots','yes')]:
            cfg=settings();cfg[change[0]]=change[1];self.config.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_local_vol_config(self.config)
        for key in settings()['validation']:
            self.cfg=settings();self.cfg['validation'][key]=-1
            with self.assertRaises(ValueError):self.convert()

    def test_no_network_and_no_mutation_and_content_sensitive_identity(self):
        before=deepcopy(self.surface.tables)
        with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):result=self.convert()
        self.assertEqual(self.surface.tables,before)
        self.cfg['grid']['moneyness_points']=9
        self.assertNotEqual(result.run_id,self.convert().run_id)
        self.cfg=settings();self.surface.tables['cohort_status'][0]['local_vol_eligibility']['warnings'].append('new warning')
        self.assertNotEqual(result.run_id,self.convert().run_id)

    def test_completed_export_reuses_without_writes_and_rejects_corruption(self):
        result=self.convert();target=result.save()
        def signatures():return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in target.iterdir()}
        before=signatures();self.assertEqual(result.save(),target);self.assertEqual(before,signatures())
        html=(target/'report.html').read_text(encoding='utf-8');self.assertIn('€STR',html)
        self.assertIn('Full pricing and simulation are not ready',html)
        manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['files'].values():self.assertEqual(hashlib.sha256((target/item['file']).read_bytes()).hexdigest(),item['sha256'])
        (target/'local_vol_grid.csv').write_text('corrupted')
        with self.assertRaisesRegex(ValueError,'checksum'):result.save()

    def test_failed_report_is_retryable_and_old_partial_is_preserved(self):
        result=self.convert();target=result.source_folder/'local_volatility'/result.run_id
        with patch('src.local_volatility._report',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):result.save()
        self.assertFalse(target.exists())
        target.mkdir();(target/'old.txt').write_text('preserve')
        result.save();manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual((target.parent/manifest['recovered_incomplete_output']/'old.txt').read_text(),'preserve')

    def test_html_escapes_metadata_and_png_is_checksummed(self):
        self.cfg['plots']=True
        self.surface.tables['cohort_status'][0]['underlying']='<script>bad</script>'
        result=self.convert();target=result.save();html=(target/'report.html').read_text(encoding='utf-8')
        self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
        self.assertEqual(len(list(target.glob('*.png'))),2)
        three_d=next(target.glob('*_3d.png'))
        self.assertIn(three_d.name,html)
        manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['files']['plot3d_group']['sha256'],hashlib.sha256(three_d.read_bytes()).hexdigest())
        self.assertGreater(next(target.glob('*.png')).stat().st_size,1000)

    def test_run_stage_skips_missing_and_exposes_unexpected_errors(self):
        import run
        with redirect_stdout(StringIO()):self.assertIsNone(run._local_vol_available(None))
        with patch.object(run,'LOCAL_VOL_FILE',self.root/'absent'),redirect_stdout(StringIO()):
            self.assertIsNone(run._local_vol_available(self.surface))
        with patch('src.local_volatility.convert_surface',side_effect=RuntimeError('bug')):
            with self.assertRaises(RuntimeError):run._local_vol_available(self.surface)

    def test_run_stage_passes_current_surface_and_does_not_call_mc_or_downloader(self):
        import run
        self.config.write_text(json.dumps(self.cfg))
        with patch.object(run,'LOCAL_VOL_FILE',self.config),patch.object(run,'simulate_processed',side_effect=AssertionError('no MC')), \
             patch('socket.socket.connect',side_effect=AssertionError('no downloads')),redirect_stdout(StringIO()):
            result=run._local_vol_available(self.surface)
        self.assertEqual(result.manifest['identity']['source_surface_id'],self.surface.run_id)
        self.assertEqual(result.manifest['summary']['validated_cohorts'],1)

    def test_pipeline_order_preserves_existing_historical_analysis(self):
        import run
        processed=Mock(run_id='fixture',summary={'portfolio':{'holdings':[]},'readiness':{'aligned_return_observations':0,'foreign_currencies':[]},'options':{'iv_candidates':0}})
        processed.save.return_value=self.root
        stats=Mock(run_id='stats',summary={'portfolio_status':'unavailable','warnings':[]})
        mc=Mock(run_id='mc',summary={'status':'unavailable','warnings':[]});order=[]
        def local(value):self.assertIs(value,self.surface);order.append('local')
        def statistics(*args):order.append('statistics');return stats
        def simulation(*args):order.append('historical_mc');return mc
        with patch('src.volatility_workflow.save_overview',return_value=Path('.')),patch.object(run,'process_inventory',return_value=processed),patch.object(run,'_calibrate_available'), \
             patch.object(run,'_surface_available',return_value=self.surface),patch.object(run,'_local_vol_available',side_effect=local), \
             patch.object(run,'analyze_processed',side_effect=statistics),patch.object(run,'simulate_processed',side_effect=simulation),redirect_stdout(StringIO()):
            run._process_inventory(SimpleNamespace(cob='2031-02-02'))
        self.assertEqual(order,['local','statistics','historical_mc'])


if __name__=='__main__':unittest.main()
