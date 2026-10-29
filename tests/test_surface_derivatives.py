"""Independent models and temporary fixtures; no archive or live settings dependency."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

from src.surface_checks import (compare_derivatives,maturity_jumps,validate_policy,
                                     assessment_status,refit_assessment)
from src.volatility_surface import load_surface_config,fit_calibrated
from test_surface_quality import practical_policy
from test_volatility_surface import fixture
from test_svi_surface import known_model
from src.surface_model import slice_values,black_partials


def policy():
    return dict(enabled=True,strike_points=11,relative_tolerance=.25,
        absolute_tolerances=dict(w_k=.001,w_kk=.01,w_t=.001,density_factor=.01),
        maximum_p95_score=1.,maximum_score=4.,minimum_density_factor=.05,
        negative_tolerance=1e-9,jump_step_fraction=.0001,jump_convergence_score=.01)


class Analytic:
    """Flat Black variance with optional tiny, high-frequency strike ripples."""
    times=[.2,.6,1.]
    def __init__(self,amplitude=0.,bounds=(-.2,.2),variance=.04):
        self.amplitude=amplitude;self.bounds=bounds;self.variance=variance
    def domain(self,t):return self.bounds
    def interval_domain(self,start,end):return self.bounds
    def evaluate(self,k,t):
        k=np.asarray(k);a=self.amplitude;n=200
        w=self.variance*t+a*np.cos(n*k);wk=-a*n*np.sin(n*k);wkk=-a*n*n*np.cos(n*k)
        g=(1-k*wk/(2*w))**2-.25*wk**2*(1/w+.25)+.5*wkk
        return dict(w=w,iv=np.sqrt(w/t),w_k=wk,w_kk=wkk,w_t=np.full_like(k,self.variance),density_factor=g)


class DerivativeTests(unittest.TestCase):
    def test_identical_smooth_models_have_zero_changes_and_no_jumps(self):
        model=Analytic();result=compare_derivatives(model,model,policy())
        self.assertEqual(result['status'],'passed')
        self.assertTrue(all(r['maximum_absolute_change']==0 for r in result['metrics']))
        jumps=maturity_jumps(model,policy())
        self.assertEqual(jumps['status'],'passed')
        self.assertTrue(all(r['limit_estimate_converged'] for r in jumps['metrics']))

    def test_tiny_iv_ripples_can_have_large_curvature_sensitivity(self):
        a=Analytic();b=Analytic(amplitude=.00001)
        ks=np.linspace(-.2,.2,100)
        self.assertLess(np.max(abs(a.evaluate(ks,.6)['iv']-b.evaluate(ks,.6)['iv'])),.0001)
        result=compare_derivatives(a,b,policy())
        self.assertEqual(result['status'],'review_required')
        curvature=next(r for r in result['metrics'] if r['metric']=='w_kk')
        self.assertEqual(curvature['status'],'REVIEW_REQUIRED')

    def test_narrower_domain_is_explicit_and_never_extrapolated(self):
        a=Analytic();b=Analytic(bounds=(-.1,.1));b.times=[.4,.8]
        result=compare_derivatives(a,b,policy())
        self.assertFalse(result['full_time_range'])
        self.assertTrue(all(abs(r['k'])<=.1 and .4<r['years']<.8 for r in result['points']))
        self.assertTrue(all(r['baseline_wings_excluded'] for r in result['domains']))

    def test_empty_disjoint_and_nonfinite_evidence_is_not_a_pass(self):
        for b in (Analytic(bounds=(.3,.4)),Analytic()):
            if b.bounds==(-.2,.2):b.times=[2.,3.]
            self.assertEqual(compare_derivatives(Analytic(),b,policy())['status'],'incomplete')
        class Bad(Analytic):
            def evaluate(self,k,t):
                values=super().evaluate(k,t);values['w_t']=np.full_like(np.asarray(k),np.nan);return values
        result=compare_derivatives(Analytic(),Bad(),policy())
        self.assertEqual(result['status'],'incomplete')
        json.dumps(result,allow_nan=False)

    def test_near_zero_density_is_flagged_even_without_refit_changes(self):
        class Thin(Analytic):
            def evaluate(self,k,t):
                values=super().evaluate(k,t);values['density_factor']=np.full_like(np.asarray(k),.001);return values
        result=compare_derivatives(Thin(),Thin(),policy())
        self.assertEqual(result['status'],'review_required')
        self.assertTrue(next(r for r in result['metrics'] if r['metric']=='density_factor')['near_zero_density'])

    def test_raw_svi_time_jump_agrees_with_independent_price_slope_limits(self):
        model=known_model();before=deepcopy(model.parameters);result=maturity_jumps(model,policy())
        row=next(r for r in result['points'] if r['metric']=='w_t' and abs(r['k'])<1e-12)
        t=model.times[1];values=[slice_values(np.array([0.]),p) for p in model.parameters]
        vega=black_partials(np.array([0.]),values[1][0])[2][0]
        left=(values[1][3][0]-values[0][3][0])/(model.times[1]-model.times[0])/vega
        right=(values[2][3][0]-values[1][3][0])/(model.times[2]-model.times[1])/vega
        self.assertAlmostEqual(row['baseline'],left,places=6)
        self.assertAlmostEqual(row['alternative'],right,places=6)
        self.assertAlmostEqual(row['signed_jump'],right-left,places=6)
        self.assertEqual(before,model.parameters)
        self.assertTrue(all(r['limit_estimate_converged'] for r in result['metrics']))

    def test_maturity_jumps_require_both_sides_and_an_interior_expiry(self):
        model=Analytic();model.times=[.2,1.]
        self.assertEqual(maturity_jumps(model,policy())['status'],'incomplete')
        model=known_model();model.k_min=[-.2,.05,-.2];model.k_max=[-.05,.2,-.05]
        self.assertEqual(maturity_jumps(model,policy())['status'],'incomplete')

    def test_timing_and_slice_start_evidence_does_not_create_false_refit_pass(self):
        records=[dict(check='slice_alternative_starts',category='numerical'),
                 dict(check='capture_first',category='timing',derivative_diagnostics={'status':'passed'})]
        self.assertEqual(refit_assessment(records),'incomplete')
        records.append(dict(check='held_out_strikes',category='numerical',derivative_diagnostics={'status':'passed'}))
        self.assertEqual(refit_assessment(records),'passed')
        records.append(dict(check='stricter_selection',category='numerical',status='optimizer_failed'))
        self.assertEqual(refit_assessment(records),'incomplete')

    def test_invalid_and_legacy_policies(self):
        validate_policy(policy())
        for key,value in [('enabled',1),('strike_points',True),('relative_tolerance',float('nan')),
                          ('jump_step_fraction',.5),('maximum_score',.5)]:
            p=policy();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_policy(p)
        p=policy();p['absolute_tolerances']['w_t']=0
        with self.assertRaises(ValueError):validate_policy(p)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.json';cfg=practical_policy();path.write_text(json.dumps(cfg))
            self.assertNotIn('derivative_diagnostics',load_surface_config(path))
            cfg['derivative_diagnostics']=policy();path.write_text(json.dumps(cfg))
            self.assertEqual(load_surface_config(path)['derivative_diagnostics'],policy())

    def test_integrated_export_records_diagnostics_without_changing_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);cfg=practical_policy();cfg['plots']=False;cfg['stability']['enabled']=False
            path=folder/'policy.json';path.write_text(json.dumps(cfg));cal=fixture(folder)
            before=deepcopy(cal.tables);baseline=fit_calibrated(cal,path)
            cfg['derivative_diagnostics']=policy();path.write_text(json.dumps(cfg))
            result=fit_calibrated(cal,path);target=result.save()
            self.assertEqual(baseline.tables['repricing'],result.tables['repricing'])
            # Enabling derivative tables must not overwrite cohort route/date
            # metadata (e.g. by shadowing the tuple used as the group key).
            self.assertEqual([r.get('timing_status') for r in baseline.tables['cohort_status']],
                             [r.get('timing_status') for r in result.tables['cohort_status']])
            self.assertEqual(before,cal.tables)
            self.assertEqual(result.tables['cohort_status'][0]['derivative_stability_status'],'incomplete')
            self.assertFalse(result.tables['cohort_status'][0]['local_volatility_implemented'])
            self.assertTrue(result.tables['maturity_jumps'])
            html=(target/'report.html').read_text(encoding='utf-8')
            self.assertIn('Time-derivative jumps',html)
            self.assertIn('manifest.json',html)
            self.assertIn('Derivative refits',html)
            self.assertEqual(result.save(),target)


if __name__=='__main__':unittest.main()
