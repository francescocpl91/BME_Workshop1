"""Permission invariants, domain enforcement and compact workflow regressions."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json,tempfile,unittest
import numpy as np

from src.surface_checks import assess_local_vol_eligibility,allows_surface_point,allows_surface_stencil,validate_eligibility_policy
from src.surface_model import validate_grid,SurfaceModel
from src.volatility_surface import fit_calibrated,load_surface_config
from test_svi_surface import known_model,settings
from test_surface_derivatives import policy as derivative_policy
from test_volatility_surface import fixture


def permission_policy():
    return dict(allow_intraday_approximation=True,allow_historical_closes=True,allow_maturity_jumps=True,maturity_knots='exclude')


def evidence():
    model=known_model();cfg=settings();cfg['eligibility']=permission_policy();cfg['derivative_diagnostics']=derivative_policy()
    grid,validation=validate_grid(model,cfg['grid']);gid='synthetic'
    cohort=dict(group_id=gid,underlying='ANY',route='live',market_fit_pass=True,numerical_checks_pass=True,
                stability_status='passed',derivative_stability_status='passed',unsupported_holdout_observations=2)
    from src.surface_checks import maturity_jumps
    jump=maturity_jumps(model,cfg['derivative_diagnostics'])
    tables=dict(surface_grid=[dict(group_id=gid,**r) for r in grid],
        reference_inputs=[dict(group_id=gid,years=t,expiration=f'node{i}',reference_timestamp='2030-01-01T12:00:00+00:00') for i,t in enumerate(model.times)],
        maturity_jumps=[dict(group_id=gid,**r) for r in jump['metrics']],
        derivative_refits=[dict(group_id=gid,check='held_out_strikes',category='numerical',status='passed')],
        derivative_refit_domains=[dict(group_id=gid,check='held_out_strikes',category='numerical',status='checked',
            domain_scope='whole_interval',
            interval_start=a,interval_end=b,k_min=-.15,k_max=.15) for a,b in zip(model.times[:-1],model.times[1:])])
    return model,cfg,cohort,tables


class EligibilityTests(unittest.TestCase):
    def test_explicit_permission_has_warnings_and_no_computation(self):
        model,cfg,cohort,tables=evidence();before=deepcopy(tables)
        r=assess_local_vol_eligibility(cohort,model,tables,cfg)
        self.assertEqual(r['status'],'eligible_with_warnings');self.assertTrue(r['allowed'])
        self.assertFalse(r['local_volatility_implemented']);self.assertFalse(r['full_simulation_ready'])
        self.assertTrue(any('Intraday' in w for w in r['warnings']))
        self.assertEqual(tables,before);json.dumps(r,allow_nan=False)

    def test_exact_knots_extrapolation_and_cross_knot_stencils_are_denied(self):
        model,cfg,cohort,tables=evidence();r=assess_local_vol_eligibility(cohort,model,tables,cfg)
        self.assertTrue(all(d['k_min']==-.15 and d['k_max']==.15 for d in r['domain']))
        self.assertTrue(all(not allows_surface_point(r,t,0.) for t in [0.,.2,.6,1.,1.1]))
        self.assertTrue(allows_surface_point(r,.4,.15));self.assertFalse(allows_surface_point(r,.4,.16))
        self.assertTrue(allows_surface_stencil(r,[(.39,-.01),(.4,0.),(.41,.01)]))
        self.assertFalse(allows_surface_stencil(r,[(.59,0.),(.61,0.)]))
        for t,k in [(float('nan'),0.),(.4,float('inf')),(True,0.),(.4,'0')]:
            self.assertFalse(allows_surface_point(r,t,k))

    def test_no_policy_cannot_authorize_a_future_module(self):
        model,cfg,cohort,tables=evidence();cfg.pop('eligibility')
        r=assess_local_vol_eligibility(cohort,model,tables,cfg)
        self.assertFalse(r['allowed']);self.assertIn('eligibility_policy_not_configured',r['blockers'])

    def test_two_expiries_need_no_interior_maturity_jump(self):
        model,cfg,cohort,tables=evidence()
        model.times=model.times[:2];tables['maturity_jumps']=[]
        r=assess_local_vol_eligibility(cohort,model,tables,cfg)
        self.assertTrue(r['allowed']);self.assertEqual(len(r['domain']),1)

    def test_legacy_schema_rejects_unsupported_diagnostics_as_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=settings();cfg['schema_version']=1
            cfg['diagnostics']['extended']=False
            path=Path(tmp)/'settings.json';path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_surface_config(path)

    def test_each_essential_failure_blocks_even_when_approximations_allowed(self):
        for key,value in [('market_fit_pass',False),('numerical_checks_pass',False),('stability_status','incomplete'),('derivative_stability_status','review_required')]:
            model,cfg,cohort,tables=evidence();cohort[key]=value
            with self.subTest(key=key):self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_bad_derivatives_and_missing_domain_evidence_block(self):
        for field,value in [('density_factor',.0001),('w_t',-.001),('w',float('nan'))]:
            model,cfg,cohort,tables=evidence();tables['surface_grid'][0][field]=value
            with self.subTest(field=field):self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])
        model,cfg,cohort,tables=evidence();tables['derivative_refit_domains']=[]
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_missing_or_unconverged_jump_estimates_cannot_be_waived(self):
        model,cfg,cohort,tables=evidence();tables['maturity_jumps'][0]['limit_estimate_converged']=False
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])
        tables['maturity_jumps']=[]
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_maturity_review_permission_is_explicit(self):
        model,cfg,cohort,tables=evidence();tables['maturity_jumps'][0]['status']='REVIEW_REQUIRED'
        self.assertTrue(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])
        cfg['eligibility']['allow_maturity_jumps']=False
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_jump_permission_cannot_waive_bad_signs_or_conditioning(self):
        for metric,value in [('w_t',-.01),('density_factor',.01),('density_factor',-.01)]:
            model,cfg,cohort,tables=evidence()
            row=next(r for r in tables['maturity_jumps'] if r['metric']==metric)
            row.update(status='REVIEW_REQUIRED',minimum_value=value,
                       negative_value=value<0,near_zero_density=metric=='density_factor' and value<.05)
            result=assess_local_vol_eligibility(cohort,model,tables,cfg)
            with self.subTest(metric=metric,value=value):
                self.assertFalse(result['allowed'])
                self.assertIn('invalid_or_missing_maturity_boundary_conditions',result['blockers'])
        for field,value in [('minimum_value',None),('minimum_value',float('nan')),('negative_value',None)]:
            model,cfg,cohort,tables=evidence();tables['maturity_jumps'][0][field]=value
            self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_ssvi_permission_uses_complete_interval_support(self):
        from src.surface_checks import compare_derivatives,maturity_jumps
        _,cfg,cohort,tables=evidence();gid=cohort['group_id']
        model=SurfaceModel([.2,.6,1.],[.008,.024,.04],0.,0.,.25,[-.3]*3,[.3]*3)
        other=SurfaceModel([.2,.6,1.],[.008,.024,.04],0.,0.,.25,[-.3,-.1,-.3],[.3,.1,.3])
        compared=compare_derivatives(model,other,cfg['derivative_diagnostics'])
        self.assertEqual(compared['status'],'passed')
        tables['surface_grid']=[dict(group_id=gid,**r) for r in validate_grid(model,cfg['grid'])[0]]
        tables['derivative_refit_domains']=[dict(group_id=gid,check='held_out_strikes',category='numerical',**r) for r in compared['domains']]
        tables['maturity_jumps']=[dict(group_id=gid,**r) for r in maturity_jumps(model,cfg['derivative_diagnostics'])['metrics']]
        result=assess_local_vol_eligibility(cohort,model,tables,cfg)
        self.assertTrue(result['allowed'])
        self.assertFalse(allows_surface_point(result,.59,.14))
        for interval in result['domain']:
            for t in np.linspace(interval['start_years'],interval['end_years'],101)[1:-1]:
                for k in (interval['k_min'],0.,interval['k_max']):
                    self.assertTrue(allows_surface_point(result,t,k))
                    self.assertTrue(np.isfinite(other.evaluate(k,t)['w']))

    def test_raw_svi_interval_support_never_uses_node_only_wings(self):
        model=known_model();model.k_min=[-.2,-.1,-.15];model.k_max=[.2,.1,.15]
        self.assertEqual(model.interval_domain(.2,.6),(-.1,.1))
        self.assertEqual(model.interval_domain(.3,.9),(-.1,.1))
        for a,b in ((.1,.6),(.2,1.1),(.6,.2),(.2,float('nan'))):
            with self.assertRaises(ValueError):model.interval_domain(a,b)

    def test_partial_or_old_sampled_domain_evidence_cannot_authorize(self):
        model,cfg,cohort,tables=evidence()
        tables['derivative_refit_domains'][0].pop('domain_scope')
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])
        model,cfg,cohort,tables=evidence()
        tables['derivative_refit_domains'][0]['interval_start']=.3
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_live_and_historical_permissions_are_independent(self):
        model,cfg,cohort,tables=evidence();cfg['eligibility']['allow_intraday_approximation']=False
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])
        cohort['route']='historical'
        r=assess_local_vol_eligibility(cohort,model,tables,cfg);self.assertTrue(r['allowed'])
        self.assertTrue(any('Historical' in w for w in r['warnings']))
        cfg['eligibility']['allow_historical_closes']=False
        self.assertFalse(assess_local_vol_eligibility(cohort,model,tables,cfg)['allowed'])

    def test_missing_surface_is_blocked_with_no_domain(self):
        _,cfg,cohort,_=evidence();cohort['reason']='no_usable_iv_observations'
        r=assess_local_vol_eligibility(cohort,None,{},cfg)
        self.assertFalse(r['allowed']);self.assertEqual(r['domain'],[])

    def test_policy_rejects_implicit_knot_or_extrapolation_permission(self):
        for key,value in [('allow_intraday_approximation',1),('maturity_knots','allow')]:
            p=permission_policy();p[key]=value
            with self.assertRaises(ValueError):validate_eligibility_policy(p)
        p=permission_policy();p['extrapolation']=True
        with self.assertRaises(ValueError):validate_eligibility_policy(p)

    def test_extended_diagnostics_toggle_preserves_fit_and_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);cfg=settings();cfg['fit']['model']='ssvi';cfg['plots']=False
            cfg['eligibility']=permission_policy();path=folder/'settings.json';cal=fixture(folder)
            cfg['diagnostics']['extended']=True;path.write_text(json.dumps(cfg));a=fit_calibrated(cal,path)
            cfg['diagnostics']['extended']=False;path.write_text(json.dumps(cfg))
            with patch('src.volatility_surface.diagnostics',side_effect=AssertionError('Optional research must be skipped')):
                b=fit_calibrated(cal,path)
            self.assertEqual(a.tables['repricing'],b.tables['repricing'])
            self.assertEqual(a.tables['cohort_status'],b.tables['cohort_status'])
            self.assertTrue(a.tables['quote_diagnostics']);self.assertFalse(b.tables['quote_diagnostics'])
            target=b.save();saved=json.loads((target/'eligibility.json').read_text())
            self.assertEqual(saved,{r['group_id']:r['local_vol_eligibility'] for r in b.tables['cohort_status']})
            self.assertEqual(target,b.save())
            (target/'eligibility.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):b.save()


if __name__=='__main__':unittest.main()
