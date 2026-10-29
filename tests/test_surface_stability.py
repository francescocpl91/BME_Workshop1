"""Supported-domain and timing policies use synthetic, independent settings."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import math
import tempfile
import unittest

from test_surface_quality import practical_policy
from src.surface_checks import assess_holdout, stability_decision, aggregate_stability
from src.volatility_surface import _stability, load_surface_config


def points():
    return [dict(record_id=i,instrument_id=f'C{i}',expiration='2030-05-01',
        strike=100*math.exp(k),k=k,reference_years=.5)
        for i,k in enumerate([-.25,-.2,-.15,-.1,-.05,0.,.05,.1,.15,.2,.25])]


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.cfg=practical_policy()
        self.model=SimpleNamespace(domain=lambda t:(-.2,.2))

    def test_two_unsupported_wings_remain_explicit_without_mutation(self):
        held=points();original=deepcopy(held)
        valid,r=assess_holdout(self.model,held,self.cfg['selection'],supported=True)
        self.assertEqual(len(valid),9)
        self.assertEqual(r['holdout_requested'],11)
        self.assertEqual(r['holdout_unsupported'],2)
        self.assertTrue(r['scope_coverage_complete'])
        self.assertEqual([p['instrument_id'] for p in r['unsupported_observations']],['C0','C10'])
        self.assertTrue(all(p['status']=='unvalidated_wing' for p in r['unsupported_observations']))
        self.assertEqual(original,held)
        _,legacy=assess_holdout(self.model,held,self.cfg['selection'],supported=False)
        self.assertFalse(legacy['scope_coverage_complete'])

    def test_empty_or_thin_supported_set_cannot_pass(self):
        for bounds in [(1.,2.),(-.01,.01),(-.2,-.05)]:
            with self.subTest(bounds=bounds):
                model=SimpleNamespace(domain=lambda t:bounds)
                _,r=assess_holdout(model,points(),self.cfg['selection'],supported=True)
                self.assertFalse(r['scope_coverage_complete'])

    def test_unavailable_maturity_is_not_relabelled_as_valid_wing(self):
        def unavailable(t):raise ValueError('outside maturity')
        _,r=assess_holdout(SimpleNamespace(domain=unavailable),points(),self.cfg['selection'],supported=True)
        self.assertFalse(r['scope_coverage_complete'])
        self.assertTrue(all(x['status']=='unvalidated_maturity' for x in r['unsupported_observations']))

    def test_diagnostics_are_visible_but_not_required_and_legacy_remains_required(self):
        policy=self.cfg['stability']
        numerical=dict(check='test',status='checked',max_iv_change=.001,full_maturity_coverage=True)
        timing=dict(check='capture_first',status='optimizer_failed',required=False)
        self.assertEqual(stability_decision(timing,policy),'FAIL')
        self.assertEqual(aggregate_stability([numerical,timing],policy),'passed')
        timing.pop('required')
        self.assertEqual(aggregate_stability([numerical,timing],policy),'failed')
        self.assertEqual(aggregate_stability([],policy),'incomplete')

    def test_supported_scope_does_not_claim_full_training_maturity_coverage(self):
        r=dict(status='checked',max_iv_change=.001,holdout_scaled_rmse=1.,
            holdout_requested=11,holdout_evaluated=9,full_maturity_coverage=False,
            validation_scope='supported_domain',scope_coverage_complete=True)
        self.assertEqual(stability_decision(r,self.cfg['stability']),'PASS')
        r['scope_coverage_complete']=False
        self.assertEqual(stability_decision(r,self.cfg['stability']),'INCOMPLETE')

    def test_missing_nonfinite_and_excessive_comparisons_never_pass(self):
        for field,value,expected in [('max_iv_change',None,'INCOMPLETE'),
                ('max_iv_change',float('nan'),'FAIL'),('max_iv_change',.1,'FAIL'),
                ('holdout_scaled_rmse',None,'INCOMPLETE'),('holdout_scaled_rmse',float('inf'),'FAIL'),
                ('holdout_scaled_rmse',9.,'FAIL')]:
            r=dict(status='checked',max_iv_change=.001,holdout_scaled_rmse=1.,holdout_requested=5,holdout_evaluated=5)
            r[field]=value
            with self.subTest(field=field,value=value):self.assertEqual(stability_decision(r,self.cfg['stability']),expected)

    def test_config_policy_values_validated_and_legacy_defaults_not_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'settings.json';cfg=self.cfg
            p.write_text(json.dumps(cfg));self.assertNotIn('timing_role',load_surface_config(p)['stability'])
            for key,bad in [('timing_role','ignore'),('maturity_holdout_scope','any_quotes')]:
                invalid=deepcopy(cfg);invalid['stability'][key]=bad;p.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):load_surface_config(p)
            cfg['stability'].update(timing_role='diagnostic',maturity_holdout_scope='supported_domain')
            p.write_text(json.dumps(cfg));self.assertEqual(load_surface_config(p)['stability']['timing_role'],'diagnostic')

    def test_workflow_keeps_time_subsets_out_of_numerical_gate(self):
        cfg=deepcopy(self.cfg);cfg['stability'].update(enabled=True,timing_role='diagnostic',maturity_holdout_scope='supported_domain')
        cfg['selection']['maximum_expiry_gap_years']=1.
        # This deliberately coarse fixture has 0.05-spaced k values: withholding
        # an interior strike creates a 0.10 gap. Permit it in the fixture only.
        cfg['selection']['maximum_strike_gap_k']=.11
        baseline=SimpleNamespace(times=[.2,.4,.6,.8],start_checks=[dict(check='starts',status='checked',max_iv_change=0.)])
        sample=[]
        for j,year in enumerate(baseline.times):
            for p in points()[1:-1]:
                sample.append(dict(p,expiration=f'2030-0{j+3}-01',reference_years=year,
                    option_price=5.,bid=4.,ask=6.,route='live',relative_spread=.02,
                    error_scale=1.,valuation_timestamp=f'2030-01-01T10:{j*10:02d}:00+00:00'))
        before=deepcopy(sample)
        def fit(subset,policy):
            return SimpleNamespace(times=sorted(set(p['reference_years'] for p in subset)),domain=lambda t:(-.2,.2)),[],[]
        with patch('src.volatility_surface._fit',side_effect=fit),patch('src.volatility_surface._model_distance',return_value=.001),patch('src.volatility_surface._metrics',return_value={'scaled_price_rmse':1.}):
            records,status=_stability(baseline,sample,[],cfg)
        self.assertEqual(status,'passed')
        timing=[r for r in records if r['category']=='timing']
        self.assertEqual(len(timing),3)
        self.assertTrue(all(not r['required'] and r['decision']=='INCOMPLETE' for r in timing))
        self.assertEqual(before,sample)


if __name__=='__main__':unittest.main()
