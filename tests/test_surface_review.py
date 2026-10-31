"""Report decisions must agree with execution; synthetic data, no user JSONs."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import math
import re
import tempfile
import unittest

from test_surface_quality import practical_policy, quote
from test_volatility_surface import fixture
from src.option_math import implied_volatility
from src.surface_checks import quote_quality
from src.surface_reporting import render_report, _table
from src.surface_checks import coverage_statistics
from src.surface_checks import summarize_quality
from src.volatility_surface import fit_calibrated, _coverage


class CoverageReviewTests(unittest.TestCase):
    def test_counts_boundaries_duplicates_and_zero(self):
        cfg=practical_policy();s=cfg['selection'];s.update(near_forward_abs_k=.04)
        ks=[-.1,-.04,0.,.04,.1,.1]
        points=[dict(k=k,strike=100*math.exp(k),expiration='2030-01-01',reference_years=1.) for k in ks]
        result=coverage_statistics(points,s)
        self.assertEqual(result['unique_strikes'],5)
        self.assertEqual((result['below'],result['above']),(2,2))
        self.assertEqual((result['near'],result['near_below'],result['near_above']),(3,1,1))
        self.assertAlmostEqual(result['largest_gap_k'],.06)

    def test_report_statistics_match_selection_rejections(self):
        cfg=practical_policy();s=cfg['selection']
        for ks in ([-.2,-.15,-.1,-.05,0.,.05,.1,.15,.2],[-.2,0.,.2],[.01,.02,.03,.04,.05]):
            points=[dict(k=k,strike=100*math.exp(k),expiration='2030-01-01',reference_years=1.) for k in ks]
            counts=coverage_statistics(points,s)
            _,rejected,_=_coverage(points,cfg)
            self.assertEqual('2030-01-01' not in rejected,counts['coverage_pass'])
        self.assertFalse(coverage_statistics([],s)['count_pass'])

    def test_broader_policy_is_still_three_simultaneous_limits(self):
        narrow=practical_policy()['quality'];broad={**narrow,'maximum_relative_outside_error':.02,'maximum_outside_half_spreads':1.}
        self.assertFalse(quote_quality(quote(),102.,.2125,narrow)['practical_pass'])
        self.assertTrue(quote_quality(quote(),102.,.2125,broad)['practical_pass'])
        self.assertFalse(quote_quality(quote(),102.,.2126,broad)['practical_pass'])
        # An inexpensive quote can still fail relative premium even with a wide spread.
        p=quote(bid=1.,ask=1.1,option_price=1.05)
        self.assertFalse(quote_quality(p,1.13,.21,broad)['practical_price_pass'])

    def test_seventy_percent_coverage_is_configurable_and_keeps_missing_in_denominator(self):
        policy=practical_policy()['quality'];policy['minimum_practical_coverage']=.7
        points=[quote() for _ in range(7)]+[quote(bid_iv=None) for _ in range(3)]
        rows=[quote_quality(p,100.,.2,policy) for p in points]
        self.assertTrue(summarize_quality(points,rows,policy)['practical_coverage_pass'])
        policy['minimum_practical_coverage']=.8
        self.assertFalse(summarize_quality(points,rows,policy)['practical_coverage_pass'])
        policy['minimum_practical_coverage']=.7;rows[0]=quote_quality(quote(),110.,.3,policy)
        self.assertFalse(summarize_quality(points,rows,policy)['practical_coverage_pass'])

    def test_table_escapes_untrusted_separators(self):
        table='\n'.join(_table(['Value','Result'],[('A | B','PASS')]))
        self.assertIn('A \\| B',table)



class SurfaceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.cfg=practical_policy();cls.cfg['fit']['model']='ssvi';cls.cfg['plots']=False
        cls.path=cls.root/'policy.json';cls.path.write_text(json.dumps(cls.cfg))
        cls.cal=fixture(cls.root);cls.result=fit_calibrated(cls.cal,cls.path)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_short_report_retains_decision_checks_domain_and_evidence(self):
        before=deepcopy(self.result.tables);report=render_report(self.result)
        for title in ('Outcome','Essential checks','Supported conversion domain','Warnings and limits','Price fit by expiry','Policy and evidence'):
            self.assertIn(title,report)
        self.assertIn('eligibility.json',report);self.assertIn('manifest.json',report)
        self.assertNotIn('Complete active configuration',report)
        self.assertLess(len(report),15000)
        self.assertEqual(before,self.result.tables)

    def test_all_detailed_coverage_populations_remain_exported(self):
        records=self.result.tables['coverage_checks']
        self.assertEqual(len(records),9)
        self.assertEqual({r['population'] for r in records},{'eligible_before_coverage','strict_passing_diagnostic','practical_passing_diagnostic'})
        target=self.result.save();self.assertEqual(target,self.result.save())
        self.assertTrue((target/'coverage_checks.csv').exists())
        self.assertEqual(set(self.result.manifest['identity']['code_hashes']),
                         {'volatility_surface.py','surface_model.py','surface_checks.py','surface_reporting.py','surface_view.py'})

    def test_report_uses_recorded_permission_never_infers_approval(self):
        result=deepcopy(self.result);cohort=result.tables['cohort_status'][0]
        cohort['local_vol_eligibility'].update(status='blocked',allowed=False,blockers=['test_failure'])
        report=render_report(result)
        self.assertIn('Not allowed',report);self.assertIn('test failure',report)
        self.assertNotIn('Allowed inside stated domain',report)

    def test_failed_essential_checks_remain_visible(self):
        result=deepcopy(self.result);permission=result.tables['cohort_status'][0]['local_vol_eligibility']
        permission['checks']=[dict(check='Arbitrage and derivative accuracy',passed=False)]
        self.assertIn('| Arbitrage and derivative accuracy | BLOCKED |',render_report(result))

    def test_rejected_expiries_and_empty_inputs_remain_blocked(self):
        cfg=deepcopy(self.cfg);cfg['selection']['minimum_near_forward']=100
        path=self.root/'reject.json';path.write_text(json.dumps(cfg))
        result=fit_calibrated(self.cal,path)
        self.assertFalse(result.models);self.assertEqual(len(result.tables['coverage_checks']),3)
        self.assertTrue(all(not r['count_pass'] for r in result.tables['coverage_checks']))
        self.assertIn('insufficient strike coverage',render_report(result))
        empty=deepcopy(self.cal);empty.tables['option_iv']=[]
        self.assertIn('no_option_data',render_report(fit_calibrated(empty,self.path)))

    def test_historical_report_does_not_invent_bid_ask_evidence(self):
        result=fit_calibrated(fixture(self.root,route='historical'),self.path)
        self.assertIn('Historical cohorts use only the RMSE gate',render_report(result))
        self.assertTrue(all(r['inside_spread_fraction'] is None for r in result.tables['expiry_metrics']))

    def test_tables_stay_narrow_and_thresholds_are_configured(self):
        result=deepcopy(self.result);result.manifest['identity']['config']['quality']['minimum_practical_coverage']=.7
        report=render_report(result);self.assertIn('**70%**',report)
        for line in report.splitlines():
            if line.startswith('|'):self.assertLessEqual(len(re.split(r'(?<!\\)\|',line))-2,6)

    def test_detailed_stability_evidence_not_erased_by_short_report(self):
        result=deepcopy(self.result)
        result.tables['stability'].append(dict(group_id=result.tables['cohort_status'][0]['group_id'],check='failure',status='optimizer_failed'))
        before=deepcopy(result.tables);render_report(result)
        self.assertEqual(before,result.tables)
        self.assertIn('stability.csv',render_report(result))


if __name__=='__main__':unittest.main()
