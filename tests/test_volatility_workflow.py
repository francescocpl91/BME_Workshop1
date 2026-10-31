"""Presentation regressions use synthetic inputs, never today's portfolio JSON."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from src.volatility_workflow import build_summary, save_overview, show_stage, Progress, capture_console


def fixtures(folder):
    def result(stage,tables,summary):
        target=folder/stage;target.mkdir();(target/'report.html').write_text('fixture',encoding='utf-8')
        return SimpleNamespace(run_id=stage, tables=tables, manifest={'identity':{'config':{}},'summary':summary},workflow_target=target)
    base=dict(group_id='good',underlying='CUSTOM',currency='EUR',valuation_date='2031-02-02',route='historical')
    bad={**base,'group_id':'rejected','status':'unavailable','reason':'no_usable_iv_observations'}
    good={**base,'status':'fitted_with_limitations','model':'raw_svi','selected_observations':4,'expiries':2,
          'local_vol_eligibility':dict(allowed=True,domain=[{}],warnings=['Timing assumption'],blockers=[])}
    cal=result('calibration',{},dict(option_records=5,iv_calculated=4,valuation_dates=['2031-02-02'],curve_status=[],unavailable_reasons={'missing_price':1}))
    surface=result('surface',dict(cohort_status=[good,bad],selection=[dict(group_id='good',selected=True) for _ in range(4)]+[dict(group_id='rejected',selected=False,reason='individual_iv_not_accepted')]),{})
    local=result('local',dict(cohorts=[{**base,'status':'validated_on_supported_grid','validation_passes':8,'requested_points':8}],
        pricing_summary=[{**base,'status':'validated_with_boundary_assumptions','assessed_quotes':4}],
        state_mapping=[{**base,'status':'prepared_with_boundary_assumptions'}]),{})
    mc=result('mc',dict(cohorts=[{**base,'status':'review_required','quotes':4,'paths':100,'reason':'generic_old_reason'}],
        quotes=[{**base,'fit_pass':True,'pde_pass':True,'convergence_pass':True,'precision_pass':i!=0} for i in range(4)],
        forwards=[{**base,'passed':True}]),{})
    return [cal,surface,local,mc]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.results=fixtures(self.root)

    def test_root_rejections_collapsed_but_precision_remains_visible(self):
        s=build_summary(self.results)
        self.assertEqual(len(s['groups']),1);self.assertEqual(len(s['excluded_input_groups']),1)
        self.assertEqual(s['groups'][0]['root_cause'],'sampling precision')
        self.assertEqual(s['groups'][0]['checks'][3],dict(check='sampling precision',passed=3,total=4))
        self.assertIn('risk-neutral',s['groups'][0]['outcome'])

    def test_real_fitted_group_failure_is_not_hidden(self):
        self.results[2].tables['pricing_summary'][0].update(status='review_required',reason='boundary_checks_failed')
        s=build_summary(self.results)
        self.assertEqual(s['groups'][0]['root_cause'],'boundary_checks_failed')
        self.assertIn('PDE',s['groups'][0]['outcome'])

    def test_empty_partial_and_disabled_results_do_not_claim_success(self):
        self.assertFalse(build_summary([None]*4)['groups'])
        self.results[3]=None
        self.assertIn('Not completed',build_summary(self.results)['groups'][0]['outcome'])
        row=self.results[1].tables['cohort_status'][0]
        row['reason']='intraday_approximation; generic descriptive warning'
        row['local_vol_eligibility'].update(allowed=False,blockers=['refit_stability_failed'])
        summary=build_summary(self.results)['groups'][0]
        self.assertIn('Blocked',summary['outcome'])
        self.assertEqual(summary['root_cause'],'refit_stability_failed')

    def test_cohorts_join_by_identity_not_same_ticker(self):
        row=copy.deepcopy(self.results[1].tables['cohort_status'][0]);row['group_id']='other';row['valuation_date']='2031-03-03'
        self.results[1].tables['cohort_status'].append(row)
        groups=build_summary(self.results)['groups']
        self.assertEqual(len(groups),2);self.assertIn('Dupire',groups[1]['outcome'])

    def test_console_is_compact_and_keeps_exact_failed_gate(self):
        with redirect_stdout(io.StringIO()) as out:
            for i,r in enumerate(self.results):show_stage(i,r,r.workflow_target)
        text=out.getvalue();self.assertIn('sampling precision 3/4',text)
        self.assertNotIn('generic_old_reason',text);self.assertNotIn('no_usable_iv_observations',text)

    def test_overview_utf8_escaped_links_and_immutable_verification(self):
        self.results[1].tables['cohort_status'][0]['underlying']='€STR <test>'
        folder=save_overview(self.root,self.results,'example log')
        html=(folder/'report.html').read_text(encoding='utf-8')
        self.assertIn('€STR &lt;test&gt;',html);self.assertIn('sampling precision',html)
        self.assertIn('../../surface/report.html',html)
        self.assertEqual(save_overview(self.root,self.results,'example log'),folder)
        (folder/'summary.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'corrupt'):save_overview(self.root,self.results,'example log')

    def test_overview_works_without_excluded_local_documentation(self):
        original = Path.exists
        def without_local_guide(path):
            return False if path.name == 'volatility_methodology.html' else original(path)
        with patch.object(Path, 'exists', without_local_guide):
            folder=save_overview(self.root,self.results,'fresh checkout without docs')
        html=(folder/'report.html').read_text(encoding='utf-8')
        self.assertNotIn('volatility_methodology.html',html)
        self.assertIn('../../surface/report.html',html)
        self.assertIn('sampling precision',html)

    def test_distinct_failure_logs_have_distinct_identity(self):
        a=save_overview(self.root,[None]*4,'missing source')
        b=save_overview(self.root,[None]*4,'invalid configuration')
        self.assertNotEqual(a,b)

    def test_progress_throttle_does_not_discard_diagnostics(self):
        with patch('src.volatility_workflow.monotonic',side_effect=[0,1,31,31]),redirect_stdout(io.StringIO()) as out:
            p=Progress('Test');p('first');p('second')
        self.assertEqual(p.messages,['first','second']);self.assertEqual(out.getvalue().count('working'),1)
        with redirect_stdout(io.StringIO()) as outer:
            with capture_console() as recorded:print('visible and saved')
        self.assertEqual(outer.getvalue(),recorded.getvalue())

if __name__=='__main__':unittest.main()
