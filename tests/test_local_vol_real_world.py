"""Offline invariants. Synthetic names, dates and settings, no user JSON."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import contextlib
import io
import json
import math
import socket
import unittest
import numpy as np
import pandas as pd

import test_monte_carlo as historical_fixture
from test_local_vol_monte_carlo import flat_adapter
from src.mc_common import prepare_history, moments
from src.monte_carlo import load_monte_carlo_config
from src.local_vol_real_world import (load_config,simulate_processed,evolve,
    correlation_factor,simulation_window,_local_scales,_read_stage)


def settings():
    return dict(schema_version=1,enabled=True,annualization_periods=252,
        steps_per_year=24,refinement=2,end_time='12:00',end_timezone='UTC',
        methods_by_return_basis={'split_and_dividend_adjusted_close':'total_return_proxy',
                                 'price_index_level':'direct_price'},
        allow_boundary_assumptions=True,allow_precision_only_review=True,
        drift_convention='historical_arithmetic',unmapped_holdings='historical',
        save_observations=4,maximum_path_steps=1000000,
        convergence_relative_initial=.001,convergence_z=3.)


class RealWorldTests(unittest.TestCase):
    def setUp(self):
        self.fixture=historical_fixture.MonteCarloTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.p=f.processed();self.root=f.root
        self.cfg=settings();self.file=self.root/'rw.json';self.write()
        self.mc=load_monte_carlo_config(f.file)
        self.history=prepare_history(self.p,self.mc)
        sigma=math.sqrt(moments(self.history['joined'])[1][0,0]*252)
        self.adapter=flat_adapter(sigma)
        self.adapter.reference=pd.Timestamp('2026-10-09T12:00:00Z').to_pydatetime()
        self.adapter.expiries=[self.adapter.reference+timedelta(days=365*t) for t in self.adapter.boundary.times]
        self.p.summary['price_selection']=[dict(instrument_id='A',option_reference='INDEX'),
                                           dict(instrument_id='B',option_reference=None)]
        self.local=SimpleNamespace(run_id='local',source_folder=self.root,manifest=dict(archive_cob=self.p.cob,identity=dict(source_surface_id='surface')),
            tables={'state_mapping':[dict(group_id='g',underlying='INDEX',currency='USD')]},state_evaluators={'g':self.adapter})
        self.surface=SimpleNamespace(run_id='surface',manifest=dict(archive_cob=self.p.cob,
            identity=dict(calibration_identity=dict(source_folder=str(self.root),config=dict(valuation_timezone='UTC')))))
        self.row=dict(group_id='g',status='validated_with_boundary_assumptions',fit_pass=True,pde_pass=True,
                      convergence_pass=True,precision_pass=True,forward_pass=True,failed_checks=[])
        self.rn=SimpleNamespace(run_id='rn',manifest={'archive_cob':self.p.cob,'identity':{'source_local_id':'local'}},tables={'cohorts':[self.row]})

    def write(self):self.file.write_text(json.dumps(self.cfg),encoding='utf-8')

    def run_model(self):
        return simulate_processed(self.p,self.local,self.surface,self.rn,self.fixture.file,self.file,source_folder=self.root)

    def test_constant_local_vol_equals_gaussian_control_and_coarse(self):
        r=self.run_model();self.assertTrue(r.summary['simulation_performed'],r.summary)
        self.assertEqual(r.summary['status'],'complete_with_model_assumptions')
        np.testing.assert_allclose(r.arrays['terminal_wealth'],r.arrays['gaussian_control_terminal_wealth'],rtol=2e-14)
        np.testing.assert_allclose(r.arrays['terminal_wealth'],r.arrays['coarse_terminal_wealth'],rtol=2e-14)
        expected=r.arrays['terminal_wealth'][:,:2] @ np.array([.5,.5])
        np.testing.assert_allclose(r.arrays['terminal_wealth'][:,-1],expected)
        np.testing.assert_array_equal(r.arrays['portfolio_wealth'][0],100)

    def test_covariance_scaling_and_singular_zero_factors(self):
        cov=np.array([[.04,-.0048],[-.0048,.0064]])
        sigma,corr,root=correlation_factor(cov)
        np.testing.assert_allclose((sigma[:,None]*root)@(sigma[:,None]*root).T,cov,atol=1e-16)
        new=np.array([.4,.08])[:,None]*root
        self.assertAlmostEqual((new@new.T)[0,1],-.0096)
        sigma,corr,root=correlation_factor(np.diag([.04,0.]))
        np.testing.assert_equal(root[1],0)

    def test_conditional_shocks_match_historical_dependence(self):
        cov=np.array([[.04,-.0048],[-.0048,.0064]])
        sigma,corr,root=correlation_factor(cov)
        z=np.random.default_rng(13).normal(size=(100000,root.shape[1]))@root.T
        self.assertAlmostEqual(np.corrcoef(z.T)[0,1],-.3,delta=.01)

    def test_proxy_is_initial_spot_times_cumulative_growth_times_carry(self):
        r=self.run_model();a=r.arrays
        t=a['years'][-1];f,d=self.adapter.curves.at(t)
        expected=np.exp(a['native_log_growth'][-1,:,0])*f*d
        np.testing.assert_allclose(np.exp(a['native_log_growth'][-1,:,0])*a['proxy_level_per_unit_growth'][-1,0],expected)
        # Evaluator receives k=log(G)+log(D), NOT cumulative growth times the
        # previous index, and portfolio G has no second dividend subtraction.
        model={0:dict(adapter=self.adapter,method='total_return_proxy')}
        _,e=_local_scales(np.array([[math.log(1.1),0.]]),.1,model,np.array([.2,.08]),True)
        self.assertAlmostEqual(e[0]['k'][0],math.log(1.1)-.02*.1)

    def test_direct_price_mapping(self):
        model={0:dict(adapter=self.adapter,method='direct_price')}
        _,e=_local_scales(np.array([[math.log(1.1),0.]]),.1,model,np.array([.2,.08]),True)
        self.assertAlmostEqual(e[0]['k'][0],math.log(1.1)-.01*.1)

    def test_precision_exception_is_diagnostic_and_other_failures_block(self):
        self.row.update(status='review_required',precision_pass=False,failed_checks=['precision_pass'])
        self.assertEqual(self.run_model().summary['status'],'diagnostic')
        self.cfg['allow_precision_only_review']=False;self.write()
        self.assertFalse(self.run_model().summary['simulation_performed'])
        self.cfg['allow_precision_only_review']=True;self.write();self.row['fit_pass']=False
        self.assertFalse(self.run_model().summary['simulation_performed'])

    def test_no_mapping_no_local_model_not_silent_historical_fallback(self):
        self.p.summary['price_selection'][0]['option_reference']=None
        r=self.run_model();self.assertEqual(r.summary['reason'],'no_mapped_local_volatility_holdings')
        self.p.summary['price_selection'][0]['option_reference']='OTHER'
        self.assertIn('missing_or_ambiguous_surface',self.run_model().summary['reason'])

    def test_ambiguous_and_wrong_currency_surface_block(self):
        self.local.tables['state_mapping'].append(dict(group_id='g2',underlying='INDEX',currency='USD'))
        self.local.state_evaluators['g2']=self.adapter
        self.assertIn('ambiguous',self.run_model().summary['reason'])
        self.local.tables['state_mapping']=self.local.tables['state_mapping'][:1]
        self.local.tables['state_mapping'][0]['currency']='EUR'
        self.assertFalse(self.run_model().summary['simulation_performed'])

    def test_dates_not_rebased_and_end_not_truncated(self):
        self.adapter.reference+=timedelta(days=1)
        self.assertIn('start_date_mismatch',self.run_model().summary['reason'])
        self.adapter.reference-=timedelta(days=1)
        self.fixture.cfg['forecast_end']='2028-01-01';self.fixture.write_config()
        self.assertIn('exceeds_surface_horizon',self.run_model().summary['reason'])

    def test_bad_path_aborts_all_outputs_no_clipping(self):
        self.adapter.bounds=(-1e-10,1e-10)
        r=self.run_model();self.assertFalse(r.summary['simulation_performed']);self.assertFalse(r.arrays)
        self.assertIn('no_clipping',r.summary['reason'])

    def test_boundary_opt_in_required(self):
        self.cfg['allow_boundary_assumptions']=False;self.write()
        self.assertIn('opt_in',self.run_model().summary['reason'])

    def test_seed_reproducibility(self):
        a=self.run_model();b=self.run_model()
        np.testing.assert_array_equal(a.arrays['terminal_wealth'],b.arrays['terminal_wealth'])
        self.fixture.cfg['seed']+=1;self.fixture.write_config()
        self.assertFalse(np.array_equal(a.arrays['terminal_wealth'],self.run_model().arrays['terminal_wealth']))

    def test_fx_fallback_and_joint_conversion(self):
        self.p.summary['portfolio']['reporting_currency']='EUR'
        self.assertEqual(self.run_model().summary['simulation_currency'],'USD')
        self.fixture.add_fx(self.p)
        r=self.run_model();self.assertEqual(r.summary['simulation_currency'],'EUR')
        a=r.arrays
        expected=100*np.exp(a['native_log_growth'][-1]@a['conversion_matrix'].T)
        np.testing.assert_allclose(expected,r.arrays['terminal_wealth'][:,:2])
        self.assertEqual(len(a['factor_ids']),3)

    def test_input_objects_unchanged_no_network_and_immutable_save(self):
        before=self.p.tables['selected_prices'].copy(deep=True)
        with patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
            r=self.run_model();folder=r.save();self.assertEqual(r.save(),folder)
        pd.testing.assert_frame_equal(before,self.p.tables['selected_prices'])
        self.assertTrue((folder/'report.html').exists())
        with np.load(folder/'paths.npz',allow_pickle=False) as arrays:
            self.assertEqual(arrays['terminal_wealth'].shape,(2000,3))
        (folder/'risk.csv').write_text('corrupted',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'checksum'):r.save()

    def test_config_typos_and_invalid_methods(self):
        for change in ({'enabled':1},{'annualization_periods':True},{'steps_per_year':0},
                       {'refinement':1},{'unknown':1},{'end_time':'wrong'},
                       {'methods_by_return_basis':{'price_index_level':'total_return_proxy'}}):
            self.file.write_text(json.dumps({**self.cfg,**change}))
            with self.assertRaises((ValueError,TypeError)):load_config(self.file)

    def test_budget_blocks_before_allocating_paths(self):
        self.cfg['maximum_path_steps']=1;self.write()
        self.assertIn('budget',self.run_model().summary['reason'])

    def test_integrated_handler_passes_current_objects_and_saves(self):
        import run
        r=Mock(summary={'status':'complete','warnings':[]});r.save.return_value=self.root
        with patch('src.local_vol_real_world.simulate_processed',return_value=r) as simulate,contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(run._real_world_available(self.p,self.local,self.surface,self.rn,self.root),r)
        args=simulate.call_args
        self.assertEqual(args.args[:4],(self.p,self.local,self.surface,self.rn))
        self.assertEqual(args.kwargs['source_folder'],self.root);r.save.assert_called_once()


    def test_annual_log_drift_and_covariance_have_correct_units(self):
        r=self.run_model();logs=r.arrays['native_log_growth'][-1];t=r.arrays['years'][-1]
        mean,cov=moments(self.history['joined'])
        # Constant local variance equals historical variance in this fixture.
        # The +half historical variance in mu cancels the log-Euler correction.
        for i in range(2):
            se=math.sqrt(cov[i,i]*252*t/len(logs))
            self.assertAlmostEqual(logs[:,i].mean(),mean[i]*252*t,delta=5*se)
        target=cov*252*t
        # Sampling uncertainty of a Gaussian sample covariance (Wishart),
        # including small off-diagonal entries where relative errors mislead.
        covariance_se=np.sqrt((np.outer(np.diag(target),np.diag(target))+target*target)/(len(logs)-1))
        self.assertTrue(np.all(np.abs(np.cov(logs,rowvar=False)-target) < 5*covariance_se))

    def test_same_cob_different_processing_or_validation_is_rejected(self):
        self.surface.manifest['identity']['calibration_identity']['source_folder']=str(self.root/'other')
        self.assertIn('processing_folder_mismatch',self.run_model().summary['reason'])
        self.surface.manifest['identity']['calibration_identity']['source_folder']=str(self.root)
        self.rn.manifest['identity']['source_local_id']='wrong'
        self.assertIn('identity_mismatch',self.run_model().summary['reason'])

    def test_unknown_adjustment_basis_does_not_infer_proxy(self):
        self.cfg['methods_by_return_basis']={'price_index_level':'direct_price'};self.write()
        self.assertIn('proxy_method_not_configured',self.run_model().summary['reason'])

    def test_disabled_and_unavailable_reports_are_saved(self):
        self.cfg['enabled']=False;self.write()
        r=self.run_model();self.assertEqual(r.summary['status'],'disabled')
        folder=r.save();self.assertFalse((folder/'paths.npz').exists())
        self.cfg['enabled']=True;self.write();self.local.state_evaluators={}
        r=self.run_model();folder=r.save()
        self.assertIn('missing_or_ambiguous_surface',(folder/'report.html').read_text(encoding='utf-8'))

    def test_saved_manifest_checksums_and_path_escape(self):
        import hashlib
        p=self.root/'stage';p.mkdir();raw=b'hello';(p/'data.csv').write_bytes(raw)
        m=dict(complete=True,files={'data':dict(file='data.csv',sha256=hashlib.sha256(raw).hexdigest())})
        (p/'manifest.json').write_text(json.dumps(m))
        self.assertEqual(_read_stage(p)[1]['data'],raw)
        (p/'data.csv').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):_read_stage(p)
        m['files']['data']['file']='../outside.csv';(p/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'escape'):_read_stage(p)

    def test_weekend_start_is_actual_cob_plus_one_not_rolled(self):
        self.p.cob='2026-10-10'
        for obj in (self.local,self.surface,self.rn):obj.manifest['archive_cob']=self.p.cob
        self.adapter.reference=pd.Timestamp('2026-10-11T12:00:00Z').to_pydatetime()
        self.adapter.expiries=[self.adapter.reference+timedelta(days=365*t) for t in self.adapter.boundary.times]
        r=self.run_model();self.assertTrue(r.summary['simulation_performed'],r.summary)
        self.assertTrue(r.summary['start_timestamp'].startswith('2026-10-11'))

    def test_time_dependent_local_vol_does_not_recalibrate_correlation(self):
        original=self.adapter.boundary.variance
        self.adapter.boundary.variance=lambda t,k:original(t,k)*(1+2*t)
        r=self.run_model();self.assertTrue(r.summary['simulation_performed'])
        self.assertFalse(np.allclose(r.arrays['terminal_wealth'],r.arrays['gaussian_control_terminal_wealth']))
        _,cov=moments(self.history['joined']);_,corr,_=correlation_factor(cov)
        np.testing.assert_allclose(r.tables['correlation'],corr)



    def test_saved_skipped_cohorts_have_json_null_not_nan(self):
        from src.local_vol_real_world import _csv_records
        rows=_csv_records(b'group_id,quotes,precision_pass\ng,12,True\nskipped,,\n')
        self.assertIsNone(rows[1]['quotes'])
        self.assertIsNone(rows[1]['precision_pass'])
        json.dumps(rows,allow_nan=False)


    def test_model_config_is_not_mistaken_for_provider_requests(self):
        from src.json_config import load_provider_configs
        folder=self.root/'configs';folder.mkdir()
        (folder/'local_vol_real_world.json').write_text(json.dumps(self.cfg))
        # Existing provider fixture is independent of current user settings.
        provider=Path(historical_fixture.FIXTURES)/'yahoo.json'
        (folder/'yahoo.json').write_bytes(provider.read_bytes())
        from datetime import date
        result=load_provider_configs(folder,cob_date=date(2026,10,8),history_start=date(2020,1,1))
        self.assertTrue(result)


    def test_code_fingerprint_covers_transitive_helpers_and_nested_sources(self):
        from src.local_vol_real_world import _code_fingerprints
        import src.local_vol_real_world as module
        sources=self.root/'sources';sources.mkdir()
        (sources/'local_vol_real_world.py').write_text('# entry')
        for name in ('statistics.py','mc_diagnostics.py','local_vol_boundary.py'):
            (sources/name).write_text('# original '+name)
        (sources/'nested').mkdir()
        (sources/'nested'/'helper.py').write_text('# nested')
        with patch.object(module,'__file__',str(sources/'local_vol_real_world.py')):
            before=_code_fingerprints()
            self.assertIn('nested/helper.py',before)
            for name in ('statistics.py','mc_diagnostics.py','local_vol_boundary.py'):
                (sources/name).write_text('# changed '+name)
                after=_code_fingerprints()
                self.assertNotEqual(before[name],after[name])
        # Verify the engine actually includes these hashes in its saved ID.
        with patch.object(module,'_code_fingerprints',return_value=before):
            first=self.run_model()
        with patch.object(module,'_code_fingerprints',return_value=after):
            second=self.run_model()
        self.assertNotEqual(first.run_id,second.run_id)
        np.testing.assert_array_equal(first.arrays['terminal_wealth'],second.arrays['terminal_wealth'])

    def test_csv_preserves_numeric_and_na_identifiers_without_losing_numbers(self):
        from src.local_vol_real_world import _csv_records
        rows=_csv_records(b'group_id,underlying,currency,years,precision_pass\n000123,0005,USD,0.25,False\n000124,NA,USD,,\n')
        self.assertEqual(rows[0]['group_id'],'000123')
        self.assertEqual(rows[0]['underlying'],'0005')
        self.assertEqual(rows[1]['underlying'],'NA')
        self.assertEqual(rows[0]['years'],.25)
        self.assertIsNone(rows[1]['years'])
        json.dumps(rows,allow_nan=False)

    def test_csv_loaded_numeric_mapping_can_resolve_and_simulate(self):
        from src.local_vol_real_world import _csv_records
        self.p.summary['price_selection'][0]['option_reference']='0005'
        self.local.tables['state_mapping']=_csv_records(b'group_id,underlying,currency\n000123,0005,USD\n')
        self.local.state_evaluators={'000123':self.adapter}
        self.row['group_id']='000123'
        result=self.run_model()
        self.assertTrue(result.summary['simulation_performed'],result.summary)
        self.assertEqual(result.tables['resolved_models'].iloc[0]['group_id'],'000123')

    def test_invalid_timezones_are_configuration_errors(self):
        for zone in ('Europe/NotAZone','',None,123,[],{}):
            self.cfg['end_timezone']=zone;self.write()
            with self.subTest(zone=zone),self.assertRaisesRegex(ValueError,'invalid_end_timezone'):
                load_config(self.file)

    def test_invalid_timezone_is_handled_by_integrated_stage(self):
        import run
        self.cfg['end_timezone']='Europe/NotAZone';self.write()
        output=io.StringIO()
        with patch.object(run,'REAL_WORLD_FILE',self.file),patch.object(run,'MONTE_CARLO_FILE',self.fixture.file),contextlib.redirect_stdout(output):
            result=run._real_world_available(self.p,self.local,self.surface,self.rn,self.root)
        self.assertIsNone(result)
        self.assertIn('invalid_end_timezone',output.getvalue())

    def test_clock_evidence_counts_exchange_holiday_not_weekdays(self):
        from src.local_vol_real_world import clock_comparison
        # July 3 is the observed US Independence Day holiday in 2026.
        # From Thursday July 2 to Monday July 6 there is one subsequent XNYS
        # session, not two weekdays and not four historical daily returns.
        origin=pd.Timestamp('2026-07-02T12:00:00Z').to_pydatetime()
        end=pd.Timestamp('2026-07-06T12:00:00Z').to_pydatetime()
        mc={**self.mc,'calendar':'XNYS','forecast_start':'2026-07-02','forecast_end':'2026-07-06'}
        evidence=clock_comparison(origin,end,mc,self.cfg)
        self.assertEqual(evidence['historical_session_steps'],1)
        self.assertAlmostEqual(evidence['historical_business_years'],1/252)
        self.assertAlmostEqual(evidence['local_vol_elapsed_years'],4/365)
        self.assertFalse(evidence['directly_time_aligned'])

    def test_saved_clock_evidence_uses_cob_not_editable_start(self):
        result=self.run_model()
        evidence=result.summary['clock_comparison']
        self.assertEqual(evidence['historical_initial_date'],'2026-10-09')
        self.assertEqual(evidence['historical_annualization_periods'],252)
        self.assertAlmostEqual(evidence['local_vol_elapsed_years'],result.arrays['years'][-1])



    def test_saved_numeric_values_round_trip_at_full_precision(self):
        from src.local_vol_real_world import _csv_records
        # The default fast CSV parser rounds this value to 0.3. It matters
        # for reproducibility when saved histories recalibrate an MC engine.
        original=0.30000000000000004
        raw=pd.DataFrame({'group_id':['0005'],'value':[original]}).to_csv(index=False).encode()
        self.assertEqual(_csv_records(raw)[0]['value'],original)


if __name__=='__main__':unittest.main()
