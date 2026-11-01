"""Synthetic, offline MC regressions; no dependence on user JSON or holdings."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch, Mock
import contextlib
import io
import json
import math
import unittest
import numpy as np
from src.local_vol_monte_carlo import (load_config, pair_samples, estimate, coefficient,
    simulate_cohort, _quote_statistics, validate_local_vol, ValidationResult, SimulationStopped)
from src.surface_model import black_prices
from test_local_vol_state import inputs
from test_local_vol_boundary import config as boundary_config
from test_local_volatility import permission
from src.local_volatility import LocalVolEvaluator, DateSpotLocalVol


def settings():
    return dict(schema_version=1,enabled=True,paths=2000,seed=73,steps_per_year=12,
        refinement=2,allow_boundary_assumptions=True,family_error_probability=.01,
        absolute_price_forward_fraction=.00001,relative_price_tolerance=.001,
        maximum_se_forward_fraction=.0002,maximum_se_relative_price=.05,
        forward_relative_tolerance=.001)


def flat_adapter(vol=.2):
    # Constant local variance has an EXACT log-Euler endpoint at every grid:
    # coupled coarse/fine prices must agree up to floating-point roundoff.
    times=np.array([.25,.5])
    def evaluate(k,t):
        a=np.ones_like(k)
        return dict(w=a*vol*vol*t,w_k=a*0,w_kk=a*0,w_t=a*vol*vol)
    model=SimpleNamespace(evaluate=evaluate)
    boundary=SimpleNamespace(times=times,domains=[dict(k_min=-.3,k_max=.3)],
        variance=lambda t,k:np.full_like(k,vol*vol))
    curves=SimpleNamespace(at=lambda t:(100*math.exp(.01*t),math.exp(-.02*t)))
    reference=datetime(2031,2,3,12,tzinfo=timezone.utc)
    return SimpleNamespace(boundary=boundary,curves=curves,bounds=(-3.,2.),
        ev=SimpleNamespace(model=model,minimum_density=.05,negative_tolerance=1e-9),
        reference=reference,expiries=[reference+timedelta(days=365*t) for t in times])


def quotes(adapter,vol=.2):
    result=[]
    for i,t in enumerate(adapter.boundary.times):
        f,d=adapter.curves.at(float(t))
        for k in (-.1,0.,.1):
            for right in ('call','put'):
                price=float(black_prices(f,f*math.exp(k),t,d,vol,right=='call'))
                result.append(dict(record_id=len(result),expiration=str(i),years=float(t),k=k,
                    option_type=right,reference_forward=f,reference_discount=d,
                    target_price=price,pde_price=price,fitted_iv=vol,
                    group_id='group',status='assessed'))
    return result


class MonteCarloTests(unittest.TestCase):
    def test_pair_standard_error_uses_independent_pairs(self):
        np.testing.assert_equal(pair_samples([1,4,3,8]),[2,6])
        mean,se=estimate(pair_samples([1,4,3,8]))
        self.assertEqual(mean,4);self.assertEqual(se,2)
        for x in ([1,2,3],[1,float('nan'),2,4]):
            with self.assertRaises(ValueError):pair_samples(x)

    def test_config_validation_and_no_current_json_dependency(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json'
            for changes in (dict(paths=101),dict(paths=True),dict(seed=-1),dict(refinement=1),
                            dict(enabled='true'),dict(relative_price_tolerance=float('nan')),
                            dict(family_error_probability=1),dict(unknown=1)):
                p.write_text(json.dumps({**settings(),**changes}))
                with self.assertRaises(ValueError):load_config(p)
            p.write_text(json.dumps(settings()))
            self.assertEqual(load_config(p),settings())

    def test_black_prices_forwards_and_exact_coupled_flat_variance(self):
        adapter=flat_adapter();cfg=settings();cfg['paths']=10000
        prices,fw,boundaries,details=simulate_cohort(adapter,quotes(adapter),cfg,123)
        self.assertEqual(len(prices),12);self.assertEqual(len(fw),4)
        self.assertTrue(all(r['fit_pass'] and r['pde_pass'] for r in prices))
        self.assertTrue(all(r['passed'] for r in fw))
        self.assertLess(max(abs(r['paired_difference']) for r in prices),1e-12)
        self.assertLess(max(r['paired_difference_se'] for r in prices),1e-12)
        self.assertEqual(details['fine_steps'],2*details['coarse_steps'])
        self.assertEqual(details['independent_pairs'],5000)
        self.assertTrue(boundaries[0]['fine_short_states']>0)
        self.assertTrue(all(r['clipped_paths']==r['discarded_paths']==0 for r in boundaries))
        # The shared control gives exact call-put parity at each strike/time.
        for c,p in zip(prices[::2],prices[1::2]):
            self.assertAlmostEqual(c['fine_price']-p['fine_price'],
                c['reference_discount']*(c['reference_forward']-c['reference_strike']),places=11)

    def test_reproducibility_and_seed_changes(self):
        a=flat_adapter();q=quotes(a);cfg=settings()
        first=simulate_cohort(a,q,cfg,17)
        self.assertEqual(first,simulate_cohort(a,q,cfg,17))
        self.assertNotEqual(first[0][0]['fine_price'],simulate_cohort(a,q,cfg,18)[0][0]['fine_price'])

    def test_boundary_opt_in_invalid_coefficients_and_terminal_exit_block(self):
        a=flat_adapter();cfg=settings();cfg['allow_boundary_assumptions']=False
        with self.assertRaisesRegex(ValueError,'opt_in'):simulate_cohort(a,quotes(a),cfg,1)
        for x in ([float('nan')],[-3.],[2.]):
            with self.assertRaisesRegex(ValueError,'no_clipping'):coefficient(a,.1,x,True)
        with self.assertRaisesRegex(ValueError,'opt_in'):coefficient(a,.1,[0.],False)
        a.bounds=(-1e-10,1e-10)
        with self.assertRaisesRegex(ValueError,'no_clipping'):simulate_cohort(a,quotes(a),settings(),1)
        a=flat_adapter()
        a.boundary.variance=lambda t,k:np.full_like(k,-1.)
        with self.assertRaisesRegex(ValueError,'variance'):simulate_cohort(a,quotes(a),settings(),1)

    def test_vector_coefficients_match_existing_date_spot_adapter(self):
        surface,origin=inputs(Path('.'));m=surface.models['group']
        adapter=DateSpotLocalVol(LocalVolEvaluator(m,permission(m),.05,1e-9),
            surface.tables['reference_inputs'],'2031-02-03',boundary_config())
        for t in (0.,.1,.25,.4,.5,.75):
            k=np.array([-.3,-.1,0.,.1,.3]);f,_=adapter.curves.at(t)
            old=adapter.evaluate(origin+timedelta(days=365*t),f*np.exp(k),allow_boundary_assumptions=True)
            self.assertEqual(old['status'],'available')
            v,flags=coefficient(adapter,t,k,True)
            np.testing.assert_allclose(v,old['local_variance'],rtol=1e-10,atol=1e-12)
            np.testing.assert_equal(flags['wing'],old['wing_assumption'])

    def test_bad_interior_density_cannot_be_bypassed(self):
        a=flat_adapter();a.ev.minimum_density=2
        with self.assertRaisesRegex(ValueError,'strict_evaluator'):coefficient(a,.3,np.array([0.]),True)

    def test_precision_and_convergence_are_separate_gates(self):
        a=flat_adapter();q=quotes(a)[2];cfg=settings()
        x=np.array([.8,1.4,1.2,.7])
        r=_quote_statistics(q,x,np.ones(4),4.,cfg)
        self.assertGreater(r['standard_error'],0)
        self.assertFalse(r['precision_pass'])
        # No uncertainty: a deterministic wrong price cannot pass the fit.
        bad=_quote_statistics(q,np.ones(4)*1.5,np.ones(4),4.,cfg)
        self.assertFalse(bad['fit_pass']);self.assertFalse(bad['convergence_pass'])

    def test_missing_expiry_quotes_rejected(self):
        a=flat_adapter()
        with self.assertRaisesRegex(ValueError,'without_supported_quotes'):
            simulate_cohort(a,quotes(a)[:6],settings(),1)

    def test_immutable_output_and_corruption_detection(self):
        with TemporaryDirectory() as tmp:
            r=ValidationResult(Path(tmp),'abc123',dict(identity=dict(config=settings()),summary={}),
                dict(cohorts=[],quotes=[],forwards=[],boundaries=[],expiries=[]))
            folder=r.save();before={p:p.read_bytes() for p in folder.iterdir()}
            self.assertEqual(r.save(),folder)
            self.assertTrue(all(p.read_bytes()==b for p,b in before.items()))
            (folder/'quotes.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'):r.save()

    def test_integration_missing_inputs_expected_error_and_unexpected_error(self):
        import run
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(run._local_vol_mc_available(None,None))
            with patch('src.local_vol_monte_carlo.validate_local_vol',side_effect=ValueError('missing')):
                self.assertIsNone(run._local_vol_mc_available(object(),object()))
            with patch('src.local_vol_monte_carlo.validate_local_vol',side_effect=RuntimeError('bug')):
                with self.assertRaises(RuntimeError):run._local_vol_mc_available(object(),object())

    def test_workflow_uses_current_inputs_and_does_not_download(self):
        a=flat_adapter();q=quotes(a)
        local=SimpleNamespace(run_id='local',source_folder=Path('.'),
            manifest=dict(archive_cob='2031-02-02',identity=dict(source_surface_id='surface')),
            tables=dict(state_mapping=[dict(group_id='group',underlying='ANY',currency='USD',route='historical')],option_repricing=q),
            state_evaluators={'group':a})
        surface=SimpleNamespace(run_id='surface',tables={'repricing':q},manifest=dict(archive_cob='2031-02-02',
            identity=dict(calibration_identity=dict(config=dict(valuation_timezone='UTC')))))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.json';p.write_text(json.dumps(settings()))
            with patch('socket.socket.connect',side_effect=AssertionError('network')):
                r=validate_local_vol(local,surface,p)
            self.assertEqual(r.tables['cohorts'][0]['quotes'],12)
            self.assertEqual(r.tables['cohorts'][0]['underlying'],'ANY')
            self.assertFalse(r.manifest['summary']['portfolio_simulation_ready'])
            local.state_evaluators={}
            self.assertEqual(validate_local_vol(local,surface,p).tables['cohorts'][0]['status'],'skipped')
            surface.run_id='wrong'
            with self.assertRaisesRegex(ValueError,'identity_mismatch'):validate_local_vol(local,surface,p)


    def test_cob_plus_one_date_gate_and_timezone_are_not_hardcoded(self):
        from src.local_vol_monte_carlo import validate_start_date
        a=flat_adapter()
        local=SimpleNamespace(manifest={'archive_cob':'2031-02-02'})
        surface=SimpleNamespace(manifest={'archive_cob':'2031-02-02','identity':{
            'calibration_identity':{'config':{'valuation_timezone':'America/New_York'}}}})
        got=validate_start_date(local,surface,a)
        self.assertEqual(got['start_date'],'2031-02-03')
        self.assertEqual(got['start_timestamp'],a.reference.isoformat())
        # The same UTC instant can belong to the prior valuation-calendar day.
        a.reference=datetime(2031,2,4,1,tzinfo=timezone.utc)
        self.assertEqual(validate_start_date(local,surface,a)['start_date'],'2031-02-03')
        local.manifest['archive_cob']=surface.manifest['archive_cob']='2031-02-03'
        with self.assertRaisesRegex(ValueError,'mc_start_date_mismatch'):
            validate_start_date(local,surface,a)
        a.reference=datetime(2031,2,4,12,tzinfo=timezone.utc)
        self.assertEqual(validate_start_date(local,surface,a)['start_date'],'2031-02-04')
        surface.manifest['archive_cob']='2031-02-02'
        with self.assertRaisesRegex(ValueError,'archive_cob_mismatch'):
            validate_start_date(local,surface,a)

    def test_mismatched_surface_is_unavailable_before_any_path_is_drawn(self):
        a=flat_adapter(); q=quotes(a)
        local=SimpleNamespace(run_id='local',source_folder=Path('.'),
            manifest=dict(archive_cob='2031-02-03',identity=dict(source_surface_id='surface')),
            tables=dict(state_mapping=[dict(group_id='group')],option_repricing=q),state_evaluators={'group':a})
        surface=SimpleNamespace(run_id='surface',tables={'repricing':q},manifest=dict(archive_cob='2031-02-03',
            identity=dict(calibration_identity=dict(config=dict(valuation_timezone='UTC')))))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.json';p.write_text(json.dumps(settings()))
            with patch('src.local_vol_monte_carlo.simulate_cohort',side_effect=AssertionError('must not draw')):
                result=validate_local_vol(local,surface,p)
        row=result.tables['cohorts'][0]
        self.assertEqual(row['status'],'unavailable')
        self.assertIn('mc_start_date_mismatch',row['reason'])
        self.assertEqual(result.tables['quotes'],[])

    def test_every_quote_must_pass_no_percentage_threshold_or_empty_success(self):
        from src.local_vol_monte_carlo import validation_gates
        good=dict(fit_pass=True,pde_pass=True,convergence_pass=True,precision_pass=True)
        for passed in (80,99):
            rows=[dict(good,precision_pass=i<passed) for i in range(100)]
            flags,failed=validation_gates(rows,[{'passed':True}])
            self.assertEqual(failed,['precision_pass'])
            self.assertFalse(flags['precision_pass'])
        self.assertEqual(validation_gates([good],[{'passed':True}])[1],[])
        self.assertTrue(validation_gates([],[])[1])

    def test_run_pipeline_links_current_local_result_before_statistics(self):
        import run
        processed=Mock(run_id='fixture',summary={'portfolio':{'holdings':[]},
            'readiness':{'aligned_return_observations':0,'foreign_currencies':[]},
            'options':{'iv_candidates':0}})
        processed.save.return_value=Path('.')
        stats=Mock(run_id='stats',summary={'portfolio_status':'unavailable','warnings':[]})
        historic=Mock(run_id='mc',summary={'status':'unavailable','warnings':[]})
        surface=SimpleNamespace(run_id='surface');local=SimpleNamespace(run_id='local');order=[]
        def validation(l,s):
            self.assertIs(l,local);self.assertIs(s,surface);order.append('validation')
        def statistics(*args):order.append('statistics');return stats
        def simulation(*args):order.append('historical');return historic
        with patch('src.volatility_workflow.save_overview',return_value=Path('.')),patch.object(run,'process_inventory',return_value=processed), \
             patch.object(run,'_calibrate_available'), \
             patch.object(run,'_surface_available',return_value=surface), \
             patch.object(run,'_local_vol_available',return_value=local), \
             patch.object(run,'_local_vol_mc_available',side_effect=validation), \
             patch.object(run,'analyze_processed',side_effect=statistics), \
             patch.object(run,'simulate_processed',side_effect=simulation), \
             contextlib.redirect_stdout(io.StringIO()):
            # This test isolates stage ordering; final-report behavior has separate fixtures.
            with patch.object(run, '_final_analysis'):
                run._process_inventory(SimpleNamespace(cob='2031-02-02'))
        self.assertEqual(order,['validation','statistics','historical'])


    def test_abort_preserves_early_expiry_evidence_without_partial_pass(self):
        a=flat_adapter();q=quotes(a)
        original=coefficient
        def fail_later(adapter,t,k,allow):
            if t >= .3:raise ValueError('synthetic_later_failure')
            return original(adapter,t,k,allow)
        with patch('src.local_vol_monte_carlo.coefficient',side_effect=fail_later):
            with self.assertRaises(SimulationStopped) as caught:
                simulate_cohort(a,q,settings(),123)
        failure=caught.exception
        self.assertEqual(len(failure.partial['quotes']),6)
        self.assertEqual(len(failure.partial['boundaries']),1)
        self.assertEqual(failure.details['expiry_index'],2)

    def test_exit_records_failing_level_state_and_variance(self):
        a=flat_adapter();a.bounds=(-1e-10,1e-10)
        with self.assertRaises(SimulationStopped) as caught:
            simulate_cohort(a,quotes(a),settings(),123)
        r=caught.exception.details
        self.assertEqual(r['level'],'fine')
        self.assertGreater(r['failed_paths'],0)
        self.assertAlmostEqual(r['annualized_volatility_before'],.2)
        self.assertEqual(r['log_state_before'],0.)
        self.assertIn('timestamp',r)
