"""Offline surface regressions: synthetic prices, isolated settings and outputs.

No test relies on the user's portfolio, calibration date or provider JSONs.
Known SSVI prices establish recoverability; failures remain explicit rather
than manufacturing a surface from absent inputs. No real archive is written.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from zoneinfo import ZoneInfo
import json
import math
import tempfile
import unittest
import numpy as np

from src.option_math import black_price,black_vega
from src.ois_bootstrap import DiscountCurve
from src.option_calibration import CalibrationResult,_curve_time
from src.surface_model import ssvi,SurfaceModel,black_prices,fit_ssvi,validate_grid,eta_limit
from src.volatility_surface import fit_calibrated,load_surface_config


def policy():
    return dict(schema_version=1,enabled=True,plots=False,instructions={},
        timing=dict(mode='intraday_approximation',maximum_capture_minutes=30.,maximum_spot_range_fraction=.01,
                    maximum_carry_dispersion_fraction=.001),
        selection=dict(minimum_expiries=3,minimum_unique_strikes=5,minimum_each_side=1,minimum_near_forward=1,
            near_forward_abs_k=.06,maximum_abs_k=.3,maximum_relative_spread=.1,maximum_expiry_gap_years=.6,maximum_strike_gap_k=.08),
        fit=dict(initial_rhos=[-.6,-.2,.2],maximum_evaluations=160),
        quality=dict(price_error_floor_spot_fraction=1e-6,historical_relative_error=.05,
                     minimum_inside_spread_fraction=.8,maximum_scaled_price_rmse=2.),
        stability=dict(enabled=False,holdout_stride=3,maximum_holdout_scaled_rmse=3.,maximum_iv_change=.03,
                       seed=5,narrow_window_minutes=5.),
        grid=dict(moneyness_points=11,maturity_points=7,derivative_tolerance=1e-4))


def fixture(folder,route='live'):
    zone='America/New_York'; stamp=datetime(2026,10,20,16,tzinfo=ZoneInfo(zone))
    curve=DiscountCurve('2026-10-20','USD','SOFR',(0.,2.),(1.,math.exp(-.04*2)),{'status':'assumed'})
    rows=[]
    for expiry,theta in zip(('2027-01-20','2027-04-20','2027-10-20'),(.012,.024,.047)):
        end=datetime.fromisoformat(expiry+'T16:00:00').replace(tzinfo=ZoneInfo(zone))
        years=(end.astimezone(timezone.utc)-stamp.astimezone(timezone.utc)).total_seconds()/(365*86400)
        d=curve.interval_discount(_curve_time(stamp,curve,zone),_curve_time(end,curve,zone))
        forward=100*math.exp(-.015*years)/d
        for j,k in enumerate(np.linspace(-.2,.2,9)):
            strike=forward*math.exp(float(k))
            iv=float(np.sqrt(ssvi(k,theta,-.35,.65,.3)[0]/years))
            price=black_price(forward,strike,years,d,iv)
            row=dict(instrument_id=expiry+'_'+str(j),provider='test_provider',route=route,
                underlying='ARBITRARY_INDEX',currency='USD',valuation_date='2026-10-20',curve_date='2026-10-20',
                status='iv_calculated_with_assumptions',exercise_style='european',settlement='cash',settlement_time='PM',
                option_type='call',expiration=expiry,expiry_timestamp=end.isoformat(),valuation_timestamp=stamp.isoformat(),
                price_type='bid_ask_midpoint' if route=='live' else 'MIDPOINT_bar_close',
                iv=iv,spot=100.,strike=strike,forward=forward,discount_factor=d,time_to_expiry=years,option_price=price,
                vega_per_unit_volatility=black_vega(forward,strike,years,d,iv),dividend_pv=100-forward*d)
            if route=='live':row.update(bid=price*.995,ask=price*1.005)
            rows.append(row)
    manifest=dict(archive_cob='2026-10-19',identity={'config':{'valuation_timezone':zone},'fixture':'independent'},
                  curve_definitions={'2026-10-20':{'USD':asdict(curve)}})
    return CalibrationResult(Path(folder),'fixture',manifest,{'option_iv':rows})


class SurfaceMathTests(unittest.TestCase):
    def test_ssvi_flat_smile_limit_and_independent_black_prices(self):
        w,wk,wkk,wt=ssvi(np.array([-.2,0.,.2]),.04,-.3,0.,.3)
        np.testing.assert_allclose(w,.04);np.testing.assert_allclose(wk,0);np.testing.assert_allclose(wkk,0)
        np.testing.assert_allclose(wt,1)
        for f,k,t,d,v in ((110,90,.3,.97,.25),(90,110,.6,.98,.4),(100,100,1,1,.2)):
            for call in (True,False):
                self.assertAlmostEqual(float(black_prices(f,k,t,d,v,call)),black_price(f,k,t,d,v,'call' if call else 'put'),places=11)

    def test_constraints_and_analytic_derivatives_across_parameter_range(self):
        for rho in (-.9,0.,.9):
            for gamma in (0.,.25,.5):
                model=SurfaceModel([.1,.5,1.],[.008,.025,.045],rho,1.8/(1+abs(rho)),gamma,[-.5]*3,[.5]*3)
                _,report=validate_grid(model,policy()['grid'])
                self.assertTrue(report['numerical_arbitrage_pass'],report)
                self.assertTrue(report['derivative_check_pass'],report)
                self.assertTrue(all(report['normalized_call_checks'].values()),report)

    def test_full_sufficient_parameter_bounds_do_not_overrestrict_fit(self):
        # This valid eta is above the old conservative 2/(1+abs(rho)) cap.
        rho=-.6;gamma=.5
        self.assertAlmostEqual(eta_limit(rho,gamma),2/math.sqrt(1+abs(rho)))
        model=SurfaceModel([.1,.5,2.],[.01,.1,1.],rho,1.4,gamma,[-.5]*3,[.5]*3)
        _,checks=validate_grid(model,policy()['grid'])
        self.assertTrue(checks['numerical_arbitrage_pass'])
        for gamma in (0.,.25,.5):
            eta=.999*eta_limit(rho,gamma)
            theta=np.geomspace(1e-8,1e8,1000)
            phi=eta/(theta**gamma*(1+theta)**(1-gamma))
            self.assertTrue(np.all(theta*phi*(1+abs(rho))<4))
            self.assertTrue(np.all(theta*phi**2*(1+abs(rho))<4))
        with self.assertRaises(ValueError):ssvi(0.,.1,rho,100.,.5)

    def test_public_evaluator_refuses_all_extrapolation(self):
        model=SurfaceModel([.1,.5],[.01,.02],-.3,.6,.3,[-.2,-.1],[.2,.1])
        for k,t in ((0,.01),(0,1),(.3,.2),(-.3,.2),(float('nan'),.2)):
            with self.subTest(k=k,t=t),self.assertRaises(ValueError):model.evaluate(k,t)
        with self.assertRaises(ValueError):SurfaceModel([.1,.5],[.02,.01],-.3,.6,.3,[-.2]*2,[.2]*2)


class SurfaceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cfg=policy();self.path=self.root/'surface.json'
        self.calibration=fixture(self.root)

    def fit(self):
        self.path.write_text(json.dumps(self.cfg),encoding='utf-8')
        with patch('socket.socket.connect',side_effect=AssertionError('No downloads')):
            return fit_calibrated(self.calibration,self.path)

    def test_known_surface_recovery_repricing_and_no_input_mutation(self):
        before=deepcopy((self.calibration.manifest,self.calibration.tables))
        result=self.fit()
        self.assertEqual(result.manifest['summary']['fitted_cohorts'],1)
        cohort=result.tables['cohort_status'][0]
        self.assertTrue(cohort['market_fit_pass'],cohort)
        self.assertTrue(cohort['numerical_checks_pass'],cohort)
        self.assertFalse(cohort['local_volatility_implemented'])
        self.assertEqual(cohort['stability_status'],'not_checked')
        self.assertEqual(cohort['status'],'diagnostic_fit')
        self.assertLess(max(abs(r['fitted_iv']-r['observed_iv']) for r in result.tables['repricing']),1e-4)
        self.assertEqual(before,(self.calibration.manifest,self.calibration.tables))
        self.assertEqual(len(result.tables['reference_inputs']),3)

    def test_historical_without_broker_iv_or_bid_ask_is_supported(self):
        self.calibration=fixture(self.root,'historical')
        result=self.fit()
        self.assertEqual(len(result.models),1)
        self.assertTrue(all(r['inside_spread'] is None for r in result.tables['repricing']))

    def test_intraday_normalization_preserves_original_quote_inputs(self):
        from datetime import timedelta
        curve=DiscountCurve(**self.calibration.manifest['curve_definitions']['2026-10-20']['USD'])
        zone='America/New_York'
        for i,r in enumerate(self.calibration.tables['option_iv']):
            # Generate independent prices at different captures, with known
            # constant IV at each original k and unchanged dividend yield.
            original_k=math.log(r['strike']/r['forward'])
            stamp=datetime.fromisoformat(r['valuation_timestamp'])+timedelta(seconds=30*i)
            end=datetime.fromisoformat(r['expiry_timestamp'])
            t=(end.astimezone(timezone.utc)-stamp.astimezone(timezone.utc)).total_seconds()/(365*86400)
            d=curve.interval_discount(_curve_time(stamp,curve,zone),_curve_time(end,curve,zone))
            spot=100+.005*i;forward=spot*math.exp(-.015*t)/d;strike=forward*math.exp(original_k)
            price=black_price(forward,strike,t,d,r['iv'])
            r.update(valuation_timestamp=stamp.isoformat(),time_to_expiry=t,discount_factor=d,
                     spot=spot,forward=forward,strike=strike,option_price=price,bid=price*.995,ask=price*1.005,
                     vega_per_unit_volatility=black_vega(forward,strike,t,d,r['iv']))
        original=deepcopy(self.calibration.tables)
        result=self.fit();cohort=result.tables['cohort_status'][0]
        self.assertEqual(cohort['capture_minutes'],13.)
        self.assertTrue(cohort['market_fit_pass'],cohort)
        reference=datetime.fromisoformat(self.calibration.tables['option_iv'][13]['valuation_timestamp']).astimezone(timezone.utc).isoformat()
        self.assertEqual(cohort['reference_timestamp'],reference)
        self.assertEqual(original,self.calibration.tables)
        for row in result.tables['repricing']:
            source=original['option_iv'][row['record_id']]
            self.assertEqual(row['original_forward'],source['forward'])
            self.assertEqual(row['original_years'],source['time_to_expiry'])
        for ref in result.tables['reference_inputs']:
            self.assertAlmostEqual(ref['equivalent_dividend_yield'],.015)
            self.assertAlmostEqual(ref['forward'],ref['spot']*math.exp(-.015*ref['years'])/ref['discount_factor'])

    def test_missing_options_curves_and_conventions_are_explicit(self):
        for field in ('options','curve','conventions'):
            self.calibration=fixture(self.root)
            if field=='options':self.calibration.tables['option_iv']=[]
            elif field=='curve':self.calibration.manifest['curve_definitions']={}
            else:
                for r in self.calibration.tables['option_iv']:r.pop('exercise_style')
            with self.subTest(field=field):
                result=self.fit();self.assertEqual(result.manifest['summary']['status'],'skipped')
                self.assertFalse(result.models)

    def test_date_mismatch_cannot_be_normalized_away(self):
        for r in self.calibration.tables['option_iv']:r['curve_date']='2026-10-19'
        result=self.fit()
        self.assertFalse(result.models)
        self.assertEqual({r['reason'] for r in result.tables['selection']},{'surface_date_mismatch'})

    def test_capture_window_and_unknown_route_are_explicit_failures(self):
        # Shift a complete expiry by one hour and update its lifetime so it is
        # otherwise internally consistent. Pooling it must fail the time gate.
        from datetime import timedelta
        for r in self.calibration.tables['option_iv'][:9]:
            r['valuation_timestamp']=(datetime.fromisoformat(r['valuation_timestamp'])-timedelta(hours=1)).isoformat()
            r['time_to_expiry']+=1/(365*24)
        result=self.fit()
        self.assertFalse(result.models)
        self.assertEqual(result.tables['cohort_status'][0]['reason'],'capture_window_exceeded')
        self.calibration=fixture(self.root,'unrecognized')
        result=self.fit()
        self.assertFalse(result.models)
        self.assertEqual({r['reason'] for r in result.tables['selection']},{'unknown_option_route'})

    def test_mixed_underlyings_and_currencies_are_separate_groups(self):
        other=deepcopy(self.calibration.tables['option_iv'])
        for r in other:r['underlying']='SECOND';r['instrument_id']='second_'+r['instrument_id']
        self.calibration.tables['option_iv']+=other
        result=self.fit();self.assertEqual(len(result.models),2)
        self.assertEqual({r['underlying'] for r in result.tables['cohort_status']},{'ARBITRARY_INDEX','SECOND'})

    def test_conflicting_duplicates_and_missing_wing_coverage_are_rejected(self):
        extra=deepcopy(self.calibration.tables['option_iv'][0]);extra['option_price']*=1.001
        extra['bid']=extra['option_price']*.995;extra['ask']=extra['option_price']*1.005
        self.calibration.tables['option_iv'].append(extra)
        result=self.fit()
        self.assertEqual(sum(r['reason']=='conflicting_duplicate_contract' for r in result.tables['selection']),2)
        self.calibration=fixture(self.root)
        self.calibration.tables['option_iv']=[r for r in self.calibration.tables['option_iv'] if r['strike']>=r['forward']]
        result=self.fit();self.assertFalse(result.models)
        self.assertIn('insufficient_strike_coverage',{r['reason'] for r in result.tables['selection']})

    def test_carry_dispersion_and_underlying_movement_are_blocking(self):
        self.calibration.tables['option_iv'][0]['forward']*=1.002
        result=self.fit()
        self.assertIn('inconsistent_expiry_carry',{r['reason'] for r in result.tables['selection']})
        self.calibration=fixture(self.root)
        self.calibration.tables['option_iv'][0]['spot']=110
        result=self.fit()
        self.assertFalse(result.models)
        self.assertEqual(result.tables['cohort_status'][0]['reason'],'underlying_moved_too_far')

    def test_insufficient_expiries_and_optimizer_failure_skip(self):
        self.calibration.tables['option_iv']=self.calibration.tables['option_iv'][:9]
        result=self.fit();self.assertFalse(result.models)
        self.calibration=fixture(self.root)
        with patch('src.volatility_surface.fit_ssvi',return_value=(None,[],[])):
            result=self.fit()
        self.assertEqual(result.tables['cohort_status'][0]['reason'],'optimizer_failed')

    def test_failed_market_quality_keeps_diagnostics_without_acceptance(self):
        self.cfg['quality']['maximum_scaled_price_rmse']=1e-15
        result=self.fit();cohort=result.tables['cohort_status'][0]
        self.assertEqual(cohort['status'],'diagnostic_fit')
        self.assertFalse(cohort['market_fit_pass'])
        self.assertTrue(result.tables['repricing'])

    def test_stability_checks_are_reproducible_and_do_not_mutate_observations(self):
        self.cfg['stability']['enabled']=True
        original=deepcopy(self.calibration.tables)
        first=self.fit();second=self.fit()
        self.assertEqual(first.tables['stability'],second.tables['stability'])
        self.assertEqual(original,self.calibration.tables)
        self.assertTrue(any(r['check']=='held_out_strikes' for r in first.tables['stability']))
        self.assertTrue(any(r['check'].startswith('quote_perturbation') for r in first.tables['stability']))

    def test_save_reuse_hashes_and_missing_index_records(self):
        result=self.fit();target=result.save()
        for csv_file in target.glob('*.csv'):
            self.assertNotIn(b'\r\r\n',csv_file.read_bytes())
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in target.iterdir()}
        self.assertEqual(result.save(),target)
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in target.iterdir()})
        manifest=target/'manifest.json';doc=json.loads(manifest.read_text(encoding='utf-8'))
        doc['files'].pop('repricing');manifest.write_text(json.dumps(doc),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'inconsistent'):result.save()

    def test_changed_inputs_and_policies_have_new_result_ids(self):
        first=self.fit();self.cfg['quality']['maximum_scaled_price_rmse']=3
        second=self.fit();self.assertNotEqual(first.run_id,second.run_id)
        self.calibration.tables['option_iv'][0]['source_note']='provenance changed'
        third=self.fit();self.assertNotEqual(second.run_id,third.run_id)

    def test_png_exports_are_nonempty_and_checksummed(self):
        self.cfg['plots']=True;result=self.fit();target=result.save()
        files=list(target.glob('*.png'));self.assertEqual(len(files),3)
        self.assertTrue(all(p.stat().st_size>1000 for p in files))
        self.assertEqual(result.save(),target)

    def test_invalid_settings_are_controlled_errors_and_do_not_read_data(self):
        values=[None,[],{'bad':True}]
        for base in values:
            self.path.write_text(json.dumps(base))
            with self.assertRaises(ValueError):load_surface_config(self.path)
        for section,key,value in [('timing','mode',None),('selection','minimum_expiries',True),
              ('fit','initial_rhos',None),('grid','derivative_tolerance',-1),('quality','minimum_inside_spread_fraction',2)]:
            cfg=policy();cfg[section][key]=value;self.path.write_text(json.dumps(cfg))
            with self.subTest(key=key),self.assertRaises(ValueError):load_surface_config(self.path)

    def test_run_stage_handles_missing_calibration_and_config(self):
        import run
        with redirect_stdout(StringIO()) as output:
            self.assertIsNone(run._surface_available(None))
        self.assertEqual(output.getvalue(), '')  # upstream cause is shown once, not repeated
        with patch.object(run,'SURFACE_FILE',self.root/'absent.json'),redirect_stdout(StringIO()) as output:
            self.assertIsNone(run._surface_available(self.calibration))
        self.assertIn('unavailable',output.getvalue())
        with patch('src.volatility_surface.fit_calibrated',side_effect=RuntimeError('programming error')):
            with self.assertRaises(RuntimeError):run._surface_available(self.calibration)

    def test_run_uses_current_calibration_and_continues_existing_analysis(self):
        import run
        processed=Mock(run_id='fixture',summary={'portfolio':{'holdings':[]},
            'readiness':{'aligned_return_observations':0,'foreign_currencies':[]},'options':{'iv_candidates':0}})
        processed.save.return_value=self.root
        stats=Mock(run_id='stats',summary={'portfolio_status':'unavailable','warnings':[]})
        mc=Mock(run_id='mc',summary={'status':'unavailable','warnings':[]})
        order=[]
        def surface(value):self.assertIs(value,self.calibration);order.append('surface')
        def statistics(*args):order.append('statistics');return stats
        def simulation(*args):order.append('mc');return mc
        with patch('src.volatility_workflow.save_overview',return_value=Path('.')),patch.object(run,'process_inventory',return_value=processed), \
             patch.object(run,'_calibrate_available',return_value=self.calibration), \
             patch.object(run,'_surface_available',side_effect=surface), \
             patch.object(run,'analyze_processed',side_effect=statistics), \
             patch.object(run,'simulate_processed',side_effect=simulation),redirect_stdout(StringIO()):
            run._process_inventory(SimpleNamespace(cob='2026-10-19'))
        self.assertEqual(order,['surface','statistics','mc'])


if __name__=='__main__':unittest.main()
