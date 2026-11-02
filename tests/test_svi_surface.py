"""Independent offline regressions for raw SVI, interpolation and diagnostics.

No user portfolio/configuration/date or real archive is a test dependency.
Prices come from known analytic smiles. Arbitrage and holdout tests exercise
economic invariants instead of merely asserting implementation internals.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import math
import tempfile
import unittest
import numpy as np

from test_volatility_surface import fixture,policy
from src.option_math import black_price,black_vega
from src.surface_model import validate_grid,black_prices
from src.surface_model import raw_svi,SviSurface,audit_slice,fit_raw_svi
from src.surface_checks import interval_feasibility,diagnostics
from src.volatility_surface import fit_calibrated,load_surface_config,_prepare_group,_stability


def settings():
    cfg=policy();cfg['schema_version']=2
    cfg['fit'].update(model='raw_svi',maximum_evaluations=600,svi=dict(midpoint_weight=.05,
        constraint_points=41,audit_points=201,constraint_iterations=200,constraint_refinements=2,arbitrage_tolerance=1e-9))
    cfg['diagnostics']=dict(forward_bump_fraction=.0001,block_holdout_fraction=.15,check_mid_expiry=True)
    return cfg


def known_model():
    return SviSurface([.2,.6,1.],[[.005,.04,-.5,.02,.15],[.012,.07,-.6,.04,.17],[.02,.11,-.55,.06,.2]],[-.2]*3,[.2]*3)


class SviMathTests(unittest.TestCase):
    def test_analytic_raw_derivatives(self):
        p=known_model().parameters[0];k=np.linspace(-.19,.19,25);h=1e-4
        w,wk,wkk=raw_svi(k,p);left=raw_svi(k-h,p)[0];right=raw_svi(k+h,p)[0]
        np.testing.assert_allclose(wk,(right-left)/(2*h),rtol=1e-5,atol=1e-8)
        np.testing.assert_allclose(wkk,(right-2*w+left)/h**2,rtol=1e-5,atol=1e-8)

    def test_price_interpolation_and_variance_derivatives(self):
        model=known_model();k=np.linspace(-.18,.18,21);t=.4
        expected=.5*(black_prices(1,np.exp(k),1,1,np.sqrt(raw_svi(k,model.parameters[0])[0]),True)
                     +black_prices(1,np.exp(k),1,1,np.sqrt(raw_svi(k,model.parameters[1])[0]),True))
        values=model.evaluate(k,t)
        actual=black_prices(1,np.exp(k),t,1,values['iv'],True)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        _,checks=validate_grid(model,policy()['grid'])
        self.assertTrue(checks['numerical_arbitrage_pass'],checks)
        self.assertTrue(checks['derivative_check_pass'],checks)
        self.assertFalse(checks['analytical_constraints_enforced'])

    def test_calendar_crossing_and_negative_density_are_detected(self):
        model=known_model();model.parameters[1]=[.001,.01,-.5,.02,.15]
        _,checks=validate_grid(model,policy()['grid'])
        self.assertFalse(checks['numerical_arbitrage_pass'])
        self.assertFalse(checks['node_audit']['passed'])
        checks=audit_slice([.001,1.,0.,0.,.01],-.5,.5)
        self.assertLess(checks['density'][0],0)

    def test_supported_domain_is_intersection_between_nodes(self):
        model=known_model();model.k_min=[-.2,-.1,-.15];model.k_max=[.2,.1,.15]
        self.assertEqual(tuple(float(x) for x in model.domain(.4)),(-.1,.1))
        self.assertTrue(np.isfinite(model.evaluate(.18,.2)['iv']))
        self.assertTrue(np.isnan(model.evaluate(.18,.2)['w_t']))
        for k,t in ((.18,.4),(0,.1),(0,1.1)):
            with self.assertRaises(ValueError):model.evaluate(k,t)
        self.assertEqual(float(model.evaluate(0.,.6)['time_derivative_at_knot']),1.)
        self.assertEqual(float(model.evaluate(0.,.4)['time_derivative_at_knot']),0.)

    def test_parameter_serialization_roundtrip(self):
        from dataclasses import asdict
        original=known_model();restored=SviSurface(**json.loads(json.dumps(asdict(original))))
        np.testing.assert_allclose(restored.evaluate([-.1,0,.1],.4)['iv'],original.evaluate([-.1,0,.1],.4)['iv'])
        with self.assertRaises(ValueError):SviSurface([.2,.1],original.parameters[:2],[-.2]*2,[.2]*2)


class SviWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cfg=settings();self.cal=fixture(self.root)
        expiries=sorted({r['expiration'] for r in self.cal.tables['option_iv']})
        for r in self.cal.tables['option_iv']:
            k=math.log(r['strike']/r['forward']);p=known_model().parameters[expiries.index(r['expiration'])]
            iv=math.sqrt(float(raw_svi(k,p)[0])/r['time_to_expiry'])
            price=black_price(r['forward'],r['strike'],r['time_to_expiry'],r['discount_factor'],iv)
            r.update(iv=iv,option_price=price,bid=price*.995,ask=price*1.005,
                     vega_per_unit_volatility=black_vega(r['forward'],r['strike'],r['time_to_expiry'],r['discount_factor'],iv))
        self.path=self.root/'surface.json'

    def fit(self):
        self.path.write_text(json.dumps(self.cfg),encoding='utf-8')
        with patch('socket.socket.connect',side_effect=AssertionError('Offline only')):
            return fit_calibrated(self.cal,self.path)

    def points(self):
        return _prepare_group(list(enumerate(self.cal.tables['option_iv'])),self.cal,self.cfg,'test')[0]

    def test_synthetic_live_fit_and_data_preservation(self):
        before=deepcopy(self.cal.tables);result=self.fit()
        self.assertEqual(result.manifest['summary']['fitted_cohorts'],1,result.tables['optimizer'])
        status=result.tables['cohort_status'][0]
        self.assertTrue(status['numerical_checks_pass'],status)
        self.assertTrue(status['market_fit_pass'],result.tables['expiry_metrics'])
        self.assertFalse(status['local_volatility_implemented'])
        self.assertEqual(result.models[status['group_id']].model_type,'raw_svi')
        self.assertEqual(before,self.cal.tables)
        self.assertTrue(result.tables['model_comparison'])
        self.assertTrue(result.tables['quote_diagnostics'])

    def test_historical_objective_does_not_invent_spreads(self):
        for r in self.cal.tables['option_iv']:
            r.update(route='historical',price_type='TRADES_bar_close');r.pop('bid');r.pop('ask')
        result=self.fit();self.assertTrue(result.models)
        self.assertTrue(all(r['inside_spread_fraction'] is None for r in result.tables['expiry_metrics']))
        self.assertTrue(all(r['status']=='not_checked' for r in result.tables['interval_feasibility']))

    def test_failed_arbitrage_audit_never_returns_a_surface(self):
        # Even a converged price fit cannot bypass an independently failed audit.
        with patch('src.surface_model.audit_slice',return_value={'density':(-1.,0.)}), \
             patch('src.surface_model.minimize',side_effect=lambda fun,x,**kw:SimpleNamespace(x=x,success=True,message='fixture')):
            model,records,_=fit_raw_svi(self.points(),self.cfg['fit'])
        self.assertIsNone(model);self.assertTrue(records)
        self.assertTrue(all(not r['arbitrage_pass'] for r in records))

    def test_interval_feasibility_missing_and_conflicting_bounds(self):
        points=self.points()[:9]
        for p in points:p['bid_iv']=.20;p['ask_iv']=.20
        self.assertEqual(interval_feasibility(points)['status'],'feasible')
        points[4]['bid_iv']=points[4]['ask_iv']=2.
        self.assertEqual(interval_feasibility(points)['status'],'infeasible')
        points[4].pop('bid_iv')
        self.assertEqual(interval_feasibility(points)['status'],'not_checked')

    def test_nonfinite_optimizer_result_is_a_serializable_failure(self):
        with patch('src.surface_model.audit_slice',return_value={'density':(-1.,0.)}), \
             patch('src.surface_model.minimize',return_value=SimpleNamespace(x=np.full(5,np.nan),success=False,message='fixture')):
            model,records,_=fit_raw_svi(self.points(),self.cfg['fit'])
        self.assertIsNone(model)
        json.dumps(records,allow_nan=False)

    def test_quote_conditioning_and_forward_sensitivity(self):
        points=self.points();times=sorted({p['reference_years'] for p in points})
        model=SviSurface(times,known_model().parameters,[-.21]*3,[.21]*3)
        before=deepcopy(points);tables=diagnostics(points,model,self.cfg['diagnostics'],.04)
        q=tables['quote_diagnostics'][0];p=points[0]
        self.assertAlmostEqual(q['spread_over_vega'],(p['ask']-p['bid'])/p['vega_per_unit_volatility'])
        bump=p['forward']*self.cfg['diagnostics']['forward_bump_fraction']
        upper=black_price(p['forward']+bump,p['strike'],p['time_to_expiry'],p['discount_factor'],p['iv'])
        lower=black_price(p['forward']-bump,p['strike'],p['time_to_expiry'],p['discount_factor'],p['iv'])
        self.assertAlmostEqual(q['first_order_price_change'],(upper-lower)/2,places=6)
        self.assertEqual(points,before)

    def test_contiguous_and_maturity_holdouts_are_explicit(self):
        points=self.points();times=sorted({p['reference_years'] for p in points})
        model=SviSurface(times,known_model().parameters,[-.21]*3,[.21]*3)
        self.cfg['stability']['enabled']=True;self.cfg['selection']['minimum_expiries']=2
        self.cfg['selection']['maximum_expiry_gap_years']=2
        with patch('src.volatility_surface._fit',return_value=(model,[],[])):
            records,_=_stability(model,points,[],self.cfg)
        names={r['check'] for r in records}
        self.assertIn('held_out_strike_block',names);self.assertIn('held_out_maturity',names)
        held=next(r for r in records if r['check']=='held_out_maturity')
        self.assertEqual(held['holdout_requested'],held['holdout_evaluated'])

    def test_schema_validation_and_legacy_model_remain_explicit(self):
        for name,value in [('constraint_points',True),('audit_points',10),('midpoint_weight',-1),('arbitrage_tolerance',.1)]:
            cfg=settings();cfg['fit']['svi'][name]=value;self.path.write_text(json.dumps(cfg))
            with self.subTest(name=name),self.assertRaises(ValueError):load_surface_config(self.path)
        self.path.write_text(json.dumps(policy()));self.assertEqual(load_surface_config(self.path)['schema_version'],1)

    def test_exports_include_new_diagnostics_without_nan_or_blank_csv_records(self):
        self.cfg['plots']=True;result=self.fit();target=result.save()
        for name in ('quote_diagnostics','timing_diagnostics','interval_feasibility','model_comparison'):
            self.assertTrue((target/(name+'.csv')).is_file())
        self.assertEqual(len(list(target.glob('*.png'))),3)
        self.assertNotIn('NaN',(target/'manifest.json').read_text(encoding='utf-8'))
        for p in target.glob('*.csv'):self.assertNotIn(b'\r\r\n',p.read_bytes())
        self.assertEqual(result.save(),target)


if __name__=='__main__':unittest.main()
