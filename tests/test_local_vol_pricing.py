"""Offline pricing invariants; synthetic inputs do not depend on user JSON."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import math
import unittest
import numpy as np
from test_local_volatility import flat_model, permission, source, settings
from src.local_volatility import LocalVolEvaluator, convert_surface
from src.local_vol_pricing import BoundaryLocalVol, ReferenceCurves, pricing_config, solve_calls, implied_vol, validate_repricing
from src.surface_model import black_prices
from src.surface_view import render_surface_3d, implied_patches


def fixture(root):
    surface=source(root)
    surface.manifest['identity']['config']['quality']=dict(maximum_scaled_price_rmse=2.,
        minimum_practical_coverage=.7,maximum_relative_outside_error=.02,
        maximum_outside_half_spreads=1.,maximum_outside_iv_error=.0025)
    refs=[];quotes=[];selected=[]
    for i,t in enumerate(surface.models['group'].times):
        expiry=str(i);f=100*math.exp(.01*t);d=math.exp(-.02*t)
        refs.append(dict(group_id='group',years=t,forward=f,discount_factor=d,spot=100.,expiration=expiry,
            curve_date='2031-02-03',reference_timestamp='2031-02-03T12:00:00Z'))
        for k in (-.1,0.,.1):
            for right in ('call','put'):
                strike=f*math.exp(k);price=float(black_prices(f,strike,t,d,.2,right=='call'));idx=len(quotes)
                quotes.append(dict(group_id='group',record_id=idx,years=t,k=k,strike=strike,option_type=right,
                    expiration=expiry,observed_iv=.2,original_forward=f,original_discount_factor=d,original_years=t,
                    observed_price=price,bid=price-.1,ask=price+.1,bid_iv=.19,ask_iv=.21))
                selected.append(dict(group_id='group',record_id=idx,selected=True,error_scale=.1))
    surface.tables.update(reference_inputs=refs,repricing=quotes,selection=selected)
    return surface


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.model=flat_model();self.ev=LocalVolEvaluator(self.model,permission(self.model),.05,1e-9)
        self.cfg=pricing_config(dict(enabled=True))

    def test_constant_vol_extension_and_strict_evaluator_are_separate(self):
        b=BoundaryLocalVol(self.ev,self.cfg)
        for t in (0.,.1,.25,.5,.75):
            np.testing.assert_allclose(b.variance(t,np.array([-3.,0.,2.])),.04,atol=1e-13)
            for variant in ('wing_alternative','short_end_alternative'):
                np.testing.assert_allclose(BoundaryLocalVol(self.ev,self.cfg,variant).variance(t,np.array([-2.,0.,2.])),.04,atol=1e-13)
        self.assertEqual(self.ev.evaluate(.1,0.)['status'],'unavailable')
        self.assertEqual(self.ev.evaluate(.4,1.)['status'],'unavailable')
        for t in (-.01,.751,float('nan')):
            with self.assertRaises(ValueError):b.variance(t,np.array([0.]))

    def test_missing_time_strip_and_atm_are_blocking(self):
        for change in (dict(start_years=.51),dict(k_min=.01)):
            ev=deepcopy(self.ev);ev.permission['domain'][1].update(change)
            with self.assertRaises(ValueError):BoundaryLocalVol(ev,self.cfg)

    def test_invalid_internal_variance_is_not_repaired(self):
        b=BoundaryLocalVol(self.ev,self.cfg)
        with patch.object(self.model,'evaluate',return_value=dict(w=.02,w_k=0.,w_kk=0.,w_t=-.01)):
            with self.assertRaisesRegex(ValueError,'coefficient'):b.variance(.3,np.array([0.]))

    def test_pde_recovers_black_prices_and_refines(self):
        b=BoundaryLocalVol(self.ev,self.cfg);errors=[]
        for n,steps in ((401,180),(801,360)):
            grid,prices=solve_calls(b,self.model.times,bounds=[-2.,2.],points=n,steps_per_year=steps)
            err=[]
            for t in self.model.times:
                ks=np.array([-.1,0.,.1]);actual=np.interp(ks,grid,prices[t])
                expected=black_prices(1.,np.exp(ks),t,1.,.2,True)
                err.extend(abs(actual-expected))
                self.assertGreaterEqual(float(prices[t].min()),-1e-12)
                self.assertLessEqual(float(np.diff(prices[t]).max()),1e-10)
            errors.append(max(err))
        self.assertLess(errors[1],errors[0]);self.assertLess(errors[1],3e-5)
        # Linear payoff is a null mode of the continuous PDE: put-call parity
        # is imposed analytically on normalized calls, not a second simulation.
        call=float(np.interp(.1,grid,prices[.75]));put=call-1+math.exp(.1)
        self.assertAlmostEqual(call-put,1-math.exp(.1))

    def test_forward_nodes_and_zero_anchor_preserved_negative_rates_allowed(self):
        rows=fixture(self.root).tables['reference_inputs'];curves=ReferenceCurves(rows,self.model.times,'2031-02-03')
        np.testing.assert_allclose(curves.at(0.),(100.,1.))
        for r in rows:np.testing.assert_allclose(curves.at(r['years']),(r['forward'],r['discount_factor']))
        f,d=curves.at(.375)
        self.assertAlmostEqual(f,math.sqrt(rows[0]['forward']*rows[1]['forward']))
        self.assertAlmostEqual(d,math.sqrt(rows[0]['discount_factor']*rows[1]['discount_factor']))
        for r in rows:r['discount_factor']=1.01
        ReferenceCurves(rows,self.model.times,'2031-02-03')
        with self.assertRaises(ValueError):curves.at(.8)
        rows[0]['curve_date']='2031-02-02'
        with self.assertRaisesRegex(ValueError,'date'):ReferenceCurves(rows,self.model.times,'2031-02-03')

    def test_iv_inversion_rejects_invalid_prices(self):
        self.assertIsNone(implied_vol(-1.,0.,.5));self.assertIsNone(implied_vol(1.1,0.,.5))
        self.assertAlmostEqual(implied_vol(float(black_prices(1.,1.,.5,1.,.2,True)),0.,.5),.2)

    def test_config_rejects_invalid_types_bounds_and_policies(self):
        for change in ({'enabled':1},{'space_points':True},{'space_points':400},{'short_end':'guess'},
                       {'wider_log_grid_bounds':[-2.,1.]},{'maximum_iv_error':float('nan')},
                       {'refinement':1},{'new_field':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):pricing_config(change)
        self.assertFalse(pricing_config(None)['enabled'])

    def test_live_and_historical_repricing_integrated_no_download_or_mc(self):
        s=fixture(self.root);cfg=settings();cfg['pricing']={'enabled':True,'space_points':1601}
        config=self.root/'lv.json';config.write_text(json.dumps(cfg))
        before=deepcopy(s.tables)
        with patch('socket.socket.connect',side_effect=AssertionError('offline')):
            result=convert_surface(s,config)
        self.assertEqual(s.tables,before)
        self.assertEqual(result.tables['pricing_summary'][0]['assessed_quotes'],18)
        self.assertEqual(result.tables['pricing_summary'][0]['status'],'validated_with_boundary_assumptions')
        self.assertFalse(result.tables['pricing_summary'][0]['full_simulation_ready'])
        s.tables['cohort_status'][0]['route']='historical'
        for r in s.tables['repricing']:
            for field in ('bid','ask','bid_iv','ask_iv'):r[field]=None
        historical=validate_repricing(s,result.evaluators,result.manifest['identity']['config']['pricing'])
        self.assertTrue(all(r['practical_coverage_fraction'] is None for r in historical['pricing_expiries']))
        self.assertTrue(all(r['inside_spread'] is None for r in historical['option_repricing']))

    def test_missing_reference_or_disabled_fails_closed(self):
        s=fixture(self.root);s.tables['reference_inputs']=[]
        result=validate_repricing(s,{'group':self.ev},self.cfg)
        self.assertEqual(result['pricing_summary'][0]['status'],'unavailable')
        self.assertEqual(result['option_repricing'],[])
        self.cfg['enabled']=False
        result=validate_repricing(s,{'group':self.ev},self.cfg)
        self.assertEqual(result['pricing_summary'][0]['reason'],'pricing_disabled')

    def test_plot_payload_has_no_remote_dependencies_and_escapes_injection(self):
        patches=implied_patches(self.model,permission(self.model))
        self.assertEqual(len(patches),2)
        for patch_,d in zip(patches,permission(self.model)['domain']):
            self.assertTrue(all(d['start_years']<point[0]<d['end_years'] for row in patch_['rows'] for point in row))
        patches[0]['label']='</script><script>alert(1)</script>'
        text=render_surface_3d('<bad>',patches)
        self.assertNotIn('</script><script>alert',text);self.assertIn('&lt;bad&gt;',text)
        self.assertNotIn('https://',text);self.assertNotIn('fetch(',text)
        payload=text.split('id="surface-data" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(json.loads(payload)['patches'][0]['label'],patches[0]['label'])
        self.assertIn("+'\\nT: '",text)


if __name__=='__main__':unittest.main()
