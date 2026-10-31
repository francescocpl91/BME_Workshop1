"""Independent known-answer tests for Monte Carlo variance reduction."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from test_local_vol_monte_carlo import settings,flat_adapter,quotes
from src.local_vol_monte_carlo import load_config,simulate_cohort,_quote_statistics,coefficient
from src.local_vol_pricing import pricing_config
from src.surface_model import black_prices

class PrecisionTests(unittest.TestCase):
    def test_coupled_black_flat_diffusion_is_exact_and_preserves_parity(self):
        a=flat_adapter();cfg={**settings(),'control_variate':'coupled_black'}
        rows,forwards,_,_=simulate_cohort(a,quotes(a),cfg,91)
        for row in rows:
            self.assertAlmostEqual(row['fine_price'],row['target_price'],places=11)
            self.assertLess(row['standard_error'],1e-12)
            self.assertTrue(row['precision_pass'])
        for call,put in zip(rows[::2],rows[1::2]):
            self.assertAlmostEqual(call['fine_price']-put['fine_price'],
                call['reference_discount']*(call['reference_forward']-call['reference_strike']),places=11)
        self.assertTrue(all(r['passed'] for r in forwards))

    def test_control_does_not_insert_target_price_or_hide_wrong_model(self):
        a=flat_adapter();q=quotes(a)[2];t=q['years']
        w=np.r_[np.array([-.1,.2,.4]),np.array([.1,-.2,-.4])]
        x=np.exp(-.5*.2**2*t+.2*w)
        q['target_price']+=10.;q['pde_price']+=10.
        r=_quote_statistics(q,x,x,4.,{**settings(),'control_variate':'coupled_black'},w)
        self.assertAlmostEqual(r['fine_price'],q['target_price']-10.,places=12)
        self.assertFalse(r['fit_pass']);self.assertFalse(r['pde_pass'])

    def test_different_auxiliary_volatility_still_estimates_true_payoff(self):
        a=flat_adapter(.3);q=quotes(a,.3)[2];t=q['years'];q['fitted_iv']=.18
        z=np.random.default_rng(81).standard_normal(100000)*math.sqrt(t)
        w=np.r_[z,-z];x=np.exp(-.5*.3**2*t+.3*w)
        r=_quote_statistics(q,x,x,5.,{**settings(),'control_variate':'coupled_black'},w)
        self.assertLess(abs(r['fine_price']-q['target_price']),5*r['standard_error'])
        self.assertGreater(r['standard_error'],0.)
        self.assertNotAlmostEqual(r['control_price'],q['target_price'])

    def test_paired_convergence_is_unchanged_by_shared_control(self):
        q=quotes(flat_adapter())[2];t=q['years']
        z=np.random.default_rng(91).normal(size=1000)*math.sqrt(t);w=np.r_[z,-z]
        x=np.exp(-.5*.22**2*t+.22*w);y=np.exp(-.5*.21**2*t+.21*w)
        old=_quote_statistics(q,x,y,4.,settings())
        new=_quote_statistics(q,x,y,4.,{**settings(),'control_variate':'coupled_black'},w)
        self.assertAlmostEqual(old['paired_difference'],new['paired_difference'],places=12)
        self.assertAlmostEqual(old['paired_difference_se'],new['paired_difference_se'],places=12)
        self.assertEqual(old['raw_price'],new['raw_price'])
        self.assertEqual(old['standard_error'],new['delta_control_standard_error'])

    def test_invalid_control_inputs_fail_explicitly(self):
        q=quotes(flat_adapter())[0];cfg={**settings(),'control_variate':'coupled_black'}
        for w in (None,[1,2],[0,0,0,float('nan')]):
            with self.assertRaisesRegex(ValueError,'control_brownian'):
                _quote_statistics(q,np.ones(4),np.ones(4),4.,cfg,w)
        with TemporaryDirectory() as temp:
            p=Path(temp)/'mc.json'
            p.write_text(json.dumps(cfg),encoding='utf-8')
            self.assertEqual(load_config(p)['control_variate'],'coupled_black')
            for mode in ('unknown',True,7):
                p.write_text(json.dumps({**cfg,'control_variate':mode}),encoding='utf-8')
                with self.assertRaises(ValueError):load_config(p)

    def test_wider_grid_keeps_spacing_and_rejects_unsafe_resolution(self):
        # Numerical fixtures only: never depend on the user's current JSON.
        p=pricing_config(dict(log_grid_bounds=[-6.,3.],wider_log_grid_bounds=[-7.,4.],space_points=5761))
        self.assertAlmostEqual((p['log_grid_bounds'][1]-p['log_grid_bounds'][0])/(p['space_points']-1),5/3200)
        for n in (8002,16001,True):
            with self.assertRaises(ValueError):pricing_config(dict(space_points=n))

    def test_smoothed_completion_evaluated_once_with_strict_checks_retained(self):
        a=flat_adapter();a.boundary.smoothing_width=.001;calls=[]
        def variance(t,k):
            calls.append(np.asarray(k).copy());return np.full_like(k,.04)
        a.boundary.variance=variance
        k=np.array([-.5,0.,.5]);v,flags=coefficient(a,.3,k,True)
        self.assertEqual(len(calls),1);np.testing.assert_equal(calls[0],k)
        np.testing.assert_allclose(v,.04);self.assertTrue(flags['time_smoothing'])
        a.ev.minimum_density=2.
        with self.assertRaisesRegex(ValueError,'strict_evaluator'):coefficient(a,.3,k,True)
        self.assertEqual(len(calls),1)
