"""Scalable completion regressions using synthetic surfaces, never user settings."""
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
import unittest
import numpy as np
from test_local_vol_boundary import model,config
from test_local_vol_state import inputs
from test_local_volatility import permission
from src.local_volatility import LocalVolEvaluator,DateSpotLocalVol
from src.local_vol_pricing import pricing_boundary,pricing_config
from src.local_vol_monte_carlo import coefficient
from src.surface_model import SviSurface


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.m=model();self.ev=LocalVolEvaluator(self.m,permission(self.m),.05,1e-9)
        self.cfg=config(first_wing_tail_weight=.5,time_smoothing_fraction=.01)
        self.b=pricing_boundary(self.ev,self.cfg)

    def test_original_node_core_prices_and_slopes_preserved(self):
        old=pricing_boundary(self.ev,config())
        for j in range(len(self.m.times)):
            k=np.linspace(-.15,.15,51)
            for actual,expected in zip(self.b.node(j,k),old.node(j,k)):
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
        self.assertGreater(self.b.ends[0][0]['weight'],0)

    def test_exact_tail_order_and_positive_density_across_large_domain(self):
        k=np.linspace(-8.,8.,10001);last=None
        for j in range(len(self.m.times)):
            p,_,d=self.b.node(j,k)
            self.assertTrue(np.all(p>0));self.assertTrue(np.all(d>0))
            if last is not None:self.assertTrue(np.all(p-last >= -1e-12*np.maximum(p,last)))
            last=p
        self.assertEqual(self.b.audit['outer_tail_ordering'],'analytic_stationary_minimum')
        # Minimal ordering cannot require more slow-tail mass than the old
        # sufficient bound which makes that component dominate the whole tail.
        for j in range(1,len(self.m.times)):
            for side in (0,1):
                end=self.b.ends[j][side];prior=self.b.ends[j-1][side]
                distance=(end['join']-prior['join'])*(1 if side==0 else -1)
                p=prior['premium']*(prior['weight']*np.exp(self.b.low[side]*distance)+(1-prior['weight'])*np.exp(prior['high']*distance))
                self.assertLessEqual(end['weight'],p/end['premium']+1e-12)

    def test_variance_continuous_at_internal_maturity_knots(self):
        ks=np.array([-.5,-.1,0.,.1,.5]);eps=self.b.smoothing_width*1e-6
        for t in self.b.times[:-1]:
            left=self.b.variance(t-eps,ks);right=self.b.variance(t+eps,ks)
            np.testing.assert_allclose(left,right,rtol=2e-4,atol=1e-7)

    def test_positive_time_maps_stay_in_horizon_and_preserve_monotonicity(self):
        for t in np.linspace(0.,self.b.times[-1],1001):
            h,hp=self.b._time_window(float(t))
            self.assertGreaterEqual(t-h,-1e-14)
            self.assertLessEqual(t+h,self.b.times[-1]+1e-14)
            self.assertLess(abs(hp),1.)
        self.assertEqual(self.b._time_window(0.)[0],0.)
        self.assertEqual(self.b._time_window(float(self.b.times[-1]))[0],0.)
        np.testing.assert_allclose(self.b.variance(0.,np.array([0.])),.04)
        k=np.array([-.3,0.,.3]);t=float(self.b.times[-1])
        np.testing.assert_allclose(self.b.completed_values(t,k)['premium'],self.b.node(2,k)[0])

    def test_smoothed_call_derivatives_match_dupire_and_positive_density(self):
        # Convert OTM premiums to calls before differencing through k=0.
        def call(t,k):return self.b.completed_values(t,np.asarray(k))['premium']+np.maximum(1-np.exp(k),0.)
        for t in (.1,.249,.25,.251,.5,.6):
            k=np.array([-.1,0.,.1]);hk=1e-4;ht=1e-6
            c=call(t,k);plus=call(t,k+hk);minus=call(t,k-hk)
            density=(plus-2*c+minus)/hk**2-(plus-minus)/(2*hk)
            ct=(call(t+ht,k)-call(t-ht,k))/(2*ht)
            self.assertTrue(np.all(density>0));self.assertTrue(np.all(ct>=0))
            np.testing.assert_allclose(2*ct/density,self.b.variance(t,k),rtol=2e-4,atol=1e-6)

    def test_time_scaling_does_not_introduce_asset_or_date_dependencies(self):
        factor=3.7
        m=SviSurface(np.asarray(self.m.times)*factor,self.m.parameters,self.m.k_min,self.m.k_max)
        b=pricing_boundary(LocalVolEvaluator(m,permission(m),.05,1e-9),self.cfg)
        self.assertAlmostEqual(b.smoothing_width,self.b.smoothing_width*factor)
        for t in (.1,.25,.4,.5):
            k=np.array([-.3,0.,.3])
            np.testing.assert_allclose(b.variance(t*factor,k)*factor,self.b.variance(t,k),rtol=1e-10)

    def test_quadrature_refinement_agrees_on_supported_states(self):
        b=pricing_boundary(self.ev,{**self.cfg,'time_smoothing_quadrature':24})
        for t in (.001,.01,.249,.25,.251):
            k=np.array([-.1,0.,.1])
            np.testing.assert_allclose(b.variance(t,k),self.b.variance(t,k),rtol=1e-4,atol=1e-7)

    def test_time_mesh_resolves_windows_and_refines(self):
        a=self.b.time_nodes(0.,.25,365);b=self.b.time_nodes(0.,.25,730)
        self.assertEqual(a[0],0);self.assertEqual(a[-1],.25)
        self.assertTrue(np.all(np.diff(a)>0));self.assertGreater(len(b),len(a))
        near=a[a>.25-2*self.b.smoothing_width]
        self.assertLessEqual(np.diff(near).max(),self.b.smoothing_width/4+1e-12)

    def test_state_adapter_and_mc_use_same_completed_model_with_opt_in(self):
        surface,origin=inputs(Path('.'))
        adapter=DateSpotLocalVol(self.ev,surface.tables['reference_inputs'],'2031-02-03',self.cfg)
        for t in (.1,.249,.25,.251,.4):
            k=np.array([-.3,0.,.3]);f,_=adapter.curves.at(t)
            row=adapter.evaluate(origin+timedelta(days=365*t),f*np.exp(k),allow_boundary_assumptions=True)
            v,flags=coefficient(adapter,t,k,True)
            self.assertEqual(row['status'],'available')
            self.assertTrue(row['time_smoothing_assumption']);self.assertTrue(flags['time_smoothing'])
            np.testing.assert_allclose(v,row['local_variance'],rtol=1e-10)
            np.testing.assert_allclose(v,self.b.variance(t,k),rtol=1e-10)
        strict=adapter.evaluate(origin+timedelta(days=365*.251),adapter.curves.at(.251)[0])
        self.assertFalse(strict['time_smoothing_assumption'])
        self.assertAlmostEqual(strict['local_variance'],self.ev.evaluate(.251,0.)['local_variance'])

    def test_invalid_controls_are_rejected(self):
        for values in (dict(first_wing_tail_weight=1),dict(first_wing_tail_weight=True),
                       dict(time_smoothing_fraction=-.01),dict(time_smoothing_fraction=.1),
                       dict(time_smoothing_quadrature=0),dict(time_smoothing_time_points=True)):
            with self.assertRaises(ValueError):pricing_config({**self.cfg,**values})
