"""Completion regressions use synthetic smiles, never user JSON or live data."""
from copy import deepcopy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from test_local_volatility import permission, flat_model
from test_local_vol_pricing import fixture
from src.surface_model import SviSurface, black_prices
from src.local_volatility import LocalVolEvaluator
from src.local_vol_pricing import pricing_config, pricing_boundary, solve_calls, validate_repricing


def model():
    # Constant-strike smiles; flat 20% IV at nodes, with the SAME linear-call
    # interpolation as production (not constant IV between nodes).
    return SviSurface([.25,.5,.75],[[w-1e-15,1e-14,0.,0.,.1] for w in (.01,.02,.03)],[-.2]*3,[.2]*3)


def config(**changes):
    return pricing_config(dict(enabled=True,short_end='first_smile_total_variance',
        wings='calendar_ordered_power_prices',boundary_audit_points=2001,**changes))


class CompletedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.m=model();self.ev=LocalVolEvaluator(self.m,permission(self.m),.05,1e-9)
        self.cfg=config();self.b=pricing_boundary(self.ev,self.cfg)

    def test_node_core_and_join_price_slope_preserved(self):
        for j,t in enumerate(self.m.times):
            k=np.linspace(-.15,.15,31)
            actual=self.b.node(j,k)[0]
            expected=black_prices(1.,np.exp(k),t,1.,.2,k>=0)
            np.testing.assert_allclose(actual,expected,atol=2e-15)
            for side,end in enumerate(self.b.ends[j]):
                h=1e-8;k=end['join'];p,slope,_=self.b.node(j,np.array([k-h,k,k+h]))
                self.assertAlmostEqual(float((p[2]-p[0])/(2*h)),float(slope[1]),places=6)
                self.assertLess(abs(slope[2]-slope[0]),1e-5)
                self.assertAlmostEqual(end['weight']*self.b.low[side]+(1-end['weight'])*end['high'],end['effective'])

    def test_calendar_and_density_hold_in_far_wings(self):
        k=np.linspace(-3.,2.,2501);last=None
        for j in range(3):
            p,s,d=self.b.node(j,k)
            self.assertTrue(np.all(d>0));self.assertTrue(np.all(p>0))
            if last is not None:self.assertTrue(np.all(p>=last-1e-12*np.maximum(last,p)))
            last=p
        self.assertEqual(self.b.audit['status'],'passed')
        self.assertFalse(any(r['observed_support'] for r in self.b.records))

    def test_underflowed_wing_join_is_unavailable_not_division_by_zero(self):
        tiny = SviSurface([.25,.5,.75],
            [[w,1e-14,0.,0.,.1] for w in (1e-8,2e-8,3e-8)],[-.2]*3,[.2]*3)
        ev = LocalVolEvaluator(tiny,permission(tiny),.05,1e-9)
        with self.assertRaisesRegex(ValueError,'invalid_wing_join_price_slope_or_density'):
            pricing_boundary(ev,self.cfg)
        with TemporaryDirectory() as tmp:
            surface=fixture(Path(tmp));surface.models['group']=tiny
            out=validate_repricing(surface,{'group':ev},self.cfg)
            self.assertEqual(out['pricing_summary'][0]['status'],'unavailable')
            self.assertEqual(out['pricing_summary'][0]['reason'],'invalid_wing_join_price_slope_or_density')
            self.assertEqual(out['option_repricing'],[])

    def test_high_variance_inversion_is_independent_of_batch(self):
        # This used to fail for [.3] but succeed when an unrelated far strike
        # expanded the batch-wide upper bracket. Use large but valid variance.
        high = SviSurface([.25,.5,.75],
            [[w-1e-15,1e-14,0.,0.,.1] for w in (1.,2.,3.)],[-.2]*3,[.2]*3)
        ev = LocalVolEvaluator(high,permission(high),.05,1e-9)
        boundary = pricing_boundary(ev,self.cfg)
        ks = np.array([.3,2.,-.4,0.])
        batch = boundary.variance(.1,ks)
        singles = np.array([float(boundary.variance(.1,np.array(k))) for k in ks])
        np.testing.assert_allclose(batch,singles,rtol=1e-12)
        np.testing.assert_allclose(boundary.variance(.1,ks[::-1]),batch[::-1],rtol=1e-12)
        np.testing.assert_allclose(boundary.variance(.1,ks.reshape(2,2)).ravel(),batch,rtol=1e-12)
        w,_,_ = boundary._first_variance(ks)
        np.testing.assert_allclose(black_prices(1,np.exp(ks),1,1,np.sqrt(w),ks>=0),
                                   boundary.node(0,ks)[0],rtol=1e-10)

    def test_inversion_rejects_impossible_prices_without_clipping(self):
        with patch.object(self.b,'node',return_value=(np.array([1.]),np.array([0.]),np.array([1.]))):
            with self.assertRaisesRegex(ValueError,'invalid_premium'):
                self.b._first_variance(np.array([0.]))

    def test_short_end_anchor_and_chain_rule(self):
        k=np.array([-.6,-.1,0.,.1,.6]);w,wk,wkk=self.b._first_variance(k)
        np.testing.assert_allclose(black_prices(1,np.exp(k),1,1,np.sqrt(w),k>=0),self.b.node(0,k)[0],rtol=1e-9)
        t=.1;a=t/.25
        g=(1-k*wk/(2*w))**2+a*(.5*wkk-.25*wk*wk/w)-a*a*wk*wk/16
        np.testing.assert_allclose(self.b.variance(t,k),w/(.25*g),rtol=1e-10)
        # On this flat first smile, interior short-end variance is exactly .04.
        np.testing.assert_allclose(self.b.variance(.1,np.array([-.1,0.,.1])),.04,atol=1e-13)

    def test_supported_local_variance_unchanged_and_permission_not_widened(self):
        before=deepcopy(self.ev.permission)
        for t in (.3,.4,.6,.7):
            k=np.array([-.1,0.,.1])
            expected=[self.ev.evaluate(t,float(x))['local_variance'] for x in k]
            np.testing.assert_allclose(self.b.variance(t,k),expected,rtol=1e-10)
        self.assertEqual(before,self.ev.permission)
        self.assertEqual(self.ev.evaluate(.1,0.)['status'],'unavailable')
        self.assertEqual(self.ev.evaluate(.4,.6)['status'],'unavailable')

    def test_alternatives_preserve_first_smile_but_change_assumptions(self):
        k=np.array([-.5,-.1,0.,.1,.5])
        alt=pricing_boundary(self.ev,self.cfg,'short_end_alternative')
        np.testing.assert_allclose(alt.node(0,k)[0],self.b.node(0,k)[0])
        self.assertFalse(np.allclose(alt.variance(.1,k),self.b.variance(.1,k)))
        wing=pricing_boundary(self.ev,self.cfg,'wing_alternative')
        for j in range(3):
            np.testing.assert_allclose(wing.node(j,k[1:4])[0],self.b.node(j,k[1:4])[0])
        self.assertFalse(np.allclose(wing.node(2,k)[0],self.b.node(2,k)[0]))

    def test_cell_average_preserves_constant_variance_and_resolves_joins(self):
        k=np.linspace(-.3,.3,121);h=k[1]-k[0]
        with patch.object(self.b,'variance',side_effect=lambda t,q: np.full_like(q,.04)):
            np.testing.assert_allclose(self.b.cell_variance(.1,k,h),.04,rtol=1e-13)
        self.assertTrue(np.all(self.b._cell_cache['weights']>0))
        np.testing.assert_allclose(self.b._cell_cache['mass'],1.,atol=1e-12)

    def test_completion_pde_recovers_nodes_and_refines(self):
        errors=[]
        for n,steps in ((3201,365),(6401,730)):
            mesh,prices=solve_calls(self.b,self.m.times,bounds=[-3.,2.],points=n,steps_per_year=steps)
            ks=np.array([-.1,0.,.1])
            errors.append(max(float(np.max(abs(np.interp(ks,mesh,prices[t])-black_prices(1,np.exp(ks),t,1,.2,True)))) for t in self.m.times))
        self.assertLess(errors[1],errors[0]);self.assertLess(errors[1],2e-5)

    def test_unsupported_interpolator_and_infeasible_nested_domains_block(self):
        other=LocalVolEvaluator(flat_model(),permission(flat_model()),.05,1e-9)
        with self.assertRaisesRegex(ValueError,'raw_svi'):pricing_boundary(other,self.cfg)
        m=model();m.k_min[2]=-.1
        ev=LocalVolEvaluator(m,permission(m),.05,1e-9)
        with self.assertRaisesRegex(ValueError,'node_support'):pricing_boundary(ev,self.cfg)

    def test_crossed_calendar_is_blocked_without_changing_prices(self):
        m=model();m.parameters[1][0]=.005
        ev=LocalVolEvaluator(m,permission(m),.05,1e-9)
        with self.assertRaisesRegex(ValueError,'calendar'):pricing_boundary(ev,self.cfg)

    def test_invalid_times_states_and_config(self):
        for t in (-1.,1.,float('nan')):
            with self.assertRaises(ValueError):self.b.variance(t,np.array([0.]))
        with self.assertRaises(ValueError):self.b.variance(.1,np.array([float('nan')]))
        for change in ({'wing_tail_fraction':0},{'wing_tail_fraction':True},
                       {'wing_tail_fraction_alternative':float('nan')},
                       {'short_end_power_alternative':1},{'boundary_audit_points':20},
                       {'short_end':'first_smile_total_variance'}):
            with self.subTest(change=change),self.assertRaises(ValueError):pricing_config(change)

    def test_integrated_evidence_live_and_historical_offline(self):
        with TemporaryDirectory() as tmp:
            surface=fixture(Path(tmp));surface.models['group']=self.m
            cfg=config(space_points=3201)
            for route in ('live','historical'):
                surface.tables['cohort_status'][0]['route']=route
                if route=='historical':
                    for q in surface.tables['repricing']:
                        for key in ('bid','ask','bid_iv','ask_iv'):q[key]=None
                with patch('socket.socket.connect',side_effect=AssertionError('no downloads')):
                    out=validate_repricing(surface,{'group':self.ev},cfg)
                self.assertEqual(out['pricing_summary'][0]['status'],'validated_with_boundary_assumptions')
                self.assertEqual(len(out['boundary_audit']),3)
                self.assertEqual(len(out['boundary_calibration']),18)
                self.assertFalse(out['pricing_summary'][0]['full_simulation_ready'])
                if route=='historical':self.assertTrue(all(r['inside_spread'] is None for r in out['option_repricing']))


if __name__=='__main__':unittest.main()
