"""Date/state preparation only: synthetic curves, no APIs or Monte Carlo."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import math
import unittest
import numpy as np
from test_local_vol_boundary import model, config
from test_local_vol_pricing import fixture
from test_local_volatility import permission, settings
from src.local_volatility import (LocalVolEvaluator, DateSpotLocalVol,
    _prepare_state_evaluators, convert_surface)

YEAR=365*86400.


def inputs(root):
    surface=fixture(root);surface.models['group']=model()
    origin=datetime(2031,2,3,12,tzinfo=timezone.utc)
    for r in surface.tables['reference_inputs']:
        r['expiry_timestamp']=(origin+timedelta(seconds=r['years']*YEAR)).isoformat()
    return surface,origin


class StateMappingTests(unittest.TestCase):
    def setUp(self):
        self.surface,self.origin=inputs(Path('.'))
        m=self.surface.models['group']
        self.ev=LocalVolEvaluator(m,permission(m),.05,1e-9)
        self.cfg=config()
        self.adapter=DateSpotLocalVol(self.ev,self.surface.tables['reference_inputs'],'2031-02-03',self.cfg)

    def stamp(self,t):return self.origin+timedelta(seconds=t*YEAR)

    def test_date_spot_mapping_matches_strict_coefficients(self):
        t=.4;f,d=self.adapter.curves.at(t)
        ks=np.array([-.1,0.,.1]);s=f*np.exp(ks)
        result=self.adapter.evaluate(self.stamp(t),s)
        self.assertEqual(result['status'],'available')
        np.testing.assert_allclose(result['k'],ks,atol=1e-15)
        np.testing.assert_allclose(result['local_variance'],[self.ev.evaluate(t,k)['local_variance'] for k in ks])
        self.assertFalse(any(result['boundary_assumption_used']))
        self.assertEqual(result['discount_factor'],d)
        self.assertFalse(result['full_simulation_ready'])
        json.dumps(result,allow_nan=False)

    def test_origin_short_end_wings_knots_require_opt_in(self):
        for t,k in ((0.,0.),(.1,0.),(.4,.3),(.25,0.),(.5,0.),(.75,0.)):
            f,_=self.adapter.curves.at(t);spot=f*math.exp(k)
            self.assertEqual(self.adapter.evaluate(self.stamp(t),spot)['status'],'unavailable')
            r=self.adapter.evaluate(self.stamp(t),spot,allow_boundary_assumptions=True)
            self.assertEqual(r['status'],'available',r)
            self.assertTrue(r['boundary_assumption_used'])
            self.assertAlmostEqual(r['local_variance'],float(self.adapter.boundary.variance(t,np.array(k))),places=10)
        at_zero=self.adapter.evaluate(self.origin,100.,allow_boundary_assumptions=True)
        self.assertTrue(at_zero['short_end_assumption'])
        self.assertAlmostEqual(at_zero['local_volatility'],.2)

    def test_scalar_vector_and_permutation_consistency_with_flags(self):
        t=.1;f,_=self.adapter.curves.at(t);s=f*np.exp([-.3,0.,.3])
        batch=self.adapter.evaluate(self.stamp(t),s,allow_boundary_assumptions=True)
        singles=[self.adapter.evaluate(self.stamp(t),float(x),allow_boundary_assumptions=True) for x in s]
        np.testing.assert_allclose(batch['local_variance'],[r['local_variance'] for r in singles],rtol=1e-12)
        np.testing.assert_equal(batch['wing_assumption'],[True,False,True])
        reverse=self.adapter.evaluate(self.stamp(t),s[::-1],allow_boundary_assumptions=True)
        np.testing.assert_allclose(reverse['local_variance'],batch['local_variance'][::-1])
        shaped=self.adapter.evaluate(self.stamp(t),s.reshape(3,1),allow_boundary_assumptions=True)
        np.testing.assert_equal(np.shape(shaped['local_variance']),(3,1))

    def test_timestamps_no_midnight_guess_and_no_rebasing(self):
        t=.4;stamp=self.stamp(t)
        other=stamp.astimezone(timezone(timedelta(hours=2)))
        self.assertEqual(self.adapter.years(stamp),self.adapter.years(other.isoformat()))
        for invalid in ('2031-02-03','03/02/2031',self.origin.replace(tzinfo=None),self.origin.date(),None):
            self.assertEqual(self.adapter.evaluate(invalid,100.)['status'],'unavailable')
        win=self.adapter.validate_window(self.stamp(.1),self.stamp(.6))
        self.assertAlmostEqual(win['start_years'],.1)
        self.assertAlmostEqual(win['end_years'],.6)
        self.assertEqual(len(win['knot_timestamps']),2)
        # A later start still queries the original reference surface.
        self.assertEqual(self.adapter.evaluate(self.stamp(.4),100.)['years'],.4)

    def test_invalid_states_and_horizon_are_not_clamped(self):
        for s in (0.,-1.,True,[True,100.],'100',[100.,float('nan')],float('inf'),[],{'spot':100}):
            self.assertEqual(self.adapter.evaluate(self.stamp(.4),s,allow_boundary_assumptions=True)['status'],'unavailable')
        for stamp in (self.origin-timedelta(microseconds=1),self.stamp(.75)+timedelta(microseconds=1)):
            self.assertEqual(self.adapter.evaluate(stamp,100.,allow_boundary_assumptions=True)['status'],'unavailable')
        for k in (-3.01,2.01):
            f,_=self.adapter.curves.at(.4)
            r=self.adapter.evaluate(self.stamp(.4),f*math.exp(k),allow_boundary_assumptions=True)
            self.assertIn('no_clipping',r['reason'])
        for a,b in ((.4,.4),(.5,.4),(-.1,.4),(.1,.8)):
            with self.assertRaises(ValueError):self.adapter.validate_window(self.stamp(a),self.stamp(b))
        self.assertEqual(self.adapter.evaluate(self.stamp(.4),100.,allow_boundary_assumptions=1)['status'],'unavailable')

    def test_timestamp_and_curve_inconsistencies_block_preparation(self):
        for key,value in (('reference_timestamp','2031-02-03T12:00:00'),
                          ('expiry_timestamp','2031-05-05T00:00:00Z'),
                          ('curve_date','2031-02-02')):
            refs=deepcopy(self.surface.tables['reference_inputs']);refs[0][key]=value
            with self.assertRaises(ValueError):DateSpotLocalVol(self.ev,refs,'2031-02-03',self.cfg)
        refs=deepcopy(self.surface.tables['reference_inputs']);del refs[0]['expiry_timestamp']
        with self.assertRaises(KeyError):DateSpotLocalVol(self.ev,refs,'2031-02-03',self.cfg)

    def test_valuation_calendar_date_is_not_reclassified_as_utc_date(self):
        # 20:00 New York on the valuation day is 01:00 UTC the next day.
        # Curves keep the upstream valuation date; elapsed time uses instants.
        refs=deepcopy(self.surface.tables['reference_inputs'])
        reference=datetime(2031,2,3,20,tzinfo=timezone(timedelta(hours=-5)))
        for r in refs:
            r['reference_timestamp']=reference.isoformat()
            r['expiry_timestamp']=(reference+timedelta(seconds=r['years']*YEAR)).isoformat()
        adapter=DateSpotLocalVol(self.ev,refs,'2031-02-03',self.cfg)
        self.assertEqual(adapter.reference.date().isoformat(),'2031-02-04')
        self.assertEqual(adapter.years(reference),0.)
        self.assertEqual(adapter.evaluate(reference,100.,allow_boundary_assumptions=True)['status'],'available')

    def test_strict_failure_cannot_be_bypassed_with_completion(self):
        with patch.object(self.ev,'evaluate',return_value=dict(status='unavailable',reason='bad_density')):
            result=self.adapter.evaluate(self.stamp(.4),100.,allow_boundary_assumptions=True)
            self.assertEqual(result['status'],'unavailable')
            self.assertIn('bad_density',result['reason'])

    def test_failed_pricing_or_missing_data_never_exposes_adapter(self):
        tables=dict(pricing_summary=[dict(group_id='group',status='review_required')]);store={}
        found=_prepare_state_evaluators(self.surface,{'group':self.ev},tables,self.cfg,store)
        self.assertEqual(found,{})
        self.assertEqual(tables['state_mapping'][0]['status'],'unavailable')
        self.assertFalse(tables['state_mapping_checks'])

    def test_plot_never_bridges_unavailable_strike_column(self):
        from types import SimpleNamespace
        from src.surface_view import local_patches
        rows=[dict(group_id='g',interval=0,years=t,k=k,status='available',local_volatility=.2)
              for t in (.3,.4,.45) for k in (-.2,-.1,0.,.1,.2)]
        for r in rows:
            if r['k']==0:r['status']='unavailable';del r['local_volatility']
        result=SimpleNamespace(tables=dict(local_vol_grid=rows,
            cohorts=[dict(group_id='g',status='review_required')]))
        patches=local_patches(result,'g')
        self.assertTrue(patches)
        for p in patches:
            self.assertIn('Diagnostic',p['label'])
            x=[q[1] for row in p['rows'] for q in row]
            self.assertTrue(max(x)<1 or min(x)>1)
        for r in rows:r['status']='unavailable'
        self.assertEqual(local_patches(result,'g'),[])

    def test_integrated_live_historical_saved_mapping_and_reuse(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);cfg=settings();cfg['pricing']=config(space_points=3201)
            path=root/'local.json';path.write_text(json.dumps(cfg),encoding='utf-8')
            for route in ('live','historical'):
                surface,_=inputs(root);surface.tables['cohort_status'][0]['route']=route
                if route=='historical':
                    for q in surface.tables['repricing']:
                        for k in ('bid','ask','bid_iv','ask_iv'):q[k]=None
                with patch('socket.socket.connect',side_effect=AssertionError('no network')):
                    result=convert_surface(surface,path)
                self.assertIn('group',result.state_evaluators)
                self.assertEqual(result.tables['state_mapping'][0]['status'],'prepared_with_boundary_assumptions')
                self.assertTrue(all(r['status']=='passed' for r in result.tables['state_mapping_checks']))
                self.assertIn('date_spot_mapping',result.evaluator_inputs['group'])
                folder=result.save();before=(folder/'manifest.json').stat().st_mtime_ns
                self.assertEqual(result.save(),folder)
                self.assertEqual((folder/'manifest.json').stat().st_mtime_ns,before)
                self.assertIn('Date and underlying mapping',(folder/'report.html').read_text(encoding='utf-8'))
                self.assertFalse(result.manifest['summary']['full_simulation_ready'])


if __name__=='__main__':unittest.main()
