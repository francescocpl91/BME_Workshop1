"""Independent practical-fit policies: synthetic quotes, no APIs/user inputs.

Tests distinguish premium/spread/IV units, missing evidence, legacy policies
and the workflow gate. Fitting, input selection and arbitrage are not relaxed.
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import json
import tempfile
import unittest
import numpy as np

from test_svi_surface import settings
from test_volatility_surface import fixture
from src.option_math import implied_volatility
from src.surface_checks import quote_quality, summarize_quality
from src.volatility_surface import load_surface_config, fit_calibrated


def practical_policy():
    cfg=settings();cfg['schema_version']=3
    cfg['quality'].pop('minimum_inside_spread_fraction')
    cfg['quality'].update(minimum_practical_coverage=.8,maximum_relative_outside_error=.01,
                          maximum_outside_half_spreads=.5,maximum_outside_iv_error=.0025)
    return cfg


def quote(**changes):
    p=dict(route='live',option_price=100.,bid=99.,ask=101.,iv=.2,bid_iv=.19,ask_iv=.21)
    p.update(changes)
    return p


class PracticalQualityTests(unittest.TestCase):
    def setUp(self):self.policy=practical_policy()['quality']

    def test_inside_quote_and_exact_practical_boundaries(self):
        inside=quote_quality(quote(),101.,.21,self.policy)
        self.assertTrue(inside['inside_spread']);self.assertTrue(inside['practical_pass'])
        boundary=quote_quality(quote(),101.5,.2125,self.policy)
        self.assertFalse(boundary['inside_spread']);self.assertTrue(boundary['practical_pass'])
        self.assertAlmostEqual(boundary['outside_spread_price_error'],.5)
        self.assertAlmostEqual(boundary['relative_outside_spread_error'],.005)
        self.assertAlmostEqual(boundary['outside_half_spreads'],.5)
        self.assertAlmostEqual(boundary['outside_iv_error'],.0025)

    def test_cheap_option_is_protected_by_premium_limit(self):
        p=quote(bid=1.9,ask=2.1,option_price=2.)
        row=quote_quality(p,2.125,.2,self.policy)
        self.assertAlmostEqual(row['relative_outside_spread_error'],.0125)
        self.assertFalse(row['practical_price_pass'])
        self.assertTrue(row['practical_spread_pass']);self.assertTrue(row['practical_iv_pass'])
        self.assertFalse(row['practical_pass'])

    def test_spread_condition_is_independently_required(self):
        row=quote_quality(quote(),101.6,.2,self.policy)
        self.assertTrue(row['practical_price_pass']);self.assertTrue(row['practical_iv_pass'])
        self.assertFalse(row['practical_spread_pass']);self.assertFalse(row['practical_pass'])

    def test_expensive_option_is_protected_by_iv_limit(self):
        row=quote_quality(quote(),100.,.213,self.policy)
        self.assertTrue(row['practical_price_pass']);self.assertTrue(row['practical_spread_pass'])
        self.assertFalse(row['practical_iv_pass']);self.assertFalse(row['practical_pass'])

    def test_boundary_uses_original_interval_not_midpoint(self):
        row=quote_quality(quote(),99.,.19,self.policy)
        self.assertAlmostEqual(row['relative_observed_price_error'],.01)
        self.assertEqual(row['relative_outside_spread_error'],0.)
        self.assertAlmostEqual(row['absolute_iv_error'],.01)
        self.assertEqual(row['outside_iv_error'],0.)
        self.assertTrue(row['practical_pass'])

    def test_invalid_iv_bounds_are_unassessed_not_zero(self):
        for bid,ask in [(None,.21),(float('nan'),.21),(0.,.21),(.22,.21),(.19,float('inf'))]:
            with self.subTest(bounds=(bid,ask)):
                row=quote_quality(quote(bid_iv=bid,ask_iv=ask),100.,.2,self.policy)
                self.assertIsNone(row['outside_iv_error']);self.assertIsNone(row['practical_pass'])
                self.assertEqual(row['practical_status'],'unassessed')
                json.dumps(row,allow_nan=False)

    def test_missing_bounds_remain_in_coverage_denominator(self):
        points=[quote() for _ in range(8)]+[quote(bid_iv=None) for _ in range(2)]
        rows=[quote_quality(p,100.,.2,self.policy) for p in points]
        result=summarize_quality(points,rows,self.policy)
        self.assertEqual(result['practical_coverage_fraction'],.8)
        self.assertEqual(result['practical_unassessed_observations'],2)
        self.assertTrue(result['practical_coverage_pass'])
        points[0]['bid_iv']=None;rows[0]=quote_quality(points[0],100.,.2,self.policy)
        self.assertFalse(summarize_quality(points,rows,self.policy)['practical_coverage_pass'])

    def test_historical_and_zero_spread_do_not_manufacture_intervals(self):
        p=quote(route='historical');p.pop('bid');p.pop('ask')
        row=quote_quality(p,101.,.21,self.policy)
        self.assertIsNone(row['inside_spread']);self.assertIsNone(row['outside_iv_error'])
        self.assertEqual(row['practical_status'],'not_applicable_historical')
        self.assertIsNone(summarize_quality([p],[row],self.policy)['practical_coverage_fraction'])
        row=quote_quality(quote(bid=100.,ask=100.),100.,.2,self.policy)
        self.assertIsNone(row['outside_half_spreads']);self.assertIsNone(row['practical_pass'])

    def test_price_unit_scaling_and_inputs_are_preserved(self):
        p=quote();before=deepcopy(p);first=quote_quality(p,101.4,.211,self.policy)
        scaled={**p,**{key:10*p[key] for key in ('bid','ask','option_price')}}
        second=quote_quality(scaled,1014.,.211,self.policy)
        for key in ('relative_outside_spread_error','outside_half_spreads','outside_iv_error'):
            self.assertAlmostEqual(first[key],second[key])
        self.assertEqual(first['practical_pass'],second['practical_pass']);self.assertEqual(p,before)

    def test_quantiles_report_available_counts_and_decimal_units(self):
        points=[quote(),quote(),quote(bid_iv=None)]
        rows=[quote_quality(p,price,.2125,self.policy) for p,price in zip(points,[100.,101.5,102.])]
        result=summarize_quality(points,rows,self.policy)
        self.assertAlmostEqual(result['p95_relative_outside_spread_error'],.0095)
        self.assertEqual(result['outside_iv_error_observations'],2)
        self.assertAlmostEqual(result['maximum_outside_iv_error'],.0025)
        json.dumps(result,allow_nan=False)


class PracticalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'policy.json'
        self.cfg=practical_policy();self.cfg['fit']['model']='ssvi'
        self.cal=fixture(self.root)
        for p in self.cal.tables['option_iv']:
            for side in ('bid','ask'):
                p[side+'_iv']=implied_volatility(p[side],p['forward'],p['strike'],p['time_to_expiry'],p['discount_factor'])

    def fit(self):
        self.path.write_text(json.dumps(self.cfg),encoding='utf-8')
        with patch('socket.socket.connect',side_effect=AssertionError('No network')):
            return fit_calibrated(self.cal,self.path)

    def test_actual_synthetic_fit_export_and_historical_regression(self):
        self.cfg['plots']=True;before=deepcopy(self.cal.tables);result=self.fit()
        self.assertTrue(result.tables['cohort_status'][0]['market_fit_pass'])
        self.assertTrue(all(r['practical_coverage_pass'] for r in result.tables['expiry_metrics']))
        target=result.save();self.assertEqual(target,result.save())
        self.assertEqual(len(list(target.glob('*.png'))),3)
        report=(target/'report.md').read_text(encoding='utf-8')
        self.assertIn('Price fit by expiry',report)
        self.assertIn('Practical coverage | Strict bid/ask coverage',report)
        with (target/'repricing.csv').open(encoding='utf-8',newline='') as f:
            self.assertIn('practical_pass',next(csv.DictReader(f)))
        self.assertEqual(before,self.cal.tables)
        self.cfg['plots']=False
        for p in self.cal.tables['option_iv']:
            p.update(route='historical',price_type='MIDPOINT_bar_close')
            for key in ('bid','ask','bid_iv','ask_iv'):p.pop(key)
        historical=self.fit()
        self.assertTrue(historical.tables['cohort_status'][0]['market_fit_pass'])
        for row in historical.tables['expiry_metrics']:
            self.assertIsNone(row['practical_coverage_fraction'])
            self.assertEqual(row['coverage_policy'],'historical_rmse_only')

    def test_practical_gate_replaces_strict_but_not_rmse_or_arbitrage(self):
        # Isolate acceptance orchestration from fitting mathematics. A controlled
        # pricing probe misses every ask slightly but satisfies all three limits.
        def probe(model,points):
            return np.array([p['option_price']*1.006 for p in points]),np.array([p['iv'] for p in points])
        with patch('src.volatility_surface.fitted_prices',side_effect=probe):result=self.fit()
        self.assertTrue(result.tables['cohort_status'][0]['market_fit_pass'])
        for row in result.tables['expiry_metrics']:
            self.assertEqual(row['inside_spread_fraction'],0.)
            self.assertEqual(row['practical_coverage_fraction'],1.)
            self.assertIsNone(row['spread_coverage_pass'])
        self.cfg['quality']['maximum_scaled_price_rmse']=.01
        with patch('src.volatility_surface.fitted_prices',side_effect=probe):result=self.fit()
        self.assertFalse(result.tables['cohort_status'][0]['market_fit_pass'])
        self.cfg['quality']['maximum_scaled_price_rmse']=2.
        with patch('src.volatility_surface.validate_grid',return_value=([],dict(numerical_arbitrage_pass=False,derivative_check_pass=True))):
            result=self.fit()
        self.assertEqual(result.tables['cohort_status'][0]['status'],'diagnostic_fit')
        self.assertFalse(result.tables['cohort_status'][0]['numerical_checks_pass'])

    def test_missing_intervals_cannot_certify_fitted_cohort(self):
        for p in self.cal.tables['option_iv']:p.pop('bid_iv')
        result=self.fit();self.assertTrue(result.models)
        self.assertFalse(result.tables['cohort_status'][0]['market_fit_pass'])
        self.assertTrue(all(m['practical_coverage_fraction']==0 for m in result.tables['expiry_metrics']))
        self.assertTrue(all(r['practical_status']=='unassessed' for r in result.tables['repricing']))

    def test_schema3_validation_and_legacy_acceptance(self):
        for key,value in [('minimum_practical_coverage',1.01),('maximum_relative_outside_error',1),
                          ('maximum_outside_half_spreads',True),('maximum_outside_iv_error',float('nan')),
                          ('maximum_outside_iv_error',25),('maximum_relative_outside_error',-.1)]:
            cfg=practical_policy();cfg['quality'][key]=value;self.path.write_text(json.dumps(cfg))
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):load_surface_config(self.path)
        self.cfg=settings();self.cfg['fit']['model']='ssvi'
        def probe(model,points):
            return np.array([p['option_price']*1.006 for p in points]),np.array([p['iv'] for p in points])
        with patch('src.volatility_surface.fitted_prices',side_effect=probe):result=self.fit()
        self.assertFalse(result.tables['cohort_status'][0]['market_fit_pass'])
        self.assertTrue(all(r['practical_coverage_fraction'] is None for r in result.tables['expiry_metrics']))


if __name__=='__main__':unittest.main()
