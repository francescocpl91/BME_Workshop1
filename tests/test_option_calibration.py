"""Offline calibration regressions with known prices and isolated temporary files.

Synthetic prices test recoverability independently of IBKR availability. These
fixtures intentionally include historical bars with no provider IV or Greeks.
Tests neither download data nor write to the user's archived observations.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import QuantLib as ql

from src.ois_bootstrap import DiscountCurve, bootstrap_ois
from src.option_math import black_price, black_vega, equity_forward, implied_volatility
from src.option_calibration import (load_calibration_config, build_curves,
    calculate_option, _historical_selection, _curve_time, _expiry,
    calibrate_processed, load_processed_inputs)

ROOT = Path(__file__).resolve().parents[1]


def settings():
    # Fixed test fixture: changing the user's valuation date or thresholds in
    # the real JSON must not change what these regression tests exercise.
    convention = dict(status='assumed', description='synthetic test conventions',
        spot_lag_business_days=2,payment_lag_business_days=2,
        fixed_payment_frequency='Annual',day_count='ACT/360',
        calendar='overnight_index_calendar',business_day_convention='ModifiedFollowing',
        payment_convention='Following',end_of_month=False)
    return dict(schema_version=1,valuation_date='2026-10-20',
        valuation_timezone='America/New_York',curve_conventions={'USD':deepcopy(convention),'EUR':deepcopy(convention)},
        live=dict(maximum_model_age_seconds=30,maximum_relative_spread=.1,capture_span_warning_minutes=5,allow_broker_dividend_pv=True),
        historical=dict(price_priority=['MIDPOINT','TRADES'],cutoff_local_time='16:00:00',maximum_bar_age_minutes=5,
            allow_daily_close_approximation=True,underlying_provider_priority=['yahoo','ibkr']),
        expiry=dict(timezone='America/New_York',times_by_settlement={'AM':'09:30:00','PM':'16:00:00'},instrument_overrides={}),
        solver=dict(maximum_volatility=5,minimum_time_value_fraction_of_spot=1e-6,minimum_vega_fraction_of_spot=1e-5),
        manual_carry_inputs=[],instructions={})


def policy_json(cfg):
    """Execution dates are internal context, never user calibration settings."""
    return json.dumps({k:v for k,v in cfg.items() if k != 'valuation_date'})


def flat_curve(observed='2026-10-20'):
    return DiscountCurve(observed,'USD','SOFR',(0.0,2.0),(1.0,math.exp(-.04*2)),{'status':'assumed'})


def contract():
    # Use an arbitrary underlying name: the engine must not depend on SPX.
    return dict(underlying='TEST_INDEX',expiration='2027-10-20',strike=100,
                option_type='call',exercise_style='european',settlement='cash',settlement_time='PM',multiplier=100)


def live_fixture():
    cfg=settings(); curve=flat_curve(); c=contract()
    stamp=datetime.fromisoformat('2026-10-20T16:00:00-04:00')
    expiry=_expiry(c,cfg)
    tau=(expiry.astimezone(timezone.utc)-stamp.astimezone(timezone.utc)).total_seconds()/(365*86400)
    discount=curve.interval_discount(_curve_time(stamp,curve,cfg['valuation_timezone']),_curve_time(expiry,curve,cfg['valuation_timezone']))
    f=equity_forward(100,1,discount)
    price=black_price(f,100,tau,discount,.25)
    row=dict(instrument_id='TEST_CALL',provider='ibkr',currency='USD',eligible=True,quote_eligible=True,
             market_data_type=1,timestamp=stamp.isoformat(),capture_timestamp_utc=stamp.isoformat(),
             model_received_at_utc=stamp.isoformat(),bid=price-.02,ask=price+.02,
             **{'modelGreeks.undPrice':100,'modelGreeks.pvDividend':1})
    return row,c,curve,cfg


def historical_fixture():
    row,c,curve,cfg=live_fixture()
    price=(row['bid']+row['ask'])/2
    h=dict(instrument_id='TEST_CALL',provider='ibkr',currency='USD',date=cfg['valuation_date'],
           timestamp='',bar_size='1 day',option_dataset='MIDPOINT',close=price,eligible=True,contract=c)
    prices=[dict(instrument_id='TEST_INDEX',provider='yahoo',currency='USD',date=cfg['valuation_date'],
                 close=100,eligible=True,price_basis='index_level',calendar='XNYS')]
    cfg['manual_carry_inputs']=[dict(underlying='TEST_INDEX',currency='USD',valuation_date=cfg['valuation_date'],
                                    expiration=c['expiration'],source='synthetic dividend forecast',dividend_pv=1)]
    return h,c,curve,cfg,prices


class OptionMathTests(unittest.TestCase):
    def test_calls_and_puts_match_independent_quantlib_and_invert(self):
        for f,k,t,v in [(100,100,1,.2),(110,90,.3,.35),(90,110,2,.5),(100,100,.01,.1)]:
            for right,qlright in [('call',ql.Option.Call),('put',ql.Option.Put)]:
                with self.subTest(f=f,k=k,t=t,right=right):
                    p=black_price(f,k,t,.97,v,right)
                    self.assertAlmostEqual(p,ql.blackFormula(qlright,k,f,v*math.sqrt(t),.97),places=11)
                    self.assertAlmostEqual(implied_volatility(p,f,k,t,.97,right),v,places=9)

    def test_boolean_inputs_are_not_numbers(self):
        with self.assertRaises(ValueError):equity_forward(100,True,.95)
        with self.assertRaises(ValueError):black_price(100,100,1,.95,True)

    def test_vega_agrees_with_price_derivative(self):
        derivative=(black_price(103,100,.7,.98,.2+1e-5)-black_price(103,100,.7,.98,.2-1e-5))/2e-5
        self.assertAlmostEqual(black_vega(103,100,.7,.98,.2),derivative,places=6)

    def test_bounds_are_rejected_not_clipped(self):
        for p in [0,9.5,95,100,float('nan')]:
            with self.subTest(price=p),self.assertRaises(ValueError):
                implied_volatility(p,100,90,1,.95)
        with self.assertRaisesRegex(ValueError,'solver_limit'):
            implied_volatility(black_price(100,100,1,1,2),100,100,1,1,maximum=1)

    def test_forward_dividend_units_and_invalid_values(self):
        self.assertEqual(equity_forward(6000,60,.96),6187.5)
        for pv in [-1,100,float('nan')]:
            with self.assertRaises(ValueError):equity_forward(100,pv,.95)


class OisBootstrapTests(unittest.TestCase):
    def quote_input(self,rate=4):
        return dict(as_of_date='2026-10-20',currency='USD',overnight_index='SOFR',
                    quote_type='spot_starting_ois_par_swap_rate',rate_unit='percent_per_annum',
                    points=[dict(tenor='1Y',rate_percent=rate)])

    def test_one_coupon_discount_matches_direct_formula(self):
        conventions=deepcopy(settings()['curve_conventions']['USD'])
        conventions.update(spot_lag_business_days=0,payment_lag_business_days=0)
        curve,nodes,repricing=bootstrap_ois(self.quote_input(),conventions)
        days=(date.fromisoformat(nodes[-1]['date'])-date(2026,10,20)).days
        self.assertAlmostEqual(curve.discounts[-1],1/(1+.04*days/360),places=10)
        self.assertLess(abs(repricing[0]['error_basis_points']),1e-5)

    def test_negative_rates_restore_global_date_and_forbid_extrapolation(self):
        before=ql.Settings.instance().evaluationDate
        curve,_,_=bootstrap_ois(self.quote_input(-1),settings()['curve_conventions']['USD'])
        self.assertGreater(curve.discounts[-1],1)
        self.assertEqual(ql.Settings.instance().evaluationDate,before)
        for t in [-.01,curve.times[-1]+.01,float('nan')]:
            with self.assertRaises(ValueError):curve.discount(t)

    def test_duplicate_quotes_and_unsupported_conventions_fail(self):
        inp=self.quote_input(); inp['points']*=2
        with self.assertRaisesRegex(ValueError,'duplicate'):
            bootstrap_ois(inp,settings()['curve_conventions']['USD'])
        cfg=settings()['curve_conventions']['USD']; cfg['day_count']='ACT/365'
        with self.assertRaisesRegex(ValueError,'convention'):
            bootstrap_ois(self.quote_input(),cfg)

    def test_date_mismatch_never_calls_bootstrap(self):
        cfg=settings()
        rows=[dict(currency='USD',date='2026-10-19')]
        with patch('src.option_calibration.bootstrap_ois',side_effect=AssertionError('must not bootstrap')):
            curves,_,_,statuses=build_curves(rows,cfg)
        self.assertFalse(curves)
        self.assertEqual(statuses[0]['reason'],'curve_date_mismatch')


class OptionAdapterTests(unittest.TestCase):
    def test_live_iv_without_provider_iv_and_bid_ask_envelope(self):
        row,c,curve,cfg=live_fixture()
        r=calculate_option(row,'live',c,{'USD':curve},[],cfg)
        self.assertEqual(r['status'],'iv_calculated_with_assumptions',r)
        self.assertAlmostEqual(r['iv'],.25,places=9)
        self.assertLess(r['bid_iv'],r['iv']); self.assertGreater(r['ask_iv'],r['iv'])
        self.assertNotIn('provider_model_iv',r)

    def test_live_curve_date_mismatch_and_missing_carry(self):
        row,c,curve,cfg=live_fixture()
        r=calculate_option(row,'live',c,{'USD':flat_curve('2026-10-19')},[],cfg)
        self.assertEqual(r['reason'],'curve_date_mismatch')
        row.pop('modelGreeks.pvDividend')
        r=calculate_option(row,'live',c,{'USD':curve},[],cfg)
        self.assertEqual(r['reason'],'missing_date_matched_dividend_or_forward')

    def test_live_stale_model_american_expiry_and_currency(self):
        row,c,curve,cfg=live_fixture()
        for changes,expected in [({'model_received_at_utc':'2026-10-19T16:00:00-04:00'},'stale_or_future_paired_model'),
                                 ({'source_currency':'EUR'},'source_currency_mismatch')]:
            r=calculate_option({**row,**changes},'live',c,{'USD':curve},[],cfg)
            self.assertEqual(r['reason'],expected)
        r=calculate_option(row,'live',{**c,'exercise_style':'american'},{'USD':curve},[],cfg)
        self.assertEqual(r['reason'],'non_european_option_not_supported')
        r=calculate_option(row,'live',{**c,'expiration':'2026-10-19'},{'USD':curve},[],cfg)
        self.assertEqual(r['reason'],'expired_option')

    def test_historical_iv_without_greeks_with_dated_dividend(self):
        h,c,curve,cfg,prices=historical_fixture()
        r=calculate_option(h,'historical',c,{'USD':curve},prices,cfg)
        self.assertEqual(r['status'],'iv_calculated_with_assumptions',r)
        self.assertAlmostEqual(r['iv'],.25,places=9)
        self.assertNotIn('provider_model_iv',r)

    def test_historical_manual_forward_matches_dividend_result(self):
        h,c,curve,cfg,prices=historical_fixture()
        first=calculate_option(h,'historical',c,{'USD':curve},prices,cfg)
        entry=cfg['manual_carry_inputs'][0]; entry.pop('dividend_pv'); entry['forward']=first['forward']
        second=calculate_option(h,'historical',c,{'USD':curve},prices,cfg)
        self.assertAlmostEqual(first['iv'],second['iv'],places=12)

    def test_historical_missing_or_wrong_date_carry_never_assumes_zero(self):
        h,c,curve,cfg,prices=historical_fixture()
        for entries in [[],[{**cfg['manual_carry_inputs'][0],'valuation_date':'2026-10-19'}]]:
            cfg['manual_carry_inputs']=entries
            r=calculate_option(h,'historical',c,{'USD':curve},prices,cfg)
            self.assertEqual(r['reason'],'missing_date_matched_dividend_or_forward')

    def test_historical_adjusted_or_different_date_underlying_is_rejected(self):
        h,c,curve,cfg,prices=historical_fixture()
        for changes in [{'price_basis':'split_adjusted_close'},{'date':'2026-10-19'}]:
            r=calculate_option(h,'historical',c,{'USD':curve},[{**prices[0],**changes}],cfg)
            self.assertEqual(r['reason'],'missing_same_date_unadjusted_underlying')

    def test_historical_selection_rejects_stale_future_and_fake_spread(self):
        h,c,curve,cfg,prices=historical_fixture()
        h['bar_size']='1 min'
        rows=[{**h,'timestamp':'2026-10-20T16:01:00-04:00'},
              {**h,'timestamp':'2026-10-20T15:40:00-04:00'},
              {**h,'timestamp':'2026-10-20T16:00:00-04:00','option_dataset':'BID'},
              {**h,'timestamp':'2026-10-20T16:00:00-04:00','option_dataset':'ASK'}]
        self.assertEqual(list(_historical_selection(rows,cfg,prices,{}))[0][1],'historical_bar_outside_close_window')
        rows.append({**h,'timestamp':'2026-10-20T15:59:00-04:00'})
        selected,reason=list(_historical_selection(rows,cfg,prices,{}))[0]
        self.assertIsNone(reason); self.assertEqual(selected['timestamp'],'2026-10-20T15:59:00-04:00')

    def test_historical_daily_close_opt_out(self):
        h,c,curve,cfg,prices=historical_fixture()
        cfg['historical']['allow_daily_close_approximation']=False
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],'daily_underlying_close_approximation_disabled')


class OfflineIntegrationTests(unittest.TestCase):
    def make_archive(self,root):
        h,c,curve,cfg,prices=historical_fixture()
        processed=root/'processed'; processed.mkdir()
        archive=root/'archive'; archive.mkdir()
        resolved=json.dumps({'requests':[{'id':'TEST_CALL','contract':c}]}).encode()
        (archive/'resolved_inputs.json').write_bytes(resolved)
        files={}
        quote=dict(instrument_id='USD_SOFR_OIS',currency='USD',date='2026-10-20',eligible=True,
                   curve_id='USD_SOFR_OIS',overnight_index='SOFR',quote_type='spot_starting_ois_par_swap_rate',
                   rate_unit='percent_per_annum',tenor='2Y',rate_percent=4)
        for name,rows in dict(selected_option_quotes=[],historical_option_bars=[h],provider_prices=prices,ois_quotes=[quote]).items():
            stream=io.StringIO(newline=''); cols=list(rows[0]) if rows else ['instrument_id']
            writer=csv.DictWriter(stream,fieldnames=cols); writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in row.items()})
            raw=stream.getvalue().encode(); (processed/(name+'.csv')).write_bytes(raw)
            files[name]={'file':name+'.csv','sha256':hashlib.sha256(raw).hexdigest()}
        manifest=dict(complete=True,cob='2026-10-20',archive=str(archive),files=files,
                      archive_input_hashes={'resolved_inputs.json':hashlib.sha256(resolved).hexdigest()})
        (processed/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        config=root/'config.json'; config.write_text(policy_json(cfg),encoding='utf-8')
        return processed,config

    def test_full_historical_pipeline_no_network_preserves_inputs_and_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed,config=self.make_archive(Path(tmp))
            before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in Path(tmp).rglob('*') if p.is_file()}
            with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
                result=calibrate_processed(processed,config)
                self.assertEqual(result.manifest['summary']['iv_calculated'],1,result.manifest['summary'])
                destination=result.save()
                out_before={p:p.stat().st_mtime_ns for p in destination.iterdir()}
                self.assertEqual(result.save(),destination)
                self.assertEqual(out_before,{p:p.stat().st_mtime_ns for p in destination.iterdir()})
            for p,snapshot in before.items():
                self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),snapshot)
            (destination/'option_iv.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'checksum'):result.save()

    def test_live_archive_cob_can_precede_actual_observation_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed,config=self.make_archive(Path(tmp))
            row,c,curve,cfg=live_fixture()
            # Restore live-only fixtures without provider IV or historical bars.
            manifest=json.loads((processed/'manifest.json').read_text())
            manifest['cob']='2026-10-19'
            for name,rows in [('selected_option_quotes',[row]),('historical_option_bars',[])]:
                stream=io.StringIO(newline=''); cols=list(rows[0]) if rows else ['instrument_id']
                writer=csv.DictWriter(stream,fieldnames=cols); writer.writeheader(); writer.writerows(rows)
                raw=stream.getvalue().encode(); (processed/(name+'.csv')).write_bytes(raw)
                manifest['files'][name]['sha256']=hashlib.sha256(raw).hexdigest()
            (processed/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            config.write_text(policy_json(cfg),encoding='utf-8')
            result=calibrate_processed(processed,config)
            self.assertEqual(result.manifest['summary']['iv_calculated'],1,result.manifest['summary'])
            self.assertEqual(result.tables['option_iv'][0]['valuation_date'],'2026-10-20')
            self.assertEqual(result.manifest['archive_cob'],'2026-10-19')

    def test_input_checksum_mismatch_stops_before_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed,config=self.make_archive(Path(tmp))
            with (processed/'provider_prices.csv').open('a') as stream:stream.write('bad')
            with patch('src.option_calibration.bootstrap_ois',side_effect=AssertionError('must not run')):
                with self.assertRaisesRegex(ValueError,'checksum'):calibrate_processed(processed,config)

    def test_new_config_does_not_interfere_with_provider_directory_loading(self):
        from src.json_config import load_provider_configs
        loaded = load_provider_configs(ROOT/'config',cob_date=date(2026,10,19),history_start=date(2000,1,3))
        self.assertTrue(loaded['requests'])
        self.assertFalse((ROOT/'config'/'option_calibration.json').exists())

    def test_bad_config_duplicate_carry_and_nan_rejected(self):
        cfg=settings()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json'
            cfg['solver']['maximum_volatility']=float('nan')
            p.write_text(policy_json(cfg))
            with self.assertRaises(ValueError):load_calibration_config(p)
            _,_,_,cfg,_=historical_fixture(); cfg['manual_carry_inputs']*=2
            p.write_text(policy_json(cfg))
            with self.assertRaisesRegex(ValueError,'ambiguous'):load_calibration_config(p)


if __name__ == '__main__':
    unittest.main()
