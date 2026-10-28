"""Offline regressions for quote timing, resolved identity and automatic workflow.

No provider requests: integration fixtures and all outputs live in temporary
folders. Configurable dates and arbitrary instrument names guard against tying
the library to the October SPX example.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import csv
import hashlib
import io
import json
import tempfile
import unittest

import test_option_calibration as fixtures
from src.option_calibration import (calculate_option, _historical_selection,
    calibrate_processed, _contract)


class TimingAndIdentityTests(unittest.TestCase):
    def test_frozen_delayed_and_unknown_are_not_current_live(self):
        row,c,curve,cfg=fixtures.live_fixture()
        for mode in (None,2,3,4,99):
            with self.subTest(mode=mode):
                result=calculate_option({**row,'market_data_type':mode},'live',c,{'USD':curve},[],cfg)
                self.assertEqual(result['reason'],'non_live_or_unknown_market_data_type')
        result=calculate_option(row,'live',c,{'USD':curve},[],cfg)
        self.assertAlmostEqual(result['iv'],.25)

    def test_resolved_identity_conflicts_for_live_and_history(self):
        row,c,curve,cfg=fixtures.live_fixture()
        h,_,_,hcfg,prices=fixtures.historical_fixture()
        for key,value,reason in [('strike',120,'strike'),('right','P','option_type'),
                                 ('symbol','OTHER','underlying'),('multiplier','10','multiplier'),
                                 ('currency','EUR','currency'),('realExpirationDate','20271021','expiration')]:
            for route,source,policy in [('live',row,cfg),('historical',h,hcfg)]:
                with self.subTest(key=key,route=route):
                    source={**source,'resolved_contract':json.dumps({key:value})}
                    result=calculate_option(source,route,c,{'USD':curve},prices,policy)
                    self.assertEqual(result['reason'],'resolved_contract_mismatch: '+reason)

    def test_am_last_trade_day_is_not_legal_expiry(self):
        row,c,curve,cfg=fixtures.live_fixture()
        c['settlement_time']='AM'
        row['resolved_contract']=json.dumps({'strike':100,'right':'C','multiplier':'100',
            'symbol':'TEST_INDEX','lastTradeDateOrContractMonth':'20271019',
            'localSymbol':'TEST  271020C00100000'})
        row['contract_dates']=json.dumps({'realExpirationDate':'20271020'})
        result=calculate_option(row,'live',c,{'USD':curve},[],cfg)
        self.assertEqual(result['status'],'iv_calculated_with_assumptions')
        row['resolved_contract']=json.dumps({'localSymbol':'TEST  271021C00100000'})
        result=calculate_option(row,'live',c,{'USD':curve},[],cfg)
        self.assertEqual(result['reason'],'resolved_contract_mismatch: localSymbol')

    def test_flattened_contract_conflict_is_not_silently_overwritten(self):
        h,c,_,_,_=fixtures.historical_fixture()
        h['strike']=120
        with self.assertRaisesRegex(ValueError,'conflicting_configured_contract'):
            _contract(h,{})

    def test_bar_must_finish_before_close_in_selection_and_direct_helper(self):
        h,c,curve,cfg,prices=fixtures.historical_fixture()
        h.update(timestamp='2026-10-20T16:00:00-04:00',bar_size='1 min')
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],
                         'historical_bar_outside_close_window')
        self.assertEqual(calculate_option(h,'historical',c,{'USD':curve},prices,cfg)['reason'],
                         'historical_bar_outside_close_window')
        h['timestamp']='2026-10-20T15:59:00-04:00'
        selected,reason=list(_historical_selection([h],cfg,prices,{}))[0]
        self.assertIsNone(reason)
        result=calculate_option(selected,'historical',c,{'USD':curve},prices,cfg)
        self.assertEqual(result['bar_age_minutes'],0)
        self.assertEqual(result['bar_end_timestamp'],'2026-10-20T20:00:00+00:00')

    def test_multiminute_bar_end_and_missing_duration(self):
        h,c,curve,cfg,prices=fixtures.historical_fixture()
        h.update(timestamp='2026-10-20T15:55:00-04:00',bar_size='5 mins')
        self.assertIsNone(list(_historical_selection([h],cfg,prices,{}))[0][1])
        h['timestamp']='2026-10-20T15:59:00-04:00'
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],
                         'historical_bar_outside_close_window')
        h['bar_size']=''
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],
                         'missing_or_unsupported_option_bar_size')
        h['timestamp']=''
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],
                         'missing_or_unsupported_option_bar_size')

    def test_noon_cutoff_cannot_use_later_daily_underlying(self):
        h,c,curve,cfg,prices=fixtures.historical_fixture()
        cfg['historical']['cutoff_local_time']='12:00:00'
        h.update(timestamp='2026-10-20T11:59:00-04:00',bar_size='1 min')
        result=calculate_option(h,'historical',c,{'USD':curve},prices,cfg)
        self.assertEqual(result['reason'],'historical_cutoff_precedes_underlying_close')
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],result['reason'])

    def test_half_day_uses_13h_close_and_holidays_are_rejected(self):
        h,c,curve,cfg,prices=fixtures.historical_fixture()
        observed='2026-11-27'
        h.update(date=observed,timestamp=observed+'T12:59:00-05:00',bar_size='1 min')
        cfg['valuation_date']=observed
        cfg['manual_carry_inputs'][0]['valuation_date']=observed
        prices[0]['date']=observed
        curve=fixtures.flat_curve(observed)
        result=calculate_option(h,'historical',c,{'USD':curve},prices,cfg)
        self.assertEqual(result['status'],'iv_calculated_with_assumptions',result)
        self.assertEqual(result['valuation_timestamp'],'2026-11-27T18:00:00+00:00')
        h['timestamp']=observed+'T15:59:00-05:00'
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],
                         'historical_bar_outside_close_window')
        cfg['valuation_date']=h['date']=prices[0]['date']='2026-11-26'
        self.assertEqual(list(_historical_selection([h],cfg,prices,{}))[0][1],
                         'underlying_date_not_trading_session')

    def test_missing_calendar_is_not_assumed_us(self):
        h,c,curve,cfg,prices=fixtures.historical_fixture()
        del prices[0]['calendar']
        self.assertEqual(calculate_option(h,'historical',c,{'USD':curve},prices,cfg)['reason'],
                         'missing_or_conflicting_underlying_calendar')

    def test_inventory_to_cleaning_keeps_per_response_duration_and_identity(self):
        from test_preprocessing import InventoryTests
        from src.data_cleaning import _extract
        f=InventoryTests();f.setUp();self.addCleanup(f.doCleanups)
        c=fixtures.contract()
        one=f.bars([('2026-10-08T19:59:00+00:00',3)])
        one['request']['bar_size']='1 min'
        five=deepcopy(one);five['request']['bar_size']='5 mins'
        details={'method':'reqContractDetails','contract_details':[{
            'realExpirationDate':'20271020','contract':{'symbol':'TEST_INDEX','strike':100,'right':'C'}}]}
        f.add('ibkr','TEST_CALL',[details,one,five],operation='option_history',contract=c)
        f.finish()
        from src.preprocessing import inspect_archive
        inventory=inspect_archive(f.archive,include_observations=True)
        *_,bars=_extract(inventory,[])
        self.assertEqual([r['bar_size'] for r in bars],['1 min','5 mins'])
        self.assertEqual(bars[0]['resolved_contract']['strike'],100)
        self.assertEqual(bars[0]['contract_dates']['realExpirationDate'],'20271020')


def replace_table(folder,name,rows):
    """Replace only a temporary test table and update its integrity manifest."""
    stream=io.StringIO(newline='')
    columns=list(dict.fromkeys(k for r in rows for k in r)) or ['instrument_id']
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in row.items()})
    raw=stream.getvalue().encode()
    (folder/(name+'.csv')).write_bytes(raw)
    p=folder/'manifest.json';manifest=json.loads(p.read_text())
    manifest['files'][name]['sha256']=hashlib.sha256(raw).hexdigest()
    p.write_text(json.dumps(manifest),encoding='utf-8')


class AutomaticCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder,self.config=fixtures.OfflineIntegrationTests().make_archive(Path(self.temp.name))

    def test_current_folder_and_observed_dates_are_execution_context(self):
        with patch('socket.socket.connect',side_effect=AssertionError('No network')):
            result=calibrate_processed(self.folder,self.config)
        self.assertEqual(result.manifest['summary']['iv_calculated'],1,result.manifest['summary'])
        self.assertEqual(result.manifest['summary']['valuation_dates'],['2026-10-20'])
        self.assertEqual(result.manifest['identity']['source_folder'],str(self.folder.resolve()))
        self.assertNotIn('processed_directory',result.manifest['identity']['config'])
        self.assertNotIn('valuation_date',result.manifest['identity']['config'])

    def test_config_cannot_redirect_archive_or_observation_date(self):
        from src.option_calibration import load_calibration_config
        base=json.loads(self.config.read_text())
        for field,value in [('processed_directory','wrong-folder'),('valuation_date','2000-01-03')]:
            self.config.write_text(json.dumps({**base,field:value}))
            with self.assertRaisesRegex(ValueError,'invalid_calibration_schema'):
                load_calibration_config(self.config)

    def test_mixed_live_and_historical_dates_keep_curve_checks(self):
        row,c,_,_=fixtures.live_fixture()
        replace_table(self.folder,'selected_option_quotes',[row])
        h,_,_,cfg,prices=fixtures.historical_fixture()
        h['date']=prices[0]['date']='2026-10-19'
        cfg['manual_carry_inputs'][0]['valuation_date']='2026-10-19'
        self.config.write_text(fixtures.policy_json(cfg))
        replace_table(self.folder,'historical_option_bars',[h])
        replace_table(self.folder,'provider_prices',prices)
        result=calibrate_processed(self.folder,self.config)
        self.assertEqual(result.manifest['summary']['iv_calculated'],1)
        self.assertEqual(result.manifest['summary']['valuation_dates'],['2026-10-19','2026-10-20'])
        rejected=[r for r in result.tables['option_iv'] if r['route']=='historical'][0]
        self.assertEqual(rejected['reason'],'missing_valid_date_matched_curve')

    def test_no_options_skips_without_bootstrap_and_saves_reason(self):
        replace_table(self.folder,'historical_option_bars',[])
        with patch('src.option_calibration.bootstrap_ois',side_effect=AssertionError('No bootstrap needed')):
            result=calibrate_processed(self.folder,self.config)
        self.assertEqual(result.manifest['summary']['skip_reason'],'no_option_data')
        target=result.save()
        self.assertTrue((target/'report.md').exists())

    def test_missing_curve_carry_or_dependency_is_reported_not_raised(self):
        for missing in ('curve','carry','dependency'):
            with self.subTest(missing=missing):
                folder,config=self.folder,self.config
                if missing=='curve':
                    with patch('src.option_calibration.build_curves',return_value=({},[],[],[])):
                        result=calibrate_processed(folder,config)
                elif missing=='carry':
                    cfg=json.loads(config.read_text());cfg['manual_carry_inputs']=[]
                    config.write_text(fixtures.policy_json(cfg))
                    result=calibrate_processed(folder,config)
                else:
                    with patch('src.option_calibration.bootstrap_ois',side_effect=ImportError('QuantLib missing')):
                        result=calibrate_processed(folder,config)
                    self.assertIn('QuantLib missing',result.manifest['summary']['curve_status'][0]['reason'])
                self.assertEqual(result.manifest['summary']['status'],'skipped')

    def test_run_stage_saves_success_and_missing_input_diagnostics(self):
        import run
        with patch.object(run,'CALIBRATION_FILE',self.config), \
             patch.object(run.MarketDataDownloader,'from_json',side_effect=AssertionError('No downloads')), \
             patch('socket.socket.connect',side_effect=AssertionError('No network')),redirect_stdout(StringIO()):
            result=run._calibrate_available(self.folder)
            self.assertEqual(result.manifest['summary']['iv_calculated'],1)
            replace_table(self.folder,'ois_quotes',[])
            result=run._calibrate_available(self.folder)
            self.assertEqual(result.manifest['summary']['status'],'skipped')
            self.assertEqual(len(list((self.folder/'option_calibration').glob('*/manifest.json'))),2)

    def test_main_processing_continues_statistics_and_mc_after_calibration_unavailable(self):
        import run
        # Keep the existing analysis stages isolated: the assertion concerns
        # workflow ordering and continuation, not numerical Monte Carlo output.
        processed=Mock(run_id='fixture',summary={'portfolio':{'holdings':[]},
            'readiness':{'aligned_return_observations':0,'foreign_currencies':[]},'options':{'iv_candidates':0}})
        processed.save.return_value=self.folder
        stats=Mock(summary={'portfolio_status':'unavailable','warnings':[]},run_id='stats')
        simulation=Mock(summary={'status':'unavailable','warnings':[]},run_id='mc')
        order=[]
        def unavailable(*args,**kwargs):
            order.append('calibration');raise ValueError('synthetic insufficient input')
        def statistics(*args):order.append('statistics');return stats
        def mc(*args):order.append('mc');return simulation
        with patch.object(run,'process_inventory',return_value=processed), \
             patch.object(run,'analyze_processed',side_effect=statistics), \
             patch.object(run,'simulate_processed',side_effect=mc), \
             patch('src.option_calibration.calibrate_processed',side_effect=unavailable), \
             patch('socket.socket.connect',side_effect=AssertionError('No network')),redirect_stdout(StringIO()) as output:
            run._process_inventory(SimpleNamespace(cob='2026-10-19'))
        self.assertEqual(order,['calibration','statistics','mc'])
        self.assertIn('Option calibration unavailable',output.getvalue())

    def test_capture_spans_are_separate_for_underlying_and_date(self):
        row,c,_,_=fixtures.live_fixture()
        row['contract']=c
        later={**row,'instrument_id':'NEXT_DAY',
               'capture_timestamp_utc':'2026-10-21T16:00:00-04:00',
               'timestamp':'2026-10-21T16:00:00-04:00',
               'model_received_at_utc':'2026-10-21T16:00:00-04:00'}
        other={**row,'instrument_id':'OTHER_CALL','contract':{**c,'underlying':'OTHER_INDEX'},
               'capture_timestamp_utc':'2026-10-20T17:00:00-04:00',
               'timestamp':'2026-10-20T17:00:00-04:00',
               'model_received_at_utc':'2026-10-20T17:00:00-04:00'}
        replace_table(self.folder,'selected_option_quotes',[row,later,other])
        replace_table(self.folder,'historical_option_bars',[])
        result=calibrate_processed(self.folder,self.config)
        summary=result.manifest['summary']
        self.assertEqual(len(summary['live_capture_windows']),3)
        self.assertEqual(summary['live_capture_span_minutes'],0)
        self.assertFalse(any('live observations span' in x for x in result.manifest['limitations']))

    def test_manifest_cannot_hide_missing_output_from_reuse_verification(self):
        result=calibrate_processed(self.folder,self.config)
        target=result.save()
        manifest=target/'manifest.json'
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        saved['files'].pop('option_iv')
        (target/'option_iv.csv').unlink()
        manifest.write_text(json.dumps(saved),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'output_index_incomplete'):
            result.save()

    def test_unexpected_programming_errors_are_not_silently_skipped(self):
        import run
        with patch('src.option_calibration.calibrate_processed',side_effect=RuntimeError('unexpected bug')), \
             redirect_stdout(StringIO()),self.assertRaisesRegex(RuntimeError,'unexpected bug'):
            run._calibrate_available(self.folder)


class CalibrationConfigurationTests(unittest.TestCase):
    """User input errors must be diagnosed without blocking other analytics."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / 'calibration.json'
        self.base = json.loads(fixtures.policy_json(fixtures.settings()))

    def load(self, cfg):
        from src.option_calibration import load_calibration_config
        self.config.write_text(json.dumps(cfg), encoding='utf-8')
        return load_calibration_config(self.config)

    def test_invalid_root_and_nested_container_types_are_value_errors(self):
        for cfg in (None, [], 'invalid', 7):
            with self.subTest(root=cfg), self.assertRaises(ValueError):
                self.load(cfg)
        for key in ('live', 'historical', 'solver', 'expiry', 'curve_conventions', 'instructions'):
            cfg = deepcopy(self.base); cfg[key] = []
            with self.subTest(section=key), self.assertRaises(ValueError):
                self.load(cfg)
        cfg = deepcopy(self.base); cfg['curve_conventions']['USD'] = []
        with self.assertRaisesRegex(ValueError, 'currency_conventions'):
            self.load(cfg)

    def test_bad_timezones_are_controlled_configuration_errors(self):
        for value in ('Invalid/Zone', '/absolute/zone', None, 5, ''):
            for target in ('valuation', 'expiry', 'override'):
                cfg = deepcopy(self.base)
                if target == 'valuation': cfg['valuation_timezone'] = value
                elif target == 'expiry': cfg['expiry']['timezone'] = value
                else: cfg['expiry']['instrument_overrides'] = {'ARBITRARY': {'timezone': value}}
                with self.subTest(value=value, target=target), self.assertRaises(ValueError):
                    self.load(cfg)

    def test_expiry_override_shapes_and_local_times_are_validated(self):
        for overrides in ([], None, {'TEST': []}, {'TEST': {'times_by_settlement': []}},
                          {'TEST': {'times_by_settlement': {'PM': 1600}}},
                          {'TEST': {'unknown_key': True}}):
            cfg = deepcopy(self.base); cfg['expiry']['instrument_overrides'] = overrides
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.load(cfg)
        for value in ('not-a-time', '16:00:00+02:00', None, 16):
            cfg = deepcopy(self.base); cfg['historical']['cutoff_local_time'] = value
            with self.subTest(cutoff=value), self.assertRaises(ValueError):
                self.load(cfg)
        for value in ([], None, {}, {'PM': '16:00:00Z'}):
            cfg = deepcopy(self.base); cfg['expiry']['times_by_settlement'] = value
            with self.subTest(times=value), self.assertRaises(ValueError):
                self.load(cfg)

    def test_manual_carry_requires_complete_string_identity(self):
        entry = dict(underlying='OTHER_INDEX', currency='USD', valuation_date='2026-10-20',
                     expiration='2027-10-20', source='Test forecast', dividend_pv=1)
        for key in ('underlying', 'currency', 'valuation_date', 'expiration'):
            for mode in ('missing', 'null', 'list'):
                bad = deepcopy(entry)
                if mode == 'missing': del bad[key]
                else: bad[key] = None if mode == 'null' else []
                cfg = deepcopy(self.base); cfg['manual_carry_inputs'] = [bad]
                with self.subTest(key=key, mode=mode), self.assertRaises(ValueError):
                    self.load(cfg)
        for carry in (None, {}, [None], [[]]):
            cfg = deepcopy(self.base); cfg['manual_carry_inputs'] = carry
            with self.subTest(carry=carry), self.assertRaises(ValueError):
                self.load(cfg)
        for source in (None, 7, '   '):
            cfg = deepcopy(self.base); cfg['manual_carry_inputs'] = [{**entry, 'source': source}]
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.load(cfg)

    def test_historical_priority_settings_are_lists(self):
        for key in ('price_priority', 'underlying_provider_priority'):
            for value in (None, {}, 'MIDPOINT', [], [None]):
                cfg = deepcopy(self.base); cfg['historical'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(cfg)

    def test_valid_custom_inputs_are_preserved_without_ticker_assumptions(self):
        cfg = deepcopy(self.base)
        cfg['expiry']['instrument_overrides'] = {'CUSTOM_EU_INDEX': {
            'timezone': 'Europe/Berlin', 'times_by_settlement': {'PM': '17:30:00'}}}
        cfg['historical']['underlying_provider_priority'] = ['custom_provider', 'ibkr']
        cfg['manual_carry_inputs'] = [dict(underlying='CUSTOM_EU_INDEX', currency='EUR',
            valuation_date='2026-10-20', expiration='2027-10-20', source='Test forward', forward=105)]
        self.assertEqual(self.load(cfg), cfg)
        cfg['manual_carry_inputs'][0].pop('forward')
        cfg['manual_carry_inputs'][0]['dividend_pv'] = 0
        self.assertEqual(self.load(cfg), cfg)

    def test_real_configuration_failures_do_not_stop_statistics_or_mc(self):
        import run
        cases = []
        cfg = deepcopy(self.base); cfg['manual_carry_inputs'] = [{
            'underlying': 'TEST', 'valuation_date': '2026-10-20', 'expiration': '2027-10-20',
            'source': 'Test', 'dividend_pv': 1}]
        cases.append(cfg)
        cfg = deepcopy(self.base); cfg['valuation_timezone'] = 'Invalid/Zone'; cases.append(cfg)
        cfg = deepcopy(self.base); cfg['expiry']['instrument_overrides'] = []; cases.append(cfg)
        for cfg in cases:
            self.config.write_text(json.dumps(cfg), encoding='utf-8')
            processed = Mock(run_id='fixture', summary={'portfolio': {'holdings': []},
                'readiness': {'aligned_return_observations': 0, 'foreign_currencies': []},
                'options': {'iv_candidates': 0}})
            processed.save.return_value = Path(self.temp.name)
            stats = Mock(summary={'portfolio_status': 'unavailable', 'warnings': []}, run_id='stats')
            simulation = Mock(summary={'status': 'unavailable', 'warnings': []}, run_id='mc')
            order = []
            def statistics(*args): order.append('statistics'); return stats
            def mc(*args): order.append('mc'); return simulation
            # Exercise the real config loader and real run.py stage handler.
            # Invalid settings must fail before opening any processed inputs.
            with self.subTest(cfg=cfg), patch.object(run, 'CALIBRATION_FILE', self.config), \
                 patch.object(run, 'process_inventory', return_value=processed), \
                 patch.object(run, 'analyze_processed', side_effect=statistics), \
                 patch.object(run, 'simulate_processed', side_effect=mc), \
                 patch('src.option_calibration.load_processed_inputs', side_effect=AssertionError('No data reads')), \
                 patch('socket.socket.connect', side_effect=AssertionError('No network')), \
                 redirect_stdout(StringIO()) as output:
                run._process_inventory(SimpleNamespace(cob='2026-10-19'))
            self.assertEqual(order, ['statistics', 'mc'])
            self.assertIn('Option calibration unavailable', output.getvalue())


if __name__=='__main__':
    unittest.main()
