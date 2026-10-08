"""Fixed option-universe and one-trigger retrieval regression checks; no network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src import MarketDataDownloader
from src.json_config import load_provider_configs
from src.config import parse_requests
from src.providers.ibkr import validate_ib_input
from src.providers.ibkr import IBKRProvider
from src.models import RetrievalError

ROOT = Path(__file__).resolve().parents[1]


class FixedOptionInputTests(unittest.TestCase):
    def test_every_selected_call_is_explicit_consistent_and_enabled(self):
        universe = None
        counts = {}
        for provider in ('yahoo', 'marketdata', 'massive', 'ibkr'):
            document = json.loads((ROOT / 'config' / (provider + '.json')).read_text())
            rows = [r for r in document['requests'] if r['product'] == 'index_option']
            if provider!='ibkr':
                self.assertEqual(rows,[])
                continue
            self.assertEqual(len(rows), 4181)
            actual = {r['ticker']: r['contract'] for r in rows}
            self.assertEqual(len(actual), 4181)
            if universe is None: universe = actual
            self.assertEqual(actual, universe)
            for r in rows:
                c = r['contract']
                self.assertTrue(r['enabled'])
                self.assertEqual((c['root'], c['option_type'], c['settlement_time']), ('SPX', 'call', 'AM'))
                self.assertLessEqual(c['expiration'], '2027-12-17')
                if provider == 'ibkr':
                    self.assertEqual(r['operation'], 'option_history')
                    self.assertEqual(r['timeframe'], {'start':'COB','end':'COB'})
                    self.assertNotIn('require_iv_greeks', r['parameters'])
                    self.assertNotIn('live_fallback', r['parameters'])
                    self.assertEqual(r['input']['localSymbol'], 'SPX   ' + r['ticker'][3:])
                    self.assertNotIn('lastTradeDateOrContractMonth', r['input'])
                else:
                    self.assertEqual(r['operation'], 'option_history')
                    self.assertEqual(r['timeframe'], {'start': '2026-10-07', 'end': '2026-10-07'} if provider == 'marketdata' else {'lookback': '1mo', 'end': None})
            counts[provider] = len(rows)
        expiries = {c['expiration'] for c in universe.values()}
        self.assertEqual(len(expiries), 14)
        self.assertIn('2027-06-17', expiries)
        self.assertEqual(max(expiries), '2027-12-17')
        self.assertFalse((ROOT / 'src/option_catalogue.py').exists())
        self.assertFalse((ROOT / 'config/option_catalogue.json').exists())
        self.assertFalse(hasattr(MarketDataDownloader, 'download_option_catalogue'))

    def test_local_symbol_is_sufficient_and_inconsistent_identifiers_fail(self):
        row = json.loads((ROOT / 'config/ibkr.json').read_text())['requests'][2]
        self.assertEqual(validate_ib_input(row['input'], 'option', 'USD', row['contract']), row['ticker'])
        for changes in ({'localSymbol': row['input']['localSymbol'].replace('C', 'P')},
                        {'localSymbol': row['ticker']}, {'lastTradeDateOrContractMonth': '20200101'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_ib_input({**row['input'], **changes}, 'option', 'USD', row['contract'])
        missing = deepcopy(row['input']); missing.pop('localSymbol')
        with self.assertRaises(ValueError): validate_ib_input(missing, 'option', 'USD', row['contract'])

    def test_resolved_contract_must_match_expiry_symbol_and_id(self):
        from src.providers.ibkr import _resolved_contract
        from test_cob import fixture
        request=parse_requests(fixture())[0]
        inputs=request.instrument.provider_inputs['ibkr']
        resolved=SimpleNamespace(**inputs,conId=123)
        detail=SimpleNamespace(contract=resolved,realExpirationDate='20261218')
        self.assertIs(_resolved_contract([detail],request.instrument,inputs,request.contract),resolved)
        for changes in ({'realExpirationDate':'20261219'}, {'realExpirationDate':''}):
            bad=deepcopy(detail)
            for key,value in changes.items():setattr(bad,key,value)
            with self.assertRaises(RetrievalError):_resolved_contract([bad],request.instrument,inputs,request.contract)
        for field,value in (('conId',0),('strike',9999),('localSymbol','WRONG')):
            bad=deepcopy(detail);setattr(bad.contract,field,value)
            with self.assertRaises(RetrievalError):_resolved_contract([bad],request.instrument,inputs,request.contract)
        with self.assertRaises(RetrievalError):_resolved_contract([detail,detail],request.instrument,inputs,request.contract)

    def test_option_contract_validation_retained(self):
        from test_cob import fixture
        for field,value in [('strike',True),('strike',7000.0001),('expiration','2026-02-30'),
                            ('option_type','put'),('exercise_style','american'),('settlement','physical'),
                            ('settlement_time','PM'),('underlying','SPY'),('multiplier',10)]:
            config=fixture();config['requests'][0]['contract'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):parse_requests(config)
        for provider in ('yahoo','marketdata','massive'):
            config=fixture();config['requests'][0]['symbols']={provider:'SPX261218C07800000'}
            with self.assertRaises(ValueError):parse_requests(config)

    def test_run_main_triggers_cob_once(self):
        import run
        from datetime import date
        result=SimpleNamespace(folder=Path('fixture'),reused=False,summary=pd.DataFrame([{
            'request_id':'SPX','ticker':'SPX','provider':'ibkr','status':'response_saved','retrieval_method':'reqHistoricalData'}]))
        with patch.object(run.MarketDataDownloader,'from_json') as create,patch('builtins.print'):
            create.return_value.download_cob.return_value=result
            self.assertIs(run.main('07/10/2026'),result)
            create.return_value.download_cob.assert_called_once_with(date(2026,10,7),run.RAW_DUMP_DIR,portfolio_file=run.PORTFOLIO_FILE,retry_failed=False)
            create.return_value.download_raw.assert_not_called()
