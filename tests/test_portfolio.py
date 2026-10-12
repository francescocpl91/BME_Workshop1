"""Portfolio weights are independent from multi-provider retrieval requests."""
from datetime import date
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src import load_portfolio
from src.json_config import load_provider_configs

ROOT = Path(__file__).resolve().parents[1]


class PortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_provider_configs(ROOT / "config", history_start=date(2000,1,3))

    def setUp(self):
        self.data = json.loads((ROOT / 'config/portfolio.json').read_text(encoding='utf-8'))

    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'portfolio.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            return load_portfolio(path, self.config)

    def test_requested_spy_ief_portfolio_has_50_50_weights(self):
        result = self.load(self.data)
        self.assertTrue(result.weights_ready)
        self.assertEqual(result.reporting_currency, 'EUR')
        self.assertEqual(result.target_allocation, {'equity': .5, 'sovereign_fixed_income': .5})
        self.assertEqual([(h['request_id'], h['weight']) for h in result.holdings], [('SPY', .5), ('IEF', .5)])
        self.assertEqual(len(self.config['requests']), 4189)

    def test_complete_weights_are_validated_without_normalizing(self):
        # Synthetic fixture allocations, not proposed portfolio weights.
        self.data['target_allocation'] = {'equity': .5, 'sovereign_fixed_income': .5}
        for row, weight in zip(self.data['holdings'], [.5, .5]): row['weight'] = weight
        result = self.load(self.data)
        self.assertTrue(result.weights_ready)
        self.assertEqual([h['weight'] for h in result.holdings], [.5, .5])
        self.data['holdings'][0]['weight'] = .6
        with self.assertRaises(ValueError): self.load(self.data)

    def test_bad_weights_and_unknown_or_duplicate_instruments_rejected(self):
        for value in [True, -0.1, 1.1, '0.5', float('nan')]:
            data = deepcopy(self.data); data['holdings'][0]['weight'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.load(data)
        for key in ['UNKNOWN', 'SOFR', 'IEF']:
            data = deepcopy(self.data); data['holdings'][0]['request_id'] = key
            with self.subTest(key=key), self.assertRaises(ValueError): self.load(data)

    def test_reference_holdings_overlap_and_incomplete_totals_rejected(self):
        self.data['reference_data'].append('SPY')
        with self.assertRaises(ValueError): self.load(self.data)
        self.data['reference_data'].remove('SPY')
        for row in self.data['holdings']: row['weight'] = 0
        with self.assertRaises(ValueError): self.load(self.data)

    def test_overnight_rates_are_not_sovereign_yields(self):
        rows = {r['id']: r for r in self.config['requests']}
        for key in ['SOFR', 'ESTR']:
            self.assertEqual(rows[key]['kind'], 'observation')
            self.assertEqual(rows[key]['data_type'], 'overnight_reference_rate')
        self.assertEqual(rows['EURUSD']['unit'], 'USD per EUR')

    def test_unset_weights_remain_supported_without_guessing(self):
        for row in self.data['holdings']: row['weight'] = None
        result = self.load(self.data)
        self.assertFalse(result.weights_ready)
        self.assertTrue(all(h['weight'] is None for h in result.holdings))
