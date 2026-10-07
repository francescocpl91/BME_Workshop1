"""Config drives symbols, routing, dates and result identity; no live API calls."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from src import MarketDataDownloader, PriceSeries, ObservationSeries
from src.providers.yahoo import YahooProvider
from src.providers.ecb import ECBProvider
from src.providers.alpha_vantage import AlphaVantageProvider


def request(name="SPY", start="2024-01-02", end="2024-01-03"):
    return {"id": name, "ticker": "SPY", "kind": "price", "data_type": "etf_price",
            "asset_class": "equity", "description": "Test ETF", "currency": "USD", "unit": "USD/share",
            "exchange": "TEST", "frequency": "1d", "symbols": {"yahoo": "SPY"},
            "timeframe": {"start": start, "end": end}}


def fake_price(provider, instrument, symbol, window):
    return PriceSeries(pd.DataFrame({"date": [str(window.start)], "close": [100.0]}),
                       provider.provenance(currency=instrument.currency, close_basis="split_adjusted"))


class RequestConfigTests(unittest.TestCase):
    def test_different_timeframes_and_same_ticker_do_not_overwrite(self):
        first, second = request("SPY_short"), request("SPY_long", "2023-01-02")
        with patch.object(YahooProvider, "fetch", autospec=True, side_effect=fake_price) as fetch:
            result = MarketDataDownloader.from_config({"requests": [first, second]}).download()
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(set(result.series), {("SPY_short", "yahoo"), ("SPY_long", "yahoo")})
        self.assertEqual(result.windows["SPY_long"].start, date(2023, 1, 2))
        self.assertIsNone(result.window)
        self.assertEqual(result.summary.requested_start.tolist(), ["2024-01-02", "2023-01-02"])
        self.assertEqual(result.series[("SPY_short", "yahoo")].metadata["configured_exchange"], "TEST")

    def test_explicit_provider_symbol_options_and_observation_routing(self):
        price = request()
        price["symbols"] = {"alpha_vantage": "SPY.TEST"}
        price["provider_options"] = {"ALPHA_VANTAGE_OUTPUTSIZE": "full"}
        rate = {"id": "EUR_rate", "ticker": "EUR10Y", "kind": "observation", "data_type": "sovereign_yield",
                "unit": "Percent", "frequency": "1d", "symbols": {"ecb": "YC/B.TEST"},
                "timeframe": {"start": "2024-01-02", "end": "2024-01-03"}}
        def rate_response(provider, instrument, symbol, window):
            return ObservationSeries(pd.DataFrame({"date": [str(window.start)], "value": [2.0]}), provider.provenance(unit="Percent"))
        with patch.object(AlphaVantageProvider, "fetch", autospec=True, side_effect=fake_price) as alpha, \
             patch.object(ECBProvider, "fetch", autospec=True, side_effect=rate_response) as ecb:
            result = MarketDataDownloader.from_config({"settings": {"ALPHA_VANTAGE_API_KEY": "fixture"},
                                                      "requests": [price, rate]}).download()
        self.assertEqual(alpha.call_args.args[2], "SPY.TEST")
        self.assertEqual(alpha.call_args.args[0].cache_settings()["outputsize"], "full")
        self.assertEqual(ecb.call_args.args[2], "YC/B.TEST")
        self.assertEqual(len(result.summary), 2)
        self.assertIsInstance(result.series[("EUR_rate", "ecb")], ObservationSeries)

    def test_all_entries_validated_before_fetch(self):
        variants = []
        bad = request(); bad["timeframe"]["lookback"] = "1y"; variants.append(bad)
        bad = request(); bad["frequency"] = "1h"; variants.append(bad)
        bad = request(); bad["symbols"] = {"fred": "DGS10"}; variants.append(bad)
        bad = request(); bad["symbols"] = {"yahoo": ""}; variants.append(bad)
        bad = request(); bad["timeframe"] = {"start": "2024-02-01", "end": "2024-01-01"}; variants.append(bad)
        bad = request(); bad["timeframe"] = {"lookback": "0y"}; variants.append(bad)
        for bad in variants:
            bad["id"] = "bad"
            with self.subTest(bad=bad), patch.object(YahooProvider, "fetch") as fetch:
                with self.assertRaises(ValueError):
                    MarketDataDownloader.from_config({"requests": [request(), bad]})
                fetch.assert_not_called()
        with self.assertRaises(ValueError):
            MarketDataDownloader.from_config({"requests": [request(), request()]})

    def test_export_metadata_and_summary_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"requests": [request("one"), request("two")], "cache_dir": Path(tmp)/"cache",
                      "output_dir": Path(tmp)/"exports"}
            with patch.object(YahooProvider, "fetch", autospec=True, side_effect=fake_price) as fetch:
                result = MarketDataDownloader.from_config(config).download()
            self.assertEqual(fetch.call_count, 1)
            summary = pd.read_csv(result.folder/"summary.csv")
            for row in summary.itertuples():
                file = result.folder/row.data_file
                self.assertTrue(file.is_file())
                metadata = json.loads(file.with_suffix(".metadata.json").read_text())
                self.assertEqual(metadata["request_id"], row.request_id)
                self.assertEqual(metadata["requested_start"], "2024-01-02")
            self.assertEqual(len(json.loads((result.folder/"manifest.json").read_text())["requests"]), 2)

    def test_missing_keys_retain_requested_dates_and_disabled_requests_skip(self):
        price = request(); price["symbols"] = {"alpha_vantage": "SPY"}
        disabled = request("disabled"); disabled["enabled"] = False
        result = MarketDataDownloader.from_config({"settings": {"ALPHA_VANTAGE_API_KEY": ""},
                                                  "requests": [price, disabled]}).download()
        self.assertEqual(result.summary.status.tolist(), ["missing_key"])
        self.assertEqual(result.summary.requested_end.tolist(), ["2024-01-03"])
        self.assertEqual(result.summary.request_id.tolist(), ["SPY"])
        self.assertEqual(result.series, {})

    def test_mixed_success_and_failure_summary_has_no_dtype_warning(self):
        failed = request("missing_key"); failed["symbols"] = {"alpha_vantage": "SPY"}
        config = {"settings": {"ALPHA_VANTAGE_API_KEY": ""}, "requests": [request(), failed]}
        with warnings.catch_warnings(), patch.object(YahooProvider, "fetch", autospec=True, side_effect=fake_price):
            warnings.simplefilter("error", FutureWarning)
            result = MarketDataDownloader.from_config(config).download()
        self.assertEqual(result.summary.status.tolist(), ["ok", "missing_key"])
        self.assertEqual(result.summary.rows.tolist(), [1, 0])

    def test_lookback_resolves_at_download_time_and_config_is_copied(self):
        class ClockDate(date):
            current = (2024, 1, 31)
            @classmethod
            def today(cls): return cls(*cls.current)
        entry = request(); entry["symbols"] = {"alpha_vantage": "SPY"}
        entry["timeframe"] = {"lookback": "1mo", "end": None}
        config = {"settings": {"ALPHA_VANTAGE_API_KEY": ""}, "requests": [entry]}
        with patch("src.config.date", ClockDate):
            downloader = MarketDataDownloader.from_config(config)
            config["requests"].clear()
            ClockDate.current = (2024, 2, 29)
            result = downloader.download()
        self.assertEqual(result.window.end, date(2024, 2, 29))
        self.assertEqual(result.window.start, date(2024, 1, 29))


if __name__ == "__main__":
    unittest.main()
