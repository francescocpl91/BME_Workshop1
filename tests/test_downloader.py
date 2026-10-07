"""Public library contract tests, with synthetic provider responses only."""
import contextlib
from datetime import date
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import MarketDataDownloader, PriceSeries, ObservationSeries
from src.providers.yahoo import YahooProvider
from src.providers.alpha_vantage import AlphaVantageProvider
from src.providers.ecb import ECBProvider


def price(provider, value):
    return PriceSeries(pd.DataFrame({"date": ["2024-01-02"], "close": [value]}),
                       provider.provenance(currency="USD", close_basis="unadjusted"))


class DownloaderTests(unittest.TestCase):
    def test_in_memory_from_another_directory_and_independent_providers(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            with patch.object(YahooProvider, "fetch", return_value=price(YahooProvider({}), 100)), \
                 patch.object(AlphaVantageProvider, "fetch", return_value=price(AlphaVantageProvider({}), 101)), \
                 patch("builtins.input", side_effect=AssertionError("Library must not prompt")), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                downloader = MarketDataDownloader(["yahoo", "alpha_vantage", "massive"],
                                                  settings={"ALPHA_VANTAGE_API_KEY": "fixture", "MASSIVE_API_KEY": ""})
                result = downloader.download("spy SPY", "2024-01-02", end="2024-01-03")
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertIsNone(result.folder)
            self.assertEqual(result.summary.status.tolist(), ["ok", "ok", "missing_key"])
            self.assertEqual(result.series[("SPY", "yahoo")].data.close.iloc[0], 100)
            self.assertEqual(result.series[("SPY", "alpha_vantage")].data.close.iloc[0], 101)
            self.assertTrue(result.summary.data_file.isna().all())
            self.assertEqual(result.window.end, date(2024, 1, 3))

    def test_python_rate_alias_is_not_a_price(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            observation = ObservationSeries(pd.DataFrame({"date": ["2024-01-02"], "value": [2.5]}),
                                             ECBProvider({}).provenance(unit="PCPA"))
            with patch.object(ECBProvider, "fetch", return_value=observation) as fetch:
                result = MarketDataDownloader(["yahoo", "ecb"]).download(["EUR10Y"], "2024-01-02", end="2024-01-03")
            self.assertEqual(result.summary.status.tolist(), ["not_applicable", "ok"])
            self.assertIsInstance(result.series[("EUR10Y", "ecb")], ObservationSeries)
            self.assertEqual(fetch.call_args.args[1], "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_optional_exports_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(YahooProvider, "fetch", return_value=price(YahooProvider({}), 100)) as fetch:
                downloader = MarketDataDownloader("yahoo", cache_dir=folder / "cache")
                args = dict(tickers=["SPY"], lookback="2024-01-02", end="2024-01-03", output=folder / "runs")
                first, second = downloader.download(**args), downloader.download(**args)
            self.assertEqual(fetch.call_count, 1)
            self.assertFalse(first.series[("SPY", "yahoo")].metadata["cache_hit"])
            self.assertTrue(second.series[("SPY", "yahoo")].metadata["cache_hit"])
            self.assertTrue((second.folder / "summary.csv").exists())
            exported = pd.read_csv(second.folder / "prices_long.csv")
            self.assertEqual(exported.close.tolist(), [100])

    def test_custom_symbol_configuration(self):
        instruments = {"MYETF": {"kind": "price", "currency": "USD", "symbols": {"yahoo": "SPY"}}}
        with patch.object(YahooProvider, "fetch", return_value=price(YahooProvider({}), 100)) as fetch:
            result = MarketDataDownloader("yahoo", instruments=instruments).download(
                ["MYETF"], "2024-01-02", end="2024-01-03")
        self.assertEqual(fetch.call_args.args[1], "SPY")
        self.assertIn(("MYETF", "yahoo"), result.series)

    def test_access_status_does_not_expose_keys_or_claim_remote_permission(self):
        downloader = MarketDataDownloader(["yahoo", "alpha_vantage", "fred"],
                                          settings={"ALPHA_VANTAGE_API_KEY": "private-fixture", "FRED_API_KEY": ""})
        with patch.object(AlphaVantageProvider, "fetch", side_effect=AssertionError("No network")):
            access = downloader.access_status()
        self.assertEqual(access.credential_status.tolist(), ["not_required", "configured", "missing_key"])
        self.assertTrue(access.remote_permission.eq("unverified").all())
        self.assertNotIn("private-fixture", access.to_string())
        self.assertEqual(access.api.iloc[1], "https://www.alphavantage.co/query")

    def test_invalid_input_raises_and_all_failed_still_returns_summary(self):
        with self.assertRaises(ValueError):
            MarketDataDownloader("unknown")
        with self.assertRaises(ValueError):
            MarketDataDownloader("yahoo").download([])
        result = MarketDataDownloader("massive", settings={"MASSIVE_API_KEY": ""}).download(["SPY"])
        self.assertEqual(result.series, {})
        self.assertEqual(result.summary.status.tolist(), ["missing_key"])


if __name__ == "__main__":
    unittest.main()
