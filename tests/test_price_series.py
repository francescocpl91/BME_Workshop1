"""Automatic price statistics and the keyless first-run workflow."""
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

import run
from src import DateWindow, ObservationSeries, PriceSeries
from src.json_config import load_provider_configs
from src.models import normalize
from src.portfolio import load_portfolio
from src.preprocessing import ArchiveNotFound
from src.providers.ecb import ECBProvider
from src.providers.yahoo import YahooProvider


def prices(values, column="close"):
    dates = pd.bdate_range("2026-01-05", periods=len(values)).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, column: values})


class PriceSeriesStatisticsTests(unittest.TestCase):
    def test_mean_and_std_are_computed_on_creation(self):
        series = PriceSeries(prices([100, 110, 99, 108.9]))
        returns = np.array([.1, -.1, .1])
        self.assertAlmostEqual(series.mean_return, returns.mean())
        self.assertAlmostEqual(series.std_return, returns.std(ddof=1))

    def test_adjusted_close_is_preferred_when_available(self):
        frame = prices([100, 100, 100])
        frame["adj_close"] = [50, 55, 60.5]
        series = PriceSeries(frame)
        self.assertAlmostEqual(series.mean_return, .1)
        self.assertAlmostEqual(series.std_return, 0)

    def test_empty_adjusted_close_falls_back_to_close(self):
        frame = prices([100, 110])
        frame["adj_close"] = np.nan
        self.assertAlmostEqual(PriceSeries(frame).mean_return, .1)

    def test_a_gap_is_not_bridged_by_a_return(self):
        series = PriceSeries(prices([100, 110, None, 200, 220]))
        self.assertAlmostEqual(series.mean_return, .1)

    def test_input_order_does_not_change_returns(self):
        frame = prices([100, 110, 121])
        self.assertAlmostEqual(PriceSeries(frame.iloc[::-1]).mean_return, .1)

    def test_too_little_data_gives_none_rather_than_zero(self):
        for values in ([], [100]):
            series = PriceSeries(prices(values))
            self.assertIsNone(series.mean_return)
            self.assertIsNone(series.std_return)
        one_return = PriceSeries(prices([100, 110]))
        self.assertAlmostEqual(one_return.mean_return, .1)
        self.assertIsNone(one_return.std_return)
        self.assertIsNone(PriceSeries(pd.DataFrame({"date": ["2026-01-05"]})).mean_return)

    def test_normalize_recomputes_on_the_clipped_window(self):
        series = PriceSeries(prices([100, 200, 220, 242]))
        self.assertGreater(series.mean_return, .3)
        window = DateWindow(date(2026, 1, 6), date(2026, 1, 8))
        result = normalize(series, window, "price")
        self.assertAlmostEqual(result.mean_return, .1)

    def test_observations_stay_without_return_statistics(self):
        self.assertFalse(hasattr(ObservationSeries(pd.DataFrame({"date": [], "value": []})), "mean_return"))


class FirstRunTests(unittest.TestCase):
    def test_demo_inputs_need_no_keys_or_broker(self):
        config = load_provider_configs(run.DEMO_CONFIG_FILES, cob_date=date(2026, 10, 8),
                                       history_start=date(2005, 1, 1))
        providers = {name for request in config["requests"] for name in request["symbols"]}
        self.assertEqual(providers, {"yahoo", "ecb"})
        portfolio = load_portfolio(run.DEMO_PORTFOLIO_FILE, config)
        self.assertEqual([h["request_id"] for h in portfolio.holdings], ["SPY", "IEF"])
        self.assertLess(date(int(run.DEMO_HISTORY_START[-4:]), 1, 1), date(2008, 1, 1))

    def test_demo_asks_nothing_and_keeps_its_data_apart(self):
        result = SimpleNamespace(folder=Path("demo_archive"), reused=False,
                                 summary=pd.DataFrame([dict(ticker="SPY", provider="yahoo", status="response_saved",
                                                            retrieval_method="chart")]))
        with TemporaryDirectory() as folder, \
                patch.object(run, "DEMO_DUMP_DIR", Path(folder)), \
                patch.object(run, "london_today", return_value=date(2026, 10, 9)), \
                patch.object(run.MarketDataDownloader, "from_json") as factory, \
                patch.object(run, "_inventory_archive") as inventory, \
                patch("builtins.input", side_effect=AssertionError("No prompts")), \
                redirect_stdout(StringIO()):
            factory.return_value.download_cob.return_value = result
            self.assertIs(run.main(mode="demo"), result)
            self.assertEqual(factory.call_args.args[0], run.DEMO_CONFIG_FILES)
            self.assertEqual(factory.call_args.kwargs["cob_date"], date(2026, 10, 8))
            args = factory.return_value.download_cob.call_args
            self.assertEqual(args.args, (date(2026, 10, 8), Path(folder)))
            self.assertEqual(args.kwargs["portfolio_file"], run.DEMO_PORTFOLIO_FILE)
            self.assertTrue(args.kwargs["retry_failed"])
            inventory.assert_called_once_with(result.folder, run.DEMO_PORTFOLIO_FILE)

    def test_ecb_allows_for_slow_first_responses(self):
        self.assertGreaterEqual(ECBProvider({}).client.timeout, 60)
        self.assertEqual(YahooProvider({}).client.timeout, 20)

    def test_demo_rejects_dates_and_retry(self):
        for options in (dict(cob_text="08/10/2026"), dict(history_start_text="03/01/2000"), dict(retry_failed=True)):
            with self.subTest(options), self.assertRaises(ValueError):
                run.main(mode="demo", **options)

    def test_menu_offers_demo(self):
        with patch("builtins.input", side_effect=["3"]), redirect_stdout(StringIO()):
            self.assertEqual(run.choose_mode(), "demo")

    def test_missing_archive_has_its_own_error_type(self):
        with TemporaryDirectory() as folder, patch.object(run, "RAW_DUMP_DIR", Path(folder)/"missing"), \
                redirect_stdout(StringIO()), self.assertRaises(ArchiveNotFound):
            run.main("08/10/2026", mode="inventory")
        self.assertTrue(issubclass(ArchiveNotFound, ValueError))


if __name__ == "__main__":
    unittest.main()
