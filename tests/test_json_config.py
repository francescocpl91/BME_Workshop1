"""Provider-native JSON inputs must map correctly or fail before network I/O."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import MarketDataDownloader, PriceSeries
from src.json_config import load_provider_configs
from src.providers.yahoo import YahooProvider
from src.providers.alpha_vantage import AlphaVantageProvider

PROJECT = Path(__file__).resolve().parents[1]


def entry(provider="yahoo"):
    field = {"yahoo": "symbol", "alpha_vantage": "symbol", "marketdata": "symbol", "massive": "ticker"}[provider]
    return {"id": "SPY", "enabled": True, "ticker": "SPY", "product": "etf", "operation": "price_history",
            "description": "ETF fixture", "asset_class": "equity", "currency": "USD", "unit": "USD/share",
            "exchange": "TEST", "frequency": "1d", "input": {field: "SPY"},
            "timeframe": {"start": "2024-01-02", "end": "2024-01-03"}, "parameters": {}}


def write_file(folder, provider, rows, **extra):
    path = folder / (provider + ".json")
    path.write_text(json.dumps({"schema_version": 1, "provider": provider, "requests": rows, **extra}), encoding="utf-8")
    return path


class JsonConfigTests(unittest.TestCase):
    def test_legacy_max_cannot_silently_substitute_a_fixed_start(self):
        for end in ('1999-12-31', 'COB'):
            with self.subTest(end=end), tempfile.TemporaryDirectory() as tmp:
                row = entry()
                row['timeframe'] = {'lookback':'max','end':end}
                path = write_file(Path(tmp),'yahoo',[row])
                with self.assertRaisesRegex(ValueError, 'lookback=max is no longer'):
                    load_provider_configs(path,cob_date=date(2026,10,8))
                loaded = load_provider_configs(path,cob_date=date(2026,10,8),history_start=date(2010,1,4))
                self.assertEqual(loaded['requests'][0]['timeframe'],{'start':'2010-01-04','end':'2026-10-08'})

    def test_explicit_start_before_2000_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = entry()
            row['timeframe'] = {'start':'1990-01-02', 'end':'1999-12-31'}
            path = write_file(Path(tmp),'yahoo',[row])
            loaded = load_provider_configs(path,cob_date=date(2026,10,8))
            self.assertEqual(loaded['requests'][0]['timeframe'],row['timeframe'])

    def test_supplied_configs_include_reference_rates_and_ibkr(self):
        config = load_provider_configs(PROJECT / "config", history_start=date(2000,1,3))
        self.assertEqual(len(config["requests"]), 4189)
        self.assertEqual(sum(len(row["symbols"]) for row in config["requests"]), 4201)
        rows = {row["id"]: row for row in config["requests"]}
        self.assertEqual(rows["US10Y"]["symbols"], {"fred": "DGS10"})
        self.assertEqual(rows["EURUSD"]["symbols"], {"ecb": "EXR/D.USD.EUR.SP00.A"})

    def test_different_provider_symbols_are_kept_and_retrieved_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            yahoo, alpha = entry(), entry("alpha_vantage")
            alpha["input"]["symbol"] = "SPY.DIFFERENT"
            alpha["parameters"] = {"function": "TIME_SERIES_DAILY_ADJUSTED", "outputsize": "full"}
            write_file(folder, "yahoo", [yahoo]); write_file(folder, "alpha_vantage", [alpha])
            def fetch(provider, instrument, symbol, window):
                return PriceSeries(pd.DataFrame({"date": [str(window.start)], "close": [100]}), provider.provenance(currency="USD"))
            with patch.object(YahooProvider, "fetch", autospec=True, side_effect=fetch) as yf, \
                 patch.object(AlphaVantageProvider, "fetch", autospec=True, side_effect=fetch) as av:
                result = MarketDataDownloader.from_json(folder, settings={"ALPHA_VANTAGE_API_KEY": "fixture"}).download()
            self.assertEqual(yf.call_args.args[2], "SPY")
            self.assertEqual(av.call_args.args[2], "SPY.DIFFERENT")
            self.assertEqual(av.call_args.args[0].cache_settings(), {"outputsize": "full", "adjusted": "true"})
            self.assertEqual(set(result.series), {("SPY", "yahoo"), ("SPY", "alpha_vantage")})

    def test_wrong_native_fields_and_unsupported_parameters_fail(self):
        variants = []
        bad = entry(); bad["input"] = {"ticker": "SPY"}; variants.append(bad)
        bad = entry(); bad["parameters"] = {"expiration": "2027-01-15"}; variants.append(bad)
        bad = entry(); bad["parameters"] = {"auto_adjust": True}; variants.append(bad)
        bad = entry(); bad["operation"] = "option_chain"; variants.append(bad)
        bad = entry(); bad["input"] = {"symbol": "O:AAPL211119C00085000"}; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                path = write_file(Path(tmp), "yahoo", [bad])
                with patch.object(YahooProvider, "fetch") as fetch, self.assertRaises(ValueError):
                    MarketDataDownloader.from_json(path)
                fetch.assert_not_called()

    def test_option_guide_cannot_enable_unimplemented_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = entry(); row["product"] = "option"; row["operation"] = "option_chain"
            path = write_file(Path(tmp), "yahoo", [row], product_guides={"option": {"implemented": True}})
            with self.assertRaisesRegex(ValueError, "not implemented"):
                MarketDataDownloader.from_json(path)
            row["enabled"] = False
            path = write_file(Path(tmp), "yahoo", [entry(), row])
            self.assertEqual(len(load_provider_configs(path)["requests"]), 1)

    def test_conflicting_ids_and_duplicate_pairs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            yahoo, alpha = entry(), entry("alpha_vantage")
            alpha["timeframe"]["start"] = "2023-01-02"
            write_file(folder, "yahoo", [yahoo]); write_file(folder, "alpha_vantage", [alpha])
            with self.assertRaisesRegex(ValueError, "different instrument/timeframe"):
                load_provider_configs(folder)
            path = write_file(folder, "yahoo", [yahoo, yahoo])
            with self.assertRaisesRegex(ValueError, "duplicate request id"):
                load_provider_configs(path)

    def test_json_errors_duplicates_and_credentials_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "broken.json"
            for text in ['{"provider":', '{"schema_version":1,"schema_version":1}', '{"x":NaN}']:
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "could not load"):
                    load_provider_configs(path)
            path = write_file(folder, "yahoo", [entry()], api_key="not-a-real-key")
            with self.assertRaisesRegex(ValueError, "credentials/settings"):
                load_provider_configs(path)

    def test_ecb_fields_and_frequency_are_validated(self):
        base = json.loads((PROJECT / "config/ecb.json").read_text())["requests"][1]
        for inputs in [{"symbol": "EURUSD"}, {"dataflow": "EXR", "series_key": "D..EUR.SP00.A"},
                       {"dataflow": "EXR", "series_key": "M.USD.EUR.SP00.A"}]:
            with tempfile.TemporaryDirectory() as tmp:
                row = deepcopy(base); row["input"] = inputs
                path = write_file(Path(tmp), "ecb", [row])
                with self.assertRaises(ValueError): load_provider_configs(path)

    def test_supported_product_templates_validate_when_copied_into_requests(self):
        for path in (PROJECT / "config").glob("*.json"):
            document = json.loads(path.read_text())
            if path.name in {"portfolio.json", "ois.json", "cleaning.json"}: continue
            for guide in document["product_guides"].values():
                if not guide.get("implemented") or "template" not in guide: continue
                row = deepcopy(guide["template"]); row["enabled"] = True
                with self.subTest(provider=document["provider"], product=row["product"]), tempfile.TemporaryDirectory() as tmp:
                    example = write_file(Path(tmp), document["provider"], [row])
                    self.assertEqual(len(load_provider_configs(example, history_start=date(2000,1,3))["requests"]), 1)

    def test_parameter_conversion_and_disabled_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = entry("marketdata"); row["parameters"] = {"adjustsplits": True, "adjustdividends": True}
            path = write_file(folder, "marketdata", [row])
            config = load_provider_configs(path)
            self.assertEqual(config["requests"][0]["provider_options"], {"MARKETDATA_ADJUSTMENT": "split_and_dividend"})
            row["parameters"]["adjustsplits"] = False
            path = write_file(folder, "marketdata", [row])
            with self.assertRaises(ValueError): load_provider_configs(path)
            write_file(folder, "marketdata", [row], enabled=False)
            write_file(folder, "yahoo", [entry()])
            self.assertEqual(load_provider_configs(folder)["requests"][0]["symbols"], {"yahoo": "SPY"})


if __name__ == "__main__":
    unittest.main()
