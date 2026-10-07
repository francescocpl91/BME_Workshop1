"""Option identity and raw routing are checked without paid API calls."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import MarketDataDownloader
from src.json_config import load_provider_configs
from src.models import DateWindow, Instrument, RetrievalError
from src.providers.yahoo import YahooProvider
from src.providers.marketdata import MarketDataAppProvider
from src.providers.massive import MassiveProvider

PROJECT = Path(__file__).resolve().parents[1]
WINDOW = DateWindow(date(2026, 9, 7), date(2026, 10, 6))


def entry(provider="yahoo", side="call"):
    rows = json.loads((PROJECT / "config" / (provider + ".json")).read_text())["requests"]
    return deepcopy(next(r for r in rows if r["product"] == "index_option" and r["contract"]["option_type"] == side))


def load_entry(row, provider="yahoo"):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "provider.json"
        path.write_text(json.dumps({"schema_version": 1, "provider": provider, "requests": [row]}))
        return load_provider_configs(path)


class FakeClient:
    def __init__(self, *payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get_json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.payloads)


class IndexOptionTests(unittest.TestCase):
    def test_configs_merge_same_contract_with_provider_native_symbols(self):
        config = load_provider_configs(PROJECT / "config")
        rows = [r for r in config["requests"] if r["kind"] == "option"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(set(row["symbols"]), {"yahoo", "marketdata", "massive"})
            occ = row["symbols"]["yahoo"]
            self.assertEqual(row["symbols"]["marketdata"], occ)
            self.assertEqual(row["symbols"]["massive"], "O:" + occ)
            self.assertEqual(row["contract"]["exercise_style"], "european")
            self.assertEqual(row["timeframe"], {"lookback": "1mo", "end": None})

    def test_wrong_strike_expiry_side_and_settlement_rejected(self):
        for field, value in [("strike", 7100), ("strike", True), ("strike", 7000.0001),
                             ("expiration", "2026-12-19"), ("expiration", "2026-02-30"),
                             ("option_type", "put"), ("exercise_style", "american"),
                             ("settlement", "physical"), ("settlement_time", "PM"),
                             ("underlying", "SPY"), ("multiplier", 10)]:
            row = entry(); row["contract"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                load_entry(row)
        for provider in ("yahoo", "marketdata", "massive"):
            row = entry(provider); row["input"] = {"symbol": "SPY"}
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                load_entry(row, provider)

    def test_contract_required_and_stock_adjustments_rejected(self):
        row = entry(); del row["contract"]
        with self.assertRaises(ValueError): load_entry(row)
        row = entry(); row["parameters"] = {"auto_adjust": True}
        with self.assertRaises(ValueError): load_entry(row)

    def test_standardized_option_download_rejected_before_network(self):
        downloader = MarketDataDownloader.from_config(load_entry(entry()))
        with patch.object(YahooProvider, "fetch") as fetch, self.assertRaisesRegex(ValueError, "download_raw"):
            downloader.download()
        fetch.assert_not_called()

    def test_raw_dispatch_preserves_contract_and_bypasses_stock_fetch(self):
        row = entry()
        def fetch(provider, instrument, symbol, window, contract):
            self.assertEqual(contract, row["contract"])
            self.assertEqual(instrument.kind, "option")
            provider.client.capture_response(b'{ "original": null }\n', 200, "application/json", "https://example.com/chart")
        downloader = MarketDataDownloader.from_config(load_entry(row))
        with tempfile.TemporaryDirectory() as tmp, patch.object(YahooProvider, "fetch_option_raw", autospec=True, side_effect=fetch), \
             patch.object(YahooProvider, "fetch_raw", side_effect=AssertionError("stock dispatch")):
            result = downloader.download_raw(tmp)
            self.assertEqual(result.summary.status.tolist(), ["response_saved"])
            self.assertEqual(result.summary.response_type.tolist(), ["daily_option_chart_bars"])
            self.assertEqual(next(result.folder.rglob("response_*.json")).read_bytes(), b'{ "original": null }\n')
            self.assertEqual(json.loads(next(result.folder.rglob("request.json")).read_text())["contract"], row["contract"])

    def test_marketdata_uses_quote_endpoint_exclusive_end_and_checks_identity(self):
        row = entry("marketdata")
        symbol = row["input"]["optionSymbol"]
        client = FakeClient({"s": "ok", "optionSymbol": [symbol], "updated": [123], "iv": [None]})
        provider = MarketDataAppProvider({"MARKETDATA_API_TOKEN": "fixture"}, client)
        provider.fetch_option_raw(Instrument(symbol, "option"), symbol, WINDOW, row["contract"])
        url, args = client.calls[0]
        self.assertIn("/options/quotes/", url)
        self.assertEqual(args["params"], {"from": "2026-09-07", "to": "2026-10-07"})
        provider.client = FakeClient({"s": "ok", "optionSymbol": ["WRONG"], "updated": [123]})
        with self.assertRaises(RetrievalError) as error:
            provider.fetch_option_raw(Instrument(symbol, "option"), symbol, WINDOW, row["contract"])
        self.assertEqual(error.exception.code, "contract_mismatch")

    def test_massive_reference_validation_precedes_bars(self):
        row = entry("massive"); c = row["contract"]; symbol = row["input"]["ticker"]
        details = {"ticker": symbol, "underlying_ticker": "SPX", "expiration_date": c["expiration"],
                   "strike_price": c["strike"], "contract_type": "call", "exercise_style": "european", "shares_per_contract": 100}
        client = FakeClient({"status": "OK", "results": details}, {"status": "DELAYED", "results": [{"t": 1, "c": 2}]})
        provider = MassiveProvider({"MASSIVE_API_KEY": "fixture"}, client)
        with patch("pandas.DataFrame", side_effect=AssertionError("table conversion")):
            provider.fetch_option_raw(Instrument(symbol, "option"), symbol, WINDOW, c)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("/v3/reference/options/contracts/", client.calls[0][0])
        self.assertIn("/v2/aggs/ticker/", client.calls[1][0])
        details["exercise_style"] = "american"
        provider.client = FakeClient({"status": "OK", "results": details})
        with self.assertRaises(RetrievalError):
            provider.fetch_option_raw(Instrument(symbol, "option"), symbol, WINDOW, c)
        self.assertEqual(len(provider.client.calls), 1)

    def test_yahoo_requires_option_identity(self):
        row = entry(); symbol = row["input"]["symbol"]
        client = FakeClient({"chart": {"result": [{"meta": {"symbol": symbol, "instrumentType": "OPTION"}, "timestamp": [123]}]}})
        provider = YahooProvider({}, client)
        provider.fetch_option_raw(Instrument(symbol, "option"), symbol, WINDOW, row["contract"])
        provider.client = FakeClient({"chart": {"result": [{"meta": {"symbol": "SPY", "instrumentType": "ETF"}, "timestamp": [123]}]}})
        with self.assertRaises(RetrievalError):
            provider.fetch_option_raw(Instrument(symbol, "option"), symbol, WINDOW, row["contract"])

    def test_alpha_example_stays_disabled_and_cannot_silently_fetch_stock(self):
        row = entry("alpha_vantage")
        self.assertFalse(row["enabled"])
        row["enabled"] = True
        with self.assertRaisesRegex(ValueError, "not implemented"):
            load_entry(row, "alpha_vantage")


if __name__ == "__main__":
    unittest.main()
