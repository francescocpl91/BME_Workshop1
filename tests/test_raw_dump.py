"""Verify byte preservation, failure evidence and bypassing table normalization."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from market_data import MarketDataDownloader
from market_data.http import HttpClient
from market_data.raw import ResponseRecorder
from market_data.providers.ecb import ECBProvider
from market_data.providers.yahoo import YahooProvider
from market_data.providers.massive import MassiveProvider
from market_data.providers.fred import FredProvider
from market_data.providers.alpha_vantage import AlphaVantageProvider
from market_data.providers.marketdata import MarketDataAppProvider
from market_data.models import Instrument, DateWindow, RetrievalError


def config(symbols, kind="price"):
    return {"settings": {"MASSIVE_API_KEY": "", "FRED_API_KEY": "",
                         "ALPHA_VANTAGE_API_KEY": "", "MARKETDATA_API_TOKEN": "",
                         "ECB_MIN_INTERVAL": "0", "YAHOO_MIN_INTERVAL": "0"},
            "requests": [{"id": "SAMPLE", "ticker": "SAMPLE", "kind": kind,
                          "currency": "USD", "unit": "native", "symbols": symbols,
                          "frequency": "1d", "timeframe": {"start": "2024-01-02", "end": "2024-01-03"}}]}


def response(body, content_type="application/json", status=200):
    result = Mock(status=status, headers={"Content-Type": content_type})
    result.read.return_value = body
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class RawDumpTests(unittest.TestCase):
    def test_ecb_bytes_preserved_without_pandas_or_normalization(self):
        body = b'\xef\xbb\xbfTIME_PERIOD,OBS_VALUE,EXTRA\r\n2024-01-03,.,x\r\n2023-01-01,-999,y\r\n2024-01-03,2,z\r\n'
        downloader = MarketDataDownloader.from_config(config({"ecb": "EXR/D.TEST"}, "observation"))
        downloader.providers[0].client.opener = Mock()
        downloader.providers[0].client.opener.open.return_value = response(body, "text/csv")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("market_data.providers.ecb.pd.read_csv", side_effect=AssertionError("parsed table")), \
             patch("market_data.runner.normalize", side_effect=AssertionError("normalized")):
            result = downloader.download_raw(tmp)
            self.assertEqual(result.summary.status.tolist(), ["response_saved"])
            saved = next(result.folder.rglob("response_*.csv"))
            self.assertEqual(saved.read_bytes(), body)
            metadata = json.loads(saved.with_name("responses.json").read_text())
            self.assertEqual(metadata[0]["sha256"], hashlib.sha256(body).hexdigest())
            self.assertFalse(metadata[0]["credentials_redacted"])
            self.assertEqual(metadata[0]["parameters"]["startPeriod"], "2024-01-02")

    def test_raw_yahoo_and_missing_credentials_are_independent_and_repeatable(self):
        downloader = MarketDataDownloader.from_config(config({"yahoo": "SPY", "massive": "SPY"}))
        body = b'{ "chart": {"result": [{"timestamp": [3, 1, 1], "untouched": null}], "error": null} }\n'
        downloader.providers[0].client.opener = Mock()
        downloader.providers[0].client.opener.open.return_value = response(body)
        downloader.providers[1].client.opener = Mock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(YahooProvider, "fetch", side_effect=AssertionError("normal pipeline")):
            result = downloader.download_raw(tmp)
            second = downloader.download_raw(tmp)
            self.assertNotEqual(result.folder, second.folder)
            self.assertEqual(result.summary.status.tolist(), ["response_saved", "missing_key"])
            self.assertEqual(next(result.folder.rglob("response_*.json")).read_bytes(), body)
            self.assertEqual(len(list((result.folder / "massive").rglob("status.json"))), 1)
            self.assertEqual(len(list((result.folder / "massive").rglob("response_*"))), 0)
            downloader.providers[1].client.opener.open.assert_not_called()
            self.assertIsNone(downloader.providers[0].client.response_sink)

    def test_http_error_and_invalid_json_are_saved_before_failure(self):
        for body, status in [(b'{"error":"denied"}', 403), (b'not JSON\r\n', 200)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                client = HttpClient(retries=0)
                client.response_sink = ResponseRecorder(tmp)
                client.opener = Mock()
                if status == 403:
                    client.opener.open.side_effect = HTTPError("https://example.com/data", status, "denied",
                                                              {"Content-Type": "application/json"}, BytesIO(body))
                else:
                    client.opener.open.return_value = response(body)
                with self.assertRaises(RetrievalError):
                    client.get_json("https://example.com/data")
                self.assertEqual(next(Path(tmp).glob("response_*.json")).read_bytes(), body)

    def test_credential_echoes_redacted_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ResponseRecorder(tmp, ["secret+value"])
            recorder(b'{"echo":"secret+value", "price":123}', 200, "application/json",
                     "https://example.com/data?api_key=secret%2Bvalue&symbol=SPY")
            contents = "\n".join(p.read_text() for p in Path(tmp).iterdir())
            self.assertNotIn("secret+value", contents)
            self.assertNotIn("secret%2Bvalue", contents)
            self.assertTrue(recorder.responses[0]["credentials_redacted"])
            self.assertEqual(recorder.responses[0]["parameters"]["symbol"], "SPY")

    def test_all_json_adapters_save_pages_before_table_conversion(self):
        window = DateWindow.parse("2024-01-02", __import__("datetime").date(2024, 1, 3))
        cases = [
            (AlphaVantageProvider, {"ALPHA_VANTAGE_API_KEY": "fixture-key"},
             [{"Time Series (Daily)": {"2024-01-02": {"4. close": "1"}}}]),
            (MarketDataAppProvider, {"MARKETDATA_API_TOKEN": "fixture-key"}, [{"s": "ok", "t": [1], "extra": "kept"}]),
            (MassiveProvider, {"MASSIVE_API_KEY": "fixture-key"},
             [{"status": "OK", "results": [{"c": 1}], "next_url": "https://api.massive.com/next"},
              {"status": "OK", "results": [{"c": 2}]}]),
            (FredProvider, {"FRED_API_KEY": "fixture-key"},
             [{"seriess": [{"units": "Percent"}]},
              {"count": 2, "observations": [{"value": "."}]},
              {"count": 2, "observations": [{"value": "2"}]}]),
        ]
        for cls, settings, payloads in cases:
            with self.subTest(provider=cls.name), tempfile.TemporaryDirectory() as tmp:
                client = HttpClient()
                client.response_sink = ResponseRecorder(tmp)
                client.opener = Mock()
                bodies = [json.dumps(payload).encode() for payload in payloads]
                client.opener.open.side_effect = [response(body) for body in bodies]
                with patch("pandas.DataFrame", side_effect=AssertionError("table conversion")):
                    cls(settings, client).fetch_raw(Instrument("SPY"), "SPY", window)
                files = sorted(Path(tmp).glob("response_*.json"))
                self.assertEqual([path.read_bytes() for path in files], bodies)

    def test_provider_error_payload_is_kept(self):
        setup = config({"alpha_vantage": "SPY"})
        setup["settings"]["ALPHA_VANTAGE_API_KEY"] = "fixture-key"
        setup["settings"]["ALPHA_VANTAGE_MIN_INTERVAL"] = "0"
        downloader = MarketDataDownloader.from_config(setup)
        body = b'{"Information":"premium endpoint"}'
        downloader.providers[0].client.opener = Mock()
        downloader.providers[0].client.opener.open.return_value = response(body)
        with tempfile.TemporaryDirectory() as tmp:
            result = downloader.download_raw(tmp)
            self.assertEqual(result.summary.status.tolist(), ["entitlement_error"])
            self.assertEqual(result.summary.responses_saved.tolist(), [1])
            self.assertEqual(next(result.folder.rglob("response_*.json")).read_bytes(), body)


if __name__ == "__main__":
    unittest.main()
