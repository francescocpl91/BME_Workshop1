from urllib.parse import quote

import pandas as pd

from .base import MarketDataProvider
from ..models import PriceSeries, RetrievalError


class MarketDataAppProvider(MarketDataProvider):
    name = "marketdata"
    api_url = "https://api.marketdata.app/v1"
    key_env = "MARKETDATA_API_TOKEN"

    def cache_settings(self):
        return {"adjustment": self.settings.get("MARKETDATA_ADJUSTMENT", "split_only")}



    def fetch(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=False)

    def fetch_raw(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=True)

    def _fetch(self, instrument, symbol, window, *, raw):
        self.check_access()
        mode = self.cache_settings()["adjustment"]
        if mode not in {"split_only", "split_and_dividend"}:
            raise RetrievalError("configuration_error", "MARKETDATA_ADJUSTMENT must be split_only or split_and_dividend.")
        payload = self.client.get_json(
            f"{self.api_url}/stocks/candles/D/{quote(symbol, safe='')}/",
            params={"from": window.start.isoformat(), "to": window.end.isoformat(),
                    "adjustsplits": "true", "adjustdividends": str(mode == "split_and_dividend").lower()},
            headers={"Authorization": f"Bearer {self.key}"},
        )
        if payload.get("s") != "ok":
            code = "no_data" if payload.get("s") == "no_data" else "provider_error"
            raise RetrievalError(code, "MarketData returned no successful candle response; check symbol, dates and entitlement.")
        if raw:
            return  # Bodies already saved by HttpClient before parsing.
        required = ("t", "o", "h", "l", "c", "v")
        if any(k not in payload for k in required):
            raise RetrievalError("invalid_response", "MarketData candle fields are missing.")
        frame = pd.DataFrame({k: payload[k] for k in required}).rename(columns={
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        })
        frame["date"] = pd.to_datetime(frame.pop("t"), unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        return PriceSeries(frame, self.provenance(
            currency=instrument.currency, close_basis="split_adjusted" if mode == "split_only" else "split_and_dividend_adjusted",
            adj_close_basis=None, exchange_timezone="America/New_York", adjustment=mode,
        ))
