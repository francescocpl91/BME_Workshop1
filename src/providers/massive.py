from urllib.parse import quote, urlsplit

import pandas as pd

from .base import MarketDataProvider
from ..models import PriceSeries, RetrievalError


class MassiveProvider(MarketDataProvider):
    name = "massive"
    api_url = "https://api.massive.com"
    key_env = "MASSIVE_API_KEY"


    def fetch(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=False)

    def fetch_raw(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=True)

    def _fetch(self, instrument, symbol, window, *, raw):
        self.check_access()
        url = f"{self.api_url}/v2/aggs/ticker/{quote(symbol, safe='')}/range/1/day/{window.start}/{window.end}"
        is_index = symbol.startswith("I:")
        params = {"sort": "asc", "limit": 50000}
        if not is_index: params["adjusted"] = "true"
        headers = {"Authorization": f"Bearer {self.key}"}
        rows, seen = [], set()
        for _ in range(100):
            if url in seen:
                raise RetrievalError("invalid_response", "Massive repeated a pagination link.")
            seen.add(url)
            payload = self.client.get_json(url, params=params, headers=headers)
            if payload.get("status") not in {"OK", "DELAYED"}:
                raise RetrievalError("provider_error", "Massive did not return successful aggregates; check account entitlement.")
            if is_index and payload.get("ticker") != symbol:
                raise RetrievalError("contract_mismatch", "Massive returned a different index identifier.")
            if not is_index and payload.get("adjusted") is False:
                raise RetrievalError("invalid_response", "Massive returned unadjusted data for an adjusted request.")
            rows.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            if not next_url:
                break
            parsed = urlsplit(next_url)
            if parsed.scheme != "https" or parsed.netloc != "api.massive.com" or parsed.username:
                raise RetrievalError("invalid_response", "Refused an unexpected Massive pagination host.")
            url, params = next_url, None
        else:
            raise RetrievalError("invalid_response", "Massive pagination exceeded the safety limit; shorten the window.")
        if not rows:
            raise RetrievalError("no_data", "Massive returned no daily bars for this window.")
        if raw:
            return  # Bodies already saved by HttpClient before parsing.
        frame = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        frame["date"] = pd.to_datetime(frame.pop("t"), unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        return PriceSeries(frame[[c for c in ("date", "open", "high", "low", "close", "volume") if c in frame]], self.provenance(
            currency=instrument.currency, close_basis="index_level" if is_index else "split_adjusted", adj_close_basis=None,
            exchange_timezone="America/New_York", pages=len(seen),
        ), [] if is_index else ["Massive adjusted aggregates correct splits, not cash dividends."])
