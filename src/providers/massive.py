from urllib.parse import quote, urlsplit

import pandas as pd

from .base import MarketDataProvider
from ..models import PriceSeries, RetrievalError


class MassiveProvider(MarketDataProvider):
    name = "massive"
    api_url = "https://api.massive.com"
    key_env = "MASSIVE_API_KEY"

    def fetch_option_raw(self, instrument, symbol, window, contract):
        """Verify contract reference, then save daily option-trade aggregates."""
        from datetime import date
        self.check_access()
        payload = self.client.get_json(
            f"{self.api_url}/v3/reference/options/contracts/{quote(symbol, safe='')}",
            params={"as_of": str(min(window.end, date.fromisoformat(contract["expiration"])))},
            headers={"Authorization": f"Bearer {self.key}"},
        )
        if payload.get("status") != "OK":
            raise RetrievalError("provider_error", "Massive option contract reference failed; inspect saved response.")
        details = payload.get("results", {})
        expected = {"ticker": symbol, "underlying_ticker": contract["underlying"],
                    "expiration_date": contract["expiration"], "strike_price": contract["strike"],
                    "contract_type": contract["option_type"], "exercise_style": contract["exercise_style"],
                    "shares_per_contract": contract["multiplier"]}
        if any(details.get(key) != value for key, value in expected.items()):
            raise RetrievalError("contract_mismatch", "Massive contract reference differs from the configured identity.")
        self.fetch_raw(instrument, symbol, window)

    def fetch(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=False)

    def fetch_raw(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=True)

    def _fetch(self, instrument, symbol, window, *, raw):
        self.check_access()
        url = f"{self.api_url}/v2/aggs/ticker/{quote(symbol, safe='')}/range/1/day/{window.start}/{window.end}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        headers = {"Authorization": f"Bearer {self.key}"}
        rows, seen = [], set()
        for _ in range(100):
            if url in seen:
                raise RetrievalError("invalid_response", "Massive repeated a pagination link.")
            seen.add(url)
            payload = self.client.get_json(url, params=params, headers=headers)
            if payload.get("status") not in {"OK", "DELAYED"}:
                raise RetrievalError("provider_error", "Massive did not return successful aggregates; check account entitlement.")
            if payload.get("adjusted") is False:
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
            currency=instrument.currency, close_basis="split_adjusted", adj_close_basis=None,
            exchange_timezone="America/New_York", pages=len(seen),
        ), ["Massive adjusted aggregates correct splits, not cash dividends."])
