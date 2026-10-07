import pandas as pd

from .base import MarketDataProvider
from ..models import ObservationSeries, RetrievalError


class FredProvider(MarketDataProvider):
    name = "fred"
    api_url = "https://api.stlouisfed.org/fred"
    kind = "observation"
    key_env = "FRED_API_KEY"

    def fetch(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=False)

    def fetch_raw(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=True)

    def _fetch(self, instrument, symbol, window, *, raw):
        self.check_access()
        common = {"api_key": self.key, "file_type": "json", "series_id": symbol}
        description = self.client.get_json(f"{self.api_url}/series", params=common)
        series = description.get("seriess", [])
        if len(series) != 1:
            raise RetrievalError("invalid_response", "FRED series metadata unavailable.")
        details = series[0]
        rows, offset = [], 0
        for _ in range(100):
            payload = self.client.get_json(f"{self.api_url}/series/observations", params={
                **common, "observation_start": str(window.start), "observation_end": str(window.end),
                "sort_order": "asc", "limit": 100000, "offset": offset,
            })
            if "observations" not in payload:
                raise RetrievalError("invalid_response", "FRED observations missing from response.")
            page = payload["observations"]
            rows.extend(page)
            offset += len(page)
            if offset >= int(payload.get("count", offset)):
                break
            if not page:
                raise RetrievalError("invalid_response", "FRED pagination stopped before advertised count.")
        else:
            raise RetrievalError("invalid_response", "FRED pagination safety limit reached.")
        if not rows:
            raise RetrievalError("no_data", "FRED returned no observations for this window.")
        if raw:
            return  # Bodies already saved by HttpClient before parsing.
        frame = pd.DataFrame(rows)[["date", "value"]]
        return ObservationSeries(frame, self.provenance(
            unit=details.get("units"), frequency=details.get("frequency"), title=details.get("title"),
            seasonal_adjustment=details.get("seasonal_adjustment"), source_last_updated=details.get("last_updated"),
            source_history_start=details.get("observation_start"), source_history_end=details.get("observation_end"),
            vintage_policy="latest available vintage; not point-in-time historical data",
        ), ["Macro/rate values retain provider units; a yield is not a traded bond price."])
