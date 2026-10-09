import pandas as pd

from .base import MarketDataProvider
from ..models import PriceSeries, RetrievalError


class AlphaVantageProvider(MarketDataProvider):
    name = "alpha_vantage"
    api_url = "https://www.alphavantage.co/query"
    key_env = "ALPHA_VANTAGE_API_KEY"

    def cache_settings(self):
        return {"outputsize": self.settings.get("ALPHA_VANTAGE_OUTPUTSIZE", "compact"),
                "adjusted": self.settings.get("ALPHA_VANTAGE_ADJUSTED", "false").lower()}

    def fetch(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=False)

    def fetch_raw(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=True)

    def _fetch(self, instrument, symbol, window, *, raw):
        self.check_access()
        config = self.cache_settings()
        if config["outputsize"] not in {"compact", "full"} or config["adjusted"] not in {"true", "false"}:
            raise RetrievalError("configuration_error", "Invalid Alpha Vantage outputsize/adjusted setting.")
        adjusted = config["adjusted"] == "true"
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        payload = self.client.get_json(self.api_url, params={
            "function": function, "symbol": symbol, "outputsize": config["outputsize"], "apikey": self.key,
        })
        if "Time Series (Daily)" not in payload:
            message = str(payload.get("Information", payload.get("Note", payload.get("Error Message", "")))).lower()
            if any(term in message for term in ("rate limit", "call frequency", "requests per", "calls per")):
                code, detail = "rate_limited", "Alpha Vantage quota reached; retry after quota reset."
            elif "premium" in message or "subscribe" in message:
                code, detail = "entitlement_error", "Alpha Vantage requires a plan entitlement for this request. Try compact/unadjusted or update the plan."
            else:
                code, detail = "provider_error", "Alpha Vantage returned no daily series; check API key, symbol and access."
            raise RetrievalError(code, detail)
        if not payload["Time Series (Daily)"]:
            raise RetrievalError("no_data", "Alpha Vantage returned an empty daily series.")
        if raw:
            return  # Bodies already saved by HttpClient before parsing.
        frame = pd.DataFrame.from_dict(payload["Time Series (Daily)"], orient="index")
        frame.index.name = "date"
        frame = frame.reset_index().rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
            "5. adjusted close": "adj_close", ("6. volume" if adjusted else "5. volume"): "volume",
            "7. dividend amount": "dividends", "8. split coefficient": "stock_splits",
        })
        warnings = []
        if config["outputsize"] == "compact":
            warnings.append("Alpha Vantage compact returns only its latest 100 daily observations; older requested history may be absent.")
        if not adjusted:
            warnings.append("Unadjusted prices: splits and distributions can distort returns; do not use directly for Monte Carlo calibration.")
        return PriceSeries(frame, self.provenance(
            currency=instrument.currency, close_basis="unadjusted",
            adj_close_basis="split_and_dividend_adjusted" if adjusted else None,
            function=function, outputsize=config["outputsize"],
        ), warnings)
