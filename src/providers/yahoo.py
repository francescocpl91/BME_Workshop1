"""Yahoo adapter uses yfinance's session/cookie handling (unofficial API)."""
import time

from .base import MarketDataProvider
from ..models import PriceSeries, RetrievalError


class YahooProvider(MarketDataProvider):
    name = "yahoo"
    api_url = "yfinance.Ticker.history (Yahoo Finance)"

    def fetch_raw(self, instrument, symbol, window):
        """Read Yahoo chart JSON directly, bypassing yfinance history processing.

        Period boundaries are UTC midnight; native timestamps/timezone metadata
        are preserved for inspection. No exchange-calendar clipping is applied.
        This unofficial endpoint can have different access limits from yfinance.
        """
        from datetime import datetime, time as day_time, timedelta, timezone
        from urllib.parse import quote
        start = datetime.combine(window.start, day_time.min, timezone.utc)
        end = datetime.combine(window.end + timedelta(days=1), day_time.min, timezone.utc)
        payload = self.client.get_json(
            "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
            params={"period1": int(start.timestamp()), "period2": int(end.timestamp()),
                    "interval": "1d", "events": "div,splits,capitalGains",
                    "includeAdjustedClose": "true"},
        )
        chart = payload.get("chart", {})
        if chart.get("error"):
            raise RetrievalError("provider_error", "Yahoo chart returned an error; inspect the saved response.")
        if not chart.get("result"):
            raise RetrievalError("no_data", "Yahoo chart returned no results; inspect the saved response.")
        return chart

    def fetch_option_raw(self, instrument, symbol, window, contract):
        chart = self.fetch_raw(instrument, symbol, window)
        results = chart["result"]
        if len(results) != 1 or results[0].get("meta", {}).get("symbol") != symbol or results[0].get("meta", {}).get("instrumentType") != "OPTION":
            raise RetrievalError("contract_mismatch", "Yahoo returned a different instrument; inspect the saved response.")
        if not results[0].get("timestamp"):
            raise RetrievalError("no_data", "Yahoo returned no option history timestamps.")

    def fetch(self, instrument, symbol, window):
        try:
            import yfinance as yf
        except ImportError:
            raise RetrievalError("missing_dependency", "Install the project dependencies to enable yfinance.") from None
        delay = self.client.min_interval - (time.monotonic() - self.client.last_request)
        if delay > 0:
            time.sleep(delay)
        self.client.last_request = time.monotonic()
        try:
            from datetime import timedelta
            ticker = yf.Ticker(symbol)
            history = ticker.history(
                start=window.start.isoformat(), end=(window.end + timedelta(days=1)).isoformat(),
                interval="1d", auto_adjust=False, back_adjust=False,
                actions=True, repair=False, keepna=True, raise_errors=True, timeout=20,
            )
            if history.empty:
                raise RetrievalError("no_data", "Yahoo returned no history; check the exchange-qualified symbol.")
            metadata = ticker.history_metadata or {}
        except RetrievalError:
            raise
        except Exception as exc:
            code = "rate_limited" if "ratelimit" in type(exc).__name__.lower() else "provider_error"
            raise RetrievalError(code, f"Yahoo request failed ({type(exc).__name__}); check connectivity, symbol and access.") from None
        # Preserve the exchange-local date; converting Asian dates to UTC can shift a session.
        history.index = history.index.tz_localize(None) if history.index.tz is not None else history.index
        frame = history.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividends", "Stock Splits": "stock_splits",
        })
        columns = [c for c in ("date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits") if c in frame]
        return PriceSeries(frame[columns], self.provenance(
            currency=metadata.get("currency") or instrument.currency,
            exchange=metadata.get("exchangeName"), exchange_timezone=metadata.get("exchangeTimezoneName"),
            close_basis="split_adjusted", adj_close_basis="split_and_dividend_adjusted" if "adj_close" in frame else None,
            source="Yahoo Finance via yfinance", source_version=getattr(yf, "__version__", "unknown"),
        ), ["Yahoo is an unofficial data source; licensing, session access and availability may change."])
