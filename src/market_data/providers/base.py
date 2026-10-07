from abc import ABC, abstractmethod
from datetime import datetime, timezone

from ..http import HttpClient
from ..models import DateWindow, Instrument, RetrievedSeries, RetrievalError


class MarketDataProvider(ABC):
    name = "abstract"
    api_url = ""
    kind = "price"
    key_env: str | None = None

    def __init__(self, settings: dict[str, str], client: HttpClient | None = None):
        self.settings = settings
        self.key = settings.get(self.key_env, "").strip() if self.key_env else ""
        interval = float(settings.get(f"{self.name.upper()}_MIN_INTERVAL", "0"))
        if not 0 <= interval <= 60:
            raise ValueError("Provider minimum interval must be between 0 and 60 seconds.")
        self.client = client or HttpClient(min_interval=interval)

    def check_access(self):
        if self.key_env and not self.key:
            raise RetrievalError("missing_key", f"Set {self.key_env} in Python settings or environment variables to enable this provider.")
        if self.key.lower() in {"demo", "your_api_key", "your_token"}:
            raise RetrievalError("missing_key", "Use your own API key; demo/sample credentials are disabled.")

    def provenance(self, **kwargs):
        return {"provider": self.name, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "frequency": "daily", **kwargs}

    def cache_settings(self):
        """Only non-secret configuration may enter cached metadata."""
        return {}

    def fetch_option_raw(self, instrument, symbol, window, contract):
        raise RetrievalError("unsupported_product", "This provider has no SPX option-history adapter.")

    def fetch_raw(self, instrument, symbol, window):
        """Retrieve native responses through client.response_sink, without tables."""
        raise NotImplementedError("Raw retrieval is not implemented for this provider.")

    @abstractmethod
    def fetch(self, instrument: Instrument, symbol: str, window: DateWindow) -> RetrievedSeries:
        raise NotImplementedError
