"""Portable outputs and an expiring cache of normalized data, without secrets."""
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import uuid
import time

import pandas as pd

from .models import ObservationSeries, PriceSeries


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # Sync clients and virus scanners can briefly hold the destination on Windows.
    # Keep atomic replacement; persistent permission errors still reach the caller.
    for attempt in range(7):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))


def safe_name(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")[:70] or "series"
    return "series_" + stem + "_" + hashlib.sha256(text.encode()).hexdigest()[:8]


class SeriesCache:
    def __init__(self, path: Path, ttl_hours=24.0):
        self.path, self.ttl_hours = path, ttl_hours

    def location(self, provider, instrument, symbol, window):
        # Hash of a complete retrieval identity, including non-secret provider options.
        identity = {"schema": 1, "provider": provider.name, "ticker": instrument.ticker,
                    "kind": instrument.kind, "symbol": symbol, "start": str(window.start), "end": str(window.end),
                    "currency": instrument.currency, "unit": instrument.unit, "description": instrument.description,
                    "options": provider.cache_settings(), "provider_input": instrument.provider_inputs.get(provider.name)}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.path / provider.name / (digest + ".json")

    def get(self, provider, instrument, symbol, window):
        file = self.location(provider, instrument, symbol, window)
        if self.ttl_hours <= 0 or not file.exists():
            return None
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            retrieved = datetime.fromisoformat(payload["metadata"]["retrieved_at_utc"])
            age = (datetime.now(timezone.utc) - retrieved).total_seconds() / 3600
            if not 0 <= age <= self.ttl_hours:
                return None
            frame = pd.read_json(StringIO(payload["table"]), orient="table")
            cls = PriceSeries if instrument.kind == "price" else ObservationSeries
            return cls(frame, payload["metadata"], payload["warnings"])
        except (ValueError, KeyError, TypeError, OSError):
            return None

    def put(self, provider, instrument, symbol, window, result):
        if self.ttl_hours <= 0:
            return
        write_json(self.location(provider, instrument, symbol, window), {
            "metadata": result.metadata, "warnings": result.warnings,
            "table": result.data.to_json(orient="table", date_format="iso"),
        })
