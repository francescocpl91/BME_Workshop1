from io import StringIO
import csv
from urllib.parse import quote

import pandas as pd

from .base import MarketDataProvider
from ..models import ObservationSeries, RetrievalError


class ECBProvider(MarketDataProvider):
    name = "ecb"
    api_url = "https://data-api.ecb.europa.eu/service/data"
    kind = "observation"

    def fetch(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=False)

    def fetch_raw(self, instrument, symbol, window):
        return self._fetch(instrument, symbol, window, raw=True)

    def _fetch(self, instrument, symbol, window, *, raw):
        try:
            flow, key = symbol.split("/", 1)
        except ValueError:
            raise RetrievalError("configuration_error", "ECB identifier must be DATAFLOW/SERIES.KEY.") from None
        if not flow or not key or ".." in key or "+" in key or "/" in key:
            raise RetrievalError("configuration_error", "Provide one complete ECB series key; wildcard/multi-series queries are disabled.")
        body = self.client.get(
            f"{self.api_url}/{quote(flow, safe='')}/{quote(key, safe='.')}",
            params={**({"startPeriod": str(window.start)} if window.start is not None else {}), "endPeriod": str(window.end), "format": "csvdata"},
            headers={"Accept": "text/csv"},
        )
        if not body.strip():
            raise RetrievalError("no_data", "ECB returned an empty response.")
        if raw:
            reader = csv.DictReader(StringIO(body))
            if next(reader, None) is None:
                raise RetrievalError("no_data", "ECB returned no observations for this window.")
            return  # Keep the native CSV bytes; no dataframe processing.
        try:
            source = pd.read_csv(StringIO(body), dtype=str)
        except (ValueError, pd.errors.ParserError):
            raise RetrievalError("invalid_response", "ECB response could not be parsed as CSV.") from None
        if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(source):
            raise RetrievalError("invalid_response", "ECB CSV lacks TIME_PERIOD/OBS_VALUE.")
        if source.empty:
            raise RetrievalError("no_data", "ECB returned no observations for this window.")
        if "KEY" in source and source["KEY"].nunique() > 1:
            raise RetrievalError("invalid_response", "ECB returned more than one series.")
        def first(column, default=None):
            values = source[column].dropna() if column in source else pd.Series(dtype=str)
            return values.iloc[0] if len(values) else default
        frequency = first("FREQ", key.split(".")[0])
        if frequency not in {"D", "B"}:
            raise RetrievalError("unsupported_frequency", "Phase 1 ECB adapter accepts daily/business-daily series only.")
        frame = source[[c for c in ("TIME_PERIOD", "OBS_VALUE", "OBS_STATUS") if c in source]].rename(columns={
            "TIME_PERIOD": "date", "OBS_VALUE": "value", "OBS_STATUS": "observation_status",
        })
        return ObservationSeries(frame, self.provenance(
            unit=first("UNIT", instrument.unit), unit_multiplier=first("UNIT_MULT"),
            currency=first("CURRENCY", instrument.currency), frequency=frequency,
            title=first("TITLE_COMPL", first("TITLE", instrument.description)), dataflow=flow,
            source_series_key=first("KEY", f"{flow}.{key}"),
            vintage_policy="latest available vintage; not point-in-time historical data",
        ), ["ECB values are retained in reported units and scale; inspect unit/unit_multiplier before transformations."])
