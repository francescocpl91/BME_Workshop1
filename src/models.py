"""Provider-independent contracts; prices and macro observations stay distinct."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import re
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta


class RetrievalError(Exception):
    """A safe, user-facing failure. Never include a credential-bearing URL."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start date must be on or before end date.")
        if self.end > date.today():
            raise ValueError("End date cannot be in the future on this computer.")

    @classmethod
    def parse(cls, lookback: str, end: date | None = None) -> DateWindow:
        end = end or date.today()
        text = lookback.strip().lower()
        match = re.fullmatch(r"([1-9][0-9]*)(y|mo|m|d|w)", text)
        if match:
            number, unit = int(match[1]), match[2]
            if unit == "y":
                start = end - relativedelta(years=number)
            elif unit in {"m", "mo"}:
                start = end - relativedelta(months=number)
            else:
                start = end - timedelta(days=number * (7 if unit == "w" else 1))
        else:
            try:
                start = date.fromisoformat(text)
            except ValueError:
                raise ValueError("Use 5y, 6mo, 30d, 4w, or a start date YYYY-MM-DD.") from None
        return cls(start, end)


@dataclass(frozen=True)
class Instrument:
    ticker: str
    kind: str = "price"
    symbols: dict[str, str | None] = field(default_factory=dict)
    currency: str | None = None
    unit: str | None = None
    description: str = ""
    mapped: bool = False
    provider_inputs: dict[str, dict] = field(default_factory=dict)

    def symbol_for(self, provider: str, provider_kind: str) -> str | None:
        if self.kind != provider_kind:
            return None
        if provider in self.symbols:
            return self.symbols[provider]
        return self.ticker if self.kind == "price" else None


@dataclass
class RetrievedSeries:
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PriceSeries(RetrievedSeries):
    """Daily market-price observations, not automatically total returns.

    Creating a series computes the mean and sample standard deviation of its
    daily simple returns. Undefined values stay None, never zero.
    """
    mean_return: float | None = field(init=False, default=None)
    std_return: float | None = field(init=False, default=None)

    def __post_init__(self):
        self.update_statistics()

    def update_statistics(self):
        """Recalculate after data changes. A return never bridges a missing price."""
        self.mean_return = self.std_return = None
        # Adjusted prices include distributions, so they describe investor returns.
        columns = [c for c in ("adj_close", "close") if c in self.data and self.data[c].notna().any()]
        if not columns:
            return
        frame = self.data.sort_values("date") if "date" in self.data else self.data
        prices = pd.to_numeric(frame[columns[0]], errors="coerce")
        returns = (prices/prices.shift()-1).replace([float("inf"), -float("inf")], float("nan")).dropna()
        if len(returns) >= 1:
            self.mean_return = float(returns.mean())
        if len(returns) >= 2:
            self.std_return = float(returns.std(ddof=1))


@dataclass
class ObservationSeries(RetrievedSeries):
    """Rates/macro levels in original units, never silently treated as prices."""


def normalize(result: RetrievedSeries, window: DateWindow, kind: str) -> RetrievedSeries:
    """Sort/clip observations; reject conflicting duplicates; never fill gaps."""
    df = result.data.copy()
    value_column = "close" if kind == "price" else "value"
    if df.empty:
        raise RetrievalError("no_data", "No observations returned for this request.")
    if not {"date", value_column}.issubset(df.columns):
        raise RetrievalError("invalid_response", "Required date/value columns are missing.")
    try:
        # Providers must supply exchange-session dates, already stripped of time zones.
        df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    except (ValueError, TypeError):
        raise RetrievalError("invalid_response", "Provider returned invalid dates.") from None
    df = df.loc[df["date"].between(pd.Timestamp(window.start), pd.Timestamp(window.end))].copy()
    fields = [c for c in ("open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "value") if c in df]
    for column in fields:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].replace([float("inf"), -float("inf")], float("nan"))
    duplicate_rows = int(df.duplicated("date").sum())
    if duplicate_rows:
        for _, group in df[df.duplicated("date", keep=False)].groupby("date"):
            if len(group.drop_duplicates()) > 1:
                raise RetrievalError("invalid_response", "Conflicting observations for the same date.")
        df = df.drop_duplicates("date")
        result.warnings.append(f"Removed {duplicate_rows} identical duplicate date rows.")
    df = df.sort_values("date").reset_index(drop=True)
    valid = df[value_column].notna()
    if not valid.any():
        raise RetrievalError("no_data", "No numeric observations within the requested window.")
    if kind == "price":
        if (df.loc[valid, "close"] <= 0).any():
            raise RetrievalError("invalid_response", "Non-positive equity/ETF closing prices.")
        if "volume" in df and (df["volume"] < 0).any():
            raise RetrievalError("invalid_response", "Negative traded volume.")
        if {"high", "low", "open", "close"}.issubset(df):
            inconsistent = ((df.high < df[["open", "low", "close"]].max(axis=1) - 1e-6)
                            | (df.low > df[["open", "high", "close"]].min(axis=1) + 1e-6))
            if inconsistent.any():
                raise RetrievalError("invalid_response", "OHLC fields are internally inconsistent.")
    valid_dates = df.loc[valid, "date"]
    first, last = valid_dates.iloc[0].date(), valid_dates.iloc[-1].date()
    start_lag, end_lag = (first - window.start).days, (window.end - last).days
    result.metadata.update({
        "requested_start": window.start.isoformat(), "requested_end": window.end.isoformat(),
        "first_observation": first.isoformat(), "last_observation": last.isoformat(),
        "rows": len(df), "numeric_observations": int(valid.sum()),
        "missing_values": int((~valid).sum()), "start_gap_calendar_days": start_lag,
        "end_gap_calendar_days": end_lag,
        "coverage_check": "boundary heuristic only; no exchange calendar or inception-date validation",
    })
    if start_lag > 7 or end_lag > 7:
        result.warnings.append("Requested window has a boundary gap over 7 calendar days; inspect history limits, inception and publication lag.")
    if (~valid).any():
        result.warnings.append("Missing observations retained as empty values; no forward-fill applied.")
    if last == date.today():
        result.warnings.append("Today's observation may be provisional or an incomplete daily candle.")
    result.data = df
    if isinstance(result, PriceSeries):
        result.update_statistics()
    return result
