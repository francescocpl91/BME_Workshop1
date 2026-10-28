"""Offline cleaning and source selection. Raw archives are never modified.

Historical prices and option snapshots follow independent paths. All source
observations remain traceable; candidates are not certified calibration inputs.

Reading order:
    process_archive -> inspect_archive(include_observations=True)
    -> process_inventory -> _extract -> clean_prices / clean_options /
    clean_auxiliary -> ProcessedArchive -> save (explicit export).

Data moves from provider-shaped archived responses to common observation
dictionaries, then to pandas tables. Only derived dictionaries/tables change;
neither cleaning nor export rewrites the source archive.

Vocabulary used throughout:
    eligible: this row passed the checks applied to it so far.
    price_basis: what a provider's close means (unadjusted, split-adjusted, etc.).
    analysis_field: the selected numerical column, usually close or adj_close.
    return_basis / local variable basis: financial meaning of that selected
        series; a descriptive label, NOT a numerical scaling factor.
    primary: preferred provider for one instrument's entire selected series.
    compatible: a secondary provider is allowed to fill ordinary-price gaps.
        An eligible row does not automatically make its provider compatible.
    iv_candidate: a preliminary option screen, not a calibrated surface input.

Provider priorities and thresholds come from cleaning JSON. Stock/ETF adjusted
preference and the ban on splicing adjusted series are currently Python rules,
not separate JSON switches. Indices have their own return-convention branch.

"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import math

import pandas as pd
import exchange_calendars as xcals

from .json_config import _unique_object, _invalid_constant
from .portfolio import load_portfolio
from .preprocessing import inspect_archive, _number
from .storage import write_json

# These describe the processing implementation and deduplication fields.
# ADJUSTED is a convention label, not an instruction to adjust prices ourselves.
ENGINE_VERSION = 1
PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count")
ADJUSTED = "split_and_dividend_adjusted_close"


def _load(path):
    """Read UTF-8 policy/metadata; reject duplicate keys and NaN/Infinity constants."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _digest(value):
    """Fingerprint structured content consistently, independent of dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive(value):
    """Accept finite positive numbers, excluding bool even though bool subclasses int."""
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def _date(value):
    """Validate a complete ISO date; return None rather than guessing a date format."""
    try:
        return date.fromisoformat(value).isoformat() if isinstance(value, str) else None
    except ValueError:
        return None


def _utc(value):
    """Convert an aware ISO timestamp to UTC; reject timestamps without a timezone."""
    try:
        stamp = datetime.fromisoformat(value)
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (TypeError, ValueError):
        return None


def load_cleaning_config(path):
    """Return a validated policy dictionary, raising ValueError for invalid settings.

    Require known section/field names so misspelled flags cannot silently change
    the intended policy. The instructions section is descriptive only: it does
    not execute rules. Actual thresholds and switches live in the other sections.
    Validation runs before cleaning/export (an inventory may already be loaded).
    """
    p = _load(path)
    required = {"schema_version", "prices", "options", "references", "instruments"}
    if not isinstance(p, dict) or not required <= p.keys() or set(p) - required - {"instructions"}:
        raise ValueError("Cleaning JSON contains missing or unknown sections.")
    if type(p["schema_version"]) is not int or p["schema_version"] != 1:
        raise ValueError("Cleaning schema_version must be 1.")
    # Each operational section has an exact set of supported keys. This schema
    # does not yet expose separate stock/ETF policies; index overrides are below.
    sections = {
        "prices": {"default_calendar", "provider_priority", "allow_price_only",
                   "allow_compatible_gap_fill", "cross_provider_tolerance",
                   "large_return_threshold", "minimum_return_observations"},
        "options": {"provider_priority", "max_relative_spread", "require_dividend_pv",
                    "minimum_vega", "expiration_timezone", "expiration_time_by_settlement",
                    "day_count", "snapshot_max_span_minutes"},
        "references": {"provider_priority"},
    }
    def priorities(values):
        """Require a non-empty, ordered list of distinct provider names."""
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Provider priorities must be a non-empty list of distinct names.")
    def zone(value):
        """Check a real timezone identifier, rather than a guessed UTC offset."""
        try:
            ZoneInfo(value)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Invalid configured timezone.") from exc
    def clock(value):
        """Validate a local expiry-time convention expressed as HH:MM."""
        try:
            if not isinstance(value, str) or len(value) != 5:
                raise ValueError()
            time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Expiry time must use HH:MM.") from exc
    for name, fields in sections.items():
        if not isinstance(p[name], dict) or set(p[name]) != fields:
            raise ValueError(f"Invalid fields in cleaning.{name}.")
        priorities(p[name]["provider_priority"])
    for section, field in (("prices", "allow_price_only"), ("prices", "allow_compatible_gap_fill"),
                           ("options", "require_dividend_pv")):
        if type(p[section][field]) is not bool:
            raise ValueError(field + " must be a boolean.")
    for section, field in (("prices", "cross_provider_tolerance"), ("prices", "large_return_threshold"),
                           ("options", "max_relative_spread"), ("options", "minimum_vega"),
                           ("options", "snapshot_max_span_minutes")):
        if not _positive(p[section][field]):
            raise ValueError(field + " must be a positive finite number.")
    n = p["prices"]["minimum_return_observations"]
    if type(n) is not int or n < 2:
        raise ValueError("minimum_return_observations must be at least 2.")
    if p["options"]["day_count"] != "ACT/365F":
        raise ValueError("Only the explicitly labelled ACT/365F expiry convention is implemented.")
    zone(p["options"]["expiration_timezone"])
    times = p["options"]["expiration_time_by_settlement"]
    if not isinstance(times, dict) or set(times) != {"AM", "PM"}:
        raise ValueError("Expiry-time defaults must specify AM and PM.")
    for value in times.values():
        clock(value)
    calendars = set(xcals.get_calendar_names())
    if p["prices"]["default_calendar"] not in calendars:
        raise ValueError("Unknown default exchange calendar.")
    if not isinstance(p["instruments"], dict):
        raise ValueError("Instrument overrides must be an object keyed by request ID.")
    # Overrides are keyed by archived request ID, not hardcoded Python tickers.
    # option_reference records a modelling relationship only; it does not apply
    # an index-to-ETF transformation or validate that risk mapping.
    allowed = {"calendar", "provider_priority", "index_return_type", "option_reference",
               "expiration_timezone", "expiration_time"}
    for key, row in p["instruments"].items():
        if not isinstance(key, str) or not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("Invalid instrument cleaning override.")
        if "calendar" in row and row["calendar"] not in calendars:
            raise ValueError("Unknown instrument exchange calendar.")
        if "provider_priority" in row:
            priorities(row["provider_priority"])
        if "index_return_type" in row and row["index_return_type"] not in {"price", "gross_total_return", "net_total_return"}:
            raise ValueError("Invalid index return convention.")
        if "option_reference" in row and (not isinstance(row["option_reference"], str) or not row["option_reference"]):
            raise ValueError("option_reference must identify an archived instrument.")
        if "expiration_timezone" in row:
            zone(row["expiration_timezone"])
        if "expiration_time" in row:
            clock(row["expiration_time"])
    return p


@dataclass
class ProcessedArchive:
    """In-memory result: tables are DataFrames; summary is policy/provenance/readiness.

    Creating this object does not export files. Call save() explicitly, or use
    run.py, which calls save() after inventory and cleaning complete.
    """
    archive: Path
    cob: str
    run_id: str
    tables: dict
    summary: dict

    def save(self, destination):
        """Export CSV tables, a detailed manifest and a readable report.

        Write complete=False first, export each table through a temporary file,
        record output hashes/counts, then mark complete=True. Completion means
        export finished, not that all rows or modelling inputs passed checks.
        The caller normally uses processed/<COB>/<run_id> as destination.
        """
        destination = Path(destination).resolve()
        if (destination == self.archive or destination.is_relative_to(self.archive)
                or self.archive.is_relative_to(destination)):
            raise ValueError("Processed outputs must be outside the raw archive and its ancestors.")
        destination.mkdir(parents=True, exist_ok=True)
        # Keep an interrupted export distinguishable from a completed one.
        manifest = {**self.summary, "complete": False, "files": {}}
        write_json(destination / "manifest.json", manifest)
        for name, table in self.tables.items():
            path = destination / (name + ".csv")
            if path.is_symlink():
                raise ValueError("Refusing to overwrite a derived table through a symlink.")
            # Convert nested lists/dictionaries to JSON text in CSV cells, while
            # retaining the original Python values in self.tables for library users.
            export = table.copy()
            for column in export:
                if export[column].map(lambda x: isinstance(x, (dict, list))).any():
                    export[column] = export[column].map(
                        lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x)
            temporary = path.with_suffix(".csv.tmp")
            if temporary.is_symlink():
                raise ValueError("Refusing a derived temporary-file symlink.")
            export.to_csv(temporary, index=False, encoding="utf-8", na_rep="")
            temporary.replace(path)
            manifest["files"][name] = {"file": path.name, "rows": len(table),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        # The report is generated from computed results, not hand-entered counts.
        # Full row-level reasons live in CSVs; policy/readiness lives in manifest.
        lines = ["# Cleaning and merging results", "", f"Archive COB: {self.cob}",
                 f"Processing ID: {self.run_id}", "",
                 "Raw archives are unchanged. These tables are not a calibrated surface or simulation.", "",
                 "## Selected historical sources", "",
                 "| Instrument | Provider | Return convention | Observations | Valid returns |",
                 "|---|---|---|---:|---:|"]
        for s in self.summary["price_selection"]:
            lines.append(f"| {s['instrument_id']} | {s['provider']} | {s['return_basis']} | {s['observations']} | {s['valid_returns']} |")
        lines += ["", "## Option candidates", "",
                  f"Selected quotes: {self.summary['options']['selected_quotes']}",
                  f"IV candidates after field/spread filters: {self.summary['options']['iv_candidates']}",
                  "Candidate status does not certify quote freshness, forward consistency or absence of arbitrage.",
                  "", "## Portfolio", "", ", ".join(f"{h['request_id']}: {h['weight']:.0%}" for h in self.summary["portfolio"]["holdings"]),
                  "Reporting currency: " + self.summary["portfolio"]["reporting_currency"] + ".",
                  "Current weights apply to this derived run. The original archive portfolio is retained in manifest.json.",
                  "", "## Readiness and limitations", ""]
        lines += ["- " + x for x in self.summary["limitations"]]
        path = destination / "report.md"
        if path.is_symlink():
            raise ValueError("Refusing to write a report through a symlink.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["complete"] = True
        manifest["report"] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        write_json(destination / "manifest.json", manifest)
        return destination


def _priority(provider, order):
    """Rank configured providers first; unlisted providers follow alphabetically."""
    return (order.index(provider) if provider in order else len(order), provider)


def _deduplicate(rows, keys, values, issues):
    """Collapse equal observations; quarantine all conflicting versions of a key.

    keys defines observation identity (e.g. instrument/provider/date).
    values defines which contents must agree. Equal records retain all source
    filenames and a duplicate count. Conflicts stay visible but become ineligible;
    this function never chooses an arbitrary winner or averages their values.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for key, group in groups.items():
        fingerprints = {_digest({k: r.get(k) for k in values}) for r in group}
        first = dict(group[0])
        first["source_files"] = sorted({r["source_file"] for r in group})
        first["duplicate_count"] = len(group) - 1
        if len(fingerprints) > 1:
            issues.append({"instrument_id": first["instrument_id"], "provider": first["provider"],
                           "code": "conflicting_duplicates", "key": str(key), "rows": len(group)})
            for r in group:
                r = dict(r)
                r["eligible"] = False
                r["reasons"] = list(r.get("reasons", [])) + ["conflicting_duplicates"]
                r["source_files"] = [r["source_file"]]
                output.append(r)
        else:
            output.append(first)
    return output


def _extract(inventory, issues):
    """Convert parsed rows into six typed collections with common provenance.

    Return prices, corporate actions, option snapshots, reference observations,
    OIS curves and historical option bars. New row dictionaries receive numeric
    conversions and flags; inventory observations/source files are not rewritten.
    """
    prices, actions, options, references, curves, bars = [], [], [], [], [], []
    for item in inventory.requests:
        meta = item["metadata"]
        # Readers may expose Windows or portable separators for the same file.
        # Match normalized names before checking the response's trust/parse role.
        sources = {s["file"].replace("\\", "/"): s for s in item["response_files"]}
        tables = inventory.observations.get((item["provider"], item["request_id"]), {})
        for name, observations in tables.items():
            for raw in observations:
                source_name = raw["source"].replace("\\", "/")
                source = sources.get(source_name, {})
                # A parser can append rows before a later record in that file
                # fails. Exclude ALL rows from a file marked unreadable/untrusted.
                if source.get("role") != "observations":
                    issues.append({"instrument_id": item["request_id"], "provider": item["provider"],
                                   "code": "unreadable_source_excluded", "source_file": raw["source"]})
                    continue
                base = dict(instrument_id=item["request_id"], ticker=item["ticker"],
                            provider=item["provider"], currency=meta.get("currency"), unit=meta.get("unit"),
                            source_currency=meta.get("source_currency"),
                            resolved_currency=(meta.get("resolved_contract") or {}).get("currency"),
                            data_type=meta.get("data_type"), date=_date(raw["date"]), timestamp=raw["timestamp"],
                            source_file=source_name, saved_status=item["saved_status"],
                            requested_start=item["requested_start"], requested_end=item["requested_end"],
                            eligible=True, reasons=[], invalid_fields=[], unavailable_fields=[])
                _validate_currency(base, issues)
                # _number distinguishes absent, malformed and provider-sentinel
                # values. Financial eligibility checks follow in each cleaner.
                vals = {}
                for field, value in raw["values"].items():
                    number, state = _number(value, field, item["provider"] == "ibkr")
                    vals[field] = number
                    if state in {"invalid", "unavailable"}:
                        base[state + "_fields"].append(field)
                row = {**base, **vals}
                if source.get("checksum_matches") is False:
                    row["reasons"].append("checksum_mismatch")
                if item["provider"] != "manual_ois" and source.get("checksum_matches") is None:
                    row["reasons"].append("missing_checksum")
                # eligible begins with source-integrity checks and can become
                # False later. It is not the provider gap-filling permission.
                row["eligible"] = not row["reasons"]
                # Split product paths before generic observations/prices: option
                # history must not accidentally become an underlying price series.
                if item["is_option"]:
                    row["contract"] = meta.get("contract") or {}
                    row["option_dataset"] = name
                    # IBKR intraday timestamps label the START of a bar. Keep
                    # its duration so calibration can check when its close was available.
                    row["bar_size"] = (meta.get("historical_sources", {}).get(source_name)
                                       or meta.get("historical_channels", {}).get(name) or {}).get("bar_size")
                    row["market_data_type"] = meta.get("market_data_type")
                    row["snapshot_completed"] = meta.get("snapshot_completed")
                    row["resolved_contract"] = meta.get("resolved_contract")
                    row["model_received_at_utc"] = meta.get("model_received_at_utc")
                    row["contract_dates"] = meta.get("contract_dates")
                    (options if name == "snapshot" else bars).append(row)
                elif item["provider"] == "manual_ois":
                    row.update(tenor=raw["key"], curve_id=meta.get("curve_id"),
                               quote_type=meta.get("quote_type"), rate_unit=meta.get("rate_unit"),
                               overnight_index=meta.get("overnight_index"), conventions=meta.get("conventions"))
                    curves.append(row)
                elif name == "observations":
                    # ECB describes units in a dict under UNIT; FRED stores a
                    # list of series descriptions with units. This copies units,
                    # not dates or Unix timestamps. FRED vintage bounds describe
                    # the data-version context, not the first/last observation.
                    definition = meta.get("series_definition") or {}
                    row["source_units"] = definition.get("UNIT") if isinstance(definition, dict) else (
                        definition[0].get("units") if definition else None)
                    row["vintage_start"] = meta.get("vintage_start")
                    row["vintage_end"] = meta.get("vintage_end")
                    references.append(row)
                elif name in {"prices", "TRADES", "ADJUSTED_LAST"}:
                    bases = item["price_bases"]
                    # Multiple/unknown recorded close conventions are not guessed.
                    # adjusted_close_basis describes adj_close separately.
                    row["price_basis"] = bases[0] if len(bases) == 1 else "unknown"
                    row["adjusted_close_basis"] = meta.get("adjusted_close_basis")
                    row["frequency"] = meta.get("requested_frequency")
                    prices.append(row)
                    # Alpha Vantage may embed distributions and split factors in daily rows.
                    for field, kind in (("dividends", "dividends"), ("stock_splits", "splits")):
                        if vals.get(field) is not None and vals[field] != (1 if field == "stock_splits" else 0):
                            actions.append({**base, "event_type": kind, "amount": vals[field],
                                            "event_date_semantics": "provider_event_date; payment date not inferred"})
                else:
                    actions.append({**row, "event_type": name,
                                    "event_date_semantics": "provider_event_date; payment date not inferred"})
    return prices, actions, options, references, curves, bars


def _reject(row, reason):
    """Retain the row for inspection but exclude it from eligible price/quote use."""
    if reason not in row["reasons"]:
        row["reasons"].append(reason)
    row["eligible"] = False


def _validate_currency(row, issues):
    """Quarantine explicit denomination conflicts without guessing conversions.

    currency remains the configured denomination; source/resolved currencies are
    retained separately. Missing provider metadata is not a conflict. Comparison
    is case-sensitive because GBp (pence), for example, is not GBP (pounds).
    """
    currencies = {field: row.get(field) for field in
                  ("currency", "source_currency", "resolved_currency") if row.get(field)}
    if len(set(currencies.values())) > 1 and "currency_mismatch" not in row["reasons"]:
        _reject(row, "currency_mismatch")
        issues.append(dict(instrument_id=row["instrument_id"], provider=row["provider"],
                           date=row.get("date"), code="currency_mismatch", **currencies))


def _calendar(name, start, end, cache):
    """Cache dated sessions with UTC open/close times for an exchange calendar.

    Padding gives the calendar library surrounding context. clean_prices limits
    emitted sessions to the requested window; holidays are not missing prices.
    Reference series and option snapshot capture times do not use this calendar.
    """
    key = (name, start, end)
    if key not in cache:
        first = date.fromisoformat(start) - timedelta(days=14)
        last = date.fromisoformat(end) + timedelta(days=7)
        cal = xcals.get_calendar(name, start=first.isoformat(), end=last.isoformat())
        schedule = cal.schedule
        cache[key] = {stamp.date().isoformat(): {
            "calendar": name, "date": stamp.date().isoformat(),
            "open_utc": row["open"].isoformat(), "close_utc": row["close"].isoformat()
        } for stamp, row in schedule.iterrows()}
    return cache[key]


def _price_value(row):
    """Read the chosen analysis column; currently unused by the active pipeline."""
    return row.get(row.get("analysis_field", "close"))


def clean_prices(rows, policy, issues):
    """Validate provider histories, choose sources and compute session returns.

    Return five lists: all provider rows with flags, selected session rows,
    cross-provider comparisons, per-instrument selection summaries, and calendars.
    Selection is per instrument, then per provider; it is not an average of feeds.
    Provider series are ranked by adjustment suitability and configured priority,
    not by length or completeness. No adjusted-price scales are concatenated.
    """
    # Stage 1: establish a single trustworthy value per provider/instrument/day.
    # Conflicting versions remain in provider output, but cannot be selected.
    # Direct helper callers receive the same currency checks as archive readers.
    # Validate before deduplication so conflicting provenance cannot disappear.
    for row in rows:
        _validate_currency(row, issues)
    rows = _deduplicate(rows, ("instrument_id", "provider", "date"),
                        PRICE_FIELDS + ("currency", "source_currency", "resolved_currency",
                                        "unit", "price_basis", "adjusted_close_basis",
                                        "eligible", "reasons"), issues)
    groups, cache, schedules = defaultdict(list), {}, {}
    selected, comparisons, selections = [], [], []
    for row in rows:
        groups[row["instrument_id"]].append(row)
    # Each instrument is selected independently. These loops do not name any
    # particular ticker; product meaning comes from archived metadata/overrides.
    for instrument, group in sorted(groups.items()):
        override = policy["instruments"].get(instrument, {})
        settings = policy["prices"]
        calendar = override.get("calendar", settings["default_calendar"])
        starts = [r["requested_start"] for r in group if _date(r["requested_start"])]
        ends = [r["requested_end"] for r in group if _date(r["requested_end"])]
        dates = [r["date"] for r in group if r["date"]]
        if not dates:
            for r in group:
                _reject(r, "invalid_date")
            continue
        start, end = min(starts or dates), max(ends or dates)
        sessions = _calendar(calendar, start, end, cache)
        schedules.update({(calendar, d): v for d, v in sessions.items() if start <= d <= end})
        # Stage 2: validate rows against their own requested windows and expected
        # daily exchange sessions. Raw fallback responses can extend outside COB.
        by_provider = defaultdict(list)
        for r in group:
            r["calendar"] = calendar
            if r["date"] is None:
                _reject(r, "invalid_date")
            elif not (r["requested_start"] or start) <= r["date"] <= (r["requested_end"] or end):
                _reject(r, "outside_requested_window")
            elif r["date"] not in sessions:
                _reject(r, "not_exchange_session")
            if not r["currency"] or not r["unit"]:
                _reject(r, "missing_currency_or_unit")
            if r["price_basis"] in {None, "unknown"}:
                _reject(r, "unknown_price_basis")
            if r.get("frequency") not in {"1d", "1 day", "daily", "D"}:
                _reject(r, "unsupported_frequency")
            for field in ("open", "high", "low", "close", "adj_close"):
                if r.get(field) is not None and not _positive(r[field]):
                    _reject(r, "nonpositive_" + field)
            if r.get("volume") is not None and r["volume"] < 0:
                r["reasons"].append("invalid_volume")  # Does not invalidate an otherwise valid price.
                r["volume"] = None
            ohlc = [r.get(f) for f in ("open", "high", "low", "close")]
            if all(v is not None for v in ohlc):
                op, hi, lo, cl = ohlc
                if lo > min(op, cl, hi) or hi < max(op, cl, lo):
                    _reject(r, "inconsistent_ohlc")
            by_provider[r["provider"]].append(r)
        # Stage 3: build one candidate series per provider from eligible rows.
        # No rows => no candidate. Price-only permission applies to selecting an
        # entire series; it does NOT allow ordinary closes to fill adjusted gaps.
        candidates = []
        order = override.get("provider_priority", settings["provider_priority"])
        for provider, data in by_provider.items():
            good = [r for r in data if r["eligible"]]
            if not good:
                continue
            types = {r["data_type"] for r in good}
            identities = {(r["currency"], r["unit"], r["price_basis"]) for r in good}
            if len(types) != 1 or len(identities) != 1:
                for r in good:
                    _reject(r, "inconsistent_instrument_metadata")
                continue
            # price_basis describes close; adjusted_close_basis describes
            # adj_close. A positive adj_close must carry the documented
            # split-and-dividend convention to trigger this preference.
            # Indices use close and their explicit index-return convention.
            is_index = types == {"index_price"}
            adjusted = [r for r in good if _positive(r.get("adj_close"))
                        and r.get("adjusted_close_basis") == "split_and_dividend_adjusted"]
            field = "adj_close" if adjusted and not is_index else "close"
            if field == "adj_close":
                basis = ADJUSTED
            elif is_index:
                basis = override.get("index_return_type", "unspecified") + "_index_level"
            else:
                # Some providers deliver fully adjusted prices in close itself
                # (e.g. an ADJUSTED_LAST channel), without an adj_close column.
                # With allow_price_only=False, reject other stock/ETF bases.
                basis = good[0]["price_basis"]
                if basis != ADJUSTED and not settings["allow_price_only"]:
                    continue
            # Map dates to positive values in the selected column. A provider
            # may still have gaps/short coverage: this is not a completeness test.
            chosen = {}
            for r in good:
                # One adjusted observation does not establish the convention
                # for every row. Never fill an adjusted series with an unknown
                # or split-only adjustment, even when price-only use is allowed.
                if field == "adj_close" and r.get("adjusted_close_basis") != "split_and_dividend_adjusted":
                    _reject(r, "incompatible_adjusted_close_basis")
                    continue
                r["analysis_field"], r["return_basis"] = field, basis
                if _positive(r.get(field)):
                    chosen[r["date"]] = r
            if chosen:
                # Tuple: (basis rank, provider rank, name, analysis field,
                #         return basis, {date: observation}).
                # Fully adjusted stock/ETF series rank 0; price-only rank 1.
                # Index candidates rank 0 and are ordered by provider priority.
                candidates.append((0 if basis == ADJUSTED or is_index else 1,
                                   _priority(provider, order), provider, field, basis, chosen))
        if not candidates:
            issues.append({"instrument_id": instrument, "code": "no_usable_price_series"})
            continue
        # Stage 4: first sort key wins before provider order. Thus a lower-priority
        # fully adjusted source beats a higher-priority price-only source.
        # minimum_return_observations is checked AFTER selection, not here.
        candidates.sort(key=lambda c: c[:2])
        _, _, primary, field, basis, primary_rows = candidates[0]
        exemplar = next(iter(primary_rows.values()))
        compatible = {}
        # Stage 5: compare CLOSE fields on overlapping dates with matching
        # currency, units and close adjustment basis. This is diagnostic even
        # when adj_close was selected for analysis. It does not compare/rebase
        # adjusted-price scales, re-rank providers, or remove the primary.
        for other in candidates[1:]:
            provider, other_rows = other[2], other[5]
            checked, failures = 0, 0
            for d in sorted(primary_rows.keys() & other_rows.keys()):
                a, b = primary_rows[d], other_rows[d]
                same = (a["currency"], a["unit"], a["price_basis"]) == (
                    b["currency"], b["unit"], b["price_basis"])
                if not same or not _positive(a.get("close")) or not _positive(b.get("close")):
                    continue
                # Relative difference: 100.05 vs 100.00 is 0.0005 = 0.05%.
                # A 0.001 policy tolerance means 0.1%, NOT 0.001%.
                diff = abs(b["close"] / a["close"] - 1)
                checked += 1
                failures += diff > settings["cross_provider_tolerance"]
                comparisons.append(dict(instrument_id=instrument, date=d, primary_provider=primary,
                    other_provider=provider, price_basis=a["price_basis"], primary_close=a["close"],
                    other_close=b["close"], relative_difference=diff,
                    exceeds_tolerance=diff > settings["cross_provider_tolerance"]))
            # Gap permission is separate from row eligibility. At least two
            # overlaps must be comparable and ALL must pass; one failure anywhere
            # blocks this provider's gap filling for the instrument in this run.
            # The primary/secondary RETURN bases must also match.
            # basis != ADJUSTED is an unconditional current restriction: even
            # perfectly matching close comparisons do not enable adjusted-price
            # filling. Reconciling adjusted scales is not implemented.
            compatible[provider] = (checked >= 2 and failures == 0 and basis != ADJUSTED
                                    and other[4] == basis)
            if not checked:
                issues.append({"instrument_id": instrument, "provider": provider,
                               "code": "no_comparable_overlap"})
        # Stage 6: emit every expected session in the requested window. Keep the
        # primary whenever present; only consider secondary data for absent dates.
        # This may fill interior gaps or extend coverage at either end of a series.
        instrument_rows = []
        for d in sorted(sessions):
            if not start <= d <= end:
                continue
            source = primary_rows.get(d)
            filled = False
            # Try candidates in ranked order and take the first compatible row.
            # Never average sources or substitute an ordinary close into an
            # adjusted series. If no fallback qualifies, retain a missing session.
            if source is None and settings["allow_compatible_gap_fill"]:
                for candidate in candidates[1:]:
                    if compatible.get(candidate[2]) and d in candidate[5]:
                        source, filled = candidate[5][d], True
                        break
            # primary_provider describes the series; provider/source_files
            # identify the observation actually used on this individual date.
            out = dict(instrument_id=instrument, date=d, calendar=calendar, primary_provider=primary,
                       provider=source["provider"] if source else None, currency=exemplar["currency"],
                       unit=exemplar["unit"], analysis_field=field, return_basis=basis,
                       analysis_price=source.get(field) if source else None,
                       close=source.get("close") if source else None,
                       adj_close=source.get("adj_close") if source else None,
                       volume=source.get("volume") if source else None,
                       source_files=source["source_files"] if source else [],
                       gap_filled=filled, status="available" if source else "missing_session",
                       simple_return=None, log_return=None, return_start_date=None,
                       return_status="missing_price",
                       large_return=False)
            # Stage 7: only adjacent expected sessions form a one-session return.
            # Friday->Monday can qualify; a missing Tuesday prevents a Wednesday
            # return from using Monday. Large returns are flagged, not deleted.
            previous = instrument_rows[-1] if instrument_rows else None
            if source:
                out["return_status"] = "no_previous_session_price"
                if previous and _positive(previous["analysis_price"]):
                    ratio = out["analysis_price"] / previous["analysis_price"]
                    out.update(simple_return=ratio-1, log_return=math.log(ratio), return_status="available",
                               return_start_date=previous["date"],
                               large_return=abs(ratio-1) > settings["large_return_threshold"])
            instrument_rows.append(out)
        # Missing coverage includes requested sessions before first available
        # data, not just interior gaps. Report it rather than inventing prices.
        present = [r for r in instrument_rows if r["analysis_price"] is not None]
        valid = sum(r["simple_return"] is not None for r in instrument_rows)
        selections.append(dict(instrument_id=instrument, provider=primary, analysis_field=field,
            return_basis=basis, observations=len(present), valid_returns=valid,
            first_observation=present[0]["date"] if present else None,
            last_observation=present[-1]["date"] if present else None,
            missing_sessions=len(instrument_rows)-len(present),
            compatible_gap_fills=sum(r["gap_filled"] for r in instrument_rows),
            sufficient_history=valid >= settings["minimum_return_observations"],
            option_reference=override.get("option_reference")))
        selected.extend(instrument_rows)
    return rows, selected, comparisons, selections, list(schedules.values())


def clean_options(rows, policy, issues):
    """Screen option snapshots and select whole observations, without pricing.

    quote_eligible: integrity, contract, expiry and bid/ask/spread checks pass.
    iv_candidate: quote checks plus required model fields/vega checks pass.
    european_surface_candidate: iv_candidate with European exercise style.
    selected: representative observation for a contract/day; can still be invalid.
    Return all flagged rows and per-underlying/day capture-span diagnostics.
    """
    values = sorted({k for r in rows for k in r if k.startswith(
        ("bidGreeks.", "askGreeks.", "lastGreeks.", "modelGreeks."))})
    rows = _deduplicate(rows, ("instrument_id", "provider", "timestamp"),
                        tuple(values) + ("bid", "ask", "last", "close", "bidSize", "askSize", "lastSize", "volume",
                                         "contract", "currency", "unit", "market_data_type", "eligible", "reasons"), issues)
    groups = defaultdict(list)
    cfg = policy["options"]
    # First retain contract identity and assess each snapshot independently.
    for r in rows:
        c = r.pop("contract")
        r.update(underlying=c.get("underlying"), expiration=c.get("expiration"),
                 strike=c.get("strike"), option_type=c.get("option_type"),
                 exercise_style=c.get("exercise_style"), settlement=c.get("settlement"),
                 settlement_time=c.get("settlement_time"), multiplier=c.get("multiplier"),
                 midpoint=None, relative_spread=None, time_to_expiry=None,
                 expiry_timestamp=None, expiry_time_source=None, selected=False)
        stamp = _utc(r["timestamp"])
        if stamp is None:
            _reject(r, "missing_capture_timestamp")
        else:
            r["capture_timestamp_utc"] = stamp.isoformat()
            r["capture_date"] = stamp.date().isoformat()
        for key in ("strike", "multiplier"):
            r[key], _ = _number(r.get(key))
            if not _positive(r[key]):
                _reject(r, "missing_or_invalid_" + key)
        for key in ("underlying", "currency", "exercise_style", "settlement"):
            if not r.get(key):
                _reject(r, "missing_" + key)
        if r["exercise_style"] not in {"european", "american"}:
            _reject(r, "unknown_exercise_style")
        if r["option_type"] not in {"call", "put"}:
            _reject(r, "unknown_option_type")
        expiry = _date(r["expiration"])
        # Underlying defaults can be overridden by an individual contract.
        # Expiry comes from configured contract metadata plus a local clock,
        # not automatic validation against the saved exchange contract details.
        override = {**policy["instruments"].get(r.get("underlying"), {}),
                    **policy["instruments"].get(r["instrument_id"], {})}
        expiry_time = override.get("expiration_time", cfg["expiration_time_by_settlement"].get(
            r["settlement_time"]))
        if expiry and expiry_time:
            expiry_stamp = datetime.combine(date.fromisoformat(expiry), time.fromisoformat(expiry_time),
                ZoneInfo(override.get("expiration_timezone", cfg["expiration_timezone"])))
            r["expiry_timestamp"] = expiry_stamp.isoformat()
            r["expiry_time_source"] = "configured_convention_estimate"
            if stamp:
                # ACT/365F uses elapsed calendar time from actual capture, not
                # 252 trading days and not the archive's COB label.
                r["time_to_expiry"] = (expiry_stamp - stamp).total_seconds() / (365*86400)
                if r["time_to_expiry"] <= 0:
                    _reject(r, "expired_at_capture")
        else:
            _reject(r, "missing_expiry_or_time_convention")
        if not (_positive(r.get("bid")) and _positive(r.get("ask"))):
            _reject(r, "missing_positive_bid_ask")
        elif r["bid"] > r["ask"]:
            _reject(r, "crossed_market")
        else:
            r["midpoint"] = (r["bid"] + r["ask"])/2
            r["relative_spread"] = (r["ask"] - r["bid"])/r["midpoint"]
            if r["relative_spread"] > cfg["max_relative_spread"]:
                _reject(r, "wide_spread")
        # Freeze quote eligibility before model checks. A good bid/ask can
        # exist without model IV. Missing optional Greeks/volume do not alone
        # fail the screen; supplied near-zero/negative vega does.
        r["quote_eligible"] = r["eligible"]
        iv_reasons = list(r["reasons"])
        for key in ("modelGreeks.impliedVol", "modelGreeks.undPrice"):
            if not _positive(r.get(key)):
                iv_reasons.append("missing_positive_" + key)
        pv = r.get("modelGreeks.pvDividend")
        if cfg["require_dividend_pv"] and (pv is None or pv < 0):
            iv_reasons.append("missing_nonnegative_dividend_pv")
        vega = r.get("modelGreeks.vega")
        if vega is not None and vega < cfg["minimum_vega"]:
            iv_reasons.append("low_or_negative_model_vega")
        r["iv_reasons"] = iv_reasons
        r["iv_candidate"] = not iv_reasons
        r["european_surface_candidate"] = r["iv_candidate"] and r["exercise_style"] == "european"
        # Keep the provider's model IV, option price, underlying and dividend PV
        # together. They are NOT assumed to be the IV of our quoted midpoint.
        # Dividend PV is a model input estimate, not a cashflow forecast schedule.
        r["implied_volatility_unit"] = "annualized_decimal"
        r["dividend_pv_meaning"] = "provider_model_PV_of_expected_dividends_to_expiry; not a cashflow_schedule"
        r["quote_timestamp_semantics"] = "SDK_receipt; exchange_age_unverified"
        # Contract identity and observation day prevent blending distinct expiries or days.
        key = (r["underlying"], r["expiration"], r["strike"], r["option_type"],
               r["exercise_style"], r["settlement_time"], r["currency"], r["multiplier"],
               r.get("capture_date"))
        groups[key].append(r)
    # Pick a whole row, never individual fields from different snapshots.
    # Keep a diagnostic representative even when no observation passes.
    for group in groups.values():
        eligible = [r for r in group if r["iv_candidate"]]
        pool = eligible or [r for r in group if r["quote_eligible"]] or group
        # Prefer configured source and then latest complete observation from that source.
        source = min(pool, key=lambda r: _priority(r["provider"], cfg["provider_priority"]))["provider"]
        chosen = max((r for r in pool if r["provider"] == source), key=lambda r: r["timestamp"] or "")
        chosen["selected"] = True
    # A wide receipt-time span warns that quotes may reflect different markets.
    # This does not prove exchange quote freshness or automatically reject the
    # entire candidate set; the warning is reported for later calibration checks.
    spans = []
    by_capture = defaultdict(list)
    for r in rows:
        if r["selected"] and r.get("capture_date"):
            by_capture[(r["underlying"], r["capture_date"])].append(r)
    for (underlying, capture_date), group in by_capture.items():
        stamps = [_utc(r["timestamp"]) for r in group]
        span = (max(stamps)-min(stamps)).total_seconds()/60
        spans.append(dict(underlying=underlying, capture_date=capture_date,
                          first_capture=min(stamps).isoformat(), last_capture=max(stamps).isoformat(),
                          span_minutes=span, exceeds_span_limit=span > cfg["snapshot_max_span_minutes"]))
    return rows, spans


def clean_auxiliary(actions, references, curves, bars, policy, issues):
    """Clean non-price categories while preserving their distinct meanings.

    Actions remain separate from adjusted returns (no dividend double counting).
    Reference observations retain native dates/vintages rather than XNYS dates.
    Fixed OIS par quotes retain actual dates even outside the requested COB window.
    Historical option bars are not given invented IVs, Greeks or settlement labels.
    """
    actions = _deduplicate(actions, ("instrument_id", "provider", "event_type", "date"),
        ("amount", "numerator", "denominator", "currency", "eligible", "reasons"), issues)
    references = _deduplicate(references, ("instrument_id", "provider", "date"),
        ("value", "unit", "currency", "eligible", "reasons"), issues)
    curves = _deduplicate(curves, ("instrument_id", "provider", "date", "tenor"),
        ("rate_percent", "currency", "quote_type", "rate_unit", "eligible", "reasons"), issues)
    bars = _deduplicate(bars, ("instrument_id", "provider", "option_dataset", "timestamp", "date", "bar_size"),
        PRICE_FIELDS + ("eligible", "reasons"), issues)
    # OIS is intentionally absent: fixed curves follow their explicit reuse
    # policy, whereas the other dated observations are checked against the window.
    for row in actions + references + bars:
        if row["date"] is None:
            _reject(row, "invalid_date")
        elif ((row["requested_start"] and row["date"] < row["requested_start"])
                or (row["requested_end"] and row["date"] > row["requested_end"])):
            _reject(row, "outside_requested_window")
    for r in actions:
        if r["event_type"] == "dividends" and not _positive(r.get("amount")):
            _reject(r, "missing_positive_distribution")
        if r["event_type"] == "splits":
            numerator, denominator = r.get("numerator"), r.get("denominator")
            r["split_factor"] = numerator / denominator if _positive(numerator) and _positive(denominator) else r.get("amount")
            if not _positive(r["split_factor"]):
                _reject(r, "invalid_split")
    # Rate values can be negative; FX cannot. Units, rather than magnitude,
    # decide whether a value such as 4.5 should also be represented as 0.045.
    selected_refs = []
    groups = defaultdict(list)
    for r in references:
        if r.get("value") is None:
            _reject(r, "missing_value")
        if r["data_type"] == "fx_reference_rate" and not _positive(r.get("value")):
            _reject(r, "nonpositive_fx_rate")
        percent = (r.get("unit") or "").lower().startswith("percent") or r.get("source_units") in {"Percent", "PCPA"}
        r["rate_decimal"] = r["value"]/100 if percent and r.get("value") is not None else None
        if r["eligible"]:
            groups[r["instrument_id"]].append(r)
    # One primary source per reference instrument; no cross-provider averaging
    # or missing-date filling is performed for rates/FX in this function.
    for group in groups.values():
        primary = min(group, key=lambda r: _priority(r["provider"], policy["references"]["provider_priority"]))["provider"]
        selected_refs.extend(r for r in group if r["provider"] == primary)
    # Conversion from percent to decimal does not bootstrap a zero/discount curve.
    # Original par quote type, tenor, date and conventions remain alongside rates.
    for r in curves:
        if not r["date"] or r.get("rate_percent") is None:
            _reject(r, "missing_curve_observation")
        if r["rate_unit"] != "percent_per_annum":
            _reject(r, "unknown_curve_rate_unit")
        r["rate_decimal"] = r["rate_percent"]/100 if r["eligible"] else None
        r["is_discount_factor"] = False
    for r in bars:
        # No IV/Greek inference, and no daily EOD claim for intraday historical bars.
        r["timestamp_semantics"] = "provider_bar_timestamp"
        if not _positive(r.get("close")):
            _reject(r, "missing_positive_close")
    return actions, references, selected_refs, curves, bars


def _table(rows, required=()):
    """Build a DataFrame with stable required columns, even when no rows exist."""
    frame = pd.DataFrame(rows)
    for name in required:
        if name not in frame:
            frame[name] = pd.Series(dtype="object")
    return frame


def process_inventory(inventory, cleaning_file, *, portfolio_file=None, progress=None):
    """Clean parsed local observations, using current weights and archived data.

    Set include_observations=True when creating the inventory. No provider
    adapter is constructed and no raw input is rewritten.

    Archived inputs define instruments and original observations. The optional
    current portfolio file defines holdings/weights for this derived analysis.
    Without it, use the archive's portfolio. Both snapshots remain in summary.
    Return a ProcessedArchive; exporting is a separate save() operation.
    """
    if inventory.observations is None:
        raise ValueError("Processing requires inspect_archive(..., include_observations=True).")
    p = load_cleaning_config(cleaning_file)
    inputs = _load(inventory.folder / "resolved_inputs.json")
    portfolio_path = Path(portfolio_file) if portfolio_file is not None else inventory.folder / "portfolio.json"
    portfolio = load_portfolio(portfolio_path, inputs)
    if not portfolio.weights_ready:
        raise ValueError("Set complete portfolio weights before processing portfolio returns.")
    if progress:
        progress("Cleaning provider histories and corporate actions; checking exchange sessions.")
    # Carry reader problems forward; add processing problems to the same log.
    # Extraction separates product paths before any source-selection decisions.
    issues = [{**issue, "provider": item["provider"], "instrument_id": item["request_id"]}
              for item in inventory.requests for issue in item["issues"]]
    prices, actions, options, references, curves, bars = _extract(inventory, issues)
    # Capture raw count before equal observations are collapsed by deduplication.
    raw_option_count = len(options)
    prices, selected, comparisons, selection, calendar = clean_prices(prices, p, issues)
    if progress:
        progress("Screening saved option quotes; keeping model fields and capture dates separate.")
    options, spans = clean_options(options, p, issues)
    actions, references, selected_refs, curves, bars = clean_auxiliary(
        actions, references, curves, bars, p, issues)
    holdings = [h["request_id"] for h in portfolio.holdings]
    # Explicit missing sessions remain in the long table. The wide table contains
    # only complete, adjacent-session returns with identical start AND end
    # dates for every holding. Different calendars can share an ending date
    # while measuring different intervals; those rows are not joint returns.
    return_maps = {key: {r["date"]: r for r in selected if r["instrument_id"] == key
                        and r["simple_return"] is not None} for key in holdings}
    common = sorted(set.intersection(*(set(return_maps[k]) for k in holdings))) if holdings else []
    aligned = []
    mismatched_intervals = 0
    for d in common:
        starts = {key: return_maps[key][d]["return_start_date"] for key in holdings}
        if len(set(starts.values())) != 1 or None in starts.values():
            mismatched_intervals += 1
            issues.append(dict(code="unaligned_return_intervals", date=d,
                               return_start_dates=starts))
            continue
        aligned.append({"date": d,
                        **{key: return_maps[key][d]["simple_return"] for key in holdings}})
    # The wide table keeps its existing date/holding columns; interval starts
    # remain explicit in selected_prices and rejected intervals in issues.
    # Readiness is diagnostic. The aligned table contains individual local-
    # currency SIMPLE returns, not weighted portfolio returns, FX-adjusted
    # returns or a covariance/volatility estimate. Weights are recorded only.
    summaries = {s["instrument_id"]: s for s in selection}
    missing = [k for k in holdings if k not in summaries or not summaries[k]["observations"]]
    insufficient = [k for k in holdings if k not in summaries or not summaries[k]["sufficient_history"]]
    # Report foreign currencies and available FX IDs; this is not yet a full
    # currency-pair/direction/date-coverage validation or an FX conversion.
    required_fx = sorted({r["currency"] for r in selected if r["instrument_id"] in holdings
                          and r["currency"] and r["currency"] != portfolio.reporting_currency})
    available_fx = sorted({r["instrument_id"] for r in selected_refs if r["data_type"] == "fx_reference_rate"})
    # One option can have several exclusion reasons; counts must not be summed
    # as a number of distinct rejected contracts.
    counts = Counter(reason for r in options for reason in r["iv_reasons"])
    chosen_options = [r for r in options if r["selected"]]
    candidates = [r for r in chosen_options if r["iv_candidate"]]
    current_portfolio = _load(portfolio_path)
    old_path = inventory.folder / "portfolio.json"
    archived_portfolio = _load(old_path) if old_path.exists() else None
    # Identity binds this derived result to its sources, policy and portfolio.
    # Actual receipt times are source metadata; the time of this offline rerun
    # is not added as an arbitrary new identity.
    source_identity = [
        {"provider": r["provider"], "id": r["request_id"], "attempt": r["selected_attempt"],
         "start": r["requested_start"], "end": r["requested_end"], "metadata": r["metadata"],
         "sources": r["response_files"], "status": r["saved_status"]}
        for r in inventory.requests]
    input_hashes = {name: hashlib.sha256((inventory.folder/name).read_bytes()).hexdigest()
                    for name in ("manifest.json", "resolved_inputs.json", "summary.json")}
    # Full source-file hashes include comments/docstrings: documenting these
    # modules can change run_id even when the numerical behavior is unchanged.
    identity = dict(engine_version=ENGINE_VERSION, policy=p, portfolio=current_portfolio,
                    archive_inputs=input_hashes, sources=source_identity,
                    calendar_library=version("exchange-calendars"),
                    code_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("data_cleaning.py", "preprocessing.py", "portfolio.py")},
                    pandas_version=version("pandas"))
    run_id = _digest(identity)[:16]
    # These statements become report content; they are not executable rules.
    # In particular, IV candidates still need synchronization/arbitrage/forward
    # checks, and option_reference does not calibrate an ETF/index mapping.
    limitations = [
        "Raw archives are unchanged. Current portfolio weights are used for derived outputs only.",
        "Adjusted-close returns approximate reinvested distributions; dividends are retained separately and are not added again.",
        "No adjusted-price scales are spliced. Missing sessions are not interpolated or forward-filled; gap returns are excluded.",
        "Option candidates are preliminary screens, not an arbitrage-free or calibrated surface.",
        "Option IVs and model underlying/dividend PV remain paired; midpoint is not assumed to imply the model IV.",
        "Dividends in historical tables are realized events. Option dividend PV is a model estimate, not a forecast cashflow schedule.",
        "SDK receipt timestamps do not prove quote freshness. Snapshot batches may cover different market conditions.",
        "Expiry time uses the configured convention estimate; exchange settlement details need validation before calibration.",
        "Greeks/volume are retained where supplied. Zero volume does not alone reject a quote; open interest is not fabricated.",
        "Reference series retain native dates/current vintages. Fixed par OIS quotes retain their actual observation dates.",
        "Aligned returns are in each instrument's local currency. No EUR conversion, portfolio valuation or simulation is performed.",
        "The configured option-reference relationship is metadata only; its risk mapping to holdings has not been calibrated.",
    ]
    if any(s["exceeds_span_limit"] for s in spans):
        limitations.append("Option captures exceed the configured synchronization window; candidates must not be treated as simultaneous quotes.")
    if required_fx and not available_fx:
        limitations.append("No usable FX history is available for the foreign-currency holdings; reporting-currency analysis remains incomplete.")
    # An empty dataset dictionary/list is not evidence of usable observations.
    # Keep provider status separate from whether any rows were parsed.
    request_status = [dict(instrument_id=r["request_id"], provider=r["provider"], status=r["saved_status"],
                           has_parsed_observations=any(inventory.observations.get((r["provider"],r["request_id"]), {}).values()))
                      for r in inventory.requests]
    summary = dict(schema_version=1, engine_version=ENGINE_VERSION, run_id=run_id,
        cob=inventory.cob, archive=str(inventory.folder), archive_input_hashes=input_hashes,
        policy=p, dependencies={"exchange-calendars": version("exchange-calendars"), "pandas": version("pandas")},
        code_hashes=identity["code_hashes"],
        portfolio={"reporting_currency":portfolio.reporting_currency, "holdings":list(portfolio.holdings),
                   "reference_data":list(portfolio.reference_data),
                   "changed_from_archived_snapshot":current_portfolio != archived_portfolio,
                   "current_input":current_portfolio, "archived_input":archived_portfolio},
        price_selection=selection,
        options={"raw_observations":raw_option_count, "selected_quotes":len(chosen_options),
                 "iv_candidates":len(candidates),
                 "european_surface_candidates":sum(r["european_surface_candidate"] for r in candidates),
                 "rejection_reasons":dict(counts), "capture_windows":spans},
        readiness={"missing_holdings":missing, "insufficient_history":insufficient,
                   "aligned_return_observations":len(aligned),
                   "excluded_return_interval_mismatches":mismatched_intervals,
                   "local_history_ready":not insufficient and len(aligned) >= p["prices"]["minimum_return_observations"],
                   "foreign_currencies":required_fx, "available_fx_series":available_fx,
                   "reporting_currency_returns_computed":False,
                   "calibration_performed":False, "simulation_performed":False},
        limitations=limitations)
    # Export layers: provider observations retain flags, selected tables show
    # source decisions, and candidate tables apply additional option screens.
    # option_iv_candidates is not necessarily European-only: consumers must
    # check european_surface_candidate for a European surface workflow.
    # Empty categories still receive headers, so absence of options is supported.
    tables = {
        "provider_prices":_table(prices, ("instrument_id","provider","date","close","adj_close","eligible","reasons")),
        "selected_prices":_table(selected, ("instrument_id","date","analysis_price","return_basis","simple_return","log_return","return_start_date")),
        "aligned_holding_returns":_table(aligned, ("date",*holdings)),
        "price_comparisons":_table(comparisons, ("instrument_id","date","primary_provider","other_provider","relative_difference")),
        "corporate_actions":_table(actions, ("instrument_id","provider","date","event_type","amount","eligible")),
        "option_quotes":_table(options, ("instrument_id","provider","bid","ask","strike","expiration","iv_candidate","iv_reasons")),
        "selected_option_quotes":_table(chosen_options, ("instrument_id","provider","bid","ask","iv_candidate")),
        "option_iv_candidates":_table(candidates, ("instrument_id","provider","bid","ask","strike","expiration",
                                                  "modelGreeks.impliedVol","modelGreeks.undPrice","modelGreeks.pvDividend")),
        "historical_option_bars":_table(bars, ("instrument_id","provider","date","timestamp","option_dataset","close","eligible")),
        "reference_observations":_table(references, ("instrument_id","provider","date","value","rate_decimal","eligible")),
        "selected_references":_table(selected_refs, ("instrument_id","provider","date","value","rate_decimal")),
        "ois_quotes":_table(curves, ("instrument_id","currency","date","tenor","rate_percent","rate_decimal","quote_type")),
        "exchange_sessions":_table(calendar, ("calendar","date","open_utc","close_utc")),
        "request_status":_table(request_status, ("instrument_id","provider","status","has_parsed_observations")),
        "issues":_table(issues, ("instrument_id","provider","code")),
    }
    return ProcessedArchive(inventory.folder.resolve(), inventory.cob, run_id, tables, summary)


def process_archive(folder, cleaning_file, *, portfolio_file=None, progress=None):
    """Read an archive and return its offline processing result.

    This wraps inspect_archive(include_observations=True) and process_inventory.
    No API connections or file exports occur here; call result.save(destination)
    to write the derived tables/report. Existing raw files remain untouched.
    """
    inventory = inspect_archive(folder, progress=progress, include_observations=True)
    return process_inventory(inventory, cleaning_file, portfolio_file=portfolio_file, progress=progress)
