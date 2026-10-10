"""Inspect a completed COB archive without APIs, cleaning or source selection.

Step 1 of preprocessing: describe the observations actually saved. Provider
parsers below only read local response bytes. They never call download adapters
or models.normalize(), which would clip, deduplicate or change the data.

The summary selects the recorded attempt. Earlier attempts remain in the archive
but are not concatenated. Repeated pages/chunks inside that attempt are counted
and diagnosed. Field counts describe raw records, not cleaned observations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .storage import write_json

GREEK_FIELDS = ("impliedVol", "optPrice", "undPrice", "delta", "gamma",
                "vega", "theta", "pvDividend")
GREEK_GROUPS = ("bidGreeks", "askGreeks", "lastGreeks", "modelGreeks")
SNAPSHOT_FIELDS = ("bid", "ask", "last", "close", "bidSize", "askSize",
                   "lastSize", "volume")
LIMITS = [
    "Inventory only: no provider selection, filling, clipping, deduplication, returns or calibration.",
    "Field counts include repeated raw rows; unique keys and conflicts are reported separately.",
    "Date gaps are calendar-day diagnostics, not exchange-calendar missing-session checks.",
    "Snapshot timestamps are SDK receipt times, not guaranteed exchange quote times.",
    "Numeric field presence and ordered bid/ask quotes do not establish liquidity or calibration suitability.",
    "All model fields numeric includes finite but questionable values; negative vega is flagged separately.",
    "Fixed OIS quotes retain their real observation date; no bootstrapping or convention inference.",
]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(root, relative):
    """Resolve saved relative paths portably and reject paths outside their owner."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing archive-relative path.")
    path = (root / relative.replace("\\", "/")).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Archive path escapes its containing directory.")
    return path


def _number(value, field="", ibkr=False):
    """Separate missing, malformed and provider-unavailable numeric values."""
    if value is None or (isinstance(value, str) and value.strip().lower() in
                         {"", ".", "nan", "null", "none", "n/a"}):
        return None, "missing"
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"
    if not math.isfinite(number):
        return None, "invalid"
    # IBKR -1 is unavailable for these nonnegative quote fields, but a delta
    # of -1 and negative theta/rates are legitimate numbers.
    leaf = field.rsplit(".", 1)[-1]
    nonnegative = set(SNAPSHOT_FIELDS) | {"impliedVol", "optPrice", "undPrice"}
    if ibkr and (abs(number) >= 1e307 or (leaf in nonnegative and number < 0)):
        return None, "unavailable"
    return number, "numeric"


def _day(value):
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None


def _timestamp(value, zone="America/New_York", milliseconds=False):
    """Epoch arithmetic also supports pre-1970 timestamps on Windows."""
    try:
        if isinstance(value, str) and not value.replace(".", "", 1).lstrip("-").isdigit():
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                return None  # Do not invent a timezone for an intraday observation.
        else:
            number, state = _number(value)
            if state != "numeric":
                return None
            stamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
                seconds=number / (1000 if milliseconds else 1))
        return stamp.astimezone(ZoneInfo(zone)).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass
class ArchiveInventory:
    """Aggregate inventory, with a readable report and a detailed JSON report."""
    folder: Path
    cob: str
    requests: list[dict]
    option_expiries: list[dict]
    metadata: dict

    def to_dict(self):
        return {"schema_version": 1, "archive": str(self.folder), "cob": self.cob,
                "limitations": LIMITS, **self.metadata,
                "requests": self.requests, "option_expiries": self.option_expiries}

    def report(self):
        lines = [
            "# Saved market-data inventory", "",
            f"Archive COB: **{self.cob}**. Requests inspected: **{len(self.requests)}**.",
            "This report describes saved data; it does not certify that it is ready for modelling.", "",
            "## Historical series and fixed curves", "",
            "| Provider | Product | Saved status | Dataset | Raw rows | Unique keys | First | Last |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
        def cell(value):
            return str(value if value is not None else "-").replace("|", "/").replace("\n", " ")
        for item in self.requests:
            if item["is_option"]:
                continue
            for name, table in (item["datasets"].items() or [("-", {})]):
                lines.append("| " + " | ".join(map(cell, [
                    item["provider"], item["ticker"], item["saved_status"], name,
                    table.get("row_count", 0), table.get("unique_keys", 0),
                    table.get("first_observation"), table.get("last_observation")])) + " |")
        lines += ["", "## Option coverage by expiry", "",
                  "Counts are per configured contract, not per response file. Quote and IV counts can overlap.",
                  "", "| Underlying | Expiry | Contracts | Ordered positive bid/ask | Positive model IV | All model fields numeric |",
                  "|---|---|---:|---:|---:|---:|"]
        for group in self.option_expiries:
            lines.append("| " + " | ".join(map(cell, [
                group["underlying"], group["expiration"], group["contracts"],
                group["contracts_with_positive_ordered_quotes"],
                group["contracts_with_positive_model_iv"],
                group["contracts_with_all_numeric_model_fields"]])) + " |")
        lines += ["", "## Field availability", "",
                  "Counts below are numeric/raw rows. The JSON also records missing, malformed and unavailable values.",
                  ""]
        for item in self.requests:
            if item["is_option"]:
                continue
            for name, table in item["datasets"].items():
                fields = ", ".join(f"{key} {value['numeric']}/{table['row_count']}"
                                   for key, value in table["fields"].items())
                lines.append(f"- **{cell(item['provider'])} / {cell(item['ticker'])} / {cell(name)}:** {fields}")
        lines += ["", "## Diagnostics", ""]
        issue_counts = Counter(issue["code"] for item in self.requests for issue in item["issues"])
        for code, count in sorted(issue_counts.items()):
            lines.append(f"- {code}: {count} request(s).")
        if not issue_counts:
            lines.append("- No reader issues detected.")
        lines += ["", "Details for historical/reference requests:", ""]
        for item in self.requests:
            if item["is_option"]:
                continue
            for issue in item["issues"]:
                lines.append(f"- {cell(item['provider'])} / {cell(item['ticker'])}: "
                             f"{cell(issue['code'])} - {cell(issue['detail'])}")
            for name, table in item["datasets"].items():
                if table["duplicate_rows"] or table["conflicting_keys"]:
                    lines.append(f"- {cell(item['provider'])} / {cell(item['ticker'])} / {cell(name)}: "
                                 f"{table['duplicate_rows']} duplicate-key rows; "
                                 f"{table['conflicting_keys']} keys contain conflicting values.")
        lines += ["", "## Interpretation", ""]
        lines += ["- " + text for text in LIMITS]
        lines += ["", "See inventory.json for every request, field, response checksum, price basis, "
                  "actual date range, option capture span and metadata.", ""]
        return "\n".join(lines)

    def save(self, destination):
        """Write derived reports outside the source archive; raw files stay intact."""
        destination = Path(destination).resolve()
        # Also reject an ancestor: reports must not replace files alongside raw inputs.
        if (destination == self.folder or destination.is_relative_to(self.folder)
                or self.folder.is_relative_to(destination)):
            raise ValueError("Inventory reports must be saved outside the source archive and its ancestors.")
        destination.mkdir(parents=True, exist_ok=True)
        json_path, text_path = destination / "inventory.json", destination / "inventory.md"
        write_json(json_path, self.to_dict())
        if text_path.is_symlink():
            raise ValueError("Refusing to write a report through a symbolic link.")
        text_path.write_text(self.report(), encoding="utf-8")
        return json_path, text_path


class _RequestReader:
    def __init__(self, archive, row, config):
        self.archive, self.row = archive, row
        self.tables = defaultdict(list)
        self.meta = {"currency": config.get("currency"), "unit": config.get("unit"),
                     "asset_class": config.get("asset_class"),
                     "data_type": config.get("data_type"),
                     "provider_symbol": config.get("symbols", {}).get(row["provider"]),
                     "requested_frequency": config.get("frequency"),
                     "description": config.get("description"),
                     "contract": config.get("contract")}
        self.bases, self.sources, self.issues = set(), [], []
        self.field_maps = defaultdict(dict)
        self.ibkr = row["provider"] == "ibkr"

    def issue(self, code, detail):
        value = {"code": code, "detail": str(detail)}
        if value not in self.issues:
            self.issues.append(value)

    def add(self, table, key, values, source, *, stamp=None, observed=None):
        self.tables[table].append({"key": key, "date": observed or _day(key),
                                   "timestamp": stamp, "values": values, "source": source})

    def arrays(self, table, timestamps, arrays, source, *, zone="America/New_York",
               milliseconds=False, prefix=""):
        lengths = {len(timestamps), *(len(v) for v in arrays.values())}
        if len(lengths) > 1:
            self.issue("array_length_mismatch", f"{source}: timestamps and value arrays differ in length.")
        self.field_maps[table].update({k: prefix + k for k in arrays})
        for i in range(max(lengths, default=0)):
            stamp = _timestamp(timestamps[i], zone, milliseconds) if i < len(timestamps) else None
            self.add(table, _day(stamp), {k: v[i] if i < len(v) else None for k, v in arrays.items()},
                     source, stamp=stamp)

    def parse(self, body, response, source):
        """Dispatch by saved provider/response schema, never by current config."""
        provider = self.row["provider"]
        params = response.get("parameters", {})
        if provider == "ecb":
            rows = list(csv.DictReader(StringIO(body)))
            if rows and {"TIME_PERIOD", "OBS_VALUE"}.issubset(rows[0]):
                for row in rows:
                    self.add("observations", row["TIME_PERIOD"], {"value": row["OBS_VALUE"]}, source)
                self.meta["series_definition"] = {k: rows[0].get(k) for k in
                    ("KEY", "TITLE", "TITLE_COMPL", "UNIT", "UNIT_MULT", "FREQ")}
                self.meta["observation_status_counts"] = dict(Counter(r.get("OBS_STATUS") for r in rows))
                self.field_maps["observations"]["value"] = "OBS_VALUE"
                return "observations"
            self.issue("unrecognised_response", source + ": no ECB observations.")
            return "unrecognised"
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        if provider == "yahoo" and data.get("chart", {}).get("result"):
            for result in data["chart"]["result"]:
                metadata = result.get("meta", {})
                zone = metadata.get("exchangeTimezoneName", "America/New_York")
                self.meta.update(source_symbol=metadata.get("symbol"),
                                 source_currency=metadata.get("currency"), timezone=zone)
                arrays = dict(result["indicators"]["quote"][0])
                adjusted = result["indicators"].get("adjclose", [])
                if adjusted:
                    arrays["adj_close"] = adjusted[0].get("adjclose", [])
                self.arrays("prices", result.get("timestamp", []), arrays, source, zone=zone)
                self.field_maps["prices"].update({k: "indicators.quote[0]." + k for k in arrays if k != "adj_close"})
                if "adj_close" in arrays:
                    self.field_maps["prices"]["adj_close"] = "indicators.adjclose[0].adjclose"
                index = self.meta.get("data_type") == "index_price"
                self.bases.add("index_level" if index else "split_adjusted_close")
                self.meta["adjusted_close_basis"] = "index_level" if index else "split_and_dividend_adjusted"
                for event, observations in result.get("events", {}).items():
                    for value in observations.values():
                        stamp = _timestamp(value.get("date"), zone)
                        fields = {k: v for k, v in value.items() if k not in {"date", "splitRatio"}}
                        self.add(event, _day(stamp), fields, source, stamp=stamp)
            return "observations"
        if provider == "alpha_vantage" and "Time Series (Daily)" in data:
            mapping = {"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
                       "5. adjusted close": "adj_close", "5. volume": "volume", "6. volume": "volume",
                       "7. dividend amount": "dividends", "8. split coefficient": "stock_splits"}
            for day, fields in data["Time Series (Daily)"].items():
                values = {mapping.get(k, k): v for k, v in fields.items()}
                self.field_maps["prices"].update({mapping.get(k, k): k for k in fields})
                self.add("prices", day, values, source)
            self.bases.add("unadjusted_close")
            self.meta["adjusted_close_basis"] = ("split_and_dividend_adjusted"
                if any("adj_close" in x["values"] for x in self.tables["prices"]) else None)
            return "observations"
        if provider == "marketdata" and data.get("s") == "ok" and "t" in data:
            mapping = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            self.arrays("prices", data["t"], {v: data[k] for k, v in mapping.items() if k in data}, source)
            self.field_maps["prices"].update({v: k for k, v in mapping.items() if k in data})
            splits, dividends = params.get("adjustsplits"), params.get("adjustdividends")
            basis = ("split_and_dividend_adjusted_close" if splits == dividends == "true" else
                     "split_adjusted_close" if splits == "true" and dividends == "false" else
                     "unadjusted_close" if splits == dividends == "false" else "unknown")
            self.bases.add(basis)
            return "observations"
        if provider == "massive" and isinstance(data.get("results"), list):
            mapping = {"o": "open", "h": "high", "l": "low", "c": "close",
                       "v": "volume", "vw": "vwap", "n": "trade_count"}
            for value in data["results"]:
                stamp = _timestamp(value.get("t"), milliseconds=True)
                self.add("prices", _day(stamp), {mapping.get(k, k): v for k, v in value.items() if k != "t"},
                         source, stamp=stamp)
            self.field_maps["prices"].update({v: k for k, v in mapping.items()})
            self.bases.add("index_level" if self.meta.get("data_type") == "index_price" else
                           "split_adjusted_close" if data.get("adjusted") is True else
                           "unadjusted_close" if data.get("adjusted") is False else "unknown")
            return "observations"
        if provider == "fred" and "observations" in data:
            for value in data["observations"]:
                self.add("observations", value.get("date"), {"value": value.get("value")}, source)
            self.meta["vintage_start"] = data.get("realtime_start")
            self.meta["vintage_end"] = data.get("realtime_end")
            self.field_maps["observations"]["value"] = "observations[].value"
            return "observations"
        if provider == "fred" and "seriess" in data:
            self.meta["series_definition"] = [{k: series.get(k) for k in
                ("id", "title", "units", "frequency", "seasonal_adjustment")} for series in data["seriess"]]
            return "metadata"
        if provider == "ibkr" and "bars" in data:
            request = data.get("request", {})
            table = request.get("what_to_show", "UNKNOWN")
            for bar in data["bars"]:
                raw_date = bar.get("date")
                stamp = None if isinstance(raw_date, str) and len(raw_date) == 10 else _timestamp(raw_date, "UTC")
                key = raw_date if isinstance(raw_date, str) and len(raw_date) == 10 else stamp
                self.add(table, key, {k: v for k, v in bar.items() if k != "date"}, source, stamp=stamp)
            self.meta.setdefault("historical_channels", {})[table] = {
                k: request.get(k) for k in ("bar_size", "use_rth", "what_to_show")}
            if data.get("errors"):
                self.issue("saved_provider_errors", source + ": historical callback errors are recorded.")
            if not data.get("completed", False):
                self.issue("incomplete_response", source)
            self.bases.add("index_level" if self.meta.get("data_type") == "index_price" else
                           "split_adjusted_close" if table == "TRADES" else
                           "split_and_dividend_adjusted_close" if table == "ADJUSTED_LAST" else table)
            return "observations"
        if provider == "ibkr" and isinstance(data.get("ticker"), dict):
            ticker = data["ticker"]
            stamp = _timestamp(ticker.get("time"), "UTC")
            values = {k: ticker.get(k) for k in SNAPSHOT_FIELDS}
            for group in GREEK_GROUPS:
                values.update({group + "." + k: (ticker.get(group) or {}).get(k) for k in GREEK_FIELDS})
            self.add("snapshot", stamp, values, source, stamp=stamp)
            self.field_maps["snapshot"] = {k: "ticker." + k for k in values}
            self.meta["market_data_type"] = ticker.get("marketDataType")
            self.meta["snapshot_completed"] = data.get("snapshot_completed")
            self.meta["resolved_contract"] = {k: data.get("resolved_contract", {}).get(k) for k in
                ("conId", "symbol", "localSymbol", "strike", "right", "multiplier",
                 "lastTradeDateOrContractMonth", "tradingClass", "exchange", "currency")}
            details = data.get("contract_details") or []
            if details:
                self.meta["contract_dates"] = {k: details[0].get(k) for k in
                    ("realExpirationDate", "lastTradeTime", "timeZoneId", "tradingHours", "liquidHours")}
            self.meta["model_received_at_utc"] = self.row.get("model_received_at_utc")
            self.meta["iv_units"] = "annualized_decimal; 0.20 means 20%, as supplied by IBKR"
            self.bases.add("snapshot_quotes_and_separate_model_values")
            return "observations"
        if provider == "ibkr" and data.get("method") in {"reqContractDetails", "reqHeadTimeStamp"}:
            return "metadata"
        if data.get("capture_format") == "serialized_sdk_contract_details":
            return "metadata"
        if any(k in data for k in ("Information", "Note", "Error Message", "error", "message", "errmsg")) or data.get("s") == "no_data":
            return "diagnostic"
        self.issue("unrecognised_response", source + ": unsupported response schema.")
        return "unrecognised"

    def profile(self, name, rows):
        fields = sorted({field for row in rows for field in row["values"]})
        stats = {field: dict(present=0, numeric=0, missing=0, invalid=0, unavailable=0,
                             zero=0, negative=0, first_numeric_date=None, last_numeric_date=None)
                 for field in fields}
        versions = defaultdict(set)
        dates, timestamps = [], []
        quality = Counter()
        invalid_dates = before = after = after_cob = 0
        for row in rows:
            day, stamp = row["date"], row["timestamp"]
            if day:
                dates.append(day)
                before += bool(self.row.get("requested_start") and day < self.row["requested_start"])
                after += bool(self.row.get("requested_end") and day > self.row["requested_end"])
                after_cob += bool(self.row.get("cob") and day > self.row["cob"])
            else:
                invalid_dates += 1
            if stamp:
                timestamps.append(stamp)
            numeric, comparable = {}, {}
            for field in fields:
                stat = stats[field]
                stat["present"] += field in row["values"]
                value, state = _number(row["values"].get(field), field, self.ibkr)
                stat[state] += 1
                numeric[field] = value
                comparable[field] = value if state in {"numeric", "missing"} else [state, str(row["values"].get(field))]
                if state == "numeric":
                    stat["zero"] += value == 0
                    stat["negative"] += value < 0
                    if day:
                        stat["first_numeric_date"] = min(stat["first_numeric_date"] or day, day)
                        stat["last_numeric_date"] = max(stat["last_numeric_date"] or day, day)
            if row["key"] is not None:
                versions[str(row["key"])].add(json.dumps(comparable, sort_keys=True))
            if numeric.get("close") is not None and numeric["close"] <= 0:
                quality["nonpositive_close"] += 1
            if numeric.get("volume") is not None and numeric["volume"] < 0:
                quality["negative_volume"] += 1
            if all(numeric.get(k) is not None for k in ("open", "high", "low", "close")):
                if numeric["high"] < max(numeric[k] for k in ("open", "low", "close")) - 1e-6 or numeric["low"] > min(numeric[k] for k in ("open", "high", "close")) + 1e-6:
                    quality["inconsistent_ohlc"] += 1
            if name == "snapshot":
                bid, ask = numeric.get("bid"), numeric.get("ask")
                if bid is not None and ask is not None:
                    if 0 < bid <= ask:
                        quality["positive_ordered_bid_ask"] += 1
                    if bid > ask:
                        quality["crossed_bid_ask"] += 1
                    if bid == 0:
                        quality["zero_bid"] += 1
                iv = numeric.get("modelGreeks.impliedVol")
                if iv is not None and iv > 0:
                    quality["positive_model_iv"] += 1
                if all(numeric.get("modelGreeks." + k) is not None for k in GREEK_FIELDS):
                    quality["all_numeric_model_fields"] += 1
        unique_days = sorted(set(dates))
        gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(unique_days, unique_days[1:])]
        duplicate_rows = sum(row["key"] is not None for row in rows) - len(versions)
        return {
            "row_count": len(rows), "unique_keys": len(versions), "unique_dates": len(unique_days),
            "duplicate_rows": duplicate_rows,
            "conflicting_keys": sum(len(values) > 1 for values in versions.values()),
            "invalid_date_rows": invalid_dates,
            "first_observation": min(dates, default=None), "last_observation": max(dates, default=None),
            "first_timestamp": min(timestamps, default=None), "last_timestamp": max(timestamps, default=None),
            "rows_before_requested_start": before, "rows_after_requested_end": after,
            "rows_after_archive_cob": after_cob,
            "largest_gap_calendar_days": max(gaps, default=None),
            "fields": stats, "source_fields": self.field_maps[name], "quality_counts": dict(quality),
        }

    def read(self):
        row = self.row
        folder = _inside(self.archive, row["folder"])
        if row["provider"] == "manual_ois":
            curve_path = folder / "curve.json"
            body = curve_path.read_bytes()
            payload = json.loads(body.decode("utf-8-sig"))
            curve = payload["curve"]
            self.meta.update({k: curve.get(k) for k in
                              ("currency", "curve_id", "overnight_index", "quote_type",
                               "rate_unit", "as_of_date", "conventions")})
            self.meta["usage_policy"] = payload.get("usage_policy")
            for point in curve["points"]:
                self.add("par_ois_quotes", point["tenor"], {"rate_percent": point["rate_percent"]},
                         str(curve_path.relative_to(self.archive)), observed=_day(curve.get("as_of_date")))
            self.sources.append({"file": curve_path.relative_to(self.archive).as_posix(),
                                 "sha256": hashlib.sha256(body).hexdigest(), "role": "observations"})
            self.bases.add("par_swap_quote_not_zero_rate")
        else:
            attempt = _inside(self.archive, row["attempt"]) if row.get("attempt") else folder
            if attempt != folder and not attempt.is_relative_to(folder):
                raise ValueError("Selected attempt is outside its request folder.")
            request_path = attempt / "request.json"
            if not request_path.exists() and attempt != folder:
                request_path = folder / "request.json"  # Legacy first-attempt metadata.
            request = _json(request_path)
            for source, target in {
                "configured_currency": "currency", "configured_unit": "unit",
                "configured_exchange": "exchange", "asset_class": "asset_class",
                "data_type": "data_type", "requested_frequency": "requested_frequency",
                "instrument_description": "description", "contract": "contract",
            }.items():
                if source in request:
                    self.meta[target] = request[source]
            for key in ("requested_start", "requested_end"):
                if key in request:
                    row[key] = request[key]
            index_path = attempt / "responses.json"
            if not index_path.exists():
                if row.get("responses_saved", 0):
                    self.issue("missing_response_index", str(index_path.relative_to(self.archive)))
                responses = []
            else:
                responses = _json(index_path)
                if not isinstance(responses, list):
                    raise ValueError("Response index must contain a list.")
            seen = set()
            for response in responses:
                path = _inside(attempt, response["file"])
                source = path.relative_to(self.archive).as_posix()
                if source in seen:
                    self.issue("duplicate_response_index_entry", source)
                    continue
                seen.add(source)
                info = {"file": source, "received_at_utc": response.get("received_at_utc"),
                        "http_status": response.get("http_status")}
                self.sources.append(info)
                try:
                    body = path.read_bytes()
                    digest = hashlib.sha256(body).hexdigest()
                    info["sha256"] = digest
                    expected = response.get("sha256")
                    info["checksum_matches"] = digest == expected if expected else None
                    if expected and digest != expected:
                        self.issue("checksum_mismatch", source)
                        info["role"] = "untrusted"
                        continue
                    if not expected:
                        self.issue("missing_checksum", source)
                    info["role"] = self.parse(body.decode("utf-8-sig"), response, source)
                # Nested JSON values can be null/scalars where a record was expected.
                # Isolate that response so later pages and requests still get inspected.
                except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                    info["role"] = "unreadable"
                    self.issue("response_read_error", source + ": " + type(exc).__name__)
            if len(responses) != row.get("responses_saved", len(responses)):
                self.issue("response_count_mismatch", "Summary and saved response index disagree.")
        return self.finish()

    def finish(self):
        datasets = {name: self.profile(name, rows) for name, rows in self.tables.items()}
        if not datasets:
            self.issue("no_observations", "No recognised observations were read from the selected attempt.")
        if self.row.get("status") != "response_saved":
            self.issue("retrieval_status", self.row.get("status", "unknown"))
        if any(table["invalid_date_rows"] for table in datasets.values()):
            self.issue("invalid_observation_dates", "Some records have missing or invalid dates.")
        if any(table["rows_after_archive_cob"] for table in datasets.values()):
            self.issue("observations_after_cob", "Actual observations are later than the archive label.")
        if any(table["rows_before_requested_start"] or table["rows_after_requested_end"] for table in datasets.values()):
            self.issue("observations_outside_requested_window", "Raw observations retained; no clipping applied.")
        if len(self.bases) > 1:
            self.issue("multiple_price_bases", "Recorded responses use more than one price convention.")
        if datasets.get("snapshot", {}).get("fields", {}).get("modelGreeks.vega", {}).get("negative", 0):
            self.issue("negative_model_vega", "Numeric vega is negative; review possible rounding or model issues before use.")
        if "snapshot" in datasets:
            self.meta["timestamp_meaning"] = "SDK receipt time; individual quote ages are not established"
        return {
            "provider": self.row["provider"], "request_id": self.row["request_id"],
            "ticker": self.row["ticker"], "saved_status": self.row.get("status"),
            "operation": self.row.get("operation"),
            "is_option": "option" in str(self.row.get("operation", "")),
            "selected_attempt": self.row.get("attempt"),
            "requested_start": self.row.get("requested_start"),
            "requested_end": self.row.get("requested_end"),
            "metadata": self.meta, "price_bases": sorted(self.bases),
            "datasets": datasets, "response_files": self.sources, "issues": self.issues,
        }


def _option_coverage(requests):
    groups = {}
    for item in requests:
        if not item["is_option"]:
            continue
        contract = item["metadata"].get("contract") or {}
        key = (item["provider"], contract.get("underlying", "unknown"), contract.get("expiration", "unknown"))
        group = groups.setdefault(key, {
            "provider": key[0], "underlying": key[1], "expiration": key[2],
            "contracts": 0, "contracts_with_positive_ordered_quotes": 0,
            "contracts_with_positive_model_iv": 0, "contracts_with_all_numeric_model_fields": 0,
            "contracts_with_observations": 0, "contracts_with_negative_model_vega": 0, "first_capture_utc": None, "last_capture_utc": None,
            "strikes": [], "saved_status_counts": {}, "field_numeric_contract_counts": {},
        })
        group["contracts"] += 1
        group["contracts_with_observations"] += any(t["row_count"] for t in item["datasets"].values())
        status = item["saved_status"]
        group["saved_status_counts"][status] = group["saved_status_counts"].get(status, 0) + 1
        strike, state = _number(contract.get("strike"))
        if state == "numeric":
            group["strikes"].append(strike)
        table = item["datasets"].get("snapshot", {})
        counts = table.get("quality_counts", {})
        group["contracts_with_negative_model_vega"] += table.get("fields", {}).get("modelGreeks.vega", {}).get("negative", 0) > 0
        for target, source in [
            ("contracts_with_positive_ordered_quotes", "positive_ordered_bid_ask"),
            ("contracts_with_positive_model_iv", "positive_model_iv"),
            ("contracts_with_all_numeric_model_fields", "all_numeric_model_fields"),
        ]:
            group[target] += counts.get(source, 0) > 0
        for field, stat in table.get("fields", {}).items():
            current = group["field_numeric_contract_counts"].get(field, 0)
            group["field_numeric_contract_counts"][field] = current + (stat["numeric"] > 0)
        for source, target, fn in [
            ("first_timestamp", "first_capture_utc", min), ("last_timestamp", "last_capture_utc", max)
        ]:
            stamp = table.get(source)
            if stamp:
                group[target] = fn(group[target] or stamp, stamp)
    for group in groups.values():
        group["strikes"] = sorted(set(group["strikes"]))
    return sorted(groups.values(), key=lambda g: (g["provider"], g["underlying"], g["expiration"]))


def inspect_archive(folder, *, progress: Callable[[str], None] | None = None, max_workers=4):
    """Read all recorded requests in a completed COB archive; never request data.

    Corrupt/missing response files become explicit inventory issues. An absent,
    unfinished or actively written archive is rejected before inspecting data.
    Configuration is read from the archive, not from today's config/*.json.
    """
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be an integer between 1 and 8.")
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"No saved archive at {folder}. Inventory mode never downloads data.")
    if (folder / ".running").exists():
        raise ValueError("The archive is being written. Wait for retrieval to finish.")
    try:
        manifest_bytes = (folder / "manifest.json").read_bytes()
        summary_bytes = (folder / "summary.json").read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        summary = json.loads(summary_bytes.decode("utf-8-sig"))
        inputs = _json(folder / "resolved_inputs.json")
    except (OSError, ValueError) as exc:
        raise ValueError("Archive metadata is missing or unreadable.") from exc
    if manifest.get("complete") is not True:
        raise ValueError("Inventory requires a completed archive.")
    if not isinstance(summary, list) or any(not isinstance(row, dict) for row in summary):
        raise ValueError("Archive summary must be a list of request results.")
    identities = [(row.get("provider"), row.get("request_id")) for row in summary]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate provider/request identities in archive summary.")
    if manifest.get("request_count") != len(summary):
        raise ValueError("Manifest request count disagrees with summary.")
    config = {value.get("id"): value for value in inputs.get("requests", [])}
    results = []
    emit = progress or (lambda _: None)
    emit(f"Inventory: reading {len(summary):,} saved requests for COB {manifest.get('cob')}.")
    def read_one(original):
        row = dict(original)
        row.setdefault("cob", manifest.get("cob"))
        reader = _RequestReader(folder, row, config.get(row.get("request_id"), {}))
        try:
            result = reader.read()
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            reader.issue("request_read_error", type(exc).__name__ + ": " + str(exc))
            result = reader.finish()
        return result

    # Only local file reads run concurrently. map preserves the saved request order.
    # The worker bound limits file handles and disk load; no provider code is called.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for index, result in enumerate(pool.map(read_one, summary), 1):
            results.append(result)
            if index % 250 == 0 or index == len(summary):
                emit(f"Inventory: inspected {index:,}/{len(summary):,} requests.")
    if ((folder / ".running").exists()
            or (folder / "manifest.json").read_bytes() != manifest_bytes
            or (folder / "summary.json").read_bytes() != summary_bytes):
        raise ValueError("Archive changed during inspection; discard this inventory and rerun when idle.")
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "archive_execution_date": manifest.get("execution_date"),
        "archive_all_requests_succeeded": manifest.get("all_requests_succeeded"),
        "request_count": len(results),
        "saved_status_counts": dict(Counter(row.get("status", "unknown") for row in summary)),
        "requests_with_reader_errors": sum(any(issue["code"] in {
            "request_read_error", "response_read_error", "checksum_mismatch",
            "missing_response_index", "response_count_mismatch", "unrecognised_response"
        } for issue in item["issues"]) for item in results),
        "dataset_start_dates": sorted({table["first_observation"] for item in results
                                  for table in item["datasets"].values() if table["first_observation"]}),
    }
    return ArchiveInventory(folder, manifest["cob"], results, _option_coverage(results), meta)

