"""Read provider-specific JSON inputs and normalize them for the downloader.

JSON guidance is documentation, never executable code. Capability checks live
here and cannot be changed by editing a product guide's implemented flag.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from .config import parse_requests

SUPPORTED_PRODUCTS = {
    "yahoo": {"stock", "etf", "index", "index_option"},
    "alpha_vantage": {"stock", "etf"},
    "marketdata": {"stock", "etf", "index_option"},
    "massive": {"stock", "etf", "index_option"},
    "fred": {"yield", "fx", "macro"},
    "ecb": {"yield", "fx", "macro"},
}
DATA_TYPES = {"stock": "stock_price", "etf": "etf_price", "index": "index_price",
              "yield": "sovereign_yield", "fx": "fx_reference_rate", "macro": "macro_observation", "index_option": "index_option_history"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Invalid JSON constant: " + value)


def _input_symbol(provider, product, inputs, location):
    fields = {"yahoo": {"symbol"}, "alpha_vantage": {"symbol"}, "marketdata": {"symbol"},
              "massive": {"ticker"}, "fred": {"series_id"}, "ecb": {"dataflow", "series_key"}}[provider]
    if product == "index_option" and provider == "marketdata":
        fields = {"optionSymbol"}
    if not isinstance(inputs, dict) or set(inputs) != fields:
        raise ValueError(location + ": input must contain exactly " + ", ".join(sorted(fields)))
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in inputs.values()):
        raise ValueError(location + ": identifiers must be non-empty text without surrounding whitespace.")
    if provider == "ecb":
        flow, key = inputs["dataflow"], inputs["series_key"]
        if not re.fullmatch(r"[A-Z0-9_]+", flow) or not re.fullmatch(r"[A-Z0-9_]+(?:\.[A-Z0-9_]+)+", key):
            raise ValueError(location + ": use a dataflow and one complete dotted series key; no wildcards.")
        if key.split(".")[0] not in {"D", "B"}:
            raise ValueError(location + ": this ECB adapter supports daily/business-daily keys only.")
        return flow + "/" + key
    symbol = next(iter(inputs.values()))
    if provider == "fred":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", symbol):
            raise ValueError(location + ": FRED input.series_id must be an economic-series identifier.")
    elif not re.fullmatch(r"[A-Za-z0-9^][A-Za-z0-9^._=+:/-]{0,179}", symbol):
        raise ValueError(location + ": invalid provider symbol format.")
    # Obvious contract/index encodings must not be mislabeled as equity prices.
    if product in {"stock", "etf"}:
        if symbol.upper().startswith(("O:", "I:", "^")) or re.fullmatch(r"[A-Z.]+[0-9]{6}[CP][0-9]{8}", symbol.upper()):
            raise ValueError(location + ": this looks like an option/index identifier, not a stock or ETF symbol.")
    return symbol


def _provider_options(provider, parameters, location):
    if not isinstance(parameters, dict):
        raise ValueError(location + ": parameters must be an object.")
    allowed = {"yahoo": {"auto_adjust"}, "alpha_vantage": {"function", "outputsize"},
               "marketdata": {"adjustsplits", "adjustdividends"}, "massive": {"adjusted"},
               "fred": {"units"}, "ecb": set()}[provider]
    if set(parameters) - allowed:
        raise ValueError(location + ": unsupported parameters for " + provider + "; check this provider's product guide.")
    if provider == "alpha_vantage":
        function = parameters.get("function", "TIME_SERIES_DAILY")
        size = parameters.get("outputsize", "compact")
        if function not in ("TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED") or size not in ("compact", "full"):
            raise ValueError(location + ": invalid Alpha Vantage daily-history function/outputsize.")
        return {"ALPHA_VANTAGE_ADJUSTED": str(function.endswith("_ADJUSTED")).lower(),
                "ALPHA_VANTAGE_OUTPUTSIZE": size}
    if provider == "marketdata":
        splits, dividends = parameters.get("adjustsplits", True), parameters.get("adjustdividends", False)
        if splits is not True or not isinstance(dividends, bool):
            raise ValueError(location + ": this adapter requires adjustsplits=true and boolean adjustdividends.")
        return {"MARKETDATA_ADJUSTMENT": "split_and_dividend" if dividends else "split_only"}
    if provider == "yahoo" and parameters.get("auto_adjust", False) is not False:
        raise ValueError(location + ": use auto_adjust=false; the adapter returns close and adjusted close separately.")
    if provider == "massive" and parameters.get("adjusted", True) is not True:
        raise ValueError(location + ": this adapter currently requires adjusted=true (splits only).")
    if provider == "fred" and parameters.get("units", "lin") != "lin":
        raise ValueError(location + ": only native levels (units=lin) are currently supported.")
    return {}


def load_provider_configs(paths) -> dict:
    """Load a JSON file, directory of provider JSONs, or explicit iterable of files.

    Each file contains schema_version, provider, instructions/product_guides and
    requests. Only enabled entries under requests are executed. Files may be
    disabled as a whole. API keys and runtime/storage settings belong in Python.

    Equal IDs across providers combine only when they describe identical logical
    requests. Provider symbols and options can differ. For different timeframes
    or instruments use different IDs. Duplicate IDs within a provider are errors.
    """
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    else:
        paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("No provider JSON files selected.")
    merged, pairs = {}, set()
    root_fields = {"schema_version", "provider", "enabled", "instructions", "product_guides", "sources", "requests"}
    request_fields = {"id", "enabled", "product", "operation", "ticker", "description", "asset_class",
                      "currency", "unit", "exchange", "input", "timeframe", "frequency", "parameters", "notes", "contract"}
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object,
                                  parse_constant=_invalid_constant)
        except (ValueError, OSError) as error:
            # Do not echo file values; users may have accidentally put a key here.
            detail = f"invalid JSON at line {error.lineno}, column {error.colno}" if isinstance(error, json.JSONDecodeError) else type(error).__name__
            raise ValueError(f"{path.name}: could not load provider config ({detail}).") from None
        if not isinstance(document, dict) or set(document) - root_fields:
            raise ValueError(path.name + ": unsupported top-level fields; keep credentials/settings in run.py.")
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError(path.name + ": schema_version must be 1.")
        provider = document.get("provider")
        if not isinstance(provider, str) or provider not in SUPPORTED_PRODUCTS:
            raise ValueError(path.name + ": unknown provider.")
        if not isinstance(document.get("enabled", True), bool):
            raise ValueError(path.name + ": enabled must be a JSON boolean.")
        if not document.get("enabled", True):
            continue
        entries = document.get("requests")
        if not isinstance(entries, list):
            raise ValueError(path.name + ": requests must be a JSON array.")
        for index, entry in enumerate(entries):
            location = f"{path.name}: requests[{index}]"
            if not isinstance(entry, dict) or not isinstance(entry.get("enabled", True), bool):
                raise ValueError(location + ": expected an object with boolean enabled.")
            if not entry.get("enabled", True):
                continue
            if set(entry) - request_fields:
                raise ValueError(location + ": unknown request fields; consult product_guides.")
            product = entry.get("product")
            if not isinstance(product, str) or product not in SUPPORTED_PRODUCTS[provider]:
                raise ValueError(location + ": product is not implemented for " + provider +
                                 ". SPX history is implemented only as index_option for Yahoo, MarketData.app and Massive.")
            kind = "option" if product == "index_option" else "price" if product in {"stock", "etf", "index"} else "observation"
            operation = "option_history" if kind == "option" else "price_history" if kind == "price" else "observations"
            if entry.get("operation") != operation:
                raise ValueError(location + ": product requires operation=" + operation)
            request_id = entry.get("id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(location + ": id is required.")
            if (provider, request_id) in pairs:
                raise ValueError(location + ": duplicate request id within provider: " + request_id)
            pairs.add((provider, request_id))
            symbol = _input_symbol(provider, product, entry.get("input"), location)
            if kind == "option":
                if entry.get("parameters", {}) != {}:
                    raise ValueError(location + ": index_option currently requires parameters={} (no stock adjustments).")
                options = {}
            else:
                options = _provider_options(provider, entry.get("parameters", {}), location)
            normalized = {key: deepcopy(entry[key]) for key in
                          ("id", "ticker", "description", "asset_class", "currency", "unit", "exchange", "timeframe", "frequency", "contract")
                          if key in entry}
            normalized.update(kind=kind, data_type=DATA_TYPES[product], symbols={provider: symbol}, provider_options=options)
            # Run semantic validation on each row before merging or starting I/O.
            try:
                parse_requests({"requests": [normalized]})
            except (ValueError, TypeError) as error:
                raise ValueError(location + ": " + str(error)) from None
            if request_id in merged:
                existing = merged[request_id]
                common = {key: value for key, value in normalized.items() if key not in {"symbols", "provider_options"}}
                prior = {key: value for key, value in existing.items() if key not in {"symbols", "provider_options"}}
                if common != prior:
                    raise ValueError(location + ": id " + request_id + " has different instrument/timeframe details across providers; use a distinct id.")
                existing["symbols"].update(normalized["symbols"])
                existing["provider_options"].update(options)
            else:
                merged[request_id] = normalized
    config = {"requests": list(merged.values())}
    parse_requests(config)
    return config
