"""Provider registry, options, credentials and instrument mappings in Python."""
import os
from dataclasses import dataclass
from datetime import date
import re

from .models import Instrument, DateWindow
from .options import RAW_OPTION_PROVIDERS, validate_contract
from .providers.alpha_vantage import AlphaVantageProvider
from .providers.ecb import ECBProvider
from .providers.fred import FredProvider
from .providers.marketdata import MarketDataAppProvider
from .providers.massive import MassiveProvider
from .providers.yahoo import YahooProvider
from .providers.ibkr import IBKRProvider, validate_ib_input

PROVIDERS = {
    "yahoo": YahooProvider, "alpha_vantage": AlphaVantageProvider,
    "marketdata": MarketDataAppProvider, "massive": MassiveProvider,
    "fred": FredProvider, "ecb": ECBProvider, "ibkr": IBKRProvider,
}
# Credentials can be passed by run.py or read from these environment variables.
# Values are never printed by the library. Account entitlements are verified only
# when the provider responds to a real request, not by the presence of a key.
DEFAULT_SETTINGS = {
    "ALPHA_VANTAGE_API_KEY": "",
    "MARKETDATA_API_TOKEN": "",
    "MASSIVE_API_KEY": "",
    "FRED_API_KEY": "",
    "ALPHA_VANTAGE_OUTPUTSIZE": "compact",
    "ALPHA_VANTAGE_ADJUSTED": "false",
    "MARKETDATA_ADJUSTMENT": "split_only",
    "YAHOO_MIN_INTERVAL": "1",
    "ALPHA_VANTAGE_MIN_INTERVAL": "13",
    "MARKETDATA_MIN_INTERVAL": "1",
    "MASSIVE_MIN_INTERVAL": "13",
    "FRED_MIN_INTERVAL": "0.6",
    "ECB_MIN_INTERVAL": "0.3",
    "IBKR_HOST": "127.0.0.1",
    "IBKR_PORT": "7497",
    "IBKR_CLIENT_ID": "81",
    "IBKR_TIMEOUT": "20",
    "IBKR_MIN_INTERVAL": "1",
    "IBKR_LIVE_EXPECTED_DATE": "",
    "IBKR_OPTION_MAX_CONCURRENT": "5",  # One TWS session; at most five option snapshots.
    "IBKR_OPTION_MARKET_DATA_TYPE": "1",  # Current live snapshot mode.
    "IBKR_OPTION_CAPTURE_SECONDS": "60",
    "IBKR_HISTORY_MAX_CONCURRENT": "5",
    "IBKR_HISTORY_REQUEST_INTERVAL": "0.25",
    "IBKR_HISTORY_TIMEOUT": "60",
    "IBKR_HISTORY_DATA_TYPES": '["TRADES","BID","ASK","MIDPOINT"]',
    "IBKR_HISTORY_CUTOFF": "16:00:00",
    "IBKR_HISTORY_TIMEZONE": "America/New_York",
}


def provider_settings(overrides=None):
    """Python defaults < named environment variables < explicit Python overrides."""
    return {**DEFAULT_SETTINGS,
            **{name: os.environ[name] for name in DEFAULT_SETTINGS if name in os.environ},
            **dict(overrides or {})}


# Logical instruments and their provider-specific identifiers. Edit or override
# these Python mappings to add other listings. null/None disables a provider.
DEFAULT_INSTRUMENTS = {'SPY': {'kind': 'price',
         'currency': 'USD',
         'description': 'US-listed SPDR S&P 500 ETF Trust',
         'symbols': {'yahoo': 'SPY',
                     'alpha_vantage': 'SPY',
                     'marketdata': 'SPY',
                     'massive': 'SPY'}},
 'QQQ': {'kind': 'price',
         'currency': 'USD',
         'description': 'US-listed Invesco QQQ Trust',
         'symbols': {'yahoo': 'QQQ',
                     'alpha_vantage': 'QQQ',
                     'marketdata': 'QQQ',
                     'massive': 'QQQ'}},
 'NVDA': {'kind': 'price',
          'currency': 'USD',
          'description': 'US-listed NVIDIA common stock',
          'symbols': {'yahoo': 'NVDA',
                      'alpha_vantage': 'NVDA',
                      'marketdata': 'NVDA',
                      'massive': 'NVDA'}},
 'AAPL': {'kind': 'price',
          'currency': 'USD',
          'description': 'US-listed Apple common stock',
          'symbols': {'yahoo': 'AAPL',
                      'alpha_vantage': 'AAPL',
                      'marketdata': 'AAPL',
                      'massive': 'AAPL'}},
 'IEF': {'kind': 'price',
         'currency': 'USD',
         'description': 'US-listed iShares 7-10 Year Treasury Bond ETF',
         'symbols': {'yahoo': 'IEF',
                     'alpha_vantage': 'IEF',
                     'marketdata': 'IEF',
                     'massive': 'IEF'}},
 'US10Y': {'kind': 'observation',
           'unit': 'Percent',
           'description': 'US 10-year constant-maturity Treasury yield, not a bond price',
           'symbols': {'fred': 'DGS10', 'ecb': None}},
 'EUR10Y': {'kind': 'observation',
            'currency': 'EUR',
            'unit': 'Percent per annum',
            'description': 'ECB euro-area AAA 10-year spot yield (continuously compounded); not a '
                           'German benchmark bond price',
            'symbols': {'ecb': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y', 'fred': None}},
 'EURUSD': {'kind': 'observation',
            'unit': 'USD per EUR',
            'description': 'ECB reference exchange rate, USD per EUR',
            'symbols': {'ecb': 'EXR/D.USD.EUR.SP00.A', 'fred': None}}}


def split_inputs(text: str) -> list[str]:
    return list(dict.fromkeys(x.upper() for x in re.split(r"[\s,;]+", text.strip()) if x))


def load_instruments(tickers: list[str], overrides: dict | None = None) -> list[Instrument]:
    """Resolve tickers from Python mappings, with optional per-instrument overrides."""
    catalog = {**DEFAULT_INSTRUMENTS, **(overrides or {})}
    result = []
    for ticker in tickers:
        if not re.fullmatch(r"[A-Z0-9^][A-Z0-9^._:/=+\-]{0,179}", ticker):
            raise ValueError(f"Invalid ticker/series identifier: {ticker!r}")
        if ticker in catalog:
            entry = catalog[ticker]
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid mapping for {ticker}.")
            kind = entry.get("kind", "price")
            symbols = entry.get("symbols", {})
            if kind not in {"price", "observation", "option"} or not isinstance(symbols, dict):
                raise ValueError(f"Invalid kind/symbols for {ticker}.")
            for provider, symbol in symbols.items():
                if provider not in PROVIDERS or (symbol is not None and (not isinstance(symbol, str) or not symbol.strip())):
                    raise ValueError(f"Invalid provider-symbol mapping for {ticker}.")
            result.append(Instrument(ticker, kind, symbols, entry.get("currency"), entry.get("unit"), entry.get("description", ""), True, entry.get("provider_inputs", {})))
        elif ticker.startswith("FRED:"):
            if not ticker[5:]:
                raise ValueError("FRED: requires a series identifier.")
            result.append(Instrument(ticker, "observation", {"fred": ticker[5:]}, mapped=True))
        elif ticker.startswith("ECB:"):
            if not ticker[4:]:
                raise ValueError("ECB: requires a dataflow and series key.")
            result.append(Instrument(ticker, "observation", {"ecb": ticker[4:]}, mapped=True))
        else:
            result.append(Instrument(ticker))
    if not result:
        raise ValueError("Enter at least one ticker or macro-series identifier.")
    return result


@dataclass(frozen=True)
class ConfiguredRequest:
    request_id: str
    instrument: Instrument
    providers: tuple[str, ...]
    window: DateWindow | None
    frequency: str
    data_type: str
    asset_class: str
    exchange: str | None
    provider_options: dict
    contract: dict | None = None
    operation: str = ""

    def metadata(self):
        return {"request_id": self.request_id, "operation": self.operation, "data_type": self.data_type,
                "asset_class": self.asset_class, "configured_exchange": self.exchange,
                "configured_currency": self.instrument.currency,
                "configured_unit": self.instrument.unit,
                "instrument_description": self.instrument.description,
                "requested_frequency": self.frequency, "contract": self.contract}


def parse_requests(config: dict) -> list[ConfiguredRequest]:
    """Validate the entire retrieval config before any network call or export.

    Each request has an explicit symbol per selected provider. No symbol guessing
    or implicit fallback to the built-in ticker catalog occurs in this workflow.
    Time windows are recalculated when download() calls this function.
    """
    if not isinstance(config, dict):
        raise ValueError("Retrieval config must be a Python dictionary.")
    extra = set(config) - {"settings", "cache_dir", "cache_hours", "output_dir", "requests"}
    if extra:
        raise ValueError("Unknown config fields: " + ", ".join(sorted(extra)))
    if not isinstance(config.get("settings", {}), dict):
        raise ValueError("config.settings must be a dictionary.")
    unknown_settings = set(config.get("settings", {})) - set(DEFAULT_SETTINGS)
    if unknown_settings:
        raise ValueError("Unknown provider settings: " + ", ".join(sorted(unknown_settings)))
    entries = config.get("requests")
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("config.requests must be a non-empty list of requests.")
    allowed = {"id", "ticker", "kind", "data_type", "asset_class", "description",
               "currency", "unit", "exchange", "symbols", "timeframe", "frequency",
               "provider_options", "enabled", "contract", "provider_inputs", "operation"}
    option_values = {"ALPHA_VANTAGE_OUTPUTSIZE": {"compact", "full"},
                     "ALPHA_VANTAGE_ADJUSTED": {"true", "false"},
                     "MARKETDATA_ADJUSTMENT": {"split_only", "split_and_dividend"}}
    for key, valid in option_values.items():
        if key in config.get("settings", {}) and config["settings"][key] not in valid:
            raise ValueError("Invalid value for " + key)
    result, ids = [], set()
    today = date.today()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("Each request must be a dictionary using the documented fields.")
        if not isinstance(entry.get("enabled", True), bool):
            raise ValueError("Request enabled must be True or False.")
        if not entry.get("enabled", True):
            continue
        request_id = entry.get("id")
        ticker = entry.get("ticker")
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", request_id):
            raise ValueError("Every request needs a unique id using letters, digits, dots, hyphens or underscores.")
        if request_id in ids:
            raise ValueError("Duplicate request id: " + request_id)
        ids.add(request_id)
        if not isinstance(ticker, str) or len(split_inputs(ticker)) != 1:
            raise ValueError(request_id + ": provide exactly one logical ticker.")
        ticker = ticker.upper()
        kind, symbols = entry.get("kind"), entry.get("symbols")
        if kind not in {"price", "observation", "option"} or not isinstance(symbols, dict) or not symbols:
            raise ValueError(request_id + ": kind and non-empty provider symbols are required.")
        names = []
        for name, symbol in symbols.items():
            if name not in PROVIDERS or (symbol is not None and (not isinstance(symbol, str) or not symbol.strip())):
                raise ValueError(request_id + ": invalid provider/symbol mapping.")
            if symbol is not None:
                if not (kind == "option" and name in RAW_OPTION_PROVIDERS) and PROVIDERS[name].kind != kind:
                    raise ValueError(request_id + ": provider does not support this data kind: " + name)
                names.append(name)
        if not names:
            raise ValueError(request_id + ": enable at least one provider symbol.")
        for key in ("description", "data_type", "asset_class", "currency", "unit", "exchange"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                raise ValueError(request_id + ": " + key + " must be text or None.")
        contract = validate_contract(entry.get("contract"), symbols, entry.get("currency")) if kind == "option" else None
        provider_inputs = entry.get("provider_inputs", {})
        if not isinstance(provider_inputs, dict) or set(provider_inputs) - {"ibkr"}:
            raise ValueError(request_id + ": provider_inputs supports IBKR contracts only.")
        if "ibkr" in names:
            native_symbol = validate_ib_input(provider_inputs.get("ibkr"), kind, entry.get("currency"), contract)
            if native_symbol != symbols["ibkr"]:
                raise ValueError(request_id + ": IBKR symbol does not match its structured input.")
        elif provider_inputs:
            raise ValueError(request_id + ": IBKR input has no enabled IBKR symbol.")
        if kind != "option" and entry.get("contract") is not None:
            raise ValueError(request_id + ": contract belongs only to option requests.")
        if kind in {"price", "option"} and not entry.get("currency"):
            raise ValueError(request_id + ": specify the quote currency for prices.")
        if kind == "observation" and not entry.get("unit"):
            raise ValueError(request_id + ": specify the observation units.")
        snapshot = entry.get("operation") == "option_model_snapshot"
        operation = entry.get("operation", "option_history" if kind == "option" else "price_history" if kind == "price" else "observations")
        expected_operation = operation if kind == "option" and names == ["ibkr"] and operation in {"option_history", "option_model_snapshot"} else "option_history" if kind == "option" else "price_history" if kind == "price" else "observations"
        if operation != expected_operation or (snapshot and (kind != "option" or names != ["ibkr"])):
            raise ValueError(request_id + ": IBKR options require a separate option_model_snapshot request; other providers retain history operations.")
        frequency = entry.get("frequency")
        if snapshot:
            if frequency != "snapshot" or entry.get("timeframe") is not None:
                raise ValueError(request_id + ": option model snapshots require frequency=snapshot and no historical timeframe.")
            window = None
        else:
            ib_history = kind == "option" and names == ["ibkr"] and operation == "option_history"
            if frequency not in ({"1min"} if ib_history else {"1d", "native"}):
                raise ValueError(request_id + ": frequency must be 1d or native; intraday/resampling is not implemented.")
            if "fred" in names and frequency != "native":
                raise ValueError(request_id + ": use native for FRED; the returned metadata describes the series frequency.")
            if kind in {"price", "option"} and not ib_history and frequency != "1d":
                raise ValueError(request_id + ": price adapters support daily data only.")
            timeframe = entry.get("timeframe")
            if not isinstance(timeframe, dict) or set(timeframe) - {"lookback", "start", "end"}:
                raise ValueError(request_id + ": timeframe must contain lookback OR start, and optional end.")
            lookback, start = timeframe.get("lookback"), timeframe.get("start")
            if (lookback is None) == (start is None):
                raise ValueError(request_id + ": choose exactly one of lookback and start.")
            end = timeframe.get("end") or today
            if isinstance(end, str):
                end = date.fromisoformat(end)
            if not isinstance(end, date):
                raise ValueError(request_id + ": end must be an ISO date, date object or None.")
            if start is not None:
                start = date.fromisoformat(start) if isinstance(start, str) else start
                if not isinstance(start, date):
                    raise ValueError(request_id + ": start must be an ISO date or date object.")
                window = DateWindow(start, end)
            else:
                if not isinstance(lookback, str) or not re.fullmatch(r"[1-9][0-9]*(y|mo|m|d|w)", lookback):
                    raise ValueError(request_id + ": use a lookback like 1y, 6mo or 30d.")
                window = DateWindow.parse(lookback, end)
        options = entry.get("provider_options", {})
        if snapshot:
            permitted = {"IBKR_OPTION_MARKET_DATA_TYPE", "IBKR_OPTION_CAPTURE_SECONDS"}
            if not isinstance(options, dict) or set(options) - permitted:
                raise ValueError(request_id + ": snapshots support only IBKR option capture settings.")
            effective = {**DEFAULT_SETTINGS, **config.get("settings", {}), **options}
            if effective["IBKR_OPTION_MARKET_DATA_TYPE"] not in {"1", "2", "3", "4"}:
                raise ValueError(request_id + ": request live (1), frozen (2), delayed (3) or delayed-frozen (4) option data.")
            try:
                seconds = float(effective["IBKR_OPTION_CAPTURE_SECONDS"])
                if not 1 <= seconds <= 120: raise ValueError
            except (ValueError, TypeError):
                raise ValueError(request_id + ": option capture seconds must be between 1 and 120.") from None
        elif kind == 'option' and names == ['ibkr'] and operation == 'option_history':
            import json
            from datetime import time as day_time
            from zoneinfo import ZoneInfo
            if not isinstance(options, dict) or set(options) - {'IBKR_HISTORY_DATA_TYPES','IBKR_HISTORY_CUTOFF','IBKR_HISTORY_TIMEZONE'}:
                raise ValueError(request_id + ': invalid historical option parameters.')
            effective = {**DEFAULT_SETTINGS, **options}
            types = json.loads(effective['IBKR_HISTORY_DATA_TYPES'])
            if not isinstance(types,list) or not types or len(types)!=len(set(types)) or set(types)-{'TRADES','BID','ASK','MIDPOINT'}:
                raise ValueError(request_id + ': invalid historical option data types.')
            day_time.fromisoformat(effective['IBKR_HISTORY_CUTOFF'])
            ZoneInfo(effective['IBKR_HISTORY_TIMEZONE'])
            if window.start != window.end:
                raise ValueError(request_id + ': option history requires a single COB date.')
        else:
            if not isinstance(options, dict) or set(options) - set(option_values):
                raise ValueError(request_id + ": provider_options supports adjustment/history options only; credentials belong in settings.")
            if any(value not in option_values[key] for key, value in options.items()):
                raise ValueError(request_id + ": invalid adjustment/history option.")
            if kind == "option" and options:
                raise ValueError(request_id + ": stock provider_options cannot be applied to index options.")
        instrument = load_instruments([ticker], {ticker: {
            "kind": kind, "symbols": symbols, "currency": entry.get("currency"),
            "unit": entry.get("unit"), "description": entry.get("description", ""), "provider_inputs": provider_inputs,
        }})[0]
        result.append(ConfiguredRequest(request_id, instrument, tuple(names), window, frequency,
                                        entry.get("data_type", kind), entry.get("asset_class", ""),
                                        entry.get("exchange"), dict(options), contract, operation))
    if not result:
        raise ValueError("Enable at least one retrieval request.")
    return result
