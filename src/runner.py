"""One request per ticker/provider pair; failures never erase other results."""
from datetime import datetime, timezone
from pathlib import Path
import uuid

import pandas as pd

from .models import RetrievalError, normalize
from .providers.base import check_request_support
from .storage import SeriesCache, safe_name, write_json

SUMMARY_COLUMNS = ["request_id", "requested_start", "requested_end", "data_type", "requested_frequency", "ticker", "provider", "provider_symbol", "kind", "status", "rows", "numeric_observations",
                   "first_observation", "last_observation", "currency", "unit", "close_basis", "adj_close_basis",
                   "missing_values", "cache_hit", "retrieved_at_utc", "message", "data_file"]


def retrieve(instruments, providers, window, output: Path | None, cache: SeriesCache | None, refresh=False, progress=print, *, results=None, blocked=None, request_metadata=None):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    folder = Path(output) / run_id if output is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=False)
    records, price_frames, observation_frames, wide_frames = [], [], [], []
    blocked = {} if blocked is None else blocked
    for provider in providers:
        for instrument in instruments:
            symbol = instrument.symbol_for(provider.name, provider.kind)
            row = {column: None for column in SUMMARY_COLUMNS}
            row.update(ticker=instrument.ticker, provider=provider.name, provider_symbol=symbol,
                       kind=instrument.kind, rows=0, numeric_observations=0, cache_hit=False)
            row.update({key: value for key, value in (request_metadata or {}).items() if key in SUMMARY_COLUMNS})
            row.update(requested_start=str(window.start), requested_end=str(window.end))
            if symbol is None:
                row.update(status="not_applicable", message="No matching data type or explicitly disabled symbol mapping.")
                records.append(row)
                continue
            progress(f"Fetching {instrument.ticker} / {provider.name} ({symbol}) ...")
            try:
                check_request_support(provider.name, (request_metadata or {}).get("data_type"))
                result = None if refresh or cache is None else cache.get(provider, instrument, symbol, window)
                cache_hit = result is not None
                if result is None:
                    if provider.name in blocked:
                        raise RetrievalError("provider_paused", blocked[provider.name])
                    provider.check_access()
                    result = normalize(provider.fetch(instrument, symbol, window), window, instrument.kind)
                    result.metadata.update(ticker=instrument.ticker, provider_symbol=symbol, kind=instrument.kind,
                                           symbol_mapping="configured" if instrument.mapped else "literal passthrough; identity not verified")
                    if instrument.kind == "price" and not result.metadata.get("currency"):
                        result.warnings.append("Quote currency is unknown; confirm it in the Python instrument mapping before portfolio use.")
                    if not instrument.mapped:
                        result.warnings.append("Ticker passed literally to each price provider. Confirm exchange, instrument and currency equivalence.")
                    if cache is not None:
                        cache.put(provider, instrument, symbol, window, result)
                stem = safe_name(instrument.ticker)
                relative = Path("series") / provider.name / (stem + ".csv")
                result.metadata["cache_hit"] = cache_hit
                result.metadata.update(request_metadata or {})
                if folder is not None:
                    destination = folder / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    result.data.to_csv(destination, index=False, date_format="%Y-%m-%d")
                    write_json(destination.with_suffix(".metadata.json"), {**result.metadata, "cache_hit": cache_hit, "warnings": result.warnings})
                row.update({key: result.metadata[key] for key in SUMMARY_COLUMNS if key in result.metadata})
                boundary_gap = (result.metadata["start_gap_calendar_days"] > 7 or result.metadata["end_gap_calendar_days"] > 7)
                row.update(status="partial" if boundary_gap else "ok", cache_hit=cache_hit,
                           message=" | ".join(result.warnings), data_file=relative.as_posix() if folder is not None else None)
                if results is not None:
                    results[(instrument.ticker, provider.name)] = result
                if folder is not None:
                    frame = result.data.copy()
                    frame.insert(0, "provider_symbol", symbol)
                    frame.insert(0, "provider", provider.name)
                    frame.insert(0, "ticker", instrument.ticker)
                    if instrument.kind == "price":
                        for field in ("currency", "close_basis", "adj_close_basis"):
                            frame[field] = result.metadata.get(field)
                        price_frames.append(frame)
                        for column in ("close", "adj_close"):
                            if column in result.data:
                                wide = result.data.set_index("date")[[column]].rename(columns={column: f"{instrument.ticker}|{provider.name}|{column}"})
                                wide_frames.append(wide)
                    else:
                        frame["unit"] = result.metadata.get("unit")
                        observation_frames.append(frame)
            except RetrievalError as exc:
                row.update(status=exc.code, message=str(exc))
                if exc.code in {"rate_limited", "missing_key", "authentication_error", "network_error", "missing_dependency"}:
                    blocked[provider.name] = f"Further network requests paused after {exc.code}; see the first failure for this provider."
            except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
                # Do not expose arbitrary API payloads or credential-bearing exception URLs.
                row.update(status="invalid_response", message=f"Unexpected response shape ({type(exc).__name__}); adapter review required.")
            records.append(row)
            progress(f"  {row['status']}: {row['rows']} rows")
            # Checkpoint the completed pairs so cancellation doesn't hide completed work.
            if folder is not None:
                pd.DataFrame(records, columns=SUMMARY_COLUMNS).to_csv(folder / "summary.csv", index=False)
    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    if folder is None:
        return None, summary
    summary.to_csv(folder / "summary.csv", index=False)
    if price_frames:
        pd.concat(price_frames, ignore_index=True).to_csv(folder / "prices_long.csv", index=False, date_format="%Y-%m-%d")
        pd.concat(wide_frames, axis=1).sort_index().to_csv(folder / "prices_wide.csv", date_format="%Y-%m-%d")
    if observation_frames:
        pd.concat(observation_frames, ignore_index=True).to_csv(folder / "observations_long.csv", index=False, date_format="%Y-%m-%d")
    write_json(folder / "manifest.json", {
        "schema_version": 1, "run_id": run_id, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": str(window.start), "requested_end": str(window.end), "end_inclusive": True,
        "tickers": [i.ticker for i in instruments], "providers": [p.name for p in providers],
        "status_counts": {str(k): int(v) for k, v in summary.status.value_counts().items()},
        "notes": ["No calendar filling, currency conversion or cross-provider stitching.",
                  "ok means usable observations and no boundary gap over 7 days, not verified complete history.",
                  "Inspect adjustment conventions before comparing price levels or computing returns.",
                  "Options, portfolio analytics and Monte Carlo are not part of phase 1."],
    })
    return folder, summary
