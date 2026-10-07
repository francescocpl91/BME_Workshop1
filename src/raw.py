"""Save native response bodies before decoding, table conversion or cleaning."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from urllib.parse import parse_qsl, quote, quote_plus, urlsplit
import uuid

import pandas as pd

from .config import PROVIDERS, parse_requests
from .models import RetrievalError
from .options import OPTION_RESPONSE_TYPES
from .storage import safe_name, write_json


@dataclass
class RawDownloadResult:
    folder: Path
    summary: pd.DataFrame


class ResponseRecorder:
    """Keep body bytes unchanged except explicitly recorded credential redaction.

    Transport framing/compression has already been handled by the HTTP client.
    Headers, cookies and authentication requests are never persisted.
    """
    def __init__(self, folder, secrets=()):
        self.folder = Path(folder)
        self.responses = []
        self.secrets = sorted({form for secret in secrets if secret
                               for form in (secret, quote(secret, safe=""), quote_plus(secret))},
                              key=len, reverse=True)

    def scrub(self, value):
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def __call__(self, body, status, content_type, url):
        stored = body
        for secret in self.secrets:
            stored = stored.replace(secret.encode("utf-8"), b"[REDACTED]")
        parsed = urlsplit(url)
        parameters = {key: ("[REDACTED]" if key.lower() in
                           {"apikey", "api_key", "token", "access_token", "crumb"}
                           else self.scrub(value)) for key, value in parse_qsl(parsed.query)}
        suffix = ".json" if "json" in content_type.lower() else ".csv" if "csv" in content_type.lower() else ".txt"
        filename = f"response_{len(self.responses) + 1:03d}{suffix}"
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / filename).write_bytes(stored)
        self.responses.append({
            "file": filename, "http_status": status, "content_type": content_type,
            "endpoint": self.scrub(f"{parsed.scheme}://{parsed.netloc}{parsed.path}"),
            "parameters": parameters, "received_at_utc": datetime.now(timezone.utc).isoformat(),
            "received_bytes": len(body), "stored_bytes": len(stored),
            "sha256": hashlib.sha256(stored).hexdigest(),
            "credentials_redacted": stored != body,
        })
        write_json(self.folder / "responses.json", self.responses)


def dump_raw(config, providers, settings, output, progress):
    requests = parse_requests(config)  # Validate everything before network or file writes.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    folder = Path(output).resolve() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    bases = {provider.name: provider for provider in providers}
    secrets = [p.key for p in providers if p.key]
    records, blocked = [], {}
    notes = [
        "Native HTTP response bodies; no table conversion, sorting, date clipping, missing-value filling or normalization.",
        "Provider-side adjustments selected by the JSON requests still apply; raw means unchanged received payload, not necessarily unadjusted prices.",
        "Only response decoding and minimal status/pagination inspection occur; each page and HTTP retry response is saved separately.",
        "response_saved confirms capture, not completeness, usable observations or valid financial data.",
        "Yahoo uses chart JSON directly with UTC-midnight query boundaries, bypassing yfinance history processing.",
        "No API keys, request headers or cookies are logged. Echoed configured credentials are redacted and flagged.",
    ]
    write_json(folder / "manifest.json", {"mode": "raw", "run_id": run_id, "complete": False, "notes": notes})
    for request in requests:
        for name in request.providers:
            provider = PROVIDERS[name]({**settings, **request.provider_options}, client=bases[name].client)
            relative = Path(name) / safe_name(request.request_id)
            destination = folder / relative
            destination.mkdir(parents=True, exist_ok=False)
            recorder = ResponseRecorder(destination, secrets)
            symbol = request.instrument.symbol_for(name, request.instrument.kind)
            row = {"request_id": request.request_id, "ticker": request.instrument.ticker,
                   "provider": name, "provider_symbol": symbol,
                   "requested_start": str(request.window.start), "requested_end": str(request.window.end),
                   "status": "pending", "message": "", "responses_saved": 0,
                   "folder": relative.as_posix(),
                   "response_type": OPTION_RESPONSE_TYPES.get(name) if request.instrument.kind == "option" else request.data_type}
            metadata = {**request.metadata(), **row, "provider_options": provider.cache_settings(),
                        "mode": "raw", "normalization_applied": False}
            write_json(destination / "request.json", metadata)
            write_json(destination / "status.json", row)
            progress(f"Raw download: {request.request_id} / {name} ...")
            previous_sink = provider.client.response_sink
            provider.client.response_sink = recorder
            try:
                provider.check_access()
                if name in blocked:
                    raise RetrievalError("provider_paused", blocked[name])
                if request.instrument.kind == "option":
                    provider.fetch_option_raw(request.instrument, symbol, request.window, request.contract)
                else:
                    provider.fetch_raw(request.instrument, symbol, request.window)
                row.update(status="response_saved" if recorder.responses else "no_response",
                           message="Native responses saved; data quality and completeness not assessed.")
            except RetrievalError as error:
                row.update(status=error.code, message=recorder.scrub(str(error)))
                if error.code in {"rate_limited", "authentication_error", "network_error"}:
                    blocked[name] = f"Further requests paused after {error.code}; inspect the first failure."
            except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
                row.update(status="invalid_response", message=f"Response inspection failed ({type(error).__name__}); any received bodies are retained.")
            finally:
                provider.client.response_sink = previous_sink
            row["responses_saved"] = len(recorder.responses)
            write_json(destination / "status.json", row)
            records.append(row)
            pd.DataFrame(records).to_csv(folder / "summary.csv", index=False)
            progress(f"  {row['status']}: {row['responses_saved']} response files")
    summary = pd.DataFrame(records)
    write_json(folder / "manifest.json", {
        "mode": "raw", "run_id": run_id, "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_counts": {str(key): int(value) for key, value in summary.status.value_counts().items()},
        "notes": notes,
    })
    return RawDownloadResult(folder, summary)
