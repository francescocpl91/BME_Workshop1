"""The single public class coordinating market-data retrieval.

run.py -> MarketDataDownloader -> provider adapters -> normalization -> results
Optional cache and export operations are delegated to storage.py via runner.py.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

from .config import PROVIDERS, load_instruments, provider_settings, split_inputs, parse_requests
from .models import DateWindow, RetrievedSeries, RetrievalError
from .runner import retrieve
from .storage import SeriesCache, write_json


@dataclass
class DownloadResult:
    """Successful typed series plus a diagnostic row for every provider/ticker pair.

    series[(request_id, provider)].data is a pandas DataFrame for configured requests.
    Direct ticker calls use (ticker, provider). windows records each configured range.
    .metadata and .warnings retain provenance and coverage information.
    Failed pairs appear in summary; folder is None unless export was requested.
    """
    window: DateWindow | None
    summary: pd.DataFrame
    series: dict[tuple[str, str], RetrievedSeries]
    folder: Path | None = None
    windows: dict[str, DateWindow] = field(default_factory=dict)


class MarketDataDownloader:
    """Configure once, then call download() from any Python program or prompt.

    providers: "all", a comma-separated string, or a list of provider names.
    settings: API keys and options, with names listed in config.DEFAULT_SETTINGS.
    instruments: Python dictionary overriding default symbol mappings.
    cache_dir: optional persistent cache; None disables the toolbox cache.
    progress: optional callback, e.g. print. The class never calls input().

    A configured API key does not prove that an account has permission for an
    endpoint/history range. access_status() checks local configuration only;
    download().summary records the actual response and access failures.
    """

    def __init__(self, providers: str | Sequence[str] = "all", *,
                 settings: Mapping[str, str] | None = None,
                 instruments: dict | None = None,
                 cache_dir: str | Path | None = None,
                 cache_hours: float = 24,
                 progress: Callable[[str], None] | None = None):
        if isinstance(providers, str):
            names = list(PROVIDERS) if providers.strip().lower() == "all" else providers.split(",")
        else:
            names = list(providers)
        names = list(dict.fromkeys(n.strip().lower() for n in names if n.strip()))
        if not names or any(n not in PROVIDERS for n in names):
            raise ValueError("Choose providers from: " + ", ".join(PROVIDERS))
        if not 0 <= cache_hours <= 8760:
            raise ValueError("cache_hours must be between 0 and 8760.")
        config = provider_settings(settings)
        self._settings = config
        self._retrieval_config = None
        self.providers = [PROVIDERS[name](config) for name in names]
        self.instruments = instruments
        self.cache = SeriesCache(Path(cache_dir), cache_hours) if cache_dir is not None else None
        self.progress = progress if progress is not None else lambda _: None

    @classmethod
    def from_json(cls, paths, *, settings=None, cache_dir=None, cache_hours=24,
                  output_dir=None, progress=None, cob_date=None, history_start=None):
        """Load provider-specific JSON request files; keep runtime settings in Python."""
        from .json_config import load_provider_configs
        config = load_provider_configs(paths, cob_date=cob_date, history_start=history_start)
        config.update(settings=dict(settings or {}), cache_dir=cache_dir,
                      cache_hours=cache_hours, output_dir=output_dir)
        return cls.from_config(config, progress=progress)

    @classmethod
    def from_config(cls, config: dict, *, progress=None):
        """Build the downloader from an already normalized configuration dictionary.

        Every request explicitly specifies its instrument, provider symbols and
        timeframe. All requests are validated before making any network call.
        """
        snapshot = deepcopy(config)
        requests = parse_requests(snapshot)
        names = list(dict.fromkeys(name for request in requests for name in request.providers))
        instance = cls(names, settings=snapshot.get("settings"),
                       cache_dir=snapshot.get("cache_dir"),
                       cache_hours=snapshot.get("cache_hours", 24), progress=progress)
        instance._retrieval_config = snapshot
        return instance

    def download_cob(self, cob, output, *, portfolio_file=None, retry_failed=False, history_start=None, ois_file=None):
        """Persistent COB archive with date-routed options; completed archives need no APIs."""
        from .cob import download_cob
        return download_cob(self, cob, output, portfolio_file=portfolio_file, retry_failed=retry_failed, history_start=history_start, ois_file=ois_file)

    def _download_configured(self, *, output=None, refresh=False):
        requests = parse_requests(self._retrieval_config)
        if any(request.instrument.kind == "option" for request in requests):
            raise ValueError("Configured option_history requests require download_cob(); explicit option_model_snapshot requests use download_raw(). Standardized option tables are not implemented.")
        output = output if output is not None else self._retrieval_config.get("output_dir")
        folder = None
        if output is not None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
            folder = Path(output) / run_id
            folder.mkdir(parents=True, exist_ok=False)
        bases = {provider.name: provider for provider in self.providers}
        frames, series, windows, blocked = [], {}, {}, {}
        for request in requests:
            # Reuse transport clients to preserve pacing across request entries.
            adapters = [PROVIDERS[name]({**self._settings, **request.provider_options}, client=bases[name].client)
                        for name in request.providers]
            collected = {}
            request_folder, summary = retrieve(
                [request.instrument], adapters, request.window, folder, self.cache,
                refresh, self.progress, results=collected, blocked=blocked,
                request_metadata=request.metadata(),
            )
            if folder is not None:
                present = summary.data_file.notna()
                summary.loc[present, "data_file"] = (request_folder.name + "/" + summary.loc[present, "data_file"])
            frames.append(summary)
            for (_, provider), value in collected.items():
                series[(request.request_id, provider)] = value
            windows[request.request_id] = request.window
        summary = pd.DataFrame([row for frame in frames for row in frame.to_dict("records")],
                               columns=frames[0].columns)
        common_window = requests[0].window if len(set(windows.values())) == 1 else None
        if folder is not None:
            summary.to_csv(folder / "summary.csv", index=False)
            write_json(folder / "manifest.json", {
                "requests": [{"request_id": request.request_id, "ticker": request.instrument.ticker,
                              "kind": request.instrument.kind, "data_type": request.data_type,
                              "providers": list(request.providers), "frequency": request.frequency,
                              "start": str(request.window.start), "end": str(request.window.end)}
                             for request in requests],
                "status_counts": {str(k): int(v) for k, v in summary.status.value_counts().items()},
            })
        return DownloadResult(common_window, summary, series, folder, windows)

    def download_raw(self, output: str | Path):
        """Dump native responses from configured JSON requests, without cleaning.

        Returns RawDownloadResult(folder, summary). Always uses live requests;
        the normalized-data cache and normal export pipeline are bypassed.
        Missing credentials and failures get status files, never invented data.
        """
        if self._retrieval_config is None:
            raise ValueError("Raw downloads require from_json() or from_config().")
        from .raw import dump_raw
        return dump_raw(self._retrieval_config, self.providers, self._settings, output, self.progress)

    def access_status(self) -> pd.DataFrame:
        """Show provider APIs and local credential readiness without revealing keys.

        No network requests occur. "configured" means a key was supplied, not
        that the provider accepted it or granted a particular subscription.
        "not_required" means the adapter has no API-key requirement.
        IBKR "session_unverified" means the SDK exists; TWS login/access is unchecked.
        "missing_key" includes empty/sample credentials.
        """
        rows = []
        for provider in self.providers:
            try:
                provider.check_access()
                status = "session_unverified" if provider.name == "ibkr" else "configured" if provider.key_env else "not_required"
            except RetrievalError as error:
                status = error.code
            rows.append({"provider": provider.name, "data_type": provider.kind,
                         "api": provider.api_url, "key_variable": provider.key_env,
                         "credential_status": status, "remote_permission": "unverified"})
        return pd.DataFrame(rows)

    def download(self, tickers: str | Sequence[str] | None = None, lookback: str = "1y", *,
                 end: date | str | None = None,
                 output: str | Path | None = None,
                 refresh: bool = False) -> DownloadResult:
        """Retrieve from every applicable configured provider, independently.

        lookback accepts 5y, 6mo, 30d, 4w or an ISO start date. The inclusive
        end defaults to the computer's current date at call time.
        Results stay in memory unless output is given. No price stitching,
        currency conversion, gap filling or yield-to-price conversion occurs.
        Provider failures are summary rows; invalid input raises ValueError.
        """
        if tickers is None:
            if self._retrieval_config is None:
                raise ValueError("Supply tickers, or construct the downloader with from_config().")
            if lookback != "1y" or end is not None:
                raise ValueError("Set timeframes in the config when downloading configured requests.")
            return self._download_configured(output=output, refresh=refresh)
        symbols = split_inputs(tickers if isinstance(tickers, str) else " ".join(tickers))
        window = DateWindow.parse(lookback, date.fromisoformat(end) if isinstance(end, str) else end)
        instruments = load_instruments(symbols, self.instruments)
        series = {}
        folder, summary = retrieve(
            instruments, self.providers, window,
            Path(output) if output is not None else None, self.cache,
            refresh, self.progress, results=series,
        )
        return DownloadResult(window, summary, series, folder)
