# PythonToolBox_Taller1Master - Market data

Inputs live in `config/*.json`; operational setup and credential environment-variable names live in `run.py`. Library modules are directly under `src/`.

## One run, one COB archive

Run `run.py` using the project `.venv` interpreter in VS Code or Python. It displays today's date in **Europe/London** and prompts for a COB date in **DD/MM/YYYY**. Today and future dates are rejected.

1. A completed matching `tests/data_dumps/YYYY-MM-DD` archive is reused without API requests, even if the program runs again on a later day.
2. Otherwise SPX, IEF and the enabled FRED/ECB references request the maximum accessible history through COB.
3. Options use the date rule below. All responses and failures are saved in the COB archive, then the run stops. No cleansing or surface calibration is performed.

| Input COB relative to today's date | IBKR option API | Observation date |
|---|---|---|
| Yesterday | Current live snapshots, including IV/Greeks where IBKR supplies them | Actual capture date/time today |
| Older than yesterday | Historical one-minute TRADES, BID, ASK and MIDPOINT bars for COB | Requested COB only |

Example: when today is 08/10/2026, COB 07/10/2026 selects live options on 08/10/2026; COB 06/10/2026 selects historical options for 06/10/2026. This uses calendar days. Weekends/holidays are not silently moved to another trading date. A historical failure never triggers live quotes for an older COB.

**A yesterday-COB archive deliberately contains historical asset prices through COB and current option snapshots from the following day. They are not a synchronized historical surface.** Every live result records its capture date separately. The adapter checks the London date before each batch and stops starting new snapshots if the date changes.

## Inputs and portfolio

The portfolio is 50% S&P 500 index exposure and 50% IEF, reported in EUR, in `config/portfolio.json`. Options are reference inputs, not holdings. SPX is an index exposure for modelling, not a directly purchasable fund.

- `yahoo.json`, `massive.json`, `ibkr.json`: SPX and IEF historical requests.
- `alpha_vantage.json`, `marketdata.json`: SPX and IEF are enabled. Their current adapters cannot retrieve SPX index history, so these SPX requests save `unsupported_operation` in JSON without sending a stock request. IEF proceeds normally. There is no SPY substitution.
- `fred.json`, `ecb.json`: existing rate, overnight-rate and FX references.
- `ibkr.json`: the existing **4,181 explicit SPX AM call contracts**, 14 expiries through **17/12/2027**. No SPXW weekly options, puts or discovery during execution.
- Other provider JSONs contain no option requests in the normal run.

Provider identifiers differ: Yahoo uses `^GSPC`, Massive uses `I:SPX`, and IBKR uses structured contracts with symbol, security type and exchange. Each JSON contains relevant instructions and input details. Contract lookup validates the already selected option rather than discovering new options.

Non-options use `{"lookback":"max","end":"COB"}`. New HTTP history requests start on 03/01/2000 (or COB if earlier). IBKR uses its earliest available timestamp when discoverable, even if it precedes 2000, and otherwise falls back to 03/01/2000. These choices do not bypass provider retention, plan limits or missing data. IBKR retrieves daily history in yearly chunks. If that timestamp lookup fails, it still requests bars backwards from COB, stopping at the first empty yearly interval. This boundary is recorded explicitly; data before a gap is not assumed unavailable. HTTP adapters request the full range, with accessible-range fallbacks for supported providers. Alpha Vantage may supply only compact history if full history is denied.

IBKR options retain `operation: "option_history"` and `{"start":"COB","end":"COB"}` as their configured COB inputs. `src/cob.py` chooses the actual historical or live endpoint by date. Historical parameters specify 1-minute bars, requested data channels, regular trading hours and a 16:00 America/New_York cutoff. Expiry is separate from COB. Only the explicitly listed contracts are requested, including for older dates; the library cannot infer which contracts were listed on a historical date.

## Historical and live option limits

Historical requests use `reqHistoricalData`, an explicit UTC end time, one-day duration, and `keepUpToDate=False`. IBKR historical option bars do **not** provide per-contract historical IV or Greeks. The adapter records these fields as unavailable rather than inventing them. IBKR does not supply expired-option history; a listed input does not guarantee historical availability. Intraday bars are not official settlement prices. Missing historical channels remain visible in the output.

Live requests use the tutorial's ib_insync workflow: one read-only option connection, qualification of explicit contracts, then `reqTickersAsync` snapshots with at most 50 simultaneous option requests. These are current market-data snapshots; paid regulatory snapshots are disabled. Quotes, model IV/Greeks, dividend present value and underlying price are preserved where returned. Missing model fields receive `partial_model_data`; subscriptions do not guarantee every field for every strike. No IV or Greeks are calculated locally.

Historical requests share a separate read-only connection, with at most five concurrent contracts and bounded pacing. A historical option can require four requests. Three consecutive timeout/rate/network failures for a channel pause that channel for the remainder of the run, with explicit failure records. Large option universes can still take considerable time.

TWS or IB Gateway must remain logged in. In TWS, enable ActiveX and Socket Clients and Read-Only API. Match `IBKR_PORT` (currently 7496) and use a unique client ID (81). Authentication happens in TWS; there is no IBKR API key in the JSON. HTTP provider credentials come from environment variables named in `run.py`. Missing providers are recorded without stopping independent providers. No order requests are implemented.

## Inspecting and reusing a dump

Each dated folder contains:

- `manifest.json`: COB, execution date, option route, configuration fingerprint and status counts.
- `summary.json`: one result per provider/request, including retrieval method and missing fields.
- `resolved_inputs.json` and `portfolio.json`: the input snapshot.
- Provider/request folders with `request.json`, `status.json`, and `attempt_001/response_*` plus response metadata.

Raw HTTP bodies and serialized IBKR callbacks are retained before cleaning. Configured credentials are redacted if a response echoes them. Provider padding, current metadata and source adjustments may remain. Alpha Vantage has no historical end-date parameter: its dated observations may extend beyond COB and must be bounded before analysis. Macro observations are current vintages, not historical revision archives.

`complete` means every configured request has been attempted, **not** that every request supplied all data. Inspect `all_requests_succeeded`, per-channel statuses and missing fields. The default rerun reuses a completed archive even if it contains failures. An interrupted run resumes unfinished checkpoints. A different configuration for an existing COB is rejected before API calls to protect the archive. A `.running` lock prevents simultaneous writes; after a forced termination, inspect the previous process before removing a stale lock.

Archive schema 3 fingerprints effective data settings as well as request inputs. Credentials and transport settings are excluded, so credential rotation does not invalidate an archive. Compatible schema 2 archives are read and reused as originally captured. Each new retrieval attempt saves its executed operation and requested start/end in its own `attempt_NNN/request.json`; the request-level metadata and summary expose the latest attempt range. The original `resolved_inputs.json` remains the archive identity, so a retry with the new fallback does not erase the old input history. Their saved instruments, parameters and dates must match; a broader saved history is accepted. Schema 2 did not record effective runtime settings, so those settings cannot be retrospectively verified and a console notice makes that limitation explicit. Completed archives are not rewritten or downloaded again. Changed instruments or recorded adjustment settings still fail the compatibility check; adding tickers is not yet incremental. Saved failures remain visible and are retried only with retry_failed=True.

For an intentional retry of failed entries, from Python:

```python
import run
result = run.main("07/10/2026", retry_failed=True)
```

Retries preserve earlier attempt files and use the date rule at the time of the retry. A live result that could not complete yesterday cannot be recreated as live data after its capture day has passed; a later retry will use historical bars with explicit provenance.

Direct library use:

```python
from datetime import date
from src import MarketDataDownloader

dl = MarketDataDownloader.from_json("config", cob_date=date(2026, 10, 7))
result = dl.download_cob("07/10/2026", "tests/data_dumps",
                         portfolio_file="config/portfolio.json")
```

Use absolute paths from another working directory. The library never calls `input()`; only `run.py` prompts. Legacy `download_raw()` and normalized `download()` APIs remain for their supported workflows; use `download_cob()` for the current JSON configuration and date policy.

## Verification and later analytics

Offline unit/regression tests use fake responses and temporary folders; they do not validate subscriptions or call real providers. Run `python -m unittest discover -s tests -v`. The new COB tests cover date validation, both API routes, cache reuse, interrupted-run recovery, configuration mismatch and incomplete model fields.

Install the project with Python 3.11+ using `python -m pip install -e .`, including the declared ib_insync dependency. The separate official ibapi SDK is no longer used. Import the library as `from src import MarketDataDownloader`; do not add the `src` directory itself to PYTHONPATH because `src/http.py` would shadow Python's standard library.

Data cleansing, aligned observations, dividend forecasts, discount/forward curves, implied-volatility surfaces and Monte Carlo calibration remain later work. ESTR/SOFR observations and sovereign yields are not substitutes for an OIS discount curve. EUR/USD is USD per EUR; conversion to EUR requires the corresponding dated FX observation.

IBKR documentation: [historical bars](https://interactivebrokers.github.io/tws-api/historical_bars.html), [historical limitations](https://interactivebrokers.github.io/tws-api/historical_limitations.html).

Console results distinguish `[HISTORICAL]` from `[LIVE SNAPSHOT]`. Existing request IDs ending in `_HISTORY` are stable identifiers retained for archive reuse; they do not select the endpoint. New result metadata records the executed `operation` separately from `configured_operation`.
