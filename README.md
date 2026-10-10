# PythonToolBox_Taller1Master - Market data

Inputs live in `config/*.json`; operational setup and credential environment-variable names live in `run.py`. Library modules are directly under `src/`.

## Choose a workflow

Run `run.py` with the project interpreter. It first asks you to choose:

- **1 - Inventory:** enter the saved archive COB and inspect downloaded data.
- **2 - Download + inventory:** enter the COB and history start to retrieve data or reuse a compatible archive, then automatically inspect that archive.

There is no default mode; blank or invalid choices prompt again. You can also
type `inventory` or `download`. Inventory has no history-start prompt,
API connection, credential check, or download if the archive is missing.
A completed archive is required; an archive with a `.running` lock is refused.

Download runs inventory after retrieval or archive reuse finishes, using the
same COB without another prompt. Saved provider failures and limited data
are included in the report. If downloading stops with an exception, inventory
does not start. If inventory fails, the raw archive remains available; resolve
the reported issue and use Inventory to retry the inspection without APIs.

`src/preprocessing.py` reads saved metadata and the recorded attempt for each
request, verifies response checksums, and inventories all its pages/chunks.
Up to four local readers inspect files concurrently, preserving request order.
Earlier attempts are not concatenated. Repeated rows and conflicting duplicate
keys are reported; none are removed.

Outputs are separate from the raw archive:

- `reports/YYYY-MM-DD/inventory.md`: readable coverage, fields and diagnostics.
- `reports/YYYY-MM-DD/inventory.json`: per-request metadata, actual date ranges,
  numerical/missing/invalid/unavailable field counts, duplicates, response files
  and checksums, plus option coverage by underlying, expiry and strike.

This is **step 1 only**. No provider ranking, merging, adjustments, filling,
date clipping, returns, fitting or simulation. Counts refer to raw records.
Date gaps are calendar-day diagnostics, not exchange-holiday checks. Numeric
model fields and ordered bid/ask quotes do not establish option liquidity.
SDK receipt times are not guaranteed exchange quote times. Fixed OIS curves
retain their actual observation date and par-rate units. The configured option
expiry and IBKR last trading date remain separate.

Download statuses (for example, limited history) remain separate from reader
errors (for example, corrupt files or unknown schemas). Missing values never
become zeros. A valid number is not automatically a financially usable value.
For example, a small negative model vega is counted as numeric and separately flagged.

Library use:

```python
from src import inspect_archive

inventory = inspect_archive("tests/data_dumps/2026-10-08", progress=print)
inventory.save("reports/2026-10-08")
# inventory.requests and inventory.option_expiries are available in Python.
```

Inventory reports can be regenerated; the inventory reader never writes raw archive files.
For programmatic use, bypass the mode menu with `main(..., mode="inventory")`
or `main(..., mode="download")`. Supply the dates too to avoid date prompts.
Download mode generates both reports and returns the original download result;
Inventory mode returns the inventory object. Direct library `download_cob()`
calls remain retrieval-only; automatic inventory is orchestrated by `run.py`.

## Download mode: one run, one COB archive

Choose **2 - Download** when running `run.py` using the project `.venv` interpreter in VS Code or Python. It displays today's date in **Europe/London** and prompts for a COB date and a historical start date in **DD/MM/YYYY**. COB must be yesterday or earlier, and history start must be on or before COB.

1. A completed matching `tests/data_dumps/YYYY-MM-DD` archive is reused without API requests, even if the program runs again on a later day.
2. Otherwise SPX, SPY, IEF and the enabled FRED/ECB references request history from the user-selected start through COB.
3. Options use the date rule below. All responses and failures are saved in the COB archive.
4. The completed new or reused archive is automatically inspected and the inventory reports are saved in `reports/YYYY-MM-DD/`, then the run stops. No cleansing or surface calibration is performed.

| Input COB relative to today's date | IBKR option API | Observation date |
|---|---|---|
| Yesterday | Current live snapshots, including IV/Greeks where IBKR supplies them | Actual capture date/time today |
| Older than yesterday | Historical one-minute TRADES, BID, ASK and MIDPOINT bars for COB | Requested COB only |

Example: when today is 08/10/2026, COB 07/10/2026 selects live options on 08/10/2026; COB 06/10/2026 selects historical options for 06/10/2026. This uses calendar days. Weekends/holidays are not silently moved to another trading date. A historical failure never triggers live quotes for an older COB.

**A yesterday-COB archive deliberately contains historical asset prices through COB and current option snapshots from the following day. They are not a synchronized historical surface.** Every live result records its capture date separately. The adapter checks the London date before each batch and stops starting new snapshots if the date changes.

## Inputs and portfolio

The portfolio is 50% S&P 500 index exposure and 50% IEF, reported in EUR, in `config/portfolio.json`. Options are reference inputs, not holdings. SPX is an index exposure for modelling, not a directly purchasable fund.

- `yahoo.json`, `massive.json`, `ibkr.json`: SPX, SPY and IEF historical requests.
- `alpha_vantage.json`, `marketdata.json`: SPX, SPY and IEF are enabled. Their current adapters cannot retrieve SPX index history, so these SPX requests save `unsupported_operation` in JSON without sending a stock request. SPY and IEF use ETF history requests; SPY is additional data, not a replacement for SPX.
- `fred.json`, `ecb.json`: existing rate, overnight-rate and FX references.
- `ibkr.json`: the existing **4,181 explicit SPX AM call contracts**, 14 expiries through **17/12/2027**. No SPXW weekly options, puts or discovery during execution.
- Other provider JSONs contain no option requests in the normal run.

Provider identifiers differ: Yahoo uses `^GSPC`, Massive uses `I:SPX`, and IBKR uses structured contracts with symbol, security type and exchange. Each JSON contains relevant instructions and input details. Contract lookup validates the already selected option rather than discovering new options.

After choosing Download, `run.py` asks for **COB** and **history start**, both in `DD/MM/YYYY`. The start must be on or before COB (minimum 01/01/1900). All non-option JSON requests use `{"start":"HISTORY_START","end":"COB"}`. These placeholders are resolved in memory to the user's actual dates and saved in the archive's `resolved_inputs.json`; the source JSON files are not rewritten at every run. Programmatic use: `MarketDataDownloader.from_json(..., cob_date=date(...), history_start=date(...))`. Explicit dates in custom JSONs are respected when no runtime start override is supplied.

Legacy `lookback: "max"` without a supplied start is rejected with a migration message; it cannot silently choose January 2000. The chosen window is requested first. If data begins later, the available part is retained without filling missing dates. A date/data-range or plan-range error triggers a provider-specific maximum accessible-history fallback, still bounded by COB where the API supports date bounds. Missing keys, authentication failures, transport failures and pacing errors do not trigger a wider date query. Unsupported index endpoints remain explicit failures.

- Yahoo: discover the first-trade timestamp, then request through COB.
- FRED/ECB: omit the lower date bound, retain COB as the upper bound.
- Alpha Vantage: request full daily history; if denied, try compact history. This endpoint does not accept start/end dates, so raw data can extend outside the requested window and may not cover an older COB.
- MarketData.app: page backwards with countback through COB; reduce page size at access boundaries.
- Massive: request backwards in date windows, narrowing rejected windows. Its aggregate search stops at the nonnegative Unix-timestamp boundary (1970) or first unavailable interval.
- IBKR: use daily chunks within the selected window, respecting a later first-available timestamp. On a range failure, retry from the earliest reported timestamp; if that lookup failed, search backwards with a 1900 safety boundary. No automatic January 2000 substitution exists.

Maximum means **accessible provider history**, not a guarantee of complete coverage. First unavailable intervals, account limits and search safety boundaries are recorded; earlier gaps/coverage remain unverified. The original request, fallback reason and executed fallback windows are saved in request status/summary metadata, alongside unchanged raw responses. Fallbacks may include observations before the requested start. No cleaning, clipping or filling is performed here. Existing compatible broader archives can be reused without downloading or modifying their original files; a request for earlier history than the saved identity rejects reuse instead of silently expanding it.

IBKR options retain `operation: "option_history"` and `{"start":"COB","end":"COB"}` as their configured COB inputs. `src/cob.py` chooses the actual historical or live endpoint by date. Historical parameters specify 1-minute bars, requested data channels, regular trading hours and a 16:00 America/New_York cutoff. Expiry is separate from COB. Only the explicitly listed contracts are requested, including for older dates; the library cannot infer which contracts were listed on a historical date.

## Manual OIS inputs

`config/ois.json` holds manual USD SOFR and EUR ESTR **spot-starting par OIS swap quotes** from `OIS.docx`, dated 09/10/2026: 86 SOFR tenors (1W–50Y) and 44 ESTR tenors (1W–60Y). Only the source's Live column supplies the curve; historical comparison columns are excluded. Rates are percentages per annum, so 3.890 means 3.890%. They are not zero rates. Contract conventions remain null where the source does not provide them; bootstrap, discount factors and forwards are later work.

`run.py` passes `OIS_FILE` to `download_cob(..., ois_file=...)`. `src.ois.load_ois_config()` validates the input before provider calls. No subscription, key, browser session or API is involved. The JSON is separate from the provider request files; directory-based provider loading excludes it.

Each regional list contains exactly one fixed curve. With `date_policy=fixed_for_all_cobs`, every COB uses those same SOFR and ESTR quotes, still dated 09/10/2026. The dump records `usage_policy=fixed_for_all_cobs` and whether the observation date is before, after or equal to COB. `suitable_for_strict_cob` is true only when the dates match; it is provenance metadata, not a restriction on including the fixed inputs. There is no date-based selection, refresh or missing-date exclusion.

New dumps include `ois/input.json`, `ois/us/curve.json`, `ois/us/status.json`, `ois/euro/curve.json`, and `ois/euro/status.json`. The two OIS results also appear in the main summary, even if an API fails. OIS data participates in archive identity checks: changing quotes or adding OIS to a pre-existing archive rejects reuse without overwriting the archive. Completed matching archives are checked and read without API calls or file writes.

Keep exactly one fixed curve per region in `config/ois.json`, preserving its observation date, percent units and quote type. The JSON contains all required OIS inputs; `OIS.docx` is not required for execution or offline tests. Its recorded filename and hash identify the original source only. The comparison against that document was completed when the rates were imported; the tests now cover JSON validation, fixed-curve inclusion and archive reuse. Forward-chart metadata is provenance only; no numerical forward quotes were extracted.

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

Archive schema 3 fingerprints effective data settings as well as request inputs. Credentials and transport settings are excluded, so credential rotation does not invalidate an archive. Compatible schema 2 archives are read and reused as originally captured. Each new retrieval attempt saves its executed operation and requested start/end in its own `attempt_NNN/request.json`; the request-level metadata and summary expose the latest attempt range. The original `resolved_inputs.json` remains the archive identity, so a retry does not erase the old input history. Reuse also checks each saved attempt's requested range: a successful narrower retry cannot satisfy a later broader request merely because the original identity was broader. Default reuse rejects that case without API calls/writes; retry_failed=True retrieves the insufficient range in a new attempt, retaining earlier files. Their saved instruments, parameters and dates must match; a broader saved history is accepted. Schema 2 did not record effective runtime settings, so those settings cannot be retrospectively verified and a console notice makes that limitation explicit. Completed archives are not rewritten or downloaded again. Changed instruments or recorded adjustment settings still fail the compatibility check; adding tickers is not yet incremental. Saved failures remain visible and are retried only with retry_failed=True.

For an intentional retry of failed entries, from Python:

```python
import run
result = run.main("07/10/2026", "03/01/2000", mode="download", retry_failed=True)
```

Retries preserve earlier attempt files and use the date rule at the time of the retry. A live result that could not complete yesterday cannot be recreated as live data after its capture day has passed; a later retry will use historical bars with explicit provenance.

Direct library use:

```python
from datetime import date
from src import MarketDataDownloader

dl = MarketDataDownloader.from_json(
    "config", cob_date=date(2026, 10, 7), history_start=date(2000, 1, 3)
)
result = dl.download_cob("07/10/2026", "tests/data_dumps",
                         portfolio_file="config/portfolio.json", ois_file="config/ois.json")
```

Use absolute paths from another working directory. The library never calls `input()`; only `run.py` prompts. Legacy `download_raw()` and normalized `download()` APIs remain for their supported workflows; use `download_cob()` for the current JSON configuration and date policy.

## Verification and later analytics

Offline unit/regression tests use fake responses and temporary folders; they do not validate subscriptions or call real providers. Run `python -m unittest discover -s tests -v`. The new COB tests cover date validation, both API routes, cache reuse, interrupted-run recovery, configuration mismatch and incomplete model fields.

Install the project with Python 3.11+ using `python -m pip install -e .`, including the declared ib_insync dependency. The separate official ibapi SDK is no longer used. Import the library as `from src import MarketDataDownloader`; do not add the `src` directory itself to PYTHONPATH because `src/http.py` would shadow Python's standard library.

Data cleansing, aligned observations, dividend forecasts, discount/forward curves, implied-volatility surfaces and Monte Carlo calibration remain later work. ESTR/SOFR observations and sovereign yields are not substitutes for an OIS discount curve. EUR/USD is USD per EUR; conversion to EUR requires the corresponding dated FX observation.

IBKR documentation: [historical bars](https://interactivebrokers.github.io/tws-api/historical_bars.html), [historical limitations](https://interactivebrokers.github.io/tws-api/historical_limitations.html).

Console results distinguish `[HISTORICAL]` from `[LIVE SNAPSHOT]`. Existing request IDs ending in `_HISTORY` are stable identifiers retained for archive reuse; they do not select the endpoint. New result metadata records the executed `operation` separately from `configured_operation`.
