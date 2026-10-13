# PythonToolBox_Taller1Master - Market data

Inputs live in `config/*.json`; operational setup and credential environment-variable names live in `run.py`. Library modules are directly under `src/`.

## Choose a workflow

Run `run.py` with the project interpreter. It asks you to choose:

- **1 - Process saved archive:** enter a saved COB. Inventory, clean and merge the
  local data without API connections or a history-start prompt, then generate statistics.
- **2 - Download + process:** enter COB and history start. Retrieve/reuse a compatible
  archive, then automatically inventory, clean, merge and analyse it.

Both choices use the same processing pipeline. The aliases `process`,
`inventory` and `download` work too. The standalone library function
`inspect_archive()` remains inventory-only; `download_cob()` remains retrieval-only.
The entry point joins these stages, with no additional date prompt.
Missing/incomplete archives or an active `.running` lock are refused.

## Historical statistics

Both run.py workflows now compute statistics after cleaning. Settings live in
config/modeling.json; tickers and weights remain in config/portfolio.json.
This stage is offline and analyses configured holdings only. Options and reference
instruments do not enter portfolio weights. No Monte Carlo or pricing is run yet.

src/statistics.py exposes AssetStatistics (automatic sample statistics) and
PortfolioStatistics (report(), plots_report(), save()). These analytical objects
wrap cleaned observations; they do not replace the retrieval PriceSeries.

Set statistics.return_types to ["simple", "log"] (the supplied default), or
select either convention alone. One calculation engine produces both analyses:
simple = P_t/P_previous - 1; log = log1p(simple). Both use identical valid intervals.
Each gets daily mean/std, annualized mean/volatility, skewness, excess kurtosis,
empirical quantiles/extremes, rolling volatility, normal/Student-t fits, covariance,
correlation and stress comparisons. No parametric quantiles or simulations are added.

All return-dependent CSV rows include return_type. Annualized mean is daily mean
times periods_per_year: arithmetic mean for simple returns, additive log growth
for log returns. It is not CAGR. Volatility uses sample std (ddof=1) times sqrt(252)
by default. Excess kurtosis uses zero for the normal reference. Undefined values
are blank/null, never zero. Charts display simple returns in percent and log
returns multiplied by 100, explicitly labelled as log-return units.

Normalized wealth, cumulative return, CAGR, percentage drawdown and coverage are
calculated once from prices/wealth and shared between return types. CAGR uses
elapsed calendar years. The report shows actual dates alongside these metrics:
individual histories may start earlier than the first common portfolio date.
Portfolio log returns come from the constructed portfolio wealth; they are never
a weighted average of holding log returns.

For direct library use, asset.return_statistics["simple"] and ["log"] contain
metrics, stress, rolling and distribution results; asset.path_metrics and
asset.stress_paths contain shared wealth results. The original metrics/stress/
rolling/distribution convenience attributes remain: simple is preferred when
enabled, otherwise they expose the labelled log analysis. For unambiguous access,
use return_statistics[kind].

Portfolio statistics represent a hypothetical buy-and-hold backtest: apply the
current initial weights once at the first common observed price date; reinvested
sleeve weights then drift. They are not daily-rebalanced 50/50 returns. Adjusted
series already include distribution adjustments; dividend cash is not added again.
Price-only portfolios remain labelled price_return.

Every active holding must be available in one common currency with compatible
return meaning. Missing holdings are not dropped or renormalized. Mixed currencies
or incompatible bases make portfolio wealth unavailable, while individual
statistics remain accessible. USD histories produce USD-labelled portfolio
statistics even when the configuration requests EUR reporting. No FX conversion
or constant-FX assumption is made.

Missing sessions are never filled. A daily return cannot bridge a gap. Correlation
and covariance use complete observations with identical interval starts and ends.
Rolling statistics retain missing dates, so gaps break the window. Cumulative
return, CAGR and maximum drawdown are withheld when an observed path contains
missing prices or deliberately excluded intervals. Data before inception does not
count as an interior gap.

GFC, COVID and 2022 stress windows are configurable illustrative periods. Their
actual coverage, sample size, volatility, drawdown and cross-asset correlation
are exported. A return starting before a window is excluded. Full-history
statistics use each holding's own available dates; cross-asset dependence uses
the common sample, whose count is reported separately.

Normal/Student-t density fits are descriptive, not simulation parameters.
Student-t scale is not standard deviation; fits with undefined variance are
flagged. Short/constant histories skip the fit. Fit failures are reported without
preventing the other statistics.

Outputs are written below:
processed/<COB>/<processing-id>/statistics/<statistics-id>/

- statistics.csv: per-series return moments, labelled simple/log.
- path_statistics.csv: one set of growth, CAGR, drawdown and coverage per series.
- daily_statistics.csv: selected prices, both return columns, interval starts,
  normalized wealth and drawdown; each date occurs once per series.
- rolling_volatility.csv, dependence.csv, rolling_correlations.csv: typed diagnostics.
- aligned_returns.csv: common intervals per return type, with return::<id> columns.
- stress_statistics.csv, stress_dependence.csv: typed crisis-return comparisons.
- stress_path_statistics.csv: shared crisis-window growth/drawdown and actual coverage.
- distributions.csv: typed normal/t parameters and fit status.
- report.md, manifest.json: assumptions, warnings, input identity and file hashes.
- plots/simple/ and plots/log/: per-series volatility/histogram/Q-Q diagnostics
  and correlation comparison for the selected convention.
- plots/wealth.png and plots/drawdown.png: shared investment-path charts.

Statistics output schema is now 2: path metrics moved out of statistics.csv into
path_statistics.csv (and the equivalent stress table), while return tables gained
return_type. Older reports remain unchanged in their original run directories.
aligned_observations counts common intervals once, not twice; the manifest also
records aligned_observations_by_return_type. Plot manifest keys use relative paths
so identically named simple/log charts each retain their own checksum.

Statistics identity includes the cleaned inputs, weights, settings, code and
dependency versions. Raw and cleaned source files are preserved. Completion means
the report was exported, not that every holding or modeling input is available.

To analyse an existing cleaned run directly (no raw-file rescan):

    from pathlib import Path
    from src import analyze_saved_processing

    folder = Path("processed/2026-10-08/<processing-id>")
    analysis = analyze_saved_processing(folder, "config/modeling.json")
    print(analysis.report())
    analysis.save(folder / "statistics" / analysis.run_id)

Select the intended processing ID explicitly; the library does not guess which
older run or portfolio settings to use. This path checks the saved price-file
checksum before analysis. Within Python, analyze_processed(result, config_path)
also accepts the in-memory result returned by process_archive/process_inventory.

## Cleaning and merging

`src/preprocessing.py` reads saved response bytes and verifies their checksums.
`src/data_cleaning.py` processes those observations separately from raw files.
`config/cleaning.json` contains provider priorities, exchange calendars, thresholds
and optional instrument overrides. New tickers do not require Python edits.
Stocks/ETFs, indices, option snapshots, historical option bars, corporate actions,
reference observations and manual OIS quotes follow separate rules.

The order is:

1. Read the recorded attempt for each provider/request. Inventory failures remain
   visible; unreadable or checksum-failed observations cannot become selected data.
2. Clean within each provider: finite numbers, date window, daily frequency,
   positive prices, consistent OHLC, duplicate and calendar checks. Equal duplicates
   collapse; conflicting versions are retained but quarantined. Explicit conflicts
   between configured and provider currencies/denominations are also quarantined;
   both values are retained, with no automatic currency or unit conversion.
3. Choose the price basis and primary provider for each product. Prefer documented
   split/dividend-adjusted close for stocks and ETFs. Indices keep their declared
   price/total-return convention. Each selected adjusted observation must have the
   documented adjustment convention. Ordinary closes and corporate actions are retained.
4. Compare ordinary closes only when currency, units and adjustment basis agree.
   Never average providers. The initial relative discrepancy threshold is **0.1%**.
   Same-basis ordinary-price gaps may be filled only after at least two overlapping
   observations agree within tolerance and no compared overlap fails.
   **Adjusted-close scales are never spliced automatically.**
5. Align to exchange sessions and calculate simple/log returns between adjacent
   sessions only. Missing observations remain missing; no interpolation or forward
   filling occurs. A missing session invalidates the following one-session return.
   The common holding-return table requires matching start and end dates for all
   holdings. Different-calendar interval mismatches are excluded and reported;
   individual return start dates remain in the selected-price table.
6. Screen option snapshots independently: positive ordered bid/ask, contract details,
   expiry, model IV, model underlying and (by policy) nonnegative dividend PV.
   Initial relative-spread limit: **10% of midpoint**. Missing optional Greeks or volume
   do not automatically reject a quote; supplied very low/negative model vega is flagged.
   Select a whole observation, never combine fields across providers or timestamps.
7. Preserve native reference dates, rate units, corporate actions and fixed par OIS
   observation dates. Save derived tables, exclusions, source comparisons and a report.

The default calendar is **XNYS**, using `exchange-calendars`, with historical holidays,
exceptional closures and early closes. Override it for other listing markets.
Do not apply the equity calendar to rates, FX, or option snapshot capture times.
Dates before an instrument's first observation remain visible as missing coverage,
not fabricated prices.

Adjusted-close returns include the provider's distribution adjustment. **Do not add
the saved dividends to those returns again.** These historical events are not a
future dividend forecast. The option model dividend PV is not a forecast cashflow schedule.

Option model IV, model price, underlying value and dividend PV remain separate
from bid/ask, midpoint and bid/ask IVs. IV is an annualized decimal. The configured
AM/PM expiry clock is explicitly labelled a convention estimate. Timestamp means
SDK receipt, not verified exchange quote freshness. A capture span above five
minutes is flagged. Candidate status does not establish simultaneity, liquidity,
forward consistency or absence of arbitrage. American options are distinguished
from European surface candidates. Historical bars do not acquire invented IV/Greeks.

### Outputs

Raw inputs remain in `tests/data_dumps/YYYY-MM-DD/`.
Inventory reports go to `reports/YYYY-MM-DD/inventory.json` and `inventory.md`.
Derived outputs go to `processed/YYYY-MM-DD/<processing-id>/`:

| File | Purpose |
|---|---|
| `provider_prices.csv` | Provider observations, quality flags and duplicate provenance |
| `selected_prices.csv` | Calendar-aligned selected prices, source, basis, simple/log returns and gaps |
| `aligned_holding_returns.csv` | Matching date intervals with valid local-currency returns for all holdings |
| `price_comparisons.csv` | Comparable provider differences and tolerance flags |
| `corporate_actions.csv` | Distributions and splits, separate from adjusted returns |
| `option_quotes.csv` | All cleaned option observations, exclusions, IVs, Greeks and model fields |
| `selected_option_quotes.csv` | Selected whole observations, including failed candidates for inspection |
| `option_iv_candidates.csv` | Preliminary quotes passing field/spread/vega checks |
| `historical_option_bars.csv` | Historical bars, kept distinct from snapshots |
| `reference_observations.csv`, `selected_references.csv` | Native rate/FX observations and source selection |
| `ois_quotes.csv` | Par quotes in percent and decimal units, with actual dates |
| `exchange_sessions.csv` | Calendar session open/close timestamps |
| `request_status.csv`, `issues.csv` | Saved provider statuses and reader/processing diagnostics |
| `manifest.json`, `report.md` | Policy, current/archived portfolio, coverage, readiness and output hashes |

A processing ID fingerprints inputs, source checksums, policy, portfolio and calendar
version. Changing weights or cleaning settings creates a separate derived run;
identical inputs regenerate the same derived output location. Raw responses are never
changed. Complete means processing finished, not that all data passed quality checks.

Processing uses the **current** portfolio JSON, while retaining the archived portfolio
in the report. To apply new weights to an old archive choose **1**. Download archive
compatibility rules remain strict: a changed portfolio/configuration can reject reuse
in Download mode. Local processing does not bypass those rules or make provider requests.

The cleaning stage performs no portfolio valuation, currency conversion, rebalancing, dividend forecast,
SPX-to-SPY calibration, OIS bootstrap, surface fitting or Monte Carlo is performed
in this stage. Foreign-currency holdings need matching FX history before EUR analysis.

Library use:

```python
from src import process_archive

result = process_archive(
    "tests/data_dumps/2026-10-08", "config/cleaning.json",
    portfolio_file="config/portfolio.json", progress=print,
)
result.save(f"processed/{result.cob}/{result.run_id}")
# result.tables contains pandas DataFrames; result.summary contains diagnostics.
```

Use absolute paths outside the project directory. Programmatic entry-point use:
`run.main("08/10/2026", mode="process")`; the legacy `mode="inventory"` alias
now also cleans/merges. It returns the inventory object for compatibility.
Download mode returns the original download result. Use `process_archive()`
when you want the processed tables directly.

## Download mode: one run, one COB archive

Choose **2 - Download** when running `run.py` using the project `.venv` interpreter in VS Code or Python. It displays today's date in **Europe/London** and prompts for a COB date and a historical start date in **DD/MM/YYYY**. COB must be yesterday or earlier, and history start must be on or before COB.

1. A completed matching `tests/data_dumps/YYYY-MM-DD` archive is reused without API requests, even if the program runs again on a later day.
2. Otherwise SPX, SPY, IEF and the enabled FRED/ECB references request history from the user-selected start through COB.
3. Options use the date rule below. All responses and failures are saved in the COB archive.
4. The new or reused archive is automatically inventoried, cleaned and merged into separate derived outputs, then the run stops. No surface calibration is performed.

| Input COB relative to today's date | IBKR option API | Observation date |
|---|---|---|
| Yesterday | Current live snapshots, including IV/Greeks where IBKR supplies them | Actual capture date/time today |
| Older than yesterday | Historical one-minute TRADES, BID, ASK and MIDPOINT bars for COB | Requested COB only |

Example: when today is 08/10/2026, COB 07/10/2026 selects live options on 08/10/2026; COB 06/10/2026 selects historical options for 06/10/2026. This uses calendar days. Weekends/holidays are not silently moved to another trading date. A historical failure never triggers live quotes for an older COB.

**A yesterday-COB archive deliberately contains historical asset prices through COB and current option snapshots from the following day. They are not a synchronized historical surface.** Every live result records its capture date separately. The adapter checks the London date before each batch and stops starting new snapshots if the date changes.

## Inputs and portfolio

The portfolio is **50% SPY and 50% IEF**, reported in EUR, in `config/portfolio.json`. SPX and SPX options are reference inputs for modelling SPY movements. They are not holdings. The configurable option-reference mapping records this intention; its calibration is later work.

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

Cleaning, aligned local-currency returns and descriptive historical statistics are implemented. Dividend forecasts, FX conversion, discount/forward curves, implied-volatility surfaces and Monte Carlo calibration remain later work. ESTR/SOFR observations and sovereign yields are not substitutes for an OIS discount curve. EUR/USD is USD per EUR; conversion to EUR requires the corresponding dated FX observation.

IBKR documentation: [historical bars](https://interactivebrokers.github.io/tws-api/historical_bars.html), [historical limitations](https://interactivebrokers.github.io/tws-api/historical_limitations.html).

Console results distinguish `[HISTORICAL]` from `[LIVE SNAPSHOT]`. Existing request IDs ending in `_HISTORY` are stable identifiers retained for archive reuse; they do not select the endpoint. New result metadata records the executed `operation` separately from `configured_operation`.
