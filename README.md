# PythonToolBox_Taller1Master - market_data

**Market-data requests live in JSON. Python setup stays in run.py.**

Edit the six files under `config/`: `yahoo.json`, `alpha_vantage.json`,
`marketdata.json`, `massive.json`, `fred.json` and `ecb.json`.
Each has active `requests`, provider-specific `instructions`, `product_guides`
with examples, and official documentation links in `sources`.

## Which identifiers to enter

| Provider | JSON input | Products implemented by this toolbox |
|---|---|---|
| Yahoo | `input.symbol` | Daily stock, ETF and index prices; raw SPX/SPXW option bars |
| Alpha Vantage | `input.symbol` from its symbol search | Daily stock/ETF prices |
| MarketData.app | `input.symbol` | Daily stock/ETF candles; raw SPX/SPXW option quotes |
| Massive | `input.ticker` from its reference data | Daily stock/ETF aggregates; raw SPX/SPXW option bars |
| FRED | `input.series_id`, e.g. DGS10 | Yields, FX and macro observations at native frequency |
| ECB | `input.dataflow` plus `input.series_key` | Daily/business-daily yields, FX and macro observations |

Provider identifiers are not interchangeable. Confirm instrument, listing,
currency and units even if two providers happen to use the same ticker string.
FRED/ECB observations remain separate from tradable asset prices.

## Fill a request

Edit the `requests` array in the provider's JSON file. A stock/ETF request needs:

- `id`: your logical request name, e.g. SPY; used in result keys.
- `product`: stock, etf or index where supported; `operation`: price_history.
- `ticker`: your readable label; `input`: the exact provider-native identifier.
- `description`, `asset_class`, `currency`, `unit`, `exchange`: listing metadata.
- `timeframe`: lookback OR start, plus optional end. For example
  `{"lookback":"1y","end":null}` or
  `{"start":"2024-01-01","end":"2025-12-31"}`.
- `frequency`: 1d for prices/ECB, native for FRED; intraday is not implemented.
- `parameters`: only the choices documented for that provider/product.
- `enabled`: true to retrieve, false to skip.

Rate/FX/macro entries use product yield/fx/macro and operation observations;
their input identifiers and units differ from stock prices. JSON uses double
quotes, true/false/null and no comments. Instructions are regular JSON fields.

For a comparison across providers, repeat the same logical id and identical
instrument/timeframe metadata in their JSON files, using each provider's input
identifier. Results stay separate by provider. If the dates, product or listing
differ, use a different id. Duplicate IDs within a provider are rejected.

A provider file can be disabled with its top-level enabled=false. Only entries
under requests execute. Examples under product_guides are documentation only.

## Options are a separate product

The guides include supported single-contract history examples and the
provider identifier field needed to select one contract. An option requires an underlying,
expiry, call/put and strike (or an exact discovered contract ID), with currency,
exercise style, multiplier and deliverable checked against provider metadata.
Expiry and the observation date are different dates.

**Raw SPX/SPXW option history is implemented for Yahoo, MarketData.app and Massive.**
Their JSON requests now include a European call and put, strike 7000, expiration
2026-12-18, standard SPX AM cash settlement, multiplier USD 100 per index point.
These are index options; SPY ETF options are a different product.

Use product=index_option, operation=option_history, and a separate contract
object for underlying, root, expiration, strike, option_type, exercise_style,
settlement, settlement_time and multiplier. Native identifiers are input.symbol
for Yahoo, input.optionSymbol for MarketData.app, and input.ticker (O: prefix)
for Massive. The loader rejects identifiers that disagree with contract details.
Use an empty parameters object. SPX uses AM settlement; SPXW uses PM settlement.

The observation timeframe remains lookback=1mo/end=null; expiration is independent
and does not roll automatically. MarketData returns historical end-of-day quotes;
Yahoo chart data and Massive trade aggregates are different response types.
They remain separate, with response_type and contract metadata in the dumps.
MarketData documents historical IV/Greeks as null; none are calculated here.
Massive saves a contract-reference response before the price-bar responses.

Alpha Vantage's current key returned a premium endpoint error for historical
options. Its two examples are disabled; SPX coverage remains unverified and its
date-by-date option-history adapter is not implemented. FRED and ECB have no
individual-option history requests. Each provider JSON explains its status under
product_guides.index_option for market-data providers. FRED/ECB guides cover
economic observations only. Option-chain retrieval is not implemented.
download_raw() handles options; download() rejects option configurations before
network access because standardized option tables are not implemented yet.

## Run and inspect in Python or VS Code

Use the existing `.venv` interpreter. Run `run.py`; it loads the JSON files using
MarketDataDownloader.from_json(), checks local credentials, and saves raw responses.
Each run creates `tests/data_dumps/<UTC timestamp>_<unique ID>/`, with one folder
per provider and request. Open `summary.csv` first, then the native `response_*.json`
or `response_*.csv` files. `request.json`, `status.json` and `responses.json`
record inputs, failures, response timestamps, HTTP status and file checksums.
Missing credentials produce status files without fake data. Earlier runs are kept.
In a terminal opened here:

```powershell
.\.venv\Scripts\python.exe -i run.py
```

The -i flag leaves the returned Python objects available:

```python
result.summary
result.folder  # Open this folder to inspect the native response files
```

Direct library use:

```python
from market_data import MarketDataDownloader

downloader = MarketDataDownloader.from_json("config")  # directory, file, or list of files
result = downloader.download_raw("tests/data_dumps")
```

Use an absolute config path when running from another directory. The library
also accepts a normalized dictionary through from_config() and direct ticker
calls, but run.py's inputs come exclusively from JSON.

## Runtime settings and permissions

`run.py` retains API credential environment-variable names, pacing settings,
optional cache/export paths and the list of JSON files to load. Credentials do
not belong in JSON. Request parameters in JSON override corresponding Python
defaults. The JSON loader validates supported input fields and product/endpoint
combinations before making requests.

`access_status()` shows local credential readiness; it does not prove remote
account permission. Authentication, quota, subscription and network failures
appear in result.summary. Four keyed providers require your own credentials.

Raw mode saves received response bodies before table conversion or cleaning.
It does not use the normalized-data cache. Pagination pages and HTTP retries stay
separate. Provider-selected price adjustments still apply at source; no additional
adjustments are made locally. Yahoo uses its chart JSON endpoint directly, with
UTC-midnight period boundaries; native timestamps and timezone metadata are kept.
No headers, cookies or keys are recorded; echoed configured keys are redacted
and the response metadata flags that exception to byte preservation.
`response_saved` means capture succeeded, not that coverage or quality passed.

For later toolbox work, `downloader.download()` remains available explicitly;
it returns standardized tables and performs the existing normalization checks.
Those normal downloads stay in memory unless output/cache paths are supplied.
Requested dates and actual coverage remain separate. An ok result is based on
a seven-day boundary heuristic, not a full exchange-calendar audit. Do not mix
raw, split-adjusted and dividend-adjusted prices without checking metadata.

## Code and installation

`run.py` -> `MarketDataDownloader.from_json()` -> `json_config.py` validation ->
MarketDataDownloader.download_raw() -> provider adapters -> native response files.
The optional download() method follows the separate normalization pipeline.
All implementation is in `src/market_data`; tests are in `tests`.
With Python 3.11+, install using `python -m pip install -e .` from this folder.
Run tests with `python -m unittest discover -s tests -v`.
The original PDFs are project references. Analytics and simulation are later work.
