"""Process a saved archive, or download/reuse data and then process it.

python run.py       : choose local processing, download or demo, then enter any required dates
python run.py demo  : first run: download from Yahoo and ECB, no keys or date prompts
python -i run.py    : execute and keep result available in a Python prompt
"""
import os
import sys
from datetime import timedelta
from pathlib import Path
from src import MarketDataDownloader, inspect_archive, process_inventory, Portfolio
from src.cob import parse_cob, parse_history_start, london_today
from src.preprocessing import ArchiveNotFound

ROOT = Path(__file__).resolve().parent

INVENTORY_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "processed"

# Input data details live ONLY in these JSON files. Disable a provider/request
# with enabled=false in JSON, or change which files are loaded here.
CONFIG_DIR = ROOT / "config"
CONFIG_FILES = [CONFIG_DIR / (name + ".json") for name in
                ("yahoo", "alpha_vantage", "marketdata", "massive", "fred", "ecb", "ibkr")]
PORTFOLIO_FILE = CONFIG_DIR / "portfolio.json"
CLEANING_FILE = CONFIG_DIR / "cleaning.json"
MODELING_FILE = CONFIG_DIR / "modeling.json"
MONTE_CARLO_FILE = CONFIG_DIR / "monte_carlo.json"
OIS_FILE = CONFIG_DIR / "ois.json"  # Fixed OIS par quotes reused for every COB; no API.

# Credentials/operational setup remain in Python; never commit actual keys.
SETTINGS = {
    "ALPHA_VANTAGE_API_KEY": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "MARKETDATA_API_TOKEN": os.environ.get("MARKETDATA_API_TOKEN", ""),
    "MASSIVE_API_KEY": os.environ.get("MASSIVE_API_KEY", ""),
    "FRED_API_KEY": os.environ.get("FRED_API_KEY", ""),
    "ALPHA_VANTAGE_OUTPUTSIZE": "full",   # Full history requires account access.
    "ALPHA_VANTAGE_ADJUSTED": "false",
    "MARKETDATA_ADJUSTMENT": "split_only",  # Or "split_and_dividend".
    "YAHOO_MIN_INTERVAL": "1",
    "ALPHA_VANTAGE_MIN_INTERVAL": "13",
    "MARKETDATA_MIN_INTERVAL": "1",
    "MASSIVE_MIN_INTERVAL": "13",
    "FRED_MIN_INTERVAL": "0.6",
    "ECB_MIN_INTERVAL": "0.3",
    # TWS session authentication: no API key. Match the socket port shown in TWS.
    "IBKR_HOST": "127.0.0.1",
    "IBKR_PORT": os.environ.get("IBKR_PORT", "7496"),  # Matches current TWS; paper sessions usually use 7497.
    "IBKR_CLIENT_ID": "81",  # Unique to this library; never use another connected client ID.
    "IBKR_TIMEOUT": "20",
    "IBKR_MIN_INTERVAL": "1",
    "IBKR_OPTION_MAX_CONCURRENT": "50",  # One option connection; at most fifty concurrent snapshots, used when COB is yesterday.
    "IBKR_HISTORY_MAX_CONCURRENT": "5",
    "IBKR_HISTORY_REQUEST_INTERVAL": "0.25",
    "IBKR_HISTORY_TIMEOUT": "60",
}

# Preserve native provider responses. Inventory reports are saved separately.
RAW_DUMP_DIR = ROOT / "tests" / "data_dumps"

# The demo uses only providers that need no key or broker session. Its archive is
# kept apart: a COB archive with different inputs must not replace a full one.
# History starts before the 2008 stress window configured for Monte Carlo.
DEMO_CONFIG_FILES = [CONFIG_DIR / "yahoo.json", CONFIG_DIR / "ecb.json"]
DEMO_PORTFOLIO_FILE = CONFIG_DIR / "demo" / "portfolio.json"
DEMO_DUMP_DIR = ROOT / "data" / "demo_dumps"
DEMO_HISTORY_START = "01/01/2005"

def download_main(cob_text=None, history_start_text=None, *, retry_failed=False):
    today = london_today()
    yesterday = today - timedelta(days=1)
    print("\nMKT DATA DOWNLOADER")
    print("=" * 58)
    print(f"Today (Europe/London): {today:%d/%m/%Y}")
    print(f"Recommended COB (close of business): {yesterday:%d/%m/%Y} (yesterday)")
    print("\nChoose yesterday for the fullest available set of simulation inputs:")
    print("  - Configured products: your selected history start through COB.")
    print("  - Related options configured in JSON: current live quotes, implied volatilities")
    print("    and Greeks, where available. Captured today, not at yesterday's close.")
    print("\nFor an older COB without a completed local archive:")
    print("  - Only historical data is requested; no current live option quotes.")
    print("  - IBKR historical option bars do not include historical IV or Greeks.")
    print("    The planned simulation would therefore use historical volatility.")
    print("\nA completed matching COB archive is reused without new API requests.")
    print("Previously saved option volatilities remain available if captured successfully;")
    print("their actual observation dates must be checked before using them.")
    print("\nCurrent stage: retrieve/reuse inputs, then process the saved archive.")
    print("Inventory, cleaning, statistics and configured historical Monte Carlo follow automatically.")
    print("Manual OIS par curves are loaded from config/ois.json and saved beside API data.")
    print("The same fixed OIS curves are used for every COB, retaining their original observation dates.")
    print("Live data and usable volatility coverage depend on IBKR availability and access.\n")
    # The prompt is at the entry point; the Python library never calls input().
    if cob_text is None:
        while True:
            try:
                cob = parse_cob(input("COB date (DD/MM/YYYY; yesterday or earlier): "))
                break
            except ValueError as exc:
                print(exc)
    else:
        cob = parse_cob(cob_text)
    print("\nChoose the historical start date for all configured products/reference series.")
    print("If trading/data began later, only available observations are returned.")
    print("If the range is rejected, we try the provider's maximum accessible history")
    print("through COB and record the fallback and any limits. Access/connectivity errors")
    print("cannot be fixed by changing dates. Options keep their separate COB/live policy.")
    if history_start_text is None:
        while True:
            try:
                history_start = parse_history_start(input("History start (DD/MM/YYYY; on or before COB): "), cob)
                break
            except ValueError as exc:
                print(exc)
    else:
        history_start = parse_history_start(history_start_text, cob)
    print(f"Requested history: {history_start:%d/%m/%Y} through {cob:%d/%m/%Y}")
    downloader = MarketDataDownloader.from_json(CONFIG_FILES, settings=SETTINGS,
        cob_date=cob, history_start=history_start, progress=print)
    result = downloader.download_cob(cob, RAW_DUMP_DIR,
        portfolio_file=PORTFOLIO_FILE, retry_failed=retry_failed, ois_file=OIS_FILE)
    print(result.summary[["ticker", "provider", "status", "retrieval_method"]].to_string(index=False))
    print(f"{'Reused' if result.reused else 'Saved'} COB archive: {result.folder}")
    print("Inspect manifest.json and summary.json for unavailable fields or provider limits.")
    print("\nArchive ready. Starting inventory for the same COB; no further API requests.")
    try:
        _inventory_archive(result.folder)
    except (OSError, ValueError):
        print(f"Local processing could not finish. The raw archive remains at: {result.folder}")
        print("After resolving the reported issue, choose Process saved archive to try again without APIs.")
        raise
    return result


def demo_main():
    """Keyless first run: download with fixed choices, then the same local pipeline."""
    cob = london_today() - timedelta(days=1)
    history_start = parse_history_start(DEMO_HISTORY_START, cob)
    print("\nDEMO: Yahoo and ECB data, no API keys, broker session or date prompts")
    print("=" * 58)
    print(f"COB: {cob:%d/%m/%Y} | History start: {history_start:%d/%m/%Y}")
    downloader = MarketDataDownloader.from_json(DEMO_CONFIG_FILES, settings=SETTINGS,
        cob_date=cob, history_start=history_start, progress=print)
    result = downloader.download_cob(cob, DEMO_DUMP_DIR,
        portfolio_file=DEMO_PORTFOLIO_FILE, ois_file=OIS_FILE, retry_failed=True)
    print(result.summary[["ticker", "provider", "status", "retrieval_method"]].to_string(index=False))
    print(f"{'Reused' if result.reused else 'Saved'} COB archive: {result.folder}")
    _inventory_archive(result.folder, DEMO_PORTFOLIO_FILE)
    return result


def inventory_main(cob_text=None):
    """Inspect local data without constructing a downloader or checking API keys."""
    print("\nPORTFOLIO SIMULATOR")
    print("Saved market data: inventory, cleaning, statistics and historical Monte Carlo")
    print("=" * 58)
    print("Reads an existing completed archive. No provider connections are made.")
    print("Checks coverage, cleans data, then saves statistics and configured historical simulations.")
    print("Use the archive's COB label, even if option captures occurred later.\n")
    if cob_text is None:
        while True:
            try:
                cob = parse_cob(input("Saved archive COB (DD/MM/YYYY): "))
                break
            except ValueError as exc:
                print(exc)
    else:
        cob = parse_cob(cob_text)
    return _inventory_archive(RAW_DUMP_DIR / cob.isoformat())


def _inventory_archive(folder, portfolio_file=None):
    """Run the same local inspection after a download or as a standalone step."""
    inventory = inspect_archive(folder, progress=print, include_observations=True)
    json_path, report_path = inventory.save(INVENTORY_DIR / inventory.cob)
    print(f"\nInspected {len(inventory.requests):,} saved requests.")
    print(f"Reader errors: {inventory.metadata['requests_with_reader_errors']}")
    print("A reader error means a saved response could not be trusted or understood.")
    print("Provider failures, partial data and later option capture dates are listed separately.")
    print(f"Readable report: {report_path}")
    print(f"Detailed inventory: {json_path}")
    _process_inventory(inventory, portfolio_file)
    return inventory


def _process_inventory(inventory, portfolio_file=None):
    result = process_inventory(inventory, CLEANING_FILE, portfolio_file=portfolio_file or PORTFOLIO_FILE, progress=print)
    destination = result.save(PROCESSED_DIR / inventory.cob / result.run_id)
    weights = ", ".join(f"{h['request_id']} {h['weight']:.0%}" for h in result.summary["portfolio"]["holdings"])
    print(f"Processing portfolio: {weights}")
    print(f"Aligned holding returns: {result.summary['readiness']['aligned_return_observations']:,}")
    print(f"Preliminary option IV candidates: {result.summary['options']['iv_candidates']:,}")
    print("Candidate quotes still require synchronization and calibration checks.")
    if result.summary["readiness"]["foreign_currencies"] and not result.summary["readiness"]["available_fx_series"]:
        print("FX history is missing: reporting-currency analysis is not ready.")
    print(f"Cleaned/merged data and report: {destination}")
    print("Computing historical statistics and stress comparisons for configured holdings.")
    # Creating the Portfolio computes its statistics; the later simulation,
    # report and plots are methods of this same object.
    portfolio = Portfolio(result, MODELING_FILE, MONTE_CARLO_FILE)
    statistics = portfolio.statistics
    statistics.source_folder = destination.resolve()
    statistics_folder = statistics.save(destination / "statistics" / statistics.run_id)
    print(f"Statistics portfolio status: {statistics.summary['portfolio_status']}")
    for notice in statistics.summary["warnings"]:
        print("Statistics: " + notice)
    print(f"Historical statistics, tables and plots: {statistics_folder}")
    print("Running configured historical Monte Carlo scenarios (offline).")
    # Monte Carlo consumes cleaned holdings/references directly, not the
    # descriptive statistics tables. Its own JSON controls calibration and
    # scenarios; this offline stage makes no further provider requests.
    # The Portfolio passes on the cleaned source to protect it during export.
    # Save in a separate simulation-ID folder tracking inputs, settings and code.
    simulation = portfolio.monte_carlo()
    simulation_folder = simulation.save(destination / "monte_carlo" / simulation.run_id)
    print(f"Monte Carlo status: {simulation.summary['status']}")
    for notice in simulation.summary["warnings"]:
        print("Monte Carlo: " + notice)
    print(f"Monte Carlo report and results: {simulation_folder}")
    portfolio_folder = destination / "portfolio"
    portfolio_folder.mkdir(exist_ok=True)
    (portfolio_folder / "report.md").write_text(portfolio.report(), encoding="utf-8")
    portfolio.plots_report(portfolio_folder / "plots")
    print(f"Portfolio report and plots: {portfolio_folder}")
    print("Raw archive unchanged. Option/local-volatility calibration is not run.")
    return result


def choose_mode():
    """Ask at the entry point; no workflow starts until a valid choice is made."""
    print("\nPORTFOLIO SIMULATOR - Choose a workflow")
    print("  1. Process saved archive: inventory, clean, merge and analyse (no API requests).")
    print("  2. Download + process: retrieve/reuse, inventory, clean, merge and analyse.")
    print("  3. Demo: first run with Yahoo and ECB only; no keys, broker or date prompts.")
    choices = {"1": "inventory", "inventory": "inventory", "process": "inventory",
               "2": "download", "download": "download", "3": "demo", "demo": "demo"}
    while True:
        choice = input("Choose mode (1 = process saved archive, 2 = download + process, 3 = demo): ").strip().lower()
        if choice in choices:
            return choices[choice]
        print("Please enter 1 (inventory), 2 (download) or 3 (demo).")


def main(cob_text=None, history_start_text=None, *, mode=None, retry_failed=False):
    """Both workflows clean and analyse; local processing never downloads."""
    selected = choose_mode() if mode is None else mode
    if selected in {"inventory", "process"}:
        if history_start_text is not None or retry_failed:
            raise ValueError("History start/retry parameters require mode='download'.")
        return inventory_main(cob_text)
    if selected == "download":
        return download_main(cob_text, history_start_text, retry_failed=retry_failed)
    if selected == "demo":
        if cob_text is not None or history_start_text is not None or retry_failed:
            raise ValueError("The demo uses fixed dates; date/retry parameters require mode='download'.")
        return demo_main()
    raise ValueError("Mode must be 'inventory', 'download' or 'demo'.")


if __name__ == "__main__":
    try:
        result = main(mode=sys.argv[1] if len(sys.argv) > 1 else None)
    except ArchiveNotFound as exc:
        sys.exit(f"\n{exc}\nDownload it first with option 2, or run the keyless demo: python run.py demo")
