"""Run provider-specific requests from config/*.json in Python or VS Code.

python run.py       : ask for COB, reuse its archive or retrieve configured data
python -i run.py    : execute and keep result available in a Python prompt
"""
import os
from datetime import timedelta
from pathlib import Path
from src import MarketDataDownloader
from src.cob import parse_cob, london_today

ROOT = Path(__file__).resolve().parent

# Input data details live ONLY in these JSON files. Disable a provider/request
# with enabled=false in JSON, or change which files are loaded here.
CONFIG_DIR = ROOT / "config"
CONFIG_FILES = [CONFIG_DIR / (name + ".json") for name in
                ("yahoo", "alpha_vantage", "marketdata", "massive", "fred", "ecb", "ibkr")]
PORTFOLIO_FILE = CONFIG_DIR / "portfolio.json"

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

# Inspect native provider responses first. No cleaning/normalization in this mode.
RAW_DUMP_DIR = ROOT / "tests" / "data_dumps"

def main(cob_text=None, *, retry_failed=False):
    today = london_today()
    yesterday = today - timedelta(days=1)
    print("\nPORTFOLIO SIMULATOR")
    print("Monte Carlo: Historical versus Option-Implied Volatility")
    print("=" * 58)
    print(f"Today (Europe/London): {today:%d/%m/%Y}")
    print(f"Recommended COB (close of business): {yesterday:%d/%m/%Y} (yesterday)")
    print("\nChoose yesterday for the fullest available set of simulation inputs:")
    print("  - Configured products: available history through COB; earliest IBKR date, otherwise from 03/01/2000.")
    print("  - Related options configured in JSON: current live quotes, implied volatilities")
    print("    and Greeks, where available. Captured today, not at yesterday's close.")
    print("\nFor an older COB without a completed local archive:")
    print("  - Only historical data is requested; no current live option quotes.")
    print("  - IBKR historical option bars do not include historical IV or Greeks.")
    print("    The planned simulation would therefore use historical volatility.")
    print("\nA completed matching COB archive is reused without new API requests.")
    print("Previously saved option volatilities remain available if captured successfully;")
    print("their actual observation dates must be checked before using them.")
    print("\nCurrent stage: retrieve and save inputs. Simulation/calibration comes later.")
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
    downloader = MarketDataDownloader.from_json(CONFIG_FILES, settings=SETTINGS,
        cob_date=cob, progress=print)
    result = downloader.download_cob(cob, RAW_DUMP_DIR,
        portfolio_file=PORTFOLIO_FILE, retry_failed=retry_failed)
    print(result.summary[["ticker", "provider", "status", "retrieval_method"]].to_string(index=False))
    print(f"{'Reused' if result.reused else 'Saved'} COB archive: {result.folder}")
    print("Inspect manifest.json and summary.json for unavailable fields or provider limits.")
    return result


if __name__ == "__main__":
    result = main()
