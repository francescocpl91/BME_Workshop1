"""Run provider-specific requests from config/*.json in Python or VS Code.

python run.py       : execute the JSON requests
python -i run.py    : execute and keep result available in a Python prompt
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from market_data import MarketDataDownloader

# Input data details live ONLY in these JSON files. Disable a provider/request
# with enabled=false in JSON, or change which files are loaded here.
CONFIG_DIR = ROOT / "config"
CONFIG_FILES = [CONFIG_DIR / (name + ".json") for name in
                ("yahoo", "alpha_vantage", "marketdata", "massive", "fred", "ecb")]

# Credentials/operational setup remain in Python; never commit actual keys.
SETTINGS = {
    "ALPHA_VANTAGE_API_KEY": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "MARKETDATA_API_TOKEN": os.environ.get("MARKETDATA_API_TOKEN", ""),
    "MASSIVE_API_KEY": os.environ.get("MASSIVE_API_KEY", ""),
    "FRED_API_KEY": os.environ.get("FRED_API_KEY", ""),
    "ALPHA_VANTAGE_OUTPUTSIZE": "compact",   # Full history requires account access.
    "ALPHA_VANTAGE_ADJUSTED": "false",
    "MARKETDATA_ADJUSTMENT": "split_only",  # Or "split_and_dividend".
    "YAHOO_MIN_INTERVAL": "1",
    "ALPHA_VANTAGE_MIN_INTERVAL": "13",
    "MARKETDATA_MIN_INTERVAL": "1",
    "MASSIVE_MIN_INTERVAL": "13",
    "FRED_MIN_INTERVAL": "0.6",
    "ECB_MIN_INTERVAL": "0.3",
}

# Inspect native provider responses first. No cleaning/normalization in this mode.
RAW_DUMP_DIR = ROOT / "tests" / "data_dumps"

# JSON parameters override provider-specific defaults for each request.
CACHE_DIR = None       # Optional: ROOT / "data" / "cache"
CACHE_HOURS = 24
OUTPUT_DIR = None      # Optional: ROOT / "data" / "runs"


def main():
    downloader = MarketDataDownloader.from_json(
        CONFIG_FILES, settings=SETTINGS, cache_dir=CACHE_DIR,
        cache_hours=CACHE_HOURS, output_dir=OUTPUT_DIR, progress=print,
    )
    print(downloader.access_status().to_string(index=False))
    result = downloader.download_raw(RAW_DUMP_DIR)
    print(result.summary[["request_id", "provider", "status", "responses_saved", "message"]].to_string(index=False))
    print(f"Raw response files: {result.folder}")
    return result


if __name__ == "__main__":
    result = main()
