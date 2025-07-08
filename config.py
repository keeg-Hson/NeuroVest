#config.py

from pathlib import Path

# ─── Project folders ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()   
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)                # make sure it exists

# ─── Project file paths ─────────────────────────────────────────
SPY_DAILY_CSV = DATA_DIR / "spy_daily.csv"


