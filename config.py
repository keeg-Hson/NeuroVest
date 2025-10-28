# config.py
from pathlib import Path

# ─── Project folders ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "data_cache"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist (idempotent)
for p in (DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ─── Project file paths ──────────────────────────────────────
SPY_DAILY_CSV = DATA_DIR / "SPY.csv"

# ─── Training configuration ──────────────────────────────────
TRAIN_CFG = {
    "price_col": "Close",
    "horizon": 5,
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "long_only": True,
    "pos_threshold": 0.0,
    "min_weight": 0.25,
    "max_weight": 5.0,
    "weight_power": 1.25,
    "use_forward_returns": True,
}

# ─── Prediction / decision gating ────────────────────────────
PREDICT_CFG = {
    "p_min": 0.55,  # minimum probability to consider a trade
    "ev_min": 0.0005,  # 5 bps minimum expected value
    "avg_gain": 0.0040,  # 40 bps typical winner magnitude
    "avg_loss": 0.0030,  # 30 bps typical loser magnitude
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
}
