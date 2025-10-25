#config.py

from pathlib import Path

# ─── Project folders ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()   
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)                # make sure it exists

# ─── Project file paths ─────────────────────────────────────────
SPY_DAILY_CSV = DATA_DIR / "SPY.csv"

# ─── Training configuration ─────────────────────────────────────────
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
    "use_forward_returns": True,  # flip to True to try the forward-returns labeling path (5590)

}

# Prediction / decision gating configuration
PREDICT_CFG = {
    # Decision gates (tune with backtests/sweeps)
    "p_min": 0.55,         # minimum probability to even consider a trade
    "ev_min": 0.0005,      # min expected value (e.g., 5 bps) to allow a trade

    # Expectancy components (set from rolling stats/backtests)
    "avg_gain": 0.0040,    # 40 bps typical winner magnitude
    "avg_loss": 0.0030,    # 30 bps typical loser magnitude

    # Frictions
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
}


