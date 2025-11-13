# config.py
"""
Global configuration for NeuroVest.

Paths
-----
Defines base folders for data, models, logs, and outputs.

Training configuration (TRAIN_CFG)
----------------------------------
Core label and weighting settings for model training:
- horizon      : forward return horizon in days (1d event to match eval_join/evaluate.py).
- pos_threshold: minimum forward net return to label an event (0.5% = 0.005).
- fee_bps      : per-trade fee in basis points.
- slippage_bps : assumed slippage in basis points.
- min_edge_bps : additional edge over costs; used by some label/EV logic.
- long_only    : when True, model is trained/evaluated only for long events.
- weighting    : sample weights emphasize larger forward returns.

Prediction configuration (PREDICT_CFG)
--------------------------------------
Default probability/EV gating for live decisions. Thresholds may be overridden
by sweep outputs (configs/best_thresholds.json, models/thresholds*.json).
"""

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
    # Price column used when deriving forward returns
    "price_col": "Close",
    # Forward horizon in days; must match evaluation horizon (Actual_Event)
    # and DuckDB eval_join forward-return construction.
    "horizon": 1,
    # Trading cost and edge assumptions (in basis points)
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "min_edge_bps": 10.0,
    # Long-only labeling; 1 = positive event, 0 = no-trade
    "long_only": True,
    # Labeling threshold: minimum net forward return to label a positive event.
    # 0.005 = +0.5% forward move; used by label logic to match Actual_Event_1d.
    "pos_threshold": 0.005,
    # Sample-weighting parameters for training:
    # weights ∝ |forward_return|^weight_power, clipped to [min_weight, max_weight].
    "min_weight": 0.50,
    "max_weight": 5.0,
    "weight_power": 1.75,
    # Use forward returns-based labeling pipeline
    "use_forward_returns": True,
}

# ─── Prediction / decision gating ────────────────────────────
PREDICT_CFG = {
    # Minimum probability of event (long) to consider a trade.
    # This acts as a default; live runs may override via thresholds_fwd.json
    # or configs/best_thresholds.json.
    "p_min": 0.55,
    # Minimum expected value (in return units) for a trade to pass EV gating.
    "ev_min": 0.0005,  # 5 bps minimum expected value
    # Typical winner/loser magnitudes used for EV heuristics.
    "avg_gain": 0.0040,  # 40 bps typical winner magnitude
    "avg_loss": 0.0030,  # 30 bps typical loser magnitude
    # Trading cost assumptions for EV gating (basis points).
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
}
