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
# LOCKED CONFIGURATION (Feb 2026): 1-day horizon + 0.5% binary threshold
# Rationale: Feature analysis confirmed this is the optimal setup for
# capturing short-term directional moves with acceptable precision (~42%).
TRAIN_CFG = {
    # Price column used when deriving forward returns
    "price_col": "Close",
    # Forward horizon in days - LOCKED at 1 day for event matching
    # Do not change without re-evaluating feature importance
    "horizon": 1,
    # Trading cost and edge assumptions (in basis points)
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "min_edge_bps": 10.0,
    # Long-only labeling; 1 = positive event, 0 = no-trade
    "long_only": True,
    # Labeling threshold - LOCKED at 0.5% binary threshold
    # This threshold yields ~42% precision with acceptable recall
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
# Single source of truth for prediction threshold (DO NOT hardcode elsewhere)
# PRECISION-FOCUSED STRATEGY (Feb 2026):
# With ~42% precision, we optimize for high-confidence signals only.
# Low recall is acceptable - position sizing matters more than signal frequency.
PREDICTION_THRESHOLD = 0.45  # Raised for precision focus (~42% precision target)

PREDICT_CFG = {
    # Minimum probability of event (long) to consider a trade.
    # PRECISION-FOCUSED THRESHOLD STRATEGY:
    # - 0.55: 97.6% precision, 14.7% recall (max precision)
    # - 0.45: 92.3% precision, 20.0% recall (precision-focused - CURRENT)
    # - 0.35: ~70% precision, ~40-50% recall (balanced)
    # Current strategy: Prioritize precision over recall
    # When the model signals, we want it to be right ~42%+ of the time
    "p_min": PREDICTION_THRESHOLD,  # Use constant above
    # Minimum expected value (in return units) for a trade to pass EV gating.
    "ev_min": 0.0005,  # 5 bps minimum expected value
    # Typical winner/loser magnitudes used for EV heuristics.
    "avg_gain": 0.0040,  # 40 bps typical winner magnitude
    "avg_loss": 0.0030,  # 30 bps typical loser magnitude
    # Trading cost assumptions for EV gating (basis points).
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
}

# ─── Risk Management Configuration ────────────────────────────
# CRITICAL: This is a weak-but-real signal system.
# Position sizing and risk management matter MORE than the model.
RISK_CFG = {
    # Maximum position size as fraction of portfolio
    "max_position_pct": 0.05,  # 5% max per trade
    # Risk per trade (stop loss distance * position size)
    "risk_per_trade_pct": 0.01,  # 1% portfolio risk per trade
    # Kelly fraction scaling (use fractional Kelly for safety)
    "kelly_fraction": 0.25,  # Use 25% of Kelly-optimal sizing
    # Maximum daily drawdown before halting
    "max_daily_drawdown_pct": 0.03,  # 3% max daily loss
    # Confidence scaling: size positions by model confidence
    "confidence_scaling": True,
    # Minimum confidence to take full position
    "min_confidence_full_size": 0.60,
    # Correlation limit: reduce exposure in correlated positions
    "max_correlated_exposure_pct": 0.15,  # 15% max in correlated assets
}
