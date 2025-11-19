# NeuroVest

Economic forecasting system using ensemble ML. Predicts market opportunities by analyzing regime shifts, cross-asset dynamics, and macro indicators.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## What This Does

NeuroVest trains XGBoost, LightGBM, and CatBoost models on 126 features to predict multi-day price movements. The system uses percentile-based thresholds to generate CRASH/NORMAL/SPIKE signals that feed into a backtest with ATR-based stops and confidence-weighted position sizing.

Current performance: ~59% accuracy, ~0.6 AUC, balanced 30/40/30 prediction distribution.

If you're seeing 98%+ accuracy, something is wrong (probably data leakage).

---

## Quick Start

```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt

# Download data
python3 update_spy_data.py
python3 download_crypto_data.py

# Train models
python3 train_multi_asset.py

# Generate predictions
python3 predict_multi_asset_ensemble.py

# Run backtest
python3 backtest.py
```

Each script shows a menu at the end with next steps you can choose from.

---

## Training Options

**Standard training** (5-10 min):
```bash
python3 train_multi_asset.py
```

**With hyperparameter tuning** - uses RandomizedSearchCV with TimeSeriesSplit to find optimal parameters for each model:
```bash
python3 train_multi_asset.py --tune        # Full search, 15-30 min
python3 train_multi_asset.py --tune-fast   # Quick search, 5-10 min
```

Tuned parameters get saved to `models/best_hyperparameters.json` and automatically used in future training runs.

**Multi-horizon training** - train separate models for different time horizons. Thresholds auto-calibrate based on expected price movement (threshold × √days):
```bash
python3 train_multi_horizon_signals.py                    # 1d, 3d, 5d
python3 train_multi_horizon_signals.py --horizons 1 3     # specific horizons
python3 train_multi_horizon_signals.py --tune             # with tuning
```

---

## Backtesting

The backtest includes confidence-based position sizing, volatility targeting, and ATR-based stop losses. Run with the optimized config:

```bash
python3 backtest.py --config configs/backtest_optimized.json
```

Key settings in that config:
- `target_ann_vol: 0.15` - targets 15% annualized volatility
- `conf_size_bounds: [0.5, 1.5]` - scales position 0.5x to 1.5x based on model confidence
- `sl_atr: 0.75`, `tp_atr: 1.25` - stop loss and take profit in ATR multiples
- `use_regime_filter: true` - only takes longs in uptrends, shorts in downtrends

---

## Multi-Asset Support

The system trains on SPY + BTC/ETH/SOL by default (~10k samples total). You can also:

**Train individual assets:**
```bash
python3 framework/train_unified.py --asset GLD
```

**Backtest different assets:**
```bash
python3 backtest.py --asset BTC/USDT
python3 backtest.py --asset-group crypto --compare
```

**Portfolio backtesting:**
```bash
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance monthly
```

---

## How Predictions Work

The prediction pipeline:

1. Loads trained XGBoost, LightGBM, CatBoost models
2. Generates probability scores from each
3. Averages them for ensemble probability
4. Uses percentile-based thresholds (30th/70th) to classify:
   - Bottom 30% → CRASH (short signal)
   - Middle 40% → NORMAL (hold)
   - Top 30% → SPIKE (long signal)

This approach ensures balanced signal distribution regardless of the actual probability range the models output.

---

## Project Structure

```
train_multi_asset.py              # Main training script
train_multi_horizon_signals.py    # Multi-horizon training
hyperparameter_tuning.py          # Standalone HP optimization
predict_multi_asset_ensemble.py   # Generate predictions
backtest.py                       # Backtesting engine
diagnose_system.py                # System diagnostics
utils.py                          # Feature engineering (126 features)
config.py                         # Global config

models/                           # Trained models (.pkl)
logs/                             # Predictions, trade logs
configs/                          # Backtest configurations
data/                             # Price data
data_cache/                       # Downloaded asset data
```

---

## Feature Engineering

126 features across 5 categories:

**Technical (40):** RSI, Stochastic, Bollinger Bands, Keltner Channels, ATR, various moving averages and momentum indicators.

**Cross-Asset (24):** Credit spreads (HYG/LQD), bond yields, dollar strength (DXY), crypto volatility, sector dispersion.

**Macro (18):** 10Y yield, yield curve slope, rate changes, recession signals.

**Regime (32):** VIX-based regimes, trend strength, volatility regimes, business cycle indicators.

**Interactions (12):** Non-linear combinations like Near_52w_High × Volatility, Credit_Ratio × Volatility.

All features are lagged by at least 1 day to prevent lookahead bias.

---

## Configuration

Training parameters in `config.py`:

```python
TRAIN_CFG = {
    "horizon": 1,              # Forward return horizon (days)
    "pos_threshold": 0.005,    # 0.5% minimum return for positive label
    "fee_bps": 1.5,            # Transaction fees
    "slippage_bps": 2.0,       # Slippage assumption
    "weight_power": 1.75,      # Sample weighting exponent
}
```

Asset-specific thresholds are calibrated to ~0.5x daily volatility:
- SPY: 0.6% (daily vol ~1.2%)
- BTC: 2.2% (daily vol ~4.4%)
- ETH: 2.8% (daily vol ~5.7%)
- SOL: 4.0% (daily vol ~8.0%)

---

## Diagnostics

If things aren't working, run:

```bash
python3 diagnose_system.py
```

This checks data files, model loading, prediction distribution, and backtest signals. Common issues:

- **99% accuracy:** Data leakage. Check that forward-looking columns are excluded from features.
- **0% NORMAL predictions:** Threshold issue. The system now uses percentile-based thresholds to fix this.
- **Only a few trades:** Thresholds too strict. Percentile approach should give ~30% crash, 40% normal, 30% spike.

---

## Framework API

There's also a REST API and framework for managing 80+ assets:

```bash
python framework/download_all_assets.py
python framework/train_unified.py --all
python framework/api_server.py
```

See [FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md) for details.

---

## What This Isn't

This is a research/educational project. It's not:
- A trading bot you should run with real money
- Financial advice
- Guaranteed to be profitable

Test accuracy is ~59%. That means ~41% of signals will be wrong. The backtest shows modest positive returns over 25 years, but past performance doesn't predict future results.

If you want to use this for real trading, paper trade for at least 6-12 months first, use proper position sizing, and start with amounts you can afford to lose entirely.

---

## Documentation

- [FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md) - Full framework documentation
- [TRAINING_SYSTEMS_GUIDE.md](TRAINING_SYSTEMS_GUIDE.md) - Different training approaches
- [ACCURACY_OPTIMIZATION_GUIDE.md](ACCURACY_OPTIMIZATION_GUIDE.md) - Threshold tuning
- [CRASH_PREDICTION_ANALYSIS.md](CRASH_PREDICTION_ANALYSIS.md) - Why crash detection is tricky
- [MULTI_ASSET_ANALYSIS_SUMMARY.md](MULTI_ASSET_ANALYSIS_SUMMARY.md) - Portfolio analysis tools

---

## Requirements

```
Python 3.8+
pandas, numpy, scikit-learn
xgboost, lightgbm, catboost
yfinance, ccxt
ta-lib
```

---

## License

MIT

---

## Author

keeg-Hson

Last updated: November 2025
