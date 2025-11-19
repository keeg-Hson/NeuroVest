# NeuroVest Training Systems Guide

**Understanding the three training approaches and when to use each**

---

## Overview: Three Training Systems

NeuroVest has **three different training systems** for different use cases:

| System | File | Best For | Output |
|--------|------|----------|--------|
| **1. SPY Ensemble** | `train_multi_asset.py` | SPY predictions | Multi-asset models for SPY |
| **2. Per-Asset Framework** | `framework/train_unified.py` | Individual assets | Separate models per asset |
| **3. Original SPY** | `train.py` | Advanced SPY training | Single SPY model |

---

## System 1: SPY Multi-Asset Ensemble (Recommended for SPY)

**Purpose:** Train models on multiple assets (SPY + crypto) to improve SPY predictions.

**How it works:**
1. Loads SPY data (6,500+ samples)
2. Loads crypto data (BTC, ETH, SOL - 4,000+ samples)
3. Combines into single training set (~10,500 samples)
4. Trains XGBoost, LightGBM, CatBoost
5. Creates ensemble predictions for SPY

### Usage

```bash
# Train the multi-asset ensemble
python3 train_multi_asset.py

# Generate predictions
python3 predict_multi_asset_ensemble.py

# Evaluate
python3 evaluate.py

# Backtest
python3 backtest.py
```

### Output Files

```
models/
├── xgboost_multi_asset.pkl
├── lightgbm_multi_asset.pkl
├── catboost_multi_asset.pkl
└── multi_asset_features.txt

logs/
├── daily_predictions.csv      # Used by backtest
└── labeled_predictions.csv    # Used by evaluate
```

### Expected Performance

- **Accuracy:** 60-65%
- **Precision:** 70-80%
- **Recall:** 40-55% (depends on threshold)
- **Sharpe Ratio:** 0.5-0.8

---

## System 2: Per-Asset Framework (Recommended for Other Assets)

**Purpose:** Train individual models for any configured asset (GLD, SLV, BTC, etc.)

**How it works:**
1. Downloads asset data (if not cached)
2. Adds technical features (50+ indicators)
3. Creates binary labels (profitable/unprofitable)
4. Trains 3 models per asset
5. Saves to `models/{asset}_*.pkl`

### Usage

```bash
# Download assets first
python3 framework/download_all_assets.py --asset GLD

# Train single asset
python3 framework/train_unified.py --asset GLD

# Train all commodities
python3 framework/train_unified.py --type commodity

# Train everything
python3 framework/train_unified.py --all

# Generate predictions
python3 predict_per_asset.py --asset GLD

# Backtest
python3 backtest.py --asset GLD
```

### Output Files

```
models/
├── gld_xgboost.pkl
├── gld_lightgbm.pkl
├── gld_catboost.pkl
├── slv_xgboost.pkl
└── ...

logs/predictions/
└── GLD_predictions.csv
```

### Expected Performance

- **Accuracy:** 55-65% (varies by asset)
- **Best for:** GLD, SLV, individual crypto
- **Note:** Some assets harder to predict than others

---

## System 3: Original SPY Training (Advanced)

**Purpose:** Advanced SPY training with triple-barrier labeling and regime detection.

**How it works:**
1. Loads SPY data only
2. Advanced labeling (triple barrier or forward returns)
3. Regime-aware feature engineering
4. Sample weighting by profit potential
5. Extensive hyperparameter tuning

### Usage

```bash
# Train SPY model
python3 train.py

# Generate predictions
python3 predict.py

# Backtest
python3 backtest.py
```

### When to Use

- When you want the most sophisticated SPY model
- When you need triple-barrier labeling
- When you want regime-aware training

---

## Which System Should I Use?

### For SPY Trading

**Use System 1 (Multi-Asset Ensemble):**
```bash
python3 train_multi_asset.py
python3 predict_multi_asset_ensemble.py
python3 backtest.py
```

**Why:** More training data (10,500 vs 6,500), learns cross-asset patterns.

### For Other Assets (GLD, SLV, BTC, etc.)

**Use System 2 (Per-Asset Framework):**
```bash
python3 framework/train_unified.py --asset GLD
python3 predict_per_asset.py --asset GLD
python3 backtest.py --asset GLD
```

**Why:** Asset-specific optimization, proper thresholds per asset.

### For Advanced SPY Research

**Use System 3 (Original):**
```bash
python3 train.py
python3 predict.py
python3 backtest.py
```

**Why:** Most sophisticated labeling and feature engineering.

---

## Quick Start: Complete Workflow

### A. SPY Trading (Most Common)

```bash
# 1. Update SPY data
python3 update_spy_data.py

# 2. Train multi-asset ensemble
python3 train_multi_asset.py

# 3. Generate predictions
python3 predict_multi_asset_ensemble.py

# 4. Evaluate model
python3 evaluate.py

# 5. Backtest strategy
python3 backtest.py

# 6. (Optional) Optimize threshold
python3 optimize_threshold.py
```

### B. Precious Metals Trading

```bash
# 1. Download precious metals data
python3 framework/download_all_assets.py --asset GLD
python3 framework/download_all_assets.py --asset SLV
python3 framework/download_all_assets.py --asset GDX

# 2. Train models
python3 framework/train_unified.py --asset GLD
python3 framework/train_unified.py --asset SLV
python3 framework/train_unified.py --asset GDX

# 3. Generate predictions
python3 predict_per_asset.py --asset GLD
python3 predict_per_asset.py --asset SLV
python3 predict_per_asset.py --asset GDX

# 4. Backtest individual assets
python3 backtest.py --asset GLD
python3 backtest.py --asset SLV
python3 backtest.py --asset GDX

# 5. Portfolio backtest
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3
```

### C. Crypto Trading

```bash
# 1. Download crypto data
python3 framework/download_all_assets.py --type crypto

# 2. Train all crypto
python3 framework/train_unified.py --type crypto

# 3. Generate predictions
python3 predict_per_asset.py --asset BTC/USDT
python3 predict_per_asset.py --asset ETH/USDT

# 4. Backtest
python3 backtest.py --asset BTC/USDT
```

---

## Threshold Configuration

All systems use thresholds from `configs/best_thresholds.json`:

```json
{
  "threshold": 0.30,
  "spike_thresh": 0.30,
  "crash_thresh": 0.30,
  "confidence_thresh": 0.25
}
```

### Threshold Trade-offs

| Threshold | Recall | Precision | Use Case |
|-----------|--------|-----------|----------|
| **0.55** | 15% | 98% | Very conservative, few trades |
| **0.45** | 25% | 92% | Conservative |
| **0.35** | 40% | 84% | Balanced |
| **0.30** | 55% | 75% | Aggressive, more trades |
| **0.25** | 65% | 68% | Very aggressive |

**Current setting:** 0.30 (catch 50%+ of market moves)

---

## Common Issues

### Issue: "98-99% accuracy"

**Problem:** Data leakage - model sees future prices.
**Solution:** Use latest version of `framework/train_unified.py` (fixed in commit 738cd395).

### Issue: "0 models trained" or "'Label' error"

**Problem:** Column naming mismatch.
**Solution:** Use latest version of `framework/train_unified.py` (fixed in commit 762bd730).

### Issue: "No predictions" in backtest

**Problem:** Need to generate predictions first.
**Solution:**
```bash
# For SPY
python3 predict_multi_asset_ensemble.py

# For other assets
python3 predict_per_asset.py --asset GLD
```

### Issue: KeyError when downloading

**Problem:** yfinance version compatibility.
**Solution:** Use latest version of `update_spy_data.py` and `framework/download_all_assets.py`.

---

## Performance Expectations

### Realistic Accuracy Ranges

| Asset | Expected Accuracy | Notes |
|-------|-------------------|-------|
| **SPY** | 60-65% | Most data, most stable |
| **QQQ** | 58-63% | Similar to SPY |
| **GLD** | 58-62% | Mean-reverting |
| **SLV** | 55-60% | More volatile |
| **BTC** | 55-65% | High volatility |
| **ETH** | 54-62% | Correlated with BTC |

**Warning signs:**
- Accuracy > 70% = likely overfitting
- Accuracy > 90% = definitely data leakage
- Accuracy = 50% = no predictive value

### Backtest Metrics to Watch

| Metric | Good | Excellent | Red Flag |
|--------|------|-----------|----------|
| **Sharpe** | > 0.5 | > 1.0 | > 3.0 (overfit) |
| **Max DD** | < 30% | < 20% | > 50% |
| **Win Rate** | > 52% | > 58% | > 70% (overfit) |

---

## Next Steps After Training

1. **Evaluate:** `python3 evaluate.py` - Check accuracy metrics
2. **Backtest:** `python3 backtest.py` - Check profit/loss
3. **Analyze:** `python3 analyze_feature_importance.py` - Understand what features matter
4. **Optimize:** `python3 optimize_threshold.py` - Fine-tune threshold
5. **Compare:** `python3 compare_strategies.py` - Compare different approaches

---

## File Structure Reference

```
NeuroVest/
├── train_multi_asset.py          # System 1: Multi-asset ensemble
├── predict_multi_asset_ensemble.py
├── train.py                      # System 3: Original SPY
├── predict.py
├── framework/
│   ├── train_unified.py          # System 2: Per-asset
│   └── download_all_assets.py
├── predict_per_asset.py
├── backtest.py                   # Works with all systems
├── backtest_portfolio.py         # Multi-asset portfolios
├── evaluate.py
├── config.py                     # Central configuration
└── configs/
    └── best_thresholds.json      # Threshold settings
```

---

## Summary

1. **For SPY:** Use `train_multi_asset.py` → `predict_multi_asset_ensemble.py`
2. **For other assets:** Use `framework/train_unified.py --asset X` → `predict_per_asset.py --asset X`
3. **Expect 55-65% accuracy** (anything higher is suspicious)
4. **Generate predictions before backtesting**
5. **Check threshold settings** in `configs/best_thresholds.json`

---

*For more details, see:*
- **ARCHITECTURE_GUIDE.md** - System architecture
- **ACCURACY_OPTIMIZATION_GUIDE.md** - Threshold tuning
- **PRECIOUS_METALS_GUIDE.md** - Precious metals workflow
- **CRASH_PREDICTION_ANALYSIS.md** - Crash detection
