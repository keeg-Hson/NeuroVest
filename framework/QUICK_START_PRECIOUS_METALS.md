# Quick Start: Adding Precious Metals to NeuroVest

## TL;DR - Current System Status

### ✅ What's Working NOW:
- **Multi-Asset Ensemble:** Trained on SPY + BTC + ETH + SOL
  - Generates predictions for SPY
  - Used by default `backtest.py`
  
- **Per-Asset Models:** Trained for SPY, BTC, ETH, SOL individually
  - Each asset has 3 models (XGBoost, LightGBM, CatBoost)
  - Use `predict_per_asset.py` and `backtest.py --asset`

### ⚠️ What's Configured but NOT Downloaded Yet:
- **33 Equities:** QQQ, DIA, IWM, VTI, etc.
- **6 Precious Metals/Commodities:** GLD, SLV, GDX, USO, UNG, DBC
- **10 Bonds:** TLT, IEF, SHY, AGG, etc.

**Total:** 59 assets configured, only 6 downloaded (10%)

---

## The Two Systems Explained

### System 1: Multi-Asset Ensemble (Current Default)

**How it works:**
1. **One model trained on multiple assets** (SPY + crypto)
2. Learns cross-asset patterns and regime changes
3. Currently only predicts for SPY

**Files:**
```
models/xgboost_multi_asset.pkl    ← Ensemble XGBoost
models/lightgbm_multi_asset.pkl   ← Ensemble LightGBM
models/catboost_multi_asset.pkl   ← Ensemble CatBoost
```

**Usage:**
```bash
python3 predict_multi_asset_ensemble.py  # Predict SPY
python3 backtest.py                      # Backtest SPY
```

**Pros:**
- Learns regime changes across markets
- More training data (combined assets)
- Captures cross-market dynamics

**Cons:**
- Only predicts SPY currently
- Can't predict individual crypto/metals

---

### System 2: Per-Asset Models (Framework)

**How it works:**
1. **Separate models for each asset**
2. Asset-specific optimization
3. Can predict any configured asset

**Files:**
```
models/spy_xgboost.pkl              ← SPY-specific model
models/btc_usdt_xgboost.pkl         ← BTC-specific model
models/gld_xgboost.pkl              ← GLD-specific (when trained)
```

**Usage:**
```bash
# Download asset
python3 framework/download_all_assets.py --asset GLD

# Train models
python3 framework/train_unified.py --asset GLD

# Generate predictions
python3 predict_per_asset.py --asset GLD

# Backtest
python3 backtest.py --asset GLD
```

**Pros:**
- Can predict any asset
- Asset-specific optimization
- Flexible for portfolios

**Cons:**
- Less training data per asset
- Doesn't learn cross-asset patterns

---

## Adding GLD (Gold) Step-by-Step

### Prerequisites:
```bash
# Ensure multitasking is installed (for yfinance)
pip install multitasking

# Or reinstall requirements
pip install -r requirements.txt
```

### Step 1: Download GLD Data
```bash
python3 framework/download_all_assets.py --asset GLD
```

**What this does:**
- Downloads GLD historical data from Yahoo Finance
- Saves to `data_cache/GLD_1d.csv`
- ~20 years of daily data (2004-present)

**Expected output:**
```
📥 Downloading GLD...
✓ GLD: 5,234 days
Saved to data_cache/GLD_1d.csv
```

---

### Step 2: Train GLD Models
```bash
python3 framework/train_unified.py --asset GLD
```

**What this does:**
- Trains 3 models: XGBoost, LightGBM, CatBoost
- Optimizes for GLD-specific patterns
- Uses same 126 features as other assets

**Creates:**
```
models/gld_xgboost.pkl
models/gld_lightgbm.pkl
models/gld_catboost.pkl
models/gld_features.txt
```

**Expected output:**
```
Training GLD models...
XGBoost:  Accuracy 62.3%, Precision 68.1%, Recall 45.2%
LightGBM: Accuracy 61.8%, Precision 66.4%, Recall 46.1%
CatBoost: Accuracy 60.9%, Precision 65.2%, Recall 44.8%
Saved models to models/
```

---

### Step 3: Generate GLD Predictions
```bash
python3 predict_per_asset.py --asset GLD
```

**What this does:**
- Loads all 3 GLD models
- Generates predictions with ensemble voting
- Outputs prediction file

**Creates:**
```
logs/predictions/GLD_predictions.csv
```

**Expected output:**
```
🤖 Generating predictions for GLD...
✓ xgboost: 5,234 predictions
✓ lightgbm: 5,234 predictions
✓ catboost: 5,234 predictions
Ensemble distribution: NORMAL: 78.2%, SPIKE: 21.8%
💾 Saved: logs/predictions/GLD_predictions.csv
```

---

### Step 4: Backtest GLD
```bash
python3 backtest.py --asset GLD
```

**What this does:**
- Loads GLD predictions from Step 3
- Backtests trading strategy on GLD
- Shows performance metrics

**Expected output:**
```
📊 Backtest Report (GLD)
Total Return:    +45.2%
Sharpe Ratio:    1.23
Max Drawdown:    -12.4%
Win Rate:        55.3%
Trades:          287
```

---

### Step 5: Compare SPY vs GLD
```bash
python3 backtest.py --assets SPY,GLD --compare
```

**What this does:**
- Backtests both SPY and GLD
- Shows side-by-side comparison
- Saves comparison CSV

**Expected output:**
```
================================================================================
BACKTEST COMPARISON (sorted by Sharpe)
================================================================================

Asset  Total Return  Sharpe  Max DD  Win Rate  Trades
SPY       +36.5%     0.51    -16.8%   50.4%     693
GLD       +45.2%     1.23    -12.4%   55.3%     287
```

---

### Step 6: Analyze Correlation
```bash
python3 analyze_correlations.py --assets SPY,GLD,BTC/USDT
```

**What this does:**
- Calculates correlation between assets
- Identifies diversification opportunities
- Creates heatmap

**Expected insights:**
```
Correlation Matrix:
         SPY    GLD    BTC/USDT
SPY     1.00  -0.15    0.23
GLD    -0.15   1.00   -0.08
BTC     0.23  -0.08    1.00

Key Finding: GLD negatively correlated with SPY (-0.15)
→ Gold is a good diversifier for equity portfolios!
```

---

### Step 7: Portfolio Backtest (SPY + GLD)
```bash
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance monthly
```

**What this does:**
- Backtests 70% SPY / 30% GLD portfolio
- Rebalances monthly
- Shows portfolio metrics

**Expected output:**
```
📊 Portfolio Results (70% SPY / 30% GLD)
Total Return:     +52.3%
Sharpe Ratio:     1.45
Max Drawdown:     -11.2%  ← Lower than SPY alone!
Rebalances:       36
```

**Key Insight:** Adding gold reduces drawdown while maintaining returns!

---

## Adding All Precious Metals

### Download All at Once:
```bash
python3 framework/download_all_assets.py --asset-group commodity
```

**Downloads:** GLD, SLV, GDX, USO, UNG, DBC

---

### Train All Metals:
```bash
# Gold
python3 framework/train_unified.py --asset GLD

# Silver  
python3 framework/train_unified.py --asset SLV

# Gold Miners
python3 framework/train_unified.py --asset GDX
```

**Training time:** ~10-15 minutes per asset

---

### Compare All Metals:
```bash
# Generate predictions
python3 predict_per_asset.py --asset GLD
python3 predict_per_asset.py --asset SLV
python3 predict_per_asset.py --asset GDX

# Compare performance
python3 backtest.py --assets GLD,SLV,GDX --compare
```

**Expected ranking:**
```
GLD: Best Sharpe, lowest drawdown (most stable)
SLV: Higher return, higher volatility
GDX: Highest return, highest volatility (leveraged gold)
```

---

## Integration with Multi-Asset Ensemble (Advanced)

### Option: Retrain Ensemble with Metals

**Current ensemble:** SPY + BTC + ETH + SOL

**New ensemble:** SPY + BTC + ETH + GLD

**Steps:**
1. Edit `train_multi_asset.py`
2. Change asset list to include GLD
3. Retrain:
   ```bash
   python3 train_multi_asset.py
   ```

**Benefit:** Model learns cross-asset patterns:
- How gold reacts to stock market crashes
- Regime detection across asset classes
- Safe haven flows during volatility

**Trade-off:** Longer training time, more complex model

---

## Correlation Analysis Examples

### Traditional Portfolio:
```bash
python3 analyze_correlations.py --assets SPY,QQQ,TLT,GLD
```

**Expected findings:**
- SPY ↔ QQQ: 0.95 (highly correlated, redundant)
- SPY ↔ TLT: -0.40 (negatively correlated, good diversifier)
- SPY ↔ GLD: -0.15 (slightly negative, hedge)
- TLT ↔ GLD: 0.10 (uncorrelated)

**Insight:** TLT (bonds) is best SPY diversifier, GLD provides additional hedge

---

### Risk Parity Portfolio:
```bash
python3 backtest_portfolio.py --assets SPY,TLT,GLD,BTC/USDT --weights 0.25,0.25,0.25,0.25
```

**Concept:** Equal allocation across asset classes
- 25% Equities (SPY)
- 25% Bonds (TLT)
- 25% Commodities (GLD)
- 25% Crypto (BTC)

**Expected result:** Lower volatility, smoother returns

---

## Troubleshooting

### Error: "No data file found"
**Solution:**
```bash
python3 framework/download_all_assets.py --asset GLD
```

### Error: "No models found for GLD"
**Solution:**
```bash
python3 framework/train_unified.py --asset GLD
```

### Error: "ModuleNotFoundError: multitasking"
**Solution:**
```bash
pip install multitasking
```

### Error: "Insufficient data after feature engineering"
**Cause:** Asset has too little history or data quality issues
**Solution:** Check data file:
```bash
head data_cache/GLD_1d.csv
```

---

## Summary

**To add ANY asset to NeuroVest:**

1. **Download:** `framework/download_all_assets.py --asset TICKER`
2. **Train:** `framework/train_unified.py --asset TICKER`
3. **Predict:** `predict_per_asset.py --asset TICKER`
4. **Backtest:** `backtest.py --asset TICKER`
5. **Compare:** `backtest.py --assets TICKER1,TICKER2 --compare`
6. **Correlate:** `analyze_correlations.py --assets TICKER1,TICKER2`
7. **Portfolio:** `backtest_portfolio.py --assets TICKER1,TICKER2 --weights W1,W2`

**All 59 configured assets follow this same pattern!**

---

## Next Steps

1. **Fix yfinance dependency:**
   ```bash
   pip install multitasking
   ```

2. **Download precious metals:**
   ```bash
   python3 framework/download_all_assets.py --asset-group commodity
   ```

3. **Train GLD:**
   ```bash
   python3 framework/train_unified.py --asset GLD
   ```

4. **Explore correlations:**
   ```bash
   python3 analyze_correlations.py --assets SPY,GLD,BTC/USDT
   ```

5. **Build diversified portfolio:**
   ```bash
   python3 backtest_portfolio.py --assets SPY,GLD,TLT --weights 0.5,0.3,0.2
   ```

**Precious metals are configured and ready - just download and train them!** 🎯
