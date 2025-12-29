# NeuroVest Architecture Guide

## Two Parallel Approaches

### 1. Multi-Asset Ensemble (Current Default)
**What:** One unified model trained on multiple assets simultaneously
**Assets:** SPY + BTC/USDT + ETH/USDT + SOL/USDT (9,786 combined samples)
**Models:** 3 ensemble models in `models/`:
- xgboost_multi_asset.pkl
- lightgbm_multi_asset.pkl
- catboost_multi_asset.pkl

**Used by:**
- `predict_multi_asset_ensemble.py` (generates predictions for SPY)
- `backtest.py` (backtests SPY using multi-asset predictions)

**Advantage:** Learns cross-asset patterns, regime changes across markets
**Limitation:** Currently only generates predictions for SPY

---

### 2. Per-Asset Models (Framework Approach)
**What:** Separate models trained for each individual asset
**Assets:** Any of the 59 configured assets
**Models:** Per-asset model files in `models/`:
- btc_usdt_xgboost.pkl, btc_usdt_lightgbm.pkl, btc_usdt_catboost.pkl
- eth_usdt_xgboost.pkl, eth_usdt_lightgbm.pkl, eth_usdt_catboost.pkl
- sol_usdt_xgboost.pkl, sol_usdt_lightgbm.pkl, sol_usdt_catboost.pkl
- spy_xgboost.pkl (per-asset SPY model, different from ensemble)

**Used by:**
- `framework/train_unified.py --asset <TICKER>` (trains per-asset models)
- `predict_per_asset.py --asset <TICKER>` (generates predictions)
- `backtest.py --asset <TICKER>` (backtests using per-asset predictions)

**Advantage:** Asset-specific optimization, can predict for any asset
**Current Status:** Only 3 crypto + SPY trained so far

---

## Current Status by Asset Type

### Equity (33 configured):
- **SPY:** ✅ Downloaded, ✅ Per-asset model, ✅ Part of multi-asset ensemble
- **QQQ:** ⚠️ Configured but NOT downloaded or trained
- **Other 31:** ⚠️ Configured but NOT downloaded or trained

### Crypto (10 configured):
- **BTC/USDT:** ✅ Downloaded, ✅ Per-asset model, ✅ Part of multi-asset ensemble
- **ETH/USDT:** ✅ Downloaded, ✅ Per-asset model, ✅ Part of multi-asset ensemble
- **SOL/USDT:** ✅ Downloaded, ✅ Per-asset model, ✅ Part of multi-asset ensemble
- **AVAX/USDT:** ⚠️ Downloaded but NOT trained
- **MATIC/USDT:** ⚠️ Downloaded but NOT trained
- **Other 5:** ⚠️ Configured but NOT downloaded or trained

### Precious Metals/Commodities (6 configured):
- **GLD (Gold):** ⚠️ Configured but NOT downloaded or trained
- **SLV (Silver):** ⚠️ Configured but NOT downloaded or trained
- **GDX (Gold Miners):** ⚠️ Configured but NOT downloaded or trained
- **USO (Oil):** ⚠️ Configured but NOT downloaded or trained
- **UNG (Natural Gas):** ⚠️ Configured but NOT downloaded or trained
- **DBC (Commodities):** ⚠️ Configured but NOT downloaded or trained

### Bonds (10 configured):
- **All 10:** ⚠️ Configured but NOT downloaded or trained

**Total:** 59 assets configured, only 6 downloaded (SPY + 5 crypto)

---

## How the System Works Today

### Default Workflow (Multi-Asset):
```bash
# 1. Generate predictions for SPY using multi-asset ensemble
python3 predict_multi_asset_ensemble.py
# → Uses multi-asset models trained on SPY+BTC+ETH+SOL
# → Outputs: logs/daily_predictions.csv (SPY predictions)

# 2. Backtest SPY using those predictions
python3 backtest.py
# → Tests trading SPY with multi-asset ensemble predictions
```

### Per-Asset Workflow (Framework):
```bash
# 1. Download asset data
python3 framework/download_all_assets.py --asset GLD

# 2. Train per-asset model
python3 framework/train_unified.py --asset GLD

# 3. Generate predictions
python3 predict_per_asset.py --asset GLD

# 4. Backtest the asset
python3 backtest.py --asset GLD
```

---

## Adding New Assets (e.g., Precious Metals)

### Option A: Quick Start (GLD only)
```bash
# 1. Download GLD data
python3 framework/download_all_assets.py --asset GLD

# 2. Train GLD-specific models (3 models: XGBoost, LightGBM, CatBoost)
python3 framework/train_unified.py --asset GLD

# 3. Generate GLD predictions
python3 predict_per_asset.py --asset GLD

# 4. Backtest GLD
python3 backtest.py --asset GLD

# 5. Compare with SPY
python3 backtest.py --assets SPY,GLD --compare
```

### Option B: Download All Precious Metals
```bash
# Download all commodity data (GLD, SLV, GDX, USO, UNG, DBC)
python3 framework/download_all_assets.py --asset-group commodity

# Train metals only (skip oil/gas for now)
python3 framework/train_unified.py --asset GLD
python3 framework/train_unified.py --asset SLV
python3 framework/train_unified.py --asset GDX

# Generate predictions for all
python3 predict_per_asset.py --asset GLD
python3 predict_per_asset.py --asset SLV
python3 predict_per_asset.py --asset GDX

# Compare precious metals
python3 backtest.py --assets GLD,SLV,GDX --compare
```

### Option C: Add QQQ (Nasdaq)
```bash
# Download QQQ
python3 framework/download_all_assets.py --asset QQQ

# Train QQQ models
python3 framework/train_unified.py --asset QQQ

# Generate predictions
python3 predict_per_asset.py --asset QQQ

# Compare SPY vs QQQ
python3 backtest.py --assets SPY,QQQ --compare
```

### Option D: Add to Multi-Asset Ensemble (Advanced)
```bash
# Retrain multi-asset ensemble to include precious metals
# Edit train_multi_asset.py to add GLD, SLV to asset list

python3 train_multi_asset.py
# This creates new multi-asset models that include precious metals
# Learns cross-asset patterns between equities, crypto, and metals
```

---

## Correlation Analysis with New Assets

Once you've downloaded data for new assets:

```bash
# Analyze SPY + Crypto + Gold correlation
python3 analyze_correlations.py --assets SPY,BTC/USDT,ETH/USDT,GLD,SLV

# Expected insights:
# - Gold often negatively correlated with stocks (diversifier)
# - Silver tracks gold but more volatile
# - Crypto may be uncorrelated with traditional assets
```

---

## Portfolio Backtesting with Mixed Assets

```bash
# Traditional 60/40 portfolio + gold hedge
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.6,0.4

# Multi-asset diversified portfolio
python3 backtest_portfolio.py --assets SPY,QQQ,GLD,BTC/USDT --weights 0.4,0.2,0.2,0.2

# Risk parity across asset classes
python3 backtest_portfolio.py --assets SPY,TLT,GLD,BTC/USDT --weights 0.25,0.25,0.25,0.25
```

---

## Key Differences

| Feature | Multi-Asset Ensemble | Per-Asset Models |
|---------|---------------------|------------------|
| **Training** | One model, many assets | Separate model per asset |
| **Prediction** | Currently SPY only | Any asset |
| **Pattern Learning** | Cross-asset patterns | Asset-specific patterns |
| **Current Usage** | Default backtest | Framework tools |
| **Data Efficiency** | More samples (combined) | Fewer samples (per asset) |
| **Flexibility** | Limited to trained combo | Any configured asset |
| **Cross-Asset Signals** | Yes (regime detection) | No (asset-specific only) |
| **Customization** | One size fits all | Optimized per asset |

---

## Which Approach for What?

### Use Multi-Asset Ensemble When:
- Want to capture cross-market dynamics
- Trading SPY primarily
- Need regime change detection across markets
- Have limited data per asset
- Want to learn how crypto crash affects equities

### Use Per-Asset Models When:
- Trading specific assets (QQQ, GLD, BTC, etc.)
- Want asset-specific optimization
- Need predictions for many different assets
- Want portfolio backtesting across multiple assets
- Each asset has unique patterns

### Can Use Both Simultaneously!
- Multi-asset ensemble for SPY trading (regime detection)
- Per-asset models for diversified portfolio (asset-specific signals)
- Compare strategies with `compare_strategies.py`
- Build portfolios with mix of approaches

---

## Complete Example: Adding Precious Metals

### Step 1: Download Gold Data
```bash
python3 framework/download_all_assets.py --asset GLD
# Downloads GLD historical data to data_cache/GLD_1d.csv
```

### Step 2: Train Gold Models
```bash
python3 framework/train_unified.py --asset GLD
# Creates:
# - models/gld_xgboost.pkl
# - models/gld_lightgbm.pkl
# - models/gld_catboost.pkl
# - models/gld_features.txt (feature list)
```

### Step 3: Generate Predictions
```bash
python3 predict_per_asset.py --asset GLD
# Creates: logs/predictions/GLD_predictions.csv
```

### Step 4: Backtest Gold
```bash
python3 backtest.py --asset GLD
# Backtests GLD using its per-asset predictions
```

### Step 5: Analyze Correlations
```bash
# See how gold correlates with stocks and crypto
python3 analyze_correlations.py --assets SPY,GLD,BTC/USDT
# Creates correlation matrix and heatmap
```

### Step 6: Portfolio Backtest
```bash
# Test SPY + Gold diversification
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance monthly
# Backtests 70/30 SPY/GLD portfolio with monthly rebalancing
```

---

## Data Requirements by Asset Type

### Equities (SPY, QQQ, etc.):
- **Source:** Yahoo Finance (yfinance)
- **History:** Typically 20+ years available
- **Update Frequency:** Daily
- **Cost:** Free

### Crypto (BTC/USDT, ETH/USDT, etc.):
- **Source:** CCXT (Binance)
- **History:** Varies (BTC: ~2017, SOL: ~2020)
- **Update Frequency:** Daily
- **Cost:** Free

### Precious Metals (GLD, SLV, etc.):
- **Source:** Yahoo Finance (yfinance)
- **History:** 15-20 years (GLD since 2004)
- **Update Frequency:** Daily
- **Cost:** Free

### Bonds (TLT, IEF, etc.):
- **Source:** Yahoo Finance (yfinance)
- **History:** 15-20 years
- **Update Frequency:** Daily
- **Cost:** Free

**All configured assets use free data sources!**

---

## Current System Limitations

1. **Multi-asset ensemble only predicts SPY**
   - Solution: Use per-asset models for other assets

2. **Only 6 assets downloaded so far**
   - Solution: Download more with `framework/download_all_assets.py`

3. **Crypto has limited history (2-3 years)**
   - Impact: Smaller training datasets, less diverse market conditions
   - Mitigation: Use multi-asset ensemble to share knowledge

4. **Per-asset vs ensemble predictions use different formats**
   - Per-asset: Binary (0/1)
   - Ensemble: 3-class (0/1/2)
   - Impact: Hard to directly compare
   - Solution: Documented in ACCURACY_OPTIMIZATION_GUIDE.md

---

## Recommended Next Steps

1. **Download more equities:**
   ```bash
   python3 framework/download_all_assets.py --asset QQQ
   python3 framework/download_all_assets.py --asset DIA  # Dow Jones
   ```

2. **Download precious metals:**
   ```bash
   python3 framework/download_all_assets.py --asset-group commodity
   ```

3. **Download bonds for diversification:**
   ```bash
   python3 framework/download_all_assets.py --asset TLT  # 20+ year bonds
   python3 framework/download_all_assets.py --asset SHY  # 1-3 year bonds
   ```

4. **Train per-asset models:**
   ```bash
   python3 framework/train_unified.py --asset QQQ
   python3 framework/train_unified.py --asset GLD
   python3 framework/train_unified.py --asset TLT
   ```

5. **Analyze cross-asset correlations:**
   ```bash
   python3 analyze_correlations.py --assets SPY,QQQ,GLD,TLT,BTC/USDT
   ```

6. **Build diversified portfolios:**
   ```bash
   python3 backtest_portfolio.py --assets SPY,GLD,TLT,BTC/USDT --weights 0.4,0.2,0.2,0.2
   ```

---

## Technical Architecture

```
NeuroVest/
├── Multi-Asset Ensemble Pipeline:
│   ├── train_multi_asset.py        # Train ensemble on SPY+crypto
│   ├── predict_multi_asset_ensemble.py  # Generate SPY predictions
│   └── backtest.py (default)       # Backtest SPY
│
├── Per-Asset Framework Pipeline:
│   ├── framework/download_all_assets.py  # Download any asset
│   ├── framework/train_unified.py   # Train per-asset models
│   ├── predict_per_asset.py        # Generate asset predictions
│   └── backtest.py --asset TICKER  # Backtest specific asset
│
├── Multi-Asset Analysis Tools:
│   ├── analyze_correlations.py     # Correlation analysis
│   ├── backtest_portfolio.py       # Portfolio backtesting
│   ├── compare_strategies.py       # Strategy comparison
│   └── backtest.py --compare       # Asset comparison
│
└── Data & Models:
    ├── data_cache/*.csv            # Downloaded asset data
    ├── models/*_xgboost.pkl        # Per-asset models
    ├── models/*_multi_asset.pkl    # Multi-asset ensemble
    └── logs/predictions/           # Per-asset predictions
```

---

## Summary

**You have TWO systems working in parallel:**

1. **Multi-Asset Ensemble** - Current default for SPY (trained on SPY+crypto)
2. **Per-Asset Framework** - Can train/predict any of 59 configured assets

**Both systems work together:**
- Use ensemble for SPY regime detection
- Use per-asset for individual asset trading
- Use portfolio tools to combine them

**Adding assets is easy:**
- Download: `framework/download_all_assets.py --asset GLD`
- Train: `framework/train_unified.py --asset GLD`
- Predict: `predict_per_asset.py --asset GLD`
- Backtest: `backtest.py --asset GLD`

**Precious metals ARE configured and ready to use!**
- Just download and train them
- Same process as crypto
- All free data sources

---

*For more details, see:*
- **MULTI_ASSET_ANALYSIS_SUMMARY.md** - Complete multi-asset tools guide
- **ACCURACY_OPTIMIZATION_GUIDE.md** - Threshold tuning
- **FRAMEWORK_GUIDE.md** - Framework documentation
- **README.md** - Main documentation
