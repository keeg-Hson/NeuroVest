# Multi-Asset Training: Which Assets to Add?

## Current State
- **Asset**: SPY only
- **Training samples**: 6,501
- **Test accuracy**: 71.0%
- **Backtest Sharpe**: 1.63

---

## Options Analysis

### Option 1: Add Equity ETFs (RECOMMENDED)

**What**: QQQ, IWM, DIA, VTI, EEM, XLF, XLK, XLE

**Pros**:
- ✅ Same market structure (trading hours, fundamentals)
- ✅ Similar volatility (15-25% annualized)
- ✅ Proven approach in quantitative finance
- ✅ 7-8x more training data (~45,000 samples)
- ✅ Low risk of distribution shift
- ✅ IWM/EEM have different enough dynamics to add value

**Cons**:
- ⚠️ High correlation (SPY/QQQ ~0.95) may not add much diversity
- ⚠️ Model might learn sector rotations that don't apply universally

**Expected impact**:
- Accuracy: 71% → 73-75% (modest improvement)
- Robustness: Significant improvement (less overfitting)
- Generalization: Better performance in unseen market regimes

**Execution**:
```bash
python download_equity_etfs.py
# Update train_multi_asset.py to load from data_cache/*.csv
python train_multi_asset.py
```

---

### Option 2: Add Crypto (NOT RECOMMENDED for SPY-only)

**What**: BTC, ETH, SOL

**Pros**:
- ✅ Very different market dynamics (diversity)
- ✅ ~4,000 more samples
- ✅ If you want to trade crypto too (multi-strategy)

**Cons**:
- ❌ Different market structure (24/7 vs market hours)
- ❌ 5-10x higher volatility (80-200% annualized)
- ❌ Different drivers (sentiment vs fundamentals)
- ❌ Only ~8 years of data (2017-2025) vs SPY's 25 years
- ❌ High risk of negative transfer learning
- ❌ Model learns crypto patterns that don't apply to SPY

**Expected impact**:
- Accuracy: 71% → 68-70% (likely degradation for SPY)
- Robustness: Questionable (may learn wrong patterns)
- Generalization: Risk of learning crypto-specific noise

**When to use**: Only if you plan to trade crypto separately with different thresholds

**Execution**:
```bash
python download_crypto_data.py
python train_multi_asset.py  # Already configured for crypto
```

---

### Option 3: Add More SPY History (CONSERVATIVE)

**What**: Extend SPY from 2000-2025 to 1993-2025

**Pros**:
- ✅ Zero distribution shift risk
- ✅ +30% more training data (~8,000 samples)
- ✅ Captures different market regimes (1990s bull market, 2000 crash)
- ✅ Simplest to implement

**Cons**:
- ⚠️ Still only one asset (same correlations)
- ⚠️ Pre-2000 market structure was different (less algorithmic)

**Expected impact**:
- Accuracy: 71% → 71-73% (small improvement)
- Robustness: Moderate improvement
- Safest option

**Execution**:
```bash
# Update download_spy_data.py
START_DATE = "1993-01-29"  # SPY inception
python download_spy_data.py
python train_multi_asset.py
```

---

## Recommendation Matrix

| Your Goal | Best Option | Why |
|-----------|-------------|-----|
| **Maximize SPY accuracy** | Option 1 (Equity ETFs) | Same asset class, 7x data |
| **Build general forecaster** | Option 1 (Equity ETFs) | Proven for cross-asset learning |
| **Trade crypto too** | Option 2 (Crypto) | But train separate models |
| **Minimize risk** | Option 3 (More SPY) | Zero distribution shift |
| **Research/learning** | Try all 3, compare | A/B test on held-out data |

---

## My Recommendation

**Start with Option 1 (Equity ETFs)**:

1. Run baseline first:
   ```bash
   # Save current model performance
   cp logs/model_performance.csv logs/model_performance_spy_only.csv
   ```

2. Add equity ETFs:
   ```bash
   python download_equity_etfs.py
   # Update train_multi_asset.py (see below)
   python train_multi_asset.py
   ```

3. Compare results:
   ```bash
   python backtest.py
   # Compare Sharpe ratio, accuracy, drawdown to baseline
   ```

4. If worse, revert to SPY-only (you have baseline saved)

---

## Code Changes Needed for Equity ETFs

Update `train_multi_asset.py` lines 77-112:

```python
# OLD: Crypto assets
crypto_assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

# NEW: Equity ETFs
equity_assets = ['QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLK', 'XLE']
equity_dfs = []

for ticker in equity_assets:
    filepath = CACHE_DIR / f"{ticker}_1d.csv"
    if not filepath.exists():
        print(f"   ⚠️  {ticker} not found, skipping")
        continue

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Add features
    df, _ = add_features(df)

    # Add asset type (all equity, but different sectors)
    df['asset_type_stock'] = 1
    df['asset_type_crypto'] = 0

    # Use same threshold as SPY (0.5%)
    df = add_forward_returns_and_labels(
        df,
        price_col="Close",
        horizon=TRAIN_CFG["horizon"],
        pos_threshold=TRAIN_CFG["pos_threshold"],  # Same 0.5% as SPY
        fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
        slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
    )

    equity_dfs.append(df)
    print(f"   {ticker:6s}: {len(df)} rows")

# Combine with SPY
all_dfs = [spy_df] + equity_dfs
```

---

## Success Criteria

After adding new assets, you should see:

**Good signs**:
- Test accuracy: ≥71% (maintain or improve)
- Sharpe ratio: ≥1.60 (maintain or improve)
- Max drawdown: ≤-22% (maintain or improve)
- Model agreement: ≥85% (ensemble consensus)

**Warning signs**:
- Test accuracy drops >2% (distribution shift)
- Training accuracy >> test accuracy (overfitting)
- Backtest Sharpe drops (learning irrelevant patterns)

**If warning signs appear**: Revert to SPY-only baseline

---

## Bottom Line

**For SPY-only predictions**: Add equity ETFs (Option 1)

**For multi-strategy trading**: Train separate models:
- SPY model (current approach)
- Crypto model (separate with 2% threshold)
- Don't mix them

**Not sure?**: Run all three options and compare. The best model is the one that performs best on held-out data, not the one with the most training samples.
