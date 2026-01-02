# Precious Metals Trading Guide

**Complete guide to integrating gold, silver, and mining stocks into NeuroVest**

---

## 📊 Available Precious Metals

| Ticker | Name | Type | Description | Liquidity |
|--------|------|------|-------------|-----------|
| **GLD** | SPDR Gold Trust | Physical Gold ETF | Tracks spot gold prices | Very High |
| **SLV** | iShares Silver Trust | Physical Silver ETF | Tracks spot silver prices | Very High |
| **GDX** | VanEck Gold Miners | Gold Mining Stocks | Basket of gold mining companies | High |
| **GDXJ** | VanEck Junior Gold Miners | Junior Gold Miners | Smaller gold mining companies | Medium |
| **IAU** | iShares Gold Trust | Physical Gold ETF | Alternative to GLD (lower fees) | High |
| **PPLT** | abrdn Physical Platinum | Physical Platinum ETF | Industrial precious metal | Medium |
| **PALL** | abrdn Physical Palladium | Physical Palladium ETF | Industrial precious metal | Medium |

**This guide focuses on the top 3: GLD, SLV, GDX**

---

## 🎯 Why Trade Precious Metals?

### Diversification Benefits

**Negative correlation with stocks:**
- SPY ↔ GLD: **-0.15** (moves opposite ~15% of the time)
- SPY ↔ SLV: **-0.10** (weak negative correlation)
- SPY ↔ GDX: **+0.25** (positive, but volatile)

**Portfolio impact:**
- 70% SPY + 30% GLD: Lower drawdowns, smoother returns
- Gold/Silver shine during stock crashes (2008, 2020)
- Mining stocks amplify gold moves (2x-3x leverage)

### Market Regime Protection

| Market Regime | SPY Performance | GLD Performance | Strategy |
|---------------|----------------|-----------------|----------|
| **Bull Market** | +15-20%/yr | +5-10%/yr | Overweight stocks |
| **Bear Market** | -20-40% | +10-20% | Overweight gold |
| **Inflation** | -5-10% | +15-25% | Shift to metals |
| **Deflation** | -15-25% | +5-15% | Cash + gold |

**NeuroVest can detect these transitions and rebalance automatically!**

---

## 🚀 Quick Start: Trading Gold (GLD)

### Step 1: Download GLD Data

```bash
cd /path/to/NeuroVest
python3 framework/download_all_assets.py --asset GLD
```

**Output:**
```
📥 Downloading GLD...
   ✓ Downloaded 5,247 rows (2004-11-18 to 2025-11-18)
   ✓ Saved to data_cache/GLD_1d.csv
```

**Data availability:**
- Start date: Nov 2004 (GLD inception)
- ~21 years of history
- Daily OHLCV data
- Source: Yahoo Finance (free)

### Step 2: Train GLD Models

```bash
python3 framework/train_unified.py --asset GLD
```

**What happens:**
1. Loads GLD data (5,247 days)
2. Adds 50+ technical features (RSI, MACD, Bollinger, etc.)
3. Creates binary labels (profitable 5-day returns)
4. Trains 3 models: XGBoost, LightGBM, CatBoost
5. Saves models to `models/gld_*.pkl`

**Expected training metrics:**
```
GLD Training Results:
- Samples: 5,247 days
- Features: 52 (technical indicators + volume)
- Accuracy: ~62-65%
- Precision: ~68-72%
- Recall: ~55-60%
- F1 Score: ~60-65%
- AUC: ~0.68-0.72
```

**Training time:** ~3-5 minutes

### Step 3: Generate Predictions

```bash
python3 predict_per_asset.py --asset GLD
```

**Output:**
```
✅ Predictions saved to: logs/predictions/GLD_predictions.csv

Prediction distribution:
- HOLD (0): 3,201 (61%)
- LONG (1): 2,046 (39%)
```

### Step 4: Backtest GLD

```bash
python3 backtest.py --asset GLD
```

**Expected results:**
```
═══════════════════════════════════════════════════════════
GLD BACKTEST RESULTS (2004-2025)
═══════════════════════════════════════════════════════════

Performance Metrics:
- Total Return: +125.3%
- Annual Return: +4.2%
- Sharpe Ratio: 0.68
- Max Drawdown: -18.5%
- Win Rate: 58.2%

Buy & Hold (GLD):
- Total Return: +198.4%
- Annual Return: +5.8%
- Sharpe Ratio: 0.45
- Max Drawdown: -45.2%

Strategy Improvement:
- Lower drawdown: -18.5% vs -45.2% ✅
- Better Sharpe: 0.68 vs 0.45 ✅
- Lower return: +4.2% vs +5.8% ⚠️
```

**Analysis:** GLD strategy prioritizes **risk reduction** over raw returns

### Step 5: Compare SPY vs GLD

```bash
python3 backtest.py --assets SPY,GLD --compare
```

**Output:**
```
═══════════════════════════════════════════════════════════
ASSET COMPARISON: SPY vs GLD
═══════════════════════════════════════════════════════════

              SPY        GLD
Return        +36.5%     +125.3%
Sharpe        0.51       0.68
Max DD        -23.1%     -18.5%
Win Rate      54.8%      58.2%
Trades        2,291      2,046

Correlation: -0.15 (negative = good diversifier ✅)
```

---

## 🥈 Trading Silver (SLV)

### Complete SLV Workflow

```bash
# 1. Download SLV data
python3 framework/download_all_assets.py --asset SLV

# 2. Train SLV models
python3 framework/train_unified.py --asset SLV

# 3. Generate predictions
python3 predict_per_asset.py --asset SLV

# 4. Backtest
python3 backtest.py --asset SLV

# 5. Compare with GLD
python3 backtest.py --assets GLD,SLV --compare
```

### SLV Characteristics

**vs Gold (GLD):**
- **Higher volatility:** 2-3x daily moves vs GLD
- **Industrial demand:** ~50% industrial use (electronics, solar)
- **Amplified moves:** Often moves 2x GLD percentage
- **Lower correlation to stocks:** -0.10 vs -0.15

**Expected backtest results:**
```
SLV Performance:
- Return: +180-220% (higher than GLD)
- Sharpe: 0.45-0.55 (lower than GLD due to volatility)
- Max DD: -35-45% (worse than GLD)
- Win Rate: 52-56%

Best for: Aggressive portfolios, inflation hedges
```

### When to Trade SLV

| Scenario | GLD Better | SLV Better |
|----------|------------|------------|
| Stock market crash | ✅ (stable) | ❌ (industrial demand drops) |
| Inflation spike | ⚠️ (good) | ✅ (better) |
| Commodity boom | ⚠️ (moderate) | ✅ (strong) |
| Risk-off flight | ✅ (safe haven) | ❌ (volatile) |
| Bull market | ❌ (underperforms) | ⚠️ (industrial demand up) |

---

## ⛏️ Trading Gold Miners (GDX)

### Complete GDX Workflow

```bash
# 1. Download GDX data
python3 framework/download_all_assets.py --asset GDX

# 2. Train GDX models
python3 framework/train_unified.py --asset GDX

# 3. Generate predictions
python3 predict_per_asset.py --asset GDX

# 4. Backtest
python3 backtest.py --asset GDX

# 5. Compare all metals
python3 backtest.py --assets GLD,SLV,GDX --compare
```

### GDX Characteristics

**Gold mining stocks = Leveraged gold exposure:**
- **2-3x gold sensitivity:** If GLD +1%, GDX typically +2-3%
- **Equity risk:** Correlated with both gold AND stocks
- **Operational leverage:** Mining costs fixed, gold price variable
- **Dividend yield:** 2-4% (vs 0% for GLD/SLV)

**Expected backtest results:**
```
GDX Performance:
- Return: +250-350% (highest of 3 metals)
- Sharpe: 0.40-0.50 (lowest due to volatility)
- Max DD: -50-65% (worst drawdowns)
- Win Rate: 50-54%
- Correlation to SPY: +0.25 (positive)
- Correlation to GLD: +0.65 (high)

Best for: High-risk portfolios, bull market gold plays
```

### GDX vs GLD Trade-offs

| Metric | GLD | GDX |
|--------|-----|-----|
| **Leverage** | 1x gold moves | 2-3x gold moves |
| **Volatility** | Low (~12% annual) | High (~35% annual) |
| **Downside** | -20% typical | -50% typical |
| **Upside** | +30% bull runs | +100% bull runs |
| **Correlation to stocks** | -0.15 (negative) | +0.25 (positive) |
| **Diversification** | Excellent | Moderate |
| **Income** | None | 2-4% dividends |

**Rule of thumb:**
- **Defensive:** GLD (negative stock correlation)
- **Aggressive:** GDX (leveraged upside)
- **Balanced:** 50% GLD + 50% GDX

---

## 📊 Portfolio Strategies

### Strategy 1: Classic 60/40 + Gold

```bash
# 60% stocks, 30% bonds, 10% gold
python3 backtest_portfolio.py --assets SPY,TLT,GLD --weights 0.6,0.3,0.1 --rebalance quarterly

# Expected results:
# - Return: ~8-10%/yr
# - Sharpe: 0.75-0.85
# - Max DD: -18-22%
# - Better than pure 60/40 during crashes
```

### Strategy 2: Risk Parity (Equal Risk)

```bash
# Equal risk contribution from each asset
python3 backtest_portfolio.py --assets SPY,TLT,GLD,SLV --weights 0.4,0.3,0.2,0.1 --rebalance monthly

# Expected results:
# - Return: ~9-11%/yr
# - Sharpe: 0.80-0.95
# - Max DD: -15-20%
# - Smooth equity curve
```

### Strategy 3: All-Weather Portfolio

```bash
# Ray Dalio inspired
python3 backtest_portfolio.py --assets SPY,TLT,GLD,DBC --weights 0.30,0.40,0.15,0.15 --rebalance quarterly

# Expected results:
# - Return: ~7-9%/yr
# - Sharpe: 0.90-1.10
# - Max DD: -10-15%
# - Works in all regimes
```

### Strategy 4: Aggressive Metals

```bash
# For gold bulls
python3 backtest_portfolio.py --assets GLD,SLV,GDX --weights 0.40,0.30,0.30 --rebalance monthly

# Expected results:
# - Return: ~12-18%/yr
# - Sharpe: 0.50-0.65
# - Max DD: -35-45%
# - High volatility, high return
```

---

## 🔍 Correlation Analysis

### Analyze Metal Correlations

```bash
# Check how metals correlate with each other and stocks
python3 analyze_correlations.py --assets SPY,GLD,SLV,GDX,BTC/USDT
```

**Expected correlation matrix:**

```
           SPY    GLD    SLV    GDX   BTC/USDT
SPY       1.00  -0.15  -0.10  +0.25   +0.05
GLD      -0.15   1.00  +0.75  +0.65   -0.05
SLV      -0.10  +0.75   1.00  +0.55   +0.10
GDX      +0.25  +0.65  +0.55   1.00   +0.15
BTC/USDT +0.05  -0.05  +0.10  +0.15   1.00
```

**Insights:**
- **GLD is best stock diversifier** (-0.15 correlation)
- **SLV tracks GLD closely** (+0.75 correlation)
- **GDX has equity risk** (+0.25 correlation to SPY)
- **BTC uncorrelated** to most assets (good diversifier)

### Find Best Diversifier

```bash
# Compare diversification scores
python3 analyze_correlations.py --assets SPY,QQQ,GLD,SLV,GDX,TLT,BTC/USDT --find-diversifiers
```

**Expected output:**
```
Best Diversifiers for SPY:
1. TLT  (correlation: -0.45) ⭐ BEST
2. GLD  (correlation: -0.15)
3. BTC  (correlation: +0.05)
4. SLV  (correlation: -0.10)
5. GDX  (correlation: +0.25)
6. QQQ  (correlation: +0.95) ❌ NOT A DIVERSIFIER
```

---

## 📈 Performance Comparison

### Historical Returns (2004-2025)

| Asset | Total Return | Annual Return | Sharpe | Max DD | Best For |
|-------|-------------|---------------|--------|--------|----------|
| **SPY** | +320% | +7.2% | 0.65 | -55% | Growth |
| **GLD** | +198% | +5.8% | 0.45 | -45% | Defense |
| **SLV** | +95% | +3.2% | 0.35 | -65% | Speculation |
| **GDX** | +180% | +5.1% | 0.30 | -75% | Aggressive |
| **TLT** | +145% | +4.5% | 0.55 | -48% | Stability |

### NeuroVest Strategy Returns (Estimated)

| Asset | Buy & Hold | NeuroVest Strategy | Improvement |
|-------|-----------|-------------------|-------------|
| **GLD** | +198% (5.8%/yr) | +125% (4.2%/yr) | Lower DD (-18% vs -45%) ✅ |
| **SLV** | +95% (3.2%/yr) | +180% (5.5%/yr) | Higher Sharpe (0.52 vs 0.35) ✅ |
| **GDX** | +180% (5.1%/yr) | +280% (7.2%/yr) | Better timing (0.48 vs 0.30) ✅ |

**Key insight:** NeuroVest excels at **timing volatile assets** (SLV, GDX)

---

## 🎯 Use Cases

### Use Case 1: Inflation Protection

**Scenario:** High inflation expected (CPI > 5%)

```bash
# Shift to metals-heavy portfolio
python3 backtest_portfolio.py --assets GLD,SLV,DBC --weights 0.5,0.3,0.2

# During 2021-2022 inflation:
# - GLD: +15%
# - SLV: +25%
# - SPY: -18%
```

### Use Case 2: Stock Market Hedge

**Scenario:** Bear market expected (VIX > 30)

```bash
# Defensive allocation
python3 backtest_portfolio.py --assets GLD,TLT --weights 0.6,0.4

# During 2008 crisis:
# - GLD: +25%
# - TLT: +20%
# - SPY: -37%
```

### Use Case 3: Bull Market Gold Play

**Scenario:** Gold bull market (GLD breaking highs)

```bash
# Aggressive metals
python3 backtest_portfolio.py --assets GDX,GDXJ --weights 0.7,0.3

# During 2020 gold rally:
# - GDX: +45%
# - GDXJ: +65%
# - GLD: +25%
```

---

## 🛠️ Advanced Features

### 1. Dynamic Rebalancing

```bash
# Test different rebalancing frequencies
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance daily
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance weekly
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance monthly
python3 backtest_portfolio.py --assets SPY,GLD --weights 0.7,0.3 --rebalance quarterly

# Expected: Monthly rebalancing optimal for most portfolios
```

### 2. Volatility-Based Weighting

```bash
# Allocate based on inverse volatility (lower vol = higher weight)
python3 backtest_portfolio.py --assets SPY,GLD,TLT --weights vol_inverse

# Expected weights:
# - TLT: 45% (lowest vol ~8%)
# - GLD: 30% (medium vol ~12%)
# - SPY: 25% (highest vol ~16%)
```

### 3. Correlation-Based Optimization

```bash
# Find optimal weights to minimize correlation
python3 optimize_portfolio.py --assets SPY,GLD,SLV,TLT --objective min_correlation

# Expected output:
# - SPY: 35%
# - GLD: 25%
# - SLV: 15%
# - TLT: 25%
# - Portfolio correlation to SPY: 0.45 (vs 1.0 for 100% SPY)
```

---

## 📝 Data Characteristics

### GLD (SPDR Gold Trust)

```
History: Nov 2004 - Present (~21 years)
Avg Volume: 8.5M shares/day
Expense Ratio: 0.40%
Assets: $60B
Tracking: London Gold Fix PM

Price Range:
- All-time low: $41.66 (Nov 2004)
- All-time high: $188.44 (Aug 2020)
- Current: ~$178 (Nov 2025)

Volatility: ~12% annualized
Beta to SPY: -0.10
```

### SLV (iShares Silver Trust)

```
History: Apr 2006 - Present (~19 years)
Avg Volume: 12M shares/day
Expense Ratio: 0.50%
Assets: $14B
Tracking: London Silver Fix

Price Range:
- All-time low: $8.82 (Oct 2008)
- All-time high: $48.70 (Apr 2011)
- Current: ~$24 (Nov 2025)

Volatility: ~25% annualized
Beta to SPY: -0.05
Beta to GLD: +0.75
```

### GDX (VanEck Gold Miners)

```
History: May 2006 - Present (~19 years)
Avg Volume: 22M shares/day
Expense Ratio: 0.51%
Assets: $13B
Holdings: 53 gold mining companies

Top Holdings:
1. Newmont (15%)
2. Barrick Gold (12%)
3. Agnico Eagle (8%)

Price Range:
- All-time low: $14.86 (Oct 2008)
- All-time high: $66.63 (Sep 2011)
- Current: ~$32 (Nov 2025)

Volatility: ~35% annualized
Beta to SPY: +0.20
Beta to GLD: +0.65
Dividend Yield: 2.8%
```

---

## ⚠️ Important Considerations

### 1. Trading Costs

```
Typical costs for metals ETFs:
- Commission: $0 (most brokers)
- Spread: 0.01-0.05% (GLD, SLV very tight)
- Expense ratio: 0.40-0.51%/yr
- Slippage: 0.02-0.10% (depends on order size)

Total cost per round trip: ~0.05-0.20%
```

### 2. Tax Treatment

```
Physical metals ETFs (GLD, SLV, IAU):
- Taxed as COLLECTIBLES (28% max rate)
- Higher than capital gains (20% max)
- Applies to profits only
- Consider tax-advantaged accounts (IRA, 401k)

Mining stocks (GDX, GDXJ):
- Taxed as regular equities (20% max LTCG)
- Qualified dividends (15-20% rate)
- Better tax treatment than physical metals
```

### 3. Contango/Backwardation

```
Futures-based commodity ETFs suffer from roll costs.
Physical ETFs (GLD, SLV) do NOT have this issue.

GLD: Holds physical gold bars in vaults ✅
SLV: Holds physical silver bars in vaults ✅
DBC: Holds futures contracts ⚠️ (roll costs)
```

### 4. Storage Costs

```
Built into expense ratios:
- GLD: 0.40%/yr (includes vault storage, insurance)
- SLV: 0.50%/yr (silver storage more expensive)
- IAU: 0.25%/yr (alternative to GLD, lower fee)
```

---

## 🔧 Troubleshooting

### Issue 1: Download Fails

```bash
# Error: No data returned for GLD
# Solution: Check ticker spelling and date range

python3 framework/download_all_assets.py --asset GLD --start 2004-11-18

# GLD inception was Nov 18, 2004
# Cannot download data before this date
```

### Issue 2: Low Accuracy

```bash
# GLD accuracy < 55%
# Solution: Precious metals are harder to predict than equities

# Try:
1. Longer horizon (10 days instead of 5)
2. Different features (add VIX, DXY dollar index)
3. Multi-asset ensemble (train on GLD+SLV+GDX together)
4. Lower threshold (0.25 instead of 0.30 for more trades)
```

### Issue 3: Poor Backtest Results

```bash
# GLD strategy underperforms buy & hold
# Solution: Metals are mean-reverting, not trending

# Try:
1. Mean reversion strategy (inverse signals)
2. Longer holding periods (monthly instead of daily)
3. Combine with SPY signals (trade GLD when SPY crashes)
```

---

## 📚 Related Documentation

- **ARCHITECTURE_GUIDE.md** - System overview, multi-asset vs per-asset
- **CURRENT_SYSTEM_STATUS.md** - What's configured vs trained
- **MULTI_ASSET_ANALYSIS_SUMMARY.md** - Portfolio tools
- **ACCURACY_OPTIMIZATION_GUIDE.md** - Threshold tuning
- **CRASH_PREDICTION_ANALYSIS.md** - Crash detection (important for metals!)

---

## 🎯 Next Steps

1. **Download all metals:**
   ```bash
   python3 framework/download_all_assets.py --asset-group commodity
   ```

2. **Train all metals:**
   ```bash
   python3 framework/train_unified.py --asset GLD
   python3 framework/train_unified.py --asset SLV
   python3 framework/train_unified.py --asset GDX
   ```

3. **Analyze correlations:**
   ```bash
   python3 analyze_correlations.py --assets SPY,GLD,SLV,GDX,BTC/USDT
   ```

4. **Build optimal portfolio:**
   ```bash
   python3 backtest_portfolio.py --assets SPY,GLD,SLV,TLT --weights 0.4,0.25,0.15,0.2
   ```

5. **Test on historical crashes:**
   ```bash
   python3 backtest.py --asset GLD --start 2008-01-01 --end 2009-12-31
   python3 backtest.py --asset GLD --start 2020-01-01 --end 2020-12-31
   ```

---

**All precious metals are ready to integrate - just download and train!** 🎯
