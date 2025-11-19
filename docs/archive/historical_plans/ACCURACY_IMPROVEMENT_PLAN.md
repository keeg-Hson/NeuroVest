# Complete Accuracy Improvement Action Plan

## Executive Summary

**Current State**: 56-60% accuracy (barely better than random 50%)
**Target State**: 62-68% accuracy (realistic with these fixes)
**Ceiling**: 70-75% accuracy (would require alternative data sources)

---

## Phase 1: QUICK WINS (1-2 hours) - Expected +5-8% accuracy

### 1.1 Reduce Features (106 → 15)
**File**: `utils.py` → `get_feature_list()`
**Action**: Replace with 15 core features (see FEATURE_REDUCTION_PLAN.md)
**Impact**: Reduces overfitting, +2-3% test accuracy
**Effort**: 30 minutes

### 1.2 Fix Data Leakage  
**File**: `utils.py` → `add_features()`
**Action**: 
- Replace `cumsum()` with `rolling().sum()` for OBV
- Remove VWAP or use rolling window
**Impact**: More honest evaluation, +1-2% test accuracy
**Effort**: 20 minutes

### 1.3 Fix Sample Weighting
**File**: `config.py` → `TRAIN_CFG`
**Action**: Set `min_weight=1.0, max_weight=1.0, weight_power=1.0`
**Impact**: Reduces outlier overfitting, +1-2% test accuracy
**Effort**: 5 minutes

### 1.4 Extend SPY Data (2010 → 1993)
**Command**:
```bash
python -c "
import yfinance as yf
spy = yf.download('SPY', start='1993-01-29')
spy.to_csv('data/SPY.csv')
"
```
**Impact**: 2x more training data, +1-2% test accuracy
**Effort**: 5 minutes

**Phase 1 Total**: ~1 hour, **+5-8% accuracy improvement**

---

## Phase 2: MEDIUM EFFORT (1-2 days) - Expected +3-5% accuracy

### 2.1 Multi-Asset Training
**Create**: `train_multi_asset.py`
**Action**: Train on SPY, QQQ, IWM, DIA, EFA, EEM, GLD, AGG (8 assets)
**Impact**: 64K training samples vs 5K, +3-4% test accuracy
**Effort**: 4 hours

### 2.2 Walk-Forward Validation
**Modify**: `comprehensive_model_evaluation.py`
**Action**: Replace single 80/20 split with 5-fold walk-forward
**Impact**: More robust evaluation, identifies overfitting
**Effort**: 2 hours

### 2.3 Longer Prediction Horizon
**File**: `config.py` → `TRAIN_CFG["horizon"]`
**Action**: Change from `horizon=1` (daily) to `horizon=5` (weekly)
**Impact**: Less noise, easier to predict, +2-3% accuracy
**Effort**: 30 minutes + retrain

### 2.4 Remove SMOTE (Making Class Imbalance Worse)
**File**: `train.py`
**Action**: Comment out SMOTE resampling
**Why**: SMOTE creates synthetic samples that don't reflect market reality
**Impact**: Better generalization, +1-2% accuracy
**Effort**: 10 minutes

**Phase 2 Total**: 1-2 days, **+3-5% accuracy improvement**

---

## Phase 3: ADVANCED (1-2 weeks) - Expected +2-4% accuracy

### 3.1 Add Crypto Markets
**Why**: Crypto less efficient than SPY, easier to predict
**Action**: 
- Download BTC, ETH, SOL, AVAX, MATIC daily data (2018-2025)
- Train separate crypto models
- Compare performance
**Expected**: Crypto models may hit 65-70% accuracy
**Effort**: 3-4 days

### 3.2 Intraday Data (Hourly Bars)
**Action**: Use 1-hour SPY data instead of daily
**Gain**: 6x more samples per time period
**Trade-off**: Different market dynamics, more noise
**Effort**: 2-3 days to adapt features

### 3.3 Regime-Aware Models
**Action**: Train separate models for:
- Bull market (above 200-day MA)
- Bear market (below 200-day MA)  
- High volatility periods
- Low volatility periods
**Impact**: Specialized models perform better, +2-3% accuracy
**Effort**: 3-4 days

### 3.4 Alternative Data (Advanced)
**Options**:
- Google Trends data (search volume for "recession", "stock crash")
- Reddit sentiment (WSB, investing subreddits)
- Options flow data (unusual options activity)
- Economic indicators (unemployment, GDP, inflation)
**Impact**: Could push accuracy to 70-75%
**Effort**: 1-2 weeks per data source

**Phase 3 Total**: 1-2 weeks, **+2-4% accuracy improvement**

---

## Expected Final Results

| Phase | Accuracy | Time | Cumulative |
|-------|----------|------|------------|
| Baseline | 56-60% | - | 58% |
| Phase 1 | +5-8% | 1 hour | **63-66%** |
| Phase 2 | +3-5% | 1-2 days | **66-71%** |
| Phase 3 | +2-4% | 1-2 weeks | **68-75%** |

---

## Realistic Expectations

### With Phase 1 Only (63-66% accuracy)
```
Win Rate: 55-58%
Avg Profit/Trade: 0.10-0.15%
After 0.15% costs: -0.05 to 0.00% (BREAK EVEN)
```
**Still loses money**, but much closer to profitability.

### With Phase 1+2 (66-71% accuracy)
```
Win Rate: 58-62%
Avg Profit/Trade: 0.20-0.30%
After 0.15% costs: +0.05 to +0.15% (PROFITABLE!)
```
**Marginally profitable** (but barely worth the risk).

### With Phase 1+2+3 (68-75% accuracy)
```
Win Rate: 60-65%
Avg Profit/Trade: 0.30-0.50%
After 0.15% costs: +0.15 to +0.35% (PROFITABLE)
```
**Actually profitable**, but still modest returns.

---

## What Won't Work (Don't Waste Time On)

❌ **More hyperparameter tuning** - Already close to optimal
❌ **More complex models** (deep neural nets) - Will overfit worse
❌ **More engineered features** - Makes overfitting worse
❌ **Ensemble of similar models** - Doesn't add diversity
❌ **Trying to predict daily 0.5% SPY moves** - Too much noise

---

## Recommended Implementation Order

### Week 1: Quick Wins
- [ ] Day 1: Feature reduction (106 → 15)
- [ ] Day 1: Fix data leakage (OBV, VWAP)
- [ ] Day 1: Fix sample weighting
- [ ] Day 1: Extend SPY data to 1993
- [ ] Day 1: Retrain and evaluate

**Checkpoint**: Should see 63-66% accuracy

### Week 2: Multi-Asset
- [ ] Day 2: Download 8 ETFs
- [ ] Day 2-3: Create multi-asset training pipeline
- [ ] Day 3-4: Train and evaluate
- [ ] Day 4-5: Walk-forward validation

**Checkpoint**: Should see 66-71% accuracy

### Week 3+: Advanced (Optional)
- [ ] Week 3: Crypto models
- [ ] Week 4: Regime-aware models
- [ ] Week 5+: Alternative data sources

**Checkpoint**: Should see 68-75% accuracy

---

## Success Metrics

**Minimum Viable** (Phase 1):
- ✅ Test accuracy: 63%+
- ✅ Win rate: 55%+
- ✅ Overfitting gap: <5% (train-test accuracy difference)

**Target** (Phase 1+2):
- ✅ Test accuracy: 68%+
- ✅ Win rate: 60%+
- ✅ Sharpe ratio: 1.5+
- ✅ Max drawdown: <15%
- ✅ Actually profitable after costs

**Stretch Goal** (Phase 1+2+3):
- ✅ Test accuracy: 72%+
- ✅ Win rate: 65%+
- ✅ Sharpe ratio: 2.0+
- ✅ Consistently profitable

---

## Bottom Line

**START WITH PHASE 1** (1 hour investment):
1. Reduce features to 15
2. Fix data leakage
3. Fix sample weighting
4. Get more SPY data

This should get you to **63-66% accuracy** which is a **7-10 percentage point improvement** over current 56-60%.

If that works, **proceed to Phase 2** for multi-asset training and longer horizons.

**Only attempt Phase 3** if you want to build a production trading system.
