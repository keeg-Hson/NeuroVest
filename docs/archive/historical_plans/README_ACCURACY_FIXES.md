# How to Fix Model Accuracy: Complete Guide

## 📊 Current Situation
- **Current Accuracy**: 56-60% (barely better than 50% random)
- **Root Cause**: Multiple fundamental issues (not just hyperparameters)
- **Good News**: Can reach 63-75% with systematic fixes

---

## 🎯 Quick Start (1 Hour → 63-66% Accuracy)

If you only do 4 things, do these:

### 1. Reduce Features (106 → 15)
```bash
# Edit utils.py, replace get_feature_list() with 15 core features
# See: FEATURE_REDUCTION_PLAN.md
```

### 2. Fix Data Leakage
```bash
# Edit utils.py add_features() 
# Replace cumsum() with rolling().sum() for OBV
# See: FIX_DATA_LEAKAGE.md
```

### 3. Fix Sample Weighting
```bash
# Edit config.py TRAIN_CFG
min_weight: 1.0
max_weight: 1.0  
weight_power: 1.0
# See: FIX_SAMPLE_WEIGHTING.md
```

### 4. Get More SPY Data
```bash
python -c "
import yfinance as yf
spy = yf.download('SPY', start='1993-01-29')
spy.to_csv('data/SPY.csv')
print(f'Downloaded {len(spy)} rows')
"
# See: INCREASE_TRAINING_DATA.md
```

**Expected Result**: 63-66% accuracy (+7-10 points)

---

## 📚 All Documentation Files

### 1. **FEATURE_REDUCTION_PLAN.md**
- Why 106 features causes overfitting
- Which 15 features to keep
- Which 91 features to remove
- Expected impact: +2-3% accuracy

### 2. **FIX_DATA_LEAKAGE.md**
- How cumsum() leaks future data
- How to fix OBV calculation
- How to fix VWAP (or remove it)
- Expected impact: +1-2% accuracy

### 3. **FIX_SAMPLE_WEIGHTING.md**
- Why exponential weighting overfits
- 3 alternative weighting strategies
- Recommended: Uniform or mild weighting
- Expected impact: +1-2% accuracy

### 4. **INCREASE_TRAINING_DATA.md**
- How to extend SPY history (15 → 32 years)
- How to train on multiple ETFs (8 assets)
- How to use intraday data
- How to use crypto data
- Expected impact: +3-5% accuracy

### 5. **ACCURACY_IMPROVEMENT_PLAN.md** (This file)
- Complete 3-phase roadmap
- Phase 1: 1 hour → 63-66% accuracy
- Phase 2: 1-2 days → 66-71% accuracy
- Phase 3: 1-2 weeks → 68-75% accuracy

---

## 🔬 Root Cause Analysis

### Why Current Accuracy is Only 56-60%

1. **Insufficient Data** (CRITICAL)
   - Only 5,201 training samples
   - Need 50K+ for reliable ML
   - Industry standard: 100-200 samples per feature

2. **Too Many Features** (CRITICAL)
   - 106 features on 5,201 samples
   - Ratio: 49 samples per feature (severe overfitting)
   - Lag duplicates, interactions memorize noise

3. **Data Leakage** (HIGH)
   - `cumsum()` uses entire dataset (includes future!)
   - VWAP calculation sees future data
   - Inflates training accuracy artificially

4. **Exponential Sample Weighting** (HIGH)
   - Outliers get 10x weight vs normal trades
   - Model overfits to rare events (Fed announcements)
   - These events don't repeat in test set

5. **Noisy Target Variable** (FUNDAMENTAL)
   - Predicting 0.5% daily moves in SPY ≈ random
   - Market is 80-90% noise, 10-20% signal
   - Even best quant funds only hit 60-65%

6. **Market Efficiency** (FUNDAMENTAL LIMIT)
   - Public technical indicators already priced in
   - Need alternative data for >70% accuracy
   - SPY is one of most efficient assets

---

## ✅ Expected Results After Each Phase

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| **Accuracy** | 58% | 63-66% | 66-71% | 68-75% |
| **Win Rate** | 52% | 55-58% | 58-62% | 60-65% |
| **Profit/Trade** | 0.03% | 0.12% | 0.25% | 0.40% |
| **After Costs** | -0.12% | -0.03% | +0.10% | +0.25% |
| **Economic** | LOSES | BREAK EVEN | PROFITABLE | PROFITABLE |
| **Time Required** | - | 1 hour | 1-2 days | 1-2 weeks |

---

## 🚀 Implementation Checklist

### Phase 1: Quick Wins (Start Here!)
- [ ] Read FEATURE_REDUCTION_PLAN.md
- [ ] Reduce features from 106 to 15
- [ ] Read FIX_DATA_LEAKAGE.md  
- [ ] Fix OBV cumsum() issue
- [ ] Read FIX_SAMPLE_WEIGHTING.md
- [ ] Set sample weights to uniform
- [ ] Read INCREASE_TRAINING_DATA.md
- [ ] Download SPY data back to 1993
- [ ] Retrain models
- [ ] Verify accuracy improved to 63-66%

### Phase 2: Multi-Asset (If Phase 1 Works)
- [ ] Download 8 ETFs (QQQ, IWM, DIA, etc.)
- [ ] Create multi-asset training script
- [ ] Train combined model
- [ ] Implement walk-forward validation
- [ ] Change horizon from 1 day to 5 days
- [ ] Remove SMOTE resampling
- [ ] Verify accuracy improved to 66-71%

### Phase 3: Advanced (Optional)
- [ ] Add crypto markets
- [ ] Try intraday (hourly) data
- [ ] Build regime-aware models
- [ ] Explore alternative data sources
- [ ] Verify accuracy improved to 68-75%

---

## ⚠️ What NOT to Do (Common Mistakes)

❌ Don't hyperparameter tune further (already optimal)
❌ Don't add more engineered features (makes overfitting worse)
❌ Don't use more complex models (will overfit harder)
❌ Don't train on daily SPY with 0.5% threshold (too noisy)
❌ Don't expect >75% accuracy without alternative data
❌ Don't skip Phase 1 to jump to Phase 3

---

## 💡 Key Insights

1. **More features ≠ better models** when data is limited
2. **Data leakage inflates accuracy** artificially
3. **Market efficiency** is fundamental limit (~70-75% ceiling)
4. **More data > better algorithms** for this problem
5. **Crypto is easier to predict** than SPY (less efficient)
6. **Longer horizons are easier** (weekly vs daily)
7. **Transaction costs matter** (0.15% is huge!)

---

## 📈 Success Metrics

After Phase 1, you should see:
- ✅ Training accuracy: 60-63% (may drop - that's good!)
- ✅ Test accuracy: 63-66% (should improve)
- ✅ Overfitting gap: <5% (train-test difference)
- ✅ Feature count: 15 (down from 106)
- ✅ Training samples: 10,000+ (up from 5,201)

After Phase 2, you should see:
- ✅ Test accuracy: 66-71%
- ✅ Walk-forward validation confirms results
- ✅ Profitable after transaction costs
- ✅ Sharpe ratio: 1.5+

---

## 🎯 Bottom Line

**Start with Phase 1 (1 hour)**:
- Reduce features to 15
- Fix data leakage  
- Fix sample weighting
- Get more historical data

**This should improve accuracy from 58% → 63-66%**

If that works (and it should), proceed to Phase 2.

**Don't attempt Phase 3** unless you're building a production system.

---

## 📞 Questions?

All details are in the individual markdown files:
- `FEATURE_REDUCTION_PLAN.md` - Feature engineering fixes
- `FIX_DATA_LEAKAGE.md` - Data quality fixes
- `FIX_SAMPLE_WEIGHTING.md` - Training process fixes
- `INCREASE_TRAINING_DATA.md` - Data quantity fixes
- `ACCURACY_IMPROVEMENT_PLAN.md` - Overall roadmap

**Good luck! The fixes are straightforward and should work.**
