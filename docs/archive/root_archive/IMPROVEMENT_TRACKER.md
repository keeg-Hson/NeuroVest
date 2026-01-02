# Accuracy Improvement Tracker

**Goal**: Increase accuracy from 57.1% to 70-75%
**Start Date**: 2025-11-16
**Status**: 📋 Planning Phase

---

## Progress Summary

| Tier | Status | Expected Impact | Actual Impact | Completion Date |
|------|--------|----------------|---------------|-----------------|
| Tier 1 | ⏳ Pending | +6% (57% → 63%) | TBD | TBD |
| Tier 2 | ⏳ Pending | +3-5% (63% → 66-68%) | TBD | TBD |
| Tier 3 | ⏳ Pending | +4-8% (66-68% → 70-75%) | TBD | TBD |
| Tier 4 | ⏳ Pending | +0-5% (experimental) | TBD | TBD |

---

## TIER 1: Immediate Improvements (Week 1)

### ⏳ Task 1.1: Lower Prediction Threshold

**Expected Impact**: +5.1% accuracy (57.1% → 62.2%)
**Effort**: 15 minutes
**Status**: ⏳ Not Started

**Steps**:
- [ ] Edit `predict_multi_asset_ensemble.py` line 132: `threshold = 0.45`
- [ ] Edit `predict.py` line 218: `threshold = 0.45` (if using single model)
- [ ] Generate new predictions: `python predict_multi_asset_ensemble.py`
- [ ] Run backtest: `python backtest.py`
- [ ] Document results below

**Results**:
```
Before (threshold=0.55):
- Accuracy: 57.1%
- Precision: 99.5%
- Recall: 7.3%
- Trades: 221

After (threshold=0.45):
- Accuracy: TBD
- Precision: TBD
- Recall: TBD
- Trades: TBD
```

**Validation**:
- [ ] Backtest return improved?
- [ ] Sharpe ratio maintained or improved?
- [ ] Max drawdown acceptable (<25%)?
- [ ] Win rate improved?

---

### ⏳ Task 1.2: Switch to Ensemble Predictor

**Expected Impact**: +1% accuracy
**Effort**: 5 minutes
**Status**: ⏳ Not Started

**Steps**:
- [ ] Edit `run_daily_pipeline.py` line 18
- [ ] Change from: `subprocess.run(["python3", "predict.py"], check=True)`
- [ ] Change to: `subprocess.run(["python3", "predict_multi_asset_ensemble.py"], check=True)`
- [ ] Test pipeline: `python run_daily_pipeline.py` (or just prediction step)
- [ ] Verify `logs/daily_predictions.csv` is updated
- [ ] Document results below

**Results**:
```
Before (single XGBoost):
- Accuracy: TBD

After (ensemble):
- Accuracy: TBD
- Improvement: TBD
```

---

### 📊 Tier 1 Results

**Target**: 57.1% → 63% accuracy (+5.9pp)

**Actual**:
- Starting accuracy: 57.1%
- After threshold adjustment: TBD
- After ensemble switch: TBD
- Final Tier 1 accuracy: TBD
- **Total improvement: TBD**

**Date Completed**: TBD

---

## TIER 2: Feature Engineering (Week 2-3)

### ⏳ Task 2.1: Remove Zero-Importance Features

**Expected Impact**: +0.5-1% accuracy
**Effort**: 30 minutes
**Status**: ⏳ Not Started

**Steps**:
- [ ] Backup current `utils.py`: `cp utils.py utils.py.backup`
- [ ] Edit `utils.py` get_feature_list() function
- [ ] Remove these 20 features:
  - [ ] RSI_x_NewsZ
  - [ ] Sent_x_Vol
  - [ ] RSI_x_RedditZ
  - [ ] News_Sent_Z20
  - [ ] Reddit_Sent_Z20
  - [ ] RSI_Neutral
  - [ ] RSI_Oversold
  - [ ] RSI_Overbought
  - [ ] High_Fear
  - [ ] Bull_Market
  - [ ] VIX_Spike (the feature, not the CSV feature Vol_Spike)
  - [ ] Near_52w_Low
  - [ ] Strong_Trend
  - [ ] MA_20_50_Cross
  - [ ] Pct_Above_MA20
  - [ ] Sector_MedianRet_20
  - [ ] Sector_Dispersion_20
  - [ ] asset_type_crypto
  - [ ] Vol_Expanding
  - [ ] Return_Reversal
- [ ] Verify feature count reduced: `python -c "from utils import get_feature_list; print(len(get_feature_list()))"`
- [ ] Expected: ~130 features (150 - 20)
- [ ] Document results below

**Results**:
```
Before: 150 features
After: TBD features
Removed: TBD features
```

---

### ⏳ Task 2.2: Verify Cross-Asset/Macro Features Are Loading

**Expected Impact**: Prerequisite for 2.3
**Effort**: 10 minutes
**Status**: ⏳ Not Started

**Steps**:
- [ ] Run test: `python test_data_integration.py`
- [ ] Verify cross-asset features have real values (not zeros):
  - [ ] Credit_Ratio
  - [ ] Yield_10Y
  - [ ] DXY_Level
  - [ ] Realized_Vol_20
- [ ] Verify macro features have real values:
  - [ ] Macro_10Y_Yield
  - [ ] Rate_Change_3m
  - [ ] Low_Rate_Regime
- [ ] Document results below

**Results**:
```
Cross-asset features loaded: YES/NO
Macro features loaded: YES/NO
Sample values: TBD
```

---

### ⏳ Task 2.3: Update Feature Selection k_choices

**Expected Impact**: +1% accuracy
**Effort**: 2 minutes
**Status**: ⏳ Not Started

**Steps**:
- [ ] Edit `train.py` line 598
- [ ] Change from: `k_choices = sorted(set([15, 20, 25, 30, max(15, max_k // 2), max_k]))`
- [ ] Change to: `k_choices = sorted(set([30, 40, 50, 60, max(30, max_k // 2), max_k]))`
- [ ] Document change below

**Results**:
```
Before: k_choices = [15, 20, 25, 30, ...]
After: k_choices = TBD
```

---

### ⏳ Task 2.4: Retrain Multi-Asset Models

**Expected Impact**: +2-3% accuracy (with all Tier 2 changes)
**Effort**: 30-60 minutes (mostly training time)
**Status**: ⏳ Not Started

**Steps**:
- [ ] Ensure Tasks 2.1, 2.2, 2.3 are complete
- [ ] Run: `python train_multi_asset.py`
- [ ] Monitor training progress
- [ ] Check CV scores (should improve from baseline)
- [ ] Verify models created:
  - [ ] xgboost_multi_asset.pkl
  - [ ] lightgbm_multi_asset.pkl
  - [ ] catboost_multi_asset.pkl
- [ ] Check feature count in models/multi_asset_features.txt
- [ ] Expected: ~130 features
- [ ] Document results below

**Training Results**:
```
XGBoost:
- CV Score: TBD
- Test Accuracy: TBD
- Best k: TBD
- Training time: TBD

LightGBM:
- CV Score: TBD
- Test Accuracy: TBD

CatBoost:
- CV Score: TBD
- Test Accuracy: TBD
```

---

### ⏳ Task 2.5: Generate Predictions and Backtest

**Expected Impact**: Validate Tier 2 improvements
**Effort**: 10 minutes
**Status**: ⏳ Not Started

**Steps**:
- [ ] Generate predictions: `python predict_multi_asset_ensemble.py`
- [ ] Run backtest: `python backtest.py`
- [ ] Compare to baseline (Tier 1 results)
- [ ] Document results below

**Backtest Results**:
```
Before Tier 2:
- Accuracy: TBD (from Tier 1)
- Return: TBD
- Sharpe: TBD

After Tier 2:
- Accuracy: TBD
- Return: TBD
- Sharpe: TBD
- Improvement: TBD
```

---

### 📊 Tier 2 Results

**Target**: 63% → 66-68% accuracy (+3-5pp)

**Actual**:
- Starting accuracy (after Tier 1): TBD
- After feature removal: TBD
- After k_choices update: TBD
- After retraining: TBD
- Final Tier 2 accuracy: TBD
- **Total improvement: TBD**

**Date Completed**: TBD

---

## TIER 3: Advanced Improvements (Month 2)

### ⏳ Task 3.1: Add Interaction Features

**Expected Impact**: +1-2% accuracy
**Effort**: 2 hours
**Status**: ⏳ Not Started

**Steps**:
- [ ] Edit `utils.py` add_features() function (around line 750)
- [ ] Add these interaction features:
  - [ ] Near_52w_High_x_Volatility
  - [ ] Near_52w_High_x_KC_Width
  - [ ] Stoch_K_x_Volatility
  - [ ] Return_Lag3_x_Volatility
  - [ ] Return_Lag5_x_ATR
  - [ ] Near_52w_High_x_Return_Lag3
  - [ ] BB_PctB_x_Stoch_K
- [ ] Add to get_feature_list()
- [ ] Test: `python test_data_integration.py`
- [ ] Retrain: `python train_multi_asset.py`
- [ ] Document results below

**Results**:
```
Features added: TBD
Accuracy improvement: TBD
```

---

### ⏳ Task 3.2: Label Optimization Experiments

**Expected Impact**: +1-3% accuracy
**Effort**: 4 hours
**Status**: ⏳ Not Started

**Experiments to Run**:
- [ ] **Exp 1**: Fixed threshold (no volatility adjustment)
  - [ ] `add_forward_returns_and_labels(df, volatility_adjusted=False, pos_threshold=0.005)`
  - [ ] Train model and record CV score: TBD
- [ ] **Exp 2**: Different threshold values
  - [ ] Test pos_threshold = 0.003, CV score: TBD
  - [ ] Test pos_threshold = 0.005, CV score: TBD
  - [ ] Test pos_threshold = 0.007, CV score: TBD
  - [ ] Test pos_threshold = 0.010, CV score: TBD
- [ ] **Exp 3**: Different holding periods
  - [ ] Test hold_period = 1, CV score: TBD
  - [ ] Test hold_period = 2, CV score: TBD
  - [ ] Test hold_period = 3, CV score: TBD
  - [ ] Test hold_period = 5, CV score: TBD
- [ ] Select best configuration
- [ ] Retrain with optimal labels
- [ ] Document results below

**Best Configuration**:
```
Volatility-adjusted: YES/NO
Threshold: TBD
Hold period: TBD
CV Score: TBD
Improvement: TBD
```

---

### ⏳ Task 3.3: Build Regime-Specific Models

**Expected Impact**: +2-3% accuracy
**Effort**: 1 week
**Status**: ⏳ Not Started

**Steps**:
- [ ] Create `train_regime_models.py` script
- [ ] Split data by VIX:
  - [ ] High vol: VIX > 20
  - [ ] Low vol: VIX < 15
  - [ ] Normal vol: 15 <= VIX <= 20
- [ ] Train 3 separate models
- [ ] Create routing logic in prediction script
- [ ] Backtest regime-switching strategy
- [ ] Compare to single model
- [ ] Document results below

**Results**:
```
High-vol model accuracy: TBD
Low-vol model accuracy: TBD
Normal-vol model accuracy: TBD
Regime-switching accuracy: TBD
Improvement vs single model: TBD
```

---

### 📊 Tier 3 Results

**Target**: 66-68% → 70-75% accuracy (+4-7pp)

**Actual**:
- Starting accuracy (after Tier 2): TBD
- After interaction features: TBD
- After label optimization: TBD
- After regime models: TBD
- Final Tier 3 accuracy: TBD
- **Total improvement: TBD**

**Date Completed**: TBD

---

## TIER 4: Experimental (Optional)

### ⏳ Task 4.1: Deep Learning Models

**Expected Impact**: +2-5% accuracy (if successful)
**Effort**: 2-3 weeks
**Status**: ⏳ Not Started

**Notes**: Only pursue if Tier 3 results are positive and we want to push beyond 70-75%.

---

### ⏳ Task 4.2: Alternative Data Sources

**Expected Impact**: +2-4% accuracy (if successful)
**Effort**: 1-3 months
**Status**: ⏳ Not Started

**Notes**: Long-term project, requires data acquisition and quality assessment.

---

## Final Results Summary

**Overall Progress**:

| Metric | Baseline | After T1 | After T2 | After T3 | After T4 | Target |
|--------|----------|----------|----------|----------|----------|--------|
| Accuracy | 57.1% | TBD | TBD | TBD | TBD | 70-75% |
| Win Rate | 52.1% | TBD | TBD | TBD | TBD | 60%+ |
| Sharpe | 1.71 | TBD | TBD | TBD | TBD | 2.0+ |
| Max DD | -18.6% | TBD | TBD | TBD | TBD | <25% |

**Date Started**: 2025-11-16
**Date Completed**: TBD
**Total Time Investment**: TBD
**Total Accuracy Improvement**: TBD

---

## Lessons Learned

### What Worked
- TBD

### What Didn't Work
- TBD

### Surprises
- TBD

### Next Steps After 70%
- TBD

---

**Last Updated**: 2025-11-16
