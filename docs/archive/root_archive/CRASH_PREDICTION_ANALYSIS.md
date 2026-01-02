# Crash Prediction Analysis - Why No CRASH Predictions?

**Date:** 2025-11-18
**Issue:** Model never predicts CRASH scenarios (class 0), only NORMAL and SPIKE
**Impact:** Cannot detect or warn about market crashes

---

## 🔍 Root Cause Analysis

### Current Prediction Distribution

```
CRASH  (0):     0  ( 0.00%)  ❌ NEVER PREDICTED
NORMAL (1):  4210  (64.76%)
SPIKE  (2):  2291  (35.24%)
```

### The Problem: Binary Training, 3-Class Prediction

**The system has a fundamental mismatch:**

1. **Models are trained on BINARY classification:**
   - Label 0: Forward return < threshold (unprofitable trade)
   - Label 1: Forward return >= threshold (profitable trade)
   - Source: `utils.py:add_forward_returns_and_labels()` line 1189/1195

2. **Predictions use 3-CLASS convention:**
   - Class 0: CRASH (severe drop)
   - Class 1: NORMAL (hold/neutral)
   - Class 2: SPIKE (strong gain)
   - Source: `predict_multi_asset_ensemble.py` line 161-163

3. **Mapping logic is broken:**
   ```python
   # Line 159: Binary prediction (0 or 1)
   ensemble_pred = (ensemble_prob >= threshold).astype(int)

   # Line 163: Maps to 3-class but SKIPS class 0!
   pred_012 = np.where(ensemble_pred == 1, 2, 1)
   # Binary 1 → Class 2 (SPIKE) ✓
   # Binary 0 → Class 1 (NORMAL) ✓
   # No path to Class 0 (CRASH) ❌
   ```

**Result:** Class 0 (CRASH) is **never assigned**, making crash prediction impossible.

---

## 📊 Evidence from Code

### Training (`train_multi_asset.py`)

```python
# Binary labels created here
spy_df = add_forward_returns_and_labels(
    spy_df,
    price_col="Close",
    horizon=5,  # 5-day forward return
    pos_threshold=0.005,  # 0.5% threshold
)

# Creates binary label:
# y = 0 if fwd_ret_net < 0.5%
# y = 1 if fwd_ret_net >= 0.5%
```

### Prediction (`predict_multi_asset_ensemble.py`)

```python
# Line 120: Extract probability of class 1 (positive)
probabilities[name] = probs[:, 1]

# Line 129: Average across ensemble
ensemble_prob = np.mean(list(probabilities.values()), axis=0)

# Line 159: Binary prediction
ensemble_pred = (ensemble_prob >= 0.30).astype(int)  # 0 or 1

# Line 163: Map to 3-class (but never assigns 0!)
pred_012 = np.where(ensemble_pred == 1, 2, 1)
# This ONLY outputs 1 or 2, never 0
```

### Prediction Statistics

Analysis of `daily_predictions.csv` confirms:

```
Crash Confidence Statistics:
- Mean Crash_Conf: 0.7318
- Max Crash_Conf:  0.9853
- Rows with Crash_Conf > 0.50: 6,264 / 6,501 (96.4%)
```

**Important:** Despite the column name "Crash_Conf", this actually represents **confidence in class 1 (NORMAL)**, not class 0 (CRASH). It's calculated as `1 - ensemble_prob` where `ensemble_prob` is the probability of class 2 (SPIKE).

---

## 🎯 Why This Matters

### Missing Crash Detection is Critical

**Without crash prediction, the system cannot:**

1. **Protect capital** during bear markets (2008, 2020, 2022)
2. **Go to cash** before major corrections
3. **Short positions** or buy puts for hedging
4. **Detect regime changes** from bull to bear
5. **Warn users** of elevated downside risk

**Examples of missed crash signals:**
- **Mar 2020 COVID crash:** -34% drop in 23 days
- **Feb 2018 VIX spike:** -10% in 9 days
- **Aug 2015 China selloff:** -11% in 6 days
- **Dec 2018 rate fears:** -20% in 3 months

**Current system behavior:**
- Would stay LONG during crashes (predicting SPIKE)
- Or HOLD (predicting NORMAL)
- **Never CASH/SHORT** (no CRASH predictions)

---

## 💡 Solution Options

### Option 1: Fix Binary Mapping (Quick Fix)

**Keep binary classification, clarify labels:**

```python
# Rename to match actual meaning
# Binary 0 → HOLD (low confidence, stay in cash)
# Binary 1 → LONG (high confidence, enter position)

pred_binary = (ensemble_prob >= threshold).astype(int)
# No "crash" class - just HOLD vs LONG
```

**Pros:**
- No retraining required
- Matches actual training data
- Clear semantics

**Cons:**
- Cannot detect crashes (no short/defensive signal)
- Binary only (no nuance)

### Option 2: Implement True 3-Class Training (Recommended)

**Train models on 3 classes: CRASH, NORMAL, SPIKE**

#### Step 1: Modify Label Creation

```python
def add_three_class_labels(
    df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 5,
    spike_threshold: float = 0.01,   # +1% = SPIKE
    crash_threshold: float = -0.01,  # -1% = CRASH
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
):
    """
    Create 3-class labels:
    - Class 0 (CRASH): fwd_ret_net < crash_threshold (-1%)
    - Class 1 (NORMAL): crash_threshold <= fwd_ret_net < spike_threshold
    - Class 2 (SPIKE): fwd_ret_net >= spike_threshold (+1%)
    """
    d = df.copy()
    cost = (fee_bps + slippage_bps) * 1e-4

    d["fwd_price"] = d[price_col].shift(-horizon)
    d["fwd_ret_raw"] = (d["fwd_price"] / d[price_col]) - 1.0
    d["fwd_ret_net"] = d["fwd_ret_raw"] - cost

    # 3-class labels
    d["y"] = 1  # Default NORMAL
    d.loc[d["fwd_ret_net"] >= spike_threshold, "y"] = 2  # SPIKE
    d.loc[d["fwd_ret_net"] < crash_threshold, "y"] = 0   # CRASH

    return d
```

#### Step 2: Retrain Models

```bash
# Update train_multi_asset.py to use add_three_class_labels()
python3 train_multi_asset.py

# This creates models that output 3 classes:
# probs[:, 0] = P(CRASH)
# probs[:, 1] = P(NORMAL)
# probs[:, 2] = P(SPIKE)
```

#### Step 3: Update Prediction Logic

```python
# predict_multi_asset_ensemble.py

# Extract all 3 class probabilities
for name, model in models.items():
    probs = model.predict_proba(X.values)
    probabilities[name] = probs  # Shape: (n_samples, 3)

# Average across ensemble
ensemble_probs = np.mean(list(probabilities.values()), axis=0)
# ensemble_probs[:, 0] = avg P(CRASH)
# ensemble_probs[:, 1] = avg P(NORMAL)
# ensemble_probs[:, 2] = avg P(SPIKE)

# Predict class with highest probability
pred_012 = np.argmax(ensemble_probs, axis=1)

# Apply confidence thresholds
crash_conf = ensemble_probs[:, 0]
spike_conf = ensemble_probs[:, 2]

# Override to NORMAL if confidence too low
low_confidence = (crash_conf < 0.30) & (spike_conf < 0.30)
pred_012[low_confidence] = 1  # NORMAL
```

**Pros:**
- Proper crash detection
- Can go to cash before crashes
- Enables short/hedge strategies
- More granular market view

**Cons:**
- Requires retraining all models
- More complex threshold tuning
- Class imbalance (crashes are rare ~5-10% of samples)
- Need separate thresholds for each class

### Option 3: Dual-Threshold Binary (Intermediate Solution)

**Use two thresholds on binary model to create 3 regions:**

```python
# Binary model outputs P(positive move)
ensemble_prob = np.mean(probabilities, axis=0)

# Three regions
pred_012 = np.where(
    ensemble_prob >= 0.60,  # High confidence positive
    2,                      # SPIKE
    np.where(
        ensemble_prob <= 0.20,  # High confidence negative
        0,                      # CRASH (inverted binary)
        1                       # NORMAL (uncertain)
    )
)
```

**Explanation:**
- High prob (>0.60): Strong upside → SPIKE
- Low prob (<0.20): Strong downside → CRASH (since model is "prob of up")
- Middle (0.20-0.60): Uncertain → NORMAL

**Pros:**
- No retraining required
- Can detect crashes immediately
- Simple to implement

**Cons:**
- Imprecise (assumes low P(up) = high P(down))
- Thresholds are arbitrary
- Not trained for 3-class

---

## 📈 Expected Class Distribution (3-Class)

Based on SPY historical data (2000-2025):

```
Market Regime Distribution:
- CRASH (0):  ~10-15% (major drops, bear markets)
- NORMAL (1): ~60-70% (sideways, small moves)
- SPIKE (2):  ~15-25% (strong rallies, bull runs)
```

**Threshold calibration for 3-class:**

```python
# Conservative (precision-focused)
spike_threshold = +1.5%  # Only clear rallies
crash_threshold = -1.5%  # Only clear drops

# Balanced
spike_threshold = +1.0%
crash_threshold = -1.0%

# Aggressive (recall-focused)
spike_threshold = +0.5%
crash_threshold = -0.5%
```

---

## 🔧 Implementation Plan

### Phase 1: Immediate Fix (Option 3 - Dual Threshold)

**Timeline:** 10 minutes
**Impact:** Can detect crashes today

```bash
# 1. Update predict_multi_asset_ensemble.py with dual threshold
# 2. Set thresholds: spike_thresh=0.60, crash_thresh=0.20
# 3. Regenerate predictions
python3 predict_multi_asset_ensemble.py

# 4. Verify crash predictions
python3 -c "
import pandas as pd
df = pd.read_csv('logs/daily_predictions.csv')
print(df['Prediction'].value_counts().sort_index())
"

# 5. Backtest with crash detection
python3 backtest.py
```

### Phase 2: Proper 3-Class Training (Option 2)

**Timeline:** 1-2 hours
**Impact:** Accurate crash detection

```bash
# 1. Implement add_three_class_labels() in utils.py
# 2. Update train_multi_asset.py to use 3-class labels
# 3. Retrain all models
python3 train_multi_asset.py

# 4. Update predict_multi_asset_ensemble.py for 3-class
# 5. Tune separate thresholds for CRASH, NORMAL, SPIKE
python3 threshold_tuning.py --num-classes 3

# 6. Regenerate predictions and backtest
python3 predict_multi_asset_ensemble.py
python3 backtest.py
```

### Phase 3: Validation

```bash
# Test on known crash periods
python3 backtest.py --start 2020-02-01 --end 2020-04-01  # COVID crash
python3 backtest.py --start 2008-09-01 --end 2009-03-01  # Financial crisis
python3 backtest.py --start 2022-01-01 --end 2022-10-01  # 2022 bear market

# Verify crash detection rate
# Expected: ~10-15% CRASH predictions in these periods
```

---

## 📚 Related Files

| File | Role | Needs Update? |
|------|------|---------------|
| **utils.py** | Label creation | ✅ Yes - add 3-class function |
| **train_multi_asset.py** | Model training | ✅ Yes - use 3-class labels |
| **predict_multi_asset_ensemble.py** | Prediction | ✅ Yes - handle 3 classes |
| **backtest.py** | Backtesting | ⚠️ Maybe - verify CRASH handling |
| **evaluate.py** | Evaluation metrics | ⚠️ Maybe - 3-class confusion matrix |
| **configs/best_thresholds.json** | Thresholds | ✅ Yes - add crash_thresh |

---

## 🎯 Recommendation

**Use Phase 1 (Dual Threshold) immediately for quick crash detection, then implement Phase 2 (3-class training) for accuracy.**

**Rationale:**
1. Phase 1 provides crash detection **today** (no retraining)
2. Protects capital during next correction
3. Phase 2 ensures proper long-term solution
4. Can compare both approaches and choose best

**Expected improvement:**

```
Current (Binary mapped to 3-class):
- CRASH:   0% (0/6501) ❌
- NORMAL: 65% (4210/6501)
- SPIKE:  35% (2291/6501)

Phase 1 (Dual threshold):
- CRASH:  ~15% (estimated)
- NORMAL: ~55%
- SPIKE:  ~30%

Phase 2 (True 3-class):
- CRASH:  ~12% (based on actual drops)
- NORMAL: ~63%
- SPIKE:  ~25%
```

---

## ⚠️ Important Notes

1. **Class imbalance:** Crashes are rare (~10%), may need class weights or SMOTE
2. **Threshold sensitivity:** Crash predictions very sensitive to threshold
3. **Trading costs:** Going to cash has opportunity cost
4. **False alarms:** Too many crash predictions = missed gains
5. **Backtesting:** Must test on 2008, 2020, 2022 crash periods

---

## 📖 Next Steps

1. **Decide on approach:** Quick fix (Phase 1) or proper 3-class (Phase 2)?
2. **Implement chosen solution**
3. **Validate on historical crashes**
4. **Update documentation** (README, ARCHITECTURE_GUIDE)
5. **Add crash detection metrics** to evaluation reports

---

*For questions or implementation help, see:*
- **ARCHITECTURE_GUIDE.md** - System overview
- **ACCURACY_OPTIMIZATION_GUIDE.md** - Threshold tuning
- **train_multi_asset.py** - Current training code
- **predict_multi_asset_ensemble.py** - Current prediction code
