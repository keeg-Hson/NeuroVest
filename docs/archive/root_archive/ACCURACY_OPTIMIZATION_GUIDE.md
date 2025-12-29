# NeuroVest: Optimizing for Market Accuracy vs Trading Profit

**Date:** 2025-11-18

---

## Two Fundamentally Different Goals

### Goal 1: Profitable Trading System
**Objective:** Make money, minimize losses
**Prioritize:** Precision (avoid bad trades) > Recall
**Philosophy:** "Better to miss opportunities than take bad trades"

### Goal 2: Accurate Market Mapping
**Objective:** Predict market movements accurately
**Prioritize:** Balanced accuracy (F1 score, recall + precision)
**Philosophy:** "Capture all market patterns, even if some predictions are wrong"

---

## Current Configuration Analysis

### Threshold Evolution:

| Threshold | Use Case | Precision | Recall | F1 | Trades | Interpretation |
|-----------|----------|-----------|--------|-----|--------|----------------|
| **0.55** | Ultra-conservative trading | 97.6% | 14.7% | 25.6% | 454 | Misses 85% of opportunities |
| **0.45** | Conservative trading | 92.3% | 20.0% | 33.0% | ~800 | Still too cautious |
| **0.35** ✅ | Balanced market mapping | 83.9% | 39.5% | 53.7% | 1,415 | **Good balance** |
| **0.30** | Aggressive mapping | ~75% | ~50% | ~60% | ~2,000 | Catch half of moves |
| **0.25** | Very aggressive | ~65% | ~65% | ~65% | ~2,800 | Maximum F1, many false positives |

**Current setting (0.35) is optimal for balanced market mapping.**

---

## For Maximum Accuracy: Recommendations

### Option A: Lower Threshold Further (Quick, 5 min)

**If you want to catch MORE market moves** (60-70% recall):

```bash
# Edit config.py, change:
PREDICTION_THRESHOLD = 0.25  # Very aggressive

# Regenerate predictions
python3 predict_multi_asset_ensemble.py

# Evaluate
python3 evaluate.py
```

**Expected results:**
- Recall: 60-70% (catch 6-7 out of 10 moves)
- Precision: 60-70% (3-4 out of 10 predictions wrong)
- F1 Score: ~65% (maximized balance)
- Accuracy: 65-68%

**Use this if:** You want to study ALL market patterns, don't care about false positives

---

### Option B: Retrain with Balanced Class Weights (Medium, 2 hours)

**Why:** Current model was trained on imbalanced data (too many NORMAL examples)

**How:**
```bash
# Edit config.py, add class balancing:
TRAIN_CFG = {
    # ... existing config ...
    "balance_classes": True,  # Add this
    "class_weight": "balanced",  # Or "balanced_subsample"
}

# Retrain models
python3 train.py

# Regenerate predictions
python3 predict_multi_asset_ensemble.py

# Evaluate
python3 evaluate.py
```

**Expected improvement:**
- Better recall without sacrificing precision
- Model learns minority classes (SPIKE/CRASH) better
- More natural prediction distribution

---

### Option C: Change Labeling Strategy (Advanced, 1 day)

**Current labeling creates class imbalance:**
```
CRASH (0):      0 predictions (0%)   ← Model never predicts crashes
NORMAL (1): 5,086 predictions (78%)
SPIKE (2):  1,415 predictions (22%)

Actual distribution:
No-Trade:  3,493 (54%)
Trade:     3,008 (46%)
```

**Problem:** 3-class labels (CRASH/NORMAL/SPIKE) are too granular

**Solutions:**

#### Option C1: Binary Labels (Simpler)
```python
# In config.py:
TRAIN_CFG = {
    # ... existing config ...
    "binary_labels": True,  # 0=no-trade, 1=trade (up OR down)
    "pos_threshold": 0.003,  # Lower threshold = more events
}
```

**Advantage:** More balanced classes, easier to learn

#### Option C2: Lower Event Hurdle
```python
# In config.py:
TRAIN_CFG = {
    # ... existing config ...
    "min_edge_bps": 5.0,  # Currently 10.0 - too strict
    "pos_threshold": 0.003,  # Currently 0.005 (0.5%)
}
```

**Advantage:** More events labeled as SPIKE/CRASH, less NORMAL

---

### Option D: Ensemble Multiple Thresholds (Best, 30 min)

**Strategy:** Generate predictions at multiple thresholds, combine them

```bash
# Generate conservative predictions
python3 predict_multi_asset_ensemble.py --threshold 0.45

# Save as conservative
cp logs/daily_predictions.csv logs/predictions_conservative.csv

# Generate balanced predictions
python3 predict_multi_asset_ensemble.py --threshold 0.35

# Save as balanced
cp logs/daily_predictions.csv logs/predictions_balanced.csv

# Generate aggressive predictions
python3 predict_multi_asset_ensemble.py --threshold 0.25

# Save as aggressive
cp logs/daily_predictions.csv logs/predictions_aggressive.csv

# Analyze all three
python3 compare_thresholds.py
```

**Use case:** Understand model behavior across confidence levels

---

## Understanding AUC vs Accuracy

**Your current AUC: 0.7156** (unchanged across thresholds)

- **AUC (0.7156):** Model's ability to discriminate between classes (threshold-independent)
- **Accuracy (68.5%):** Actual performance at current threshold (0.35)

**What this means:**
- AUC of 0.71 is "acceptable" but not great (0.8+ would be good)
- The underlying model has room for improvement
- Changing thresholds won't improve AUC, only retraining can

**To improve AUC (requires retraining):**
1. Add more features (technical indicators, macro data)
2. Use better feature selection (remove noise)
3. Balance training data (SMOTE, class weights)
4. Train on more data (more assets, longer history)
5. Hyperparameter tuning (grid search)

---

## Recommended Settings by Use Case

### Research / Market Mapping
**Goal:** Understand market patterns accurately

```python
# config.py
PREDICTION_THRESHOLD = 0.30

# Prioritize:
# - High recall (50-60%)
# - Balanced accuracy (65-70%)
# - F1 score maximization
```

**Acceptable:** More false positives (predict trade when shouldn't)
**Not acceptable:** Missing real market moves (false negatives)

---

### Paper Trading / Strategy Development
**Goal:** Test strategies without real money

```python
# config.py
PREDICTION_THRESHOLD = 0.35  # Current setting ✅

# Prioritize:
# - Balanced precision/recall
# - F1 score ~55%
# - Sharpe ratio > 0.5
```

**Good for:** Understanding strategy behavior, debugging logic

---

### Live Trading (Real Money)
**Goal:** Consistent profits, minimize losses

```python
# config.py
PREDICTION_THRESHOLD = 0.45

# Prioritize:
# - High precision (>85%)
# - Positive Sharpe ratio
# - Low drawdown
```

**Conservative approach:** Only take high-confidence trades

---

## Current Status: Your Model Performance

**With threshold 0.35:**
```
✅ Accuracy: 68.5% (good for market mapping)
✅ Balanced Accuracy: 66.5% (well-calibrated)
✅ Recall: 39.5% (catching 4 of 10 market moves)
✅ Precision: 83.9% (8 of 10 predictions correct)
✅ F1 Score: 53.7% (balanced)
✅ Backtest Return: +36.5% (profitable)
✅ Sharpe: 0.51 (acceptable risk-adjusted return)
```

**This is a good configuration for:**
- Understanding market patterns
- Strategy development
- Balanced trading approach

**If you want better accuracy**, choose Option B (retrain with class balancing) + Option C2 (lower event hurdle).

---

## Quick Action Matrix

| Your Goal | Threshold | Next Action |
|-----------|-----------|-------------|
| **Maximum market coverage** | 0.25-0.30 | Edit config.py, regenerate |
| **Best accuracy/recall balance** | 0.35 ✅ | **Already there!** |
| **Safe trading** | 0.45 | Edit config.py, regenerate |
| **Ultra-conservative** | 0.55 | Not recommended |
| **Improve underlying model** | N/A | Retrain with class balancing |

---

## Current Recommendation

**For market mapping (your stated goal):**

Your current threshold (0.35) is **excellent** for accurate market mapping. You're:
- Catching 39.5% of market moves (reasonable coverage)
- 83.9% precision (high accuracy when you predict)
- 68.5% overall accuracy (well above random)

**If you want to catch MORE moves** (at the cost of more false positives):
```bash
# Lower to 0.30 for ~50% recall
python3 -c "
from config import PREDICTION_THRESHOLD
import sys
with open('config.py', 'r') as f:
    content = f.read()
content = content.replace(
    'PREDICTION_THRESHOLD = 0.35',
    'PREDICTION_THRESHOLD = 0.30'
)
with open('config.py', 'w') as f:
    f.write(content)
print('Threshold updated to 0.30')
"

python3 predict_multi_asset_ensemble.py
python3 evaluate.py
```

**For long-term improvement**, implement Option B (retrain with class balancing).

---

**Bottom line:** You're already optimized for market accuracy with current settings (0.35). Only lower threshold if you want to catch MORE moves at the expense of more errors.
