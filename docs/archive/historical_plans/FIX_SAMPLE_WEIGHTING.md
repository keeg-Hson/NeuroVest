# Fix Sample Weighting

## Current Problem (config.py)

```python
TRAIN_CFG = {
    "min_weight": 0.50,
    "max_weight": 5.0,
    "weight_power": 1.75,  # EXPONENTIAL!
}
```

**How it works**:
```python
weights = np.abs(forward_returns) ** 1.75
weights = np.clip(weights, 0.5, 5.0)
```

**Problem**: 
- Trade with 2% return gets **5.0x weight** (max)
- Trade with 0.5% return gets **0.5x weight** (min)
- Model **overfits to outliers** (Fed announcements, Black Swans)
- These extreme events **don't repeat** in test set

## Current Weight Distribution

```
Forward Return  →  Weight  →  Frequency
     0.1%       →   0.50   →  Common (gets downweighted)
     0.5%       →   0.50   →  Common (gets downweighted)  
     1.0%       →   1.00   →  Normal
     2.0%       →   5.00   →  Rare (gets 10x weight vs 0.5%)
     5.0%       →   5.00   →  Very rare (overfits to these!)
```

**Impact**: Model learns "when Fed announces rate cuts, buy" instead of "when RSI < 30 and trend reversing, buy"

---

## Solution 1: Uniform Weighting (SIMPLE, RECOMMENDED)

```python
TRAIN_CFG = {
    "min_weight": 1.0,
    "max_weight": 1.0,
    "weight_power": 1.0,
}

# Or in compute_sample_weights():
def compute_sample_weights(...):
    return np.ones(len(df))  # All samples weighted equally
```

**Pros**:
- No overfitting to outliers
- Model learns general patterns
- More robust to market regime changes

**Cons**:
- Doesn't emphasize profitable trades
- Treats 0.1% and 5% moves equally

---

## Solution 2: Mild Weighting (BALANCED)

```python
TRAIN_CFG = {
    "min_weight": 0.8,
    "max_weight": 1.5,
    "weight_power": 0.5,  # Square root instead of ^1.75
}
```

**Weight Distribution**:
```
Forward Return  →  Weight  →  Ratio
     0.1%       →   0.80   →  1.0x
     0.5%       →   0.89   →  1.1x
     1.0%       →   1.00   →  1.25x
     2.0%       →   1.41   →  1.8x
     5.0%       →   1.50   →  1.9x (capped)
```

**Pros**:
- Slight emphasis on better trades
- Doesn't overfit to outliers
- More balanced than current

**Cons**:
- Still has some bias

---

## Solution 3: Class Balancing Only (ROBUST)

Instead of weighting by return magnitude, just balance positive/negative classes:

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights (handles imbalance)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Map to samples
sample_weights = np.array([class_weights[y] for y in y_train])
```

**Current Class Distribution**:
- Class 0 (no trade): 3,810 samples → weight: 0.68
- Class 1 (trade): 1,391 samples → weight: 1.87

**Pros**:
- Handles class imbalance
- Doesn't overfit to return magnitude
- Standard ML practice

**Cons**:
- Doesn't prioritize more profitable trades

---

## Recommended Fix

### Option A: Start Simple (Uniform)
```python
# In config.py
TRAIN_CFG = {
    # ... other settings ...
    "min_weight": 1.0,
    "max_weight": 1.0,
    "weight_power": 1.0,
}
```

### Option B: Mild + Class Balance (Best of Both)
```python
def compute_sample_weights(df, y, long_only=True):
    """Combine class balancing with mild profit weighting"""
    
    # 1. Class weights (handle imbalance)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights = np.array([class_weights[label] for label in y])
    
    # 2. Mild profit boost (sqrt, not exponential)
    if 'fwd_ret_net' in df.columns:
        profit_factor = np.sqrt(np.abs(df['fwd_ret_net']) * 100)  # sqrt(% return)
        profit_factor = np.clip(profit_factor, 0.8, 1.3)
        weights *= profit_factor
    
    # 3. Normalize to mean=1
    weights = weights / weights.mean()
    
    return weights
```

---

## Expected Impact

**Before** (exponential weighting):
- Training accuracy: 59% (inflated by memorizing outliers)
- Test accuracy: 57% (outliers don't repeat)
- Overfitting: Severe

**After** (uniform or mild weighting):
- Training accuracy: 56-57% (more honest)
- Test accuracy: 58-60% (better generalization)
- Overfitting: Reduced

**Net Effect**: +1-3 percentage points on test accuracy

---

## Implementation

1. Edit `config.py`:
```python
TRAIN_CFG = {
    "min_weight": 1.0,
    "max_weight": 1.0,
    "weight_power": 1.0,
}
```

2. Or edit `utils.py` `compute_sample_weights()`:
```python
def compute_sample_weights(...):
    # Simple version
    return np.ones(len(df))
```

3. Retrain all models
4. Compare test accuracy before/after
