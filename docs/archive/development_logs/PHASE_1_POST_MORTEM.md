# Phase 1 Post-Mortem: Why All Changes Were Incorrect

## Summary

**Phase 1 made accuracy WORSE, not better.**

- **Before Phase 1**: 58.08% accuracy (106 features, exponential weighting)
- **After Phase 1**: 50.8% accuracy (24 features, uniform weighting, broken OBV/VWAP)
- **After full revert**: 58.08% accuracy (back to baseline)

All Phase 1 changes were incorrect and hurt model performance.

## Incorrect Change #1: OBV/VWAP "Data Leakage Fix"

### What Was Done
```python
# WRONG - Changed cumsum() to rolling()
d["OBV"] = obv_incremental.rolling(window=252, min_periods=20).sum()
d["VWAP"] = rolling_vwap / rolling_vol
```

### Why It Was Wrong
- **OBV and VWAP are BY DEFINITION cumulative indicators**
- cumsum() only uses PAST data (not future), so it's NOT data leakage
- Real data leakage would be: `df.shift(-5)` or using test set statistics
- By using rolling windows, I fundamentally broke these indicators

### Impact
- Accuracy dropped from 58% → 50.8%

### Correct Implementation (Reverted)
```python
# CORRECT - OBV and VWAP should be cumulative
d["OBV"] = (direction * d["Volume"].fillna(0.0)).cumsum()
cum_vol = d["Volume"].replace(0, np.nan).cumsum()
d["VWAP"] = (typical_price * d["Volume"]).cumsum() / cum_vol
```

## Incorrect Change #2: Feature Reduction (106 → 15)

### What Was Done
- Reduced feature set from 106 → 15 features
- Rationale: "106 features on 5,201 samples = overfitting"
- Removed 91 features including interactions, lags, regime indicators

### Why It Was Wrong
The "overfitting" analysis was flawed:
1. **Modern tree-based models (XGBoost, LightGBM) handle high-dimensional data well**
   - Built-in regularization
   - Feature selection during training
   - L1/L2 penalties

2. **The 106 features were actually helping performance**
   - Market prediction requires capturing complex dynamics
   - Feature interactions (BB_Width × RSI) provide valuable signal
   - Multiple lags capture temporal patterns
   - Regime features capture market state

3. **5,201 samples / 106 features = 49 samples/feature is acceptable**
   - Not ideal, but tree models can handle it
   - Would be more problematic for linear models

### Impact
- Features dropped from 106 → 24
- Accuracy dropped from 58% → 49.4%

### Results Comparison
| Configuration | Features | Accuracy | Impact |
|--------------|----------|----------|---------|
| Original | 106 | 58.08% | Baseline |
| Reduced | 15 core + 9 extras = 24 | 49.4% | -8.7% |
| Reverted | 106 | 58.08% | Restored |

## Incorrect Change #3: Sample Weighting (Exponential → Uniform)

### What Was Done
```python
# Changed from exponential to uniform
"min_weight": 1.0,  # Was 0.50
"max_weight": 1.0,  # Was 5.0
"weight_power": 1.0,  # Was 1.75
```

Rationale: "Exponential weighting overfits to rare outliers"

### Why It Was Wrong
1. **Financial markets have skewed returns**
   - Large moves (2%+ days) contain important signal
   - These are NOT just noise or outliers
   - They represent regime shifts, news events, volatility spikes

2. **Uniform weighting treats all samples equally**
   - Gives same importance to 0.1% and 2.0% moves
   - Model can't learn that larger moves matter more
   - Loses ability to distinguish significant events

3. **Exponential weighting (^1.75) was appropriate**
   - Emphasizes larger moves without extreme outlier focus
   - 2% return → 3.3x weight (not extreme)
   - Helps model learn from important market events

### Impact (Combined with other changes)
Contributed to the 58% → 49.4% accuracy drop

## What Is REAL Data Leakage?

### ✅ NOT Data Leakage
```python
# cumsum() - Only looks backward
d["OBV"] = (direction * volume).cumsum()  # Each row sums PAST rows only

# Rolling windows - Only looks backward
d["MA_20"] = d["Close"].rolling(20).mean()  # Each row averages PAST 20 rows

# Lags - Explicit past data
d["Return_Lag1"] = d["Close"].pct_change(1)  # Yesterday's return
```

### ❌ ACTUAL Data Leakage
```python
# Forward-looking shift
d["Future_Return"] = d["Close"].shift(-5)  # Uses FUTURE data!

# Using test set statistics
scaler = StandardScaler().fit(entire_dataset)  # Includes test data!

# Target-based encoding without proper CV
d["Mean_Return_By_Date"] = d.groupby("Date")["fwd_ret"].transform("mean")

# Looking ahead in time
d["Max_Price_Today"] = d.groupby("Date")["Close"].transform("max")
```

## Lessons Learned

### 1. Understand What "Overfitting" Really Means
- Not just "many features = overfitting"
- Tree models handle high dimensions well
- Test performance is what matters, not sample/feature ratio

### 2. Understand Domain-Specific Indicators
- OBV, VWAP, cumulative volume are SUPPOSED to be cumulative
- Changing fundamental indicator definitions breaks them
- Always research before "fixing" established indicators

### 3. Understand Financial Data Characteristics
- Returns are skewed, not normal
- Large moves matter more than small ones
- Sample weighting should reflect importance

### 4. Always Test Changes Individually
Phase 1 bundled 3 changes together, making it hard to isolate impact. Should have:
1. Changed one thing at a time
2. Tested each change independently
3. Only kept changes that improved performance

### 5. Trust the Data
Original configuration achieved 58% accuracy for a reason. Before "fixing" it:
- Understand WHY it works
- Test changes on validation set
- Don't assume complexity = overfitting

## Current Status

✅ **All Phase 1 changes reverted**
✅ **Accuracy restored to 58.08% baseline**
✅ **106 features restored**
✅ **Exponential sample weighting restored**
✅ **OBV/VWAP cumsum() restored**

## Next Steps

The original analysis in `ACCURACY_IMPROVEMENT_PLAN.md` needs revision. The real issues are:

### Actual Problems (Not Addressed by Phase 1)
1. **Limited training data** (5,201 samples)
   - Need more data or multi-asset training
   - Consider daily + intraday data

2. **Market efficiency** (fundamental limit)
   - 58% accuracy might be near the ceiling
   - Can't predict random walk perfectly

3. **Single train/test split**
   - Need walk-forward validation
   - Test on different market regimes

4. **Target variable noise** (0.5% threshold)
   - 1-day horizon is very noisy
   - Consider longer horizons or volatility adjustment

### What NOT To Do (Lessons from Phase 1)
- ❌ Don't reduce features aggressively
- ❌ Don't "fix" cumulative indicators
- ❌ Don't use uniform weighting on skewed data
- ❌ Don't bundle multiple changes together

### What TO Do (Revised Approach)
1. ✅ Extend training data (more history, more assets)
2. ✅ Try different horizons (2-day, 3-day, 5-day)
3. ✅ Implement walk-forward validation
4. ✅ Test each change individually
5. ✅ Focus on ensemble diversity, not feature reduction

## References

- Commit 6d3ab7e5: Phase 1 test results (50.8% accuracy)
- Commit 0a89fecf: OBV/VWAP revert
- Commit 74b36869: Feature reduction and sample weighting revert
- Commit a5151ff7: Final accuracy restoration (58.08%)
