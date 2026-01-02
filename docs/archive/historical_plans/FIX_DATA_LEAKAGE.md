# Data Leakage Fixes

## CRITICAL: Cumulative Features Using Future Data

### Current Broken Code (utils.py lines 537, 569-570)

```python
# WRONG: Uses entire dataset's cumulative sum
direction = np.sign(price_diff).fillna(0.0)
d["OBV"] = (direction * d["Volume"].fillna(0.0)).cumsum()

# WRONG: Cumulative sum uses all future data
cum_vol = d["Volume"].replace(0, np.nan).cumsum()
d["VWAP"] = (typical_price * d["Volume"]).cumsum() / cum_vol
```

**Problem**: When calculating features for day 100, `.cumsum()` includes data from days 101-3927!

### Fixed Code (Rolling Window Approach)

```python
# CORRECT: Use rolling window for OBV
def calculate_obv_rolling(close, volume, window=252):
    """Calculate OBV using only past data within a rolling window"""
    price_diff = close.diff()
    direction = np.sign(price_diff).fillna(0.0)
    obv_incremental = direction * volume.fillna(0.0)
    # Rolling sum instead of cumsum
    return obv_incremental.rolling(window=window, min_periods=20).sum()

d["OBV"] = calculate_obv_rolling(d["Close"], d["Volume"])

# CORRECT: Remove VWAP entirely (it's a intraday indicator anyway)
# If needed, use rolling VWAP:
def calculate_vwap_rolling(high, low, close, volume, window=20):
    """Calculate VWAP using only past data"""
    typical_price = (high + low + close) / 3.0
    rolling_vol = volume.rolling(window=window, min_periods=1).sum()
    rolling_vwap = (typical_price * volume).rolling(window=window, min_periods=1).sum()
    return rolling_vwap / rolling_vol.replace(0, np.nan)

# Better: Just remove VWAP from features list
```

## Other Potential Leakage Issues

### 1. Percentile Calculations

```python
# WRONG: Uses entire dataset
d["Vol_Percentile_252"] = d["Volatility"].rank(pct=True)

# CORRECT: Use expanding window
d["Vol_Percentile_252"] = d["Volatility"].expanding(min_periods=252).apply(
    lambda x: pd.Series(x[:-1]).rank(pct=True).iloc[-1] if len(x) > 1 else 0.5
)
```

### 2. Z-Score Calculations

```python
# Verify these use rolling(), not global mean/std:
# GOOD: Uses rolling window (no leakage)
roll_mom_mean = d["Price_Momentum_10"].rolling(60).mean()
roll_mom_std = d["Price_Momentum_10"].rolling(60).std()
d["ZMomentum"] = (d["Price_Momentum_10"] - roll_mom_mean) / roll_mom_std
```

## Implementation Checklist

- [ ] Replace `cumsum()` with `rolling().sum()` for OBV
- [ ] Remove VWAP or use rolling VWAP
- [ ] Verify all `.rank(pct=True)` use expanding windows
- [ ] Verify all percentiles use expanding windows
- [ ] Add `ensure_no_future_leakage()` validation function

## Expected Impact
- **Training accuracy**: Will DROP (good - means it was inflated)
- **Test accuracy**: Will IMPROVE (better generalization)
- **Realistic performance**: More honest evaluation
