# Feature Reduction Plan

## Current State
- **106 features** on **5,201 training samples**
- Ratio: 49 samples per feature (SEVERE overfitting)
- Industry best practice: 100-200 samples per feature

## Target State
- **15-20 core features** maximum
- Ratio: 260-347 samples per feature (acceptable)

## Features to KEEP (15 core features)

### Price Action (5 features)
1. `Return_Lag1` - Yesterday's return (momentum)
2. `Daily_Return` - Today's return
3. `Price_Momentum_10` - 10-day momentum
4. `MA_20` - 20-day moving average
5. `MA_200` - Long-term trend

### Volatility (3 features)
6. `Volatility` - 20-day volatility
7. `ATR_14` - Average True Range
8. `BB_Width` - Bollinger Band width

### Volume (2 features)
9. `Vol_Ratio` - Volume vs 20-day average
10. `OBV` - On-Balance Volume (trend confirmation)

### Technical Indicators (3 features)
11. `RSI` - Relative Strength Index
12. `MACD` - Moving Average Convergence Divergence
13. `Stoch_K` - Stochastic oscillator

### Market Regime (2 features)
14. `Bull_Market` - Above/below 200-day MA
15. `High_Volatility` - Volatility regime flag

## Features to REMOVE (91 features)

### Remove ALL Lag Duplicates
- ❌ `Return_Lag3`, `Return_Lag5`, `Return_Lag7`, `Return_Lag10`, `Return_Lag15`
- ❌ `RSI_Lag_1`, `RSI_Lag_3`, `RSI_Lag_5`, `RSI_Lag7`, `RSI_Lag10`
- ❌ `BB_Width_Lag1`, `BB_Width_Lag3`
- **Why**: These are redundant with Lag1 + autocorrelation

### Remove ALL Interaction Features
- ❌ `BB_Width_x_RSI`, `BB_Width_x_Return_Lag1`, `Return_Lag1_x_Return_Lag3`
- ❌ `RSI_x_Vol_Ratio`, `OBV_x_Return_Lag1`, `MACD_x_RSI`
- **Why**: Interaction features memorize training noise

### Remove Redundant Indicators
- ❌ `EMA_12`, `EMA_26` (keep `MACD` which uses these)
- ❌ `MACD_Signal`, `MACD_Histogram` (keep just `MACD`)
- ❌ `Stoch_D` (keep just `Stoch_K`)
- ❌ `RSI_Delta` (redundant with `RSI`)

### Remove Statistical Features
- ❌ `Ret_Skew_20`, `Ret_Kurt_20`, `Ret_Skew_10`, `Ret_Kurt_10`
- **Why**: Higher moments are noisy on small samples

### Remove Engineered Features
- ❌ All "enhanced" features from feature importance analysis
- ❌ `Return_Trend_Strength`, `Return_Momentum_Ratio`, `Return_Acceleration`
- ❌ `Vol_Expanding`, `Vol_Percentile`, `Volatility_Acceleration`
- **Why**: Created from overfitting analysis, not fundamental patterns

## Implementation

```python
# In utils.py, replace get_feature_list() with:
def get_feature_list():
    return [
        # Price Action (5)
        "Return_Lag1",
        "Daily_Return", 
        "Price_Momentum_10",
        "MA_20",
        "MA_200",
        # Volatility (3)
        "Volatility",
        "ATR_14",
        "BB_Width",
        # Volume (2)
        "Vol_Ratio",
        "OBV",
        # Technical (3)
        "RSI",
        "MACD",
        "Stoch_K",
        # Regime (2)
        "Bull_Market",
        "High_Volatility"
    ]
```

## Expected Impact
- **Training time**: 5-10x faster
- **Overfitting**: Significantly reduced
- **Test accuracy**: Should improve 2-5 percentage points
- **Generalization**: Much better out-of-sample performance
