# Market Regime Detection Features

## Overview

Market regime features help the model identify and adapt to different market conditions (bull markets, bear markets, high volatility periods, trending vs sideways markets). These features were added to improve model accuracy from **56.46% to 58.38%** (+3.41%).

## Why Regime Features?

Financial markets behave differently in different "regimes":
- **Bull markets**: Stocks generally rise, buy signals work better
- **Bear markets**: Stocks generally fall, different patterns emerge
- **High volatility**: Larger price swings, more uncertainty
- **Strong trends**: Momentum strategies work well
- **Sideways markets**: Mean reversion strategies work better

By detecting these regimes, the model can adjust its predictions based on market context.

## Feature Categories

### 1. Bull/Bear Market Detection (200-day MA)

These features use the 200-day moving average, a classic indicator of long-term market direction.

| Feature | Description | Impact |
|---------|-------------|--------|
| `MA_200` | 200-day moving average | Moderate (120) |
| `Price_vs_MA200` | Current price / MA200 ratio | High (102) |
| `Bull_Market` | Binary: 1 if price > MA200 | Low (1) |
| `MA200_Slope` | Rate of change in MA200 | Moderate (85) |
| `MA200_Distance_Vol` | Distance from MA200 normalized by volatility | **Very High (125)** |

**Key Insight**: `MA200_Distance_Vol` is the 3rd most important regime feature - it measures how far price has deviated from the long-term trend, adjusted for current volatility.

### 2. Volatility-Based Features

Detect high-fear/high-volatility regimes when markets are unstable.

| Feature | Description | Impact |
|---------|-------------|--------|
| `Vol_Percentile_252` | Volatility percentile rank (1 year) | - |
| `High_Volatility` | Binary: 1 if volatility > 75th percentile | Low (6) |

**Note**: VIX-based features (`VIX_Percentile`, `High_Fear`, `VIX_Spike`) are supported but require VIX data.

### 3. Market Breadth Indicators

Measure how widespread a market move is.

| Feature | Description | Impact |
|---------|-------------|--------|
| `Near_52w_High` | Binary: 1 if within 2% of 52-week high | Moderate (24) |
| `Near_52w_Low` | Binary: 1 if within 2% of 52-week low | None (0) |
| `Pct_Above_MA20` | Binary: 1 if price > 20-day MA | None (0) |
| `MA_20_50_Cross` | Binary: 1 if MA20 > MA50 (golden cross) | None (0) |

**Key Insight**: Being near 52-week highs matters (24), but near lows doesn't (0) - suggests bull momentum is predictive.

### 4. Trend Strength (ADX)

The Average Directional Index (ADX) measures how strong a trend is, regardless of direction.

| Feature | Description | Impact |
|---------|-------------|--------|
| `ADX` | Average Directional Index (trend strength) | **Highest (280)** |
| `Plus_DI` | Positive directional indicator | **2nd Highest (155)** |
| `Minus_DI` | Negative directional indicator | High (86) |
| `Strong_Trend` | Binary: 1 if ADX > 25 | Very Low (2) |
| `Days_Above_MA20` | Days above 20-day MA (rolling 20d) | Moderate (26) |
| `Trend_Consistency` | Fraction of days above MA20 | Low (9) |

**Key Insight**: `ADX` is the **single most important regime feature**, contributing 280 importance points. The directional indicators (`Plus_DI`, `Minus_DI`) are also critical.

### 5. Composite Score

| Feature | Description | Impact |
|---------|-------------|--------|
| `Regime_Score` | Average of Bull_Market, Low_Fear, Strong_Trend | Low (19) |

A composite indicator combining multiple regime signals.

## Performance Impact

From `regime_features_impact.csv`:

| Metric | Baseline (78 features) | With Regime (103 features) | Improvement |
|--------|----------------------|---------------------------|-------------|
| **Accuracy** | 56.46% | 58.38% | **+3.41%** |
| **F1 Score** | 45.68% | 45.85% | **+0.36%** |
| **Precision** | 36.39% | 37.48% | +2.99% |
| **Recall** | 61.34% | 59.02% | -3.78% |

**Regime features contribute 12.54% of total model importance** despite being only 16.5% of features (17 out of 103).

## Top 5 Regime Features by Importance

1. **ADX** (280) - Trend strength is the #1 regime signal
2. **Plus_DI** (155) - Positive directional movement
3. **MA200_Distance_Vol** (125) - Volatility-adjusted deviation from trend
4. **MA_200** (120) - Long-term trend baseline
5. **Price_vs_MA200** (102) - Relative position to long-term trend

## Usage

The regime features are automatically added when you call `add_features()`:

```python
from utils import load_SPY_data, add_features

# Load data
df = load_SPY_data()

# Add all features including regime detection
df, feature_cols = add_features(df)

# Regime features are now in the dataframe
regime_features = [col for col in df.columns if any(x in col for x in
    ['MA_200', 'Bull_Market', 'ADX', 'Plus_DI', 'Minus_DI',
     'High_Volatility', 'Regime_Score'])]
```

## Models Using Regime Features

- **`models/market_crash_model_lightgbm_regime.pkl`**: LightGBM with full regime features (58.38% accuracy)

## Recommendation

**Use regime-enhanced models** for better market adaptability. The modest performance improvement (+3.41% accuracy) combined with strong feature importance (12.54%) suggests these features add valuable market context without overfitting.

---

**Added**: 2025-11-14
**Tested**: `test_regime_features.py`
**Impact Analysis**: `regime_features_impact.csv`
