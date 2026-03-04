# NeuroVest Optimization Guide

## Bugs Fixed

### 1. Random Splitting / Lookahead Bias Bug (CRITICAL)

**Location:** `utils.py` - `add_forward_returns_and_labels()` and `compute_sample_weights()`

**Problem:**
The volatility-adjusted threshold calculation used `d["Volatility"].median()` which calculates the median across the **entire dataset**, including future data. This is a subtle but significant form of lookahead bias.

```python
# BEFORE (BUG):
median_vol = d["Volatility"].median()  # Uses ALL data including future!
vol_ratio = d["Volatility"] / median_vol
```

**Fix:**
Changed to use **expanding window median** which only uses data available at each point in time:

```python
# AFTER (FIXED):
expanding_median_vol = vol_series.expanding(min_periods=60).median()
vol_ratio = vol_series / expanding_median_vol  # Only uses past data
```

**Impact:**
- More realistic backtest results (no peeking at future volatility)
- Better out-of-sample performance
- Proper time-series cross-validation integrity

---

## Optimization Strategies for Optimal Performance

### 1. Regime-Specific Model Training

**New Feature:** `core/regime_backtest.py`

Train separate models or adjust parameters for different market regimes:

```python
from core.regime_backtest import RegimeBacktester, run_regime_analysis

# Analyze performance across regimes
rb = RegimeBacktester(price_df, predictions_df)
rb.detect_regimes()
rb.backtest_by_regime('volatility')  # Test high/low vol separately
rb.backtest_by_regime('trend')       # Test bull/bear markets
report = rb.generate_report('logs/regime_analysis.json')
```

**Key Regime Types:**
- **Trend:** Bull / Bear / Sideways
- **Volatility:** Low / Medium / High
- **Risk Appetite:** Risk-On / Risk-Off
- **Market Phase:** Expansion / Peak / Contraction / Recovery

**Usage:**
```bash
python -m core.regime_backtest --regime-type all
```

### 2. Walk-Forward Optimization

Instead of single train/test split, use rolling windows:

```python
from core.model_improvements import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    n_splits=10,
    train_window=504,  # 2 years
    test_window=63,    # 1 quarter
    purge_gap=5,       # 5 day embargo
)
validator = WalkForwardValidator(config)
results = validator.validate(model, X, y)
```

### 3. Feature Selection Improvements

Current feature count may be too high. Consider:

1. **Aggressive correlation filtering:** Reduce from 0.85 to 0.75
2. **SHAP-based pruning:** Keep only features with mean |SHAP| > threshold
3. **RFE with cross-validation:** Recursive elimination with CV stability check

```python
from core.feature_selection import FeatureSelector, FeatureSelectionConfig

config = FeatureSelectionConfig(
    correlation_threshold=0.75,  # More aggressive
    min_features=30,             # Tighter bound
    max_features=50,
    shap_sample_size=2000,
    use_shap=True,
    use_rfe=True,
)
selector = FeatureSelector(config)
X_selected = selector.fit_transform(X, y)
```

### 4. Regime-Adaptive Position Sizing

Adjust position sizes based on detected regime:

```python
def get_regime_position_multiplier(current_regime):
    """Scale positions based on market regime."""
    multipliers = {
        ('bull', 'low'): 1.3,      # Bull market, low vol - increase size
        ('bull', 'high'): 0.8,     # Bull market, high vol - reduce size
        ('bear', 'low'): 0.7,      # Bear market, low vol - cautious
        ('bear', 'high'): 0.5,     # Bear market, high vol - very cautious
        ('sideways', 'medium'): 1.0,
    }
    return multipliers.get(
        (current_regime['trend'], current_regime['volatility']),
        1.0
    )
```

### 5. Ensemble Diversity

Current ensemble uses XGBoost, LightGBM, CatBoost. Consider adding:

1. **Linear models:** Ridge/Lasso for interpretability baseline
2. **Neural networks:** Small MLP for non-linear interactions
3. **Different feature subsets:** Train each model on different features

```python
from core.model_improvements import EnhancedEnsemble, EnsembleConfig

config = EnsembleConfig(
    n_models=5,
    model_types=['xgboost', 'lightgbm', 'catboost', 'ridge', 'mlp'],
    use_calibration=True,
    aggregation='stacked',  # Use meta-learner instead of weighted avg
)
```

### 6. Threshold Optimization by Regime

Different regimes may require different decision thresholds:

```python
regime_thresholds = {
    'bull_low_vol': 0.45,   # More aggressive in calm bull markets
    'bull_high_vol': 0.55,  # More conservative in volatile bull
    'bear_low_vol': 0.60,   # Very conservative in bear markets
    'bear_high_vol': 0.65,  # Most conservative
}
```

### 7. Dynamic Feature Importance

Track feature importance over time to detect concept drift:

```python
from core.model_drift import DriftDetector

detector = DriftDetector()
detector.set_baseline(X_train, y_train, model_probs)

# On new data
drift_detected, drift_metrics = detector.check_drift(X_new, y_new, new_probs)
if drift_detected:
    # Trigger retraining
    pass
```

### 8. Cost-Sensitive Learning

Adjust class weights based on actual trading costs and expected profits:

```python
# Weight positive class by expected profit, negative by expected loss
expected_profit_per_trade = 0.015  # 1.5%
expected_loss_per_trade = 0.010    # 1%
total_cost_per_trade = 0.0035      # fees + slippage

# Asymmetric weights
pos_weight = expected_profit_per_trade - total_cost_per_trade
neg_weight = expected_loss_per_trade + total_cost_per_trade
scale_pos_weight = neg_weight / pos_weight
```

### 9. Temporal Feature Engineering

Add more time-aware features:

```python
# Rolling quantiles (regime detection)
d['price_quantile_252'] = d['Close'].rolling(252).rank(pct=True)

# Momentum decomposition
d['short_mom'] = d['Close'].pct_change(5)
d['medium_mom'] = d['Close'].pct_change(21)
d['long_mom'] = d['Close'].pct_change(63)
d['mom_divergence'] = d['short_mom'] - d['medium_mom']
```

### 10. Backtesting Best Practices

1. **Use regime-specific backtests** to understand model weaknesses
2. **Run Monte Carlo simulations** to test robustness
3. **Test on multiple assets** (SPY, QQQ, sector ETFs)
4. **Walk-forward validation** instead of simple train/test

```bash
# Run regime analysis
python -m core.regime_backtest

# Run Monte Carlo
python backtest.py --monte-carlo --n-simulations 1000

# Multi-asset comparison
python backtest.py --asset-group equity --compare
```

---

## Priority Action Items

### High Priority (Do First)

1. **Retrain models** after lookahead bias fix
2. **Run regime analysis** to identify weak regimes
3. **Implement regime-adaptive thresholds**

### Medium Priority

4. Add more aggressive feature selection
5. Implement walk-forward optimization
6. Add Monte Carlo simulation for strategy robustness

### Lower Priority (Future)

7. Neural network addition to ensemble
8. Real-time regime detection API
9. Automated retraining pipeline

---

## Monitoring

After implementing optimizations, track:

1. **Sharpe ratio** by regime
2. **Maximum drawdown** by regime
3. **Win rate stability** over rolling windows
4. **Feature importance drift** over time

```python
# Example monitoring
from core.model_drift import monitor_model

drift_report = monitor_model(
    model_path='models/market_crash_model_fwd.pkl',
    new_data=recent_df,
    baseline_path='models/baseline_metrics.json',
)
```
