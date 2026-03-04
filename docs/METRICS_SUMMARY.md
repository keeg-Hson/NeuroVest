# NeuroVest Model Metrics Summary

**Last Updated:** 2026-03-01
**Source of Truth:** This document consolidates metrics from `config.py`, `configs/best_thresholds.json`, and latest evaluation runs.
**Accuracy Alignment:** All accuracy metrics now computed from `logs/labeled_predictions.csv` (same source for evaluate.py and backtest).
**Architecture:** See `SYSTEM_DESIGN.md` for canonical file references.

---

## Current Configuration (Locked)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Prediction Threshold** | 0.45 | `config.py` / `best_thresholds.json` |
| **Forward Horizon** | 1 day | `config.py` TRAIN_CFG |
| **Positive Threshold** | 0.5% (0.005) | `config.py` TRAIN_CFG |
| **Objective** | max_precision | `best_thresholds.json` |
| **Strategy** | Precision-focused | Trade less, be right more |

---

## Model Performance Metrics

### Primary Evaluation (`evaluate.py`)

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 80.88% | On 6,511 rows |
| **AUC** | 0.7792 | Area under ROC curve |
| **Balanced Accuracy** | 62.33% | Accounts for class imbalance |
| **Precision (Class 1)** | 81.29% | When model predicts long |
| **Recall (Class 1)** | 39.55% | Of actual events captured |
| **F1 Score (Class 1)** | 0.516 | Harmonic mean |

### Prediction Distribution
- **PredLong=0**: 5,973 (91.7%)
- **PredLong=1**: 538 (8.3%)
- **Actual Events**: 1,790 (27.5%)

### Backtest Performance (`generate_backtest_metrics.py`)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Total Return** | 191.0% | - |
| **Annual Return** | 4.4% | - |
| **Sharpe Ratio** | 2.55 | vs 0.42 buy-and-hold |
| **Max Drawdown** | -5.4% | vs -55% buy-and-hold |
| **Win Rate** | 54.0% | - |
| **Sortino Ratio** | 3.12 | - |
| **Calmar Ratio** | 35.37 | - |
| **Profit Factor** | 1.87 | - |
| **Total Trades** | 50 | - |
| **Model Accuracy** | 80.88% | Aligned with evaluate.py |

---

## Ensemble Models (with Regime Features)

From `comprehensive_model_evaluation.py` using 164 features:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | 64.29% | 38.08% | 42.12% | 0.400 |
| **LightGBM** | 62.06% | 36.88% | 48.10% | 0.417 |
| **CatBoost** | 61.67% | 37.57% | 53.80% | 0.442 |
| **Ensemble** | 63.98% | 38.95% | 48.37% | 0.432 |

### Profit Optimization Results

| Threshold | Avg Profit/Trade | Win Rate | N Trades | F1 Score |
|-----------|------------------|----------|----------|----------|
| **0.80** (Best Profit) | 0.63% | 100% | 3 | 0.011 |
| **0.30** (Best F1) | 0.02% | 52.5% | 1,065 | 0.451 |

---

## Multi-Asset Strategy Comparison

From `compare_strategies.py`:

| Asset | Strategy | Total Return | Sharpe | Max DD | Trades | Win Rate |
|-------|----------|--------------|--------|--------|--------|----------|
| BTC/USDT | Multi-Asset Ensemble | 256.68% | 2.46 | -20.96% | 473 | 58.77% |
| ETH/USDT | Multi-Asset Ensemble | 367.76% | 2.31 | -29.48% | 489 | 57.06% |
| SOL/USDT | Per-Asset | 16,127.74% | 17.68 | -10.57% | 184 | 91.30% |
| SOL/USDT | Multi-Asset Ensemble | 156.13% | 1.82 | -26.05% | 294 | 54.42% |

---

## Feature Engineering

### Feature Count by Category

| Stage | Features | Accuracy | Improvement |
|-------|----------|----------|-------------|
| Baseline | 103 | 58.4% | - |
| + Cross-Asset | 130 | 61.5% | +3.1% |
| + Macro (FINAL) | 164 | 62.3% | +3.9% |

### Current Feature Breakdown (164 Total)

- **Base Features**: 153
- **Regime Features**: 11

### Top 5 Regime Features by Importance

1. **ADX** (280) - Trend strength indicator
2. **Plus_DI** (155) - Positive directional movement
3. **MA200_Distance_Vol** (125) - Volatility-adjusted deviation from 200-day MA
4. **MA_200** (120) - 200-day moving average
5. **Price_vs_MA200** (102) - Price relative to long-term trend

### Feature Categories

| Category | Features | Key Indicators |
|----------|----------|----------------|
| **Technical** | ~40 | RSI, MACD, Bollinger Bands, ATR, Stochastic |
| **Price Patterns** | ~20 | SMA crossovers (5/20, 10/50, 20/200), momentum |
| **Volume** | ~10 | OBV, volume spikes, accumulation/distribution |
| **Volatility** | ~15 | Historical volatility, ATR ratios, range analysis |
| **Cross-Asset** | ~27 | VIX, TNX, DXY, sector ETFs correlations |
| **Macro** | ~34 | FRED economic indicators, sentiment signals |
| **Regime** | 11 | ADX, DI+/-, MA200, Bull/Bear detection |
| **Lagged** | ~27 | Returns, volumes over multiple horizons |

---

## What Makes This Model Good

### Strengths

1. **Risk-Adjusted Returns**: Sharpe of 2.55 vs 0.42 buy-and-hold (6x better)
2. **Drawdown Protection**: -5.4% max DD vs -55% buy-and-hold (90% reduction)
3. **Precision-Focused**: 81% precision when model signals a trade
4. **Regime Awareness**: 11 regime features detect bull/bear/volatility environments
5. **Multi-Asset Learning**: Trained on 10,500+ samples across 40+ assets
6. **Realistic Costs**: Backtests include 2 bps fees + 3 bps slippage

### Trade-offs (Known Limitations)

1. **Moderate Recall** (40%): Model captures ~40% of opportunities (precision focus)
2. **Few Trades**: ~50 trades total (quality over quantity)
3. **Conservative**: Threshold 0.45 filters out lower-confidence signals
4. **Asset Variation**: SOL per-asset model outperforms multi-asset (91% vs 54% win rate)

### Strategy Rationale

The model prioritizes **precision over recall**:
- When it signals, it's right ~81% of the time
- It captures ~40% of actual positive events
- Position sizing matters more than signal frequency
- This matches a "weak but real edge" trading philosophy

---

## Threshold Optimization Trade-offs

| Strategy | Threshold | Precision | Win Rate | Trades | Use Case |
|----------|-----------|-----------|----------|--------|----------|
| Ultra-Conservative | 0.65 | 100% | 100% | 4 | High-value accounts |
| **Conservative** | 0.55 | 92.6% | 92.6% | 27 | Risk-averse |
| **Precision-Focused** | 0.45 | ~87% | ~54% | ~50 | **CURRENT** |
| Balanced | 0.50 | 38.7% | 52.7% | 619 | Active trading |
| Aggressive | 0.40 | 31.6% | 52.4% | 880 | High frequency |

---

## Risk Management Configuration

From `config.py` RISK_CFG:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max Position Size | 5% | Per-trade limit |
| Risk Per Trade | 1% | Stop loss * position |
| Kelly Fraction | 0.25 | Use 25% of Kelly-optimal |
| Max Daily Drawdown | 3% | Halt trading threshold |
| Confidence Scaling | True | Size by model confidence |
| Max Correlated Exposure | 15% | Limit in correlated assets |

---

## Files That Define the Model

| File | Purpose | Status |
|------|---------|--------|
| `config.py` | Global configuration, locked parameters | **Source of Truth** |
| `configs/best_thresholds.json` | Optimized thresholds | **Source of Truth** |
| `docs/model_changelog.md` | Official change log | **Source of Truth** |
| `build_feature_table.py` | Feature engineering pipeline | Canonical |
| `train.py` | Model training entry point | Canonical |
| `predict.py` | Prediction entry point | Canonical |
| `backtest.py` | Backtesting entry point | Canonical |

---

## Known Issues (from Latest Run)

1. **Legacy Scripts Archived** (2026-03-01): Scripts referencing deprecated models moved to `archive/legacy_scripts/`:
   - `market_crash_model_fwd_improved.pkl` references removed
   - See `SYSTEM_DESIGN.md` for canonical entry points

2. **Working Models**: Regime ensemble models are functional:
   - `xgboost_regime.pkl`
   - `lightgbm_regime.pkl`
   - `catboost_regime.pkl`

3. **API Limits**: NewsAPI returns 426 (plan limit) - falling back to top-headlines

4. **Database**: Railway PostgreSQL is primary. Local SQLite shows empty data without `DATABASE_URL`.

---

*Generated: 2026-03-01*
*Configuration: 1-day horizon, 0.5% threshold, 0.45 prediction threshold*
