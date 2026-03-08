# NeuroVest

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)
![Platform](https://img.shields.io/badge/Deployed-Railway-blueviolet?logo=railway)
![Models](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-informational)
![Accuracy](https://img.shields.io/badge/Accuracy-80.88%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC-0.7792-brightgreen)

**Economic Forecasting Platform**

**Live Dashboard:** [neurovestdemo.up.railway.app](https://neurovestdemo.up.railway.app)

---

## Overview

NeuroVest is a market analysis and economic forecasting platform built around an ensemble machine learning engine. It analyzes live and historical market data to detect regime shifts, spikes, and drawdown risk using a combination of technical, sentiment, and macroeconomic indicators.

The system runs on Railway with automated daily predictions and weekly retraining, backed by a PostgreSQL data store.

---

## Model Performance

### Current Production Metrics (March 2026)

Sourced from `evaluate.py` on 6,511 labeled samples (`logs/labeled_predictions.csv`). Run `update_metrics_docs.py` to regenerate `docs/METRICS_SUMMARY.md` after new evaluation runs.

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 80.88% | 6,511 rows, threshold 0.380 |
| Balanced Accuracy | 68.05% | Corrects for class imbalance (4,721 vs 1,790) |
| AUC | 0.7792 | Threshold-independent, Proba column |
| Precision (Long signals) | 81.3% | Class 1 at threshold 0.380 |
| Recall (Long signals) | 39.6% | Precision-focused by design |
| F1 Score (Long signals) | 0.532 | |

### Ensemble Models (164 Features)

From `comprehensive_model_evaluation.py`, 80/20 train/test split.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost (Regime) | 64.06% | 37.71% | 42.70% | 0.400 |
| LightGBM (Regime) | 60.94% | 36.36% | 51.89% | 0.428 |
| CatBoost (Regime) | 62.77% | 38.59% | 54.86% | 0.453 |
| Ensemble | 63.83% | 39.05% | 51.08% | 0.443 |

### Threshold Strategy

| Strategy | Threshold | Precision | Win Rate | Trades |
|----------|-----------|-----------|----------|--------|
| Ultra-Conservative | 0.80 | 100% | 100% | 2 |
| Conservative | 0.65 | ~100% | ~100% | ~4 |
| **Precision-Focused (Production)** | **0.380** | **81.3%** | **66.0%** | **~871** |
| Balanced | 0.40 | ~52% | ~52% | ~880 |

### Feature Engineering Impact

| Stage | Features | Accuracy | Improvement |
|-------|----------|----------|-------------|
| Baseline | 103 | 58.4% | -- |
| + Cross-Asset | 130 | 61.5% | +3.1% |
| + Macro (Final) | 164 | 62.3% | +3.9% |

See `docs/METRICS_SUMMARY.md` for full breakdown of all metrics, features, and validation details.
See `SYSTEM_DESIGN.md` for architecture and canonical file references.

---

## Features

- **Forecasting Engine** - Predicts market regimes (spike, crash, neutral) using an ensemble of XGBoost, LightGBM, and CatBoost.
- **Signal Integration** - Merges macroeconomic indicators (FRED), sentiment data (Reddit and NewsAPI), and technical indicators (RSI, MACD, momentum).
- **Automated Backtesting** - Simulates performance with Sharpe ratio, Sortino ratio, max drawdown, and profit factor.
- **Monte Carlo Validation** - 1,000-run simulation with statistical significance testing and confidence intervals.
- **Walk-Forward Testing** - 33 out-of-sample windows from 2015 to 2026 with no look-ahead bias.
- **Parameter Sweeps** - Threshold optimization across precision/recall trade-off curves.
- **Multi-Asset Coverage** - 63 assets: 36 equities/ETFs, 10 bond ETFs, 7 precious metals, and 10 cryptocurrencies.
- **Production Automation** - Railway-deployed worker with scheduled training (weekly) and predictions (daily).
- **Modular Design** - Extendable for new models or data sources.

---

## Architecture

```
+-------------------------------------------------------------+
|                   Railway PostgreSQL                         |
|              (PRIMARY DATA STORE)                            |
+------------------------------+------------------------------+
                               |
              +----------------+-----------------+
              |   DataManager                    |
              |   (core/data_manager_postgres.py)|
              +----------------+-----------------+
                               |
              +----------------+-----------------+
              |   Feature Engineering            |
              |   build_feature_table.py         |
              +----------------+-----------------+
                               |
    +----------+---------------+---------------+-----------+
    |          |               |               |           |
    v          v               v               v           v
train.py  predict.py      backtest.py   dashboard_     api_server.py
                                        comprehensive.py
```

See `SYSTEM_DESIGN.md` for complete architecture documentation.

---

## Repository Structure

```
.
+-- SYSTEM_DESIGN.md              # Architecture and single source of truth
+-- config.py                     # Global configuration (canonical)
+-- train.py                      # Model training entry point
+-- predict.py                    # Prediction entry point
+-- backtest.py                   # Backtesting entry point
+-- advanced_backtesting.py       # Monte Carlo and walk-forward validation
+-- evaluate.py                   # Model evaluation
+-- update_metrics_docs.py        # Regenerates docs/METRICS_SUMMARY.md
+-- main.py                       # Interactive menu
+-- neurovest_cli.py              # CLI interface
+-- build_feature_table.py        # Feature engineering (canonical)
+-- dashboard_comprehensive.py    # Streamlit dashboard
|
+-- core/
|   +-- data_manager_postgres.py  # Data management (PostgreSQL/SQLite)
|   +-- data_pipeline.py          # Training pipeline
|   +-- prediction_engine.py      # Prediction system
|   +-- models/base_models.py     # Model architectures
|
+-- config/assets.yaml            # Asset definitions (63 assets)
+-- configs/                      # Trading profiles and thresholds
+-- data/                         # Local data cache
+-- logs/                         # Predictions and metrics
+-- models/                       # Trained model files
|
+-- docs/
|   +-- METRICS_SUMMARY.md        # Detailed performance metrics (auto-generated)
|
+-- outputs/                      # Backtest output files
+-- archive/                      # Deprecated code (do not use)
```

---

## Setup

### Requirements

- Python 3.10+

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for production (Railway PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:port/db

# Optional integrations
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=your_agent
NEWSAPI_KEY=your_newsapi_key
FRED_API_KEY=your_fred_api_key
```

Without `DATABASE_URL`, the system falls back to local SQLite which may return empty data.

### Quick Start

```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt

# Update market data
python3 update_data.py update

# Generate predictions
python3 predict.py

# Run backtest
python3 backtest.py

# Evaluate model
python3 evaluate.py

# Run advanced validation (Monte Carlo, walk-forward, stress tests)
python3 advanced_backtesting.py

# Regenerate metrics documentation
python3 update_metrics_docs.py

# Full pipeline
python3 run_all.py
```

### Diagnostic Commands

```bash
# System health
python3 health_check.py
python3 diagnose_system.py

# Database
python3 diagnose_database.py

# Evaluation
python3 evaluate.py
python3 extract_metrics.py --comprehensive
python3 evaluate_horizons.py
python3 evaluate_targets.py
```

---

## Example Outputs

- `logs/daily_predictions.csv` - Model predictions with confidence scores
- `logs/backtest_results.csv` - Simulated capital growth and key metrics
- `logs/model_performance.csv` - Detailed classification report
- `logs/evaluate_metrics.json` - Evaluation metrics summary
- `outputs/advanced_backtest_results.json` - Monte Carlo and walk-forward results
- `configs/best_thresholds.json` - Optimal threshold configuration

---

## Development Status

**Version:** Beta 1.1
**Stage:** Production - full ML pipeline deployed on Railway with automated scheduling.
**Last Updated:** March 2026

### Completed

- End-to-end pipeline: data, feature engineering, training, prediction, evaluation
- Sentiment and macroeconomic signal integration
- Production deployment on Railway with PostgreSQL
- Multi-asset support (36 equities/ETFs, 10 bonds, 7 precious metals, 10 crypto)
- Streamlit dashboard for visualization
- Weighted ensemble with 164 features across 8 categories
- Automated metrics documentation via `update_metrics_docs.py`

### In Progress

- Broker API integration (Alpaca, IBKR)
- Natural language market summarization

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core ML Forecasting: training, prediction, logging | Complete |
| 2 | Backtesting and Optimization: threshold sweeps, strategy simulation | Complete |
| 3 | Automation and Scheduling: Railway worker | Complete |
| 4 | Dashboard and Visualization: Streamlit | Complete |
| 5 | Multi-Asset Coverage: 63 assets | Complete |
| 6 | Ensemble Models: 164 features, regime detection | Complete |
| 7 | Trade Execution: broker API integration | In Progress |

---

## License

MIT License - free for public and commercial use. Attribution appreciated.

---

## Author

Built by [Keegan Hutchinson](https://github.com/keeg-Hson)

Feedback, contributions, and collaboration are welcome.
