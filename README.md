# 🧠 NeuroVest (Beta)

**AI-Powered Economic Forecasting Platform**
Predict market trends. Identify undervalued assets. Translate data into financial insight.

---

## 📘 Overview

**NeuroVest** is an advanced **economic forecasting and market analysis platform** that helps investors and analysts anticipate financial trends before they happen.

Built around a **machine learning forecasting engine**, NeuroVest analyzes both live and historical data to detect spikes, crashes, and valuation shifts using a combination of quantitative, sentiment, and macroeconomic indicators.

It bridges the gap between traditional financial modeling and intelligent automation, offering a data-driven lens for understanding the markets.

---

## 📈 Model Performance

### Current Production Metrics (Feb 2026)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 78.62% | - |
| **AUC** | 0.7792 | - |
| **Sharpe Ratio** | 2.55 | vs 0.42 buy-and-hold |
| **Max Drawdown** | -5.4% | vs -55% buy-and-hold |
| **Win Rate** | 54.0% | - |
| **Precision** | 86.99% | When model signals |
| **Sortino Ratio** | 3.12 | - |
| **Profit Factor** | 1.87 | - |

### Ensemble Models (164 Features)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **XGBoost (Regime)** | 64.29% | 38.08% | 42.12% | 0.400 |
| **LightGBM (Regime)** | 62.06% | 36.88% | 48.10% | 0.417 |
| **CatBoost (Regime)** | 61.67% | 37.57% | 53.80% | 0.442 |
| **Ensemble** | 63.98% | 38.95% | 48.37% | 0.432 |

### Threshold Strategy (Precision-Focused)

| Strategy | Threshold | Precision | Win Rate | Trades |
|----------|-----------|-----------|----------|--------|
| Ultra-Conservative | 0.65 | 100% | 100% | 4 |
| Conservative | 0.55 | 92.6% | 92.6% | 27 |
| **Precision-Focused** | **0.45** | **~87%** | **54%** | **~50** |
| Balanced | 0.50 | 38.7% | 52.7% | 619 |
| Aggressive | 0.40 | 31.6% | 52.4% | 880 |

### Feature Engineering Impact

| Stage | Features | Accuracy | Improvement |
|-------|----------|----------|-------------|
| Baseline | 103 | 58.4% | - |
| + Cross-Asset | 130 | 61.5% | +3.1% |
| + Macro (FINAL) | 164 | 62.3% | +3.9% |

> See `docs/METRICS_SUMMARY.md` for detailed breakdown of all metrics, features, and model strengths.

---

## 🚀 Features

- **Forecasting Engine** - Predicts market regimes (spike, crash, neutral) using ensemble of XGBoost, LightGBM, CatBoost, and LSTM models.
- **Signal Integration** – Merges macroeconomic indicators (FRED), sentiment data (Reddit + NewsAPI), and technical metrics (RSI, MACD, momentum).
- **Automated Backtesting** – Simulates performance, calculating Sharpe ratio, Sortino ratio, max drawdown, and profit factor.
- **Parameter Sweeps** – Optimizes model thresholds for maximum profitability or precision.
- **Multi-Asset Coverage** – 31 stocks/ETFs/commodities + 10 cryptocurrencies.
- **Production Automation** – Railway-deployed worker with scheduled training (weekly) and predictions (daily).
- **Comprehensive Logging** – Stores predictions, metrics, and thresholds for reproducibility.
- **Modular Design** – Easily extendable for new models or data sources.

---

## 🧩 Architecture

```
DATA SOURCES -> FEATURE ENGINEERING -> MODEL TRAINING -> PREDICTION -> BACKTESTING -> OPTIMIZATION
│                     │                          │                    │                     │
│                     │                          │                    │                     │
│              utils.py + external_signals.py     │                    │                     │
│                         train.py                │                    │                     │
│                                  predict.py      │                    │                     │
│                                           backtest.py -> sweep_runner.py
│
└── run_all.py  (Master pipeline orchestrator)
```

---

## 📁 Repository Structure

```
.
├── configs/              # Parameter and sweep configurations
├── data/                 # Market and macroeconomic datasets
├── logs/                 # Predictions, backtests, and optimization results
├── models/               # Trained model files (XGBoost / RandomForest)
├── train.py              # Model training pipeline
├── predict.py            # Live and historical prediction module
├── backtest.py           # Capital simulation and evaluation
├── external_signals.py   # Reddit, NewsAPI, and FRED integration
├── sweep_runner.py       # Threshold optimization sweeps
├── utils.py              # Shared helper and feature functions
├── run_all.py            # Full end-to-end pipeline
└── .env                  # Environment variables (API keys)
```

---

## ⚙️ Setup

### Requirements
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Environment Variables
Create a `.env` file in the project root with your credentials:
```bash
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=your_agent
NEWSAPI_KEY=your_newsapi_key
FRED_API_KEY=your_fred_api_key
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest

# Install dependencies
pip install -r requirements.txt

# Update market data
python3 update_data.py update

# Generate live predictions
python3 predict.py

# Run backtest
python3 backtest.py

# Optimize thresholds
python3 sweep_runner.py

# Execute full pipeline
python3 run_all.py
```

### Diagnostic Commands
```bash
# System Health
python3 health_check.py               # Full 7-check audit
python3 diagnose_system.py            # Data, models, predictions, pipeline

# Database (Railway)
python3 diagnose_database.py          # PostgreSQL connection check

# Model Evaluation
python3 evaluate.py                   # Unified evaluation
python3 extract_metrics.py --comprehensive  # Extract real metrics

# Horizon/Target Testing
python3 evaluate_horizons.py          # Test prediction timeframes
python3 evaluate_targets.py           # Test labeling strategies
```

---

## 📊 Example Outputs
- `logs/daily_predictions.csv` – Model predictions with confidence scores.
- `logs/backtest_results.csv` – Simulated capital growth and key performance metrics.
- `configs/best_thresholds.json` – Optimal crash/spike threshold configuration.
- `all_models_comparison.csv` – Full model comparison with metrics.
- `optimization_metric_comparison.csv` – Threshold optimization results.

---

## 🧭 Development Status

**Version:** Beta 1.0
**Stage:** Production - Full ML pipeline deployed on Railway with automated scheduling.

### Completed
- End-to-end pipeline: data → model → prediction → evaluation
- Integration of sentiment and macroeconomic signals
- Automated optimization and backtesting modules
- Production deployment on Railway with PostgreSQL
- Multi-asset support (31 stocks/ETFs + 10 cryptocurrencies)
- Streamlit dashboard for visualization
- Weighted ensemble achieving 68.8% accuracy
- Cross-asset and macro feature engineering (+3.9% accuracy gain)

### In Progress
- Broker API integration (Alpaca, IBKR)
- Natural language market summarization

---

## 🛠️ Roadmap

| Phase | Focus | Status |
|-------|--------|---------|
| 1. Core ML Forecasting | Model training, prediction, logging | Complete |
| 2. Backtesting & Optimization | Strategy simulation and threshold sweeps | Complete |
| 3. Automation & Scheduling | Railway worker with daily/weekly jobs | Complete |
| 4. Dashboard / Visualization | Streamlit web interface | Complete |
| 5. Multi-Asset Coverage | 41 assets (stocks, crypto, commodities) | Complete |
| 6. Ensemble Models | Weighted ensemble (68.8% accuracy) | Complete |
| 7. Trade Execution | Broker API integration | In Progress |

---

## 📜 License
MIT License - free for public and commercial use. Attribution appreciated.

---

## 👤 Author

Built by [**Keegan Hutchinson**](https://github.com/keeg-Hson)  
Feedback, contributions, and collaboration are always welcome.

> NeuroVest - Invest with intelligence.




