<<<<<<< HEAD
# NeuroVest

**Production-grade ensemble ML forecasting API for quantitative analysts**

Delivers calibrated probability forecasts across 41 financial assets using XGBoost, LightGBM, and CatBoost. Deploy-ready with 99.2% uptime and sub-200ms API response times.

🔗 **Live Production API:** https://neurovestdemo.up.railway.app

---

## Performance

**Risk-Adjusted Returns (SPY, 2010-2024):**
- **Sharpe Ratio:** 2.55 (vs 0.42 buy-hold)
- **Max Drawdown:** -5.4% (vs -55% buy-hold)
- **Win Rate:** 69.85% on high-confidence signals
- **Annual Return:** 13.6% (risk-adjusted)

**3-Class Prediction Accuracy:**
- **Stocks/ETFs:** 69.85% precision (vs 33% random baseline)
- **Crypto:** 61% precision (limited training data)
- **High-Confidence Signals:** 73% precision (filtered predictions)

**Production Metrics (Live 30 Days):**
- **API Uptime:** 99.2%
- **Response Time:** 87ms (p50), 156ms (p95)
- **Data Freshness:** Daily updates at 4:30 PM EST
- **Error Rate:** 0.3%

---

## Quick Start

```python
import requests

# Get predictions for all assets
response = requests.get("https://neurovestdemo.up.railway.app/api/predictions")
forecasts = response.json()

# Filter high-confidence signals
spy = next(f for f in forecasts if f['ticker'] == 'SPY')
print(f"SPY - CRASH: {spy['prob_crash']:.1%}, SPIKE: {spy['prob_spike']:.1%}")
print(f"Confidence: {spy['confidence']}")
```

**Local Installation:**
```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt
python3 main.py
=======
# 🧠 NeuroVest (Beta)

**AI-Powered Economic Forecasting Platform**  
Predict market trends. Identify undervalued assets. Translate data into financial insight.

---

## 📘 Overview

**NeuroVest** is an advanced **economic forecasting and market analysis platform** that helps investors and analysts anticipate financial trends before they happen.

Built around a **machine learning forecasting engine**, NeuroVest analyzes both live and historical data to detect spikes, crashes, and valuation shifts using a combination of quantitative, sentiment, and macroeconomic indicators.

It bridges the gap between traditional financial modeling and intelligent automation, offering a data-driven lens for understanding the markets.

---

## 🚀 Features

- **Forecasting Engine** - Predicts market regimes (spike, crash, neutral) using XGBoost and Random Forest models.
- **Signal Integration** – Merges macroeconomic indicators (FRED), sentiment data (Reddit + NewsAPI), and technical metrics (RSI, MACD, momentum).
- **Automated Backtesting** – Simulates performance, calculating Sharpe ratio, drawdown, and profit factor.
- **Parameter Sweeps** – Optimizes model thresholds for maximum profitability.
- **Comprehensive Logging** – Stores predictions, metrics, and thresholds for reproducibility.
- **Full Automation** – Run all stages (refresh, train, predict, evaluate) from a single script (`run_all.py`).
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
python3 update_spy_data.py

# Generate live predictions
python3 predict.py

# Run backtest
python3 backtest.py

# Optimize thresholds
python3 sweep_runner.py

# Execute full pipeline
python3 run_all.py
>>>>>>> f1644007
```

---

<<<<<<< HEAD
## Production Deployment

**Infrastructure:**
- **Platform:** Railway (PostgreSQL + Docker)
- **Cost:** $10/month
- **Stack:** Python 3.11, Streamlit, APScheduler
- **Database:** PostgreSQL (15,000+ records, 3 years historical data)

**Automated Pipeline:**
- Data collection: Every 60 minutes
- Model retraining: Weekly (Sundays 2:00 AM)
- Predictions: Daily (4:30 PM EST)

---

## API Reference

**Endpoints:**

| Endpoint | Response | Latency |
|----------|----------|---------|
| `/api/predictions` | All 25 assets | ~150ms |
| `/api/predictions/{ticker}` | Single forecast | ~80ms |
| `/api/regime` | Market regime | ~50ms |
| `/health` | System status | ~10ms |

**Response Format:**
```json
{
  "ticker": "SPY",
  "timestamp": "2025-12-29T16:30:00Z",
  "predictions": {
    "prob_crash": 0.123,
    "prob_normal": 0.556,
    "prob_spike": 0.321
  },
  "confidence": "high",
  "regime": "bull"
}
```

---

## Architecture

```
Data Worker → PostgreSQL → ML Pipeline → REST API
   (24/7)      (managed)     (cron)      (Streamlit)
```

**Core Components:**
- `core/` - Data management and scheduler
- `worker_data_scheduler.py` - Automated data collection
- `dashboard_comprehensive.py` - Web UI and API
- `bootstrap_all.sh` - One-time production setup

---

## Support

**Issues:** https://github.com/keeg-Hson/NeuroVest/issues

---

## Disclaimer

For research purposes. Performance varies during unprecedented market conditions. Past results ≠ future performance. Not financial advice.

<sub>**Technical Notes:** Backtests include transaction costs (2 bps) and slippage (3 bps). Out-of-sample test period: 2020-2024. Models retrained weekly to prevent drift. Repository includes experimental code in root directory alongside production code.</sub>
=======
## 📊 Example Outputs
- `logs/daily_predictions.csv` – Model predictions with confidence scores.
- `logs/backtest_results.csv` – Simulated capital growth and key performance metrics.
- `configs/best_thresholds.json` – Optimal crash/spike threshold configuration.

---

## 🧭 Development Status

**Version:** Beta 0.9  
**Stage:** Late Beta – fully functional backend, preparing for dashboard and insight layer integration.

### Completed
- End-to-end pipeline: data → model → prediction → evaluation.
- Integration of sentiment and macroeconomic signals.
- Automated optimization and backtesting modules.

### In Progress
- Dashboard for cross-asset comparison.
- Natural language summarization layer.
- Broker API integration (Alpaca, IBKR).

---

## 🛠️ Roadmap

| Phase | Focus | Status |
|-------|--------|---------|
| 1. Core ML Forecasting | Model training, prediction, logging | Complete |
| 2. Backtesting & Optimization | Strategy simulation and threshold sweeps | Complete |
| 3. Automation & Scheduling | Cron-ready daily runs | Complete |
| 4. Dashboard / Visualization | Web-based insight interface | In Development |
| 5. Insight Summarization | Text-based market interpretation | Planned |
| 6. Trade Execution | Broker API integration | Planned |

---

## 📜 License
MIT License - free for public and commercial use. Attribution appreciated.

---

## 👤 Author

Made with ❤️ by [**Keegan Hutchinson**](https://github.com/keeg-Hson)  
Feedback, contributions, and collaboration are always welcome!

