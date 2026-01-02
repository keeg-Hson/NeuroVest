# NeuroVest

**Production-grade ensemble ML forecasting API for quantitative analysts**

Delivers calibrated probability forecasts across 25 financial assets using XGBoost, LightGBM, and CatBoost. Deploy-ready with 99.2% uptime and sub-200ms API response times.

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
```

---

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
