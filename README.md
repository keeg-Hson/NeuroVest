# NeuroVest

**Ensemble ML probability forecasts for financial assets. Built for quant analysts.**

Daily predictions for 25 assets using XGBoost, LightGBM, and CatBoost. Returns 3-class probabilities (CRASH/NORMAL/SPIKE) with confidence scores.

🔗 **Live API:** https://neurovestdemo.up.railway.app

---

## Why This Exists

**Problem:** Most market prediction tools either (1) give binary signals with no uncertainty quantification, or (2) require significant ML expertise to build and maintain.

**Solution:** Production-ready ensemble predictions with calibrated probabilities. No ML expertise needed to use, just call the API.

**Better Than:**
- Bloomberg's ML models: Free, open-source, customizable
- Building your own: Pre-trained, maintained, deployed
- Free alternatives: Ensemble approach, confidence calibration, production-grade

---

## Quick Start

```python
import requests

# Get predictions
r = requests.get("https://neurovestdemo.up.railway.app/api/predictions")
spy = next(p for p in r.json() if p['ticker'] == 'SPY')

print(f"CRASH: {spy['prob_crash']:.1%}, SPIKE: {spy['prob_spike']:.1%}")
print(f"Confidence: {spy['confidence']}")
```

**Local Installation:**
```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt
python3 main.py  # Run full pipeline
```

---

## Production Details

### Costs (Railway Deployment)
- **Database:** PostgreSQL, $5/mo (1GB storage)
- **Compute:** 512MB RAM, $5/mo
- **Total:** ~$10/month
- **Free tier:** Available but limited (500 hours/month)

### Actual Usage Metrics
- **Uptime:** 99.2% over 30 days
- **Response Time:**
  - p50: 87ms
  - p95: 156ms
  - p99: 243ms
- **Error Rate:** 0.3% (mostly API rate limits)
- **Data Freshness:** Updated daily at 4:30 PM EST
- **Last Model Retrain:** Sundays at 2:00 AM EST

### Monitoring
```python
# Health check endpoint
GET /health
{
  "status": "healthy",
  "last_update": "2025-12-29T16:30:00Z",
  "models_trained": 3,
  "data_age_hours": 2.5,
  "prediction_latency_p95": 156
}
```

**Alerts configured for:**
- Prediction latency >500ms
- Model age >7 days
- Data worker offline >30min
- Error rate >5%

---

## Performance (25-year SPY backtest)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Precision** | 69.85% | 33% (random) |
| **Win Rate** | 54% | 50% (coin flip) |
| **Sharpe Ratio** | 2.55 | 0.42 (buy-hold) |
| **Max Drawdown** | -5.4% | -55% (buy-hold) |
| **Total Return** | 191% | 467% (buy-hold) |

**Important:** Lower total return than buy-hold is intentional - this is for risk-adjusted strategies, not maximum returns.

---

## Known Failures

**Model Drift:**
- 2008 Financial Crisis: Precision dropped to 45% (from 70%)
- COVID-19 Crash: Missed initial spike, recovered after 2 weeks
- Solution: Weekly retraining helps but doesn't eliminate drift

**Crypto:**
- Only 300 days of training data (vs 6000+ for stocks)
- Precision: 61% (vs 70% for stocks)
- More volatile, less reliable

**Daily-Only:**
- No intraday predictions
- Updates once per day at 4:30 PM EST
- Not suitable for day trading

**Data Dependencies:**
- Requires yfinance (free, but rate-limited)
- FRED API (optional, free, but improves macro features)
- CCXT for crypto (Coinbase, some assets unavailable)

---

## API Reference

| Endpoint | Response | Latency |
|----------|----------|---------|
| `/api/predictions` | All 25 assets | ~150ms |
| `/api/predictions/{ticker}` | Single asset | ~80ms |
| `/api/regime` | Market regime | ~50ms |
| `/health` | Service health | ~10ms |

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

## Deployment

**Railway (Production):**
```bash
# Add DATABASE_URL to both DataWorker2 and Dashboard2 services
# Set start command: bash bootstrap_all.sh && bash start_combined.sh
railway up
```

**Docker (Local):**
```bash
docker build -t neurovest .
docker run -p 8501:8501 neurovest
```

**Stack:**
- Python 3.11
- PostgreSQL (Railway managed)
- Streamlit (dashboard)
- APScheduler (cron jobs)

---

## Support

- **Issues:** https://github.com/keeg-Hson/NeuroVest/issues
- **Docs:** See `docs/` for detailed guides (if needed)

---

## Disclaimer

This is a **forecasting tool**, not financial advice. Models fail during unprecedented conditions. Use at your own risk. Past performance ≠ future results.
