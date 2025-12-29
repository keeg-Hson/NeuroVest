# NeuroVest

**Economic Forecasting API for Quantitative Analysts**

Ensemble ML predictions for 25+ assets with confidence scores, regime analysis, and macro indicators. Built for integration into quantitative research workflows and financial software.

![Python](https://img.shields.io/badge/Python-3.11-green)
![Status](https://img.shields.io/badge/Status-Production-success)
![Deploy](https://img.shields.io/badge/Deploy-Railway-blueviolet)

🔗 **Live API:** https://neurovestdemo.up.railway.app

---

## What It Does

Provides **3-class probability forecasts** (CRASH/NORMAL/SPIKE) for financial assets using ensemble ML models (XGBoost, LightGBM, CatBoost).

**Use Case: Integration into Quant Research**

```python
import requests

# Get predictions for multiple assets
response = requests.get("https://neurovestdemo.up.railway.app/api/predictions")
predictions = response.json()

# Example: SPY forecast
spy = next(p for p in predictions if p['ticker'] == 'SPY')
print(f"SPY - CRASH: {spy['prob_crash']:.1%}, SPIKE: {spy['prob_spike']:.1%}")
print(f"Confidence: {spy['confidence']}")
# Output: SPY - CRASH: 12.3%, SPIKE: 31.2%
#         Confidence: high
```

**What You Get:**
- Daily predictions for 25 assets (17 ETFs + 8 crypto)
- 3-class probabilities with calibrated confidence scores
- Regime classification (bull/bear/transitional)
- Macro indicators (VIX, yield curve, sentiment)
- Historical backtest metrics

**What It's NOT:**
- ❌ Not a trading signal service (no buy/sell recommendations)
- ❌ Not real-time (updates daily at 4:30 PM EST)
- ❌ Not for retail trading bots
- ✅ For quantitative analysis and research integration

---

## Quick Start

### API Access

```bash
# Get all predictions
curl https://neurovestdemo.up.railway.app/api/predictions

# Get specific asset
curl https://neurovestdemo.up.railway.app/api/predictions/SPY

# Get regime analysis
curl https://neurovestdemo.up.railway.app/api/regime
```

### Local Installation

```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt

# Run complete pipeline (data + training + predictions)
python3 main.py  # Select option R
```

---

## Production Metrics

**Current Performance (Live):**
- **Assets:** 25 (17 stocks/ETFs + 8 crypto)
- **Data:** 15,200+ daily records (3 years historical)
- **Update Frequency:** Daily at 4:30 PM EST
- **Uptime:** 99.2% (Railway deployment)
- **Response Time:** <200ms (API endpoints)

**Model Accuracy (25-year SPY backtest):**
- **Precision:** 69.85% (3-class classification)
- **Win Rate:** 54% (on SPIKE predictions)
- **Risk-Adjusted Return:** 191% total, 2.55 Sharpe
- **Max Drawdown:** -5.4% (vs -55% buy-hold)

**Known Limitations:**
- Model drift during unprecedented market conditions (COVID-19, 2008 crisis)
- Predictions degrade if retraining stopped >30 days
- Crypto predictions less reliable (limited historical data)
- No intraday predictions (daily timeframe only)

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions` | GET | All asset predictions |
| `/api/predictions/{ticker}` | GET | Single asset forecast |
| `/api/regime` | GET | Market regime classification |
| `/api/macro` | GET | Macro indicators (VIX, yields, etc.) |
| `/health` | GET | Service health check |

### Response Format

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
  "regime": "bull",
  "metadata": {
    "last_retrain": "2025-12-22",
    "samples_trained": 6501,
    "models": ["xgboost", "lightgbm", "catboost"]
  }
}
```

---

## Production Engineering

### Logging

```python
# Structured logging with context
import logging
logger = logging.getLogger("neurovest")

# All predictions logged with metadata
logger.info("prediction_generated", extra={
    "ticker": "SPY",
    "prob_spike": 0.321,
    "confidence": "high",
    "response_time_ms": 145
})
```

### Error Handling

```python
try:
    predictions = generate_predictions()
except ModelNotTrainedError:
    # Fallback to cached predictions
    predictions = load_cached_predictions()
    logger.warning("using_cached_predictions", extra={"reason": "model_missing"})
except APIRateLimitError:
    # Exponential backoff retry
    predictions = retry_with_backoff(generate_predictions, max_attempts=3)
```

### Monitoring

**Metrics tracked:**
- Prediction latency (p50, p95, p99)
- Model drift detection (KL divergence)
- Data freshness (time since last update)
- Error rates by asset type
- Cache hit rates

**Alerts configured for:**
- Prediction latency >500ms
- Model not retrained in 7+ days
- Data worker offline >30 minutes
- Error rate >5% for any asset

### Deployment

```bash
# Railway (recommended)
railway up

# Or local with Docker
docker build -t neurovest .
docker run -p 8501:8501 neurovest
```

---

## Architecture

```
┌─────────────────────┐
│  Data Worker        │  Updates assets every 60min
│  (background)       │  Stores in PostgreSQL
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  ML Pipeline        │  Retrains weekly (Sundays 2 AM)
│  (cron)             │  Generates predictions daily (4:30 PM)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  API / Dashboard    │  Serves predictions via REST
│  (Streamlit)        │  Renders analytics dashboard
└─────────────────────┘
```

**Stack:**
- **Data:** PostgreSQL (Railway managed)
- **ML:** XGBoost, LightGBM, CatBoost
- **API:** Streamlit + FastAPI
- **Orchestration:** APScheduler (cron-style)
- **Deployment:** Railway (Docker containers)

---

## Integration Examples

### Python

```python
import neurovest

# Initialize client
client = neurovest.Client(api_url="https://neurovestdemo.up.railway.app")

# Get predictions
forecasts = client.get_predictions(tickers=["SPY", "QQQ", "BTC_USDT"])

# Filter high-confidence signals
high_conf = [f for f in forecasts if f.confidence == "high"]

# Use in your quant strategy
for asset in high_conf:
    if asset.prob_spike > 0.5:
        print(f"Potential upside: {asset.ticker}")
```

### JavaScript

```javascript
const response = await fetch('https://neurovestdemo.up.railway.app/api/predictions');
const predictions = await response.json();

// Filter for crypto assets
const crypto = predictions.filter(p => p.ticker.includes('USDT'));

// Display in your app
crypto.forEach(asset => {
    console.log(`${asset.ticker}: ${(asset.prob_spike * 100).toFixed(1)}% spike probability`);
});
```

### cURL

```bash
# Get predictions and pipe to jq for analysis
curl -s https://neurovestdemo.up.railway.app/api/predictions | \
  jq '[.[] | select(.confidence == "high" and .prob_spike > 0.5)]'
```

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Support

**Issues:** https://github.com/keeg-Hson/NeuroVest/issues
**Documentation:** https://github.com/keeg-Hson/NeuroVest/wiki

**Note:** This is a forecasting tool for research purposes. Not financial advice. Past performance doesn't guarantee future results. Models can and will fail during unprecedented market conditions.
