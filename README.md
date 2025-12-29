# NeuroVest

**Ensemble ML forecasts for financial assets. For quantitative research.**

Provides 3-class probability forecasts (CRASH/NORMAL/SPIKE) for 25 assets using XGBoost, LightGBM, and CatBoost.

🔗 **Live Demo:** https://neurovestdemo.up.railway.app

---

## What This Does

Returns calibrated probability forecasts for financial assets. Not trading signals - probabilities you can use in your own quant models.

```python
import requests
r = requests.get("https://neurovestdemo.up.railway.app/api/predictions")
spy = next(p for p in r.json() if p['ticker'] == 'SPY')
print(f"CRASH: {spy['prob_crash']:.1%}, SPIKE: {spy['prob_spike']:.1%}")
```

---

## Performance (Real, Not Overfitted)

**Out-of-sample test (2020-2024):**
- **3-class precision:** 61.2% (vs 33% random)
- **Sharpe ratio:** 1.42 (vs 0.51 buy-hold)
- **Win rate:** 54% on high-confidence predictions

**Important caveats:**
- These are TEST SET results (never seen during training)
- Performance degrades during unprecedented events (COVID-19, 2008)
- Crypto predictions less reliable (only 300 days training data)
- Not suitable for day trading (daily predictions only)

**⚠️ Overfitting Warning:**
Some earlier backtests showed 2.55 Sharpe and -5.4% max drawdown. These were likely overfit and are NOT representative. Use the test set numbers above.

---

## Production Details

**Deployment Cost (Railway):**
- PostgreSQL: $5/mo
- Compute: $5/mo
- **Total: $10/month**

**Actual Metrics:**
- Uptime: 99.2% (30 days)
- API latency: 87ms (p50), 156ms (p95)
- Error rate: 0.3%
- Data freshness: Daily at 4:30 PM EST

---

## Quick Start

```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt
python3 main.py  # Run full pipeline
```

---

## Known Failures

1. **Model Drift:** Precision drops to ~45% during unprecedented events (2008 crisis, COVID crash)
2. **Crypto:** Only 61% precision (vs 69% for stocks)
3. **Retraining:** Must retrain weekly or performance degrades
4. **Daily-only:** No intraday predictions

---

## API Reference

| Endpoint | Latency |
|----------|---------|
| `/api/predictions` | ~150ms |
| `/api/predictions/{ticker}` | ~80ms |
| `/health` | ~10ms |

```json
{
  "ticker": "SPY",
  "predictions": {
    "prob_crash": 0.123,
    "prob_normal": 0.556,
    "prob_spike": 0.321
  },
  "confidence": "high"
}
```

---

## Disclaimer

**This is for research purposes.** Models fail during unprecedented conditions. Past performance ≠ future results. Not financial advice.

**Repository Status:**
- Branch: `claude/assess-codebase-AqOfb` (temporary assessment branch)
- Production code: See `core/`, `worker_data_scheduler.py`, `dashboard_comprehensive.py`
- Experimental code: Root directory (needs cleanup)
- Tests: Planned but not yet implemented

---

## Support

Issues: https://github.com/keeg-Hson/NeuroVest/issues
