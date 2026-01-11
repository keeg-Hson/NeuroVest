# NeuroVest API - User Guide

**AI-Powered Market Forecasting API**

Get real-time market predictions for 40+ assets including stocks, ETFs, cryptocurrencies, and precious metals.

**Base URL:** `https://neurovest-api-production-f8dc.up.railway.app`

---

## Quick Start (5 Minutes)

### Step 1: Get Your API Key

```bash
curl -X POST "https://neurovest-api-production-f8dc.up.railway.app/api/auth/register?username=YOUR_NAME"
```

**Response:**
```json
{
  "message": "User created successfully",
  "user_id": 1,
  "api_key": "abc123def456...",
  "note": "⚠️ Save your API key - it won't be shown again!"
}
```

**⚠️ IMPORTANT:** Copy and save your `api_key` - you'll need it for all requests!

---

### Step 2: Get Market Predictions

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://neurovest-api-production-f8dc.up.railway.app/api/predictions/SPY"
```

**Response:**
```json
{
  "ticker": "SPY",
  "prediction_date": "2026-01-02",
  "prediction_label": "SPIKE",
  "prob_crash": 0.322,
  "prob_normal": 0.322,
  "prob_spike": 0.356,
  "confidence": "low",
  "timestamp": "2026-01-05T02:14:00.674846"
}
```

---

### Step 3: Build Something Cool!

Use the predictions in your:
- Trading bots
- Portfolio managers
- Market dashboards
- Alert systems
- Research tools

---

## API Endpoints

### 1. **Health Check** (No Auth Required)

**GET** `/health`

Check if the API is online and connected to the database.

```bash
curl https://neurovest-api-production-f8dc.up.railway.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "last_prediction": "2026-01-02",
  "assets_count": 40,
  "timestamp": "2026-01-05T02:00:00"
}
```

**Status Codes:**
- `200` - System healthy
- `503` - Service unavailable (database error)

---

### 2. **Register User** (No Auth Required)

**POST** `/api/auth/register?username={YOUR_NAME}`

Create a new user account and receive an API key.

```bash
curl -X POST "https://neurovest-api-production-f8dc.up.railway.app/api/auth/register?username=trader123"
```

**Parameters:**
- `username` (required) - Your username (3-50 characters)

**Response:**
```json
{
  "message": "User created successfully",
  "user_id": 42,
  "api_key": "OJBnO-j7_9R-aRvciIUN5Xz2jKhdACv9U_gQQa69v_0",
  "note": "⚠️ Save your API key - it won't be shown again!"
}
```

**⚠️ Security:**
- API keys are generated using cryptographically secure random tokens
- Keys are 44 characters long
- Store your key securely - it won't be shown again!

---

### 3. **Get All Predictions** (Auth Required)

**GET** `/api/predictions`

Get latest predictions for all available assets.

```bash
curl -H "X-API-Key: YOUR_KEY" \
  "https://neurovest-api-production-f8dc.up.railway.app/api/predictions"
```

**Headers:**
- `X-API-Key` (required) - Your API key

**Response:**
```json
[
  {
    "ticker": "SPY",
    "prediction_date": "2026-01-02",
    "prediction_label": "SPIKE",
    "prob_crash": 0.322,
    "prob_normal": 0.322,
    "prob_spike": 0.356,
    "confidence": "low",
    "timestamp": "2026-01-05T02:14:12"
  },
  {
    "ticker": "BTC/USDT",
    "prediction_date": "2026-01-02",
    "prediction_label": "NORMAL",
    "prob_crash": 0.15,
    "prob_normal": 0.70,
    "prob_spike": 0.15,
    "confidence": "high",
    "timestamp": "2026-01-05T02:14:12"
  }
]
```

**Field Descriptions:**
- `ticker` - Asset symbol (SPY, BTC/USDT, etc.)
- `prediction_date` - Date of prediction
- `prediction_label` - Predicted outcome: **CRASH**, **NORMAL**, or **SPIKE**
- `prob_crash` - Probability of significant decline (>0.6% for stocks, >2% for crypto)
- `prob_normal` - Probability of range-bound movement
- `prob_spike` - Probability of significant gain (>0.6% for stocks, >2% for crypto)
- `confidence` - Prediction confidence: **high** (≥70%), **medium** (≥50%), or **low** (<50%)
- `timestamp` - When the API response was generated

**Status Codes:**
- `200` - Success
- `401` - Invalid API key
- `500` - Server error

---

### 4. **Get Specific Asset Prediction** (Auth Required)

**GET** `/api/predictions/{ticker}`

Get latest prediction for a single asset.

```bash
# Stocks/ETFs
curl -H "X-API-Key: YOUR_KEY" \
  "https://neurovest-api-production-f8dc.up.railway.app/api/predictions/SPY"

# Crypto (use URL encoding for slashes)
curl -H "X-API-Key: YOUR_KEY" \
  "https://neurovest-api-production-f8dc.up.railway.app/api/predictions/BTC%2FUSDT"
```

**Parameters:**
- `ticker` - Asset symbol (case-insensitive)

**Response:**
```json
{
  "ticker": "SPY",
  "prediction_date": "2026-01-02",
  "prediction_label": "SPIKE",
  "prob_crash": 0.322,
  "prob_normal": 0.322,
  "prob_spike": 0.356,
  "confidence": "low",
  "timestamp": "2026-01-05T02:14:00"
}
```

**Status Codes:**
- `200` - Success
- `401` - Invalid API key
- `404` - Asset not found or no predictions available
- `500` - Server error

---

### 5. **List All Assets** (Auth Required)

**GET** `/api/assets`

Get list of all available assets.

```bash
curl -H "X-API-Key: YOUR_KEY" \
  "https://neurovest-api-production-f8dc.up.railway.app/api/assets"
```

**Response:**
```json
{
  "total": 40,
  "assets": [
    "ADA_USDT",
    "AVAX_USDT",
    "BNB_USDT",
    "BTC_USDT",
    "SPY",
    "QQQ",
    "GLD",
    ...
  ]
}
```

**Status Codes:**
- `200` - Success
- `401` - Invalid API key
- `500` - Server error

---

## Supported Assets (40 Total)

### Stocks & ETFs (20 Assets)
- **Major Indices:** SPY, QQQ, IWM, DIA, VTI, EEM
- **Sectors:** XLF (Financials), XLK (Technology), XLE (Energy)
- **Bonds:** HYG, LQD, TLT, IEF, SHY, TNX
- **Currency:** DXY, UUP
- **Commodities:** USO, UNG, DBA, CORN, WEAT

### Precious Metals (7 Assets)
- **Gold:** GLD, IAU, GDX, GDXJ
- **Silver:** SLV
- **Other:** PPLT (Platinum), PALL (Palladium)

### Cryptocurrencies (10 Assets)
- BTC_USDT (Bitcoin)
- ETH_USDT (Ethereum)
- SOL_USDT (Solana)
- BNB_USDT (Binance Coin)
- XRP_USDT (Ripple)
- ADA_USDT (Cardano)
- DOGE_USDT (Dogecoin)
- AVAX_USDT (Avalanche)
- MATIC_USDT (Polygon)
- LINK_USDT (Chainlink)
- DOT_USDT (Polkadot)

---

## Understanding Predictions

### Prediction Labels

| Label | Meaning | Threshold (Stocks) | Threshold (Crypto) |
|-------|---------|-------------------|-------------------|
| **CRASH** | Expect significant decline | >0.6% down | >2% down |
| **NORMAL** | Range-bound or neutral | -0.6% to +0.6% | -2% to +2% |
| **SPIKE** | Expect significant gain | >0.6% up | >2% up |

### Confidence Levels

| Level | Confidence Score | Interpretation |
|-------|-----------------|----------------|
| **high** | ≥70% | Strong signal - model is confident |
| **medium** | 50-69% | Moderate signal - use with caution |
| **low** | <50% | Weak signal - high uncertainty |

### Probability Distribution

The API returns three probabilities that sum to 1.0:

```json
{
  "prob_crash": 0.15,
  "prob_normal": 0.70,
  "prob_spike": 0.15
}
```

**Interpretation:** 70% chance of normal movement, 15% chance of crash/spike

**Trading Strategy Example:**
- If `prob_spike` > 0.60 AND `confidence` == "high" → Consider long position
- If `prob_crash` > 0.60 AND `confidence` == "high" → Consider short position or exit
- If `prob_normal` > 0.60 → Range-bound, consider theta strategies

---

## Code Examples

### Python

```python
import requests
import json

# Configuration
API_URL = "https://neurovest-api-production-f8dc.up.railway.app"
API_KEY = "your-api-key-here"

headers = {"X-API-Key": API_KEY}

# Example 1: Get all predictions
def get_all_predictions():
    response = requests.get(f"{API_URL}/api/predictions", headers=headers)
    return response.json()

# Example 2: Get specific asset
def get_asset_prediction(ticker):
    response = requests.get(
        f"{API_URL}/api/predictions/{ticker}",
        headers=headers
    )
    return response.json()

# Example 3: Filter high-confidence SPIKE signals
def get_spike_opportunities():
    predictions = get_all_predictions()
    spikes = [
        p for p in predictions
        if p['prediction_label'] == 'SPIKE'
        and p['confidence'] == 'high'
        and p['prob_spike'] >= 0.65
    ]
    return spikes

# Usage
if __name__ == "__main__":
    # Get SPY prediction
    spy = get_asset_prediction("SPY")
    print(f"SPY Prediction: {spy['prediction_label']}")
    print(f"Confidence: {spy['confidence']}")
    print(f"Spike Probability: {spy['prob_spike']:.1%}")

    # Find opportunities
    opportunities = get_spike_opportunities()
    print(f"\nFound {len(opportunities)} high-confidence opportunities:")
    for opp in opportunities:
        print(f"  {opp['ticker']}: {opp['prob_spike']:.1%} spike probability")
```

---

### JavaScript / Node.js

```javascript
const axios = require('axios');

const API_URL = 'https://neurovest-api-production-f8dc.up.railway.app';
const API_KEY = 'your-api-key-here';

const headers = {
  'X-API-Key': API_KEY
};

// Get all predictions
async function getAllPredictions() {
  const response = await axios.get(`${API_URL}/api/predictions`, { headers });
  return response.data;
}

// Get specific asset
async function getAssetPrediction(ticker) {
  const response = await axios.get(
    `${API_URL}/api/predictions/${ticker}`,
    { headers }
  );
  return response.data;
}

// Example: Build trading signal
async function getTradingSignal(ticker) {
  const pred = await getAssetPrediction(ticker);

  let signal = 'HOLD';
  if (pred.prob_spike >= 0.65 && pred.confidence === 'high') {
    signal = 'BUY';
  } else if (pred.prob_crash >= 0.65 && pred.confidence === 'high') {
    signal = 'SELL';
  }

  return {
    ticker: pred.ticker,
    signal: signal,
    confidence: pred.confidence,
    prediction: pred.prediction_label
  };
}

// Usage
(async () => {
  const spy = await getTradingSignal('SPY');
  console.log(`SPY Signal: ${spy.signal} (${spy.confidence} confidence)`);

  const btc = await getTradingSignal('BTC_USDT');
  console.log(`BTC Signal: ${btc.signal} (${btc.confidence} confidence)`);
})();
```

---

### cURL / Shell Script

```bash
#!/bin/bash

API_URL="https://neurovest-api-production-f8dc.up.railway.app"
API_KEY="your-api-key-here"

# Get SPY prediction
echo "Fetching SPY prediction..."
curl -s -H "X-API-Key: $API_KEY" \
  "$API_URL/api/predictions/SPY" | jq '.'

# Get all high-confidence signals
echo -e "\nHigh-confidence signals:"
curl -s -H "X-API-Key: $API_KEY" \
  "$API_URL/api/predictions" | \
  jq '.[] | select(.confidence == "high") | {ticker, prediction_label, prob_spike, confidence}'

# Check specific assets
for ticker in SPY QQQ BTC_USDT ETH_USDT; do
  echo -e "\nChecking $ticker..."
  curl -s -H "X-API-Key: $API_KEY" \
    "$API_URL/api/predictions/$ticker" | \
    jq '{ticker, prediction_label, confidence}'
done
```

---

## Error Handling

### Common Error Responses

**401 Unauthorized**
```json
{
  "detail": "Invalid API key. Register at /api/auth/register to get a key.",
  "timestamp": "2026-01-05T02:00:00"
}
```
**Solution:** Check your API key or register for a new one.

---

**404 Not Found**
```json
{
  "detail": "No predictions found for XYZ. Available assets: /api/assets",
  "timestamp": "2026-01-05T02:00:00"
}
```
**Solution:** Check asset ticker or call `/api/assets` for list of available assets.

---

**500 Internal Server Error**
```json
{
  "detail": "Failed to fetch predictions: database error",
  "timestamp": "2026-01-05T02:00:00"
}
```
**Solution:** Try again in a few seconds. If persists, check service status at `/health`.

---

## Best Practices

### 1. **Cache Responses**
Predictions update once daily (4:30 PM EST). Cache results to reduce API calls.

```python
import time

cache = {}
CACHE_TTL = 3600  # 1 hour

def get_prediction_cached(ticker):
    now = time.time()
    if ticker in cache and (now - cache[ticker]['time']) < CACHE_TTL:
        return cache[ticker]['data']

    # Fetch fresh data
    data = get_asset_prediction(ticker)
    cache[ticker] = {'data': data, 'time': now}
    return data
```

### 2. **Handle Errors Gracefully**
Always wrap API calls in try/except blocks.

```python
try:
    prediction = get_asset_prediction("SPY")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Asset not found")
    elif e.response.status_code == 401:
        print("Invalid API key")
    else:
        print(f"Error: {e}")
```

### 3. **Validate Inputs**
Check ticker format before making requests.

```python
def validate_ticker(ticker):
    # Stock/ETF tickers: 1-10 alphanumeric
    # Crypto tickers: XXX/USDT or XXX_USDT
    if '/' in ticker or '_' in ticker:
        # Crypto
        return ticker.replace('/', '_').upper()
    else:
        # Stock/ETF
        return ticker.upper()

ticker = validate_ticker("spy")  # Returns "SPY"
prediction = get_asset_prediction(ticker)
```

### 4. **Combine Multiple Signals**
Don't trade on a single prediction - combine with your own analysis.

```python
def should_trade(ticker, your_analysis):
    pred = get_asset_prediction(ticker)

    # Require both API and your analysis to agree
    if (pred['prediction_label'] == 'SPIKE' and
        pred['confidence'] == 'high' and
        your_analysis['trend'] == 'bullish'):
        return 'BUY'

    return 'HOLD'
```

---

## Rate Limits

**Current Limits:**
- No rate limits enforced (fair use policy)
- Recommended: <60 requests/minute
- Future plans: Tiered limits based on subscription

**Fair Use:**
- Don't poll endpoints every second
- Cache predictions (they update once daily)
- Use batch endpoint `/api/predictions` instead of individual calls

---

## Support & Feedback

**Questions?**
- Check API documentation: `/docs` (Swagger UI)
- View this guide: `API_USER_GUIDE.md`

**Issues?**
- Check system health: `/health` endpoint
- Contact: GitHub Issues

**Feature Requests?**
- Batch predictions API
- Historical predictions
- WebSocket real-time updates
- Custom asset uploads via API

---

## Changelog

### v2.0.0 (2026-01-05)
- ✅ Standalone REST API deployed
- ✅ 40 assets supported
- ✅ Authentication via API keys
- ✅ Real-time predictions from PostgreSQL
- ✅ Auto-generated Swagger documentation

---

## Legal & Disclaimer

**Not Financial Advice**
- Predictions are for informational purposes only
- Past performance does not guarantee future results
- Always do your own research
- Never invest more than you can afford to lose

**Data Accuracy**
- API relies on third-party data sources (Yahoo Finance, CCXT)
- Predictions are probabilistic, not guaranteed
- Model performance varies during unprecedented market conditions

**Terms of Use**
- API provided "as is" without warranty
- Users responsible for their own trading decisions
- Not liable for losses incurred from using predictions

---

**Ready to start?**

1. Register: `POST /api/auth/register?username=YOUR_NAME`
2. Get predictions: `GET /api/predictions`
3. Build something awesome! 🚀

**API Base URL:** `https://neurovest-api-production-f8dc.up.railway.app`
**Documentation:** `https://neurovest-api-production-f8dc.up.railway.app/docs`
