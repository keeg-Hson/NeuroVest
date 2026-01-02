## NeuroVest Trading API

REST API for programmatic access to the trading system.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic
```

### 2. Start API Server

```bash
# Development mode
python api/trading_api.py

# Production mode
uvicorn api.trading_api:app --host 0.0.0.0 --port 8000
```

Server runs at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### 3. Test API

```bash
curl http://localhost:8000/health
```

---

## Authentication

All endpoints (except `/` and `/health`) require authentication.

**Header:**
```
Authorization: Bearer demo_token_replace_in_production
```

**Example:**
```bash
curl -H "Authorization: Bearer demo_token_replace_in_production" \
     http://localhost:8000/portfolio/status
```

---

## Endpoints

### Health & Status

#### `GET /`
Root endpoint with API info

**Response:**
```json
{
  "name": "NeuroVest Trading API",
  "version": "1.0.0",
  "status": "online",
  "docs": "/docs"
}
```

#### `GET /health`
Health check

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-16T01:00:00"
}
```

---

### Portfolio Management

#### `GET /portfolio/status`
Get current portfolio status

**Response:**
```json
{
  "total_value": 154592.50,
  "cash": 50000.00,
  "stock_value": 85000.00,
  "crypto_value": 19592.50,
  "positions": 5,
  "daily_pnl": 2341.20,
  "daily_pnl_pct": 1.54,
  "total_pnl": 54592.50,
  "total_pnl_pct": 54.59,
  "updated_at": "2025-11-16T01:00:00"
}
```

#### `GET /portfolio/positions`
Get all open positions

**Response:**
```json
[
  {
    "ticker": "SPY",
    "asset_type": "stock",
    "quantity": 100,
    "entry_price": 450.00,
    "current_price": 465.50,
    "market_value": 46550.00,
    "unrealized_pnl": 1550.00,
    "unrealized_pnl_pct": 3.44,
    "entry_time": "2025-11-11T09:30:00",
    "days_held": 5
  }
]
```

#### `GET /portfolio/performance`
Get performance metrics

**Response:**
```json
{
  "total_return": 0.5459,
  "annualized_return": 0.2122,
  "sharpe_ratio": 7.47,
  "max_drawdown": -0.2740,
  "win_rate": 0.535,
  "total_trades": 475,
  "avg_profit_per_trade": 0.0347,
  "best_trade": 0.1250,
  "worst_trade": -0.0850
}
```

---

### Trading

#### `GET /signals`
Get current trading signals

**Query Parameters:**
- `asset_type` (optional): "stock" or "crypto"
- `min_confidence` (optional): Minimum confidence threshold (default: 0.60)

**Response:**
```json
[
  {
    "ticker": "SPY",
    "asset_type": "stock",
    "signal": "buy",
    "confidence": 0.87,
    "entry_price": 465.50,
    "target_price": 485.00,
    "stop_loss": 450.00,
    "position_size": 15000.00,
    "reason": "Strong bullish momentum, high model confidence",
    "timestamp": "2025-11-16T01:00:00"
  }
]
```

**Example:**
```bash
# Get all signals
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:8000/signals

# Get stock signals only with 75% minimum confidence
curl -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/signals?asset_type=stock&min_confidence=0.75"
```

#### `POST /trade`
Execute a trade

**Query Parameters:**
- `ticker` (required): Asset ticker
- `action` (required): "buy" or "sell"
- `quantity` (required): Number of shares/units
- `mode` (optional): "paper" (default) or "live"

**Response:**
```json
{
  "ticker": "SPY",
  "asset_type": "stock",
  "action": "buy",
  "quantity": 10,
  "price": 465.50,
  "total_value": 4655.00,
  "timestamp": "2025-11-16T01:00:00",
  "mode": "paper",
  "status": "executed"
}
```

**Example:**
```bash
# Paper trade (safe testing)
curl -X POST -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/trade?ticker=SPY&action=buy&quantity=10"

# Live trade (real money)
curl -X POST -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/trade?ticker=SPY&action=buy&quantity=10&mode=live"
```

---

### Risk Management

#### `GET /risk/profiles`
List all risk profiles

**Response:**
```json
{
  "profiles": ["conservative", "moderate", "aggressive"],
  "active": "moderate"
}
```

#### `GET /risk/profile/{profile_name}`
Get specific risk profile

**Response:**
```json
{
  "name": "moderate",
  "stock_allocation": 0.70,
  "crypto_allocation": 0.30,
  "max_position_size": 0.20,
  "max_positions": 5,
  "max_leverage_stocks": 1.5,
  "max_leverage_crypto": 2.0,
  "stop_loss_pct_stocks": 0.04,
  "stop_loss_pct_crypto": 0.08,
  "max_daily_loss": 0.025,
  "max_drawdown": 0.30,
  "min_confidence_stocks": 0.60,
  "min_confidence_crypto": 0.65,
  "holding_period_days_stocks": 7,
  "holding_period_days_crypto": 5
}
```

#### `POST /risk/profile/active`
Set active risk profile

**Query Parameters:**
- `profile_name` (required): Profile name

**Response:**
```json
{
  "message": "Active profile set to moderate",
  "profile": { ... }
}
```

**Example:**
```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/risk/profile/active?profile_name=aggressive"
```

#### `POST /risk/profile/create`
Create custom risk profile

**Request Body:**
```json
{
  "name": "my_custom",
  "stock_allocation": 0.65,
  "crypto_allocation": 0.35,
  "max_leverage_crypto": 2.5,
  "stop_loss_pct_crypto": 0.09
}
```

**Response:**
```json
{
  "message": "Created custom profile: my_custom",
  "profile": { ... }
}
```

**Example:**
```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name":"my_custom","stock_allocation":0.65,"crypto_allocation":0.35}' \
     http://localhost:8000/risk/profile/create
```

---

### Market Data

#### `GET /data/{ticker}`
Get market data for a ticker

**Query Parameters:**
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Response:**
```json
{
  "ticker": "SPY",
  "records": 1000,
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2025-11-16T00:00:00",
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "Open": 460.00,
      "High": 465.00,
      "Low": 458.00,
      "Close": 463.50,
      "Volume": 50000000,
      "Adj_Close": 463.50
    }
  ]
}
```

**Example:**
```bash
# Get all data
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:8000/data/SPY

# Get specific date range
curl -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/data/SPY?start_date=2024-01-01&end_date=2024-12-31"
```

#### `GET /data/stats`
Get database statistics

**Response:**
```json
{
  "total_assets": 16,
  "total_records": 45823,
  "db_size_mb": 12.4,
  "cache_hits": 1234,
  "cache_misses": 56,
  "cache_hit_rate": 95.67,
  "cache_size": 12
}
```

#### `POST /data/update`
Trigger data update

**Query Parameters:**
- `ticker` (optional): Specific ticker to update (default: all)

**Response:**
```json
{
  "message": "Data update triggered",
  "ticker": "SPY",
  "status": "processing"
}
```

---

### WebSocket (Real-Time)

#### `WS /ws/live`
WebSocket for real-time updates

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

**Message Format:**
```json
{
  "type": "portfolio_update",
  "data": {
    "total_value": 154592.50,
    "timestamp": "2025-11-16T01:00:00"
  }
}
```

---

## Python Client Example

```python
import requests

class NeuroVestClient:
    def __init__(self, base_url="http://localhost:8000", token="demo_token"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def get_portfolio_status(self):
        """Get portfolio status"""
        response = requests.get(
            f"{self.base_url}/portfolio/status",
            headers=self.headers
        )
        return response.json()

    def get_signals(self, asset_type=None, min_confidence=0.60):
        """Get trading signals"""
        params = {"min_confidence": min_confidence}
        if asset_type:
            params["asset_type"] = asset_type

        response = requests.get(
            f"{self.base_url}/signals",
            headers=self.headers,
            params=params
        )
        return response.json()

    def execute_trade(self, ticker, action, quantity, mode="paper"):
        """Execute a trade"""
        params = {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "mode": mode
        }
        response = requests.post(
            f"{self.base_url}/trade",
            headers=self.headers,
            params=params
        )
        return response.json()

    def set_risk_profile(self, profile_name):
        """Set active risk profile"""
        response = requests.post(
            f"{self.base_url}/risk/profile/active",
            headers=self.headers,
            params={"profile_name": profile_name}
        )
        return response.json()


# Usage
client = NeuroVestClient()

# Get portfolio status
portfolio = client.get_portfolio_status()
print(f"Total value: ${portfolio['total_value']:,.2f}")

# Get signals
signals = client.get_signals(asset_type="stock", min_confidence=0.75)
for signal in signals:
    print(f"{signal['ticker']}: {signal['signal']} (confidence: {signal['confidence']*100:.0f}%)")

# Execute paper trade
trade = client.execute_trade("SPY", "buy", 10, mode="paper")
print(f"Trade: {trade['status']}")

# Set risk profile
result = client.set_risk_profile("aggressive")
print(f"Risk profile: {result['message']}")
```

---

## JavaScript Client Example

```javascript
class NeuroVestClient {
  constructor(baseUrl = 'http://localhost:8000', token = 'demo_token') {
    this.baseUrl = baseUrl;
    this.headers = { 'Authorization': `Bearer ${token}` };
  }

  async getPortfolioStatus() {
    const response = await fetch(`${this.baseUrl}/portfolio/status`, {
      headers: this.headers
    });
    return await response.json();
  }

  async getSignals(assetType = null, minConfidence = 0.60) {
    const params = new URLSearchParams({ min_confidence: minConfidence });
    if (assetType) params.append('asset_type', assetType);

    const response = await fetch(`${this.baseUrl}/signals?${params}`, {
      headers: this.headers
    });
    return await response.json();
  }

  async executeTrade(ticker, action, quantity, mode = 'paper') {
    const params = new URLSearchParams({
      ticker, action, quantity, mode
    });

    const response = await fetch(`${this.baseUrl}/trade?${params}`, {
      method: 'POST',
      headers: this.headers
    });
    return await response.json();
  }

  connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/ws/live');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Update:', data);
    };

    return ws;
  }
}

// Usage
const client = new NeuroVestClient();

// Get portfolio
client.getPortfolioStatus().then(portfolio => {
  console.log(`Total value: $${portfolio.total_value}`);
});

// Get signals
client.getSignals('stock', 0.75).then(signals => {
  signals.forEach(signal => {
    console.log(`${signal.ticker}: ${signal.signal}`);
  });
});

// WebSocket connection
const ws = client.connectWebSocket();
```

---

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn

COPY . .

CMD ["uvicorn", "api.trading_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t neurovest-api .
docker run -p 8000:8000 neurovest-api
```

### Using systemd

Create `/etc/systemd/system/neurovest-api.service`:

```ini
[Unit]
Description=NeuroVest Trading API
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/NeuroVest
ExecStart=/usr/bin/python3 -m uvicorn api.trading_api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable neurovest-api
sudo systemctl start neurovest-api
```

### Using nginx reverse proxy

```nginx
server {
    listen 80;
    server_name api.neurovest.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Security Best Practices

### 1. Replace Demo Token

**Never use the demo token in production!**

```python
# Use proper JWT tokens
from fastapi_jwt_auth import AuthJWT

@app.post('/login')
def login(username: str, password: str, Authorize: AuthJWT = Depends()):
    # Validate credentials
    if username == "user" and password == "pass":
        access_token = Authorize.create_access_token(subject=username)
        return {"access_token": access_token}
```

### 2. Enable HTTPS

```python
# Use SSL certificates
uvicorn.run(app,
    host="0.0.0.0",
    port=443,
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem"
)
```

### 3. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/signals")
@limiter.limit("10/minute")
async def get_signals():
    ...
```

### 4. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (invalid token) |
| 404 | Not Found (ticker/profile doesn't exist) |
| 500 | Internal Server Error |

**Error Response Format:**
```json
{
  "detail": "Error message here"
}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| /signals | 10 req/min |
| /trade | 5 req/min |
| /portfolio/* | 60 req/min |
| /data/* | 30 req/min |

---

## Support

- **Documentation**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json
- **Issues**: GitHub repository

---

**Last Updated**: 2025-11-16
**API Version**: 1.0.0
