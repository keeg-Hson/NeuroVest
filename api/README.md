# NeuroVest Trading API

REST API for programmatic access to the trading system.

---

## Quick Start

### 1. Install API Dependencies

```bash
pip install fastapi uvicorn pydantic websockets
```

### 2. Start Server

```bash
# Development
python api/trading_api.py

# Production
uvicorn api.trading_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json
- **Full Guide**: [docs/API_GUIDE.md](../docs/API_GUIDE.md)

---

## Features

- **Portfolio Management**: Real-time portfolio status and positions
- **Trading Signals**: ML-powered buy/sell signals
- **Trade Execution**: Paper and live trading
- **Risk Management**: Configurable risk profiles
- **Market Data**: Historical and real-time data access
- **WebSocket**: Real-time updates
- **Authentication**: Token-based security

---

## Example Usage

### Python

```python
from examples.api_client import NeuroVestClient

client = NeuroVestClient()

# Get portfolio status
portfolio = client.get_portfolio_status()
print(f"Total value: ${portfolio['total_value']:,.2f}")

# Get trading signals
signals = client.get_signals(min_confidence=0.75)
for signal in signals:
    print(f"{signal['ticker']}: {signal['signal']}")

# Execute paper trade
trade = client.execute_trade("SPY", "buy", 10, mode="paper")
```

### cURL

```bash
# Get signals
curl -H "Authorization: Bearer demo_token_replace_in_production" \
     http://localhost:8000/signals

# Execute trade
curl -X POST \
     -H "Authorization: Bearer demo_token_replace_in_production" \
     "http://localhost:8000/trade?ticker=SPY&action=buy&quantity=10"
```

---

## Authentication

All endpoints require Bearer token authentication:

```
Authorization: Bearer YOUR_TOKEN_HERE
```

**Default demo token**: `demo_token_replace_in_production`

**⚠️ Replace in production with proper JWT tokens!**

---

## Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/status` | GET | Portfolio overview |
| `/portfolio/positions` | GET | Open positions |
| `/signals` | GET | Trading signals |
| `/trade` | POST | Execute trade |
| `/risk/profiles` | GET | List risk profiles |
| `/risk/profile/active` | POST | Set active profile |
| `/data/{ticker}` | GET | Market data |

See [docs/API_GUIDE.md](../docs/API_GUIDE.md) for complete documentation.

---

## WebSocket

Real-time updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

---

## Production Deployment

### Docker

```bash
docker build -t neurovest-api .
docker run -p 8000:8000 neurovest-api
```

### systemd

```bash
sudo systemctl enable neurovest-api
sudo systemctl start neurovest-api
```

See [docs/API_GUIDE.md](../docs/API_GUIDE.md) for deployment guides.

---

## Security

### Production Checklist

- [ ] Replace demo token with JWT
- [ ] Enable HTTPS
- [ ] Add rate limiting
- [ ] Configure CORS
- [ ] Use environment variables
- [ ] Enable request logging
- [ ] Set up monitoring

---

## Examples

Run example client:

```bash
# Make sure API server is running first
python api/trading_api.py

# In another terminal
python examples/api_client.py
```

---

## Documentation

- **Full API Guide**: [docs/API_GUIDE.md](../docs/API_GUIDE.md)
- **Risk Management**: [core/risk_manager.py](../core/risk_manager.py)
- **Example Client**: [examples/api_client.py](../examples/api_client.py)

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
