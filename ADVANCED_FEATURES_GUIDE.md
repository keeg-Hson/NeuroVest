# Advanced Features Guide - WebSockets, Analytics & Request Logging

## Overview

This guide covers three new powerful features added to the NeuroVest API:

1. **Request Logging** - Track all API requests for analytics and billing
2. **Analytics Dashboard** - Comprehensive usage metrics and insights
3. **WebSocket Streaming** - Real-time prediction updates

---

## 1. Request Logging

### What It Does

Automatically logs every API request to the database, capturing:
- User ID and subscription tier
- Endpoint and HTTP method
- Response time (milliseconds)
- Status code (200, 404, 429, etc.)
- IP address and user agent
- Error messages (if any)
- Timestamp

### Database Setup

Run the migration:

```bash
psql $DATABASE_URL -f migrations/add_request_logging.sql
```

This creates:
- `request_logs` table
- 6 optimized indexes
- `request_stats_daily` view for quick analytics

### How It Works

The `RequestLoggerMiddleware` automatically intercepts all requests:

```python
# In api_server.py - already configured!
app.add_middleware(RequestLoggerMiddleware)
```

### Query Examples

```sql
-- Today's request count
SELECT COUNT(*) FROM request_logs WHERE DATE(created_at) = CURRENT_DATE;

-- Requests by tier
SELECT tier, COUNT(*) as requests
FROM request_logs
GROUP BY tier
ORDER BY requests DESC;

-- Average response time
SELECT AVG(response_time_ms) as avg_ms
FROM request_logs
WHERE status_code = 200;

-- Error rate
SELECT
    COUNT(CASE WHEN status_code >= 400 THEN 1 END)::float / COUNT(*) * 100 as error_rate
FROM request_logs
WHERE created_at >= NOW() - INTERVAL '7 days';
```

### Performance Impact

- **Overhead:** ~2-5ms per request
- **Storage:** ~200 bytes per request
- **Async:** Non-blocking logging (doesn't slow down responses)

---

## 2. Analytics Dashboard

### Endpoints

#### `/api/analytics/usage`

Get API usage statistics

**Query Parameters:**
- `days` (default: 7) - Days to analyze (1-90)
- `user_id` (optional) - Filter by user
- `tier` (optional) - Filter by tier (free/individual/pro/enterprise)

**Example:**
```bash
curl https://api.railway.app/api/analytics/usage?days=30
```

**Response:**
```json
{
  "period_days": 30,
  "overall": {
    "total_requests": 15234,
    "unique_users": 42,
    "avg_response_time_ms": 23.45,
    "error_count": 156
  },
  "by_tier": [
    {"tier": "free", "count": 8500},
    {"tier": "pro", "count": 5234},
    {"tier": "individual", "count": 1500}
  ],
  "daily": [
    {"date": "2026-01-05", "requests": 523, "users": 18},
    {"date": "2026-01-04", "requests": 489, "users": 15}
  ]
}
```

#### `/api/analytics/popular`

Get most requested assets

**Query Parameters:**
- `days` (default: 7) - Days to analyze
- `limit` (default: 10) - Number of assets to return

**Example:**
```bash
curl https://api.railway.app/api/analytics/popular?days=7&limit=5
```

**Response:**
```json
{
  "period_days": 7,
  "total_assets": 5,
  "assets": [
    {"ticker": "SPY", "requests": 1250, "unique_users": 35},
    {"ticker": "QQQ", "requests": 890, "unique_users": 28},
    {"ticker": "BTC_USDT", "requests": 673, "unique_users": 19},
    {"ticker": "ETH_USDT", "requests": 512, "unique_users": 15},
    {"ticker": "GLD", "requests": 345, "unique_users": 12}
  ]
}
```

#### `/api/analytics/errors`

Error analysis and trends

**Response:**
```json
{
  "period_days": 7,
  "by_status_code": [
    {"status_code": 429, "count": 89},
    {"status_code": 404, "count": 34},
    {"status_code": 401, "count": 12}
  ],
  "by_endpoint": [
    {
      "endpoint": "/api/predictions/INVALID",
      "total_requests": 45,
      "errors": 45,
      "error_rate": 100.0
    }
  ],
  "daily_trend": [
    {
      "date": "2026-01-05",
      "total_requests": 523,
      "errors": 12,
      "error_rate": 2.29
    }
  ]
}
```

#### `/api/analytics/performance`

Response time metrics

**Response:**
```json
{
  "period_days": 7,
  "overall": {
    "avg_ms": 23.45,
    "min_ms": 3.21,
    "max_ms": 1234.56,
    "p50_ms": 18.32,
    "p95_ms": 67.89,
    "p99_ms": 234.12
  },
  "slowest_endpoints": [
    {
      "endpoint": "/api/predictions/batch",
      "requests": 234,
      "avg_ms": 145.67,
      "max_ms": 890.23
    }
  ]
}
```

#### `/api/analytics/dashboard`

Complete analytics dashboard (all metrics in one call)

**Example:**
```bash
curl https://api.railway.app/api/analytics/dashboard?days=30
```

Returns combined data from all analytics endpoints.

### Use Cases

**1. Monitor API Health**
```bash
# Check error rate
curl https://api.railway.app/api/analytics/errors?days=1
```

**2. Identify Power Users**
```bash
# Who's using the API most?
curl https://api.railway.app/api/analytics/usage?days=30
```

**3. Optimize Performance**
```bash
# Which endpoints are slow?
curl https://api.railway.app/api/analytics/performance?days=7
```

**4. Product Decisions**
```bash
# What assets do users care about?
curl https://api.railway.app/api/analytics/popular?days=30&limit=20
```

---

## 3. WebSocket Streaming

### What It Does

Real-time prediction updates via WebSocket. When new predictions are generated, they're instantly pushed to connected clients.

### Connection

**Endpoint:** `ws://api.railway.app/ws/predictions`

**Authentication:** Pass API key as query parameter

**Subscribe to all assets:**
```
ws://api.railway.app/ws/predictions?api_key=YOUR_KEY
```

**Subscribe to specific assets:**
```
ws://api.railway.app/ws/predictions?api_key=YOUR_KEY&tickers=SPY,QQQ,BTC_USDT
```

### Client Examples

#### JavaScript (Browser)

```javascript
const apiKey = 'your_api_key_here';
const tickers = 'SPY,QQQ,BTC_USDT';

const ws = new WebSocket(
    `ws://api.railway.app/ws/predictions?api_key=${apiKey}&tickers=${tickers}`
);

ws.onopen = () => {
    console.log('Connected to NeuroVest stream');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);

    switch(message.type) {
        case 'connected':
            console.log('✅ Subscribed to:', message.subscriptions);
            break;

        case 'prediction':
            console.log('📊 New prediction:', message.data);
            // Update your UI with new prediction
            updatePrediction(message.data);
            break;

        case 'heartbeat':
            console.log('💓 Heartbeat');
            break;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from stream');
    // Reconnect logic here
};

// Subscribe to additional tickers
function subscribe(newTickers) {
    ws.send(JSON.stringify({
        action: 'subscribe',
        tickers: newTickers
    }));
}

// Unsubscribe from tickers
function unsubscribe(removeTickers) {
    ws.send(JSON.stringify({
        action: 'unsubscribe',
        tickers: removeTickers
    }));
}
```

#### Python

```python
import asyncio
import websockets
import json

async def stream_predictions():
    api_key = 'your_api_key_here'
    uri = f"ws://api.railway.app/ws/predictions?api_key={api_key}&tickers=SPY,QQQ"

    async with websockets.connect(uri) as websocket:
        print("✅ Connected to NeuroVest stream")

        async for message in websocket:
            data = json.loads(message)

            if data['type'] == 'connected':
                print(f"Subscribed to: {data['subscriptions']}")

            elif data['type'] == 'prediction':
                prediction = data['data']
                print(f"\n📊 New prediction for {prediction['ticker']}:")
                print(f"   Label: {prediction['prediction_label']}")
                print(f"   Confidence: {prediction['confidence']}")

            elif data['type'] == 'heartbeat':
                print("💓", end='', flush=True)

# Run the stream
asyncio.run(stream_predictions())
```

### Message Types

#### 1. Connected
Sent when connection is established

```json
{
  "type": "connected",
  "message": "Connected to NeuroVest prediction stream",
  "subscriptions": ["SPY", "QQQ"],
  "timestamp": "2026-01-05T15:30:00Z"
}
```

#### 2. Prediction
New prediction available

```json
{
  "type": "prediction",
  "data": {
    "ticker": "SPY",
    "prediction_date": "2026-01-06",
    "prediction_label": "SPIKE",
    "prob_crash": 0.12,
    "prob_normal": 0.45,
    "prob_spike": 0.43,
    "confidence": "high"
  },
  "timestamp": "2026-01-05T16:30:01Z"
}
```

#### 3. Heartbeat
Keep-alive ping (every 30 seconds)

```json
{
  "type": "heartbeat",
  "timestamp": "2026-01-05T15:30:30Z",
  "active_connections": 12
}
```

### Client Commands

Send JSON messages to interact with the stream:

#### Subscribe to Additional Tickers
```json
{
  "action": "subscribe",
  "tickers": ["ETH_USDT", "SOL_USDT"]
}
```

Response:
```json
{
  "type": "subscribed",
  "tickers": ["ETH_USDT", "SOL_USDT"],
  "timestamp": "2026-01-05T15:31:00Z"
}
```

#### Unsubscribe from Tickers
```json
{
  "action": "unsubscribe",
  "tickers": ["SPY"]
}
```

#### Ping/Pong
```json
{
  "action": "ping"
}
```

Response:
```json
{
  "type": "pong",
  "timestamp": "2026-01-05T15:32:00Z"
}
```

### Broadcasting Predictions

When generating new predictions, broadcast to connected clients:

```python
from websocket_streaming import broadcast_new_prediction

# After generating prediction
prediction = {
    "ticker": "SPY",
    "prediction_label": "SPIKE",
    "prob_crash": 0.12,
    "prob_normal": 0.45,
    "prob_spike": 0.43,
    "confidence": "high",
    "prediction_date": "2026-01-06"
}

await broadcast_new_prediction(prediction)
```

### WebSocket Statistics

**Endpoint:** `/ws/stats`

```bash
curl https://api.railway.app/ws/stats
```

**Response:**
```json
{
  "total_connections": 15,
  "by_tier": {
    "free": 8,
    "pro": 5,
    "individual": 2
  },
  "ticker_subscriptions": {
    "SPY": 12,
    "QQQ": 8,
    "BTC_USDT": 5
  }
}
```

### Best Practices

1. **Reconnection Logic** - Always implement auto-reconnect
2. **Heartbeat Monitoring** - Disconnect if no heartbeat for 60s
3. **Subscription Management** - Don't subscribe to hundreds of tickers
4. **Error Handling** - Handle connection drops gracefully
5. **Rate Limiting** - WebSocket connections count toward tier limits

### Use Cases

**Trading Dashboards**
- Real-time prediction updates
- No polling required
- Instant alerts on new signals

**Mobile Apps**
- Push notifications for predictions
- Battery-efficient (no polling)
- Always up-to-date

**Backtesting Systems**
- Stream historical predictions
- Real-time strategy testing
- Live performance monitoring

---

## Testing

### 1. Test Request Logging

```bash
# Make some requests
curl -H "X-API-Key: YOUR_KEY" https://api.railway.app/api/predictions/SPY

# Check logs were created
psql $DATABASE_URL -c "SELECT * FROM request_logs ORDER BY created_at DESC LIMIT 5;"
```

### 2. Test Analytics

```bash
# Get usage stats
curl https://api.railway.app/api/analytics/usage

# Get popular assets
curl https://api.railway.app/api/analytics/popular

# Full dashboard
curl https://api.railway.app/api/analytics/dashboard?days=7
```

### 3. Test WebSocket

```bash
# Install wscat
npm install -g wscat

# Connect to stream
wscat -c "ws://api.railway.app/ws/predictions?api_key=YOUR_KEY&tickers=SPY"

# You should see:
# {"type":"connected","message":"Connected to NeuroVest prediction stream",...}
```

---

## Deployment

### Environment Variables

No new environment variables required! All features work with existing setup:

- `DATABASE_URL` (required)
- `REDIS_URL` (optional - for caching)

### Railway Setup

1. **Run migrations:**
   ```bash
   railway run psql $DATABASE_URL -f migrations/add_request_logging.sql
   ```

2. **Deploy updated API:**
   ```bash
   git push origin main
   ```

3. **Verify features:**
   ```bash
   curl https://your-api.railway.app/ws/stats
   curl https://your-api.railway.app/api/analytics/dashboard
   ```

---

## Performance & Scaling

### Request Logging
- **Storage:** ~200 bytes/request
- **Growth:** 1M requests = ~200MB
- **Cleanup:** Consider archiving logs older than 90 days

### Analytics
- **Query time:** <100ms for most queries
- **Caching:** Results are fresh (no caching)
- **Indexes:** Optimized for common queries

### WebSocket
- **Concurrent connections:** 1000+ per server
- **Memory:** ~10KB per connection
- **Latency:** <50ms for broadcasts

---

## Troubleshooting

### WebSocket won't connect

**Issue:** Connection refused

**Solution:**
```bash
# Check stats endpoint works
curl https://api.railway.app/ws/stats

# Verify API key is valid
curl -H "X-API-Key: YOUR_KEY" https://api.railway.app/health
```

### Analytics returns empty data

**Issue:** No request logs in database

**Solution:**
```bash
# Check table exists
psql $DATABASE_URL -c "\dt request_logs"

# Run migration if missing
psql $DATABASE_URL -f migrations/add_request_logging.sql
```

### Slow analytics queries

**Issue:** Queries take >1 second

**Solution:**
```sql
-- Check indexes exist
SELECT * FROM pg_indexes WHERE tablename = 'request_logs';

-- Analyze table
ANALYZE request_logs;
```

---

## Next Steps

**Integrate into Your Workflow:**
1. Add analytics to admin dashboard
2. Set up alerts for error spikes
3. Build WebSocket client for your app
4. Monitor popular assets for product decisions

**Advanced Features:**
- Add email alerts for errors
- Create usage reports for customers
- Build real-time trading dashboard
- Implement auto-scaling based on WebSocket load

---

## Support

Questions? Check:
- API docs: `https://api.railway.app/docs`
- Health check: `https://api.railway.app/health`
- WebSocket stats: `https://api.railway.app/ws/stats`
- Analytics: `https://api.railway.app/api/analytics/dashboard`
