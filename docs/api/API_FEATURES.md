# NeuroVest API - Advanced Features

## Overview

This document covers the advanced production features implemented for the NeuroVest Forecasting API:

1. **Rate Limiting** - Prevent abuse with tier-based request limits
2. **Premium Tiers** - Subscription levels with different rate limits
3. **Redis Caching** - Faster responses with 5-minute cache TTL
4. **CLI API Key Generator** - Easy customer onboarding tool

---

## 1. Rate Limiting

### Implementation

- **Library:** `slowapi` (Redis-backed rate limiter for FastAPI)
- **Strategy:** Tier-based rate limits with automatic enforcement
- **Enforcement:** Per IP address + API key validation

### Rate Limits by Tier

| Tier       | Requests/Min | Monthly Cost | Use Case                    |
|------------|--------------|--------------|----------------------------|
| Free       | 10           | $0           | Personal testing, demos    |
| Individual | 60           | $49          | Individual traders         |
| Pro        | 300          | $199         | Professional traders       |
| Enterprise | 10,000       | $999+        | Institutional clients      |

### Example Response (Rate Limit Exceeded)

```json
{
  "detail": "Rate limit exceeded: 10 per 1 minute",
  "timestamp": "2026-01-05T15:30:00Z"
}
```

### Headers

Rate limit info is included in response headers:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1704470400
```

### Protected Endpoints

Rate limiting applies to:
- `/api/predictions` - All predictions
- `/api/predictions/{ticker}` - Single asset
- `/api/predictions/{ticker}/history` - Historical data
- `/api/predictions/batch` - Batch requests

### Public Endpoints (No Limit)

- `/` - API info
- `/health` - Health check
- `/cache/stats` - Cache statistics
- `/docs` - API documentation

---

## 2. Premium Tiers

### Database Schema

Added `tier` column to `users` table:

```sql
ALTER TABLE users
ADD COLUMN tier VARCHAR(20) DEFAULT 'free'
CHECK (tier IN ('free', 'individual', 'pro', 'enterprise'));
```

### Tier Assignment

Set tier when creating users:

```bash
# Create free tier user (default)
python3 create_api_key.py --username "john_doe"

# Create pro tier user
python3 create_api_key.py --username "premium_customer" --tier pro
```

### Upgrade Existing Users

```bash
# Upgrade user to pro tier
python3 create_api_key.py --update "john_doe" --tier pro
```

### API Response

User tier is included in authentication logs:

```
INFO: Authenticated user: 42 (tier: pro)
```

---

## 3. Redis Caching

### Implementation

- **Library:** `redis` (Python Redis client)
- **Strategy:** Cache predictions with 5-minute TTL
- **Fallback:** Gracefully degrades to database-only if Redis unavailable

### Cache Keys

```
prediction:{TICKER}           # Single asset prediction
predictions:all:{LIMIT}       # All predictions with limit
```

### TTL (Time To Live)

- **Predictions:** 5 minutes (300 seconds)
- **Rationale:** Balance freshness vs performance

### Cache Statistics Endpoint

```bash
curl https://your-api.railway.app/cache/stats
```

**Response:**
```json
{
  "enabled": true,
  "total_commands": 15234,
  "keyspace_hits": 12891,
  "keyspace_misses": 2343,
  "hit_rate": 84.63
}
```

### Performance Improvement

**Without cache:**
- Average response time: 200-500ms
- Database queries per request: 1-2

**With cache (hit):**
- Average response time: 5-15ms
- Database queries per request: 0

**Speed improvement:** ~20-40x faster on cache hits

### Cache Invalidation

Automatic invalidation when new predictions are generated:

```python
from cache_manager import cache
cache.invalidate_predictions()  # Clear all prediction:* keys
```

### Environment Setup

Set `REDIS_URL` environment variable:

```bash
# Local development
export REDIS_URL=redis://localhost:6379

# Railway (automatic with Redis plugin)
REDIS_URL=redis://default:password@containers-us-west-xyz.railway.app:6379
```

### Graceful Degradation

If Redis is unavailable:
- API continues to work (slower)
- Logs warning: `⚠️ Redis unavailable - caching disabled`
- Falls back to direct database queries

---

## 4. CLI API Key Generator

### Features

- Create new API keys with custom tier
- List all users with tier information
- Get details for specific user
- Update user tier
- Formatted table output with statistics

### Usage

#### Create Free Tier User (Default)

```bash
python3 create_api_key.py --username "john_doe"
```

**Output:**
```
======================================================================
Creating API Key for: john_doe
Tier: free
======================================================================

✅ API Key Created Successfully!

User ID:  42
Username: john_doe
Tier:     free

──────────────────────────────────────────────────────────────────────
API Key (save this - it won't be shown again!):
──────────────────────────────────────────────────────────────────────

xYz123AbC...

──────────────────────────────────────────────────────────────────────

Rate Limit: 10 requests/minute
```

#### Create Pro Tier User

```bash
python3 create_api_key.py --username "premium_customer" --tier pro
```

#### List All Users

```bash
python3 create_api_key.py --list
```

**Output:**
```
====================================================================================================
API USERS
====================================================================================================

+----+------------------+--------+----------------------+---------------------+
| ID | Username         | Tier   | API Key Preview      | Created             |
+====+==================+========+======================+=====================+
| 42 | john_doe         | free   | xYz123AbC...         | 2026-01-05 10:30:00 |
| 41 | premium_customer | pro    | aBc456XyZ...         | 2026-01-05 09:15:00 |
| 40 | demo             | free   | qWe789RtY...         | 2026-01-04 14:00:00 |
+----+------------------+--------+----------------------+---------------------+

────────────────────────────────────────────────────────────────────────────────────────────────────
TIER DISTRIBUTION:
   free        :  25 users
   pro         :   8 users
   individual  :   3 users
   enterprise  :   1 user
────────────────────────────────────────────────────────────────────────────────────────────────────
```

#### Get User Details

```bash
python3 create_api_key.py --get "john_doe"
```

#### Update User Tier

```bash
python3 create_api_key.py --update "john_doe" --tier pro
```

**Output:**
```
✅ Updated john_doe to tier: pro
```

---

## Migration Guide

### 1. Update Database

Run the tier migration:

```bash
psql $DATABASE_URL -f migrations/add_user_tiers.sql
```

### 2. Install Dependencies

```bash
pip install slowapi redis tabulate
```

Or update from requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

**Required:**
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
```

**Optional (for caching):**
```bash
REDIS_URL=redis://localhost:6379
```

### 4. Deploy Updated API

Railway/Docker deployment:

```bash
# Rebuild and deploy
git push origin main

# Or redeploy Railway service
railway up
```

### 5. Create First API Keys

```bash
# Create demo user
python3 create_api_key.py --username "demo" --tier free

# Create test pro user
python3 create_api_key.py --username "test_pro" --tier pro
```

---

## Testing

### Test Rate Limiting

```bash
# Make 11 requests in quick succession (should hit limit on 11th)
for i in {1..11}; do
  curl -H "X-API-Key: YOUR_KEY" https://api.railway.app/api/predictions
  echo "\nRequest $i"
  sleep 0.1
done
```

**Expected:** First 10 succeed, 11th returns 429 (Rate Limit Exceeded)

### Test Cache Performance

```bash
# First request (cache miss - slow)
time curl -H "X-API-Key: YOUR_KEY" https://api.railway.app/api/predictions/SPY

# Second request (cache hit - fast!)
time curl -H "X-API-Key: YOUR_KEY" https://api.railway.app/api/predictions/SPY
```

**Expected:** Second request ~20x faster

### Test Cache Stats

```bash
curl https://api.railway.app/cache/stats
```

### Test Tier Assignment

```bash
# Create users with different tiers
python3 create_api_key.py --username "free_user" --tier free
python3 create_api_key.py --username "pro_user" --tier pro

# Verify rate limits differ
# (free_user gets 10/min, pro_user gets 300/min)
```

---

## Monitoring

### Key Metrics to Track

1. **Rate Limit Hits**
   - Monitor 429 responses
   - Identify users hitting limits frequently
   - Consider tier upgrades

2. **Cache Hit Rate**
   - Target: >80% hit rate
   - Check `/cache/stats` endpoint
   - Low hit rate = increase TTL or investigate

3. **Response Times**
   - With cache: 5-15ms
   - Without cache: 200-500ms
   - Slow responses = check database

4. **Tier Distribution**
   - Track user growth by tier
   - Revenue = (individual×$49) + (pro×$199) + (enterprise×custom)

### Logging

All features log detailed information:

```
INFO: Authenticated user: 42 (tier: pro)
INFO: User 42 requesting prediction for SPY
INFO: Returning cached prediction for SPY  # Cache hit!
```

---

## Cost Analysis

### Infrastructure Costs (Estimated)

| Service      | Free Tier      | Paid Tier        | Recommended        |
|--------------|----------------|------------------|--------------------|
| PostgreSQL   | 100 connections| Unlimited        | $7-20/month        |
| Redis        | 25 MB          | 100 MB - 5 GB    | $5-15/month        |
| API Server   | 512 MB RAM     | 1-2 GB RAM       | $10-25/month       |
| **Total**    | **$0**         | **$22-60/month** | **Scale with users**|

### Revenue Potential

With 100 users:
- 70 Free: $0
- 20 Individual: $980/month
- 8 Pro: $1,592/month
- 2 Enterprise: $2,000/month

**Total: ~$4,500/month revenue vs $50/month costs = 90x profit margin**

---

## Roadmap (Future Features)

### Short Term (1-2 weeks)
- ✅ Rate limiting (DONE)
- ✅ Premium tiers (DONE)
- ✅ Redis caching (DONE)
- ✅ CLI key generator (DONE)
- ⏱️ WebSocket real-time updates
- ⏱️ Usage analytics dashboard

### Medium Term (1-2 months)
- API usage metrics per user
- Billing integration (Stripe)
- Email notifications for limit warnings
- Custom rate limits per user
- API key rotation
- Webhook support for predictions

### Long Term (3-6 months)
- GraphQL API option
- Multi-region deployment
- Custom model training API
- Backtesting API endpoints
- Portfolio optimization endpoints

---

## Support

For questions or issues:

1. Check logs: `railway logs` or Docker logs
2. Test health: `curl https://api.railway.app/health`
3. Check cache: `curl https://api.railway.app/cache/stats`
4. Review docs: `https://api.railway.app/docs`

## License

Proprietary - NeuroVest API
