# NeuroVest REST API Deployment Guide

**Fast Track: Get API Live in 30 Minutes**

---

## What You're Deploying

A standalone FastAPI REST API that provides:
- ✅ GET /health - System health check
- ✅ GET /api/predictions - All latest predictions
- ✅ GET /api/predictions/{ticker} - Specific asset prediction
- ✅ GET /api/assets - List available assets
- ✅ POST /api/auth/register - Create user and get API key

**Authentication:** X-API-Key header required
**Documentation:** Auto-generated Swagger UI at `/docs`

---

## Prerequisites

✅ Railway account (already have PostgreSQL running)
✅ NeuroVest repo with latest code
✅ DATABASE_URL from your PostgreSQL service

---

## Step 1: Test Locally (Optional but Recommended)

### 1.1 Install Dependencies

```bash
cd /path/to/NeuroVest
pip install -r requirements.txt
```

This installs:
- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- pydantic>=2.5.3
- python-multipart>=0.0.6

### 1.2 Run API Server

```bash
# Set DATABASE_URL
export DATABASE_URL="your-postgresql-url-here"

# Start server
python3 api_server.py
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
======================================================================
NeuroVest API Server Starting
======================================================================
Version: 2.0.0
Documentation: /docs
Health Check: /health
======================================================================
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 1.3 Test Endpoints

Open browser to: **http://localhost:8000/docs**

You'll see Swagger UI with all endpoints. Try:

1. **GET /health** - Should return `{"status": "healthy"}`
2. **POST /api/auth/register?username=test** - Get an API key
3. **GET /api/predictions** - Use API key in header

Or run automated tests:
```bash
python3 test_api.py
```

Expected output:
```
============================================================
TEST SUMMARY
============================================================

Root Endpoint.................................... PASS
Health Check..................................... PASS
User Registration................................ PASS
Auth Required.................................... PASS
All Predictions.................................. PASS
Specific Asset (SPY)............................. PASS
Assets List...................................... PASS
Invalid Ticker................................... PASS

============================================================
Results: 8/8 tests passed
============================================================
```

---

## Step 2: Deploy to Railway

### 2.1 Create New Service

1. Go to Railway dashboard: https://railway.app/dashboard
2. Select your NeuroVest project
3. Click **+ New** → **Empty Service**
4. Name it: **NeuroVest-API**

### 2.2 Link to GitHub

1. In service settings → **Connect** → **GitHub Repo**
2. Select your NeuroVest repository
3. Branch: `main` (or your production branch)
4. Railway will detect Python and try to deploy

### 2.3 Configure Service

Click on **NeuroVest-API** service → **Settings**

#### Start Command:
```bash
uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

#### Root Directory:
```
/
```
(Leave as root since api_server.py is in project root)

#### Environment Variables:

Click **Variables** tab → **+ New Variable**

Add:
```
DATABASE_URL = <copy from PostgreSQL service>
```

**To get DATABASE_URL:**
1. Click on your **PostgreSQL** service
2. Go to **Variables** tab
3. Find `DATABASE_URL` → Click **Copy**
4. Paste into NeuroVest-API variables

### 2.4 Deploy

1. Click **Deploy** (or it may auto-deploy)
2. Watch logs in **Deployments** tab
3. Wait ~2-3 minutes for build

Expected build logs:
```
[build] Installing dependencies from requirements.txt
[build] Successfully installed fastapi uvicorn pydantic...
[build] Build succeeded
[deploy] Starting server...
[deploy] NeuroVest API Server Starting
[deploy] Uvicorn running on http://0.0.0.0:8000
```

### 2.5 Get Public URL

1. In **NeuroVest-API** service → **Settings** tab
2. Scroll to **Networking** → **Generate Domain**
3. Railway assigns: `https://neurovest-api-production-xyz.up.railway.app`
4. **Copy this URL** - this is your API endpoint!

---

## Step 3: Test Production API

### 3.1 Test Health Endpoint

```bash
curl https://neurovest-api-production-xyz.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "last_prediction": "2026-01-04",
  "assets_count": 40,
  "timestamp": "2026-01-04T20:15:30.123456"
}
```

### 3.2 Open Swagger Docs

Visit in browser:
```
https://neurovest-api-production-xyz.up.railway.app/docs
```

You'll see interactive API documentation!

### 3.3 Create Test User

```bash
curl -X POST "https://neurovest-api-production-xyz.up.railway.app/api/auth/register?username=testuser"
```

Response:
```json
{
  "message": "User created successfully",
  "user_id": 1,
  "api_key": "abc123def456...",
  "note": "⚠️ Save your API key - it won't be shown again!"
}
```

**Save the `api_key` value!**

### 3.4 Test Predictions

```bash
export API_KEY="your-api-key-from-above"

# Get all predictions
curl -H "X-API-Key: $API_KEY" \
  https://neurovest-api-production-xyz.up.railway.app/api/predictions

# Get specific asset
curl -H "X-API-Key: $API_KEY" \
  https://neurovest-api-production-xyz.up.railway.app/api/predictions/SPY
```

### 3.5 Run Full Test Suite

Update `test_api.py`:
```python
API_URL = "https://neurovest-api-production-xyz.up.railway.app"
```

Run:
```bash
python3 test_api.py
```

All tests should pass!

---

## Step 4: Update Dashboard with API URL

Update your dashboard or documentation to reference the new API:

**File:** `dashboard_comprehensive.py`

Add to overview page:
```python
st.markdown("""
### 🔌 REST API

The NeuroVest forecasting API is now live!

**Base URL:** `https://neurovest-api-production-xyz.up.railway.app`

**Quick Start:**
1. Register for API key: `POST /api/auth/register?username=yourname`
2. Get predictions: `GET /api/predictions` (with X-API-Key header)

**Documentation:** [API Docs](https://neurovest-api-production-xyz.up.railway.app/docs)
""")
```

---

## Architecture After Deployment

```
┌─────────────────────────────────────────────────┐
│           Railway Project: NeuroVest            │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────┐│
│  │ PostgreSQL   │  │ DataWorker2  │  │Dashboard││
│  │ (existing)   │  │ (existing)   │  │(existing)││
│  └──────┬───────┘  └──────┬───────┘  └────┬───┘│
│         │                 │                │    │
│         │        ┌────────┴────────┐       │    │
│         └────────┤  NeuroVest-API  ├───────┘    │
│                  │   (NEW!)        │            │
│                  │                 │            │
│                  │ - FastAPI       │            │
│                  │ - REST endpoints│            │
│                  │ - Swagger docs  │            │
│                  │ - Public URL    │            │
│                  └─────────────────┘            │
│                                                 │
└─────────────────────────────────────────────────┘
                       ↓
              External API Consumers
           (Web apps, traders, quants)
```

**Data Flow:**
1. DataWorker2 → Collects data hourly → PostgreSQL
2. DataWorker2 → Generates predictions → PostgreSQL
3. NeuroVest-API → Reads predictions → Returns JSON
4. Dashboard → Displays UI (Streamlit)
5. External Apps → Call API → Get predictions

---

## Costs

**Before:** $10-15/month (PostgreSQL + 2 workers)
**After:** $15-20/month (PostgreSQL + 3 workers)

Railway pricing: ~$5/month per additional service

---

## Monitoring & Maintenance

### Check API Health

Visit: `https://your-api-url/health`

Healthy response:
```json
{"status": "healthy", "database": "connected"}
```

Unhealthy response (503):
```json
{"status": "unhealthy", "detail": "Database connection failed"}
```

### View Logs

Railway dashboard → NeuroVest-API → Deployments → View Logs

Look for:
- ✅ `Application startup complete`
- ✅ `User X requesting prediction for Y`
- ⚠️ `Invalid API key attempted`
- ❌ `Error fetching predictions: ...`

### Common Issues

**Issue:** `503 Service Unavailable`
- **Cause:** Database connection failed
- **Fix:** Check DATABASE_URL is set correctly

**Issue:** `401 Unauthorized` on all requests
- **Cause:** Invalid or missing API key
- **Fix:** Register new user with `/api/auth/register`

**Issue:** `404 Not Found` for valid ticker
- **Cause:** No predictions in database yet
- **Fix:** Wait for daily prediction run (4:30 PM EST) or manually trigger

**Issue:** Slow response times (>1s)
- **Cause:** Database query not optimized
- **Fix:** Check indexes on `predictions` table

---

## API Usage Examples

### Python

```python
import requests

API_URL = "https://neurovest-api-production-xyz.up.railway.app"
API_KEY = "your-api-key-here"

headers = {"X-API-Key": API_KEY}

# Get all predictions
response = requests.get(f"{API_URL}/api/predictions", headers=headers)
predictions = response.json()

# Filter high-confidence SPIKE signals
spikes = [
    p for p in predictions
    if p['prediction_label'] == 'SPIKE'
    and p['confidence'] == 'high'
]

for s in spikes:
    print(f"{s['ticker']}: {s['prob_spike']:.1%} SPIKE probability")
```

### JavaScript

```javascript
const API_URL = 'https://neurovest-api-production-xyz.up.railway.app';
const API_KEY = 'your-api-key-here';

async function getPredictions() {
  const response = await fetch(`${API_URL}/api/predictions`, {
    headers: {
      'X-API-Key': API_KEY
    }
  });

  const predictions = await response.json();
  console.log(predictions);
}

getPredictions();
```

### cURL

```bash
# All predictions
curl -H "X-API-Key: YOUR_KEY" \
  https://neurovest-api-production-xyz.up.railway.app/api/predictions

# Specific asset
curl -H "X-API-Key: YOUR_KEY" \
  https://neurovest-api-production-xyz.up.railway.app/api/predictions/SPY

# Assets list
curl -H "X-API-Key: YOUR_KEY" \
  https://neurovest-api-production-xyz.up.railway.app/api/assets
```

---

## Next Steps

After basic deployment:

1. **Add Rate Limiting** (slowapi)
   - Prevent abuse
   - Different tiers (Free: 10/min, Pro: 60/min, Enterprise: unlimited)

2. **Add Caching** (Redis)
   - Cache predictions for 5 minutes
   - Reduce database load
   - Faster response times

3. **Add Monitoring** (Sentry)
   - Track errors
   - Performance metrics
   - Alerting

4. **Add Advanced Endpoints**
   - Historical predictions
   - Batch predictions
   - Custom asset upload via API

5. **Write Documentation**
   - User guide
   - Integration examples
   - Rate limits and pricing

---

## Success Checklist

- [ ] Railway service created and deployed
- [ ] Public URL generated
- [ ] Health endpoint returns 200 OK
- [ ] Swagger docs accessible at /docs
- [ ] Can register user and get API key
- [ ] Can fetch predictions with API key
- [ ] All test_api.py tests pass
- [ ] Dashboard updated with API URL
- [ ] API documented for users

---

## Support

**Issues?**
1. Check Railway logs for errors
2. Verify DATABASE_URL is set
3. Test health endpoint first
4. Review API docs at `/docs`

**Need Help?**
- GitHub Issues: Create issue with logs
- Railway Discord: Share deployment ID

---

**🎉 Congratulations! Your REST API is live!**

Users can now integrate NeuroVest predictions into their own applications, trading bots, and analytics tools.
