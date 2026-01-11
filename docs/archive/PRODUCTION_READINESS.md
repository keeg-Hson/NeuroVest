# Production Readiness Checklist

## ✅ COMPLETED (Equivalent Substitutions)

- [x] **API/UI Deployment** - Railway Dashboard2 (replaces Render API)
- [x] **Worker Service** - Railway DataWorker2 running
- [x] **Database** - Railway PostgreSQL (replaces Supabase)
- [x] **Data Pipeline** - Hourly updates, daily predictions
- [x] **Model Training** - 3 models trained, weekly retraining
- [x] **Predictions** - Daily generation at 4:30 PM EST

---

## 🔴 CRITICAL - Must Do Before Launch

### 1. Fix Custom Asset Uploads (Security + Persistence)

**Problem:**
- Files saved to ephemeral filesystem (lost on restart)
- No user isolation (everyone sees all uploads)
- No authentication

**Solution:**

```bash
# Step 1: Update database schema
psql $DATABASE_URL -f fix_custom_assets.sql

# Step 2: Update dashboard to use auth
# See auth_middleware.py for implementation

# Step 3: Test upload persistence
# Upload asset → restart container → verify asset still exists
```

**Files Created:**
- `fix_custom_assets.sql` - Database schema updates
- `auth_middleware.py` - User authentication + custom asset storage

**Integration Steps:**
1. Run SQL migrations on Railway PostgreSQL
2. Update `dashboard_comprehensive.py` to import `AuthManager`
3. Replace custom asset upload code (lines 1786-1790) with `save_custom_asset_to_db()`
4. Add auth UI to sidebar with `AuthManager.require_auth()`
5. Filter assets by user_id in all queries

---

### 2. Add API Authentication

**Current State:** No authentication - anyone can access

**Options:**

**Option A: Simple API Keys (Recommended)**
```python
# Already implemented in auth_middleware.py
# Features:
# - User accounts with API keys
# - Key generation/validation
# - Session management
```

**Option B: Full OAuth2 (Complex)**
- Requires external provider (Auth0, Supabase Auth, etc.)
- More secure but slower to implement

**Recommendation:** Start with Option A (API keys)

---

### 3. REST API Endpoints

**Check if FastAPI exists:**

```bash
# Look for api_server.py or similar
ls -la *api*.py framework/api*.py
```

**If NOT exists, create minimal API:**

```python
# framework/api_server.py
from fastapi import FastAPI, HTTPException, Header
from auth_middleware import AuthManager

app = FastAPI(title="NeuroVest API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/predict/{asset}")
def get_prediction(asset: str, api_key: str = Header(...)):
    user = AuthManager.validate_api_key(api_key)
    if not user:
        raise HTTPException(401, "Invalid API key")

    # Get prediction from database
    # ... implementation ...

    return {"asset": asset, "prediction": "..."}
```

**Deploy to Railway:**
1. Create new service: "NeuroVest-API"
2. Start command: `uvicorn framework.api_server:app --host 0.0.0.0 --port $PORT`
3. Add DATABASE_URL environment variable
4. Deploy

---

## 🟡 MEDIUM PRIORITY - Important But Not Blocking

### 4. Error Handling & Monitoring

**Add:**
- [ ] Sentry or similar error tracking
- [ ] Logging to file/service (not just stdout)
- [ ] Health check endpoints that actually verify DB connectivity
- [ ] Alerting when predictions fail to generate

**Quick Win:**
```python
# Add to worker and dashboard
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 5. Rate Limiting

**Prevent abuse:**

```python
# Add to FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/predict/{asset}")
@limiter.limit("60/minute")  # 60 requests per minute
def get_prediction(...):
    ...
```

### 6. Data Validation

**Validate inputs:**
- [ ] Custom asset uploads (check date format, required columns)
- [ ] API request parameters (ticker format, date ranges)
- [ ] Prediction requests (asset exists, data available)

---

## 🟢 NICE-TO-HAVE - Optional Enhancements

### 7. Redis Cache (Performance)

**Why:** Speed up repeated database queries

**Implementation:**
```bash
# Add Redis service on Railway
# Update code to cache predictions

from redis import Redis
import json

redis_client = Redis.from_url(os.getenv('REDIS_URL'))

def get_prediction_cached(ticker):
    # Check cache first
    cached = redis_client.get(f"prediction:{ticker}")
    if cached:
        return json.loads(cached)

    # Query database
    prediction = query_database(ticker)

    # Cache for 1 hour
    redis_client.setex(
        f"prediction:{ticker}",
        3600,
        json.dumps(prediction)
    )

    return prediction
```

**Cost:** ~$5-10/month on Railway

### 8. Automated Testing

**Add:**
- [ ] Unit tests for core functions
- [ ] Integration tests for API endpoints
- [ ] E2E tests for critical user flows

### 9. Documentation

**Create:**
- [ ] API documentation (auto-generated with FastAPI/Swagger)
- [ ] User guide for custom asset uploads
- [ ] Admin guide for managing users

---

## 📋 LAUNCH CHECKLIST

Before going live:

- [ ] **Security**
  - [ ] Custom assets save to PostgreSQL (not filesystem)
  - [ ] User authentication working
  - [ ] API keys required for all endpoints
  - [ ] Rate limiting enabled

- [ ] **Reliability**
  - [ ] Worker runs continuously without crashes
  - [ ] Predictions generate daily (verify logs)
  - [ ] Data updates hourly (verify database)
  - [ ] Error monitoring in place

- [ ] **Performance**
  - [ ] Dashboard loads in <3 seconds
  - [ ] API responds in <500ms
  - [ ] Database queries optimized (check EXPLAIN plans)

- [ ] **UX**
  - [ ] Custom asset uploads work end-to-end
  - [ ] Error messages are helpful
  - [ ] Loading states show progress

- [ ] **Testing**
  - [ ] Upload custom asset → restart container → verify persisted
  - [ ] Create user → generate API key → test API access
  - [ ] Test all prediction endpoints with valid/invalid data

---

## 🚀 DEPLOYMENT STEPS

### Phase 1: Fix Critical Issues (This Week)

```bash
# Day 1: Database Schema
psql $DATABASE_URL -f fix_custom_assets.sql

# Day 2: Update Dashboard
# Integrate auth_middleware.py
# Update custom asset upload code

# Day 3: Test & Deploy
# Verify custom assets persist after restart
# Deploy to Railway

# Day 4: Create REST API
# Build minimal FastAPI server
# Deploy as separate Railway service

# Day 5: End-to-End Testing
# Test all user flows
# Fix any bugs found
```

### Phase 2: Medium Priority (Next Week)

- Add error monitoring
- Implement rate limiting
- Add data validation

### Phase 3: Nice-to-Have (Future)

- Add Redis cache
- Write tests
- Create documentation

---

## 💰 ESTIMATED COSTS

**Current:** ~$10-15/month (Railway PostgreSQL + 2 services)

**After Production Hardening:**
- Railway PostgreSQL: $10/month
- DataWorker2: $5/month
- Dashboard2: $5/month
- API Service: $5/month
- Redis (optional): $5/month

**Total:** ~$25-30/month

---

## ⚡ QUICK START (Get Production-Ready TODAY)

If you only have time for the bare minimum:

```bash
# 1. Fix custom assets (30 min)
psql $DATABASE_URL -f fix_custom_assets.sql
# Update dashboard with auth_middleware.py

# 2. Add API key auth (15 min)
# Copy auth code to dashboard
# Test login/signup flow

# 3. Deploy (5 min)
git add .
git commit -m "Add user auth and persistent custom assets"
git push

# Done! You now have:
# ✅ User authentication
# ✅ Persistent custom assets
# ✅ Basic security
```

**This gets you 80% production-ready in under 1 hour.**

The rest (API, Redis, monitoring) can be added incrementally.
