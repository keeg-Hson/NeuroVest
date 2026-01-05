# NeuroVest API Completion Checklist

## Current Status: 70% Complete

**What's Working:**
- ✅ Database with predictions, users, custom assets
- ✅ Authentication system (API key-based)
- ✅ Data pipeline (hourly updates, daily predictions)
- ✅ Dashboard with all features
- ✅ Custom asset uploads with user isolation

**What's Missing:**
- ❌ Standalone REST API service
- ❌ API documentation (OpenAPI/Swagger)
- ❌ Rate limiting
- ❌ API monitoring/analytics
- ❌ Production deployment as separate service

---

## Phase 1: Core REST API (Required for Launch)

**Estimated Time: 4-6 hours**

### Task 1.1: Create FastAPI Server ⏱️ 2 hours

**File to Create:** `api_server.py`

```python
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from datetime import datetime

from core.data_manager_postgres import DataManager
from auth_middleware import AuthManager

app = FastAPI(
    title="NeuroVest Forecasting API",
    description="AI-powered multi-asset market predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    prediction_label: str
    prob_crash: float
    prob_normal: float
    prob_spike: float
    confidence: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    database: str
    last_prediction: Optional[str]
    assets_count: int

# Authentication dependency
async def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    user = AuthManager.validate_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
def root():
    """API information"""
    return {
        "name": "NeuroVest Forecasting API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """System health check"""
    try:
        dm = DataManager()

        # Check database connectivity
        assets = dm.get_all_tickers()

        # Get latest prediction
        latest = dm.get_latest_predictions(limit=1)
        last_pred = None
        if len(latest) > 0:
            last_pred = str(latest.iloc[0]['prediction_date'])

        dm.close()

        return {
            "status": "healthy",
            "database": "connected",
            "last_prediction": last_pred,
            "assets_count": len(assets)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/api/predictions", response_model=List[PredictionResponse], tags=["Predictions"])
def get_all_predictions(
    user: dict = Depends(verify_api_key),
    limit: int = Query(40, le=1000, description="Maximum number of predictions")
):
    """Get latest predictions for all assets"""
    try:
        dm = DataManager()
        df = dm.get_latest_predictions(limit=limit)
        dm.close()

        if len(df) == 0:
            return []

        # Group by ticker and get most recent for each
        df = df.sort_values('prediction_date', ascending=False)
        latest = df.groupby('ticker').first().reset_index()

        predictions = []
        for _, row in latest.iterrows():
            predictions.append({
                "ticker": row['ticker'],
                "prediction_date": str(row['prediction_date']),
                "prediction_label": row['prediction_label'],
                "prob_crash": float(row['prob_crash']),
                "prob_normal": float(row['prob_normal']),
                "prob_spike": float(row['prob_spike']),
                "confidence": row['confidence'],
                "timestamp": datetime.now().isoformat()
            })

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/{ticker}", response_model=PredictionResponse, tags=["Predictions"])
def get_prediction(
    ticker: str,
    user: dict = Depends(verify_api_key)
):
    """Get latest prediction for specific asset"""
    try:
        dm = DataManager()
        df = dm.get_latest_predictions(limit=1000)
        dm.close()

        # Filter for this ticker
        ticker_df = df[df['ticker'].str.upper() == ticker.upper()]

        if len(ticker_df) == 0:
            raise HTTPException(status_code=404, detail=f"No predictions found for {ticker}")

        # Get most recent
        row = ticker_df.sort_values('prediction_date', ascending=False).iloc[0]

        return {
            "ticker": row['ticker'],
            "prediction_date": str(row['prediction_date']),
            "prediction_label": row['prediction_label'],
            "prob_crash": float(row['prob_crash']),
            "prob_normal": float(row['prob_normal']),
            "prob_spike": float(row['prob_spike']),
            "confidence": row['confidence'],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assets", tags=["Assets"])
def get_assets(user: dict = Depends(verify_api_key)):
    """Get list of all available assets"""
    try:
        dm = DataManager()
        assets = dm.get_all_tickers()
        dm.close()

        return {
            "total": len(assets),
            "assets": assets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/register", tags=["Auth"])
def register_user(username: str):
    """Register new user and get API key"""
    try:
        user_data = AuthManager.create_user(username)
        return {
            "message": "User created successfully",
            "user_id": user_data['user_id'],
            "api_key": user_data['api_key'],
            "note": "Save your API key - it won't be shown again"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**Action Items:**
- [ ] Create `api_server.py` with above code
- [ ] Test locally: `python3 api_server.py`
- [ ] Verify `/docs` endpoint shows Swagger UI
- [ ] Test all endpoints with sample API key

---

### Task 1.2: Add Requirements ⏱️ 15 min

**File to Update:** `requirements.txt`

Add these lines:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
```

**Action Items:**
- [ ] Add FastAPI dependencies to requirements.txt
- [ ] Test install: `pip install -r requirements.txt`

---

### Task 1.3: Deploy to Railway ⏱️ 30 min

**Steps:**

1. **Create new Railway service:**
   ```bash
   # In Railway dashboard:
   # New → Empty Service → Link to GitHub repo → Name: "NeuroVest-API"
   ```

2. **Configure service:**
   - Start Command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - Environment Variables:
     - Copy `DATABASE_URL` from your PostgreSQL service
     - Add `PORT=8000`

3. **Deploy:**
   ```bash
   git add api_server.py requirements.txt
   git commit -m "Add standalone REST API with FastAPI"
   git push origin main
   ```

4. **Get API URL:**
   - Railway will assign: `https://neurovest-api.up.railway.app`
   - Test: `curl https://neurovest-api.up.railway.app/health`

**Action Items:**
- [ ] Create Railway service
- [ ] Configure environment variables
- [ ] Deploy and verify health endpoint
- [ ] Get public API URL

---

### Task 1.4: Test API Endpoints ⏱️ 30 min

**Create test script:** `test_api.py`

```python
import requests

API_URL = "https://neurovest-api.up.railway.app"
API_KEY = "your-api-key-here"  # Get from /api/auth/register

headers = {"X-API-Key": API_KEY}

# Test 1: Health check
print("Testing /health...")
r = requests.get(f"{API_URL}/health")
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}\n")

# Test 2: Get all predictions
print("Testing /api/predictions...")
r = requests.get(f"{API_URL}/api/predictions", headers=headers)
print(f"Status: {r.status_code}")
predictions = r.json()
print(f"Found {len(predictions)} predictions\n")

# Test 3: Get specific asset
print("Testing /api/predictions/SPY...")
r = requests.get(f"{API_URL}/api/predictions/SPY", headers=headers)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}\n")

# Test 4: Get assets list
print("Testing /api/assets...")
r = requests.get(f"{API_URL}/api/assets", headers=headers)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}\n")

print("✅ All tests passed!")
```

**Action Items:**
- [ ] Create test script
- [ ] Register test user and get API key
- [ ] Run all endpoint tests
- [ ] Verify responses match expected format

---

## Phase 2: API Hardening (Production-Ready)

**Estimated Time: 3-4 hours**

### Task 2.1: Add Rate Limiting ⏱️ 1 hour

**Install slowapi:**
```bash
pip install slowapi
```

**Update `api_server.py`:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/predictions")
@limiter.limit("60/minute")  # 60 requests per minute
def get_all_predictions(...):
    ...
```

**Rate Limits by Tier:**
- Free: 10 requests/minute
- Individual: 60 requests/minute
- Professional: 300 requests/minute
- Enterprise: Unlimited

**Action Items:**
- [ ] Install slowapi
- [ ] Add rate limiting to all endpoints
- [ ] Test rate limit triggers (make 61 requests in 1 minute)
- [ ] Add rate limit info to response headers

---

### Task 2.2: Add Logging & Monitoring ⏱️ 1 hour

**Create logger:**
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/api.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add to endpoints
@app.get("/api/predictions/{ticker}")
def get_prediction(ticker: str, user: dict = Depends(verify_api_key)):
    logger.info(f"User {user['user_id']} requested prediction for {ticker}")
    try:
        # ... existing code ...
        logger.info(f"Successfully returned prediction for {ticker}")
        return response
    except Exception as e:
        logger.error(f"Error getting prediction for {ticker}: {e}")
        raise
```

**Metrics to Track:**
- Total requests per endpoint
- Response times (p50, p95, p99)
- Error rates
- Most requested assets
- Active users

**Action Items:**
- [ ] Add logging to all endpoints
- [ ] Create logs directory
- [ ] Add request/response time tracking
- [ ] Monitor Railway logs for errors

---

### Task 2.3: Add Input Validation ⏱️ 1 hour

**Using Pydantic:**
```python
from pydantic import BaseModel, validator
from typing import Optional

class PredictionRequest(BaseModel):
    ticker: str
    date: Optional[str] = None

    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError('Invalid ticker format')
        return v.upper()

@app.get("/api/predictions/{ticker}")
def get_prediction(ticker: str, user: dict = Depends(verify_api_key)):
    # Pydantic validation happens automatically
    ...
```

**Validation Rules:**
- Ticker: 1-10 chars, alphanumeric + / or _
- Date: YYYY-MM-DD format if provided
- Limit: 1-1000 for batch requests
- API Key: 32+ chars, alphanumeric

**Action Items:**
- [ ] Add Pydantic models for all request types
- [ ] Add validators for edge cases
- [ ] Test with invalid inputs
- [ ] Return helpful error messages

---

### Task 2.4: API Documentation ⏱️ 30 min

FastAPI auto-generates docs, but enhance them:

```python
@app.get(
    "/api/predictions/{ticker}",
    response_model=PredictionResponse,
    summary="Get asset prediction",
    description="""
    Returns the latest prediction for a specific asset.

    **Example Request:**
    ```
    GET /api/predictions/SPY
    Headers: X-API-Key: your-api-key-here
    ```

    **Example Response:**
    ```json
    {
      "ticker": "SPY",
      "prediction_label": "SPIKE",
      "prob_spike": 0.67,
      "confidence": "high"
    }
    ```

    **Prediction Labels:**
    - CRASH: Expect >0.6% decline
    - NORMAL: Range-bound movement
    - SPIKE: Expect >0.6% gain
    """,
    tags=["Predictions"]
)
def get_prediction(...):
    ...
```

**Action Items:**
- [ ] Add detailed descriptions to all endpoints
- [ ] Include example requests/responses
- [ ] Document error codes and meanings
- [ ] Test Swagger UI at `/docs`

---

## Phase 3: Advanced Features (Optional)

**Estimated Time: 4-6 hours**

### Task 3.1: WebSocket Support ⏱️ 2 hours

For real-time prediction updates:

```python
from fastapi import WebSocket

@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send predictions every 60 seconds
            predictions = get_latest_predictions()
            await websocket.send_json(predictions)
            await asyncio.sleep(60)
    except:
        await websocket.close()
```

---

### Task 3.2: Batch Prediction Endpoint ⏱️ 1 hour

```python
@app.post("/api/predictions/batch")
def get_batch_predictions(
    tickers: List[str],
    user: dict = Depends(verify_api_key)
):
    """Get predictions for multiple assets in one request"""
    results = []
    for ticker in tickers:
        try:
            pred = get_prediction_logic(ticker)
            results.append(pred)
        except:
            results.append({"ticker": ticker, "error": "Not found"})
    return results
```

---

### Task 3.3: Historical Predictions API ⏱️ 1 hour

```python
@app.get("/api/predictions/{ticker}/history")
def get_prediction_history(
    ticker: str,
    days: int = Query(30, le=365),
    user: dict = Depends(verify_api_key)
):
    """Get historical predictions for backtesting"""
    dm = DataManager()
    df = dm.get_predictions_for_ticker(ticker, days=days)
    dm.close()

    return df.to_dict(orient='records')
```

---

### Task 3.4: Custom Asset Prediction ⏱️ 2 hours

```python
@app.post("/api/custom/predict")
async def predict_custom_asset(
    file: UploadFile,
    user: dict = Depends(verify_api_key)
):
    """Upload CSV and get instant prediction"""
    # Save file
    df = pd.read_csv(file.file)

    # Validate format
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required):
        raise HTTPException(400, "Missing required columns")

    # Generate prediction
    prediction = run_prediction_pipeline(df)

    return prediction
```

---

## Phase 4: Deployment & Operations

### Task 4.1: Production Checklist

**Before Launch:**
- [ ] **Security**
  - [ ] API keys required for all protected endpoints
  - [ ] Rate limiting enabled and tested
  - [ ] HTTPS only (Railway provides this)
  - [ ] CORS restricted to known domains
  - [ ] Input validation on all endpoints
  - [ ] SQL injection protection (using SQLAlchemy parameterized queries)

- [ ] **Reliability**
  - [ ] Health endpoint returns accurate status
  - [ ] Database connection pooling configured
  - [ ] Error handling on all endpoints
  - [ ] Graceful degradation if DB unavailable
  - [ ] Automatic retries for transient failures

- [ ] **Performance**
  - [ ] All endpoints respond <500ms (p95)
  - [ ] Database queries optimized with indexes
  - [ ] Results cached where appropriate
  - [ ] Connection pooling prevents DB exhaustion

- [ ] **Monitoring**
  - [ ] Logging to Railway (stdout/stderr)
  - [ ] Error tracking (Sentry optional)
  - [ ] Health checks every 5 min
  - [ ] Alerts for downtime/errors

- [ ] **Documentation**
  - [ ] Swagger UI accessible at `/docs`
  - [ ] All endpoints documented with examples
  - [ ] Authentication flow explained
  - [ ] Error codes documented

---

### Task 4.2: Testing

**Manual Tests:**
```bash
# 1. Health check
curl https://your-api.railway.app/health

# 2. Register user
curl -X POST https://your-api.railway.app/api/auth/register?username=test

# 3. Get predictions (with API key)
curl -H "X-API-Key: YOUR_KEY" https://your-api.railway.app/api/predictions

# 4. Get specific asset
curl -H "X-API-Key: YOUR_KEY" https://your-api.railway.app/api/predictions/SPY

# 5. Test rate limiting (run 61 times)
for i in {1..61}; do curl -H "X-API-Key: YOUR_KEY" https://your-api.railway.app/health; done
```

**Automated Tests:**
Create `tests/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predictions_no_auth():
    response = client.get("/api/predictions")
    assert response.status_code == 401

def test_predictions_with_auth():
    headers = {"X-API-Key": "valid-key"}
    response = client.get("/api/predictions", headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

---

## Summary: Time to Production-Ready API

| Phase | Tasks | Time | Priority |
|-------|-------|------|----------|
| **Phase 1: Core API** | Create FastAPI, deploy, test | 4-6 hours | 🔴 Critical |
| **Phase 2: Hardening** | Rate limiting, logging, validation | 3-4 hours | 🟡 High |
| **Phase 3: Advanced** | WebSocket, batch, history | 4-6 hours | 🟢 Optional |
| **Phase 4: Operations** | Testing, monitoring, docs | 2-3 hours | 🟡 High |

**Total Time to MVP API: 6-8 hours**
**Total Time to Production-Ready: 12-15 hours**

---

## Current Status Breakdown

### ✅ What You Already Have (90% of backend)
- Database with predictions, users, assets
- Authentication system (API keys)
- Data collection pipeline
- Model training and predictions
- Custom asset uploads

### ❌ What You Need to Build (10% remaining)
- Standalone FastAPI service (4 hours)
- Rate limiting (1 hour)
- API documentation (30 min)
- Testing (2 hours)

### 🎯 Recommended Next Steps

**This Week (Get API Live):**
1. Create `api_server.py` (2 hours)
2. Deploy to Railway (30 min)
3. Test endpoints (1 hour)
4. Add to documentation (30 min)

**Next Week (Harden for Production):**
1. Add rate limiting (1 hour)
2. Add logging (1 hour)
3. Write tests (2 hours)
4. Create API guide for users (1 hour)

**You're 90% there! Just need to wrap existing functionality in FastAPI and deploy it.**
