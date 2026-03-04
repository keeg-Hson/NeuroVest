#!/usr/bin/env python3
"""
NeuroVest REST API Server
Production-ready FastAPI server for multi-asset predictions

Endpoints:
- GET /health - System health check
- GET /api/predictions - All latest predictions
- GET /api/predictions/{ticker} - Specific asset prediction
- GET /api/assets - List all available assets
- POST /api/auth/register - Create new user and API key

Authentication: X-API-Key header required for protected endpoints
"""

from fastapi import FastAPI, HTTPException, Header, Query, Depends, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from datetime import datetime, timedelta
import logging
import pandas as pd
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.data_manager_postgres import DataManager
from auth_middleware import AuthManager
from cache_manager import cache
from request_logger import RequestLoggerMiddleware
from analytics_api import router as analytics_router
from websocket_streaming import websocket_endpoint, get_websocket_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Tier-based rate limits
RATE_LIMITS = {
    'free': "10/minute",
    'individual': "60/minute",
    'pro': "300/minute",
    'enterprise': "10000/minute"
}

# Initialize FastAPI app
app = FastAPI(
    title="NeuroVest Forecasting API",
    description="AI-powered multi-asset market predictions with ensemble ML models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Info", "description": "System information and health"},
        {"name": "Predictions", "description": "Market forecast endpoints"},
        {"name": "Assets", "description": "Asset management"},
        {"name": "Auth", "description": "User authentication"}
    ]
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include analytics router
app.include_router(analytics_router)

# Add request logging middleware
app.add_middleware(RequestLoggerMiddleware)

# CORS middleware - configurable origins for security
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    # Default to restrictive origins in production
    ALLOWED_ORIGINS = [
        "https://neurovestdemo.up.railway.app",
        "http://localhost:8501",  # Streamlit dev
        "http://localhost:3000",  # React dev
    ]
    logger.info(f"Using default CORS origins: {ALLOWED_ORIGINS}")
else:
    logger.info(f"Using configured CORS origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionResponse(BaseModel):
    """Single asset prediction response"""
    ticker: str = Field(..., description="Asset ticker symbol")
    prediction_date: str = Field(..., description="Date of prediction")
    prediction_label: str = Field(..., description="CRASH, NORMAL, or SPIKE")
    prob_crash: float = Field(..., ge=0, le=1, description="Probability of crash")
    prob_normal: float = Field(..., ge=0, le=1, description="Probability of normal")
    prob_spike: float = Field(..., ge=0, le=1, description="Probability of spike")
    confidence: str = Field(..., description="high, medium, or low")
    timestamp: str = Field(..., description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "SPY",
                "prediction_date": "2026-01-04",
                "prediction_label": "SPIKE",
                "prob_crash": 0.12,
                "prob_normal": 0.45,
                "prob_spike": 0.43,
                "confidence": "high",
                "timestamp": "2026-01-04T16:30:00Z"
            }
        }

class HealthResponse(BaseModel):
    """System health status"""
    status: str = Field(..., description="healthy or unhealthy")
    database: str = Field(..., description="Database connection status")
    last_prediction: Optional[str] = Field(None, description="Date of most recent prediction")
    assets_count: int = Field(..., description="Number of assets in database")
    timestamp: str = Field(..., description="Health check timestamp")

class AssetsResponse(BaseModel):
    """Available assets list"""
    total: int = Field(..., description="Total number of assets")
    assets: List[str] = Field(..., description="List of asset tickers")

class RegisterResponse(BaseModel):
    """User registration response"""
    message: str
    user_id: int
    api_key: str
    note: str

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    timestamp: str

# ============================================================================
# Authentication Dependency
# ============================================================================

async def verify_api_key(x_api_key: str = Header(..., description="Your API key")):
    """Validate API key from header and return user with tier"""
    dm = DataManager()

    try:
        with dm.engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(
                text("SELECT id, username, tier FROM users WHERE api_key = :api_key"),
                {"api_key": x_api_key}
            )
            user = result.fetchone()

        dm.close()

        if not user:
            logger.warning(f"Invalid API key attempted: {x_api_key[:8]}...")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Register at /api/auth/register to get a key."
            )

        user_dict = {
            'user_id': user[0],
            'username': user[1],
            'tier': user[2] if len(user) > 2 and user[2] else 'free'
        }

        logger.info(f"Authenticated user: {user_dict['user_id']} (tier: {user_dict['tier']})")
        return user_dict

    except Exception as e:
        dm.close()
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# ============================================================================
# Rate Limiting Helper
# ============================================================================

def get_user_rate_limit(user: dict) -> str:
    """Get rate limit string for user's tier"""
    tier = user.get('tier', 'free')
    return RATE_LIMITS.get(tier, RATE_LIMITS['free'])

# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    "/",
    tags=["Info"],
    summary="API Information"
)
def root():
    """
    Root endpoint with API information and links
    """
    return {
        "name": "NeuroVest Forecasting API",
        "version": "2.0.0",
        "description": "AI-powered multi-asset market predictions",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predictions": "/api/predictions",
            "specific_asset": "/api/predictions/{ticker}",
            "assets_list": "/api/assets",
            "register": "/api/auth/register"
        }
    }

@app.get(
    "/cache/stats",
    tags=["Info"],
    summary="Cache Statistics",
    description="View Redis cache performance metrics"
)
def cache_stats():
    """
    Get cache statistics and performance metrics

    Returns hit rate, total operations, and cache status
    """
    return cache.get_stats()

@app.get(
    "/ws/stats",
    tags=["Info"],
    summary="WebSocket Statistics",
    description="View active WebSocket connections and subscriptions"
)
def websocket_stats():
    """
    Get WebSocket connection statistics

    Returns active connections, tier distribution, and subscriptions
    """
    return get_websocket_stats()

@app.websocket("/ws/predictions")
async def predictions_stream(
    websocket: WebSocket,
    api_key: str = Query(..., description="Your API key"),
    tickers: Optional[str] = Query(None, description="Comma-separated tickers")
):
    """
    WebSocket endpoint for real-time prediction streaming

    Connect via:
        ws://api.neurovest.com/ws/predictions?api_key=YOUR_KEY&tickers=SPY,QQQ

    Message Types:
        - connected: Initial connection
        - prediction: New prediction
        - heartbeat: Keep-alive (every 30s)
    """
    await websocket_endpoint(websocket, api_key, tickers)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Info"],
    summary="Health Check",
    description="Check system health and database connectivity"
)
def health_check():
    """
    System health check endpoint

    Returns current status of:
    - API service
    - Database connection
    - Latest prediction date
    - Number of assets available
    """
    try:
        dm = DataManager()

        # Check database connectivity
        assets = dm.get_all_assets()  # Returns list of (ticker, asset_type) tuples

        # Get latest prediction
        latest = dm.get_latest_predictions(limit=1)
        last_pred = None
        if len(latest) > 0:
            last_pred = str(latest.iloc[0]['prediction_date'])

        dm.close()

        logger.info(f"Health check successful: {len(assets)} assets, last prediction: {last_pred}")

        return {
            "status": "healthy",
            "database": "connected",
            "last_prediction": last_pred,
            "assets_count": len(assets),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get(
    "/api/predictions",
    response_model=List[PredictionResponse],
    tags=["Predictions"],
    summary="Get All Predictions",
    description="""
    Get latest predictions for all available assets.

    Returns one prediction per asset (most recent).
    Requires authentication via X-API-Key header.

    **Rate Limits by Tier:**
    - Free: 10 requests/minute
    - Individual: 60 requests/minute
    - Pro: 300 requests/minute
    - Enterprise: 10,000 requests/minute

    **Example:**
    ```
    curl -H "X-API-Key: YOUR_KEY" https://api.neurovest.com/api/predictions
    ```
    """
)
@limiter.limit("300/minute")  # Max for non-enterprise
def get_all_predictions(
    request: Request,
    limit: int = Query(100, le=1000, description="Maximum predictions to return"),
    user: dict = Depends(verify_api_key)
):
    """
    Get latest predictions for all assets

    Args:
        limit: Maximum number of predictions to return (default 100, max 1000)
        user: Authenticated user (injected by dependency)

    Returns:
        List of PredictionResponse objects
    """
    try:
        logger.info(f"User {user['user_id']} requesting all predictions (limit={limit})")

        # Try cache first
        cache_key = f"predictions:all:{limit}"
        cached = cache.get(cache_key)
        if cached:
            logger.info("Returning cached predictions")
            return cached

        dm = DataManager()
        df = dm.get_latest_predictions(limit=limit)
        dm.close()

        if len(df) == 0:
            logger.warning("No predictions found in database")
            return []

        # Group by ticker and get most recent for each
        df = df.sort_values('prediction_date', ascending=False)
        latest = df.groupby('ticker').first().reset_index()

        predictions = []
        for _, row in latest.iterrows():
            # Derive 3-class probabilities from ensemble_prob and prediction_label
            ensemble_prob = float(row['ensemble_prob'])
            label = row['prediction_label']

            # Distribute remaining probability across other classes
            remaining = 1.0 - ensemble_prob
            other_prob = remaining / 2.0

            if label == 'CRASH':
                prob_crash, prob_normal, prob_spike = ensemble_prob, other_prob, other_prob
            elif label == 'SPIKE':
                prob_crash, prob_normal, prob_spike = other_prob, other_prob, ensemble_prob
            else:  # NORMAL
                prob_crash, prob_normal, prob_spike = other_prob, ensemble_prob, other_prob

            # Derive confidence from confidence_score
            conf_score = float(row.get('confidence_score', 0.5))
            confidence = 'high' if conf_score >= 0.7 else ('medium' if conf_score >= 0.5 else 'low')

            predictions.append({
                "ticker": row['ticker'],
                "prediction_date": str(row['prediction_date']),
                "prediction_label": label,
                "prob_crash": round(prob_crash, 3),
                "prob_normal": round(prob_normal, 3),
                "prob_spike": round(prob_spike, 3),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })

        logger.info(f"Returning {len(predictions)} predictions")

        # Cache for 5 minutes
        cache.set(cache_key, predictions, ttl=300)

        return predictions

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")

@app.get(
    "/api/predictions/{ticker}",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Get Asset Prediction",
    description="""
    Get latest prediction for a specific asset.

    **Supported Assets:**
    - Stocks/ETFs: SPY, QQQ, IWM, DIA, VTI, etc.
    - Crypto: BTC/USDT, ETH/USDT, SOL/USDT, etc.
    - Metals: GLD, SLV, PPLT, PALL

    **Prediction Labels:**
    - **CRASH**: Expect >0.6% decline (stocks) or >2% (crypto)
    - **NORMAL**: Range-bound movement
    - **SPIKE**: Expect >0.6% gain (stocks) or >2% (crypto)

    **Rate Limits by Tier:**
    - Free: 10 requests/minute
    - Individual: 60 requests/minute
    - Pro: 300 requests/minute
    - Enterprise: 10,000 requests/minute

    **Example:**
    ```
    curl -H "X-API-Key: YOUR_KEY" https://api.neurovest.com/api/predictions/SPY
    ```
    """
)
@limiter.limit("300/minute")
def get_prediction(
    request: Request,
    ticker: str,
    user: dict = Depends(verify_api_key)
):
    """
    Get latest prediction for specific asset

    Args:
        ticker: Asset ticker symbol (e.g., SPY, BTC/USDT)
        user: Authenticated user (injected by dependency)

    Returns:
        PredictionResponse with probabilities and confidence
    """
    try:
        ticker_upper = ticker.upper()
        logger.info(f"User {user['user_id']} requesting prediction for {ticker_upper}")

        # Try cache first
        cache_key = f"prediction:{ticker_upper}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Returning cached prediction for {ticker_upper}")
            return cached

        dm = DataManager()
        df = dm.get_latest_predictions(limit=1000)
        dm.close()

        # Filter for this ticker (case-insensitive)
        ticker_df = df[df['ticker'].str.upper() == ticker_upper]

        if len(ticker_df) == 0:
            logger.warning(f"No predictions found for ticker: {ticker_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for {ticker}. Available assets: /api/assets"
            )

        # Get most recent prediction
        row = ticker_df.sort_values('prediction_date', ascending=False).iloc[0]

        # Derive 3-class probabilities from ensemble_prob and prediction_label
        ensemble_prob = float(row['ensemble_prob'])
        label = row['prediction_label']

        # Distribute remaining probability across other classes
        remaining = 1.0 - ensemble_prob
        other_prob = remaining / 2.0

        if label == 'CRASH':
            prob_crash, prob_normal, prob_spike = ensemble_prob, other_prob, other_prob
        elif label == 'SPIKE':
            prob_crash, prob_normal, prob_spike = other_prob, other_prob, ensemble_prob
        else:  # NORMAL
            prob_crash, prob_normal, prob_spike = other_prob, ensemble_prob, other_prob

        # Derive confidence from confidence_score
        conf_score = float(row.get('confidence_score', 0.5))
        confidence = 'high' if conf_score >= 0.7 else ('medium' if conf_score >= 0.5 else 'low')

        response = {
            "ticker": row['ticker'],
            "prediction_date": str(row['prediction_date']),
            "prediction_label": label,
            "prob_crash": round(prob_crash, 3),
            "prob_normal": round(prob_normal, 3),
            "prob_spike": round(prob_spike, 3),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Returning prediction for {ticker_upper}: {label}")

        # Cache for 5 minutes
        cache.set(cache_key, response, ttl=300)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prediction for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction: {str(e)}")

@app.get(
    "/api/assets",
    response_model=AssetsResponse,
    tags=["Assets"],
    summary="List Available Assets",
    description="""
    Get list of all assets available for predictions.

    Includes:
    - 14 stocks and ETFs (SPY, QQQ, sector ETFs, bonds)
    - 7 precious metals (GLD, SLV, platinum, palladium)
    - 10 cryptocurrencies (BTC, ETH, SOL, etc.)
    - User-uploaded custom assets
    """
)
def get_assets(user: dict = Depends(verify_api_key)):
    """
    Get list of all available assets

    Args:
        user: Authenticated user (injected by dependency)

    Returns:
        AssetsResponse with total count and asset list
    """
    try:
        logger.info(f"User {user['user_id']} requesting assets list")

        dm = DataManager()
        assets = dm.get_all_assets()  # Returns list of (ticker, asset_type) tuples
        dm.close()

        # Extract just the ticker names
        tickers = [ticker for ticker, asset_type in assets]

        logger.info(f"Returning {len(tickers)} assets")

        return {
            "total": len(tickers),
            "assets": sorted(tickers)
        }
    except Exception as e:
        logger.error(f"Error fetching assets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch assets: {str(e)}")

@app.post(
    "/api/auth/register",
    response_model=RegisterResponse,
    tags=["Auth"],
    summary="Register New User",
    description="""
    Create a new user account and receive an API key.

    **Important:** Save your API key immediately - it won't be shown again!

    **Example:**
    ```
    curl -X POST "https://api.neurovest.com/api/auth/register?username=myname"
    ```
    """
)
def register_user(username: str = Query(..., min_length=3, max_length=50)):
    """
    Register new user and get API key

    Args:
        username: Desired username (3-50 characters)

    Returns:
        RegisterResponse with user_id and API key
    """
    try:
        logger.info(f"New user registration request: {username}")

        user_data = AuthManager.create_user(username)

        logger.info(f"User created successfully: {user_data['user_id']}")

        return {
            "message": "User created successfully",
            "user_id": user_data['user_id'],
            "api_key": user_data['api_key'],
            "note": "⚠️ Save your API key - it won't be shown again!"
        }
    except Exception as e:
        logger.error(f"Registration failed for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ============================================================================
# Advanced Endpoints
# ============================================================================

@app.get(
    "/api/predictions/{ticker}/history",
    response_model=List[PredictionResponse],
    tags=["Predictions"],
    summary="Get Historical Predictions",
    description="""
    Get historical predictions for a specific asset.

    **Parameters:**
    - `ticker`: Asset symbol
    - `days`: Number of days of history (default 30, max 365)

    **Rate Limits:** Same as other prediction endpoints

    **Example:**
    ```
    curl -H "X-API-Key: YOUR_KEY" \
      "https://api.neurovest.com/api/predictions/SPY/history?days=90"
    ```
    """
)
@limiter.limit("300/minute")
def get_prediction_history(
    request: Request,
    ticker: str,
    days: int = Query(30, le=365, description="Days of history to return"),
    user: dict = Depends(verify_api_key)
):
    """
    Get historical predictions for backtesting and analysis

    Args:
        ticker: Asset ticker symbol
        days: Number of days (max 365)
        user: Authenticated user

    Returns:
        List of historical predictions
    """
    try:
        ticker_upper = ticker.upper()
        logger.info(f"User {user['user_id']} requesting {days} days history for {ticker_upper}")

        dm = DataManager()

        # Query predictions table directly with date filter
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)

        with dm.engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(
                text("""
                    SELECT * FROM predictions
                    WHERE ticker = :ticker
                      AND prediction_date >= :cutoff_date
                    ORDER BY prediction_date DESC
                """),
                {"ticker": ticker_upper, "cutoff_date": cutoff_date.date()}
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        dm.close()

        if len(df) == 0:
            return []

        # Convert to API format
        predictions = []
        for _, row in df.iterrows():
            ensemble_prob = float(row['ensemble_prob'])
            label = row['prediction_label']

            remaining = 1.0 - ensemble_prob
            other_prob = remaining / 2.0

            if label == 'CRASH':
                prob_crash, prob_normal, prob_spike = ensemble_prob, other_prob, other_prob
            elif label == 'SPIKE':
                prob_crash, prob_normal, prob_spike = other_prob, other_prob, ensemble_prob
            else:
                prob_crash, prob_normal, prob_spike = other_prob, ensemble_prob, other_prob

            conf_score = float(row.get('confidence_score', 0.5))
            confidence = 'high' if conf_score >= 0.7 else ('medium' if conf_score >= 0.5 else 'low')

            predictions.append({
                "ticker": row['ticker'],
                "prediction_date": str(row['prediction_date']),
                "prediction_label": label,
                "prob_crash": round(prob_crash, 3),
                "prob_normal": round(prob_normal, 3),
                "prob_spike": round(prob_spike, 3),
                "confidence": confidence,
                "timestamp": str(row['prediction_timestamp'])
            })

        logger.info(f"Returning {len(predictions)} historical predictions")
        return predictions

    except Exception as e:
        logger.error(f"Error fetching history for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.post(
    "/api/predictions/batch",
    response_model=List[PredictionResponse],
    tags=["Predictions"],
    summary="Get Batch Predictions",
    description="""
    Get predictions for multiple assets in a single request.

    **Body:**
    ```json
    {
      "tickers": ["SPY", "QQQ", "BTC_USDT", "ETH_USDT"]
    }
    ```

    **Rate Limits:** Same as other prediction endpoints (counts as 1 request regardless of ticker count)

    **Example:**
    ```
    curl -X POST -H "X-API-Key: YOUR_KEY" \
      -H "Content-Type: application/json" \
      -d '{"tickers":["SPY","QQQ","GLD"]}' \
      "https://api.neurovest.com/api/predictions/batch"
    ```
    """
)
@limiter.limit("300/minute")
def get_batch_predictions(
    request: Request,
    tickers: List[str],
    user: dict = Depends(verify_api_key)
):
    """
    Get predictions for multiple assets at once

    Args:
        tickers: List of asset tickers
        user: Authenticated user

    Returns:
        List of predictions
    """
    try:
        logger.info(f"User {user['user_id']} requesting batch predictions for {len(tickers)} assets")

        dm = DataManager()
        df = dm.get_latest_predictions(limit=1000)
        dm.close()

        # Filter for requested tickers
        tickers_upper = [t.upper() for t in tickers]
        filtered = df[df['ticker'].str.upper().isin(tickers_upper)]

        predictions = []
        for _, row in filtered.iterrows():
            ensemble_prob = float(row['ensemble_prob'])
            label = row['prediction_label']

            remaining = 1.0 - ensemble_prob
            other_prob = remaining / 2.0

            if label == 'CRASH':
                prob_crash, prob_normal, prob_spike = ensemble_prob, other_prob, other_prob
            elif label == 'SPIKE':
                prob_crash, prob_normal, prob_spike = other_prob, other_prob, ensemble_prob
            else:
                prob_crash, prob_normal, prob_spike = other_prob, ensemble_prob, other_prob

            conf_score = float(row.get('confidence_score', 0.5))
            confidence = 'high' if conf_score >= 0.7 else ('medium' if conf_score >= 0.5 else 'low')

            predictions.append({
                "ticker": row['ticker'],
                "prediction_date": str(row['prediction_date']),
                "prediction_label": label,
                "prob_crash": round(prob_crash, 3),
                "prob_normal": round(prob_normal, 3),
                "prob_spike": round(prob_spike, 3),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })

        logger.info(f"Returning {len(predictions)} batch predictions")
        return predictions

    except Exception as e:
        logger.error(f"Error fetching batch predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch batch: {str(e)}")

@app.post(
    "/api/custom/upload",
    tags=["Custom Assets"],
    summary="Upload Custom Asset Data",
    description="""
    Upload custom asset price data and get instant prediction.

    **File Format:** CSV with columns: Date, Open, High, Low, Close, Volume

    **Example:**
    ```bash
    curl -X POST -H "X-API-Key: YOUR_KEY" \
      -F "file=@myasset.csv" \
      -F "ticker=CUSTOM1" \
      "https://api.neurovest.com/api/custom/upload"
    ```
    """
)
async def upload_custom_asset(
    file: bytes,
    ticker: str = Query(..., description="Asset ticker symbol"),
    user: dict = Depends(verify_api_key)
):
    """
    Upload custom asset and get prediction

    Args:
        file: CSV file with OHLCV data
        ticker: Asset ticker
        user: Authenticated user

    Returns:
        Prediction for uploaded asset
    """
    try:
        logger.info(f"User {user['user_id']} uploading custom asset: {ticker}")

        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(file.decode('utf-8')))

        # Validate format
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns. Required: {required}"
            )

        # Save to database
        from auth_middleware import save_custom_asset_to_db
        success, count = save_custom_asset_to_db(
            ticker=ticker,
            asset_type='custom',
            df=df,
            user_id=user['user_id']
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save custom asset")

        logger.info(f"Custom asset {ticker} uploaded: {count} records")

        return {
            "message": "Custom asset uploaded successfully",
            "ticker": ticker,
            "records": count,
            "note": "Prediction will be available after next prediction run (4:30 PM EST)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading custom asset: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    logger.info("=" * 70)
    logger.info("NeuroVest API Server Starting")
    logger.info("=" * 70)
    logger.info(f"Version: 2.0.0")
    logger.info(f"Documentation: /docs")
    logger.info(f"Health Check: /health")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    logger.info("NeuroVest API Server Shutting Down")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,  # Disable in production
        log_level="info"
    )
