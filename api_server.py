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

from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import logging

from core.data_manager_postgres import DataManager
from auth_middleware import AuthManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# CORS middleware (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
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
    """Validate API key from header"""
    user = AuthManager.validate_api_key(x_api_key)
    if not user:
        logger.warning(f"Invalid API key attempted: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Register at /api/auth/register to get a key."
        )
    logger.info(f"Authenticated user: {user['user_id']}")
    return user

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
        assets = dm.get_all_tickers()

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

    **Example:**
    ```
    curl -H "X-API-Key: YOUR_KEY" https://api.neurovest.com/api/predictions
    ```
    """
)
def get_all_predictions(
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

        logger.info(f"Returning {len(predictions)} predictions")
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

    **Example:**
    ```
    curl -H "X-API-Key: YOUR_KEY" https://api.neurovest.com/api/predictions/SPY
    ```
    """
)
def get_prediction(
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

        response = {
            "ticker": row['ticker'],
            "prediction_date": str(row['prediction_date']),
            "prediction_label": row['prediction_label'],
            "prob_crash": float(row['prob_crash']),
            "prob_normal": float(row['prob_normal']),
            "prob_spike": float(row['prob_spike']),
            "confidence": row['confidence'],
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Returning prediction for {ticker_upper}: {row['prediction_label']}")
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
        assets = dm.get_all_tickers()
        dm.close()

        logger.info(f"Returning {len(assets)} assets")

        return {
            "total": len(assets),
            "assets": sorted(assets)
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
