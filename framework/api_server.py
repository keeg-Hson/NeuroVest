#!/usr/bin/env python3
"""
NeuroVest Prediction API

RESTful API for accessing model predictions and results

Endpoints:
    GET  /health                    - Health check
    GET  /assets                    - List all configured assets
    GET  /models                    - List all trained models
    GET  /predict/{asset}           - Get latest prediction for asset
    GET  /predict/{asset}/history   - Get prediction history
    GET  /results/per-asset         - Get all per-asset results
    GET  /results/macro              - Get all macro model results
    GET  /results/summary            - Get training summary
    POST /train/{asset}             - Trigger training for asset
    POST /refresh/{asset}           - Refresh data for asset

Usage:
    python framework/api_server.py
    # Server runs on http://localhost:8000
    # API docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path
import pandas as pd
import joblib
import json
from datetime import datetime
import uvicorn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset_manager import AssetManager


app = FastAPI(
    title="NeuroVest Prediction API",
    description="API for accessing multi-asset ML trading predictions",
    version="1.0.0"
)


class PredictionResponse(BaseModel):
    asset: str
    timestamp: str
    prediction: int  # 1 = long, 0 = neutral/short
    probability: float
    confidence: str  # low, medium, high
    model_type: str  # per-asset or macro
    models_agree: bool


class AssetInfo(BaseModel):
    ticker: str
    name: str
    type: str
    category: str
    threshold: float
    enabled: bool


class ModelInfo(BaseModel):
    name: str
    type: str  # per-asset or macro
    accuracy: Optional[float]
    trained_at: Optional[str]


# Initialize
manager = AssetManager()
models_dir = Path("models")
results_dir = Path("results")
data_dir = Path("data_cache")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/assets", response_model=List[AssetInfo])
async def list_assets(asset_type: Optional[str] = None, enabled_only: bool = True):
    """
    List all configured assets

    Args:
        asset_type: Filter by type (equity, bond, commodity, crypto)
        enabled_only: Only return enabled assets
    """
    if asset_type:
        assets = manager.get_assets_by_type(asset_type, enabled_only=enabled_only)
    else:
        assets = manager.get_all_assets(enabled_only=enabled_only)

    return [
        AssetInfo(
            ticker=a.ticker,
            name=a.name,
            type=a.asset_type,
            category=a.category,
            threshold=a.threshold,
            enabled=a.enabled
        )
        for a in assets
    ]


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all trained models"""
    models = []

    # Find per-asset models
    for model_file in models_dir.glob("*_xgboost.pkl"):
        asset_name = model_file.stem.replace('_xgboost', '').upper()

        # Try to load results
        results_files = list(results_dir.glob("per_asset_results_*.csv"))
        accuracy = None
        trained_at = None

        if results_files:
            latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_results)
            asset_row = df[df['asset'] == asset_name]
            if not asset_row.empty:
                accuracy = asset_row.iloc[0]['ensemble_acc']
                trained_at = asset_row.iloc[0].get('timestamp')

        models.append(ModelInfo(
            name=asset_name,
            type="per-asset",
            accuracy=accuracy,
            trained_at=trained_at
        ))

    # Find macro models
    for model_file in models_dir.glob("macro_*_xgboost.pkl"):
        group_name = model_file.stem.replace('macro_', '').replace('_xgboost', '')

        results_files = list(results_dir.glob("macro_results_*.csv"))
        accuracy = None
        trained_at = None

        if results_files:
            latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_results)
            group_row = df[df['group'] == group_name]
            if not group_row.empty:
                accuracy = group_row.iloc[0]['ensemble_acc']
                trained_at = group_row.iloc[0].get('timestamp')

        models.append(ModelInfo(
            name=group_name,
            type="macro",
            accuracy=accuracy,
            trained_at=trained_at
        ))

    return models


@app.get("/predict/{asset}", response_model=PredictionResponse)
async def get_prediction(asset: str, model_type: str = "per-asset"):
    """
    Get latest prediction for asset

    Args:
        asset: Asset ticker (e.g., SPY, BTC/USDT)
        model_type: Use per-asset or macro model
    """
    # Validate asset exists
    asset_obj = manager.get_asset(asset)
    if not asset_obj:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset}")

    # Load models
    if model_type == "per-asset":
        model_prefix = asset.replace('/', '_').lower()
    else:
        # Find appropriate macro group
        if asset_obj.asset_type == 'equity':
            model_prefix = "macro_all_equities"
        elif asset_obj.asset_type == 'crypto':
            model_prefix = "macro_all_crypto"
        else:
            model_prefix = f"macro_{asset_obj.asset_type}s_only"

    # Load ensemble models
    try:
        xgb_model = joblib.load(models_dir / f"{model_prefix}_xgboost.pkl")
        lgb_model = joblib.load(models_dir / f"{model_prefix}_lightgbm.pkl")
        cat_model = joblib.load(models_dir / f"{model_prefix}_catboost.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Models not found for {asset}. Train first.")

    # Load latest data
    if asset_obj.asset_type == 'crypto':
        filename = asset.replace('/', '_') + '_1d.csv'
    else:
        filename = f"{asset}_1d.csv"

    data_file = data_dir / filename
    if not data_file.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")

    df = pd.read_csv(data_file)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No data available")

    # Get features from latest row (simplified - would use create_features() in production)
    latest_row = df.iloc[-1:]

    # Generate actual predictions using models
    try:
        # Import feature building functions
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils import add_features, finalize_features

        # Build features from data
        df_feat, feature_cols = add_features(df)

        # Get latest row with features
        # Use last row and fill any NaN
        X = df_feat[feature_cols].iloc[-1:].fillna(0)
        X = X.replace([float('inf'), float('-inf')], 0)

        # Get predictions from each model
        xgb_pred = int(xgb_model.predict(X)[0])
        lgb_pred = int(lgb_model.predict(X)[0])
        cat_pred = int(cat_model.predict(X)[0])

        # Get probability from XGBoost
        if hasattr(xgb_model, 'predict_proba'):
            xgb_proba = xgb_model.predict_proba(X)[0]
            xgb_prob = float(xgb_proba[1]) if len(xgb_proba) > 1 else float(xgb_proba[0])
        else:
            xgb_prob = 0.5

        # Ensemble (majority vote)
        ensemble_pred = int(round((xgb_pred + lgb_pred + cat_pred) / 3))
        models_agree = (xgb_pred == lgb_pred == cat_pred)

        # Confidence level
        if xgb_prob >= 0.65:
            confidence = "high"
        elif xgb_prob >= 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResponse(
            asset=asset,
            timestamp=datetime.now().isoformat(),
            prediction=ensemble_pred,
            probability=xgb_prob,
            confidence=confidence,
            model_type=model_type,
            models_agree=models_agree
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/results/per-asset")
async def get_per_asset_results():
    """Get all per-asset model results"""
    results_files = list(results_dir.glob("per_asset_results_*.csv"))

    if not results_files:
        return {"results": [], "message": "No per-asset results found"}

    # Load latest results
    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_results)

    return {
        "file": str(latest_results),
        "timestamp": datetime.fromtimestamp(latest_results.stat().st_mtime).isoformat(),
        "count": len(df),
        "results": df.to_dict(orient='records')
    }


@app.get("/results/macro")
async def get_macro_results():
    """Get all macro model results"""
    results_files = list(results_dir.glob("macro_results_*.csv"))

    if not results_files:
        return {"results": [], "message": "No macro results found"}

    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_results)

    return {
        "file": str(latest_results),
        "timestamp": datetime.fromtimestamp(latest_results.stat().st_mtime).isoformat(),
        "count": len(df),
        "results": df.to_dict(orient='records')
    }


@app.get("/results/summary")
async def get_summary():
    """Get training summary"""
    summary_files = list(results_dir.glob("training_summary_*.json"))

    if not summary_files:
        return {"message": "No training summary found"}

    latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)

    with open(latest_summary) as f:
        summary = json.load(f)

    return summary


@app.get("/")
async def root():
    """API root - redirect to docs"""
    return {
        "message": "NeuroVest Prediction API",
        "docs": "/docs",
        "health": "/health",
        "assets": "/assets",
        "models": "/models"
    }


if __name__ == "__main__":
    # Get settings from config
    settings = manager.get_settings()

    host = settings.get('api_host', '0.0.0.0')
    port = settings.get('api_port', 8000)

    print("=" * 80)
    print("NEUROVEST PREDICTION API")
    print("=" * 80)
    print(f"\n✓ Server starting on http://{host}:{port}")
    print(f"✓ API Documentation: http://localhost:{port}/docs")
    print(f"✓ Interactive API: http://localhost:{port}/redoc")
    print("\n" + "=" * 80)

    uvicorn.run(app, host=host, port=port)
