#!/usr/bin/env python3
"""
Multi-Asset Prediction Generator
Generates predictions for ALL assets in the database (not just SPY)

Usage:
    python3 predict_all_assets.py
    python3 predict_all_assets.py --assets SPY QQQ BTC_USDT  # Specific assets only
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from config import LOGS_DIR, MODELS_DIR
from utils import add_features, finalize_features
from core.data_manager_postgres import DataManager

print("=" * 80)
print("MULTI-ASSET PREDICTION GENERATOR")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--assets', nargs='+', help='Specific assets to predict (default: all)')
args = parser.parse_args()

# Load models (check both naming conventions)
print("\n📥 Loading ensemble models...")
models = {}
for name in ['xgboost', 'lightgbm', 'catboost']:
    # Try new naming convention first (train_unified.py)
    filepath = MODELS_DIR / f"multi_asset_{name}.pkl"
    if not filepath.exists():
        # Fall back to old naming convention
        filepath = MODELS_DIR / f"{name}_multi_asset.pkl"
    if filepath.exists():
        loaded = joblib.load(filepath)
        # Handle wrapper dict format (from BaseModel.save())
        if isinstance(loaded, dict) and 'model' in loaded:
            models[name] = loaded['model']
            print(f"   ✓ {name} (unwrapped from dict)")
        else:
            models[name] = loaded
            print(f"   ✓ {name}")

if len(models) == 0:
    raise SystemExit("❌ No models found. Run train_unified.py first.")

# Load feature list (check both naming conventions)
feature_file = MODELS_DIR / "multi_asset_features.txt"
if not feature_file.exists():
    feature_file = MODELS_DIR / "multi_asset_features.txt"  # Same for both conventions
if feature_file.exists():
    saved_feats = [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]
    print(f"   Features: {len(saved_feats)}")
else:
    raise SystemExit(f"❌ Feature list not found: {feature_file}")

# Initialize data manager
dm = DataManager()

# Get assets to predict
if args.assets:
    # User-specified assets
    assets_to_predict = [(ticker, None) for ticker in args.assets]
    print(f"\n📊 Predicting {len(assets_to_predict)} specified assets...")
else:
    # All assets from database
    assets_to_predict = dm.get_all_assets()
    print(f"\n📊 Predicting ALL {len(assets_to_predict)} assets from database...")

# Prediction results
all_predictions = []
successful = 0
failed = 0

for ticker, asset_type in assets_to_predict:
    print(f"\n{'='*80}")
    print(f"Processing: {ticker}")
    print(f"{'='*80}")

    try:
        # Load asset data
        print(f"   📥 Loading data...")
        raw = dm.get_data(ticker)

        if raw is None or len(raw) == 0:
            print(f"   ⚠️  No data available - skipping")
            failed += 1
            continue

        # Prepare data
        if 'timestamp' in raw.columns:
            raw['Date'] = pd.to_datetime(raw['timestamp'])
            raw = raw.drop('timestamp', axis=1)

        if 'close' in raw.columns:
            raw = raw.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })

        raw['Date'] = pd.to_datetime(raw['Date'])
        raw = raw.sort_values('Date').reset_index(drop=True)

        print(f"   Rows: {len(raw)}, Date range: {raw['Date'].min()} to {raw['Date'].max()}")

        # Add features
        print(f"   🔧 Building features...")
        df_feat, _ = add_features(raw)

        # Add asset_type features
        if asset_type == 'crypto':
            df_feat["asset_type_stock"] = 0
            df_feat["asset_type_crypto"] = 1
        else:
            df_feat["asset_type_stock"] = 1
            df_feat["asset_type_crypto"] = 0

        df_feat = finalize_features(df_feat, saved_feats)

        if "Date" not in df_feat.columns:
            df_feat = df_feat.reset_index().rename(columns={"index": "Date"})

        # Align features
        for feat in saved_feats:
            if feat not in df_feat.columns:
                df_feat[feat] = 0.0

        X = df_feat[saved_feats].copy()
        dates = df_feat["Date"].values

        # Fill NaN
        X = X.ffill().fillna(0)

        print(f"   ✓ Features ready: {len(X)} rows, {len(saved_feats)} features")

        # Generate predictions
        print(f"   🤖 Generating predictions...")

        predictions = {}
        for name, model in models.items():
            try:
                pred = model.predict(X)
                prob = model.predict_proba(X)
                predictions[name] = {'pred': pred, 'prob': prob}
                print(f"      ✓ {name}")
            except Exception as e:
                print(f"      ✗ {name} failed: {e}")

        if len(predictions) == 0:
            print(f"   ⚠️  All models failed - skipping")
            failed += 1
            continue

        # Ensemble voting
        print(f"   🗳️  Ensemble voting...")

        all_preds = np.array([p['pred'] for p in predictions.values()])

        # Ensemble probabilities (average across models)
        all_probs = np.array([p['prob'] for p in predictions.values()])
        ensemble_prob = all_probs.mean(axis=0)

        # FIX: Convert binary model outputs (0/1) to 3-class (0=CRASH, 1=NORMAL, 2=SPIKE)
        # Strategy: Use percentile-based thresholds on probability distribution
        # This ensures balanced class distribution regardless of model calibration

        # Check if models are binary (2 classes) or ternary (3 classes)
        n_classes = ensemble_prob.shape[1] if len(ensemble_prob.shape) > 1 else 1

        if n_classes == 2:
            # Binary models: Use probability of positive class (class 1)
            prob_positive = ensemble_prob[:, 1]

            # Use percentile-based thresholds (same as predict_multi_asset_ensemble.py)
            crash_threshold = np.percentile(prob_positive, 30)   # Bottom 30% = CRASH
            spike_threshold = np.percentile(prob_positive, 70)   # Top 30% = SPIKE

            # Convert to 3-class predictions
            ensemble_pred_3class = np.where(
                prob_positive >= spike_threshold, 2,  # SPIKE
                np.where(prob_positive < crash_threshold, 0, 1)  # CRASH or NORMAL
            )

            # Calculate confidence based on distance from thresholds
            spike_conf = np.clip((prob_positive - spike_threshold) / (1 - spike_threshold + 0.001), 0, 1)
            crash_conf = np.clip((crash_threshold - prob_positive) / (crash_threshold + 0.001), 0, 1)
            confidence_array = np.maximum(spike_conf, crash_conf)

            print(f"   Binary models detected - using percentile conversion")
            print(f"   Crash threshold (30th): {crash_threshold:.3f}")
            print(f"   Spike threshold (70th): {spike_threshold:.3f}")

        else:
            # Ternary models: Use direct predictions
            ensemble_pred_3class = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=all_preds
            )
            # Confidence is max probability
            confidence_array = ensemble_prob.max(axis=1)

            print(f"   Ternary models detected - using direct predictions")

        # Get most recent prediction
        latest_idx = -1
        latest_date = dates[latest_idx]
        latest_pred = ensemble_pred_3class[latest_idx]

        # Map prediction to label
        label_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}
        label = label_map[latest_pred]

        # Confidence score
        if n_classes == 2:
            confidence_score = confidence_array[latest_idx]
            latest_prob_value = prob_positive[latest_idx]
        else:
            confidence_score = confidence_array[latest_idx]
            latest_prob_value = ensemble_prob[latest_idx, latest_pred]

        print(f"   ✓ Prediction: {label} (confidence: {confidence_score:.2%})")

        # Save to database
        # Extract individual model probabilities
        if n_classes == 2:
            # Binary models: use probability of positive class
            xgb_prob = predictions.get('xgboost', {}).get('prob', [[0, 0]])[latest_idx][1]
            lgb_prob = predictions.get('lightgbm', {}).get('prob', [[0, 0]])[latest_idx][1]
            cat_prob = predictions.get('catboost', {}).get('prob', [[0, 0]])[latest_idx][1]
        else:
            # Ternary models: use probability of predicted class
            xgb_prob = predictions.get('xgboost', {}).get('prob', [[0,0,0]])[latest_idx][latest_pred]
            lgb_prob = predictions.get('lightgbm', {}).get('prob', [[0,0,0]])[latest_idx][latest_pred]
            cat_prob = predictions.get('catboost', {}).get('prob', [[0,0,0]])[latest_idx][latest_pred]

        pred_df = pd.DataFrame([{
            'ticker': ticker,
            'prediction_date': pd.to_datetime(latest_date).date(),
            'ensemble_prob': latest_prob_value,
            'prediction_label': label,
            'xgboost_prob': xgb_prob,
            'lightgbm_prob': lgb_prob,
            'catboost_prob': cat_prob,
            'model_agreement': len(set(all_preds[:, latest_idx])) == 1 if n_classes > 2 else True,
            'confidence_score': confidence_score
        }])

        rows_saved = dm.save_predictions(pred_df)
        print(f"   ✓ Saved to PostgreSQL: {rows_saved} predictions")

        all_predictions.append({
            'ticker': ticker,
            'prediction': label,
            'confidence': f"{confidence_score:.2%}"
        })

        successful += 1

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

dm.close()

# Summary
print(f"\n{'='*80}")
print("PREDICTION SUMMARY")
print(f"{'='*80}")
print(f"Total assets: {len(assets_to_predict)}")
print(f"✓ Successful: {successful}")
print(f"✗ Failed: {failed}")
print(f"{'='*80}\n")

if len(all_predictions) > 0:
    print("Latest Predictions:")
    print(pd.DataFrame(all_predictions).to_string(index=False))
    print()

print(f"\n✅ Done! Predictions available via API:")
print(f"   curl -H 'X-API-Key: YOUR_KEY' https://neurovest-api-production-f8dc.up.railway.app/api/predictions")
