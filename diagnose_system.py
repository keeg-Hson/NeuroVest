#!/usr/bin/env python3
"""
NeuroVest System Diagnostics

Comprehensive testing and diagnostics for the entire prediction pipeline.
Run this to identify issues with training, predictions, and backtesting.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

print("=" * 80)
print("NEUROVEST SYSTEM DIAGNOSTICS")
print("=" * 80)

# =============================================================================
# 1. CHECK DATA FILES
# =============================================================================

print("\n📁 1. DATA FILES")
print("-" * 80)

data_dir = Path("data")
cache_dir = Path("data_cache")

# Check SPY data
spy_main = data_dir / "SPY.csv"
spy_cache = cache_dir / "SPY_1d.csv"

if spy_main.exists():
    df = pd.read_csv(spy_main)
    print(f"✓ data/SPY.csv: {len(df):,} rows")
else:
    print(f"✗ data/SPY.csv: NOT FOUND")

if spy_cache.exists():
    df = pd.read_csv(spy_cache)
    print(f"✓ data_cache/SPY_1d.csv: {len(df):,} rows")
else:
    print(f"  data_cache/SPY_1d.csv: NOT FOUND (optional)")

# Check crypto data
crypto_files = ['BTC_USDT_1d.csv', 'ETH_USDT_1d.csv', 'SOL_USDT_1d.csv']
for f in crypto_files:
    path = cache_dir / f
    if path.exists():
        df = pd.read_csv(path)
        print(f"✓ data_cache/{f}: {len(df):,} rows")
    else:
        print(f"✗ data_cache/{f}: NOT FOUND")

# =============================================================================
# 2. CHECK MODELS
# =============================================================================

print("\n🤖 2. MODEL FILES")
print("-" * 80)

models_dir = Path("models")
required_models = [
    'xgboost_multi_asset.pkl',
    'lightgbm_multi_asset.pkl',
    'catboost_multi_asset.pkl',
]

for model_file in required_models:
    path = models_dir / model_file
    if path.exists():
        model = joblib.load(path)
        print(f"✓ {model_file}: loaded successfully")
    else:
        print(f"✗ {model_file}: NOT FOUND - run train_multi_asset.py")

# Count per-asset models
per_asset_models = list(models_dir.glob("*_xgboost.pkl"))
print(f"\n  Per-asset models found: {len(per_asset_models)}")

# =============================================================================
# 3. CHECK PREDICTIONS
# =============================================================================

print("\n📊 3. PREDICTIONS")
print("-" * 80)

logs_dir = Path("logs")
pred_files = [
    'labeled_predictions.csv',
    'daily_predictions.csv',
]

for pred_file in pred_files:
    path = logs_dir / pred_file
    if path.exists():
        df = pd.read_csv(path)
        print(f"✓ logs/{pred_file}: {len(df):,} rows")

        if 'Prediction' in df.columns:
            pred_dist = df['Prediction'].value_counts().sort_index()
            total = len(df)
            print(f"  Prediction distribution:")
            for cls, count in pred_dist.items():
                label = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}.get(cls, f'Class {cls}')
                print(f"    {label} ({cls}): {count:,} ({100*count/total:.1f}%)")

        if 'Proba' in df.columns:
            proba = df['Proba'].dropna()
            print(f"  Probability stats:")
            print(f"    Min: {proba.min():.3f}, Max: {proba.max():.3f}")
            print(f"    Mean: {proba.mean():.3f}, Median: {proba.median():.3f}")
            print(f"    Std: {proba.std():.3f}")
    else:
        print(f"✗ logs/{pred_file}: NOT FOUND - run predict_multi_asset_ensemble.py")

# =============================================================================
# 4. ANALYZE LABEL DISTRIBUTION
# =============================================================================

print("\n🏷️  4. LABEL DISTRIBUTION ANALYSIS")
print("-" * 80)

try:
    import sys
    sys.path.insert(0, 'framework')
    from utils import add_features, add_forward_returns_and_labels

    # Load SPY data
    df = pd.read_csv("data/SPY.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    # Test different thresholds
    thresholds = [0.004, 0.005, 0.006, 0.007, 0.008]

    print(f"Testing label distribution with different thresholds:")
    print(f"{'Threshold':<12} {'Class 0':<15} {'Class 1':<15} {'Baseline':<12}")
    print("-" * 54)

    for thresh in thresholds:
        df_test = df.copy()
        df_test, _ = add_features(df_test)
        df_test = add_forward_returns_and_labels(
            df_test, price_col='Close', horizon=1,
            pos_threshold=thresh, fee_bps=2.0, slippage_bps=3.0
        )

        if 'y' in df_test.columns:
            labels = df_test['y'].dropna()
            class_0 = (labels == 0).sum()
            class_1 = (labels == 1).sum()
            total = len(labels)
            baseline = max(class_0, class_1) / total

            print(f"{thresh*100:.1f}%{' '*8} {class_0:,} ({100*class_0/total:.1f}%){' '*3} {class_1:,} ({100*class_1/total:.1f}%){' '*3} {100*baseline:.1f}%")

except Exception as e:
    print(f"✗ Error analyzing labels: {e}")

# =============================================================================
# 5. TEST PREDICTION PIPELINE
# =============================================================================

print("\n🔬 5. PREDICTION PIPELINE TEST")
print("-" * 80)

try:
    # Load models
    xgb_model = joblib.load("models/xgboost_multi_asset.pkl")
    lgb_model = joblib.load("models/lightgbm_multi_asset.pkl")
    cat_model = joblib.load("models/catboost_multi_asset.pkl")

    # Load features list
    features_file = Path("models/multi_asset_features.txt")
    if features_file.exists():
        with open(features_file) as f:
            saved_feats = [line.strip() for line in f if line.strip()]
        print(f"✓ Feature list loaded: {len(saved_feats)} features")
    else:
        print(f"✗ Feature list not found: {features_file}")
        saved_feats = None

    # Load and prepare data
    df = pd.read_csv("data/SPY.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df, feature_cols = add_features(df)

    # Add asset_type features (required by multi-asset models)
    df['asset_type_stock'] = 1
    df['asset_type_crypto'] = 0

    if saved_feats:
        # Check feature alignment
        missing = set(saved_feats) - set(df.columns)
        extra = set(feature_cols) - set(saved_feats)

        if missing:
            print(f"⚠️  Missing features: {len(missing)}")
            for feat in missing:
                df[feat] = 0.0
        else:
            print(f"✓ All saved features present in data")

        # Prepare features
        X = df[saved_feats].copy()
        X = X.ffill().fillna(0)

        # Get predictions
        xgb_prob = xgb_model.predict_proba(X)[:, 1]
        lgb_prob = lgb_model.predict_proba(X)[:, 1]
        cat_prob = cat_model.predict_proba(X)[:, 1]

        ensemble_prob = (xgb_prob + lgb_prob + cat_prob) / 3

        print(f"\n  Test predictions on {len(X):,} rows:")
        print(f"    XGBoost prob range: {xgb_prob.min():.3f} - {xgb_prob.max():.3f}")
        print(f"    LightGBM prob range: {lgb_prob.min():.3f} - {lgb_prob.max():.3f}")
        print(f"    CatBoost prob range: {cat_prob.min():.3f} - {cat_prob.max():.3f}")
        print(f"    Ensemble prob range: {ensemble_prob.min():.3f} - {ensemble_prob.max():.3f}")
        print(f"    Ensemble mean: {ensemble_prob.mean():.3f}")

        # Calculate thresholds
        crash_thresh = np.percentile(ensemble_prob, 30)
        spike_thresh = np.percentile(ensemble_prob, 70)

        pred_012 = np.where(
            ensemble_prob >= spike_thresh, 2,
            np.where(ensemble_prob < crash_thresh, 0, 1)
        )

        unique, counts = np.unique(pred_012, return_counts=True)
        print(f"\n  Predicted distribution (percentile-based):")
        for cls, count in zip(unique, counts):
            label = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}.get(cls, f'Class {cls}')
            print(f"    {label}: {count:,} ({100*count/len(pred_012):.1f}%)")

except Exception as e:
    print(f"✗ Error in pipeline test: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 6. BACKTEST SIGNAL CHECK
# =============================================================================

print("\n📈 6. BACKTEST SIGNAL CHECK")
print("-" * 80)

try:
    pred_path = logs_dir / "daily_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)

        # Check for tradeable signals
        if 'Prediction' in df.columns:
            spike_signals = (df['Prediction'] == 2).sum()
            crash_signals = (df['Prediction'] == 0).sum()
            normal_signals = (df['Prediction'] == 1).sum()

            print(f"Total rows: {len(df):,}")
            print(f"SPIKE signals (long): {spike_signals:,}")
            print(f"CRASH signals (short): {crash_signals:,}")
            print(f"NORMAL signals (hold): {normal_signals:,}")

            if spike_signals < 100:
                print(f"\n⚠️  WARNING: Only {spike_signals} SPIKE signals")
                print(f"   Backtest will have very few trades!")
                print(f"   Consider lowering spike_threshold in predict_multi_asset_ensemble.py")

        # Check date range
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            print(f"\nDate range: {dates.min()} to {dates.max()}")
            print(f"Trading days: {len(dates):,}")

except Exception as e:
    print(f"✗ Error checking backtest signals: {e}")

# =============================================================================
# 7. RECOMMENDATIONS
# =============================================================================

print("\n💡 7. RECOMMENDATIONS")
print("-" * 80)

recommendations = []

# Check if models exist
if not (models_dir / "xgboost_multi_asset.pkl").exists():
    recommendations.append("Run: python3 train_multi_asset.py")

# Check if predictions exist
if not (logs_dir / "daily_predictions.csv").exists():
    recommendations.append("Run: python3 predict_multi_asset_ensemble.py")

# Check prediction distribution
try:
    pred_df = pd.read_csv(logs_dir / "daily_predictions.csv")
    if 'Prediction' in pred_df.columns:
        spike_count = (pred_df['Prediction'] == 2).sum()
        if spike_count < 100:
            recommendations.append("Regenerate predictions - too few SPIKE signals for meaningful backtest")
except:
    pass

if recommendations:
    print("Actions needed:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("✓ System appears correctly configured!")
    print("\nTest commands:")
    print("  python3 train_multi_asset.py          # Train models")
    print("  python3 predict_multi_asset_ensemble.py  # Generate predictions")
    print("  python3 backtest.py                   # Run backtest")
    print("  python3 evaluate.py                   # Evaluate predictions")

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
