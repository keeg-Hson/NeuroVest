#!/usr/bin/env python3
"""
Per-Asset Model Training

Trains separate models for each asset independently.
Use this when you want asset-specific models rather than a unified multi-asset model.

Output:
- models/spy_xgboost.pkl, models/spy_lightgbm.pkl, models/spy_catboost.pkl
- models/qqq_xgboost.pkl, models/qqq_lightgbm.pkl, models/qqq_catboost.pkl
- etc.

Compare with train_multi_asset.py to see which approach works better.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

from utils import add_features, add_forward_returns_and_labels
from config import TRAIN_CFG
from asset_features import filter_features_for_asset, get_asset_type

print("=" * 80)
print("PER-ASSET MODEL TRAINING")
print("=" * 80)
print("\nTraining separate models for each asset independently")
print("This allows asset-specific patterns vs multi-asset generalization\n")

DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Assets to train on
ASSETS = {
    'SPY': {'path': DATA_DIR / "SPY.csv", 'type': 'stock', 'threshold': 0.005},
}

# Add equity ETFs if available
equity_etfs = ['QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLK', 'XLE']
for ticker in equity_etfs:
    filepath = CACHE_DIR / f"{ticker}_1d.csv"
    if filepath.exists():
        ASSETS[ticker] = {'path': filepath, 'type': 'stock', 'threshold': 0.005}

# Add crypto if available (separate thresholds)
crypto_assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
for ticker in crypto_assets:
    filepath = CACHE_DIR / f"{ticker}_1d.csv"
    if filepath.exists():
        ASSETS[ticker] = {'path': filepath, 'type': 'crypto', 'threshold': 0.020}

print(f"Found {len(ASSETS)} assets to train on: {list(ASSETS.keys())}\n")

# Results tracking
all_results = []

# Train each asset separately
for asset_name, asset_info in ASSETS.items():
    print("=" * 80)
    print(f"TRAINING: {asset_name}")
    print("=" * 80)

    try:
        # Load data
        df = pd.read_csv(asset_info['path'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Add features
        df, features = add_features(df)

        # CRITICAL: Verify no forward-looking columns in features (leakage prevention)
        leakage_cols = {'fwd_price', 'fwd_ret_raw', 'fwd_ret_net', 'horizon_forward',
                        'y', 'Label', 'Forward_Return', 'Future_Return', 'Next_Close'}
        leaky_features = [f for f in features if f in leakage_cols or f.startswith('fwd_')]
        if leaky_features:
            print(f"   ⚠️  WARNING: Removing leaky features: {leaky_features}")
            features = [f for f in features if f not in leakage_cols and not f.startswith('fwd_')]

        # ASSET-AWARE FEATURE FILTERING: Remove irrelevant features for this asset type
        # This improves accuracy by excluding VIX/Sector features for crypto, etc.
        original_count = len(features)
        features = filter_features_for_asset(features, asset_name, verbose=False)
        if len(features) < original_count:
            print(f"   📊 Filtered {original_count - len(features)} irrelevant features for {get_asset_type(asset_name)} asset")

        print(f"   {len(df)} rows, {len(features)} features")

        # Add labels
        df = add_forward_returns_and_labels(
            df,
            price_col="Close",
            horizon=TRAIN_CFG["horizon"],
            pos_threshold=asset_info['threshold'],
            fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
            slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
        )

        # Prepare data
        df = df[features + ['y']].copy()
        df = df.dropna(subset=['y'])
        df = df.fillna(0)

        # Train/test split (80/20 time-based)
        df = df.sort_index()
        test_size = int(len(df) * 0.2)
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        X_train = train_df[features]
        y_train = train_df['y']
        X_test = test_df[features]
        y_test = test_df['y']

        print(f"   Train: {len(X_train)} samples ({y_train.sum():.0f} positive)")
        print(f"   Test:  {len(X_test)} samples ({y_test.sum():.0f} positive)")

        # Train models
        models = {}
        results = {'Asset': asset_name, 'Type': asset_info['type'], 'Samples': len(df)}

        # XGBoost
        print(f"\n   [1/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        models['xgboost'] = xgb_model
        results['XGB_Accuracy'] = accuracy_score(y_test, xgb_pred)

        # LightGBM
        print(f"   [2/3] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        models['lightgbm'] = lgb_model
        results['LGB_Accuracy'] = accuracy_score(y_test, lgb_pred)

        # CatBoost
        print(f"   [3/3] Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        cat_model.fit(X_train, y_train)
        cat_pred = cat_model.predict(X_test)
        models['catboost'] = cat_model
        results['CAT_Accuracy'] = accuracy_score(y_test, cat_pred)

        # Ensemble
        ensemble_pred = ((xgb_pred + lgb_pred + cat_pred) >= 2).astype(int)
        results['Ensemble_Accuracy'] = accuracy_score(y_test, ensemble_pred)

        # Save models
        print(f"\n   💾 Saving models...")
        for model_type, model in models.items():
            model_path = MODELS_DIR / f"{asset_name.lower()}_{model_type}.pkl"
            joblib.dump(model, model_path)
            print(f"      ✓ {model_path}")

        # Save features
        feature_path = MODELS_DIR / f"{asset_name.lower()}_features.txt"
        with open(feature_path, 'w') as f:
            for feat in features:
                f.write(f"{feat}\n")

        all_results.append(results)

        print(f"\n   ✅ {asset_name} complete:")
        print(f"      XGBoost:  {results['XGB_Accuracy']:.1%}")
        print(f"      LightGBM: {results['LGB_Accuracy']:.1%}")
        print(f"      CatBoost: {results['CAT_Accuracy']:.1%}")
        print(f"      Ensemble: {results['Ensemble_Accuracy']:.1%}")

    except Exception as e:
        print(f"   ✗ Error training {asset_name}: {e}")
        continue

# Summary
print("\n" + "=" * 80)
print("PER-ASSET TRAINING SUMMARY")
print("=" * 80)

if all_results:
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # Save results
    results_path = MODELS_DIR / "per_asset_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to {results_path}")

    # Compare to multi-asset if available
    multi_asset_path = MODELS_DIR / "multi_asset_results.csv"
    if multi_asset_path.exists():
        print("\n" + "=" * 80)
        print("COMPARISON: Per-Asset vs Multi-Asset")
        print("=" * 80)

        multi_df = pd.read_csv(multi_asset_path)

        # Find SPY in per-asset results
        spy_per = results_df[results_df['Asset'] == 'SPY'].iloc[0]

        # Find ensemble in multi-asset
        multi_ensemble = multi_df[multi_df['Model'] == 'Ensemble'].iloc[0]

        print(f"\nSPY Performance:")
        print(f"  Per-Asset Model:   {spy_per['Ensemble_Accuracy']:.1%} accuracy")
        print(f"  Multi-Asset Model: {multi_ensemble['Accuracy']:.1%} accuracy")

        if spy_per['Ensemble_Accuracy'] > multi_ensemble['Accuracy']:
            print(f"\n✅ Per-asset approach is better by {(spy_per['Ensemble_Accuracy'] - multi_ensemble['Accuracy'])*100:.1f} percentage points")
        else:
            print(f"\n✅ Multi-asset approach is better by {(multi_ensemble['Accuracy'] - spy_per['Ensemble_Accuracy'])*100:.1f} percentage points")

else:
    print("⚠️ No models trained successfully")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Compare results with multi-asset approach (train_multi_asset.py)")
print("2. Run backtest with best performing model")
print("3. Update predict script to use per-asset models if better")
