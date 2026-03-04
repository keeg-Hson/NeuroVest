#!/usr/bin/env python3
"""
Multi-Horizon Trading Signal Training

Trains classification models for multiple prediction horizons (1-day, 3-day, 5-day).
Each horizon gets its own set of models trained to generate trading signals.

Usage:
    python3 train_multi_horizon_signals.py                    # Train all horizons
    python3 train_multi_horizon_signals.py --horizons 1 3     # Train specific horizons
    python3 train_multi_horizon_signals.py --tune             # With hyperparameter tuning

Benefits:
- Different horizons capture different market dynamics
- Short-term (1-day): High noise, day trading
- Medium-term (3-day): Swing trading
- Longer-term (5-day): Position trading, less noise
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

from utils import add_features, add_forward_returns_and_labels
from config import TRAIN_CFG, MODELS_DIR

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Multi-Horizon Trading Signal Training')
parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5],
                    help='Prediction horizons in days (default: 1 3 5)')
parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
parser.add_argument('--tune-fast', action='store_true', help='Quick hyperparameter tuning')
args = parser.parse_args()

DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")

print("=" * 80)
print("MULTI-HORIZON TRADING SIGNAL TRAINING")
print("=" * 80)
print(f"\nHorizons: {args.horizons} days")

# =============================================================================
# LOAD BASE DATA
# =============================================================================

print("\n📥 Loading base data...")

# Load SPY
spy_df = pd.read_csv(DATA_DIR / "SPY.csv")
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df.set_index('Date', inplace=True)
spy_df, feature_cols = add_features(spy_df)
spy_df['asset_type_stock'] = 1
spy_df['asset_type_crypto'] = 0

print(f"   SPY: {len(spy_df)} rows, {len(feature_cols)} features")

# Load additional equity ETFs
equity_assets = ['QQQ', 'IWM', 'DIA', 'VTI']
equity_dfs = []

for asset in equity_assets:
    filepath = CACHE_DIR / f"{asset}_1d.csv"
    if filepath.exists():
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df, _ = add_features(df)
        df['asset_type_stock'] = 1
        df['asset_type_crypto'] = 0
        equity_dfs.append(df)
        print(f"   {asset}: {len(df)} rows")

feature_cols = feature_cols + ['asset_type_stock', 'asset_type_crypto']

# =============================================================================
# TRAIN MODELS FOR EACH HORIZON
# =============================================================================

all_results = []

for horizon in args.horizons:
    print("\n" + "=" * 80)
    print(f"TRAINING HORIZON: {horizon}-DAY")
    print("=" * 80)

    # Calibrate threshold based on horizon
    # Longer horizons need larger thresholds (more price movement expected)
    base_threshold = 0.006  # 0.6% for 1-day
    threshold = base_threshold * np.sqrt(horizon)  # Scale by sqrt of time
    print(f"\nThreshold: {threshold*100:.2f}% (base × √{horizon})")

    # Add labels to SPY with horizon-specific threshold
    spy_horizon = spy_df.copy()
    spy_horizon = add_forward_returns_and_labels(
        spy_horizon,
        price_col="Close",
        horizon=horizon,
        pos_threshold=threshold,
        fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
        slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
    )

    # Add labels to equity ETFs
    all_dfs = [spy_horizon]
    for df in equity_dfs:
        df_horizon = df.copy()
        df_horizon = add_forward_returns_and_labels(
            df_horizon,
            price_col="Close",
            horizon=horizon,
            pos_threshold=threshold,
            fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
            slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
        )
        all_dfs.append(df_horizon)

    # Combine data
    combined_df = pd.concat(all_dfs, axis=0)
    combined_df = combined_df.sort_index()

    # Prepare features
    feat_cols = [c for c in feature_cols if c in combined_df.columns]
    all_cols = feat_cols + ['y', 'fwd_ret_net']
    combined_df = combined_df[all_cols].dropna(subset=['y'])
    combined_df = combined_df.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"\nDataset: {len(combined_df):,} rows, {len(feat_cols)} features")

    # Label distribution
    y_counts = combined_df['y'].value_counts()
    print(f"Labels: {dict(y_counts)}")
    baseline_acc = y_counts.max() / len(combined_df)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Train/test split
    test_size = int(len(combined_df) * 0.2)
    train_df = combined_df.iloc[:-test_size]
    test_df = combined_df.iloc[-test_size:]

    X_train = train_df[feat_cols].values
    y_train = train_df['y'].values
    X_test = test_df[feat_cols].values
    y_test = test_df['y'].values

    # Sample weights
    fwd_rets = train_df['fwd_ret_net'].abs()
    weights = fwd_rets ** TRAIN_CFG.get("weight_power", 1.75)
    weights = weights.clip(
        TRAIN_CFG.get("min_weight", 0.5),
        TRAIN_CFG.get("max_weight", 5.0)
    ).values

    # Clean data
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.nan_to_num(weights, nan=1.0, posinf=5.0, neginf=0.5)

    print(f"\nTrain: {len(train_df):,} samples ({y_train.sum():,} positive)")
    print(f"Test:  {len(test_df):,} samples ({y_test.sum():,} positive)")

    # Train models
    models = {}
    predictions = {}

    if args.tune or args.tune_fast:
        from hyperparameter_tuning import tune_models, save_best_params

        n_iter = 15 if args.tune_fast else 30
        print(f"\n🔧 Hyperparameter tuning (n_iter={n_iter})...")

        tuning_results = tune_models(X_train, y_train, weights, n_iter=n_iter, cv_splits=5)

        models['xgboost'] = tuning_results['xgboost']['model']
        models['lightgbm'] = tuning_results['lightgbm']['model']
        models['catboost'] = tuning_results['catboost']['model']

        # Save horizon-specific best params
        params_file = MODELS_DIR / f"best_hyperparameters_{horizon}d.json"
        save_best_params(tuning_results, params_file)

        for name, model in models.items():
            predictions[name] = model.predict(X_test)

    else:
        # Train with default parameters
        print("\n[1/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        models['xgboost'] = xgb_model
        predictions['xgboost'] = xgb_model.predict(X_test)

        print("[2/3] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train, sample_weight=weights)
        models['lightgbm'] = lgb_model
        predictions['lightgbm'] = lgb_model.predict(X_test)

        print("[3/3] Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.05,
            random_state=42, verbose=False
        )
        cat_model.fit(X_train, y_train, sample_weight=weights)
        models['catboost'] = cat_model
        predictions['catboost'] = cat_model.predict(X_test)

    # Ensemble
    ensemble_pred = np.round(
        (predictions['xgboost'] + predictions['lightgbm'] + predictions['catboost']) / 3
    ).astype(int)
    predictions['ensemble'] = ensemble_pred

    # Evaluate
    print(f"\n{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'vs Baseline':<12}")
    print("-" * 64)

    for name, pred in predictions.items():
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        improvement = acc - baseline_acc

        print(f"{name:<12} {acc:.4f}     {prec:.4f}     {rec:.4f}     {f1:.4f}     {improvement:+.4f}")

        all_results.append({
            'Horizon': f'{horizon}d',
            'Model': name.capitalize(),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Baseline': baseline_acc,
            'Improvement': improvement
        })

    # Save models
    print(f"\n💾 Saving {horizon}-day models...")
    for name, model in models.items():
        filepath = MODELS_DIR / f"{name}_{horizon}d.pkl"
        joblib.dump(model, filepath)
        print(f"   ✓ {filepath}")

    # Save feature list
    feature_list_path = MODELS_DIR / f"features_{horizon}d.txt"
    with open(feature_list_path, 'w') as f:
        for feat in feat_cols:
            f.write(f"{feat}\n")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("MULTI-HORIZON TRAINING SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Show ensemble results for each horizon
ensemble_results = results_df[results_df['Model'] == 'Ensemble']
print(f"\n{'Horizon':<10} {'Accuracy':<10} {'F1':<10} {'vs Baseline':<12}")
print("-" * 42)

for _, row in ensemble_results.iterrows():
    print(f"{row['Horizon']:<10} {row['Accuracy']:.4f}     {row['F1']:.4f}     {row['Improvement']:+.4f}")

# Save results
results_path = MODELS_DIR / "multi_horizon_signal_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\n💾 Saved results: {results_path}")

# Recommendations
print("\n💡 RECOMMENDATIONS:")
best_horizon = ensemble_results.loc[ensemble_results['Accuracy'].idxmax()]
print(f"   Best accuracy: {best_horizon['Horizon']} ({best_horizon['Accuracy']:.4f})")

best_f1_horizon = ensemble_results.loc[ensemble_results['F1'].idxmax()]
print(f"   Best F1 score: {best_f1_horizon['Horizon']} ({best_f1_horizon['F1']:.4f})")

print("\nUsage:")
print("   # For predictions, specify horizon:")
print("   python3 predict_multi_asset_ensemble.py --horizon 3")

print("\n" + "=" * 80)
print("✅ MULTI-HORIZON SIGNAL TRAINING COMPLETE!")
print("=" * 80)

# Interactive next steps
print("\n💡 NEXT STEPS:")
print("-" * 40)
print("1. Generate predictions")
print("2. Run backtest")
print("3. Train standard multi-asset models")
print("4. Run hyperparameter tuning")
print("0. Exit")

try:
    choice = input("\nSelect next step (0-4): ").strip()

    if choice == "1":
        print("\n▶️  Running: python3 predict_multi_asset_ensemble.py")
        import subprocess
        subprocess.run(["python3", "predict_multi_asset_ensemble.py"])
    elif choice == "2":
        print("\n▶️  Running: python3 backtest.py")
        import subprocess
        subprocess.run(["python3", "backtest.py"])
    elif choice == "3":
        print("\n▶️  Running: python3 train_multi_asset.py")
        import subprocess
        subprocess.run(["python3", "train_multi_asset.py"])
    elif choice == "4":
        print("\n▶️  Running: python3 hyperparameter_tuning.py")
        import subprocess
        subprocess.run(["python3", "hyperparameter_tuning.py"])
    else:
        print("\n👋 Done!")
except (KeyboardInterrupt, EOFError):
    print("\n👋 Done!")
