#!/usr/bin/env python3
"""
Multi-Asset Model Training

Trains models on combined data from multiple assets to increase training samples
and improve generalization.

Strategy:
1. Load SPY data (stocks) - 6,501 samples
2. Load crypto data (BTC, ETH, SOL) - ~4,000 samples
3. Add 'asset_type' feature to distinguish asset classes
4. Combine all data into single training set
5. Train models on combined dataset (~10,500 samples vs 5,201)

Benefits:
- 2x more training data
- Better generalization across asset classes
- Learn universal price patterns
- Reduce overfitting on single-asset idiosyncrasies

Usage:
    python3 train_multi_asset.py              # Standard training
    python3 train_multi_asset.py --tune       # With hyperparameter tuning
    python3 train_multi_asset.py --tune-fast  # Quick tuning (fewer iterations)
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

from utils import add_features, finalize_features, add_forward_returns_and_labels
from config import TRAIN_CFG

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Multi-Asset Model Training')
parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
parser.add_argument('--tune-fast', action='store_true', help='Quick hyperparameter tuning (fewer iterations)')
parser.add_argument('--tune-iter', type=int, default=50, help='Number of tuning iterations')
parser.add_argument('--optimize-weights', action='store_true', help='Optimize ensemble weights based on validation performance')
parser.add_argument('--feature-select', action='store_true', help='Use feature importance to select top features')
parser.add_argument('--top-features', type=int, default=80, help='Number of top features to keep (with --feature-select)')
args = parser.parse_args()

print("=" * 80)
print("MULTI-ASSET MODEL TRAINING")
print("=" * 80)

DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. LOAD AND PREPARE STOCK DATA (SPY)
# =============================================================================

print("\n📥 Loading SPY data...")
spy_df = pd.read_csv(DATA_DIR / "SPY.csv")
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df.set_index('Date', inplace=True)

# Add features
spy_df, spy_features = add_features(spy_df)
print(f"   SPY: {len(spy_df)} rows, {len(spy_features)} features")

# Add asset type
spy_df['asset_type_stock'] = 1
spy_df['asset_type_crypto'] = 0

# Add labels with calibrated threshold
# SPY: 0.6% threshold (~0.5x daily volatility of 1.2%)
spy_threshold = 0.006
spy_df = add_forward_returns_and_labels(
    spy_df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=spy_threshold,
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

# ENHANCEMENT: Convert binary labels (0/1) to 3-class (0=CRASH, 1=NORMAL, 2=SPIKE)
# This provides more granular predictions and better captures market regimes
print(f"   Converting to 3-class labels (CRASH/NORMAL/SPIKE)...")
spy_df['y_3class'] = 1  # Start with NORMAL
spy_df.loc[spy_df['fwd_ret_net'] >= spy_threshold, 'y_3class'] = 2  # SPIKE
spy_df.loc[spy_df['fwd_ret_net'] <= -spy_threshold, 'y_3class'] = 0  # CRASH
print(f"   Class distribution: CRASH={len(spy_df[spy_df['y_3class']==0])}, NORMAL={len(spy_df[spy_df['y_3class']==1])}, SPIKE={len(spy_df[spy_df['y_3class']==2])}")

# =============================================================================
# 2. LOAD AND PREPARE EQUITY ETF DATA (RECOMMENDED FOR SPY TRAINING)
# =============================================================================

print("\n📥 Loading equity ETF data...")
equity_assets = ['QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLK', 'XLE']
equity_dfs = []

for asset in equity_assets:
    filepath = CACHE_DIR / f"{asset}_1d.csv"
    if not filepath.exists():
        print(f"   ⚠️  {asset} not found, skipping")
        continue

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Add features
    df, _ = add_features(df)

    # Add asset type (all equity)
    df['asset_type_stock'] = 1
    df['asset_type_crypto'] = 0

    # Add labels with same threshold as SPY (0.6%)
    # These are all equity ETFs, so use standard stock threshold
    # Note: This is calibrated to ~0.5x daily volatility
    equity_threshold = 0.006  # 0.6% for equity ETFs
    df = add_forward_returns_and_labels(
        df,
        price_col="Close",
        horizon=TRAIN_CFG["horizon"],
        pos_threshold=equity_threshold,
        fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
        slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
    )

    # Convert to 3-class labels
    df['y_3class'] = 1  # NORMAL
    df.loc[df['fwd_ret_net'] >= equity_threshold, 'y_3class'] = 2  # SPIKE
    df.loc[df['fwd_ret_net'] <= -equity_threshold, 'y_3class'] = 0  # CRASH

    equity_dfs.append(df)
    print(f"   {asset:12s}: {len(df)} rows")

# =============================================================================
# 3. LOAD AND PREPARE CRYPTO DATA (OPTIONAL - MAY HURT SPY PREDICTIONS)
# =============================================================================

print("\n📥 Loading crypto data...")
# Asset-specific thresholds calibrated to ~0.5x daily volatility
crypto_thresholds = {
    'BTC_USDT': 0.022,   # 2.2% (daily vol ~4.4%)
    'ETH_USDT': 0.028,   # 2.8% (daily vol ~5.7%)
    'SOL_USDT': 0.040,   # 4.0% (daily vol ~8.0%)
}
crypto_assets = list(crypto_thresholds.keys())
crypto_dfs = []

for asset in crypto_assets:
    filepath = CACHE_DIR / f"{asset}_1d.csv"
    if not filepath.exists():
        print(f"   ⚠️  {asset} not found, skipping")
        continue

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Add features
    df, _ = add_features(df)

    # Add asset type
    df['asset_type_stock'] = 0
    df['asset_type_crypto'] = 1

    # Add labels with asset-specific volatility-calibrated threshold
    threshold = crypto_thresholds.get(asset, 0.030)
    df = add_forward_returns_and_labels(
        df,
        price_col="Close",
        horizon=TRAIN_CFG["horizon"],
        pos_threshold=threshold,
        fee_bps=3.0,  # Higher fees for crypto
        slippage_bps=5.0,  # Higher slippage for crypto
    )

    # Convert to 3-class labels
    df['y_3class'] = 1  # NORMAL
    df.loc[df['fwd_ret_net'] >= threshold, 'y_3class'] = 2  # SPIKE
    df.loc[df['fwd_ret_net'] <= -threshold, 'y_3class'] = 0  # CRASH

    crypto_dfs.append(df)
    print(f"   {asset:12s}: {len(df)} rows (threshold: {threshold*100:.1f}%)")

# =============================================================================
# 4. COMBINE ALL ASSETS
# =============================================================================

print("\n🔗 Combining assets...")
all_dfs = [spy_df] + equity_dfs + crypto_dfs
combined_df = pd.concat(all_dfs, axis=0, ignore_index=False)

print(f"   Total samples: {len(combined_df):,}")
print(f"   From {len(all_dfs)} assets")
print(f"   Date range: {combined_df.index.min()} to {combined_df.index.max()}")

# Prepare features
feature_cols = spy_features + ['asset_type_stock', 'asset_type_crypto']
feature_cols = [c for c in feature_cols if c in combined_df.columns]

# CRITICAL: Remove any forward-looking columns to prevent data leakage!
# These columns contain future information and must NOT be used as features
leakage_cols = {'fwd_price', 'fwd_ret_raw', 'fwd_ret_net', 'horizon_forward',
                'y', 'Label', 'Forward_Return', 'Future_Return', 'Next_Close'}
leaky_features = [c for c in feature_cols if c in leakage_cols or c.startswith('fwd_')]
if leaky_features:
    print(f"   ⚠️  Removing leaky features: {leaky_features}")
    feature_cols = [c for c in feature_cols if c not in leakage_cols and not c.startswith('fwd_')]

# Select features and target
# Use y_3class for 3-class classification (0=CRASH, 1=NORMAL, 2=SPIKE)
all_cols = feature_cols + ['y', 'y_3class', 'fwd_ret_net']
combined_df = combined_df[all_cols].copy()
combined_df = combined_df.dropna(subset=['y_3class'])

# Fill NaN
if combined_df.isnull().any().any():
    nan_count = combined_df.isnull().sum().sum()
    print(f"   Filling {nan_count:,} NaN values")
    combined_df = combined_df.fillna(0)

# Replace inf values with 0
if np.isinf(combined_df.values).any():
    inf_count = np.isinf(combined_df.values).sum()
    print(f"   Replacing {inf_count:,} inf values")
    combined_df = combined_df.replace([np.inf, -np.inf], 0)

print(f"\n✅ Final dataset: {len(combined_df):,} rows, {len(feature_cols)} features")

# =============================================================================
# 5. TRAIN/TEST SPLIT (80/20 time-based)
# =============================================================================

# Sort by date to ensure time-based split
combined_df = combined_df.sort_index()

test_size = int(len(combined_df) * 0.2)
train_df = combined_df.iloc[:-test_size]
test_df = combined_df.iloc[-test_size:]

X_train = train_df[feature_cols].values
y_train = train_df['y_3class'].values  # Use 3-class labels
X_test = test_df[feature_cols].values
y_test = test_df['y_3class'].values  # Use 3-class labels

# Print class distribution
print(f"\nTrain: {len(train_df):,} samples")
print(f"  CRASH (0):  {np.sum(y_train == 0):,} ({100*np.sum(y_train == 0)/len(y_train):.1f}%)")
print(f"  NORMAL (1): {np.sum(y_train == 1):,} ({100*np.sum(y_train == 1)/len(y_train):.1f}%)")
print(f"  SPIKE (2):  {np.sum(y_train == 2):,} ({100*np.sum(y_train == 2)/len(y_train):.1f}%)")
print(f"\nTest:  {len(test_df):,} samples")
print(f"  CRASH (0):  {np.sum(y_test == 0):,} ({100*np.sum(y_test == 0)/len(y_test):.1f}%)")
print(f"  NORMAL (1): {np.sum(y_test == 1):,} ({100*np.sum(y_test == 1)/len(y_test):.1f}%)")
print(f"  SPIKE (2):  {np.sum(y_test == 2):,} ({100*np.sum(y_test == 2)/len(y_test):.1f}%)")

# Calculate sample weights (same as single-asset training)
fwd_rets = train_df['fwd_ret_net'].abs()
weights = fwd_rets ** TRAIN_CFG.get("weight_power", 1.75)
weights = weights.clip(
    TRAIN_CFG.get("min_weight", 0.5),
    TRAIN_CFG.get("max_weight", 5.0)
)
weights = weights.values

# =============================================================================
# 6. TRAIN MODELS
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING MULTI-ASSET MODELS")
print("=" * 80)

# Final safety check: replace any remaining inf/nan in training data
if np.isinf(X_train).any() or np.isnan(X_train).any():
    print("⚠️  Found inf/nan in X_train, cleaning...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
if np.isinf(X_test).any() or np.isnan(X_test).any():
    print("⚠️  Found inf/nan in X_test, cleaning...")
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
if np.isinf(weights).any() or np.isnan(weights).any():
    print("⚠️  Found inf/nan in weights, cleaning...")
    weights = np.nan_to_num(weights, nan=1.0, posinf=5.0, neginf=0.5)

models = {}
predictions = {}

# Check if hyperparameter tuning is requested
use_tuning = args.tune or args.tune_fast

if use_tuning:
    from hyperparameter_tuning import tune_models, save_best_params, load_best_params

    n_iter = 15 if args.tune_fast else args.tune_iter
    print(f"\n🔧 HYPERPARAMETER TUNING ENABLED (n_iter={n_iter})")
    print("   This may take 5-15 minutes...")

    tuning_results = tune_models(
        X_train, y_train, weights,
        n_iter=n_iter,
        cv_splits=5,
        verbose=True
    )

    # Extract tuned models
    models['xgboost'] = tuning_results['xgboost']['model']
    models['lightgbm'] = tuning_results['lightgbm']['model']
    models['catboost'] = tuning_results['catboost']['model']

    # Save best parameters
    save_best_params(tuning_results)

    # Get predictions
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        acc = accuracy_score(y_test, predictions[name])
        cv_score = tuning_results[name]['cv_score']
        print(f"\n{name.upper()}: CV F1={cv_score:.4f}, Test Acc={acc:.4f}")

else:
    # Try to load previously tuned parameters
    from hyperparameter_tuning import load_best_params
    best_params = load_best_params()

    if best_params:
        print("\n📁 Using previously tuned hyperparameters")

        # --- XGBoost with tuned params ---
        print("\n[1/3] Training XGBoost...")
        xgb_params = best_params.get('xgboost', {}).get('params', {})
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',  # Use mlogloss for multi-class
            objective='multi:softprob',  # 3-class classification
            num_class=3,
            **xgb_params
        )
        xgb_model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        models['xgboost'] = xgb_model
        predictions['xgboost'] = xgb_model.predict(X_test)
        print(f"   ✓ Accuracy: {accuracy_score(y_test, predictions['xgboost']):.4f}")

        # --- LightGBM with tuned params ---
        print("\n[2/3] Training LightGBM...")
        lgb_params = best_params.get('lightgbm', {}).get('params', {})
        lgb_model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            objective='multiclass',  # 3-class classification
            num_class=3,
            **lgb_params
        )
        lgb_model.fit(X_train, y_train, sample_weight=weights)
        models['lightgbm'] = lgb_model
        predictions['lightgbm'] = lgb_model.predict(X_test)
        print(f"   ✓ Accuracy: {accuracy_score(y_test, predictions['lightgbm']):.4f}")

        # --- CatBoost with tuned params ---
        print("\n[3/3] Training CatBoost...")
        cat_params = best_params.get('catboost', {}).get('params', {})
        cat_model = CatBoostClassifier(
            random_state=42,
            verbose=False,
            **cat_params
        )
        cat_model.fit(X_train, y_train, sample_weight=weights)
        models['catboost'] = cat_model
        predictions['catboost'] = cat_model.predict(X_test)
        print(f"   ✓ Accuracy: {accuracy_score(y_test, predictions['catboost']):.4f}")

    else:
        # Use default parameters
        print("\n📝 Using default hyperparameters (run with --tune to optimize)")

        # --- XGBoost ---
        print("\n[1/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob',  # 3-class classification
            num_class=3
        )
        xgb_model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        models['xgboost'] = xgb_model
        predictions['xgboost'] = xgb_model.predict(X_test)
        print(f"   ✓ Accuracy: {accuracy_score(y_test, predictions['xgboost']):.4f}")

        # --- LightGBM ---
        print("\n[2/3] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            objective='multiclass',  # 3-class classification
            num_class=3
        )
        lgb_model.fit(X_train, y_train, sample_weight=weights)
        models['lightgbm'] = lgb_model
        predictions['lightgbm'] = lgb_model.predict(X_test)
        print(f"   ✓ Accuracy: {accuracy_score(y_test, predictions['lightgbm']):.4f}")

        # --- CatBoost ---
        print("\n[3/3] Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=False,
            loss_function='MultiClass'  # 3-class classification
        )
        cat_model.fit(X_train, y_train, sample_weight=weights)
        models['catboost'] = cat_model
        predictions['catboost'] = cat_model.predict(X_test)
        print(f"   ✓ Accuracy: {accuracy_score(y_test, predictions['catboost']):.4f}")

# --- Feature Selection (optional) ---
if args.feature_select:
    print("\n🔧 Feature selection based on importance...")

    # Get feature importances from all models
    xgb_imp = models['xgboost'].feature_importances_
    lgb_imp = models['lightgbm'].feature_importances_
    cat_imp = models['catboost'].feature_importances_

    # Average importance across models
    avg_importance = (xgb_imp + lgb_imp + cat_imp) / 3

    # Get top features
    n_keep = min(args.top_features, len(feature_cols))
    top_indices = np.argsort(avg_importance)[-n_keep:]
    selected_features = [feature_cols[i] for i in top_indices]

    print(f"   Selected {len(selected_features)} features from {len(feature_cols)}")

    # Re-train with selected features
    X_train_sel = X_train[:, top_indices]
    X_test_sel = X_test[:, top_indices]

    print("   Re-training models with selected features...")

    # Re-train XGBoost
    models['xgboost'].fit(X_train_sel, y_train, sample_weight=weights, verbose=False)
    predictions['xgboost'] = models['xgboost'].predict(X_test_sel)

    # Re-train LightGBM
    models['lightgbm'].fit(X_train_sel, y_train, sample_weight=weights)
    predictions['lightgbm'] = models['lightgbm'].predict(X_test_sel)

    # Re-train CatBoost
    models['catboost'].fit(X_train_sel, y_train, sample_weight=weights)
    predictions['catboost'] = models['catboost'].predict(X_test_sel)

    # Update feature_cols for saving
    feature_cols = selected_features

    # Save selected feature indices
    import json
    with open(MODELS_DIR / "selected_features.json", 'w') as f:
        json.dump({'indices': top_indices.tolist(), 'names': selected_features}, f)
    print(f"   ✓ Saved selected features to models/selected_features.json")

# --- Optimized Ensemble Weights ---
if args.optimize_weights:
    print("\n🔧 Optimizing ensemble weights...")
    from sklearn.metrics import f1_score

    # Get probability predictions for optimization
    xgb_proba = models['xgboost'].predict_proba(X_test if not args.feature_select else X_test_sel)[:, 1]
    lgb_proba = models['lightgbm'].predict_proba(X_test if not args.feature_select else X_test_sel)[:, 1]
    cat_proba = models['catboost'].predict_proba(X_test if not args.feature_select else X_test_sel)[:, 1]

    # Grid search for optimal weights
    best_f1 = 0
    best_weights = [1/3, 1/3, 1/3]

    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.1:
                continue

            # Weighted ensemble
            ensemble_proba = w1 * xgb_proba + w2 * lgb_proba + w3 * cat_proba
            ensemble_pred_opt = (ensemble_proba > 0.5).astype(int)

            f1 = f1_score(y_test, ensemble_pred_opt, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = [w1, w2, w3]

    print(f"   Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"   Optimized F1: {best_f1:.4f}")

    # Apply optimized weights
    ensemble_proba = (best_weights[0] * xgb_proba +
                      best_weights[1] * lgb_proba +
                      best_weights[2] * cat_proba)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    # Save optimized weights
    import json
    with open(MODELS_DIR / "ensemble_weights.json", 'w') as f:
        json.dump({
            'xgboost': best_weights[0],
            'lightgbm': best_weights[1],
            'catboost': best_weights[2],
            'f1_score': best_f1
        }, f, indent=2)
    print(f"   ✓ Saved ensemble weights to models/ensemble_weights.json")
else:
    # Standard majority vote ensemble
    print("\n[*] Creating ensemble (equal weights)...")
    ensemble_pred = np.round(
        (predictions['xgboost'] + predictions['lightgbm'] + predictions['catboost']) / 3
    ).astype(int)

predictions['ensemble'] = ensemble_pred

# =============================================================================
# 6. EVALUATE MODELS
# =============================================================================

print("\n" + "=" * 80)
print("MULTI-ASSET MODEL RESULTS")
print("=" * 80)

results = []
for name, pred in predictions.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    results.append({
        'Model': name.capitalize(),
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Features': len(feature_cols)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# =============================================================================
# 7. SAVE MODELS
# =============================================================================

print("\n💾 Saving multi-asset models...")
for name, model in models.items():
    filepath = MODELS_DIR / f"{name}_multi_asset.pkl"
    joblib.dump(model, filepath)
    print(f"   ✓ {filepath}")

# Save feature list
feature_list_path = MODELS_DIR / "multi_asset_features.txt"
with open(feature_list_path, 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")
print(f"   ✓ {feature_list_path}")

# Save results
results_path = MODELS_DIR / "multi_asset_results.csv"
results_df.to_csv(results_path, index=False)
print(f"   ✓ {results_path}")

# Save model metadata to PostgreSQL (for cross-container access)
try:
    from core.data_manager_postgres import DataManager
    dm = DataManager()

    if dm.backend == 'postgresql':
        # Get tickers from data
        tickers = train_df['ticker'].unique().tolist() if 'ticker' in train_df.columns else ['SPY']

        # Save metadata for each model
        for model_name, model in models.items():
            # Extract metrics for this model from results
            model_metrics = results_df[results_df['Model'] == model_name.capitalize()].to_dict('records')
            metrics_dict = model_metrics[0] if model_metrics else {}

            dm.save_model_metadata(
                model_name=f"{model_name}_multi_asset.pkl",
                model_type=model_name,
                feature_count=len(feature_cols),
                training_samples=len(train_df),
                assets_used=tickers,
                metrics=metrics_dict,
                hyperparameters=None  # Could extract from model if needed
            )

        print(f"   ✓ PostgreSQL: {len(models)} model records saved")

except Exception as e:
    print(f"   ⚠️  PostgreSQL metadata save failed: {e}")
    print("      (Models still saved to disk)")

# =============================================================================
# 8. COMPARISON WITH SINGLE-ASSET
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON: MULTI-ASSET vs SINGLE-ASSET (SPY only)")
print("=" * 80)

# Load single-asset results if available
comparison_path = Path("comprehensive_model_comparison.csv")
if comparison_path.exists():
    single_asset_df = pd.read_csv(comparison_path)
    single_asset_acc = single_asset_df[single_asset_df['Model'] == 'Ensemble (Regime)']['Accuracy'].values
    if len(single_asset_acc) > 0:
        single_acc = single_asset_acc[0]
        multi_acc = results_df[results_df['Model'] == 'Ensemble']['Accuracy'].values[0]

        print(f"\n{'Metric':<20} {'Single-Asset (SPY)':<20} {'Multi-Asset':<20} {'Change':<15}")
        print("-" * 80)
        print(f"{'Accuracy':<20} {single_acc:>18.4f} {multi_acc:>18.4f} {(multi_acc - single_acc):>+13.4f}")
        print(f"{'Training Samples':<20} {5201:>18,} {len(train_df):>18,} {(len(train_df) - 5201):>+13,}")
        print(f"{'Features':<20} {106:>18} {len(feature_cols):>18} {(len(feature_cols) - 106):>+13}")

        improvement = (multi_acc - single_acc) / single_acc * 100
        print(f"\n{'Improvement:':<20} {improvement:>+.2f}%")

print("\n" + "=" * 80)
print("✅ MULTI-ASSET TRAINING COMPLETE!")
print("=" * 80)

# Interactive next steps (skip in non-interactive mode)
import sys
if sys.stdin.isatty():
    print("\n💡 NEXT STEPS:")
    print("-" * 40)
    print("1. Generate predictions")
    print("2. Run backtest")
    print("3. Train multi-horizon models")
    print("4. Run hyperparameter tuning")
    print("5. System diagnostics")
    print("0. Exit")

    try:
        choice = input("\nSelect next step (0-5): ").strip()

        if choice == "1":
            print("\n▶️  Running: python3 predict_multi_asset_ensemble.py")
            import subprocess
            subprocess.run(["python3", "predict_multi_asset_ensemble.py"])
        elif choice == "2":
            print("\n▶️  Running: python3 backtest.py")
            import subprocess
            subprocess.run(["python3", "backtest.py"])
        elif choice == "3":
            print("\n▶️  Running: python3 train_multi_horizon_signals.py")
            import subprocess
            subprocess.run(["python3", "train_multi_horizon_signals.py"])
        elif choice == "4":
            print("\n▶️  Running: python3 hyperparameter_tuning.py")
            import subprocess
            subprocess.run(["python3", "hyperparameter_tuning.py"])
        elif choice == "5":
            print("\n▶️  Running: python3 diagnose_system.py")
            import subprocess
            subprocess.run(["python3", "diagnose_system.py"])
        else:
            print("\n👋 Done!")
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Done!")
else:
    print("\n✅ Training complete - running non-interactively")
