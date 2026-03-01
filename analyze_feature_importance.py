#!/usr/bin/env python3
"""
Feature Importance Analysis with SHAP
======================================
Analyzes feature importance across all models to identify which features
contribute most to predictions and which can be removed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP, install if missing
try:
    import shap
except ImportError:
    print("Installing SHAP...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap'])
    import shap

import lightgbm as lgb
import xgboost as xgb

# Import from utils
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels
from train import TRAIN_CFG

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS WITH SHAP")
print("=" * 80)
print()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Load data using same approach as training scripts
print("📥 Loading all feature sets...")
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

# Load cross-asset features
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)

# Load macro features
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)

# Combine all features
existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

df_all = df[existing_features + ["y", "fwd_ret_net", "Close"]].copy()

# Join cross-asset
for col in cross_asset_df.columns:
    if col not in df_all.columns:
        df_all[col] = cross_asset_df[col]

# Join macro
for col in macro_df.columns:
    if col not in df_all.columns:
        df_all[col] = macro_df[col]

# Drop NaN
df_all = df_all.dropna()

print(f"✅ Loaded {len(df_all)} rows")

# Get feature columns
feature_cols = [c for c in df_all.columns if c not in ["y", "fwd_ret_net", "Close"]]
print(f"📊 Using {len(feature_cols)} features for analysis")
print()

# Split data
split_idx = int(len(df_all) * 0.8)
X_train = df_all[feature_cols].iloc[:split_idx]
y_train = df_all['y'].iloc[:split_idx]
X_test = df_all[feature_cols].iloc[split_idx:]
y_test = df_all['y'].iloc[split_idx:]

# Store importance from each model
importance_results = {}

print("=" * 80)
print("ANALYZING MODEL-SPECIFIC FEATURE IMPORTANCE")
print("=" * 80)
print()

# 1. LightGBM Multi-Asset Model
print("[1/3] LightGBM Multi-Asset Model...")
try:
    model_path = MODELS_DIR / "lightgbm_multi_asset.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "lightgbm_regime.pkl"  # fallback
    with open(model_path, 'rb') as f:
        lgb_model = pickle.load(f)

    # SHAP analysis on subset
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"   Computing SHAP values for {sample_size} samples...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Get mean absolute SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_results['LightGBM_SHAP'] = pd.Series(mean_shap, index=feature_cols)

    print(f"   ✅ Analyzed {sample_size} samples")
    top5 = importance_results['LightGBM_SHAP'].nlargest(5)
    for feat, val in top5.items():
        print(f"      {feat}: {val:.4f}")
        
    # Also get built-in importance
    if hasattr(lgb_model, 'feature_importance'):
        lgb_importances = lgb_model.feature_importance(importance_type='gain')
        importance_results['LGB_Gain'] = pd.Series(lgb_importances, index=feature_cols)
        print(f"   ✅ Extracted LightGBM gain importance")
except Exception as e:
    print(f"   ⚠️  Error: {e}")
print()

# 2. XGBoost Multi-Asset Model
print("[2/3] XGBoost Multi-Asset Model...")
try:
    model_path = MODELS_DIR / "xgboost_multi_asset.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "xgboost_regime.pkl"  # fallback
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"   Computing SHAP values for {sample_size} samples...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_results['XGBoost_SHAP'] = pd.Series(mean_shap, index=feature_cols)

    print(f"   ✅ Analyzed {sample_size} samples")
    top5 = importance_results['XGBoost_SHAP'].nlargest(5)
    for feat, val in top5.items():
        print(f"      {feat}: {val:.4f}")
        
    # Also get built-in importance
    if hasattr(xgb_model, 'feature_importances_'):
        importance_results['XGB_Gain'] = pd.Series(xgb_model.feature_importances_, index=feature_cols)
        print(f"   ✅ Extracted XGBoost gain importance")
except Exception as e:
    print(f"   ⚠️  Error: {e}")
print()

# 3. CatBoost Multi-Asset Model
print("[3/3] CatBoost Multi-Asset Model...")
try:
    model_path = MODELS_DIR / "catboost_multi_asset.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, 'rb') as f:
        cb_model = pickle.load(f)

    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"   Computing SHAP values for {sample_size} samples...")
    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_results['CatBoost_SHAP'] = pd.Series(mean_shap, index=feature_cols)

    print(f"   ✅ Analyzed {sample_size} samples")
    top5 = importance_results['CatBoost_SHAP'].nlargest(5)
    for feat, val in top5.items():
        print(f"      {feat}: {val:.4f}")

    # Also get built-in importance
    if hasattr(cb_model, 'feature_importances_'):
        importance_results['CB_Gain'] = pd.Series(cb_model.feature_importances_, index=feature_cols)
        print(f"   ✅ Extracted CatBoost gain importance")
except Exception as e:
    print(f"   ⚠️  Error: {e}")
print()

# Aggregate results
print("=" * 80)
print("AGGREGATING IMPORTANCE ACROSS MODELS")
print("=" * 80)
print()

if importance_results:
    # Create DataFrame with all importance scores
    importance_df = pd.DataFrame(importance_results)

    # Normalize each column (0-1 scale)
    importance_normalized = importance_df.copy()
    for col in importance_normalized.columns:
        max_val = importance_normalized[col].max()
        if max_val > 0:
            importance_normalized[col] = importance_normalized[col] / max_val

    # Calculate aggregate metrics
    importance_df['Mean'] = importance_normalized.mean(axis=1)
    importance_df['Median'] = importance_normalized.median(axis=1)
    importance_df['Max'] = importance_normalized.max(axis=1)
    importance_df['Min'] = importance_normalized.min(axis=1)
    importance_df['Std'] = importance_normalized.std(axis=1)

    # Sort by mean importance
    importance_df = importance_df.sort_values('Mean', ascending=False)

    # Save results
    output_file = BASE_DIR / "feature_importance_analysis.csv"
    importance_df.to_csv(output_file)
    print(f"💾 Saved full analysis to: {output_file}")
    print()

    # Analysis summary
    print("📊 FEATURE IMPORTANCE SUMMARY")
    print("-" * 80)
    print()

    # Top 20 most important features
    print("🏆 TOP 20 MOST IMPORTANT FEATURES:")
    print(importance_df[['Mean', 'Median', 'Std']].head(20).to_string())
    print()

    # Bottom 20 least important features
    print("⚠️  BOTTOM 20 LEAST IMPORTANT FEATURES:")
    print(importance_df[['Mean', 'Median', 'Std']].tail(20).to_string())
    print()

    # Recommendations
    print("=" * 80)
    print("FEATURE SELECTION RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Features to keep (top 80%)
    threshold_keep = importance_df['Mean'].quantile(0.2)
    features_to_keep = importance_df[importance_df['Mean'] >= threshold_keep].index.tolist()

    # Features to remove (bottom 20%)
    features_to_remove = importance_df[importance_df['Mean'] < threshold_keep].index.tolist()

    print(f"📊 Current features: {len(feature_cols)}")
    print(f"✅ Recommended to KEEP: {len(features_to_keep)} features (top 80%)")
    print(f"❌ Recommended to REMOVE: {len(features_to_remove)} features (bottom 20%)")
    print()

    if features_to_remove:
        print("Features to REMOVE:")
        for i, feat in enumerate(features_to_remove[:40]):
            mean_imp = importance_df.loc[feat, 'Mean']
            print(f"   {i+1:2d}. {feat:50s} (importance: {mean_imp:.6f})")
        if len(features_to_remove) > 40:
            print(f"   ... and {len(features_to_remove) - 40} more")
        print()

    # Save feature lists
    keep_file = BASE_DIR / "features_to_keep.txt"
    remove_file = BASE_DIR / "features_to_remove.txt"

    with open(keep_file, 'w') as f:
        f.write('\n'.join(features_to_keep))

    with open(remove_file, 'w') as f:
        f.write('\n'.join(features_to_remove))

    print(f"💾 Saved feature lists:")
    print(f"   - {keep_file}")
    print(f"   - {remove_file}")
    print()

    print("=" * 80)
    print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("💡 Next Steps:")
    print("   1. Review feature_importance_analysis.csv for detailed rankings")
    print("   2. Remove bottom 20% features to reduce noise")
    print("   3. Retrain models with optimized feature set")
    print()
    print(f"   Expected impact: +0.5-1% accuracy from noise reduction")

else:
    print("⚠️  No models could be analyzed. Please check model files.")
