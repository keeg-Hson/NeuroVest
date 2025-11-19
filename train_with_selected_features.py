#!/usr/bin/env python3
"""
Retrain Models with Selected Features
======================================
Train models using only the top 80% most important features.

Expected: +0.5-1% improvement from noise reduction
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

print("=" * 80)
print("RETRAINING WITH SELECTED FEATURES (Top 80%)")
print("=" * 80)
print()

# Paths
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# Load selected features
print("📥 Loading selected features list...")
with open("selected_features.txt", 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"✅ Loaded {len(selected_features)} selected features")
print()

# Load data
print("📥 Loading SPY data...")
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels
from train import TRAIN_CFG

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

# Load cross-asset and macro
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)

existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

df_all = df[existing_features + ["y", "Close"]].copy()

for col in cross_asset_df.columns:
    df_all[col] = cross_asset_df[col].reindex(df_all.index)

for col in macro_df.columns:
    df_all[col] = macro_df[col].reindex(df_all.index)

df_all = df_all.fillna(0)
df_all = df_all.dropna(subset=["y"])

print(f"✅ Loaded {len(df_all)} samples")
print()

# Filter to selected features only
available_features = [f for f in selected_features if f in df_all.columns]
missing_features = [f for f in selected_features if f not in df_all.columns]

if missing_features:
    print(f"⚠️  Warning: {len(missing_features)} selected features not in data:")
    for feat in missing_features[:10]:
        print(f"   - {feat}")
    if len(missing_features) > 10:
        print(f"   ... and {len(missing_features)-10} more")
    print()

X = df_all[available_features].values
y = df_all["y"].values

print(f"📊 Dataset with selected features:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(available_features)} (reduced from 164)")
print(f"   Reduction: {164 - len(available_features)} features removed ({(164-len(available_features))/164*100:.1f}%)")
print()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Train/Test split:")
print(f"   Train: {len(X_train)}")
print(f"   Test:  {len(X_test)}")
print()

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("=" * 80)
print("TRAINING MODELS WITH SELECTED FEATURES")
print("=" * 80)
print()

results = {}

# 1. LightGBM
print("🌳 Training LightGBM...")
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=200)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)
lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_f1 = f1_score(y_test, lgb_pred)

print(f"   Accuracy: {lgb_acc:.4f} ({lgb_acc*100:.2f}%)")
print(f"   F1 Score: {lgb_f1:.4f}")
print()

results['LightGBM_Selected'] = {
    'accuracy': lgb_acc,
    'f1': lgb_f1,
    'predictions': lgb_pred_proba
}

# 2. XGBoost
print("🚀 Training XGBoost...")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'verbosity': 0
}

xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=200)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"   F1 Score: {xgb_f1:.4f}")
print()

results['XGBoost_Selected'] = {
    'accuracy': xgb_acc,
    'f1': xgb_f1,
    'predictions': xgb_pred_proba
}

# 3. Ensemble (Weighted Average)
print("🎯 Creating Ensemble...")
ensemble_pred_proba = 0.5 * lgb_pred_proba + 0.5 * xgb_pred_proba
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

ens_acc = accuracy_score(y_test, ensemble_pred)
ens_f1 = f1_score(y_test, ensemble_pred)

print(f"   Accuracy: {ens_acc:.4f} ({ens_acc*100:.2f}%)")
print(f"   F1 Score: {ens_f1:.4f}")
print()

results['Ensemble_Selected'] = {
    'accuracy': ens_acc,
    'f1': ens_f1
}

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================

print("=" * 80)
print("COMPARISON: SELECTED FEATURES vs BASELINE (All Features)")
print("=" * 80)
print()

# Load baseline results from final_model_comparison.csv
baseline_df = pd.read_csv("final_model_comparison.csv")
baseline_ensemble_acc = baseline_df[baseline_df['Model'] == 'Ensemble (Weighted Average)']['Accuracy'].values[0]

print(f"📊 Results Comparison:")
print(f"   Baseline Ensemble (164 features): {baseline_ensemble_acc:.4f} ({baseline_ensemble_acc*100:.2f}%)")
print(f"   NEW Ensemble ({len(available_features)} features):     {ens_acc:.4f} ({ens_acc*100:.2f}%)")
print()

improvement = (ens_acc - baseline_ensemble_acc) * 100
if improvement > 0:
    print(f"✅ IMPROVEMENT: +{improvement:.2f}pp ({improvement/baseline_ensemble_acc*100:.1f}% relative)")
elif improvement < 0:
    print(f"❌ REGRESSION: {improvement:.2f}pp")
else:
    print(f"⚖️  NO CHANGE")

print()

# Detailed comparison table
comparison_df = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'Ensemble'],
    'Baseline_Acc': [
        baseline_df[baseline_df['Model'] == 'LightGBM (Regime)']['Accuracy'].values[0],
        baseline_df[baseline_df['Model'] == 'XGBoost (Regime)']['Accuracy'].values[0],
        baseline_ensemble_acc
    ],
    'Selected_Acc': [lgb_acc, xgb_acc, ens_acc],
    'Improvement_pp': [
        (lgb_acc - baseline_df[baseline_df['Model'] == 'LightGBM (Regime)']['Accuracy'].values[0]) * 100,
        (xgb_acc - baseline_df[baseline_df['Model'] == 'XGBoost (Regime)']['Accuracy'].values[0]) * 100,
        improvement
    ]
})

print("\n📊 Detailed Comparison:")
print(comparison_df.to_string(index=False))
print()

# Save results
comparison_df.to_csv("feature_selection_comparison.csv", index=False)
print(f"💾 Saved comparison to: feature_selection_comparison.csv")

# Save models
with open(MODELS_DIR / "lightgbm_selected_features.pkl", 'wb') as f:
    pickle.dump(lgb_model, f)
with open(MODELS_DIR / "xgboost_selected_features.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)

print(f"💾 Saved models to: {MODELS_DIR}/")
print()

print("=" * 80)
print("✅ FEATURE SELECTION RETRAINING COMPLETE")
print("=" * 80)
