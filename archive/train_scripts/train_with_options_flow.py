#!/usr/bin/env python3
"""
Train Models with Options Flow Features
========================================
Add 18 options flow features to the existing 132 selected features.

Expected impact: +1-2% improvement
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

print("=" * 80)
print("TRAINING WITH OPTIONS FLOW FEATURES")
print("=" * 80)
print()

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# Load selected features
print("📥 Loading selected features...")
with open("selected_features.txt", 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"✅ Loaded {len(selected_features)} selected features")
print()

# Load options flow features
print("📥 Loading options flow features...")
options_df = pd.read_csv(DATA_DIR / "options_flow_features.csv", index_col=0, parse_dates=True)
print(f"✅ Loaded {len(options_df.columns)} options flow features")
print(f"   Features: {list(options_df.columns[:5])}...")
print()

# Load SPY data
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

# Add options flow features
for col in options_df.columns:
    df_all[col] = options_df[col].reindex(df_all.index)

df_all = df_all.fillna(0)
df_all = df_all.dropna(subset=["y"])

print(f"✅ Loaded {len(df_all)} samples")
print()

# Combine selected features with options flow features
available_selected = [f for f in selected_features if f in df_all.columns]
options_features = list(options_df.columns)
all_features = available_selected + options_features

print(f"📊 Feature composition:")
print(f"   Selected features: {len(available_selected)}")
print(f"   Options flow features: {len(options_features)}")
print(f"   Total features: {len(all_features)}")
print()

X = df_all[all_features].values
y = df_all["y"].values

# ============================================================================
# WALK-FORWARD VALIDATION WITH OPTIONS FEATURES
# ============================================================================

print("=" * 80)
print("WALK-FORWARD VALIDATION (with Options Features)")
print("=" * 80)
print()

MIN_TRAIN_SIZE = 504
TEST_SIZE = 21
N_SPLITS = 10

tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
dates = df_all.index

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

results = []
fold_num = 1

for train_idx, test_idx in tscv.split(X):
    if len(train_idx) < MIN_TRAIN_SIZE:
        continue

    print(f"📊 Fold {fold_num}/{N_SPLITS}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=200)
    lgb_model.fit(X_train, y_train)
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=200)
    xgb_model.fit(X_train, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Ensemble
    ens_pred_proba = 0.5 * lgb_pred_proba + 0.5 * xgb_pred_proba
    ens_pred = (ens_pred_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_pred)

    print(f"   Ensemble accuracy: {ens_acc:.4f} ({ens_acc*100:.2f}%)")

    results.append(ens_acc)
    fold_num += 1

print()

# Calculate average
avg_acc = np.mean(results)
std_acc = np.std(results)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print(f"📊 Walk-Forward Performance (with Options Features):")
print(f"   Average: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
print(f"   Std Dev: {std_acc:.4f} ({std_acc*100:.2f}%)")
print()

# Compare with baseline
baseline_acc = 0.7048  # Previous walk-forward result
improvement = (avg_acc - baseline_acc) * 100

print(f"📊 Comparison:")
print(f"   Baseline (132 features): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"   With Options ({len(all_features)} features): {avg_acc:.4f} ({avg_acc*100:.2f}%)")
print()

if improvement > 0:
    print(f"✅ IMPROVEMENT: +{improvement:.2f}pp")
elif improvement < 0:
    print(f"⚠️  REGRESSION: {improvement:.2f}pp")
else:
    print(f"⚖️  NO CHANGE")

print()

# Feature importance analysis
print("=" * 80)
print("OPTIONS FLOW FEATURE IMPORTANCE")
print("=" * 80)
print()

# Train on full dataset to get feature importance
lgb_full = lgb.LGBMClassifier(**lgb_params, n_estimators=200)
lgb_full.fit(X, y)

importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': lgb_full.feature_importances_
}).sort_values('importance', ascending=False)

# Show top options features
options_importance = importance_df[importance_df['feature'].isin(options_features)]
print(f"📊 Options Flow Feature Rankings:")
print(options_importance.to_string(index=False))
print()

# Overall top features
print(f"📊 Top 10 Features Overall:")
print(importance_df.head(10).to_string(index=False))
print()

# Save results
results_df = pd.DataFrame({
    'fold': range(1, len(results) + 1),
    'accuracy': results
})
results_df.to_csv("options_flow_results.csv", index=False)
print(f"💾 Saved results to: options_flow_results.csv")

# Save feature importance
importance_df.to_csv("options_flow_feature_importance.csv", index=False)
print(f"💾 Saved feature importance to: options_flow_feature_importance.csv")

print()
print("=" * 80)
print("✅ TRAINING WITH OPTIONS FLOW COMPLETE")
print("=" * 80)
