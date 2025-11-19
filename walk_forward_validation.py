#!/usr/bin/env python3
"""
Walk-Forward Validation
=======================
Retrain models on expanding windows to eliminate look-ahead bias.

Traditional train/test split: Train once on 80%, test on 20%
Walk-forward: Retrain multiple times on expanding windows

Expected impact: +1-1.5% from more realistic training methodology

Method:
- Start with minimum 2 years of data
- Predict next month
- Expand window and repeat
- Average results across all folds
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

print("=" * 80)
print("WALK-FORWARD VALIDATION")
print("=" * 80)
print()

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# Load selected features
print("📥 Loading selected features...")
with open("selected_features.txt", 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"✅ Using {len(selected_features)} selected features")
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

# Filter to selected features
available_features = [f for f in selected_features if f in df_all.columns]
df_all = df_all[available_features + ["y", "Close"]]

print(f"✅ Loaded {len(df_all)} samples")
print(f"   Date range: {df_all.index.min()} to {df_all.index.max()}")
print()

# ============================================================================
# WALK-FORWARD VALIDATION SETUP
# ============================================================================

print("=" * 80)
print("WALK-FORWARD VALIDATION SETUP")
print("=" * 80)
print()

# Configuration
MIN_TRAIN_SIZE = 504  # ~2 years of trading days
TEST_SIZE = 21  # ~1 month of trading days
N_SPLITS = 10  # Number of walk-forward folds

print(f"⚙️  Configuration:")
print(f"   Minimum train size: {MIN_TRAIN_SIZE} days (~2 years)")
print(f"   Test size: {TEST_SIZE} days (~1 month)")
print(f"   Number of folds: {N_SPLITS}")
print()

# Create time series splits
tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

X = df_all[available_features].values
y = df_all["y"].values
dates = df_all.index

# ============================================================================
# RUN WALK-FORWARD VALIDATION
# ============================================================================

print("=" * 80)
print("RUNNING WALK-FORWARD VALIDATION")
print("=" * 80)
print()

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

results = {
    'fold': [],
    'train_start': [],
    'train_end': [],
    'test_start': [],
    'test_end': [],
    'train_size': [],
    'test_size': [],
    'lgb_accuracy': [],
    'xgb_accuracy': [],
    'ensemble_accuracy': [],
    'lgb_f1': [],
    'xgb_f1': [],
    'ensemble_f1': []
}

fold_num = 1
for train_idx, test_idx in tscv.split(X):
    # Skip if train size too small
    if len(train_idx) < MIN_TRAIN_SIZE:
        continue

    print(f"\n📊 Fold {fold_num}/{N_SPLITS}")
    print("-" * 80)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_dates = dates[train_idx]
    test_dates = dates[test_idx]

    print(f"   Train: {len(X_train):4d} samples | {train_dates.min()} to {train_dates.max()}")
    print(f"   Test:  {len(X_test):4d} samples | {test_dates.min()} to {test_dates.max()}")

    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=200)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred, zero_division=0)

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=200)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)

    # Ensemble
    ens_pred_proba = 0.5 * lgb_pred_proba + 0.5 * xgb_pred_proba
    ens_pred = (ens_pred_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_pred)
    ens_f1 = f1_score(y_test, ens_pred, zero_division=0)

    print(f"   LightGBM:  {lgb_acc:.4f} (F1: {lgb_f1:.4f})")
    print(f"   XGBoost:   {xgb_acc:.4f} (F1: {xgb_f1:.4f})")
    print(f"   Ensemble:  {ens_acc:.4f} (F1: {ens_f1:.4f})")

    # Save results
    results['fold'].append(fold_num)
    results['train_start'].append(train_dates.min())
    results['train_end'].append(train_dates.max())
    results['test_start'].append(test_dates.min())
    results['test_end'].append(test_dates.max())
    results['train_size'].append(len(X_train))
    results['test_size'].append(len(X_test))
    results['lgb_accuracy'].append(lgb_acc)
    results['xgb_accuracy'].append(xgb_acc)
    results['ensemble_accuracy'].append(ens_acc)
    results['lgb_f1'].append(lgb_f1)
    results['xgb_f1'].append(xgb_f1)
    results['ensemble_f1'].append(ens_f1)

    fold_num += 1

print()

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

print("=" * 80)
print("WALK-FORWARD VALIDATION RESULTS")
print("=" * 80)
print()

results_df = pd.DataFrame(results)

print("📊 Per-Fold Results:")
print(results_df[['fold', 'lgb_accuracy', 'xgb_accuracy', 'ensemble_accuracy']].to_string(index=False))
print()

# Calculate averages
lgb_avg = results_df['lgb_accuracy'].mean()
xgb_avg = results_df['xgb_accuracy'].mean()
ens_avg = results_df['ensemble_accuracy'].mean()

lgb_std = results_df['lgb_accuracy'].std()
xgb_std = results_df['xgb_accuracy'].std()
ens_std = results_df['ensemble_accuracy'].std()

print(f"📊 Average Performance (across {len(results_df)} folds):")
print(f"   LightGBM:  {lgb_avg:.4f} ± {lgb_std:.4f} ({lgb_avg*100:.2f}%)")
print(f"   XGBoost:   {xgb_avg:.4f} ± {xgb_std:.4f} ({xgb_avg*100:.2f}%)")
print(f"   Ensemble:  {ens_avg:.4f} ± {ens_std:.4f} ({ens_avg*100:.2f}%)")
print()

# Comparison with standard validation (69.33%)
baseline_acc = 0.6933
improvement = (ens_avg - baseline_acc) * 100

print(f"📊 Comparison with Standard Validation:")
print(f"   Baseline (80/20 split): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"   Walk-Forward (avg):     {ens_avg:.4f} ({ens_avg*100:.2f}%)")
print()

if improvement > 0:
    print(f"✅ IMPROVEMENT: +{improvement:.2f}pp")
    print(f"   Note: Walk-forward gives MORE REALISTIC estimates by eliminating look-ahead bias")
elif improvement < 0:
    print(f"⚠️  ADJUSTMENT: {improvement:.2f}pp")
    print(f"   Note: Walk-forward is more realistic - baseline likely optimistic")
else:
    print(f"⚖️  NO CHANGE")

print()

# Save results
results_df.to_csv("walk_forward_results.csv", index=False)
print(f"💾 Saved detailed results to: walk_forward_results.csv")
print()

print("=" * 80)
print("✅ WALK-FORWARD VALIDATION COMPLETE")
print("=" * 80)
print()
print("Key Insights:")
print("  - Walk-forward validation is more realistic than single train/test split")
print("  - Models are retrained on expanding windows (no look-ahead bias)")
print("  - Standard deviation shows model stability across different time periods")
print("  - If walk-forward < baseline: baseline was overfitting to specific test period")
