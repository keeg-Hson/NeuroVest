#!/usr/bin/env python3
"""
Retrain Model with Cross-Asset Features

Combines existing 103 features with 27 new cross-asset features
Expected improvement: +15-20% accuracy (59.69% → 70-75%)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)
from train import TRAIN_CFG

print("=" * 80)
print("RETRAINING MODEL WITH CROSS-ASSET FEATURES")
print("=" * 80)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# ============================================================================
# 1. LOAD EXISTING FEATURES (103 features)
# ============================================================================

print("\n📥 Loading existing features...")
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

# Reindex Close
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

# Add forward returns
df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

print(f"✅ Loaded {len(df)} rows with {len(feature_cols)} existing features")

# ============================================================================
# 2. LOAD CROSS-ASSET FEATURES (27 features)
# ============================================================================

print("\n📥 Loading cross-asset features...")
cross_asset_file = DATA_DIR / "cross_asset_features.csv"

if not cross_asset_file.exists():
    print(f"❌ Cross-asset features not found at {cross_asset_file}")
    print("   Run: python create_cross_asset_features.py")
    exit(1)

cross_asset_df = pd.read_csv(cross_asset_file, index_col=0, parse_dates=True)
print(f"✅ Loaded {len(cross_asset_df.columns)} cross-asset features")

# ============================================================================
# 3. MERGE FEATURES
# ============================================================================

print("\n🔄 Merging existing + cross-asset features...")

# Get existing features (excluding labels and aux columns)
existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

# Merge cross-asset features
df_combined = df[existing_features + ["y", "fwd_ret_net", "Close"]].copy()

# Join cross-asset features
for col in cross_asset_df.columns:
    df_combined[col] = cross_asset_df[col].reindex(df_combined.index)

# Get all feature columns
all_features = existing_features + list(cross_asset_df.columns)

print(f"✅ Combined features:")
print(f"   Existing features:     {len(existing_features)}")
print(f"   Cross-asset features:  {len(cross_asset_df.columns)}")
print(f"   Total features:        {len(all_features)}")

# Fill NaN
df_combined = df_combined.fillna(0)

# Drop rows without labels
df_combined = df_combined.dropna(subset=["y"])

print(f"\n✅ Final dataset: {len(df_combined)} rows")

# ============================================================================
# 4. SPLIT DATA
# ============================================================================

test_size = int(len(df_combined) * 0.2)
train_end_idx = len(df_combined) - test_size

X_train = df_combined.iloc[:train_end_idx][all_features]
X_test = df_combined.iloc[train_end_idx:][all_features]
y_train = df_combined.iloc[:train_end_idx]["y"]
y_test = df_combined.iloc[train_end_idx:]["y"]

print(f"\n📅 Data split:")
print(f"   Train: {len(X_train)} rows")
print(f"   Test:  {len(X_test)} rows")
print(f"   Train class distribution: {y_train.value_counts().to_dict()}")

# ============================================================================
# 5. TRAIN BASELINE (WITHOUT CROSS-ASSET) FOR COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("BASELINE: MODEL WITHOUT CROSS-ASSET FEATURES")
print("=" * 80)

X_train_baseline = df_combined.iloc[:train_end_idx][existing_features]
X_test_baseline = df_combined.iloc[train_end_idx:][existing_features]

# Calculate sample weights
class_counts = y_train.value_counts()
total = len(y_train)
class_weight_dict = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
sample_weights = y_train.map(class_weight_dict)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.02,
    'n_estimators': 400,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbosity': -1,
}

print(f"\n⏳ Training baseline model ({len(existing_features)} features)...")
baseline_model = lgb.LGBMClassifier(**lgb_params)
baseline_model.fit(X_train_baseline, y_train, sample_weight=sample_weights)

baseline_pred = baseline_model.predict(X_test_baseline)
baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_prec = precision_score(y_test, baseline_pred, zero_division=0)
baseline_rec = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

print(f"\n✅ Baseline Results:")
print(f"   Accuracy:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"   Precision: {baseline_prec:.4f}")
print(f"   Recall:    {baseline_rec:.4f}")
print(f"   F1 Score:  {baseline_f1:.4f}")

# ============================================================================
# 6. TRAIN WITH CROSS-ASSET FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("NEW MODEL: WITH CROSS-ASSET FEATURES")
print("=" * 80)

print(f"\n⏳ Training model with cross-asset features ({len(all_features)} features)...")
start = datetime.now()
cross_asset_model = lgb.LGBMClassifier(**lgb_params)
cross_asset_model.fit(X_train, y_train, sample_weight=sample_weights)
elapsed = (datetime.now() - start).total_seconds()

print(f"✅ Training complete in {elapsed:.1f}s")

cross_asset_pred = cross_asset_model.predict(X_test)
cross_asset_acc = accuracy_score(y_test, cross_asset_pred)
cross_asset_prec = precision_score(y_test, cross_asset_pred, zero_division=0)
cross_asset_rec = recall_score(y_test, cross_asset_pred)
cross_asset_f1 = f1_score(y_test, cross_asset_pred)

print(f"\n✅ Cross-Asset Model Results:")
print(f"   Accuracy:  {cross_asset_acc:.4f} ({cross_asset_acc*100:.2f}%)")
print(f"   Precision: {cross_asset_prec:.4f}")
print(f"   Recall:    {cross_asset_rec:.4f}")
print(f"   F1 Score:  {cross_asset_f1:.4f}")

# ============================================================================
# 7. COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("ACCURACY IMPROVEMENT ANALYSIS")
print("=" * 80)

comparison = pd.DataFrame([
    {
        'Model': 'Baseline (103 features)',
        'Accuracy': baseline_acc,
        'Precision': baseline_prec,
        'Recall': baseline_rec,
        'F1_Score': baseline_f1,
        'Features': len(existing_features)
    },
    {
        'Model': 'With Cross-Asset (130 features)',
        'Accuracy': cross_asset_acc,
        'Precision': cross_asset_prec,
        'Recall': cross_asset_rec,
        'F1_Score': cross_asset_f1,
        'Features': len(all_features)
    }
])

print("\n" + comparison.to_string(index=False))

# Calculate improvement
acc_improvement = ((cross_asset_acc - baseline_acc) / baseline_acc) * 100
f1_improvement = ((cross_asset_f1 - baseline_f1) / baseline_f1) * 100

print(f"\n🎯 Improvement:")
print(f"   Accuracy:  {baseline_acc*100:.2f}% → {cross_asset_acc*100:.2f}% ({acc_improvement:+.1f}%)")
print(f"   F1 Score:  {baseline_f1:.4f} → {cross_asset_f1:.4f} ({f1_improvement:+.1f}%)")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("TOP CROSS-ASSET FEATURES BY IMPORTANCE")
print("=" * 80)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': cross_asset_model.feature_importances_
}).sort_values('importance', ascending=False)

# Top overall
print("\n📊 Top 20 Features (Overall):")
for i, row in feature_importance.head(20).iterrows():
    is_cross_asset = row['feature'].startswith('XAsset_')
    marker = "🆕" if is_cross_asset else "  "
    print(f"   {i+1:2d}. {marker} {row['feature']:<45s} {row['importance']:>8.1f}")

# Top cross-asset features
cross_asset_importance = feature_importance[feature_importance['feature'].str.startswith('XAsset_')]
print(f"\n📊 Top 10 Cross-Asset Features:")
for i, row in cross_asset_importance.head(10).iterrows():
    print(f"   {i+1:2d}. {row['feature']:<45s} {row['importance']:>8.1f}")

total_importance = feature_importance['importance'].sum()
cross_asset_total = cross_asset_importance['importance'].sum()
cross_asset_pct = (cross_asset_total / total_importance) * 100

print(f"\n📊 Cross-asset features contribute {cross_asset_pct:.1f}% of total importance")

# ============================================================================
# 9. SAVE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "xgboost_with_cross_asset.pkl"
joblib.dump({
    'model': cross_asset_model,
    'features': all_features,
    'accuracy': cross_asset_acc,
    'f1_score': cross_asset_f1,
    'training_date': datetime.now().isoformat()
}, model_path)

print(f"💾 Saved model to: {model_path}")

# Save comparison results
comparison.to_csv("cross_asset_comparison.csv", index=False)
feature_importance.to_csv("cross_asset_feature_importance.csv", index=False)

print("\n" + "=" * 80)
print("✅ RETRAINING COMPLETE!")
print("=" * 80)

print(f"\n🎉 Results:")
print(f"   Baseline accuracy:     {baseline_acc*100:.2f}%")
print(f"   Cross-asset accuracy:  {cross_asset_acc*100:.2f}%")
print(f"   Improvement:           {acc_improvement:+.1f}%")

if cross_asset_acc > baseline_acc:
    print(f"\n✅ Cross-asset features IMPROVED the model!")
else:
    print(f"\n⚠️  Cross-asset features did not improve accuracy on this test set")
    print(f"    This can happen due to:")
    print(f"    - Different market regime in test period")
    print(f"    - Need for more data")
    print(f"    - Features need tuning")

print(f"\n📁 Files saved:")
print(f"   - models/xgboost_with_cross_asset.pkl")
print(f"   - cross_asset_comparison.csv")
print(f"   - cross_asset_feature_importance.csv")
