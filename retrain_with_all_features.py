#!/usr/bin/env python3
"""
Retrain Model with ALL Features

Combines:
- 103 existing technical features
- 27 cross-asset lead-lag features
- 34 macro-economic features
= 164 total features

Expected cumulative improvement: +15-25% accuracy
Target: 59% baseline → 70-75% with all features
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
print("FINAL RETRAIN: ALL FEATURES (PHASE 1 + PHASE 2)")
print("=" * 80)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# ============================================================================
# 1. LOAD ALL FEATURE SETS
# ============================================================================

print("\n📥 Loading all feature sets...")

# Existing features
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

print(f"✅ Existing features: {len(feature_cols)} features")

# Cross-asset features
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)
print(f"✅ Cross-asset features: {len(cross_asset_df.columns)} features")

# Macro features
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)
print(f"✅ Macro features: {len(macro_df.columns)} features")

# ============================================================================
# 2. MERGE ALL FEATURES
# ============================================================================

print("\n🔄 Merging all feature sets...")

existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

df_all = df[existing_features + ["y", "fwd_ret_net", "Close"]].copy()

# Join cross-asset
for col in cross_asset_df.columns:
    df_all[col] = cross_asset_df[col].reindex(df_all.index)

# Join macro
for col in macro_df.columns:
    df_all[col] = macro_df[col].reindex(df_all.index)

all_features = existing_features + list(cross_asset_df.columns) + list(macro_df.columns)

print(f"\n✅ Feature sets combined:")
print(f"   Existing (technical):  {len(existing_features):>3d}")
print(f"   Cross-asset:           {len(cross_asset_df.columns):>3d}")
print(f"   Macro-economic:        {len(macro_df.columns):>3d}")
print(f"   {'─' * 35}")
print(f"   TOTAL FEATURES:        {len(all_features):>3d}")

# Fill NaN and drop rows without labels
df_all = df_all.fillna(0)
df_all = df_all.dropna(subset=["y"])

print(f"\n✅ Final dataset: {len(df_all)} rows")

# ============================================================================
# 3. SPLIT DATA
# ============================================================================

test_size = int(len(df_all) * 0.2)
train_end_idx = len(df_all) - test_size

X_train = df_all.iloc[:train_end_idx][all_features]
X_test = df_all.iloc[train_end_idx:][all_features]
y_train = df_all.iloc[:train_end_idx]["y"]
y_test = df_all.iloc[train_end_idx:]["y"]

print(f"\n📅 Data split:")
print(f"   Train: {len(X_train)} rows")
print(f"   Test:  {len(X_test)} rows")

# Sample weights
class_counts = y_train.value_counts()
total = len(y_train)
class_weight_dict = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
sample_weights = y_train.map(class_weight_dict)

# ============================================================================
# 4. TRAIN 3 MODELS FOR COMPARISON
# ============================================================================

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

print("\n" + "=" * 80)
print("TRAINING 3 MODELS FOR COMPARISON")
print("=" * 80)

# Model 1: Baseline (existing features only)
print("\n[1/3] Baseline Model (103 features)")
X_train_baseline = df_all.iloc[:train_end_idx][existing_features]
X_test_baseline = df_all.iloc[train_end_idx:][existing_features]

model_baseline = lgb.LGBMClassifier(**lgb_params)
model_baseline.fit(X_train_baseline, y_train, sample_weight=sample_weights)
pred_baseline = model_baseline.predict(X_test_baseline)

acc_baseline = accuracy_score(y_test, pred_baseline)
f1_baseline = f1_score(y_test, pred_baseline)
print(f"   Accuracy: {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
print(f"   F1 Score: {f1_baseline:.4f}")

# Model 2: With cross-asset (103 + 27 = 130 features)
print("\n[2/3] With Cross-Asset (130 features)")
cross_asset_features = existing_features + list(cross_asset_df.columns)
X_train_cross = df_all.iloc[:train_end_idx][cross_asset_features]
X_test_cross = df_all.iloc[train_end_idx:][cross_asset_features]

model_cross = lgb.LGBMClassifier(**lgb_params)
model_cross.fit(X_train_cross, y_train, sample_weight=sample_weights)
pred_cross = model_cross.predict(X_test_cross)

acc_cross = accuracy_score(y_test, pred_cross)
f1_cross = f1_score(y_test, pred_cross)
print(f"   Accuracy: {acc_cross:.4f} ({acc_cross*100:.2f}%)")
print(f"   F1 Score: {f1_cross:.4f}")

# Model 3: ALL features (103 + 27 + 34 = 164 features)
print("\n[3/3] ALL Features (164 features)")
model_all = lgb.LGBMClassifier(**lgb_params)
model_all.fit(X_train, y_train, sample_weight=sample_weights)
pred_all = model_all.predict(X_test)

acc_all = accuracy_score(y_test, pred_all)
prec_all = precision_score(y_test, pred_all, zero_division=0)
rec_all = recall_score(y_test, pred_all)
f1_all = f1_score(y_test, pred_all)

print(f"   Accuracy:  {acc_all:.4f} ({acc_all*100:.2f}%)")
print(f"   Precision: {prec_all:.4f}")
print(f"   Recall:    {rec_all:.4f}")
print(f"   F1 Score:  {f1_all:.4f}")

# ============================================================================
# 5. PROGRESSIVE IMPROVEMENT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PROGRESSIVE IMPROVEMENT ANALYSIS")
print("=" * 80)

comparison = pd.DataFrame([
    {'Stage': 'Baseline', 'Features': 103, 'Accuracy': acc_baseline, 'F1_Score': f1_baseline},
    {'Stage': '+ Cross-Asset', 'Features': 130, 'Accuracy': acc_cross, 'F1_Score': f1_cross},
    {'Stage': '+ Macro (FINAL)', 'Features': 164, 'Accuracy': acc_all, 'F1_Score': f1_all},
])

print("\n" + comparison.to_string(index=False))

# Calculate improvements
phase1_improvement = ((acc_cross - acc_baseline) / acc_baseline) * 100
phase2_improvement = ((acc_all - acc_cross) / acc_cross) * 100
total_improvement = ((acc_all - acc_baseline) / acc_baseline) * 100

print(f"\n📊 Accuracy Progression:")
print(f"   Baseline:            {acc_baseline*100:>6.2f}%")
print(f"   + Cross-Asset:       {acc_cross*100:>6.2f}% ({phase1_improvement:+.1f}%)")
print(f"   + Macro (FINAL):     {acc_all*100:>6.2f}% ({phase2_improvement:+.1f}%)")
print(f"   {'─' * 50}")
print(f"   TOTAL IMPROVEMENT:   {total_improvement:>+6.1f}%")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': model_all.feature_importances_
}).sort_values('importance', ascending=False)

# Categorize features
importance_df['category'] = 'Existing'
importance_df.loc[importance_df['feature'].str.startswith('XAsset_'), 'category'] = 'Cross-Asset'
importance_df.loc[importance_df['feature'].str.startswith('Macro_'), 'category'] = 'Macro'

print("\n📊 Top 20 Features Overall:")
for i, row in importance_df.head(20).iterrows():
    marker = {'Existing': '  ', 'Cross-Asset': '🔗', 'Macro': '📊'}[row['category']]
    print(f"   {i+1:2d}. {marker} {row['feature']:<45s} {row['importance']:>8.1f}")

# Category contribution
category_importance = importance_df.groupby('category')['importance'].sum()
total_importance = importance_df['importance'].sum()

print(f"\n📊 Feature Category Contributions:")
for category in ['Existing', 'Cross-Asset', 'Macro']:
    contribution = (category_importance[category] / total_importance) * 100
    count = (importance_df['category'] == category).sum()
    print(f"   {category:<15s} {count:>3d} features  {contribution:>5.1f}% of importance")

# Top from each category
print(f"\n📊 Top 5 Features by Category:")
for category in ['Cross-Asset', 'Macro']:
    cat_features = importance_df[importance_df['category'] == category].head(5)
    print(f"\n   {category}:")
    for i, row in cat_features.iterrows():
        print(f"      • {row['feature']:<45s} {row['importance']:>8.1f}")

# ============================================================================
# 7. SAVE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING FINAL MODEL")
print("=" * 80)

model_path = MODELS_DIR / "xgboost_all_features_final.pkl"
joblib.dump({
    'model': model_all,
    'features': all_features,
    'accuracy': acc_all,
    'f1_score': f1_all,
    'feature_counts': {
        'existing': len(existing_features),
        'cross_asset': len(cross_asset_df.columns),
        'macro': len(macro_df.columns)
    },
    'training_date': datetime.now().isoformat()
}, model_path)

print(f"💾 Saved final model to: {model_path}")

# Save results
comparison.to_csv("phase_progression_results.csv", index=False)
importance_df.to_csv("all_features_importance.csv", index=False)

print("\n" + "=" * 80)
print("✅ PHASE 2 COMPLETE!")
print("=" * 80)

print(f"\n🎉 Final Results:")
print(f"   Baseline (technical only):     {acc_baseline*100:.2f}%")
print(f"   Phase 1 (+cross-asset):        {acc_cross*100:.2f}% ({phase1_improvement:+.1f}%)")
print(f"   Phase 2 (+macro) [FINAL]:      {acc_all*100:.2f}% ({phase2_improvement:+.1f}%)")
print(f"\n   🚀 TOTAL IMPROVEMENT:          {total_improvement:+.1f}%")

if acc_all > 0.70:
    print(f"\n   🎊 ACHIEVED 70%+ ACCURACY TARGET!")
elif acc_all > 0.65:
    print(f"\n   ✅ Strong improvement achieved!")
else:
    print(f"\n   📈 Good progress - may need Phase 3 (regime-switching)")

print(f"\n📁 Files saved:")
print(f"   - models/xgboost_all_features_final.pkl")
print(f"   - phase_progression_results.csv")
print(f"   - all_features_importance.csv")

print(f"\n🎯 Next Steps (Optional Phase 3):")
print(f"   - Regime-switching models for +5-10% more")
print(f"   - Deep learning (LSTM) for sequential patterns")
print(f"   - Target: 75-80% accuracy")
