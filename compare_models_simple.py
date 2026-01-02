#!/usr/bin/env python3
"""
Simple comparison: Train LightGBM and CatBoost on the same data as XGBoost

Uses the compare_optimization_metrics approach to prep data correctly.
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Install if needed
try:
    import lightgbm as lgb
except:
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "lightgbm"])
    import lightgbm as lgb

try:
    import catboost as cb
except:
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "catboost"])
    import catboost as cb

from config import MODELS_DIR, TRAIN_CFG
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels

print("="*80)
print("MODEL ARCHITECTURE COMPARISON: XGBoost vs LightGBM vs CatBoost")
print("="*80)

# Load XGBoost model to get features
print("\n📥 Loading XGBoost model...")
xgb_payload = joblib.load(MODELS_DIR / "market_crash_model_fwd_improved.pkl")
xgb_model = xgb_payload["model"]
saved_features = xgb_payload["features"]
print(f"✅ XGBoost model loaded ({len(saved_features)} features)")

# Prepare data using same approach as compare_optimization_metrics
print("\n📊 Preparing data...")
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

# Add Close for labeling
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

# Add labels
df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

# Keep only features used by XGBoost model + labels
keep_cols = [c for c in saved_features if c in df.columns] + ["y", "fwd_ret_net"]
df = df[keep_cols].dropna()

print(f"✅ Data prepared: {len(df)} rows")
print(f"   Class 0: {(df['y'] == 0).sum()} ({(df['y'] == 0).mean()*100:.1f}%)")
print(f"   Class 1: {(df['y'] == 1).sum()} ({(df['y'] == 1).mean()*100:.1f}%)")

# Split into train/test
test_size = int(len(df) * 0.2)
X_train = df.iloc[:-test_size][saved_features]
X_test = df.iloc[-test_size:][saved_features]
y_train = df.iloc[:-test_size]["y"]
y_test = df.iloc[-test_size:]["y"]

print(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test\n")

results = {}

# ============================================================================
# 1. XGBoost (already trained)
# ============================================================================
print("="*80)
print("XGBOOST (Pre-trained)")
print("="*80)

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

results['xgboost'] = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_xgb, zero_division=0),
    'f1': f1_score(y_test, y_pred_xgb, zero_division=0),
    'probabilities': y_proba_xgb,
}

print(f"Accuracy:  {results['xgboost']['accuracy']:.4f}")
print(f"Precision: {results['xgboost']['precision']:.4f}")
print(f"Recall:    {results['xgboost']['recall']:.4f}")
print(f"F1:        {results['xgboost']['f1']:.4f}")

# ============================================================================
# 2. LightGBM
# ============================================================================
print("\n" + "="*80)
print("LIGHTGBM")
print("="*80)

start = datetime.now()

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
    'min_child_samples': 20,
    'random_state': 42,
    'verbosity': -1,
}

# Calculate class weights manually
class_counts = y_train.value_counts()
total = len(y_train)
class_weight_dict = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
sample_weights = y_train.map(class_weight_dict)

print(f"[{datetime.now():%H:%M:%S}] Training...")
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

elapsed = (datetime.now() - start).total_seconds()
print(f"[{datetime.now():%H:%M:%S}] Completed in {elapsed:.1f}s")

y_pred_lgb = lgb_model.predict(X_test)
y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

results['lightgbm'] = {
    'accuracy': accuracy_score(y_test, y_pred_lgb),
    'precision': precision_score(y_test, y_pred_lgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_lgb, zero_division=0),
    'f1': f1_score(y_test, y_pred_lgb, zero_division=0),
    'probabilities': y_proba_lgb,
    'training_time': elapsed,
}

print(f"Accuracy:  {results['lightgbm']['accuracy']:.4f}")
print(f"Precision: {results['lightgbm']['precision']:.4f}")
print(f"Recall:    {results['lightgbm']['recall']:.4f}")
print(f"F1:        {results['lightgbm']['f1']:.4f}")

# Save
joblib.dump({'model': lgb_model, 'features': saved_features}, MODELS_DIR / "market_crash_model_lightgbm.pkl")
print(f"💾 Saved: {MODELS_DIR / 'market_crash_model_lightgbm.pkl'}")

# ============================================================================
# 3. CatBoost
# ============================================================================
print("\n" + "="*80)
print("CATBOOST")
print("="*80)

start = datetime.now()

cb_params = {
    'iterations': 400,
    'depth': 5,
    'learning_rate': 0.02,
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'subsample': 0.8,
    'random_state': 42,
    'verbose': False,
    'l2_leaf_reg': 3.0,
}

# Calculate class weights
cb_class_weights = {0: class_weight_dict[0], 1: class_weight_dict[1]}

print(f"[{datetime.now():%H:%M:%S}] Training...")
cb_model = cb.CatBoostClassifier(**cb_params, class_weights=cb_class_weights)
cb_model.fit(X_train, y_train)

elapsed = (datetime.now() - start).total_seconds()
print(f"[{datetime.now():%H:%M:%S}] Completed in {elapsed:.1f}s")

y_pred_cb = cb_model.predict(X_test).flatten().astype(int)
y_proba_cb = cb_model.predict_proba(X_test)[:, 1]

results['catboost'] = {
    'accuracy': accuracy_score(y_test, y_pred_cb),
    'precision': precision_score(y_test, y_pred_cb, zero_division=0),
    'recall': recall_score(y_test, y_pred_cb, zero_division=0),
    'f1': f1_score(y_test, y_pred_cb, zero_division=0),
    'probabilities': y_proba_cb,
    'training_time': elapsed,
}

print(f"Accuracy:  {results['catboost']['accuracy']:.4f}")
print(f"Precision: {results['catboost']['precision']:.4f}")
print(f"Recall:    {results['catboost']['recall']:.4f}")
print(f"F1:        {results['catboost']['f1']:.4f}")

# Save
joblib.dump({'model': cb_model, 'features': saved_features}, MODELS_DIR / "market_crash_model_catboost.pkl")
print(f"💾 Saved: {MODELS_DIR / 'market_crash_model_catboost.pkl'}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    name: {
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1 Score': res['f1'],
    }
    for name, res in results.items()
}).T

print("\n" + str(comparison_df.round(4)))

comparison_df.to_csv('model_architecture_comparison.csv')
print(f"\n✅ Saved: model_architecture_comparison.csv")

# Find best
best_f1 = comparison_df['F1 Score'].idxmax()
best_acc = comparison_df['Accuracy'].idxmax()

print(f"\n🏆 Best F1: {best_f1.upper()} ({comparison_df.loc[best_f1, 'F1 Score']:.4f})")
print(f"🏆 Best Accuracy: {best_acc.upper()} ({comparison_df.loc[best_acc, 'Accuracy']:.4f})")

# Ensemble
print("\n" + "="*80)
print("ENSEMBLE (Average Probabilities)")
print("="*80)

ensemble_proba = np.mean([results['xgboost']['probabilities'],
                          results['lightgbm']['probabilities'],
                          results['catboost']['probabilities']], axis=0)
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

acc_ens = accuracy_score(y_test, ensemble_pred)
prec_ens = precision_score(y_test, ensemble_pred, zero_division=0)
rec_ens = recall_score(y_test, ensemble_pred, zero_division=0)
f1_ens = f1_score(y_test, ensemble_pred, zero_division=0)

print(f"Accuracy:  {acc_ens:.4f}")
print(f"Precision: {prec_ens:.4f}")
print(f"Recall:    {rec_ens:.4f}")
print(f"F1:        {f1_ens:.4f}")

best_single_f1 = comparison_df['F1 Score'].max()
if f1_ens > best_single_f1:
    improvement = ((f1_ens - best_single_f1) / best_single_f1) * 100
    print(f"\n✅ Ensemble improves F1 by {improvement:.2f}%!")
else:
    print(f"\n⚠️ Best single model performs better")

print("\n🎉 Complete!")
