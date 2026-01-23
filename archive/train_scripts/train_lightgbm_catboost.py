#!/usr/bin/env python3
"""
Train LightGBM and CatBoost models for comparison with XGBoost

This script trains both models with similar configurations and compares results.
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

# Try importing LightGBM and CatBoost
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "lightgbm"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ CatBoost not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "catboost"])
    import catboost as cb
    CATBOOST_AVAILABLE = True

from config import MODELS_DIR, TRAIN_CFG
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels

print("=" * 80)
print("LIGHTGBM & CATBOOST MODEL TRAINING")
print("=" * 80)
print("\nComparing gradient boosting algorithms:")
print("  - XGBoost (already trained)")
print("  - LightGBM (fast, accurate)")
print("  - CatBoost (robust, less tuning needed)\n")

# Load and prepare data (same as train_improved.py)
print("📥 Loading data...")
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

# Clean data
df = df.replace([np.inf, -np.inf], np.nan)
feature_cols_final = [c for c in df.columns if c not in ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]
df_clean = df[feature_cols_final + ["y", "fwd_ret_net"]].dropna()

X = df_clean[feature_cols_final]
y = df_clean["y"]

print(f"✅ Data prepared: {len(X)} samples, {len(X.columns)} features")
print(f"   Class 0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"   Class 1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# Train/test split
test_size = int(len(X) * 0.2)
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

print(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test")

results = {}

# ============================================================================
# 1. LIGHTGBM
# ============================================================================
if LIGHTGBM_AVAILABLE:
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)

    start_time = datetime.now()

    # LightGBM parameters (similar to XGBoost for fair comparison)
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # 2^max_depth - 1, where max_depth~5
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
        'class_weight': 'balanced',  # Handle class imbalance
    }

    print(f"[{datetime.now():%H:%M:%S}] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[{datetime.now():%H:%M:%S}] Training complete in {elapsed:.1f}s")

    # Evaluate
    y_pred = lgb_model.predict(X_test)
    y_proba = lgb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n📊 LightGBM Results (threshold 0.5):")
    print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"   Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")

    results['lightgbm'] = {
        'model': lgb_model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'training_time': elapsed,
        'predictions': y_pred,
        'probabilities': y_proba,
    }

    # Save model
    model_path = MODELS_DIR / "market_crash_model_lightgbm.pkl"
    payload = {"model": lgb_model, "features": list(X.columns)}
    joblib.dump(payload, model_path)
    print(f"💾 LightGBM model saved: {model_path}")

# ============================================================================
# 2. CATBOOST
# ============================================================================
if CATBOOST_AVAILABLE:
    print("\n" + "=" * 80)
    print("TRAINING CATBOOST")
    print("=" * 80)

    start_time = datetime.now()

    # CatBoost parameters
    cb_params = {
        'iterations': 400,
        'depth': 5,
        'learning_rate': 0.02,
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'subsample': 0.8,
        'random_state': 42,
        'verbose': False,
        'auto_class_weights': 'Balanced',  # Handle class imbalance
        'l2_leaf_reg': 3.0,
    }

    print(f"[{datetime.now():%H:%M:%S}] Training CatBoost...")
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[{datetime.now():%H:%M:%S}] Training complete in {elapsed:.1f}s")

    # Evaluate
    y_pred = cb_model.predict(X_test)
    y_proba = cb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n📊 CatBoost Results (threshold 0.5):")
    print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"   Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")

    results['catboost'] = {
        'model': cb_model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'training_time': elapsed,
        'predictions': y_pred,
        'probabilities': y_proba,
    }

    # Save model
    model_path = MODELS_DIR / "market_crash_model_catboost.pkl"
    payload = {"model": cb_model, "features": list(X.columns)}
    joblib.dump(payload, model_path)
    print(f"💾 CatBoost model saved: {model_path}")

# ============================================================================
# 3. LOAD XGBOOST FOR COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("LOADING XGBOOST (for comparison)")
print("=" * 80)

xgb_path = MODELS_DIR / "market_crash_model_fwd_improved.pkl"
xgb_payload = joblib.load(xgb_path)
xgb_model = xgb_payload["model"]

# Make predictions with XGBoost on same test set
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
rec_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)

print(f"\n📊 XGBoost Results (threshold 0.5):")
print(f"   Accuracy:  {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"   Precision: {prec_xgb:.4f} ({prec_xgb*100:.2f}%)")
print(f"   Recall:    {rec_xgb:.4f} ({rec_xgb*100:.2f}%)")
print(f"   F1 Score:  {f1_xgb:.4f}")

results['xgboost'] = {
    'model': xgb_model,
    'accuracy': acc_xgb,
    'precision': prec_xgb,
    'recall': rec_xgb,
    'f1': f1_xgb,
    'training_time': 0,  # Already trained
    'predictions': y_pred_xgb,
    'probabilities': y_proba_xgb,
}

# ============================================================================
# 4. COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    name: {
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1 Score': res['f1'],
        'Training Time (s)': res['training_time'],
    }
    for name, res in results.items()
}).T

print("\n" + str(comparison_df.round(4)))

# Save comparison
comparison_df.to_csv('model_architecture_comparison.csv')
print(f"\n✅ Comparison saved to: model_architecture_comparison.csv")

# Find best model
best_f1 = comparison_df['F1 Score'].idxmax()
best_acc = comparison_df['Accuracy'].idxmax()

print(f"\n🏆 Best F1 Score: {best_f1.upper()} ({comparison_df.loc[best_f1, 'F1 Score']:.4f})")
print(f"🏆 Best Accuracy: {best_acc.upper()} ({comparison_df.loc[best_acc, 'Accuracy']:.4f})")

# Simple ensemble (average probabilities)
print("\n" + "=" * 80)
print("SIMPLE ENSEMBLE (Average Probabilities)")
print("=" * 80)

ensemble_proba = np.mean([
    results['xgboost']['probabilities'],
    results['lightgbm']['probabilities'],
    results['catboost']['probabilities'],
], axis=0)

ensemble_pred = (ensemble_proba >= 0.5).astype(int)

acc_ens = accuracy_score(y_test, ensemble_pred)
prec_ens = precision_score(y_test, ensemble_pred, zero_division=0)
rec_ens = recall_score(y_test, ensemble_pred, zero_division=0)
f1_ens = f1_score(y_test, ensemble_pred, zero_division=0)

print(f"\n📊 Ensemble Results (threshold 0.5):")
print(f"   Accuracy:  {acc_ens:.4f} ({acc_ens*100:.2f}%)")
print(f"   Precision: {prec_ens:.4f} ({prec_ens*100:.2f}%)")
print(f"   Recall:    {rec_ens:.4f} ({rec_ens*100:.2f}%)")
print(f"   F1 Score:  {f1_ens:.4f}")

# Compare to best single model
best_single_f1 = comparison_df['F1 Score'].max()
improvement = ((f1_ens - best_single_f1) / best_single_f1) * 100

if f1_ens > best_single_f1:
    print(f"\n✅ Ensemble beats best single model by {improvement:.2f}%!")
else:
    print(f"\n⚠️ Best single model ({best_f1}) outperforms ensemble by {-improvement:.2f}%")

# Save ensemble
ensemble_payload = {
    'models': [results['xgboost']['model'], results['lightgbm']['model'], results['catboost']['model']],
    'features': list(X.columns),
    'weights': [1/3, 1/3, 1/3],  # Equal weights
}
joblib.dump(ensemble_payload, MODELS_DIR / "market_crash_model_ensemble.pkl")
print(f"\n💾 Ensemble saved: {MODELS_DIR / 'market_crash_model_ensemble.pkl'}")

print("\n🎉 Training complete!")
print("\n📋 Summary:")
print(f"   - Trained {len(results)} models")
print(f"   - Best individual: {best_f1} (F1: {comparison_df.loc[best_f1, 'F1 Score']:.4f})")
print(f"   - Ensemble F1: {f1_ens:.4f}")
print(f"   - All models saved to {MODELS_DIR}")
