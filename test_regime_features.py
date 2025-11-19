#!/usr/bin/env python3
"""
Test market regime features by comparing model performance

Compares models trained with and without regime detection features.
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import lightgbm as lgb

from config import MODELS_DIR, TRAIN_CFG
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels

print("="*80)
print("MARKET REGIME FEATURES IMPACT ANALYSIS")
print("="*80)
print("\nTesting new market regime detection features:")
print("  - Bull/Bear market indicators (200-day MA)")
print("  - VIX/Volatility-based fear metrics")
print("  - Market breadth indicators")
print("  - Trend strength (ADX)")
print("  - Regime classification\n")

# Prepare data
print("📥 Loading data with new regime features...")
df = load_SPY_data()
df, feature_cols = add_features(df)

# Count regime features
regime_features = [f for f in feature_cols if any(x in f for x in [
    'MA_200', 'MA200', 'Bull_Market', 'Price_vs_MA200',
    'VIX_Percentile', 'High_Fear', 'High_Volatility', 'VIX_Spike',
    'Near_52w', 'MA_20_50_Cross', 'Pct_Above_MA20',
    'ADX', 'Strong_Trend', 'Plus_DI', 'Minus_DI',
    'Days_Above_MA20', 'Trend_Consistency', 'Regime_Score'
])]

print(f"✅ Feature engineering complete")
print(f"   Total features: {len(feature_cols)}")
print(f"   New regime features: {len(regime_features)}")
print(f"\n   Regime features added:")
for f in regime_features:
    if f in df.columns:
        print(f"     - {f}")

# Prepare data using same pipeline as compare_models_simple
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

# Get all available features
all_features = [c for c in df.columns if c not in ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]
keep_cols = all_features + ["y", "fwd_ret_net"]

# Select columns and drop rows with NaN in critical columns only
df = df[keep_cols]
df = df.dropna(subset=["y"])  # Only drop rows without labels

print(f"\n✅ Data prepared: {len(df)} rows")
print(f"   Features available: {len(all_features)}")

# Handle any remaining NaN in features by filling with 0
if df.isnull().any().any():
    print(f"   Filling {df.isnull().sum().sum()} NaN values with 0")
    df = df.fillna(0)

# Check for sufficient data
if len(df) < 100:
    print(f"\n❌ ERROR: Only {len(df)} rows after preparation. Need at least 100 rows.")
    print("   This might indicate an issue with the data pipeline.")
    exit(1)

# Split data
test_size = int(len(df) * 0.2)
X_train_full = df.iloc[:-test_size][all_features]
X_test_full = df.iloc[-test_size:][all_features]
y_train = df.iloc[:-test_size]["y"]
y_test = df.iloc[-test_size:]["y"]

print(f"   Train: {len(X_train_full)}, Test: {len(X_test_full)}")
print(f"   Train class distribution: {y_train.value_counts().to_dict()}")

# Load old XGBoost model to get baseline features
print("\n📥 Loading baseline XGBoost model (without regime features)...")
xgb_payload = joblib.load(MODELS_DIR / "market_crash_model_fwd_improved.pkl")
baseline_features = xgb_payload["features"]
print(f"✅ Baseline uses {len(baseline_features)} features (no regime detection)")

# Create baseline dataset (features without regime)
X_train_baseline = X_train_full[[f for f in baseline_features if f in X_train_full.columns]]
X_test_baseline = X_test_full[[f for f in baseline_features if f in X_test_full.columns]]

print(f"   Baseline features in current data: {len(X_train_baseline.columns)}")

results = {}

# ============================================================================
# 1. BASELINE LIGHTGBM (without regime features)
# ============================================================================
print("\n" + "="*80)
print("BASELINE: LightGBM WITHOUT Regime Features")
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

# Calculate class weights
class_counts = y_train.value_counts()
total = len(y_train)

# Check both classes exist
if len(class_counts) < 2:
    print(f"\n❌ ERROR: Only one class present in training data: {class_counts.to_dict()}")
    print("   Cannot train a binary classifier with only one class.")
    exit(1)

class_weight_dict = {0: total / (2 * class_counts.get(0, 1)), 1: total / (2 * class_counts.get(1, 1))}
sample_weights = y_train.map(class_weight_dict)

print(f"[{datetime.now():%H:%M:%S}] Training baseline...")
lgb_baseline = lgb.LGBMClassifier(**lgb_params)
lgb_baseline.fit(X_train_baseline, y_train, sample_weight=sample_weights)

elapsed = (datetime.now() - start).total_seconds()

y_pred = lgb_baseline.predict(X_test_baseline)
y_proba = lgb_baseline.predict_proba(X_test_baseline)[:, 1]

results['baseline'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'training_time': elapsed,
    'n_features': len(X_train_baseline.columns),
}

print(f"[{datetime.now():%H:%M:%S}] Completed in {elapsed:.1f}s")
print(f"Accuracy:  {results['baseline']['accuracy']:.4f}")
print(f"Precision: {results['baseline']['precision']:.4f}")
print(f"Recall:    {results['baseline']['recall']:.4f}")
print(f"F1:        {results['baseline']['f1']:.4f}")

# ============================================================================
# 2. WITH REGIME FEATURES
# ============================================================================
print("\n" + "="*80)
print("NEW: LightGBM WITH Regime Detection Features")
print("="*80)

start = datetime.now()

print(f"[{datetime.now():%H:%M:%S}] Training with regime features...")
lgb_regime = lgb.LGBMClassifier(**lgb_params)
lgb_regime.fit(X_train_full, y_train, sample_weight=sample_weights)

elapsed = (datetime.now() - start).total_seconds()

y_pred = lgb_regime.predict(X_test_full)
y_proba = lgb_regime.predict_proba(X_test_full)[:, 1]

results['with_regime'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'training_time': elapsed,
    'n_features': len(X_train_full.columns),
}

print(f"[{datetime.now():%H:%M:%S}] Completed in {elapsed:.1f}s")
print(f"Accuracy:  {results['with_regime']['accuracy']:.4f}")
print(f"Precision: {results['with_regime']['precision']:.4f}")
print(f"Recall:    {results['with_regime']['recall']:.4f}")
print(f"F1:        {results['with_regime']['f1']:.4f}")

# Save the regime-enhanced model
joblib.dump({'model': lgb_regime, 'features': list(X_train_full.columns)},
            MODELS_DIR / "market_crash_model_lightgbm_regime.pkl")
print(f"💾 Saved: {MODELS_DIR / 'market_crash_model_lightgbm_regime.pkl'}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("IMPACT ANALYSIS")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Features', 'Training Time (s)'],
    'Baseline': [
        results['baseline']['accuracy'],
        results['baseline']['precision'],
        results['baseline']['recall'],
        results['baseline']['f1'],
        results['baseline']['n_features'],
        results['baseline']['training_time'],
    ],
    'With Regime': [
        results['with_regime']['accuracy'],
        results['with_regime']['precision'],
        results['with_regime']['recall'],
        results['with_regime']['f1'],
        results['with_regime']['n_features'],
        results['with_regime']['training_time'],
    ]
})

comparison_df['Improvement'] = comparison_df['With Regime'] - comparison_df['Baseline']
comparison_df['% Change'] = (comparison_df['Improvement'] / comparison_df['Baseline'].replace(0, np.nan)) * 100

print("\n" + str(comparison_df.round(4)))

comparison_df.to_csv('regime_features_impact.csv', index=False)
print(f"\n✅ Saved: regime_features_impact.csv")

# Feature importance for regime features
print("\n" + "="*80)
print("REGIME FEATURE IMPORTANCE")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': lgb_regime.feature_importances_
}).sort_values('importance', ascending=False)

regime_importance = feature_importance[feature_importance['feature'].isin(regime_features)]

if len(regime_importance) > 0:
    print(f"\n{len(regime_importance)} regime features used by model:")
    print(regime_importance.to_string(index=False))

    total_importance = feature_importance['importance'].sum()
    regime_total_importance = regime_importance['importance'].sum()
    regime_pct = (regime_total_importance / total_importance) * 100

    print(f"\n📊 Regime features contribute {regime_pct:.2f}% of total importance")
else:
    print("\n⚠️ No regime features used by the model")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

f1_improvement = (results['with_regime']['f1'] - results['baseline']['f1']) / results['baseline']['f1'] * 100
acc_improvement = (results['with_regime']['accuracy'] - results['baseline']['accuracy']) / results['baseline']['accuracy'] * 100

if results['with_regime']['f1'] > results['baseline']['f1']:
    print(f"\n✅ Regime features IMPROVE performance!")
    print(f"   F1 Score: {f1_improvement:+.2f}%")
    print(f"   Accuracy: {acc_improvement:+.2f}%")
    print(f"\n💡 Recommendation: Use regime-enhanced model for better market adaptability")
elif results['with_regime']['f1'] > results['baseline']['f1'] * 0.98:
    print(f"\n➡️ Regime features have NEUTRAL impact")
    print(f"   F1 Score: {f1_improvement:+.2f}% (minimal change)")
    print(f"\n💡 Recommendation: Regime features don't hurt, provide market context")
else:
    print(f"\n⚠️ Regime features DECREASE performance")
    print(f"   F1 Score: {f1_improvement:+.2f}%")
    print(f"\n💡 Recommendation: Stick with baseline model, regime features may add noise")

print("\n🎉 Analysis complete!")
