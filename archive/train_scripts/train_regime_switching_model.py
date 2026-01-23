#!/usr/bin/env python3
"""
Regime-Switching Model Training

Uses Hidden Markov Model to detect market regimes, then trains
separate models for each regime. This accounts for the fact that
different market conditions require different prediction strategies.

Expected improvement: +5-10% accuracy
Target: 62.31% → 67-72%
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
from hmmlearn import hmm

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)
from train import TRAIN_CFG

print("=" * 80)
print("REGIME-SWITCHING MODEL TRAINING (PHASE 3)")
print("=" * 80)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# ============================================================================
# 1. LOAD ALL FEATURES
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

# Cross-asset features
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)

# Macro features
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)

# Combine all features
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

df_all = df_all.fillna(0)
df_all = df_all.dropna(subset=["y"])

print(f"✅ Loaded {len(df_all)} rows with {len(all_features)} features")

# ============================================================================
# 2. DETECT MARKET REGIMES USING HMM
# ============================================================================

print("\n" + "=" * 80)
print("REGIME DETECTION WITH HIDDEN MARKOV MODEL")
print("=" * 80)

# Create features for regime detection
# Use returns, volatility, and volume-related features
regime_detection_features = []
for col in df_all.columns:
    if any(x in col.lower() for x in ['return', 'vol', 'rsi', 'atr', 'ma_', 'bull_market', 'vix']):
        regime_detection_features.append(col)

# Ensure we have Close for returns
if 'Close' in df_all.columns:
    returns = df_all['Close'].pct_change()
    volatility = returns.rolling(20).std()

    regime_data = pd.DataFrame({
        'returns': returns,
        'volatility': volatility,
        'abs_returns': returns.abs(),
    })

    # Add some existing features
    if 'XAsset_Realized_Vol_20' in df_all.columns:
        regime_data['realized_vol'] = df_all['XAsset_Realized_Vol_20']
    if 'Bull_Market' in df_all.columns:
        regime_data['bull_market'] = df_all['Bull_Market']
    if 'XAsset_Credit_Stress' in df_all.columns:
        regime_data['credit_stress'] = df_all['XAsset_Credit_Stress']

    regime_data = regime_data.fillna(0)

    print(f"\n📊 Using {len(regime_data.columns)} features for regime detection:")
    for feat in regime_data.columns:
        print(f"   • {feat}")

    # Train HMM with 4 states (Bull, Bear, Choppy, Crisis)
    print(f"\n⏳ Training Hidden Markov Model with 4 regimes...")

    n_regimes = 4
    hmm_model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )

    # Fit HMM
    X_regime = regime_data.values
    hmm_model.fit(X_regime)

    # Predict regimes
    regimes = hmm_model.predict(X_regime)
    df_all['regime'] = regimes

    print(f"✅ HMM training complete")

    # ============================================================================
    # 3. ANALYZE REGIMES
    # ============================================================================

    print("\n" + "=" * 80)
    print("REGIME CHARACTERISTICS ANALYSIS")
    print("=" * 80)

    regime_stats = []
    for regime_id in range(n_regimes):
        regime_mask = df_all['regime'] == regime_id
        regime_data_subset = df_all[regime_mask]

        if len(regime_data_subset) > 0:
            avg_return = regime_data_subset['Close'].pct_change().mean() * 100
            avg_vol = regime_data_subset['Close'].pct_change().rolling(20).std().mean() * 100
            label_distribution = regime_data_subset['y'].mean()

            regime_stats.append({
                'regime': regime_id,
                'count': len(regime_data_subset),
                'pct': len(regime_data_subset) / len(df_all) * 100,
                'avg_return': avg_return,
                'avg_vol': avg_vol,
                'bullish_label_pct': label_distribution * 100
            })

    regime_stats_df = pd.DataFrame(regime_stats).sort_values('avg_return', ascending=False)

    # Assign regime names based on characteristics
    regime_names = {}
    for idx, row in regime_stats_df.iterrows():
        regime_id = int(row['regime'])
        if row['avg_return'] > 0.05 and row['avg_vol'] < 1.5:
            regime_names[regime_id] = "Bull Market"
        elif row['avg_return'] < -0.05 and row['avg_vol'] > 2.0:
            regime_names[regime_id] = "Crisis"
        elif row['avg_return'] < 0:
            regime_names[regime_id] = "Bear Market"
        else:
            regime_names[regime_id] = "Choppy/Sideways"

    print(f"\n📊 Regime Statistics:\n")
    print(f"{'Regime':<20s} {'Days':>8s} {'Pct':>8s} {'Avg Ret':>10s} {'Avg Vol':>10s} {'Bullish %':>10s}")
    print("─" * 80)

    for idx, row in regime_stats_df.iterrows():
        regime_id = int(row['regime'])
        name = regime_names.get(regime_id, f"Regime {regime_id}")
        print(f"{name:<20s} {row['count']:>8.0f} {row['pct']:>7.1f}% {row['avg_return']:>+9.3f}% {row['avg_vol']:>9.2f}% {row['bullish_label_pct']:>9.1f}%")

    # Map regime IDs to names
    df_all['regime_name'] = df_all['regime'].map(regime_names)

    # Save regime analysis
    regime_stats_df['regime_name'] = regime_stats_df['regime'].map(regime_names)
    regime_stats_df.to_csv("regime_analysis.csv", index=False)
    print(f"\n💾 Saved regime analysis to regime_analysis.csv")

# ============================================================================
# 4. SPLIT DATA
# ============================================================================

test_size = int(len(df_all) * 0.2)
train_end_idx = len(df_all) - test_size

train_data = df_all.iloc[:train_end_idx].copy()
test_data = df_all.iloc[train_end_idx:].copy()

print(f"\n📅 Data split:")
print(f"   Train: {len(train_data)} rows")
print(f"   Test:  {len(test_data)} rows")

# ============================================================================
# 5. TRAIN REGIME-SPECIFIC MODELS
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING REGIME-SPECIFIC MODELS")
print("=" * 80)

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

regime_models = {}

for regime_id in range(n_regimes):
    regime_name = regime_names.get(regime_id, f"Regime {regime_id}")
    regime_train = train_data[train_data['regime'] == regime_id]

    if len(regime_train) < 50:  # Skip if too few samples
        print(f"\n[{regime_id+1}/{n_regimes}] {regime_name}: ⚠️  Too few samples ({len(regime_train)}), skipping")
        continue

    print(f"\n[{regime_id+1}/{n_regimes}] {regime_name}")
    print(f"   Training samples: {len(regime_train)}")

    X_regime_train = regime_train[all_features]
    y_regime_train = regime_train['y']

    # Calculate sample weights
    class_counts = y_regime_train.value_counts()
    if len(class_counts) < 2:
        print(f"   ⚠️  Only one class present, skipping")
        continue

    total = len(y_regime_train)
    class_weight_dict = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
    sample_weights = y_regime_train.map(class_weight_dict)

    # Train model
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_regime_train, y_regime_train, sample_weight=sample_weights)

    # Evaluate on regime-specific test data
    regime_test = test_data[test_data['regime'] == regime_id]
    if len(regime_test) > 10:
        X_regime_test = regime_test[all_features]
        y_regime_test = regime_test['y']

        pred = model.predict(X_regime_test)
        acc = accuracy_score(y_regime_test, pred)

        print(f"   Test samples: {len(regime_test)}")
        print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    regime_models[regime_id] = {
        'model': model,
        'name': regime_name,
        'train_samples': len(regime_train)
    }

print(f"\n✅ Trained {len(regime_models)} regime-specific models")

# ============================================================================
# 6. TRAIN FALLBACK MODEL (FOR ALL DATA)
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FALLBACK MODEL (ALL DATA)")
print("=" * 80)

X_train_all = train_data[all_features]
y_train_all = train_data['y']

class_counts = y_train_all.value_counts()
total = len(y_train_all)
class_weight_dict = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
sample_weights = y_train_all.map(class_weight_dict)

fallback_model = lgb.LGBMClassifier(**lgb_params)
fallback_model.fit(X_train_all, y_train_all, sample_weight=sample_weights)

print(f"✅ Fallback model trained on {len(X_train_all)} samples")

# ============================================================================
# 7. EVALUATE REGIME-SWITCHING PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("REGIME-SWITCHING MODEL EVALUATION")
print("=" * 80)

X_test = test_data[all_features]
y_test = test_data['y']
test_regimes = test_data['regime'].values

# Make regime-aware predictions
regime_switch_preds = []
regime_switch_proba = []

for i, regime_id in enumerate(test_regimes):
    if regime_id in regime_models:
        # Use regime-specific model
        model = regime_models[regime_id]['model']
        pred = model.predict(X_test.iloc[[i]])[0]
        proba = model.predict_proba(X_test.iloc[[i]])[0, 1]
    else:
        # Use fallback model
        pred = fallback_model.predict(X_test.iloc[[i]])[0]
        proba = fallback_model.predict_proba(X_test.iloc[[i]])[0, 1]

    regime_switch_preds.append(pred)
    regime_switch_proba.append(proba)

regime_switch_preds = np.array(regime_switch_preds)

# Calculate metrics
acc_regime = accuracy_score(y_test, regime_switch_preds)
prec_regime = precision_score(y_test, regime_switch_preds, zero_division=0)
rec_regime = recall_score(y_test, regime_switch_preds)
f1_regime = f1_score(y_test, regime_switch_preds)

print(f"\n🎯 Regime-Switching Model Performance:")
print(f"   Accuracy:  {acc_regime:.4f} ({acc_regime*100:.2f}%)")
print(f"   Precision: {prec_regime:.4f}")
print(f"   Recall:    {rec_regime:.4f}")
print(f"   F1 Score:  {f1_regime:.4f}")

# Compare with single model (fallback)
fallback_preds = fallback_model.predict(X_test)
acc_fallback = accuracy_score(y_test, fallback_preds)
f1_fallback = f1_score(y_test, fallback_preds)

print(f"\n📊 Comparison:")
print(f"   Single Model:         {acc_fallback*100:.2f}%")
print(f"   Regime-Switching:     {acc_regime*100:.2f}%")
print(f"   Improvement:          {(acc_regime - acc_fallback)*100:+.2f}%")

# ============================================================================
# 8. PERFORMANCE BY REGIME
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BREAKDOWN BY REGIME")
print("=" * 80)

print(f"\n{'Regime':<20s} {'Samples':>10s} {'Accuracy':>12s} {'F1 Score':>12s}")
print("─" * 60)

for regime_id in range(n_regimes):
    regime_name = regime_names.get(regime_id, f"Regime {regime_id}")
    regime_mask = test_regimes == regime_id

    if regime_mask.sum() > 0:
        y_regime = y_test.values[regime_mask]
        pred_regime = regime_switch_preds[regime_mask]

        acc = accuracy_score(y_regime, pred_regime)
        f1 = f1_score(y_regime, pred_regime) if len(np.unique(y_regime)) > 1 else 0

        print(f"{regime_name:<20s} {regime_mask.sum():>10d} {acc*100:>11.2f}% {f1:>12.4f}")

# ============================================================================
# 9. SAVE REGIME-SWITCHING MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING REGIME-SWITCHING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "regime_switching_model.pkl"
joblib.dump({
    'hmm_model': hmm_model,
    'regime_models': regime_models,
    'fallback_model': fallback_model,
    'regime_names': regime_names,
    'features': all_features,
    'regime_detection_features': list(regime_data.columns),
    'accuracy': acc_regime,
    'f1_score': f1_regime,
    'training_date': datetime.now().isoformat()
}, model_path)

print(f"💾 Saved regime-switching model to: {model_path}")

# Save test predictions for analysis
test_results = test_data.copy()
test_results['prediction'] = regime_switch_preds
test_results['probability'] = regime_switch_proba
test_results.to_csv("regime_switching_predictions.csv")

print(f"💾 Saved predictions to: regime_switching_predictions.csv")

print("\n" + "=" * 80)
print("✅ PHASE 3 COMPLETE!")
print("=" * 80)

print(f"\n🎯 Results Summary:")
print(f"   Phase 1 (Cross-Asset):    58.38% → 61.46% (+5.3%)")
print(f"   Phase 2 (Macro):          61.46% → 62.31% (+1.4%)")
print(f"   Phase 3 (Regime-Switch):  62.31% → {acc_regime*100:.2f}% ({(acc_regime - 0.6231)*100:+.1f}%)")
print(f"\n   🚀 TOTAL IMPROVEMENT:     58.38% → {acc_regime*100:.2f}% ({(acc_regime - 0.5838)*100:+.1f}%)")

if acc_regime > 0.67:
    print(f"\n   🎊 EXCEEDED 67% ACCURACY TARGET!")
elif acc_regime > 0.65:
    print(f"\n   ✅ Strong progress toward 70% target!")
else:
    print(f"\n   📈 Good progress - Phase 4 (deep learning) can push further")

print(f"\n📁 Files saved:")
print(f"   - models/regime_switching_model.pkl")
print(f"   - regime_analysis.csv")
print(f"   - regime_switching_predictions.csv")

print(f"\n🎯 Next Steps (Optional Phase 4):")
print(f"   - LSTM/Transformer for sequential patterns (+5-8%)")
print(f"   - Options flow analysis (+3-5%)")
print(f"   - Sentiment analysis with FinBERT (+2-4%)")
print(f"   - Target: 75-80% accuracy")
