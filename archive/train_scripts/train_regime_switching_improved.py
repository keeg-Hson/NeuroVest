#!/usr/bin/env python3
"""
Improved Regime-Switching Model

Uses a simpler, more effective regime detection based on:
1. Market Trend (Bull vs Bear using 200-day MA)
2. Volatility Regime (Low vs High VIX/realized vol)

This creates 3 clear regimes:
- Bull/Low Vol (normal bullish markets)
- Bull/High Vol (volatile rallies)
- Bear/High Vol (corrections/crashes)

Expected to outperform HMM approach.
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
print("IMPROVED REGIME-SWITCHING MODEL (PHASE 3)")
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
# 2. SIMPLE RULE-BASED REGIME DETECTION
# ============================================================================

print("\n" + "=" * 80)
print("REGIME DETECTION (RULE-BASED)")
print("=" * 80)

# Calculate regime indicators
close = df_all['Close']
returns = close.pct_change()
ma_200 = close.rolling(200).mean()

# Trend: Bull if above 200-day MA
is_bull = close > ma_200

# Volatility: Use 20-day realized volatility
realized_vol_20 = returns.rolling(20).std() * np.sqrt(252) * 100

# Define volatility threshold (use 75th percentile as "high vol")
vol_threshold = realized_vol_20.rolling(252).quantile(0.70)
is_high_vol = realized_vol_20 > vol_threshold

# Create regimes
regime = pd.Series('Unknown', index=df_all.index)

# Regime 0: Bull + Low Vol (normal bull market)
regime[(is_bull) & (~is_high_vol)] = 'Bull_LowVol'

# Regime 1: Bull + High Vol (volatile rally / rotation)
regime[(is_bull) & (is_high_vol)] = 'Bull_HighVol'

# Regime 2: Bear + Low Vol (slow grind down / choppy bear)
regime[(~is_bull) & (~is_high_vol)] = 'Bear_LowVol'

# Regime 3: Bear + High Vol (crash / panic selling)
regime[(~is_bull) & (is_high_vol)] = 'Bear_HighVol'

df_all['regime'] = regime

print(f"\n📊 Regime Definition:")
print(f"   • Bull/Bear: Price vs 200-day MA")
print(f"   • Low/High Vol: 20-day realized vol vs 70th percentile")

# ============================================================================
# 3. ANALYZE REGIMES
# ============================================================================

print("\n" + "=" * 80)
print("REGIME CHARACTERISTICS ANALYSIS")
print("=" * 80)

regime_stats = []
for regime_name in ['Bull_LowVol', 'Bull_HighVol', 'Bear_LowVol', 'Bear_HighVol']:
    regime_mask = df_all['regime'] == regime_name
    regime_data = df_all[regime_mask]

    if len(regime_data) > 0:
        avg_return = regime_data['Close'].pct_change().mean() * 100
        avg_vol = regime_data['Close'].pct_change().rolling(20).std().mean() * 100
        label_distribution = regime_data['y'].mean()

        regime_stats.append({
            'regime': regime_name,
            'count': len(regime_data),
            'pct': len(regime_data) / len(df_all) * 100,
            'avg_return': avg_return,
            'avg_vol': avg_vol,
            'bullish_label_pct': label_distribution * 100
        })

regime_stats_df = pd.DataFrame(regime_stats)

print(f"\n{'Regime':<20s} {'Days':>8s} {'Pct':>8s} {'Avg Ret':>10s} {'Avg Vol':>10s} {'Bullish %':>10s}")
print("─" * 80)

for idx, row in regime_stats_df.iterrows():
    print(f"{row['regime']:<20s} {row['count']:>8.0f} {row['pct']:>7.1f}% {row['avg_return']:>+9.3f}% {row['avg_vol']:>9.2f}% {row['bullish_label_pct']:>9.1f}%")

# Save regime analysis
regime_stats_df.to_csv("regime_analysis_improved.csv", index=False)
print(f"\n💾 Saved regime analysis to regime_analysis_improved.csv")

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

for i, regime_name in enumerate(['Bull_LowVol', 'Bull_HighVol', 'Bear_LowVol', 'Bear_HighVol'], 1):
    regime_train = train_data[train_data['regime'] == regime_name]

    if len(regime_train) < 50:  # Skip if too few samples
        print(f"\n[{i}/4] {regime_name}: ⚠️  Too few samples ({len(regime_train)}), skipping")
        continue

    print(f"\n[{i}/4] {regime_name}")
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
    regime_test = test_data[test_data['regime'] == regime_name]
    if len(regime_test) > 10:
        X_regime_test = regime_test[all_features]
        y_regime_test = regime_test['y']

        pred = model.predict(X_regime_test)
        acc = accuracy_score(y_regime_test, pred)

        print(f"   Test samples: {len(regime_test)}")
        print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    regime_models[regime_name] = {
        'model': model,
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
model_used = []

for i, regime_name in enumerate(test_regimes):
    if regime_name in regime_models:
        # Use regime-specific model
        model = regime_models[regime_name]['model']
        pred = model.predict(X_test.iloc[[i]])[0]
        proba = model.predict_proba(X_test.iloc[[i]])[0, 1]
        model_used.append(regime_name)
    else:
        # Use fallback model
        pred = fallback_model.predict(X_test.iloc[[i]])[0]
        proba = fallback_model.predict_proba(X_test.iloc[[i]])[0, 1]
        model_used.append('Fallback')

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
improvement = (acc_regime - acc_fallback) * 100
print(f"   Improvement:          {improvement:+.2f}%")

# ============================================================================
# 8. PERFORMANCE BY REGIME
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BREAKDOWN BY REGIME")
print("=" * 80)

print(f"\n{'Regime':<20s} {'Samples':>10s} {'Accuracy':>12s} {'F1 Score':>12s}")
print("─" * 60)

for regime_name in ['Bull_LowVol', 'Bull_HighVol', 'Bear_LowVol', 'Bear_HighVol']:
    regime_mask = test_regimes == regime_name

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
print("SAVING IMPROVED REGIME-SWITCHING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "regime_switching_improved.pkl"
joblib.dump({
    'regime_models': regime_models,
    'fallback_model': fallback_model,
    'features': all_features,
    'accuracy': acc_regime,
    'f1_score': f1_regime,
    'training_date': datetime.now().isoformat()
}, model_path)

print(f"💾 Saved regime-switching model to: {model_path}")

# Save test predictions for analysis
test_results = test_data.copy()
test_results['prediction'] = regime_switch_preds
test_results['probability'] = regime_switch_proba
test_results['model_used'] = model_used
test_results.to_csv("regime_switching_improved_predictions.csv")

print(f"💾 Saved predictions to: regime_switching_improved_predictions.csv")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ PHASE 3 COMPLETE (IMPROVED)!")
print("=" * 80)

baseline_acc = 0.5838  # From Phase 0
phase1_acc = 0.6146    # From Phase 1
phase2_acc = 0.6231    # From Phase 2

print(f"\n🎯 Progressive Results Summary:")
print(f"   Baseline (existing):      {baseline_acc*100:.2f}%")
print(f"   Phase 1 (+Cross-Asset):   {phase1_acc*100:.2f}% ({(phase1_acc-baseline_acc)*100:+.2f}%)")
print(f"   Phase 2 (+Macro):         {phase2_acc*100:.2f}% ({(phase2_acc-phase1_acc)*100:+.2f}%)")
print(f"   Phase 3 (+Regime-Switch): {acc_regime*100:.2f}% ({(acc_regime-phase2_acc)*100:+.2f}%)")
print(f"\n   🚀 TOTAL IMPROVEMENT:     {baseline_acc*100:.2f}% → {acc_regime*100:.2f}% ({(acc_regime-baseline_acc)*100:+.2f}%)")

if acc_regime > 0.70:
    print(f"\n   🎊 ACHIEVED 70%+ ACCURACY TARGET!")
    achievement = "EXCELLENT"
elif acc_regime > 0.67:
    print(f"\n   🎉 EXCEEDED 67% TARGET!")
    achievement = "VERY GOOD"
elif acc_regime > 0.65:
    print(f"\n   ✅ Strong improvement achieved!")
    achievement = "GOOD"
else:
    print(f"\n   📈 Good progress - Phase 4 can add more")
    achievement = "MODERATE"

print(f"\n📊 Model Quality: {achievement}")
print(f"   F1 Score: {f1_regime:.4f}")
print(f"   Precision: {prec_regime:.4f}")
print(f"   Recall: {rec_regime:.4f}")

print(f"\n📁 Files saved:")
print(f"   - models/regime_switching_improved.pkl")
print(f"   - regime_analysis_improved.csv")
print(f"   - regime_switching_improved_predictions.csv")

print(f"\n🎯 Next Steps (Optional Phase 4):")
print(f"   - LSTM/Transformer for sequential patterns (+5-8%)")
print(f"   - Options flow analysis (+3-5%)")
print(f"   - Sentiment analysis with FinBERT (+2-4%)")
print(f"   - Ensemble stacking across regimes (+1-3%)")
print(f"   - Target: 75-80% accuracy")
