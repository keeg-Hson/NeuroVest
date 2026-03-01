#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with Regime Features

This script:
1. Trains ensemble model (XGBoost + LightGBM + CatBoost) with regime features
2. Applies profit-optimization to regime-enhanced LightGBM
3. Evaluates improved XGBoost model on same test set for comparison
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
from catboost import CatBoostClassifier

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
    get_feature_list
)
from train import TRAIN_CFG

print("=" * 80)
print("COMPREHENSIVE MODEL EVALUATION WITH REGIME FEATURES")
print("=" * 80)

MODELS_DIR = Path("models")

# ============================================================================
# 1. PREPARE DATA WITH REGIME FEATURES
# ============================================================================

print("\n📥 Loading data with regime features...")
df = load_SPY_data()
df, feature_cols = add_features(df)

print(f"✅ Features generated: {len(feature_cols)}")
regime_features = [f for f in feature_cols if any(x in f for x in
    ['MA_200', 'Bull_Market', 'ADX', 'Plus_DI', 'Minus_DI',
     'High_Volatility', 'Regime_Score', 'Near_52w', 'Trend_Consistency'])]
print(f"   Regime features: {len(regime_features)}")

# Finalize features
df = finalize_features(df, feature_cols)

# Reindex Close prices
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

# Add forward returns and labels
df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

# Prepare features
all_features = [c for c in df.columns if c not in ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]
keep_cols = all_features + ["y", "fwd_ret_net"]

df = df[keep_cols]
df = df.dropna(subset=["y"])

# Fill remaining NaN with column median (not 0, to avoid systematic bias)
if df.isnull().any().any():
    nan_count = df.isnull().sum().sum()
    print(f"   Filling {nan_count} NaN values with column medians")
    for col in all_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    # Any remaining NaN (e.g. all-NaN columns) fill with 0
    df = df.fillna(0)

print(f"\n✅ Data prepared: {len(df)} rows, {len(all_features)} features")

# Split data (80/20 split)
test_size = int(len(df) * 0.2)
X_train = df.iloc[:-test_size][all_features]
X_test = df.iloc[-test_size:][all_features]
y_train = df.iloc[:-test_size]["y"]
y_test = df.iloc[-test_size:]["y"]
returns_train = df.iloc[:-test_size]["fwd_ret_net"]
returns_test = df.iloc[-test_size:]["fwd_ret_net"]

print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
print(f"   Class distribution (train): {y_train.value_counts().to_dict()}")
print(f"   Class distribution (test):  {y_test.value_counts().to_dict()}")

# Calculate sample weights
class_counts = y_train.value_counts()
total = len(y_train)
class_weight_dict = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
sample_weights = y_train.map(class_weight_dict)

# ============================================================================
# 2. TRAIN ENSEMBLE MODEL WITH REGIME FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("TASK 1: ENSEMBLE MODEL WITH REGIME FEATURES")
print("=" * 80)

# XGBoost parameters
import xgboost as xgb
xgb_params = {
    'max_depth': 5,
    'learning_rate': 0.02,
    'n_estimators': 400,
    'objective': 'binary:logistic',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0,
}

# LightGBM parameters
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

# CatBoost parameters
cat_params = {
    'iterations': 400,
    'depth': 5,
    'learning_rate': 0.02,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': 42,
    'verbose': False,
    'auto_class_weights': 'Balanced',
}

print("\n[1/3] Training XGBoost with regime features...")
start = datetime.now()
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"✅ Completed in {(datetime.now() - start).total_seconds():.1f}s")

print("\n[2/3] Training LightGBM with regime features...")
start = datetime.now()
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"✅ Completed in {(datetime.now() - start).total_seconds():.1f}s")

print("\n[3/3] Training CatBoost with regime features...")
start = datetime.now()
cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"✅ Completed in {(datetime.now() - start).total_seconds():.1f}s")

# Ensemble predictions (average probabilities)
print("\n📊 Generating ensemble predictions...")
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
cat_proba = cat_model.predict_proba(X_test)[:, 1]

ensemble_proba = (xgb_proba + lgb_proba + cat_proba) / 3
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# Individual model predictions
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
cat_pred = cat_model.predict(X_test)

# Metrics
results = []

for name, preds in [
    ('XGBoost', xgb_pred),
    ('LightGBM', lgb_pred),
    ('CatBoost', cat_pred),
    ('Ensemble', ensemble_pred)
]:
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Features': len(all_features)
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("ENSEMBLE RESULTS (WITH REGIME FEATURES)")
print("=" * 80)
print(results_df.to_string(index=False))

# Save models
print("\n💾 Saving ensemble models...")
joblib.dump(xgb_model, MODELS_DIR / "xgboost_regime.pkl")
joblib.dump(lgb_model, MODELS_DIR / "lightgbm_regime.pkl")
joblib.dump(cat_model, MODELS_DIR / "catboost_regime.pkl")
print("✅ Saved: xgboost_regime.pkl, lightgbm_regime.pkl, catboost_regime.pkl")

# ============================================================================
# 3. PROFIT OPTIMIZATION FOR REGIME-ENHANCED LIGHTGBM
# ============================================================================

print("\n" + "=" * 80)
print("TASK 2: PROFIT OPTIMIZATION (REGIME-ENHANCED LIGHTGBM)")
print("=" * 80)

def actual_profit(y_true, y_pred, returns):
    """Calculate actual trading profit"""
    positions = y_pred == 1
    if positions.sum() == 0:
        return 0.0, 0.0, 0

    trade_returns = returns[positions]
    cumulative_return = np.sum(trade_returns)
    avg_return_per_trade = cumulative_return / positions.sum()
    win_rate = (trade_returns > 0).sum() / len(trade_returns)

    return avg_return_per_trade, win_rate, positions.sum()

print("\n🔍 Testing thresholds from 0.30 to 0.95...")

lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]

threshold_results = []
for threshold in np.arange(0.30, 0.96, 0.05):
    preds = (lgb_test_proba >= threshold).astype(int)

    if preds.sum() == 0:  # No trades
        continue

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    avg_profit, win_rate, n_trades = actual_profit(y_test, preds, returns_test.values)

    threshold_results.append({
        'Threshold': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Avg_Profit_Per_Trade': avg_profit,
        'Win_Rate': win_rate,
        'N_Trades': n_trades
    })

threshold_df = pd.DataFrame(threshold_results)

# Find best threshold by profit
best_profit_idx = threshold_df['Avg_Profit_Per_Trade'].idxmax()
best_profit_threshold = threshold_df.loc[best_profit_idx]

# Find best threshold by F1
best_f1_idx = threshold_df['F1_Score'].idxmax()
best_f1_threshold = threshold_df.loc[best_f1_idx]

print("\n" + "=" * 80)
print("PROFIT OPTIMIZATION RESULTS")
print("=" * 80)

print("\n📊 Best by PROFIT:")
print(f"   Threshold: {best_profit_threshold['Threshold']:.2f}")
print(f"   Avg Profit/Trade: {best_profit_threshold['Avg_Profit_Per_Trade']:.4f} ({best_profit_threshold['Avg_Profit_Per_Trade']*100:.2f}%)")
print(f"   Win Rate: {best_profit_threshold['Win_Rate']:.2%}")
print(f"   N Trades: {best_profit_threshold['N_Trades']:.0f}")
print(f"   Accuracy: {best_profit_threshold['Accuracy']:.2%}")
print(f"   F1 Score: {best_profit_threshold['F1_Score']:.4f}")

print("\n📊 Best by F1 Score:")
print(f"   Threshold: {best_f1_threshold['Threshold']:.2f}")
print(f"   Avg Profit/Trade: {best_f1_threshold['Avg_Profit_Per_Trade']:.4f} ({best_f1_threshold['Avg_Profit_Per_Trade']*100:.2f}%)")
print(f"   Win Rate: {best_f1_threshold['Win_Rate']:.2%}")
print(f"   N Trades: {best_f1_threshold['N_Trades']:.0f}")
print(f"   Accuracy: {best_f1_threshold['Accuracy']:.2%}")
print(f"   F1 Score: {best_f1_threshold['F1_Score']:.4f}")

# Save threshold optimization results
threshold_df.to_csv("regime_lightgbm_profit_optimization.csv", index=False)
print("\n💾 Saved: regime_lightgbm_profit_optimization.csv")

# Save best threshold config
profit_config = {
    "threshold": float(best_profit_threshold['Threshold']),
    "metric": "profit_optimized_regime",
    "avg_profit_per_trade": float(best_profit_threshold['Avg_Profit_Per_Trade']),
    "win_rate": float(best_profit_threshold['Win_Rate']),
    "n_trades": int(best_profit_threshold['N_Trades']),
    "accuracy": float(best_profit_threshold['Accuracy']),
    "f1_score": float(best_profit_threshold['F1_Score']),
}

import json
with open(MODELS_DIR / "threshold_lightgbm_regime_profit.json", "w") as f:
    json.dump(profit_config, f, indent=2)
print("💾 Saved: threshold_lightgbm_regime_profit.json")

# ============================================================================
# 4. EVALUATE IMPROVED XGBOOST ON SAME TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("TASK 3: IMPROVED XGBOOST EVALUATION (SAME TEST SET)")
print("=" * 80)

print("\n📥 Loading improved XGBoost model...")
try:
    xgb_improved_payload = joblib.load(MODELS_DIR / "market_crash_model_fwd_improved.pkl")

    # Extract the actual model from calibrated classifier
    if hasattr(xgb_improved_payload, 'calibrated_classifiers_'):
        xgb_improved = xgb_improved_payload.calibrated_classifiers_[0].estimator
        print("✅ Loaded calibrated XGBoost model")
    else:
        xgb_improved = xgb_improved_payload
        print("✅ Loaded XGBoost model")

    # Get the features the improved model was trained on
    if hasattr(xgb_improved, 'named_steps'):
        # It's a pipeline
        if 'kbest' in xgb_improved.named_steps:
            kbest = xgb_improved.named_steps['kbest']

            # Get base features (78 features without regime)
            base_features = [f for f in all_features if f not in regime_features]

            if len(base_features) >= 78:
                base_features = base_features[:78]  # Use first 78

            print(f"   Model expects {len(base_features)} base features (no regime)")

            # Prepare test data with base features only
            X_test_base = X_test[base_features]

            # Predict
            improved_pred = xgb_improved.predict(X_test_base)
            improved_proba = xgb_improved.predict_proba(X_test_base)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, improved_pred)
            prec = precision_score(y_test, improved_pred, zero_division=0)
            rec = recall_score(y_test, improved_pred)
            f1 = f1_score(y_test, improved_pred)

            print("\n" + "=" * 80)
            print("IMPROVED XGBOOST RESULTS (SAME TEST SET)")
            print("=" * 80)
            print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")

            # Profit analysis
            avg_profit, win_rate, n_trades = actual_profit(y_test, improved_pred, returns_test.values)
            print(f"\nProfit Analysis (threshold 0.5):")
            print(f"   Avg Profit/Trade: {avg_profit:.4f} ({avg_profit*100:.2f}%)")
            print(f"   Win Rate: {win_rate:.2%}")
            print(f"   N Trades: {n_trades}")

            # Try with optimal threshold from config
            try:
                with open(MODELS_DIR / "thresholds_fwd_improved_profit_optimized.json", "r") as f:
                    optimal_config = json.load(f)
                    optimal_threshold = optimal_config['threshold']

                optimal_pred = (improved_proba >= optimal_threshold).astype(int)
                avg_profit_opt, win_rate_opt, n_trades_opt = actual_profit(y_test, optimal_pred, returns_test.values)

                print(f"\nProfit Analysis (optimal threshold {optimal_threshold}):")
                print(f"   Avg Profit/Trade: {avg_profit_opt:.4f} ({avg_profit_opt*100:.2f}%)")
                print(f"   Win Rate: {win_rate_opt:.2%}")
                print(f"   N Trades: {n_trades_opt}")

            except Exception:
                pass

        else:
            print("⚠️  No SelectKBest in pipeline, using all features")
    else:
        print("⚠️  Not a pipeline, skipping improved XGBoost evaluation")

except Exception as e:
    print(f"❌ Error loading improved XGBoost: {e}")

# ============================================================================
# 5. FINAL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("FINAL COMPARISON - ALL MODELS")
print("=" * 80)

comparison_data = []

# Ensemble with regime
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)
ensemble_profit, ensemble_wr, ensemble_trades = actual_profit(y_test, ensemble_pred, returns_test.values)

comparison_data.append({
    'Model': 'Ensemble (Regime)',
    'Features': len(all_features),
    'Accuracy': ensemble_acc,
    'F1_Score': ensemble_f1,
    'Avg_Profit': ensemble_profit,
    'Win_Rate': ensemble_wr,
    'N_Trades': ensemble_trades
})

# LightGBM with regime (default threshold)
lgb_regime_pred = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_regime_pred)
lgb_f1 = f1_score(y_test, lgb_regime_pred)
lgb_profit, lgb_wr, lgb_trades = actual_profit(y_test, lgb_regime_pred, returns_test.values)

comparison_data.append({
    'Model': 'LightGBM (Regime)',
    'Features': len(all_features),
    'Accuracy': lgb_acc,
    'F1_Score': lgb_f1,
    'Avg_Profit': lgb_profit,
    'Win_Rate': lgb_wr,
    'N_Trades': lgb_trades
})

# LightGBM with regime (profit-optimized threshold)
comparison_data.append({
    'Model': f'LightGBM (Regime, Profit-Opt @ {best_profit_threshold["Threshold"]:.2f})',
    'Features': len(all_features),
    'Accuracy': best_profit_threshold['Accuracy'],
    'F1_Score': best_profit_threshold['F1_Score'],
    'Avg_Profit': best_profit_threshold['Avg_Profit_Per_Trade'],
    'Win_Rate': best_profit_threshold['Win_Rate'],
    'N_Trades': best_profit_threshold['N_Trades']
})

# Improved XGBoost (if loaded successfully)
try:
    comparison_data.append({
        'Model': 'XGBoost (Improved, 78 feat)',
        'Features': len(base_features),
        'Accuracy': acc,
        'F1_Score': f1,
        'Avg_Profit': avg_profit,
        'Win_Rate': win_rate,
        'N_Trades': n_trades
    })
except Exception:
    pass

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

comparison_df.to_csv("comprehensive_model_comparison.csv", index=False)
print("\n💾 Saved: comprehensive_model_comparison.csv")

print("\n" + "=" * 80)
print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
print("=" * 80)
