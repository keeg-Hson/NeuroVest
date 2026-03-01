#!/usr/bin/env python3
"""
Evaluate the improved XGBoost model on the same test set
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)
from train import TRAIN_CFG

print("=" * 80)
print("IMPROVED XGBOOST EVALUATION")
print("=" * 80)

MODELS_DIR = Path("models")

# Load data
print("\n📥 Loading data...")
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

# Get regime features
regime_features = [f for f in feature_cols if any(x in f for x in
    ['MA_200', 'Bull_Market', 'ADX', 'Plus_DI', 'Minus_DI',
     'High_Volatility', 'Regime_Score', 'Near_52w', 'Trend_Consistency'])]

# All features
all_features = [c for c in df.columns if c not in ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

# Base features (without regime)
base_features = [f for f in all_features if f not in regime_features]

print(f"✅ Total features: {len(all_features)}")
print(f"   Base features (no regime): {len(base_features)}")
print(f"   Regime features: {len(regime_features)}")

# Prepare data
keep_cols = all_features + ["y", "fwd_ret_net"]
df = df[keep_cols]
df = df.dropna(subset=["y"])
df = df.fillna(0)

# Split (80/20)
test_size = int(len(df) * 0.2)
X_test_all = df.iloc[-test_size:][all_features]
X_test_base = df.iloc[-test_size:][base_features]
y_test = df.iloc[-test_size:]["y"]
returns_test = df.iloc[-test_size:]["fwd_ret_net"]

print(f"\n✅ Test set: {len(y_test)} rows")
print(f"   Class distribution: {y_test.value_counts().to_dict()}")

# Load improved XGBoost
print("\n📥 Loading improved XGBoost model...")
try:
    model_data = joblib.load(MODELS_DIR / "market_crash_model_fwd_improved.pkl")
    print(f"✅ Loaded data type: {type(model_data).__name__}")

    # Extract model and features from dict
    if isinstance(model_data, dict):
        print(f"   Dict keys: {list(model_data.keys())}")
        model = model_data['model']
        model_features = model_data['features']
        print(f"✅ Extracted model: {type(model).__name__}")
        print(f"   Model expects {len(model_features)} features")
    else:
        model = model_data
        model_features = base_features

    # Try to predict with model features
    print(f"\n🔍 Preparing features for prediction...")

    try:
        # Use the features the model was trained on
        # Check which features are available
        available_features = [f for f in model_features if f in X_test_all.columns]
        missing_features = [f for f in model_features if f not in X_test_all.columns]

        if missing_features:
            print(f"⚠️  Missing {len(missing_features)} features: {missing_features[:5]}")

        print(f"✅ Using {len(available_features)}/{len(model_features)} features")

        X_test_model = X_test_all[available_features]

        # Try direct prediction
        y_pred = model.predict(X_test_model)
        y_proba = model.predict_proba(X_test_model)[:, 1]

        print("✅ Prediction successful!")

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n" + "=" * 80)
        print(f"IMPROVED XGBOOST RESULTS ({len(available_features)} FEATURES)")
        print("=" * 80)
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Profit analysis
        def actual_profit(y_true, y_pred, returns):
            positions = y_pred == 1
            if positions.sum() == 0:
                return 0.0, 0.0, 0
            trade_returns = returns[positions]
            cumulative_return = np.sum(trade_returns)
            avg_return_per_trade = cumulative_return / positions.sum()
            win_rate = (trade_returns > 0).sum() / len(trade_returns)
            return avg_return_per_trade, win_rate, positions.sum()

        avg_profit, win_rate, n_trades = actual_profit(y_test, y_pred, returns_test.values)

        print(f"\nProfit Analysis (default threshold 0.5):")
        print(f"   Avg Profit/Trade: {avg_profit:.6f} ({avg_profit*100:.4f}%)")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   N Trades: {n_trades}")

        # Try with profit-optimized threshold
        try:
            import json
            with open(MODELS_DIR / "thresholds_fwd_improved_profit_optimized.json", "r") as f:
                config = json.load(f)
                opt_threshold = config['threshold']

            opt_pred = (y_proba >= opt_threshold).astype(int)
            opt_profit, opt_wr, opt_trades = actual_profit(y_test, opt_pred, returns_test.values)

            print(f"\nProfit Analysis (optimized threshold {opt_threshold}):")
            print(f"   Avg Profit/Trade: {opt_profit:.6f} ({opt_profit*100:.4f}%)")
            print(f"   Win Rate: {opt_wr:.2%}")
            print(f"   N Trades: {opt_trades}")

        except Exception as e:
            print(f"\n⚠️  Could not load profit-optimized threshold: {e}")

        # Save results
        results = {
            'Model': 'XGBoost (Improved)',
            'Features': len(available_features),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Avg_Profit': avg_profit,
            'Win_Rate': win_rate,
            'N_Trades': n_trades
        }

        results_df = pd.DataFrame([results])
        results_df.to_csv("improved_xgboost_evaluation.csv", index=False)
        print("\n💾 Saved: improved_xgboost_evaluation.csv")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Try with all features including regime
        print(f"\n🔍 Trying with all {len(all_features)} features (including regime)...")
        try:
            y_pred = model.predict(X_test_all)
            print("✅ Prediction successful with all features!")

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"   Accuracy: {acc:.4f}")
            print(f"   F1 Score: {f1:.4f}")

        except Exception as e2:
            print(f"❌ Also failed with all features: {e2}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Error type: {type(e).__name__}")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE")
print("=" * 80)
