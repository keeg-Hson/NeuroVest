#!/usr/bin/env python3
"""
compare_optimization_metrics.py - Compare different optimization strategies

This script:
1. Evaluates the current model with different metrics
2. Shows what metrics we're actually optimizing for
3. Provides insights on which metric to target for retraining
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import MODELS_DIR, TRAIN_CFG
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels

print("=" * 80)
print("OPTIMIZATION METRIC COMPARISON")
print("=" * 80)
print("\nEvaluating the current model with different optimization metrics")
print("to understand which objective function yields best trading results.\n")

# Load model
print("📥 Loading trained model...")
model_path = MODELS_DIR / "market_crash_model_fwd_improved.pkl"
model_payload = joblib.load(model_path)
model = model_payload["model"]
saved_features = model_payload["features"]
print(f"✅ Model loaded: {len(saved_features)} features")

# Load and prepare data (following train_improved.py pattern)
print("\n📊 Loading data...")
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

# Keep only relevant columns
keep_cols = [c for c in saved_features if c in df.columns] + ["y", "fwd_ret_net"]
df = df[keep_cols].dropna()

print(f"✅ Data prepared: {len(df)} rows")

# Split into test set (last 20%)
test_size = int(len(df) * 0.2)
X_test = df.iloc[-test_size:][saved_features]
y_test = df.iloc[-test_size:]["y"]
returns_test = df.iloc[-test_size:]["fwd_ret_net"]

print(f"\n📊 Test set: {len(X_test)} samples")
print(f"   Class 0: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"   Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

# Get predictions at different thresholds
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 80)
print("EVALUATING DIFFERENT OPTIMIZATION OBJECTIVES")
print("=" * 80)

# Define custom scoring functions
def profit_score(y_true, y_pred, returns=None):
    """Profit-based score: TP gains vs FP losses"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (tp - fp) / len(y_true) if len(y_true) > 0 else 0

def sharpe_score(y_true, y_pred, returns=None):
    """Sharpe-like ratio"""
    trade_returns = np.zeros(len(y_true))
    trade_returns[(y_true == 1) & (y_pred == 1)] = 1.0
    trade_returns[(y_true == 0) & (y_pred == 1)] = -1.0

    if len(trade_returns[trade_returns != 0]) == 0:
        return 0.0

    mean_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns)
    return mean_ret / std_ret if std_ret > 0 else 0.0

def actual_profit(y_true, y_pred, returns):
    """Actual trading profit using real returns"""
    # Only enter positions where model predicts 1
    positions = y_pred == 1
    if positions.sum() == 0:
        return 0.0

    # Calculate cumulative return
    cumulative_return = np.sum(returns[positions])
    avg_return_per_trade = cumulative_return / positions.sum()

    return avg_return_per_trade

def win_rate(y_true, y_pred):
    """Win rate: % of predictions that are correct"""
    predictions = y_pred == 1
    if predictions.sum() == 0:
        return 0.0

    correct_predictions = np.sum((y_true == 1) & (y_pred == 1))
    return correct_predictions / predictions.sum()

# Test different thresholds
thresholds_to_test = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
results = []

for threshold in thresholds_to_test:
    y_pred = (y_proba >= threshold).astype(int)

    # Standard metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Custom trading metrics
    profit = profit_score(y_test, y_pred)
    sharpe = sharpe_score(y_test, y_pred)
    actual_pnl = actual_profit(y_test, y_pred, returns_test.values)
    win_pct = win_rate(y_test, y_pred)
    n_trades = (y_pred == 1).sum()

    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'profit_score': profit,
        'sharpe': sharpe,
        'actual_profit': actual_pnl,
        'win_rate': win_pct,
        'n_trades': n_trades,
    })

results_df = pd.DataFrame(results)

print("\n" + str(results_df.round(4)))

# Find best threshold for each metric
print("\n" + "=" * 80)
print("BEST THRESHOLD BY OPTIMIZATION OBJECTIVE")
print("=" * 80)

metrics_to_optimize = ['accuracy', 'f1', 'profit_score', 'sharpe', 'actual_profit']

for metric in metrics_to_optimize:
    best_idx = results_df[metric].idxmax()
    best_row = results_df.loc[best_idx]

    print(f"\n🎯 Best for {metric.upper()}:")
    print(f"   Threshold: {best_row['threshold']:.2f}")
    print(f"   Accuracy:  {best_row['accuracy']:.4f} ({best_row['accuracy']*100:.2f}%)")
    print(f"   Precision: {best_row['precision']:.4f} ({best_row['precision']*100:.2f}%)")
    print(f"   Recall:    {best_row['recall']:.4f} ({best_row['recall']*100:.2f}%)")
    print(f"   F1:        {best_row['f1']:.4f}")
    print(f"   Profit:    {best_row['profit_score']:.4f}")
    print(f"   Sharpe:    {best_row['sharpe']:.4f}")
    print(f"   Actual P&L: {best_row['actual_profit']:.6f} (avg per trade)")
    print(f"   Win Rate:  {best_row['win_rate']:.4f} ({best_row['win_rate']*100:.2f}%)")
    print(f"   # Trades:  {int(best_row['n_trades'])}")

# Save results
results_df.to_csv('optimization_metric_comparison.csv', index=False)
print(f"\n✅ Results saved to: optimization_metric_comparison.csv")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

best_profit_idx = results_df['actual_profit'].idxmax()
best_f1_idx = results_df['f1'].idxmax()
best_accuracy_idx = results_df['accuracy'].idxmax()

print(f"\n1. Current model optimizes for F1 score")
print(f"   Best F1 threshold: {results_df.loc[best_f1_idx, 'threshold']:.2f}")
print(f"   Actual profit at this threshold: {results_df.loc[best_f1_idx, 'actual_profit']:.6f}")

print(f"\n2. If we optimize for ACCURACY instead:")
print(f"   Best accuracy threshold: {results_df.loc[best_accuracy_idx, 'threshold']:.2f}")
print(f"   Actual profit at this threshold: {results_df.loc[best_accuracy_idx, 'actual_profit']:.6f}")
print(f"   Improvement: {(results_df.loc[best_accuracy_idx, 'actual_profit'] - results_df.loc[best_f1_idx, 'actual_profit']):.6f}")

print(f"\n3. If we optimize for PROFIT directly:")
print(f"   Best profit threshold: {results_df.loc[best_profit_idx, 'threshold']:.2f}")
print(f"   Actual profit: {results_df.loc[best_profit_idx, 'actual_profit']:.6f}")
print(f"   Improvement over F1: {(results_df.loc[best_profit_idx, 'actual_profit'] - results_df.loc[best_f1_idx, 'actual_profit']):.6f}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

best_overall = results_df.loc[best_profit_idx]
print(f"\n💡 For maximum trading profit, use threshold: {best_overall['threshold']:.2f}")
print(f"\n   This gives you:")
print(f"   - Average profit per trade: {best_overall['actual_profit']:.6f}")
print(f"   - Win rate: {best_overall['win_rate']*100:.2f}%")
print(f"   - {int(best_overall['n_trades'])} trades (out of {len(X_test)} days)")
print(f"   - Precision: {best_overall['precision']*100:.2f}%")
print(f"   - Accuracy: {best_overall['accuracy']*100:.2f}%")

if best_profit_idx != best_f1_idx:
    improvement = (best_overall['actual_profit'] - results_df.loc[best_f1_idx, 'actual_profit']) / abs(results_df.loc[best_f1_idx, 'actual_profit']) * 100
    print(f"\n   📈 This is {improvement:.1f}% better than F1-optimized threshold!")

print("\n🎉 Analysis complete!")
