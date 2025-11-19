#!/usr/bin/env python3
"""
train_profit_optimized.py - Retrain model optimized for profit instead of F1

Key differences from train_improved.py:
1. Custom scoring function based on actual trading profit
2. Optimizes for Sharpe ratio or total profit instead of F1 score
3. More realistic evaluation metric for trading strategy
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from datetime import datetime

from config import MODELS_DIR, TRAIN_CFG
from utils import (
    load_SPY_data,
    add_features,
    add_forward_returns_and_labels,
    finalize_features,
    compute_sample_weights,
)

print("=" * 80)
print("PROFIT-OPTIMIZED TRAINING")
print("=" * 80)
print("\nInstead of optimizing F1 score, we optimize for actual trading profit.")
print("This should find hyperparameters that maximize real-world returns.\n")

# Custom profit-based scoring function
def profit_score(y_true, y_pred, sample_weight=None):
    """
    Calculate profit-based score for trading strategy.

    For each correct positive prediction: +1 (profitable trade)
    For each false positive: -1 (losing trade)
    For true negatives and false negatives: 0 (no trade or missed opportunity)

    This approximates actual trading P&L.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True positives: correct buy signals (profitable)
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False positives: wrong buy signals (losses)
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # Calculate profit (TP worth +1 each, FP worth -1 each)
    profit = tp - fp

    # Normalize by number of samples to get average profit per prediction
    total_predictions = tp + fp
    if total_predictions == 0:
        return 0.0

    return profit / len(y_true)

def sharpe_score(y_true, y_pred, sample_weight=None):
    """
    Calculate Sharpe-like ratio for predictions.

    Higher is better. Rewards consistent profits, penalizes volatility.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate per-trade returns
    # Correct predictions: +1, incorrect: -1, no prediction: 0
    returns = np.zeros(len(y_true))

    # True positives: +1
    returns[(y_true == 1) & (y_pred == 1)] = 1.0

    # False positives: -1
    returns[(y_true == 0) & (y_pred == 1)] = -1.0

    # True negatives and false negatives: 0 (no trade)

    if len(returns[returns != 0]) == 0:
        return 0.0

    # Calculate mean and std of returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Sharpe ratio (annualized approximation)
    sharpe = mean_return / std_return

    return sharpe

def accuracy_score_custom(y_true, y_pred, sample_weight=None):
    """Simple accuracy score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

# Load data
print("📥 Loading SPY data...")
df = load_SPY_data()
print(f"✅ Loaded {len(df)} rows")

# Add features
print("\n📊 Generating features...")
df, feature_cols = add_features(df)
print(f"✅ Feature engineering complete: {len(df.columns)} columns")

# Finalize features BEFORE labeling (same as train_improved.py)
print("\n🔧 Finalizing features...")
df = finalize_features(df, feature_cols)

# Ensure Close exists for labeling
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

# Add forward returns and labels
print("\n🎯 Adding forward returns and labels...")
df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)
print(f"✅ Labels added")

# Remove inf values
df = df.replace([np.inf, -np.inf], np.nan)

# Separate features from target and meta columns
feature_cols_final = [c for c in df.columns if c not in ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

# Drop NaN only in feature columns and target
df_clean = df[feature_cols_final + ["y", "fwd_ret_net"]].dropna()
print(f"✅ After removing NaN: {len(df_clean)} rows")

# Prepare features and target
X = df_clean[feature_cols_final]
y = df_clean["y"]
df = df_clean  # Use cleaned dataframe for sample weights

print(f"\n📊 Final dataset:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(X.columns)}")
print(f"   Class 0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"   Class 1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# Time-series cross-validation
print("\n🔄 Setting up time-series cross-validation...")
cv_splits = 4
tscv = TimeSeriesSplit(n_splits=cv_splits)

# Sample weights
print("\n⚖️ Computing sample weights...")
try:
    sample_weight = compute_sample_weights(df)
    print(f"✅ Sample weights computed")
except Exception as e:
    print(f"⚠️ Sample weight computation failed: {e}")
    print("   Using uniform weights")
    sample_weight = np.ones(len(df))

# XGBoost configuration
print("\n🤖 Configuring XGBoost...")
xgb_common = dict(
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    tree_method="hist",
    use_label_encoder=False,
)
xgb_obj = dict(objective="binary:logistic", eval_metric="logloss")

# Pipeline with feature selection
use_kbest = X.shape[1] >= 40
print(f"   Using feature selection: {use_kbest}")

if use_kbest:
    # Pre-filter with ExtraTrees
    print("   Pre-filtering features with ExtraTrees...")
    et_clf = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    et_clf.fit(X, y, sample_weight=sample_weight)

    selector = SelectFromModel(et_clf, prefit=True, threshold="median")
    X_filtered = pd.DataFrame(
        selector.transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index
    )
    print(f"   Features after pre-filtering: {X_filtered.shape[1]}")

    # Use filtered features
    X = X_filtered

    # Feature selection values
    k_choices = [20, 30, min(39, X.shape[1]), min(40, X.shape[1]), X.shape[1]]
    k_choices = sorted(list(set(k_choices)))

    smote_step = SMOTE(random_state=42, k_neighbors=min(5, (y == 1).sum() - 1))

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("varth", VarianceThreshold(threshold=0.0)),
        ("smote", smote_step),
        ("kbest", SelectKBest(mutual_info_classif)),
        ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
    ]

    param_grid = {
        "kbest__k": k_choices,
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.02, 0.05],
        "clf__subsample": [0.7, 0.8],
        "clf__colsample_bytree": [0.7, 0.8],
        "clf__min_child_weight": [5, 10, 15],
        "clf__gamma": [0, 0.5, 1.0],
        "clf__reg_alpha": [0, 0.1, 0.5],
        "clf__reg_lambda": [1.0, 2.0, 3.0],
        "clf__scale_pos_weight": [1.0, 1.5, 2.0],
    }
else:
    smote_step = SMOTE(random_state=42, k_neighbors=min(5, (y == 1).sum() - 1))

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("varth", VarianceThreshold(threshold=0.0)),
        ("smote", smote_step),
        ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
    ]

    param_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.02, 0.05],
        "clf__subsample": [0.7, 0.8],
        "clf__colsample_bytree": [0.7, 0.8],
        "clf__min_child_weight": [5, 10, 15],
        "clf__gamma": [0, 0.5, 1.0],
        "clf__reg_alpha": [0, 0.1, 0.5],
        "clf__reg_lambda": [1.0, 2.0, 3.0],
        "clf__scale_pos_weight": [1.0, 1.5, 2.0],
    }

pipe = Pipeline(steps=steps)

# Calculate grid size
grid_size = 1
for v in param_grid.values():
    grid_size *= len(v)

print(f"\n🔍 Hyperparameter search configuration:")
print(f"   Total combinations: {grid_size:,}")
print(f"   Sampling: 300 random combinations")
print(f"   CV folds: {cv_splits}")
print(f"   Total fits: 300 × {cv_splits} = {300 * cv_splits}")

# Test different scoring metrics
scoring_metrics = {
    'profit': make_scorer(profit_score, greater_is_better=True),
    'sharpe': make_scorer(sharpe_score, greater_is_better=True),
    'accuracy': make_scorer(accuracy_score_custom, greater_is_better=True),
}

results = {}

for metric_name, scorer in scoring_metrics.items():
    print("\n" + "=" * 80)
    print(f"TRAINING WITH {metric_name.upper()} OPTIMIZATION")
    print("=" * 80)

    print(f"\n[{datetime.now():%H:%M:%S}] Starting RandomizedSearchCV with {metric_name} scoring...")

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=300,
        scoring=scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        error_score=0,
        refit=True,
        random_state=42,
    )

    search.fit(X, y)

    print(f"[{datetime.now():%H:%M:%S}] Training completed!")

    print(f"\n✅ Best Parameters ({metric_name}):")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    print(f"\n🎯 Best CV Score ({metric_name}): {search.best_score_:.4f}")

    # Evaluate on test set
    test_size = int(len(X) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    y_pred = search.best_estimator_.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    profit = profit_score(y_test, y_pred)
    sharpe = sharpe_score(y_test, y_pred)

    print(f"\n📊 Test Set Performance ({metric_name}):")
    print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"   Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Profit:    {profit:.4f}")
    print(f"   Sharpe:    {sharpe:.4f}")

    results[metric_name] = {
        'model': search.best_estimator_,
        'params': search.best_params_,
        'cv_score': search.best_score_,
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'test_profit': profit,
        'test_sharpe': sharpe,
    }

    # Save model
    model_path = MODELS_DIR / f"market_crash_model_{metric_name}_optimized.pkl"
    payload = {"model": search.best_estimator_, "features": list(X.columns)}
    joblib.dump(payload, model_path)
    print(f"\n💾 Model saved → {model_path}")

# Summary comparison
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    metric: {
        'CV Score': res['cv_score'],
        'Test Accuracy': res['test_accuracy'],
        'Test Precision': res['test_precision'],
        'Test Recall': res['test_recall'],
        'Test F1': res['test_f1'],
        'Test Profit': res['test_profit'],
        'Test Sharpe': res['test_sharpe'],
    }
    for metric, res in results.items()
}).T

print("\n" + str(comparison_df))

comparison_df.to_csv('metric_comparison.csv')
print("\n✅ Comparison saved to metric_comparison.csv")

# Find best model by profit
best_metric = max(results.keys(), key=lambda k: results[k]['test_profit'])
print(f"\n🏆 Best model by PROFIT: {best_metric.upper()}")
print(f"   Profit score: {results[best_metric]['test_profit']:.4f}")
print(f"   Accuracy: {results[best_metric]['test_accuracy']*100:.2f}%")
print(f"   Precision: {results[best_metric]['test_precision']*100:.2f}%")

print("\n🎉 Profit-optimized training complete!")
print("\nNext steps:")
print("1. Review metric_comparison.csv for detailed comparison")
print("2. Use the profit-optimized model for better real-world returns")
print("3. Consider combining with threshold optimization for even better results")
