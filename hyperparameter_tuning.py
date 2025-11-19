#!/usr/bin/env python3
"""
Hyperparameter Tuning with Time-Series Cross-Validation

Uses TimeSeriesSplit for proper temporal CV and RandomizedSearchCV
for efficient hyperparameter optimization.

Usage:
    python3 hyperparameter_tuning.py

    # Or import and use in training:
    from hyperparameter_tuning import tune_models
    best_params = tune_models(X_train, y_train, weights)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

from config import MODELS_DIR

# =============================================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================================

XGB_PARAM_GRID = {
    'n_estimators': [100, 150, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 5, 10],
}

LGB_PARAM_GRID = {
    'n_estimators': [100, 150, 200, 300],
    'max_depth': [3, 4, 5, 6, 7, -1],  # -1 = no limit
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_samples': [5, 10, 20, 50],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 5, 10],
}

CAT_PARAM_GRID = {
    'iterations': [100, 150, 200, 300],
    'depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128, 255],
    'bagging_temperature': [0, 0.5, 1, 2],
}


def tune_xgboost(X_train, y_train, weights=None, n_iter=50, cv_splits=5, verbose=True):
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

    Args:
        X_train: Training features
        y_train: Training labels
        weights: Sample weights (optional)
        n_iter: Number of parameter combinations to try
        cv_splits: Number of CV folds
        verbose: Print progress

    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    if verbose:
        print("\n[XGBoost] Starting hyperparameter search...")

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Use F1 as primary metric (balances precision/recall)
    scorer = make_scorer(f1_score, zero_division=0)

    search = RandomizedSearchCV(
        model,
        XGB_PARAM_GRID,
        n_iter=n_iter,
        cv=tscv,
        scoring=scorer,
        random_state=42,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )

    if weights is not None:
        search.fit(X_train, y_train, sample_weight=weights)
    else:
        search.fit(X_train, y_train)

    if verbose:
        print(f"   Best F1: {search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_lightgbm(X_train, y_train, weights=None, n_iter=50, cv_splits=5, verbose=True):
    """
    Tune LightGBM hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
    """
    if verbose:
        print("\n[LightGBM] Starting hyperparameter search...")

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    model = lgb.LGBMClassifier(
        random_state=42,
        verbose=-1
    )

    scorer = make_scorer(f1_score, zero_division=0)

    search = RandomizedSearchCV(
        model,
        LGB_PARAM_GRID,
        n_iter=n_iter,
        cv=tscv,
        scoring=scorer,
        random_state=42,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )

    if weights is not None:
        search.fit(X_train, y_train, sample_weight=weights)
    else:
        search.fit(X_train, y_train)

    if verbose:
        print(f"   Best F1: {search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_catboost(X_train, y_train, weights=None, n_iter=30, cv_splits=5, verbose=True):
    """
    Tune CatBoost hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

    Note: CatBoost is slower, so we use fewer iterations by default.
    """
    if verbose:
        print("\n[CatBoost] Starting hyperparameter search...")

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    model = CatBoostClassifier(
        random_state=42,
        verbose=False
    )

    scorer = make_scorer(f1_score, zero_division=0)

    search = RandomizedSearchCV(
        model,
        CAT_PARAM_GRID,
        n_iter=n_iter,
        cv=tscv,
        scoring=scorer,
        random_state=42,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )

    if weights is not None:
        search.fit(X_train, y_train, sample_weight=weights)
    else:
        search.fit(X_train, y_train)

    if verbose:
        print(f"   Best F1: {search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_models(X_train, y_train, weights=None, n_iter=50, cv_splits=5, verbose=True):
    """
    Tune all three models and return best parameters.

    Args:
        X_train: Training features
        y_train: Training labels
        weights: Sample weights (optional)
        n_iter: Number of parameter combinations per model
        cv_splits: Number of CV folds
        verbose: Print progress

    Returns:
        Dictionary with best models and parameters
    """
    results = {}

    # Tune XGBoost
    xgb_model, xgb_params, xgb_score = tune_xgboost(
        X_train, y_train, weights, n_iter, cv_splits, verbose
    )
    results['xgboost'] = {
        'model': xgb_model,
        'params': xgb_params,
        'cv_score': xgb_score
    }

    # Tune LightGBM
    lgb_model, lgb_params, lgb_score = tune_lightgbm(
        X_train, y_train, weights, n_iter, cv_splits, verbose
    )
    results['lightgbm'] = {
        'model': lgb_model,
        'params': lgb_params,
        'cv_score': lgb_score
    }

    # Tune CatBoost (fewer iterations as it's slower)
    cat_model, cat_params, cat_score = tune_catboost(
        X_train, y_train, weights, max(20, n_iter // 2), cv_splits, verbose
    )
    results['catboost'] = {
        'model': cat_model,
        'params': cat_params,
        'cv_score': cat_score
    }

    return results


def save_best_params(results, filepath=None):
    """Save best parameters to JSON file."""
    import json

    if filepath is None:
        filepath = MODELS_DIR / "best_hyperparameters.json"

    # Extract just the parameters (not the models)
    params_only = {}
    for name, data in results.items():
        params_only[name] = {
            'params': data['params'],
            'cv_score': data['cv_score']
        }

    with open(filepath, 'w') as f:
        json.dump(params_only, f, indent=2, default=str)

    print(f"\n💾 Saved best parameters to {filepath}")
    return filepath


def load_best_params(filepath=None):
    """Load best parameters from JSON file."""
    import json

    if filepath is None:
        filepath = MODELS_DIR / "best_hyperparameters.json"

    if not filepath.exists():
        return None

    with open(filepath) as f:
        return json.load(f)


# =============================================================================
# MAIN - Standalone tuning
# =============================================================================

if __name__ == "__main__":
    from utils import add_features, add_forward_returns_and_labels
    from config import TRAIN_CFG

    print("=" * 80)
    print("HYPERPARAMETER TUNING WITH TIME-SERIES CROSS-VALIDATION")
    print("=" * 80)

    # Load and prepare data
    print("\n📥 Loading data...")
    DATA_DIR = Path("data")
    CACHE_DIR = Path("data_cache")

    # Load SPY data
    spy_df = pd.read_csv(DATA_DIR / "SPY.csv")
    spy_df['Date'] = pd.to_datetime(spy_df['Date'])
    spy_df.set_index('Date', inplace=True)

    # Add features
    spy_df, feature_cols = add_features(spy_df)

    # Add asset type
    spy_df['asset_type_stock'] = 1
    spy_df['asset_type_crypto'] = 0
    feature_cols = feature_cols + ['asset_type_stock', 'asset_type_crypto']

    # Add labels
    spy_df = add_forward_returns_and_labels(
        spy_df,
        price_col="Close",
        horizon=TRAIN_CFG["horizon"],
        pos_threshold=0.006,
        fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
        slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
    )

    # Load equity ETFs for more training data
    equity_assets = ['QQQ', 'IWM', 'DIA']
    all_dfs = [spy_df]

    for asset in equity_assets:
        filepath = CACHE_DIR / f"{asset}_1d.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df, _ = add_features(df)
            df['asset_type_stock'] = 1
            df['asset_type_crypto'] = 0
            df = add_forward_returns_and_labels(
                df, price_col="Close", horizon=TRAIN_CFG["horizon"],
                pos_threshold=0.006,
                fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
                slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
            )
            all_dfs.append(df)
            print(f"   ✓ Loaded {asset}: {len(df)} rows")

    # Combine data
    combined_df = pd.concat(all_dfs, axis=0)
    combined_df = combined_df.sort_index()

    # Prepare features
    feature_cols = [c for c in feature_cols if c in combined_df.columns]
    all_cols = feature_cols + ['y', 'fwd_ret_net']
    combined_df = combined_df[all_cols].dropna(subset=['y'])
    combined_df = combined_df.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"\n✅ Combined dataset: {len(combined_df):,} rows, {len(feature_cols)} features")

    # Train/test split
    test_size = int(len(combined_df) * 0.2)
    train_df = combined_df.iloc[:-test_size]
    test_df = combined_df.iloc[-test_size:]

    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values

    # Calculate sample weights
    fwd_rets = train_df['fwd_ret_net'].abs()
    weights = fwd_rets ** TRAIN_CFG.get("weight_power", 1.75)
    weights = weights.clip(
        TRAIN_CFG.get("min_weight", 0.5),
        TRAIN_CFG.get("max_weight", 5.0)
    ).values

    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    # Run hyperparameter tuning
    print("\n" + "=" * 80)
    print("TUNING HYPERPARAMETERS (this may take 5-15 minutes)")
    print("=" * 80)

    results = tune_models(
        X_train, y_train, weights,
        n_iter=30,  # Reduced for faster execution
        cv_splits=5,
        verbose=True
    )

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET RESULTS (with tuned parameters)")
    print("=" * 80)

    for name, data in results.items():
        model = data['model']
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)

        print(f"\n{name.upper()}:")
        print(f"   CV F1:   {data['cv_score']:.4f}")
        print(f"   Test Acc: {acc:.4f}")
        print(f"   Test F1:  {f1:.4f}")

    # Save results
    save_best_params(results)

    # Save tuned models
    print("\n💾 Saving tuned models...")
    for name, data in results.items():
        filepath = MODELS_DIR / f"{name}_multi_asset_tuned.pkl"
        joblib.dump(data['model'], filepath)
        print(f"   ✓ {filepath}")

    print("\n" + "=" * 80)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 80)
