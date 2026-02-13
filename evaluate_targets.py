#!/usr/bin/env python3
"""
Alternative Target Evaluation Script

Tests different target labeling strategies to find what works best:

1. BINARY_THRESHOLD: Standard +0.5% threshold
2. VOLATILITY_ADJUSTED: Threshold scaled by volatility
3. TRIPLE_BARRIER: Profit-taking and stop-loss barriers
4. MULTI_CLASS: 3-class (Crash/Normal/Spike)
5. REGRESSION_BINNED: Regression with quantile binning
6. RISK_ADJUSTED: Sharpe-ratio-like targets

Usage:
    python3 evaluate_targets.py [--quick] [--targets binary,volatility]

Outputs:
    - Per-target strategy AUC, F1, Precision, Recall
    - Recommendation for best target strategy
    - Saves results to outputs/target_evaluation.csv
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAIN_CFG


# Feature pruning configuration (from analyze_features.py Feb 2026)
FEATURES_TO_PRUNE = {
    'stoch_k_14_3',    # +0.0052 AUC when removed
    'ret_10d',         # +0.0022 AUC when removed
    'trend_strength',  # -15.70 importance score
    'ret_5d',          # -8.74 importance score
    'px_over_sma20',   # -6.86 importance score
    'rsi_price_div',   # -3.19 importance score
    'rsi_14',          # -0.51 importance, redundant with rsi_7
    'rsi_21',          # Redundant with rsi_7 (r=0.98)
    'sma_20',          # Redundant with sma_50 (r=0.999)
    'bb_up_20_2',      # Redundant with sma_50 (r=1.0)
    'trend_accel',     # Low importance (0.53)
}


class TargetStrategy(Enum):
    """Target labeling strategies"""
    BINARY_THRESHOLD = "binary_threshold"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TRIPLE_BARRIER = "triple_barrier"
    MULTI_CLASS = "multi_class"
    REGRESSION_BINNED = "regression_binned"
    RISK_ADJUSTED = "risk_adjusted"


@dataclass
class TargetResult:
    """Results for a single target evaluation"""
    strategy: str
    auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    n_samples: int
    n_classes: int
    class_distribution: Dict[int, int]
    train_time: float = 0.0
    description: str = ""


def add_binary_threshold_target(
    df: pd.DataFrame,
    threshold: float = 0.005,
    horizon: int = 1,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Standard binary threshold target.
    y=1 if forward_return >= threshold
    """
    d = df.copy()
    cost = (fee_bps + slippage_bps) * 1e-4

    d['fwd_ret_raw'] = d['Close'].pct_change(horizon).shift(-horizon)
    d['fwd_ret_net'] = d['fwd_ret_raw'] - cost
    d['y'] = (d['fwd_ret_net'] >= threshold).astype(int)

    return d


def add_volatility_adjusted_target(
    df: pd.DataFrame,
    base_threshold: float = 0.005,
    horizon: int = 1,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Volatility-adjusted threshold.
    Threshold scales with realized volatility - higher vol = higher threshold.
    """
    d = df.copy()
    cost = (fee_bps + slippage_bps) * 1e-4

    d['fwd_ret_raw'] = d['Close'].pct_change(horizon).shift(-horizon)
    d['fwd_ret_net'] = d['fwd_ret_raw'] - cost

    # Calculate rolling volatility with EXPANDING median to prevent lookahead
    d['_vol'] = d['Close'].pct_change().rolling(20).std()
    expanding_median_vol = d['_vol'].expanding(min_periods=60).median()

    valid = expanding_median_vol.notna() & (expanding_median_vol > 0)
    d['y'] = (d['fwd_ret_net'] >= base_threshold).astype(int)  # default

    if valid.any():
        vol_ratio = d.loc[valid, '_vol'] / expanding_median_vol[valid]
        adjusted_threshold = base_threshold * vol_ratio
        d.loc[valid, 'y'] = (d.loc[valid, 'fwd_ret_net'] >= adjusted_threshold).astype(int)

    d = d.drop(columns=['_vol'])
    return d


def add_triple_barrier_target(
    df: pd.DataFrame,
    pt_mult: float = 1.5,  # Profit-taking multiplier
    sl_mult: float = 1.0,  # Stop-loss multiplier
    t_max: int = 10,  # Maximum holding period
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Triple barrier labeling.
    y=1 if profit-taking barrier hit first
    y=0 if stop-loss barrier or time barrier hit first
    """
    d = df.copy()

    # Calculate ATR-like volatility
    returns = d['Close'].pct_change()
    vol = returns.rolling(vol_window).std()

    # Initialize labels
    d['y'] = 0

    for i in range(len(d) - t_max):
        entry_price = d['Close'].iloc[i]
        entry_vol = vol.iloc[i] if pd.notna(vol.iloc[i]) else 0.01

        pt_level = entry_price * (1 + pt_mult * entry_vol)
        sl_level = entry_price * (1 - sl_mult * entry_vol)

        # Look forward up to t_max bars
        for j in range(1, min(t_max + 1, len(d) - i)):
            high = d['High'].iloc[i + j]
            low = d['Low'].iloc[i + j]

            # Check barriers
            if high >= pt_level:
                d.loc[d.index[i], 'y'] = 1  # Profit-taking hit
                break
            elif low <= sl_level:
                d.loc[d.index[i], 'y'] = 0  # Stop-loss hit
                break
            # If j == t_max, time barrier hit (y stays 0)

    return d


def add_multi_class_target(
    df: pd.DataFrame,
    crash_threshold: float = -0.005,
    spike_threshold: float = 0.005,
    horizon: int = 1,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    3-class target: Crash (0), Normal (1), Spike (2)
    """
    d = df.copy()
    cost = (fee_bps + slippage_bps) * 1e-4

    d['fwd_ret_raw'] = d['Close'].pct_change(horizon).shift(-horizon)
    d['fwd_ret_net'] = d['fwd_ret_raw'] - cost

    d['y'] = np.select(
        [
            d['fwd_ret_net'] <= crash_threshold,
            d['fwd_ret_net'] >= spike_threshold,
        ],
        [0, 2],  # 0=Crash, 2=Spike
        default=1,  # 1=Normal
    )

    return d


def add_regression_binned_target(
    df: pd.DataFrame,
    n_bins: int = 3,
    horizon: int = 1,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Regression with expanding-window quantile binning.
    Uses only past data to determine bin boundaries, preventing lookahead bias.
    """
    d = df.copy()
    cost = (fee_bps + slippage_bps) * 1e-4

    d['fwd_ret_raw'] = d['Close'].pct_change(horizon).shift(-horizon)
    d['fwd_ret_net'] = d['fwd_ret_raw'] - cost

    # Expanding-window quantile binning to prevent lookahead bias
    # At each point t, bin boundaries are computed from data [0..t-1] only
    min_history = 252  # Need at least 1 year of history for stable quantiles
    y_vals = pd.Series(np.nan, index=d.index)
    fwd = d['fwd_ret_net']

    for i in range(min_history, len(d)):
        val = fwd.iloc[i]
        if pd.isna(val):
            continue
        past = fwd.iloc[:i].dropna()
        if len(past) < min_history:
            continue
        quantiles = past.quantile([k / n_bins for k in range(1, n_bins)]).values
        label = 0
        for q in quantiles:
            if val >= q:
                label += 1
        y_vals.iloc[i] = label

    d['y'] = y_vals

    return d


def add_risk_adjusted_target(
    df: pd.DataFrame,
    sharpe_threshold: float = 0.5,
    horizon: int = 1,
    lookback: int = 20,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Risk-adjusted (Sharpe-like) target.
    y=1 if expected return / volatility exceeds threshold
    """
    d = df.copy()
    cost = (fee_bps + slippage_bps) * 1e-4

    returns = d['Close'].pct_change()
    d['fwd_ret_raw'] = returns.shift(-horizon)
    d['fwd_ret_net'] = d['fwd_ret_raw'] - cost

    # Calculate rolling volatility
    vol = returns.rolling(lookback).std()

    # Calculate Sharpe-like ratio
    d['_sharpe'] = d['fwd_ret_net'] / (vol + 1e-9)

    d['y'] = (d['_sharpe'] >= sharpe_threshold).astype(int)
    d = d.drop(columns=['_sharpe'])

    return d


def evaluate_target_strategy(
    strategy: TargetStrategy,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    quick_mode: bool = False,
    prune_features: bool = True,
    **target_params,
) -> TargetResult:
    """
    Evaluate a single target strategy.

    Args:
        strategy: Target labeling strategy to evaluate
        df: DataFrame with OHLCV and features
        feature_cols: List of feature column names
        n_splits: Number of CV splits
        quick_mode: If True, use faster but less accurate evaluation
        prune_features: If True, exclude low-value features
        **target_params: Strategy-specific parameters
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import time

    print(f"\n[{strategy.value}] Evaluating target strategy...")

    start_time = time.time()

    # Apply target labeling strategy
    if strategy == TargetStrategy.BINARY_THRESHOLD:
        df_labeled = add_binary_threshold_target(df, **target_params)
        description = f"Binary threshold at {target_params.get('threshold', 0.005):.4f}"
    elif strategy == TargetStrategy.VOLATILITY_ADJUSTED:
        df_labeled = add_volatility_adjusted_target(df, **target_params)
        description = "Volatility-adjusted threshold"
    elif strategy == TargetStrategy.TRIPLE_BARRIER:
        df_labeled = add_triple_barrier_target(df, **target_params)
        description = f"Triple barrier (pt={target_params.get('pt_mult', 1.5)}x, sl={target_params.get('sl_mult', 1.0)}x)"
    elif strategy == TargetStrategy.MULTI_CLASS:
        df_labeled = add_multi_class_target(df, **target_params)
        description = "3-class (Crash/Normal/Spike)"
    elif strategy == TargetStrategy.REGRESSION_BINNED:
        df_labeled = add_regression_binned_target(df, **target_params)
        description = f"Regression binned ({target_params.get('n_bins', 3)} classes)"
    elif strategy == TargetStrategy.RISK_ADJUSTED:
        df_labeled = add_risk_adjusted_target(df, **target_params)
        description = f"Risk-adjusted (Sharpe threshold={target_params.get('sharpe_threshold', 0.5)})"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Clean data
    df_labeled = df_labeled.dropna(subset=['y'])

    # Apply feature pruning if enabled
    valid_features = [c for c in feature_cols if c in df_labeled.columns]
    if prune_features:
        original_count = len(valid_features)
        valid_features = [f for f in valid_features if f not in FEATURES_TO_PRUNE]
        pruned_count = original_count - len(valid_features)
        if pruned_count > 0:
            print(f"[{strategy.value}] Pruned {pruned_count} low-value features")

    X = df_labeled[valid_features].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df_labeled['y'].astype(int)

    # Impute NaN values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=valid_features, index=X.index)

    n_samples = len(X)
    n_classes = len(y.unique())
    class_distribution = y.value_counts().to_dict()

    print(f"[{strategy.value}] Samples: {n_samples}, Classes: {n_classes}")
    print(f"[{strategy.value}] Class distribution: {class_distribution}")

    # Model configuration - optimized based on analyze_features.py results
    if n_classes > 2:
        # Multi-class
        try:
            from xgboost import XGBClassifier
            model_class = XGBClassifier
            model_params = {
                'n_estimators': 200 if quick_mode else 300,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.08,
                'reg_lambda': 1.2,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                'objective': 'multi:softprob',
                'num_class': n_classes,
                'use_label_encoder': False,
            }
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model_class = GradientBoostingClassifier
            model_params = {
                'n_estimators': 100 if quick_mode else 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'random_state': 42,
            }
        avg_method = 'macro'
    else:
        # Binary - with class imbalance handling
        try:
            from xgboost import XGBClassifier
            model_class = XGBClassifier
            # Calculate scale_pos_weight for class imbalance
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            scale_weight = neg_count / max(pos_count, 1)

            model_params = {
                'n_estimators': 200 if quick_mode else 300,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.85,
                'min_child_weight': 8,
                'reg_alpha': 0.08,
                'reg_lambda': 1.2,
                'scale_pos_weight': scale_weight,  # Handle class imbalance
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                'use_label_encoder': False,
            }
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model_class = GradientBoostingClassifier
            model_params = {
                'n_estimators': 100 if quick_mode else 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'random_state': 42,
            }
        avg_method = 'binary'

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Handle probability for AUC
        if n_classes == 2:
            all_y_prob.extend(y_prob[:, 1].tolist())
        else:
            # For multi-class, use max probability
            all_y_prob.extend(y_prob.max(axis=1).tolist())

    # Calculate metrics
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob)
        else:
            # For multi-class, use OvR AUC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            if y_true_bin.shape[1] == 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = 0.5  # Simplified for multi-class
    except ValueError:
        auc = 0.5

    train_time = time.time() - start_time

    result = TargetResult(
        strategy=strategy.value,
        auc=auc,
        f1=f1_score(y_true, y_pred, average=avg_method, zero_division=0),
        precision=precision_score(y_true, y_pred, average=avg_method, zero_division=0),
        recall=recall_score(y_true, y_pred, average=avg_method, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        n_samples=n_samples,
        n_classes=n_classes,
        class_distribution=class_distribution,
        train_time=train_time,
        description=description,
    )

    print(f"[{strategy.value}] AUC: {result.auc:.4f}, F1: {result.f1:.4f}, "
          f"Precision: {result.precision:.4f}, Recall: {result.recall:.4f}")

    return result


def generate_recommendation(results: List[TargetResult]) -> Dict:
    """Generate recommendation based on results"""
    # Sort by AUC for binary strategies
    binary_results = [r for r in results if r.n_classes == 2]
    multiclass_results = [r for r in results if r.n_classes > 2]

    best_binary = max(binary_results, key=lambda x: x.auc) if binary_results else None
    best_multiclass = max(multiclass_results, key=lambda x: x.f1) if multiclass_results else None

    recommendation = {
        'best_binary': {
            'strategy': best_binary.strategy if best_binary else None,
            'auc': best_binary.auc if best_binary else None,
            'f1': best_binary.f1 if best_binary else None,
            'description': best_binary.description if best_binary else None,
        },
        'best_multiclass': {
            'strategy': best_multiclass.strategy if best_multiclass else None,
            'f1': best_multiclass.f1 if best_multiclass else None,
            'description': best_multiclass.description if best_multiclass else None,
        },
        'all_results': [
            {
                'strategy': r.strategy,
                'auc': r.auc,
                'f1': r.f1,
                'precision': r.precision,
                'recall': r.recall,
                'n_classes': r.n_classes,
                'description': r.description,
            }
            for r in sorted(results, key=lambda x: x.auc, reverse=True)
        ],
    }

    if best_binary and best_binary.auc > 0.55:
        recommendation['primary_recommendation'] = best_binary.strategy
        recommendation['reason'] = f"Best AUC {best_binary.auc:.4f} with {best_binary.description}"
    else:
        recommendation['primary_recommendation'] = 'binary_threshold'
        recommendation['reason'] = "Default to standard binary threshold"

    return recommendation


def print_summary(results: List[TargetResult], recommendation: Dict):
    """Print summary of evaluation results"""
    print("\n" + "=" * 70)
    print("TARGET STRATEGY EVALUATION SUMMARY")
    print("=" * 70)

    print("\nResults by AUC (descending):")
    print("-" * 70)
    print(f"{'Strategy':<25} {'AUC':<10} {'F1':<10} {'Prec':<10} {'Recall':<10} {'Classes':<10}")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda x: x.auc, reverse=True)
    for r in sorted_results:
        print(f"{r.strategy:<25} {r.auc:<10.4f} {r.f1:<10.4f} "
              f"{r.precision:<10.4f} {r.recall:<10.4f} {r.n_classes:<10}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    print(f"\nPrimary Recommendation: {recommendation['primary_recommendation']}")
    print(f"Reason: {recommendation['reason']}")

    if recommendation['best_binary']['strategy']:
        print(f"\nBest Binary Strategy: {recommendation['best_binary']['strategy']}")
        print(f"  AUC: {recommendation['best_binary']['auc']:.4f}")
        print(f"  Description: {recommendation['best_binary']['description']}")

    if recommendation['best_multiclass']['strategy']:
        print(f"\nBest Multi-class Strategy: {recommendation['best_multiclass']['strategy']}")
        print(f"  F1: {recommendation['best_multiclass']['f1']:.4f}")
        print(f"  Description: {recommendation['best_multiclass']['description']}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate target labeling strategies')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick evaluation mode')
    parser.add_argument('--targets', '-t', type=str,
                        default='binary,volatility,triple,multiclass,regression,risk',
                        help='Comma-separated list of targets to evaluate')
    parser.add_argument('--splits', '-s', type=int, default=5,
                        help='Number of CV splits')
    parser.add_argument('--no-prune', action='store_true',
                        help='Disable feature pruning (use all features)')

    args = parser.parse_args()
    prune_features = not args.no_prune

    # Map short names to strategies
    strategy_map = {
        'binary': TargetStrategy.BINARY_THRESHOLD,
        'volatility': TargetStrategy.VOLATILITY_ADJUSTED,
        'triple': TargetStrategy.TRIPLE_BARRIER,
        'multiclass': TargetStrategy.MULTI_CLASS,
        'regression': TargetStrategy.REGRESSION_BINNED,
        'risk': TargetStrategy.RISK_ADJUSTED,
    }

    strategies = []
    for name in args.targets.split(','):
        name = name.strip().lower()
        if name in strategy_map:
            strategies.append(strategy_map[name])

    print("=" * 70)
    print("TARGET STRATEGY EVALUATION")
    print("=" * 70)
    print(f"\nStrategies to evaluate: {[s.value for s in strategies]}")
    print(f"CV splits: {args.splits}")
    print(f"Feature pruning: {'Enabled' if prune_features else 'Disabled'}")
    if prune_features:
        print(f"Pruned features: {len(FEATURES_TO_PRUNE)} low-value features excluded")
    print("=" * 70)

    # Load data using build_feature_table (matches analyze_features.py namespace
    # so that FEATURES_TO_PRUNE names actually match the generated columns)
    from utils import load_SPY_data
    from build_feature_table import build_features

    print("\nLoading data...")
    raw = load_SPY_data()
    df = build_features(raw, prune_features=False)

    # Feature columns are everything except 'close'
    feature_cols = [c for c in df.columns if c != 'close']

    # Carry OHLC through for target labeling strategies that need them
    df['Close'] = raw['Close'].reindex(df.index)
    df['High'] = raw['High'].reindex(df.index)
    df['Low'] = raw['Low'].reindex(df.index)

    results = []

    for strategy in strategies:
        try:
            result = evaluate_target_strategy(
                strategy=strategy,
                df=df.copy(),
                feature_cols=feature_cols,
                n_splits=args.splits,
                quick_mode=args.quick,
                prune_features=prune_features,
            )
            results.append(result)
        except Exception as e:
            print(f"[{strategy.value}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nERROR: No strategies could be evaluated.")
        return 1

    # Generate recommendation
    recommendation = generate_recommendation(results)

    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame([
        {
            'strategy': r.strategy,
            'auc': r.auc,
            'f1': r.f1,
            'precision': r.precision,
            'recall': r.recall,
            'accuracy': r.accuracy,
            'n_samples': r.n_samples,
            'n_classes': r.n_classes,
            'description': r.description,
        }
        for r in results
    ])
    results_df.to_csv(output_dir / 'target_evaluation_latest.csv', index=False)

    with open(output_dir / 'target_recommendation.json', 'w') as f:
        json.dump(recommendation, f, indent=2, default=float)

    # Print summary
    print_summary(results, recommendation)

    return 0


if __name__ == '__main__':
    sys.exit(main())
