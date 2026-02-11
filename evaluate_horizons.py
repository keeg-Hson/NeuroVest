#!/usr/bin/env python3
"""
Horizon Evaluation Script

Tests multiple prediction horizons (1d, 2d, 3d, 5d, 10d, 21d) to find
which timeframe produces the highest AUC scores.

Usage:
    python evaluate_horizons.py [--quick] [--horizons 1,3,5,10]

Outputs:
    - Per-horizon AUC, F1, Precision, Recall scores
    - Recommendation for best horizon or weighted ensemble
    - Saves results to outputs/horizon_evaluation.csv
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAIN_CFG, MODELS_DIR, LOGS_DIR


@dataclass
class HorizonResult:
    """Results for a single horizon evaluation"""
    horizon: int
    auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    n_samples: int
    n_positive: int
    positive_rate: float
    train_time: float = 0.0


def load_data_for_horizon(horizon: int, pos_threshold: float = 0.005) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and prepare data for a specific horizon.

    Args:
        horizon: Forward return horizon in days
        pos_threshold: Minimum return to label as positive

    Returns:
        Tuple of (DataFrame with features and labels, feature column names)
    """
    from utils import add_features, add_forward_returns_and_labels, finalize_features, load_SPY_data

    print(f"\n[Horizon {horizon}d] Loading and preparing data...")

    # Load raw data
    df = load_SPY_data()

    # Add features
    df, feature_cols = add_features(df)

    # Add forward returns and labels for this horizon
    df = add_forward_returns_and_labels(
        df,
        price_col="Close",
        horizon=horizon,
        fee_bps=TRAIN_CFG.get('fee_bps', 1.5),
        slippage_bps=TRAIN_CFG.get('slippage_bps', 2.0),
        long_only=True,
        pos_threshold=pos_threshold,
        volatility_adjusted=True,
    )

    # Finalize features
    df = finalize_features(df, feature_cols)

    # Restore Close column for validation
    raw = load_SPY_data()
    df['Close'] = raw['Close'].reindex(df.index)

    # Drop rows with missing labels
    df = df.dropna(subset=['y'])

    return df, feature_cols


def evaluate_horizon(
    horizon: int,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    quick_mode: bool = False,
) -> HorizonResult:
    """
    Evaluate a single horizon using time-series cross-validation.

    Args:
        horizon: Forward return horizon in days
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        n_splits: Number of CV splits
        quick_mode: If True, use faster but less accurate evaluation

    Returns:
        HorizonResult with evaluation metrics
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import time

    start_time = time.time()

    # Prepare data
    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df['y'].astype(int)

    # Impute NaN values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)

    n_samples = len(X)
    n_positive = int(y.sum())
    positive_rate = n_positive / n_samples

    print(f"[Horizon {horizon}d] Samples: {n_samples}, Positive: {n_positive} ({positive_rate:.1%})")

    # Model configuration
    if quick_mode:
        from sklearn.ensemble import GradientBoostingClassifier
        model_class = GradientBoostingClassifier
        model_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
        }
    else:
        try:
            from xgboost import XGBClassifier
            model_class = XGBClassifier
            model_params = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,
                'reg_alpha': 0.05,
                'reg_lambda': 1.5,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                'tree_method': 'hist',
                'use_label_encoder': False,
            }
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model_class = GradientBoostingClassifier
            model_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'random_state': 42,
            }

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
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

    # Calculate metrics
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5  # Default for edge cases

    train_time = time.time() - start_time

    result = HorizonResult(
        horizon=horizon,
        auc=auc,
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        n_samples=n_samples,
        n_positive=n_positive,
        positive_rate=positive_rate,
        train_time=train_time,
    )

    print(f"[Horizon {horizon}d] AUC: {result.auc:.4f}, F1: {result.f1:.4f}, "
          f"Precision: {result.precision:.4f}, Recall: {result.recall:.4f}")

    return result


def evaluate_weighted_ensemble(
    results: List[HorizonResult],
    weight_by: str = 'auc'
) -> Dict:
    """
    Calculate weighted ensemble metrics based on horizon performance.

    Args:
        results: List of HorizonResult objects
        weight_by: Metric to weight by ('auc', 'f1', 'precision')

    Returns:
        Dict with ensemble weights and expected performance
    """
    # Extract weights based on specified metric
    if weight_by == 'auc':
        scores = [r.auc for r in results]
    elif weight_by == 'f1':
        scores = [r.f1 for r in results]
    elif weight_by == 'precision':
        scores = [r.precision for r in results]
    else:
        scores = [r.auc for r in results]

    # Normalize to get weights
    total = sum(scores)
    if total > 0:
        weights = [s / total for s in scores]
    else:
        weights = [1.0 / len(results)] * len(results)

    # Calculate weighted average of metrics
    weighted_auc = sum(r.auc * w for r, w in zip(results, weights))
    weighted_f1 = sum(r.f1 * w for r, w in zip(results, weights))
    weighted_precision = sum(r.precision * w for r, w in zip(results, weights))
    weighted_recall = sum(r.recall * w for r, w in zip(results, weights))

    return {
        'horizons': [r.horizon for r in results],
        'weights': weights,
        'weighted_auc': weighted_auc,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weight_by': weight_by,
    }


def generate_recommendation(results: List[HorizonResult]) -> Dict:
    """
    Generate recommendation for best horizon strategy.

    Args:
        results: List of HorizonResult objects

    Returns:
        Dict with recommendation details
    """
    # Sort by AUC
    sorted_results = sorted(results, key=lambda x: x.auc, reverse=True)
    best = sorted_results[0]

    # Check if ensemble would be beneficial
    auc_variance = np.var([r.auc for r in results])
    top_3_avg_auc = np.mean([r.auc for r in sorted_results[:min(3, len(sorted_results))]])

    # If top horizons are close in performance, recommend ensemble
    if len(results) >= 3:
        top_3 = sorted_results[:3]
        auc_range = top_3[0].auc - top_3[2].auc

        if auc_range < 0.02:  # Very close performance
            strategy = 'weighted_ensemble'
            ensemble = evaluate_weighted_ensemble(top_3)
            recommendation = {
                'strategy': strategy,
                'best_single_horizon': best.horizon,
                'best_single_auc': best.auc,
                'ensemble_horizons': ensemble['horizons'],
                'ensemble_weights': ensemble['weights'],
                'ensemble_expected_auc': ensemble['weighted_auc'],
                'reason': f"Top 3 horizons have similar AUC (range={auc_range:.4f}). "
                          f"Ensemble may provide more robust predictions."
            }
        else:
            strategy = 'single_best'
            recommendation = {
                'strategy': strategy,
                'best_horizon': best.horizon,
                'best_auc': best.auc,
                'best_f1': best.f1,
                'reason': f"Horizon {best.horizon}d clearly outperforms others "
                          f"(AUC range={auc_range:.4f}). Use single horizon."
            }
    else:
        strategy = 'single_best'
        recommendation = {
            'strategy': strategy,
            'best_horizon': best.horizon,
            'best_auc': best.auc,
            'best_f1': best.f1,
            'reason': f"Horizon {best.horizon}d has best AUC."
        }

    # Add all results for reference
    recommendation['all_results'] = [
        {
            'horizon': r.horizon,
            'auc': r.auc,
            'f1': r.f1,
            'precision': r.precision,
            'recall': r.recall,
            'positive_rate': r.positive_rate,
        }
        for r in sorted_results
    ]

    return recommendation


def save_results(
    results: List[HorizonResult],
    recommendation: Dict,
    output_dir: Path = None,
):
    """Save evaluation results to files"""
    if output_dir is None:
        output_dir = Path('outputs')

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results as CSV
    results_df = pd.DataFrame([
        {
            'horizon': r.horizon,
            'auc': r.auc,
            'f1': r.f1,
            'precision': r.precision,
            'recall': r.recall,
            'accuracy': r.accuracy,
            'n_samples': r.n_samples,
            'n_positive': r.n_positive,
            'positive_rate': r.positive_rate,
            'train_time': r.train_time,
        }
        for r in results
    ])
    results_df = results_df.sort_values('auc', ascending=False)
    results_df.to_csv(output_dir / f'horizon_evaluation_{timestamp}.csv', index=False)

    # Save as latest for easy access
    results_df.to_csv(output_dir / 'horizon_evaluation_latest.csv', index=False)

    # Save recommendation as JSON
    with open(output_dir / 'horizon_recommendation.json', 'w') as f:
        json.dump(recommendation, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}/")


def print_summary(results: List[HorizonResult], recommendation: Dict):
    """Print summary of evaluation results"""
    print("\n" + "=" * 70)
    print("HORIZON EVALUATION SUMMARY")
    print("=" * 70)

    print("\nResults by AUC (descending):")
    print("-" * 70)
    print(f"{'Horizon':<10} {'AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Pos%':<10}")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda x: x.auc, reverse=True)
    for r in sorted_results:
        print(f"{r.horizon:>3}d       {r.auc:<10.4f} {r.f1:<10.4f} "
              f"{r.precision:<10.4f} {r.recall:<10.4f} {r.positive_rate:<10.1%}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if recommendation['strategy'] == 'weighted_ensemble':
        print(f"\nStrategy: WEIGHTED ENSEMBLE")
        print(f"Horizons: {recommendation['ensemble_horizons']}")
        print(f"Weights: {[f'{w:.3f}' for w in recommendation['ensemble_weights']]}")
        print(f"Expected AUC: {recommendation['ensemble_expected_auc']:.4f}")
        print(f"\nReason: {recommendation['reason']}")
    else:
        print(f"\nStrategy: SINGLE BEST HORIZON")
        print(f"Best Horizon: {recommendation['best_horizon']}d")
        print(f"AUC: {recommendation['best_auc']:.4f}")
        print(f"F1: {recommendation['best_f1']:.4f}")
        print(f"\nReason: {recommendation['reason']}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate prediction horizons')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick evaluation mode (faster but less accurate)')
    parser.add_argument('--horizons', '-H', type=str, default='1,2,3,5,10,21',
                        help='Comma-separated list of horizons to evaluate')
    parser.add_argument('--threshold', '-t', type=float, default=0.005,
                        help='Positive threshold for labeling (default: 0.005 = 0.5%%)')
    parser.add_argument('--splits', '-s', type=int, default=5,
                        help='Number of CV splits (default: 5)')

    args = parser.parse_args()

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(',')]

    print("=" * 70)
    print("HORIZON EVALUATION")
    print("=" * 70)
    print(f"\nHorizons to evaluate: {horizons}")
    print(f"Positive threshold: {args.threshold:.4f} ({args.threshold*100:.2f}%)")
    print(f"CV splits: {args.splits}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print("=" * 70)

    results = []

    for horizon in horizons:
        try:
            # Load data for this horizon
            df, feature_cols = load_data_for_horizon(horizon, pos_threshold=args.threshold)

            # Evaluate
            result = evaluate_horizon(
                horizon=horizon,
                df=df,
                feature_cols=feature_cols,
                n_splits=args.splits,
                quick_mode=args.quick,
            )
            results.append(result)

        except Exception as e:
            print(f"[Horizon {horizon}d] ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nERROR: No horizons could be evaluated successfully.")
        return 1

    # Generate recommendation
    recommendation = generate_recommendation(results)

    # Save results
    save_results(results, recommendation)

    # Print summary
    print_summary(results, recommendation)

    return 0


if __name__ == '__main__':
    sys.exit(main())
