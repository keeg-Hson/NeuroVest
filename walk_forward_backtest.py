#!/usr/bin/env python3
"""
Walk-Forward Backtest for NeuroVest

Implements true walk-forward validation that simulates real trading:
- Train on historical data up to time T
- Predict on forward period T to T+step
- Roll forward and repeat
- Track cumulative performance over time

This is more rigorous than simple train/test splits because it:
1. Prevents any look-ahead bias
2. Tests model robustness to regime changes
3. Shows how the model would have performed in production

Usage:
    python3 walk_forward_backtest.py --years 5
    python3 walk_forward_backtest.py --years 3 --step-days 21 --retrain-freq 63
    python3 walk_forward_backtest.py --quick  # Fast mode for testing

Outputs:
    - outputs/walk_forward_results.csv       # Per-period predictions
    - outputs/walk_forward_summary.json      # Aggregate metrics
    - outputs/walk_forward_equity_curve.png  # Cumulative returns chart
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAIN_CFG, OUTPUT_DIR


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest"""
    # Time parameters
    total_years: float = 5.0          # Total backtest period
    min_train_years: float = 2.0      # Minimum training history required
    step_days: int = 21               # Forward step size (21 = ~1 month)
    retrain_freq: int = 63            # Retrain every N days (63 = ~quarterly)

    # Model parameters
    horizon: int = 1                  # Prediction horizon in days
    pos_threshold: float = 0.005      # Positive return threshold
    n_estimators: int = 300           # XGBoost trees
    max_depth: int = 6                # Tree depth
    learning_rate: float = 0.03       # Learning rate

    # Trading parameters
    prob_threshold: float = 0.45      # Minimum probability to trade
    fee_bps: float = 1.5              # Trading fee in bps
    slippage_bps: float = 2.0         # Slippage in bps

    # Feature pruning
    prune_features: bool = True

    # Mode
    quick_mode: bool = False          # Faster but less accurate


@dataclass
class PeriodResult:
    """Results for a single walk-forward period"""
    period_start: str
    period_end: str
    train_start: str
    train_end: str
    n_train_samples: int
    n_test_samples: int
    n_predictions: int
    n_trades: int
    accuracy: float
    precision: float
    recall: float
    auc: float
    period_return: float
    period_return_gross: float
    benchmark_return: float
    excess_return: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_period: float


@dataclass
class WalkForwardResults:
    """Aggregated walk-forward backtest results"""
    config: WalkForwardConfig
    periods: List[PeriodResult] = field(default_factory=list)
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Aggregate metrics
    total_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_auc: float = 0.0
    avg_precision: float = 0.0
    n_periods: int = 0
    n_trades: int = 0


def load_data(prune_features: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Load SPY data and build features"""
    from utils import load_SPY_data
    from build_feature_table import build_features

    print("Loading SPY data...")
    df = load_SPY_data()

    print("Building features...")
    feat_df = build_features(df, prune_features=prune_features)

    # Feature columns (everything except 'close')
    feature_cols = [c for c in feat_df.columns if c != 'close']

    # Add close for return calculations
    if 'close' not in feat_df.columns:
        feat_df['close'] = df['Close'].reindex(feat_df.index)

    print(f"Loaded {len(feat_df)} samples with {len(feature_cols)} features")
    return feat_df, feature_cols


def prepare_labels(
    df: pd.DataFrame,
    horizon: int = 1,
    pos_threshold: float = 0.005,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """Add forward returns and binary labels"""
    cost = (fee_bps + slippage_bps) * 1e-4

    df = df.copy()
    df['fwd_ret_raw'] = (df['close'].shift(-horizon) / df['close']) - 1.0
    df['fwd_ret_net'] = df['fwd_ret_raw'] - cost
    df['y'] = (df['fwd_ret_net'] >= pos_threshold).astype(int)

    # Drop rows without labels
    df = df.dropna(subset=['y'])

    return df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WalkForwardConfig,
) -> object:
    """Train XGBoost model on training data"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_train.replace([np.inf, -np.inf], np.nan))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Calculate class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=config.n_estimators if not config.quick_mode else 100,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=0.8,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_alpha=0.08,
            reg_lambda=1.2,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method='hist',
            use_label_encoder=False,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=config.n_estimators if not config.quick_mode else 100,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            random_state=42,
        )

    model.fit(X_scaled, y_train)

    # Store preprocessing objects
    model._imputer = imputer
    model._scaler = scaler

    return model


def predict_proba(model, X_test: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from model"""
    X_imputed = model._imputer.transform(X_test.replace([np.inf, -np.inf], np.nan))
    X_scaled = model._scaler.transform(X_imputed)
    return model.predict_proba(X_scaled)[:, 1]


def evaluate_period(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fwd_returns: np.ndarray,
    prob_threshold: float,
    cost: float,
) -> Dict:
    """Evaluate model performance for a period"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, roc_auc_score
    )

    y_pred = (y_prob >= prob_threshold).astype(int)

    # Classification metrics
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Trading metrics
    trades = y_pred == 1
    n_trades = trades.sum()

    if n_trades > 0:
        raw_trade_returns = fwd_returns[trades]
        # Filter out NaN trade returns
        valid_trade_mask = ~np.isnan(raw_trade_returns)
        trade_returns = raw_trade_returns[valid_trade_mask] - cost
        n_valid_trades = len(trade_returns)

        if n_valid_trades > 0:
            period_return = trade_returns.sum()
            wins = trade_returns > 0
            win_rate = wins.mean()
            avg_win = trade_returns[wins].mean() if wins.any() else 0
            avg_loss = trade_returns[~wins].mean() if (~wins).any() else 0

            # Period Sharpe (annualized approximation)
            if len(trade_returns) > 1 and trade_returns.std() > 0:
                sharpe_period = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252)
            else:
                sharpe_period = 0
        else:
            period_return = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            sharpe_period = 0
    else:
        period_return = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        sharpe_period = 0

    # Gross return (before costs)
    if n_trades > 0:
        valid_gross = fwd_returns[trades]
        valid_gross = valid_gross[~np.isnan(valid_gross)]
        period_return_gross = valid_gross.sum() if len(valid_gross) > 0 else 0
    else:
        period_return_gross = 0

    # Benchmark (buy and hold) - handle NaN values
    valid_returns = fwd_returns[~np.isnan(fwd_returns)]
    benchmark_return = valid_returns.sum() if len(valid_returns) > 0 else 0.0

    return {
        'n_trades': n_trades,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'period_return': period_return,
        'period_return_gross': period_return_gross,
        'benchmark_return': benchmark_return,
        'excess_return': period_return - benchmark_return if not np.isnan(benchmark_return) else 0,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_period': sharpe_period,
    }


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: WalkForwardConfig,
) -> WalkForwardResults:
    """Run walk-forward backtest"""

    # Prepare labels
    df = prepare_labels(
        df,
        horizon=config.horizon,
        pos_threshold=config.pos_threshold,
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
    )

    # Filter to valid features
    valid_features = [f for f in feature_cols if f in df.columns]

    # Calculate period boundaries
    total_days = int(config.total_years * 252)
    min_train_days = int(config.min_train_years * 252)

    # Ensure we have enough data
    if len(df) < total_days:
        total_days = len(df)
        print(f"Adjusted total days to available data: {total_days}")

    # Start from the end and work backwards
    end_idx = len(df)
    start_idx = max(0, end_idx - total_days)

    # First test period starts after minimum training
    test_start_idx = start_idx + min_train_days

    results = WalkForwardResults(config=config)
    all_predictions = []

    current_model = None
    last_train_idx = 0
    period_num = 0

    cost = (config.fee_bps + config.slippage_bps) * 1e-4

    test_days = end_idx - test_start_idx
    test_years = test_days / 252

    print(f"\nStarting walk-forward backtest...")
    print(f"  Data range: {df.index[start_idx].date()} to {df.index[end_idx-1].date()}")
    print(f"  Initial training: {min_train_days} days ({config.min_train_years} years)")
    print(f"  Test period: ~{test_days} days ({test_years:.1f} years)")
    print(f"  Step size: {config.step_days} days")
    print(f"  Retrain frequency: {config.retrain_freq} days")
    print()

    test_idx = test_start_idx

    while test_idx < end_idx:
        period_num += 1

        # Define test period
        test_end_idx = min(test_idx + config.step_days, end_idx)

        # Check if we need to retrain
        days_since_train = test_idx - last_train_idx
        need_retrain = (current_model is None) or (days_since_train >= config.retrain_freq)

        if need_retrain:
            # Training data: everything before test period
            train_df = df.iloc[start_idx:test_idx]

            X_train = train_df[valid_features]
            y_train = train_df['y']

            print(f"Period {period_num}: Training on {len(train_df)} samples "
                  f"({train_df.index[0].date()} to {train_df.index[-1].date()})")

            current_model = train_model(X_train, y_train, config)
            last_train_idx = test_idx

        # Test data for this period
        test_df = df.iloc[test_idx:test_end_idx]

        if len(test_df) == 0:
            test_idx = test_end_idx
            continue

        X_test = test_df[valid_features]
        y_test = test_df['y'].values
        fwd_returns = test_df['fwd_ret_raw'].values

        # Skip periods with all NaN forward returns (end of data)
        valid_fwd = ~np.isnan(fwd_returns)
        if not valid_fwd.any():
            print(f"  Skipping period (no valid forward returns)")
            test_idx = test_end_idx
            continue

        # Get predictions
        y_prob = predict_proba(current_model, X_test)

        # Evaluate period
        metrics = evaluate_period(
            y_test, y_prob, fwd_returns,
            config.prob_threshold, cost
        )

        # Store predictions
        for i, idx in enumerate(test_df.index):
            all_predictions.append({
                'date': idx,
                'period': period_num,
                'y_true': y_test[i],
                'y_prob': y_prob[i],
                'y_pred': int(y_prob[i] >= config.prob_threshold),
                'fwd_ret': fwd_returns[i],
                'close': test_df['close'].iloc[i],
            })

        # Store period result
        period_result = PeriodResult(
            period_start=str(test_df.index[0].date()),
            period_end=str(test_df.index[-1].date()),
            train_start=str(df.index[start_idx].date()),
            train_end=str(df.index[test_idx-1].date()),
            n_train_samples=test_idx - start_idx,
            n_test_samples=len(test_df),
            n_predictions=len(y_prob),
            **metrics,
        )
        results.periods.append(period_result)

        ret_str = f"{metrics['period_return']*100:+.2f}%" if not np.isnan(metrics['period_return']) else "N/A"
        bench_str = f"{metrics['benchmark_return']*100:+.2f}%" if not np.isnan(metrics['benchmark_return']) else "N/A"
        print(f"  Test: {period_result.period_start} to {period_result.period_end} | "
              f"AUC: {metrics['auc']:.3f} | Trades: {metrics['n_trades']} | "
              f"Return: {ret_str} | Benchmark: {bench_str}")

        # Move to next period
        test_idx = test_end_idx

    # Store all predictions
    results.predictions_df = pd.DataFrame(all_predictions)
    if len(results.predictions_df) > 0:
        results.predictions_df.set_index('date', inplace=True)

    # Calculate aggregate metrics
    if results.periods:
        results.n_periods = len(results.periods)
        results.n_trades = sum(p.n_trades for p in results.periods)
        results.total_return = sum(p.period_return for p in results.periods if not np.isnan(p.period_return))
        benchmark_returns = [p.benchmark_return for p in results.periods if not np.isnan(p.benchmark_return)]
        results.benchmark_return = sum(benchmark_returns) if benchmark_returns else 0.0
        results.excess_return = results.total_return - results.benchmark_return
        results.avg_auc = np.mean([p.auc for p in results.periods if not np.isnan(p.auc)])
        precision_values = [p.precision for p in results.periods if p.n_trades > 0 and not np.isnan(p.precision)]
        results.avg_precision = np.mean(precision_values) if precision_values else 0.0

        # Win rate across all trades
        if len(results.predictions_df) > 0:
            trades_df = results.predictions_df[results.predictions_df['y_pred'] == 1].copy()
            if len(trades_df) > 0:
                trade_returns = trades_df['fwd_ret'].dropna() - cost
                if len(trade_returns) > 0:
                    results.win_rate = (trade_returns > 0).mean()

        # Sharpe ratio on period returns (exclude NaN)
        period_returns = [p.period_return for p in results.periods if not np.isnan(p.period_return)]
        if len(period_returns) > 1 and np.std(period_returns) > 0:
            # Annualize: assume step_days per period
            periods_per_year = 252 / config.step_days
            results.sharpe_ratio = (
                np.mean(period_returns) / np.std(period_returns)
            ) * np.sqrt(periods_per_year)

        # Max drawdown
        if len(results.predictions_df) > 0:
            trades_df = results.predictions_df[results.predictions_df['y_pred'] == 1].copy()
            trades_df = trades_df.dropna(subset=['fwd_ret'])
            if len(trades_df) > 0:
                trades_df['trade_ret'] = trades_df['fwd_ret'] - cost
                trades_df['cum_ret'] = (1 + trades_df['trade_ret']).cumprod()
                trades_df['rolling_max'] = trades_df['cum_ret'].cummax()
                trades_df['drawdown'] = trades_df['cum_ret'] / trades_df['rolling_max'] - 1
                results.max_drawdown = trades_df['drawdown'].min()

    return results


def save_results(results: WalkForwardResults, output_dir: Path = None):
    """Save walk-forward results to files"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save predictions
    if len(results.predictions_df) > 0:
        results.predictions_df.to_csv(output_dir / f'walk_forward_predictions_{timestamp}.csv')
        results.predictions_df.to_csv(output_dir / 'walk_forward_predictions_latest.csv')

    # Save period-by-period results
    periods_df = pd.DataFrame([
        {
            'period_start': p.period_start,
            'period_end': p.period_end,
            'n_trades': p.n_trades,
            'auc': p.auc,
            'precision': p.precision,
            'recall': p.recall,
            'period_return': p.period_return,
            'benchmark_return': p.benchmark_return,
            'excess_return': p.excess_return,
            'win_rate': p.win_rate,
            'sharpe_period': p.sharpe_period,
        }
        for p in results.periods
    ])
    periods_df.to_csv(output_dir / f'walk_forward_periods_{timestamp}.csv', index=False)
    periods_df.to_csv(output_dir / 'walk_forward_periods_latest.csv', index=False)

    # Save summary
    summary = {
        'timestamp': timestamp,
        'config': {
            'total_years': results.config.total_years,
            'min_train_years': results.config.min_train_years,
            'step_days': results.config.step_days,
            'retrain_freq': results.config.retrain_freq,
            'horizon': results.config.horizon,
            'pos_threshold': results.config.pos_threshold,
            'prob_threshold': results.config.prob_threshold,
            'prune_features': results.config.prune_features,
        },
        'results': {
            'n_periods': results.n_periods,
            'n_trades': results.n_trades,
            'total_return': results.total_return,
            'total_return_pct': results.total_return * 100,
            'benchmark_return': results.benchmark_return,
            'benchmark_return_pct': results.benchmark_return * 100,
            'excess_return': results.excess_return,
            'excess_return_pct': results.excess_return * 100,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'max_drawdown_pct': results.max_drawdown * 100 if results.max_drawdown else 0,
            'win_rate': results.win_rate,
            'avg_auc': results.avg_auc,
            'avg_precision': results.avg_precision,
        },
    }

    with open(output_dir / 'walk_forward_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}/")


def plot_results(results: WalkForwardResults, output_dir: Path = None):
    """Generate equity curve and performance charts"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    if len(results.predictions_df) == 0:
        print("No predictions to plot")
        return

    cost = (results.config.fee_bps + results.config.slippage_bps) * 1e-4

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. Cumulative returns comparison
    ax1 = axes[0]

    df = results.predictions_df.copy()
    df['trade_ret'] = np.where(
        df['y_pred'] == 1,
        df['fwd_ret'] - cost,
        0
    )
    df['cum_strategy'] = (1 + df['trade_ret']).cumprod()
    df['cum_benchmark'] = (1 + df['fwd_ret']).cumprod()

    ax1.plot(df.index, df['cum_strategy'], label='Walk-Forward Strategy', linewidth=2)
    ax1.plot(df.index, df['cum_benchmark'], label='Buy & Hold Benchmark', linewidth=2, alpha=0.7)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Walk-Forward Backtest: Equity Curves')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Period-by-period returns
    ax2 = axes[1]

    periods_df = pd.DataFrame([
        {'date': p.period_end, 'strategy': p.period_return, 'benchmark': p.benchmark_return}
        for p in results.periods
    ])
    periods_df['date'] = pd.to_datetime(periods_df['date'])

    x = range(len(periods_df))
    width = 0.35
    ax2.bar([i - width/2 for i in x], periods_df['strategy'] * 100, width, label='Strategy', alpha=0.8)
    ax2.bar([i + width/2 for i in x], periods_df['benchmark'] * 100, width, label='Benchmark', alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Period Return (%)')
    ax2.set_title('Period-by-Period Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Rolling AUC
    ax3 = axes[2]

    aucs = [p.auc for p in results.periods]
    dates = [p.period_end for p in results.periods]

    ax3.plot(range(len(aucs)), aucs, marker='o', linewidth=2, markersize=4)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)', alpha=0.7)
    ax3.axhline(y=np.mean(aucs), color='green', linestyle='--',
                label=f'Mean ({np.mean(aucs):.3f})', alpha=0.7)
    ax3.set_ylabel('AUC')
    ax3.set_xlabel('Period')
    ax3.set_title('Rolling Model Performance (AUC)')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.4, 0.7)

    plt.tight_layout()

    chart_path = output_dir / 'walk_forward_equity_curve.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chart saved to {chart_path}")


def print_summary(results: WalkForwardResults):
    """Print summary of walk-forward results"""
    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST SUMMARY")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Backtest period: {results.config.total_years} years")
    print(f"  Min training: {results.config.min_train_years} years")
    print(f"  Step size: {results.config.step_days} days")
    print(f"  Retrain frequency: {results.config.retrain_freq} days")
    print(f"  Probability threshold: {results.config.prob_threshold}")

    print(f"\nResults:")
    print(f"  Periods evaluated: {results.n_periods}")
    print(f"  Total trades: {results.n_trades}")
    print(f"  Average AUC: {results.avg_auc:.4f}")
    print(f"  Average Precision: {results.avg_precision:.4f}")

    print(f"\nPerformance:")
    strat_ret = results.total_return if not np.isnan(results.total_return) else 0
    bench_ret = results.benchmark_return if not np.isnan(results.benchmark_return) else 0
    excess_ret = results.excess_return if not np.isnan(results.excess_return) else 0
    print(f"  Strategy Return: {strat_ret*100:+.2f}%")
    print(f"  Benchmark Return: {bench_ret*100:+.2f}%")
    print(f"  Excess Return: {excess_ret*100:+.2f}%")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {results.max_drawdown*100:.2f}%" if results.max_drawdown and not np.isnan(results.max_drawdown) else "  Max Drawdown: N/A")
    print(f"  Win Rate: {results.win_rate*100:.1f}%")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")

    if results.avg_auc > 0.55:
        print(f"  Model shows predictive skill (AUC={results.avg_auc:.3f} > 0.55)")
    elif results.avg_auc > 0.52:
        print(f"  Model shows weak but potentially usable signal (AUC={results.avg_auc:.3f})")
    else:
        print(f"  Model shows minimal predictive value (AUC={results.avg_auc:.3f})")

    if excess_ret > 0:
        print(f"  Strategy outperformed benchmark by {excess_ret*100:.2f}%")
    else:
        print(f"  Strategy underperformed benchmark by {abs(excess_ret)*100:.2f}%")

    if results.sharpe_ratio > 1.0:
        print(f"  Sharpe ratio is attractive (>{1.0})")
    elif results.sharpe_ratio > 0.5:
        print(f"  Sharpe ratio is acceptable (>{0.5})")
    else:
        print(f"  Sharpe ratio is weak (<{0.5})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward backtest for NeuroVest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 walk_forward_backtest.py --years 5
  python3 walk_forward_backtest.py --years 3 --step-days 21 --retrain-freq 63
  python3 walk_forward_backtest.py --quick
        """
    )

    parser.add_argument('--years', '-y', type=float, default=5.0,
                        help='Total years to backtest (default: 5)')
    parser.add_argument('--min-train', type=float, default=2.0,
                        help='Minimum training years (default: 2)')
    parser.add_argument('--step-days', '-s', type=int, default=21,
                        help='Forward step size in days (default: 21, ~1 month)')
    parser.add_argument('--retrain-freq', '-r', type=int, default=63,
                        help='Retrain frequency in days (default: 63, ~quarterly)')
    parser.add_argument('--horizon', '-H', type=int, default=1,
                        help='Prediction horizon in days (default: 1)')
    parser.add_argument('--threshold', '-t', type=float, default=0.45,
                        help='Probability threshold for trading (default: 0.45)')
    parser.add_argument('--pos-threshold', type=float, default=0.005,
                        help='Positive return threshold (default: 0.005)')
    parser.add_argument('--no-prune', action='store_true',
                        help='Disable feature pruning')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode (faster but less accurate)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    # Build configuration
    config = WalkForwardConfig(
        total_years=args.years,
        min_train_years=args.min_train,
        step_days=args.step_days,
        retrain_freq=args.retrain_freq,
        horizon=args.horizon,
        pos_threshold=args.pos_threshold,
        prob_threshold=args.threshold,
        prune_features=not args.no_prune,
        quick_mode=args.quick,
    )

    if args.quick:
        config.step_days = 42  # ~2 months
        config.retrain_freq = 126  # ~semi-annual

    print("=" * 70)
    print("WALK-FORWARD BACKTEST")
    print("=" * 70)

    # Load data
    df, feature_cols = load_data(prune_features=config.prune_features)

    # Run backtest
    results = run_walk_forward(df, feature_cols, config)

    # Save results
    save_results(results)

    # Generate plots
    if not args.no_plot:
        plot_results(results)

    # Print summary
    print_summary(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
