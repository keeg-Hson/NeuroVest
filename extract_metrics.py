#!/usr/bin/env python3
"""
Extract Real Performance Metrics

Runs actual predictions and backtests to extract accurate performance statistics.
Updates documentation with real metrics instead of placeholder values.

Usage:
    python3 extract_metrics.py --comprehensive
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import sys


def check_requirements():
    """Check if required files exist"""
    required = {
        'models': [
            'models/xgboost_multi_asset.pkl',
            'models/lightgbm_multi_asset.pkl',
            'models/catboost_multi_asset.pkl'
        ],
        'data': [
            'data/SPY.csv'
        ]
    }

    missing = []
    for category, files in required.items():
        for f in files:
            if not Path(f).exists():
                missing.append(f)

    return missing


def extract_model_metrics():
    """Extract metrics from trained models"""
    print("\n" + "="*70)
    print("EXTRACTING MODEL PERFORMANCE METRICS")
    print("="*70)

    metrics = {}

    # Check if models exist
    model_dir = Path("models")
    if not model_dir.exists():
        print("❌ No models directory found. Train models first!")
        return metrics

    # Check for best hyperparameters (indicates tuning was done)
    hp_file = model_dir / "best_hyperparameters.json"
    if hp_file.exists():
        with open(hp_file) as f:
            hp_data = json.load(f)
            metrics['hyperparameter_tuning'] = "Completed"
            metrics['tuned_models'] = list(hp_data.keys())
    else:
        metrics['hyperparameter_tuning'] = "Not performed"

    # Count trained models
    model_files = list(model_dir.glob("*.pkl"))
    metrics['total_models'] = len(model_files)
    metrics['model_types'] = [f.stem for f in model_files]

    return metrics


def extract_prediction_metrics():
    """Extract metrics from prediction files"""
    print("\n" + "="*70)
    print("EXTRACTING PREDICTION METRICS")
    print("="*70)

    metrics = {}

    # Check labeled predictions
    pred_file = Path("logs/labeled_predictions.csv")
    if not pred_file.exists():
        print("⚠️  No predictions found. Run predict_multi_asset_ensemble.py first!")
        return metrics

    df = pd.read_csv(pred_file)

    # Signal distribution
    pred_counts = df['Prediction'].value_counts()
    total = len(df)

    metrics['total_predictions'] = total
    metrics['signal_distribution'] = {
        'CRASH (0)': f"{pred_counts.get(0, 0)} ({100*pred_counts.get(0, 0)/total:.1f}%)",
        'NORMAL (1)': f"{pred_counts.get(1, 0)} ({100*pred_counts.get(1, 0)/total:.1f}%)",
        'SPIKE (2)': f"{pred_counts.get(2, 0)} ({100*pred_counts.get(2, 0)/total:.1f}%)"
    }

    # Confidence statistics
    if 'Confidence' in df.columns:
        conf = df['Confidence']
        metrics['confidence_stats'] = {
            'mean': f"{conf.mean():.3f}",
            'median': f"{conf.median():.3f}",
            'min': f"{conf.min():.3f}",
            'max': f"{conf.max():.3f}"
        }

    # Probability statistics
    if 'Proba' in df.columns:
        proba = df['Proba']
        metrics['probability_stats'] = {
            'mean': f"{proba.mean():.3f}",
            'median': f"{proba.median():.3f}",
            'std': f"{proba.std():.3f}"
        }

    # If we have labels, calculate accuracy
    if 'Label' in df.columns:
        df_labeled = df.dropna(subset=['Label'])
        if len(df_labeled) > 0:
            # Binary classification (positive vs negative)
            y_true = (df_labeled['Label'] > 0).astype(int)
            y_pred = (df_labeled['Prediction'] == 2).astype(int)

            accuracy = (y_true == y_pred).mean()
            metrics['test_accuracy'] = f"{accuracy:.1%}"

            # Calculate precision and recall for SPIKE signals
            true_positives = ((y_true == 1) & (y_pred == 1)).sum()
            false_positives = ((y_true == 0) & (y_pred == 1)).sum()
            false_negatives = ((y_true == 1) & (y_pred == 0)).sum()

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            metrics['spike_precision'] = f"{precision:.1%}"
            metrics['spike_recall'] = f"{recall:.1%}"

    return metrics


def extract_backtest_metrics():
    """Extract metrics from backtest results"""
    print("\n" + "="*70)
    print("EXTRACTING BACKTEST METRICS")
    print("="*70)

    metrics = {}

    # Check for backtest results
    results_file = Path("logs/backtest_results.csv")
    if not results_file.exists():
        print("⚠️  No backtest results found. Run backtest.py first!")
        return metrics

    try:
        df = pd.read_csv(results_file)

        # Calculate performance metrics
        if 'portfolio_value' in df.columns:
            initial_value = df['portfolio_value'].iloc[0]
            final_value = df['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value - 1) * 100

            metrics['total_return'] = f"{total_return:.1f}%"
            metrics['initial_value'] = f"${initial_value:,.0f}"
            metrics['final_value'] = f"${final_value:,.0f}"

        # Calculate drawdown
        if 'portfolio_value' in df.columns:
            portfolio = df['portfolio_value']
            running_max = portfolio.expanding().max()
            drawdown = (portfolio - running_max) / running_max * 100
            max_drawdown = drawdown.min()

            metrics['max_drawdown'] = f"{max_drawdown:.1f}%"

        # Calculate Sharpe ratio
        if 'returns' in df.columns:
            returns = df['returns'].dropna()
            if len(returns) > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                metrics['sharpe_ratio'] = f"{sharpe:.2f}"

        # Win rate
        if 'pnl' in df.columns:
            trades = df[df['pnl'].notna()]
            if len(trades) > 0:
                wins = (trades['pnl'] > 0).sum()
                total_trades = len(trades)
                win_rate = wins / total_trades * 100

                metrics['total_trades'] = total_trades
                metrics['winning_trades'] = wins
                metrics['win_rate'] = f"{win_rate:.1f}%"

    except Exception as e:
        print(f"❌ Error reading backtest results: {e}")

    return metrics


def extract_asset_coverage():
    """Extract available asset coverage"""
    print("\n" + "="*70)
    print("EXTRACTING ASSET COVERAGE")
    print("="*70)

    metrics = {}

    # Check data cache
    data_cache = Path("data_cache")
    if data_cache.exists():
        crypto_assets = list(data_cache.glob("*_USDT_1d.csv"))
        other_assets = [f for f in data_cache.glob("*_1d.csv") if "_USDT" not in f.name]

        metrics['crypto_assets'] = len(crypto_assets)
        metrics['other_assets'] = len(other_assets)
        metrics['total_cached_assets'] = len(crypto_assets) + len(other_assets)

        # List specific crypto
        crypto_list = [f.stem.replace("_1d", "").replace("_", "/") for f in crypto_assets]
        metrics['crypto_list'] = crypto_list[:10]  # First 10

    # Check for per-asset predictions
    pred_dir = Path("logs/predictions")
    if pred_dir.exists():
        pred_files = list(pred_dir.glob("*_predictions.csv"))
        metrics['assets_with_predictions'] = len(pred_files)
        metrics['prediction_files'] = [f.stem.replace("_predictions", "") for f in pred_files[:10]]

    return metrics


def run_quick_prediction_test():
    """Run a quick prediction to get real-time metrics"""
    print("\n" + "="*70)
    print("RUNNING LIVE PREDICTION TEST")
    print("="*70)

    try:
        print("\nGenerating fresh predictions...")
        result = subprocess.run(
            ["python3", "predict_multi_asset_ensemble.py"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✓ Predictions generated successfully")

            # Extract metrics from output
            output = result.stdout
            metrics = {}

            # Look for key metrics in output
            if "Prediction Distribution:" in output:
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if "CRASH" in line and "(" in line:
                        try:
                            pct = line.split('(')[1].split('%')[0].strip()
                            metrics['crash_pct'] = f"{pct}%"
                        except Exception:
                            pass
                    elif "NORMAL" in line and "(" in line:
                        try:
                            pct = line.split('(')[1].split('%')[0].strip()
                            metrics['normal_pct'] = f"{pct}%"
                        except Exception:
                            pass
                    elif "SPIKE" in line and "(" in line:
                        try:
                            pct = line.split('(')[1].split('%')[0].strip()
                            metrics['spike_pct'] = f"{pct}%"
                        except Exception:
                            pass

            return metrics
        else:
            print(f"⚠️  Prediction failed: {result.stderr}")
            return {}

    except subprocess.TimeoutExpired:
        print("⚠️  Prediction timed out")
        return {}
    except Exception as e:
        print(f"❌ Error running prediction: {e}")
        return {}


def compile_all_metrics(run_live_test=False):
    """Compile all metrics into a comprehensive report"""
    print("\n" + "="*70)
    print("COMPILING COMPREHENSIVE METRICS REPORT")
    print("="*70)

    all_metrics = {
        'timestamp': datetime.now().isoformat(),
        'models': extract_model_metrics(),
        'predictions': extract_prediction_metrics(),
        'backtest': extract_backtest_metrics(),
        'assets': extract_asset_coverage()
    }

    if run_live_test:
        all_metrics['live_test'] = run_quick_prediction_test()

    return all_metrics


def save_metrics_report(metrics, output_path="metrics_report.json"):
    """Save metrics to JSON file"""
    output_file = Path(output_path)

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n✓ Metrics saved to: {output_file}")
    return output_file


def print_metrics_summary(metrics):
    """Print a human-readable summary"""
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)

    # Models
    if 'models' in metrics and metrics['models']:
        print("\n📦 MODELS:")
        m = metrics['models']
        print(f"  Total Models: {m.get('total_models', 0)}")
        print(f"  Hyperparameter Tuning: {m.get('hyperparameter_tuning', 'Unknown')}")

    # Predictions
    if 'predictions' in metrics and metrics['predictions']:
        print("\n🎯 PREDICTIONS:")
        m = metrics['predictions']
        print(f"  Total Predictions: {m.get('total_predictions', 0)}")

        if 'signal_distribution' in m:
            print("  Signal Distribution:")
            for signal, value in m['signal_distribution'].items():
                print(f"    {signal}: {value}")

        if 'test_accuracy' in m:
            print(f"  Test Accuracy: {m['test_accuracy']}")
            print(f"  SPIKE Precision: {m.get('spike_precision', 'N/A')}")
            print(f"  SPIKE Recall: {m.get('spike_recall', 'N/A')}")

    # Backtest
    if 'backtest' in metrics and metrics['backtest']:
        print("\n📈 BACKTEST:")
        m = metrics['backtest']
        print(f"  Total Return: {m.get('total_return', 'N/A')}")
        print(f"  Sharpe Ratio: {m.get('sharpe_ratio', 'N/A')}")
        print(f"  Max Drawdown: {m.get('max_drawdown', 'N/A')}")
        print(f"  Win Rate: {m.get('win_rate', 'N/A')}")
        print(f"  Total Trades: {m.get('total_trades', 'N/A')}")

    # Assets
    if 'assets' in metrics and metrics['assets']:
        print("\n💼 ASSETS:")
        m = metrics['assets']
        print(f"  Crypto Assets: {m.get('crypto_assets', 0)}")
        print(f"  Other Assets: {m.get('other_assets', 0)}")
        print(f"  Assets with Predictions: {m.get('assets_with_predictions', 0)}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Extract Real Performance Metrics")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive test including live predictions")
    parser.add_argument("--output", default="metrics_report.json",
                        help="Output file path")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check requirements, don't extract")

    args = parser.parse_args()

    print("="*70)
    print("NEUROVEST METRICS EXTRACTION")
    print("="*70)

    # Check requirements
    missing = check_requirements()
    if missing:
        print("\n⚠️  Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease train models and run predictions first:")
        print("  1. python3 train_multi_asset.py --optimize-weights")
        print("  2. python3 predict_multi_asset_ensemble.py")
        print("  3. python3 backtest.py")

        if args.check_only:
            return

        print("\nContinuing with available data...")

    # Compile metrics
    metrics = compile_all_metrics(run_live_test=args.comprehensive)

    # Print summary
    print_metrics_summary(metrics)

    # Save report
    output_file = save_metrics_report(metrics, args.output)

    print(f"\n✅ Metrics extraction complete!")
    print(f"\nUse this data to update:")
    print("  - README.md (performance metrics section)")
    print("  - Documentation files")
    print("  - Dashboard displays")


if __name__ == "__main__":
    main()
