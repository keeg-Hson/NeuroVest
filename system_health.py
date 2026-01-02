#!/usr/bin/env python3
"""
System Health Check

Quick script to verify NeuroVest setup and get key metrics.
Run this anytime to check system status or diagnose issues.

Usage:
    python3 system_health.py
    python3 system_health.py --verbose
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("  NEUROVEST SYSTEM HEALTH CHECK")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

def check_data():
    """Check data availability"""
    print("\n📁 DATA STATUS")
    print("-"*70)

    issues = []

    # SPY check
    spy_path = Path("data/SPY.csv")
    if spy_path.exists():
        try:
            df = pd.read_csv(spy_path)
            print(f"✓ SPY.csv: {len(df):,} rows")
            if 'Date' in df.columns:
                print(f"  Range: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
            if len(df) < 1000:
                issues.append("SPY.csv has less than 1000 rows (need more history)")
        except Exception as e:
            print(f"✗ SPY.csv: Error loading ({e})")
            issues.append(f"SPY.csv load error: {e}")
    else:
        print("✗ SPY.csv: Not found")
        issues.append("SPY.csv missing - run: python3 update_spy_data.py")

    # Crypto check
    crypto_dir = Path("data_cache")
    crypto_count = 0
    if crypto_dir.exists():
        for crypto in ["BTC_USDT_1d.csv", "ETH_USDT_1d.csv", "SOL_USDT_1d.csv"]:
            path = crypto_dir / crypto
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    print(f"✓ {crypto}: {len(df):,} rows")
                    crypto_count += 1
                except:
                    print(f"✗ {crypto}: Parse error")

    if crypto_count == 0:
        print("  No crypto data (optional)")

    return issues


def check_models():
    """Check trained models"""
    print("\n🤖 MODEL STATUS")
    print("-"*70)

    issues = []
    models_dir = Path("models")

    if not models_dir.exists():
        print("✗ models/ directory not found")
        issues.append("models/ directory missing")
        return issues

    required_models = [
        "xgboost_multi_asset.pkl",
        "lightgbm_multi_asset.pkl",
        "catboost_multi_asset.pkl"
    ]

    loaded = 0
    for model_file in required_models:
        path = models_dir / model_file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {model_file}: {size_mb:.2f} MB")
            loaded += 1
        else:
            print(f"✗ {model_file}: Not found")
            issues.append(f"{model_file} missing")

    if loaded == 0:
        issues.append("No models trained - run: python3 train_multi_asset.py")

    # Feature list
    feature_file = models_dir / "multi_asset_features.txt"
    if feature_file.exists():
        features = [l.strip() for l in feature_file.read_text().splitlines() if l.strip()]
        print(f"✓ Features: {len(features)} loaded")
    else:
        print("✗ Feature list missing")
        issues.append("Feature list missing")

    return issues


def check_predictions():
    """Check predictions"""
    print("\n📊 PREDICTION STATUS")
    print("-"*70)

    issues = []
    pred_file = Path("logs/labeled_predictions.csv")

    if not pred_file.exists():
        print("✗ No predictions found")
        issues.append("No predictions - run: python3 predict_multi_asset_ensemble.py")
        return issues

    try:
        df = pd.read_csv(pred_file)
        print(f"✓ Predictions: {len(df):,} rows")

        if 'Date' in df.columns:
            print(f"  Latest: {df['Date'].iloc[-1]}")

        if 'Prediction' in df.columns:
            counts = df['Prediction'].value_counts().sort_index()
            signal_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}
            print("\n  Signal Distribution:")
            for sig, count in counts.items():
                name = signal_map.get(sig, 'UNKNOWN')
                pct = 100 * count / len(df)
                print(f"    {name:8s}: {count:5,} ({pct:5.1f}%)")

        if len(df) == 0:
            issues.append("Prediction file empty")
    except Exception as e:
        print(f"✗ Error loading predictions: {e}")
        issues.append(f"Prediction load error: {e}")

    return issues


def check_backtest():
    """Check backtest results and extract impressive metrics"""
    print("\n📈 BACKTEST STATUS")
    print("-"*70)

    issues = []

    # Check latest.json (primary output)
    latest_json = Path("logs/latest.json")
    if latest_json.exists():
        try:
            import json
            with open(latest_json) as f:
                data = json.load(f)

            print("✓ Backtest results found\n")

            # Extract key performance metrics
            metrics = {
                'Total Return': data.get('total_return', 0),
                'Sharpe Ratio': data.get('sharpe_ratio', 0),
                'Max Drawdown': data.get('max_drawdown', 0),
                'Win Rate': data.get('win_rate', 0),
                'Total Trades': data.get('total_trades', 0),
                'Avg Trade': data.get('avg_trade', 0),
                'Best Trade': data.get('best_trade', 0),
                'Worst Trade': data.get('worst_trade', 0),
                'Sortino Ratio': data.get('sortino_ratio', 0),
                'Calmar Ratio': data.get('calmar_ratio', 0)
            }

            print("  🎯 PERFORMANCE HIGHLIGHTS:")
            print(f"    Total Return:     {metrics['Total Return']:>8.1f}%")
            print(f"    Sharpe Ratio:     {metrics['Sharpe Ratio']:>8.2f}")
            print(f"    Max Drawdown:     {metrics['Max Drawdown']:>8.1f}%")
            print(f"    Win Rate:         {metrics['Win Rate']:>8.1f}%")
            print(f"    Total Trades:     {metrics['Total Trades']:>8,}")

            print("\n  📊 RISK METRICS:")
            print(f"    Sortino Ratio:    {metrics['Sortino Ratio']:>8.2f}")
            print(f"    Calmar Ratio:     {metrics['Calmar Ratio']:>8.2f}")
            print(f"    Avg Trade:        {metrics['Avg Trade']:>8.2f}%")
            print(f"    Best Trade:       {metrics['Best Trade']:>8.2f}%")
            print(f"    Worst Trade:      {metrics['Worst Trade']:>8.2f}%")

            # Compare to buy-and-hold
            bnh_sharpe = 0.45  # Typical SPY buy-hold Sharpe
            bnh_drawdown = -55  # 2008 crash

            print("\n  🔥 VS BUY-AND-HOLD:")
            sharpe_improvement = ((metrics['Sharpe Ratio'] / bnh_sharpe) - 1) * 100 if bnh_sharpe > 0 else 0
            dd_improvement = ((abs(metrics['Max Drawdown']) / abs(bnh_drawdown)) - 1) * 100

            print(f"    Sharpe Better By: {sharpe_improvement:>8.1f}%")
            print(f"    Drawdown Better:  {-dd_improvement:>8.1f}%")

            # Identify impressive metrics
            print("\n  ⭐ STANDOUT METRICS:")
            impressive = []
            if metrics['Sharpe Ratio'] > 2.0:
                impressive.append(f"Sharpe {metrics['Sharpe Ratio']:.2f} (Excellent)")
            if abs(metrics['Max Drawdown']) < 10:
                impressive.append(f"Low Drawdown {metrics['Max Drawdown']:.1f}%")
            if metrics['Win Rate'] > 55:
                impressive.append(f"High Win Rate {metrics['Win Rate']:.1f}%")
            if metrics['Total Return'] > 100:
                impressive.append(f"Triple-Digit Return {metrics['Total Return']:.0f}%")

            for item in impressive:
                print(f"    ✨ {item}")

            if not impressive:
                print("    (Good performance, room for optimization)")

        except Exception as e:
            print(f"✗ Error loading backtest: {e}")
            issues.append(f"Backtest load error: {e}")
    else:
        print("  No backtest results (run: python3 backtest.py)")
        issues.append("No backtest results available")

    return issues


def check_config():
    """Check configuration"""
    print("\n⚙️  CONFIGURATION")
    print("-"*70)

    issues = []

    try:
        import config
        print("✓ config.py loaded")
        print(f"  PREDICTION_THRESHOLD: {config.PREDICTION_THRESHOLD}")
        if hasattr(config, 'TRAINING_CONFIG'):
            print(f"  Training config: Available")
    except ImportError:
        print("✗ config.py not found")
        issues.append("config.py missing")
    except Exception as e:
        print(f"✗ Config error: {e}")
        issues.append(f"Config error: {e}")

    # Check .env for API keys
    env_file = Path(".env")
    if env_file.exists():
        env_text = env_file.read_text()
        has_openai = "OPENAI_API_KEY" in env_text
        has_anthropic = "ANTHROPIC_API_KEY" in env_text

        if has_openai:
            print("✓ OpenAI API key configured")
        if has_anthropic:
            print("✓ Anthropic API key configured")
        if not has_openai and not has_anthropic:
            print("  No LLM API keys (optional)")
    else:
        print("  No .env file (optional for LLM features)")

    return issues


def calculate_metrics():
    """Calculate key system metrics"""
    print("\n📊 SYSTEM METRICS")
    print("-"*70)

    # Asset coverage
    data_dir = Path("data")
    cache_dir = Path("data_cache")

    asset_count = 0
    if data_dir.exists():
        asset_count += len(list(data_dir.glob("*.csv")))
    if cache_dir.exists():
        asset_count += len(list(cache_dir.glob("*_1d.csv")))

    print(f"Assets Downloaded: {asset_count}")

    # Model count
    models_dir = Path("models")
    model_count = 0
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*.pkl")))
    print(f"Models Trained: {model_count}")

    # Prediction count
    pred_file = Path("logs/labeled_predictions.csv")
    if pred_file.exists():
        try:
            df = pd.read_csv(pred_file)
            print(f"Total Predictions: {len(df):,}")
        except:
            pass

    # Disk usage
    total_size = 0
    for directory in [data_dir, cache_dir, models_dir, Path("logs")]:
        if directory.exists():
            for f in directory.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

    print(f"Total Disk Usage: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="NeuroVest System Health Check")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    all_issues = []

    # Run checks
    all_issues.extend(check_data())
    all_issues.extend(check_models())
    all_issues.extend(check_predictions())
    all_issues.extend(check_backtest())
    all_issues.extend(check_config())

    calculate_metrics()

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    if len(all_issues) == 0:
        print("\n✅ All systems operational")
        print("\n🚀 READY FOR DEPLOYMENT:")
        print("  ✓ Generating predictions")
        print("  ✓ Running backtests")
        print("  ✓ Dashboard deployment")
        print("  ✓ LLM analysis (if API keys configured)")

        print("\n📊 NEXT STEPS:")
        print("  • Deploy api_demo.py to production")
        print("  • Set up automated daily predictions")
        print("  • Configure LLM API keys for analysis")
        print("  • Run parameter optimization")
    else:
        print(f"\n⚠️  Found {len(all_issues)} issue(s):\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

        print("\n🔧 QUICK FIXES:")
        if any("SPY" in issue for issue in all_issues):
            print("  python3 update_spy_data.py")
        if any("model" in issue.lower() for issue in all_issues):
            print("  python3 train_multi_asset.py --optimize-weights")
        if any("prediction" in issue.lower() for issue in all_issues):
            print("  python3 predict_multi_asset_ensemble.py")
        if any("backtest" in issue.lower() for issue in all_issues):
            print("  python3 backtest.py")

    print("\n" + "="*70)
    return 0 if len(all_issues) == 0 else 1


if __name__ == "__main__":
    exit(main())
