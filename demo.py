#!/usr/bin/env python3
"""
NeuroVest Quick Demo
Simple demonstration of core prediction functionality

Usage:
    python3 demo.py                    # Quick prediction demo
    python3 demo.py --full             # Full pipeline demo
    python3 demo.py --backtest         # Run backtest demo
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import joblib
from pathlib import Path

print("="*70)
print("  NEUROVEST QUICK DEMO")
print("="*70)

def demo_predictions():
    """Quick demo of prediction system"""
    print("\n📊 PREDICTION DEMO")
    print("-"*70)

    # Check if predictions exist
    pred_file = Path("logs/labeled_predictions.csv")
    if not pred_file.exists():
        print("❌ No predictions found. Run this first:")
        print("   python3 predict_multi_asset_ensemble.py")
        return

    # Load predictions
    df = pd.read_csv(pred_file)
    print(f"✓ Loaded {len(df):,} predictions\n")

    # Show signal distribution
    if 'Prediction' in df.columns:
        signal_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}
        counts = df['Prediction'].value_counts().sort_index()
        print("📈 Signal Distribution:")
        for signal, count in counts.items():
            name = signal_map.get(signal, 'UNKNOWN')
            pct = 100 * count / len(df)
            print(f"   {name:8s}: {count:5,} ({pct:5.1f}%)")

    # Show recent predictions
    print("\n📅 Recent Predictions (Last 10):")
    print("-"*70)
    recent = df.tail(10)[['Date', 'Prediction', 'Proba', 'Confidence']].copy()

    if not recent.empty:
        recent['Signal'] = recent['Prediction'].map({0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'})
        recent['Proba'] = recent['Proba'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        recent['Confidence'] = recent['Confidence'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        print(recent[['Date', 'Signal', 'Proba', 'Confidence']].to_string(index=False))

    print("\n" + "="*70)


def demo_models():
    """Quick demo of loaded models"""
    print("\n🤖 MODEL DEMO")
    print("-"*70)

    models_dir = Path("models")
    model_files = [
        "xgboost_multi_asset.pkl",
        "lightgbm_multi_asset.pkl",
        "catboost_multi_asset.pkl"
    ]

    loaded = 0
    for model_file in model_files:
        path = models_dir / model_file
        if path.exists():
            try:
                model = joblib.load(path)
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✓ {model_file:30s} ({size_mb:6.2f} MB)")
                loaded += 1
            except Exception as e:
                print(f"   ✗ {model_file:30s} (failed to load)")
        else:
            print(f"   ✗ {model_file:30s} (not found)")

    if loaded == 0:
        print("\n❌ No models found. Run this first:")
        print("   python3 train_multi_asset.py")
    else:
        print(f"\n✓ {loaded}/3 models loaded successfully")

    # Check feature list
    feature_file = models_dir / "multi_asset_features.txt"
    if feature_file.exists():
        features = [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]
        print(f"✓ Feature list: {len(features)} features")
    else:
        print("✗ Feature list not found")

    print("="*70)


def demo_backtest():
    """Quick demo of backtest results"""
    print("\n📊 BACKTEST DEMO")
    print("-"*70)

    backtest_file = Path("logs/backtest_results.csv")
    if not backtest_file.exists():
        print("❌ No backtest results found. Run this first:")
        print("   python3 backtest.py")
        return

    df = pd.read_csv(backtest_file)
    print(f"✓ Loaded backtest results: {len(df):,} trades\n")

    if len(df) > 0:
        # Calculate key metrics
        if 'profit_pct' in df.columns:
            total_return = df['profit_pct'].sum()
            win_rate = (df['profit_pct'] > 0).sum() / len(df) * 100
            avg_win = df[df['profit_pct'] > 0]['profit_pct'].mean() if (df['profit_pct'] > 0).any() else 0
            avg_loss = df[df['profit_pct'] < 0]['profit_pct'].mean() if (df['profit_pct'] < 0).any() else 0

            print("📈 Performance Summary:")
            print(f"   Total Return:   {total_return:7.2f}%")
            print(f"   Win Rate:       {win_rate:7.2f}%")
            print(f"   Avg Win:        {avg_win:7.2f}%")
            print(f"   Avg Loss:       {avg_loss:7.2f}%")
            print(f"   Total Trades:   {len(df):7,}")

    print("\n" + "="*70)


def demo_data():
    """Quick demo of data status"""
    print("\n📁 DATA STATUS")
    print("-"*70)

    spy_file = Path("data/SPY.csv")
    if spy_file.exists():
        df = pd.read_csv(spy_file)
        if len(df) > 0:
            first_date = df['Date'].iloc[0] if 'Date' in df.columns else "Unknown"
            last_date = df['Date'].iloc[-1] if 'Date' in df.columns else "Unknown"
            print(f"   ✓ SPY.csv: {len(df):,} rows")
            print(f"     Range: {first_date} → {last_date}")
        else:
            print("   ⚠️  SPY.csv is empty")
            print("   Run: python3 update_spy_data.py")
    else:
        print("   ✗ SPY.csv not found")
        print("   Run: python3 update_spy_data.py")

    # Check crypto data
    crypto_dir = Path("data")
    crypto_files = ["BTC.csv", "ETH.csv", "SOL.csv"]
    crypto_count = 0
    for crypto_file in crypto_files:
        path = crypto_dir / crypto_file
        if path.exists():
            df = pd.read_csv(path)
            if len(df) > 0:
                print(f"   ✓ {crypto_file}: {len(df):,} rows")
                crypto_count += 1

    if crypto_count == 0:
        print("\n   ℹ️  No crypto data found (optional)")
        print("   Run: python3 download_crypto_enhanced.py")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="NeuroVest Quick Demo")
    parser.add_argument("--full", action="store_true", help="Run full demo")
    parser.add_argument("--backtest", action="store_true", help="Show backtest demo")
    args = parser.parse_args()

    if args.full:
        # Full pipeline demo
        demo_data()
        demo_models()
        demo_predictions()
        demo_backtest()
    elif args.backtest:
        demo_backtest()
    else:
        # Quick demo - just predictions
        demo_predictions()

    print("\n💡 TIP: For comprehensive demos, run:")
    print("   python3 demo_comprehensive.py")
    print("\n💡 TIP: For web dashboard, run:")
    print("   streamlit run dashboard.py")
    print()


if __name__ == "__main__":
    main()
