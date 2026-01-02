#!/usr/bin/env python3
"""
Compare Per-Asset vs Multi-Asset Training Approaches

Analyzes results from both training strategies and recommends which to use.
"""

import pandas as pd
from pathlib import Path
import json

print("=" * 80)
print("TRAINING APPROACH COMPARISON")
print("=" * 80)

MODELS_DIR = Path("models")

# Load baseline (SPY-only before adding other assets)
baseline_path = Path("logs/baseline_spy_only.csv")
if baseline_path.exists():
    baseline = pd.read_csv(baseline_path)
    if 'accuracy' in baseline.columns:
        baseline_acc = baseline['accuracy'].values[0]
    elif 'Accuracy' in baseline.columns:
        baseline_acc = baseline['Accuracy'].values[0]
    else:
        baseline_acc = 0.71  # Fallback to known value
    print(f"\n📊 Baseline (SPY-only): {baseline_acc:.1%} accuracy")
else:
    baseline_acc = 0.71
    print(f"\n📊 Baseline (SPY-only): {baseline_acc:.1%} accuracy (default)")

# Load per-asset results
per_asset_path = MODELS_DIR / "per_asset_results.csv"
if per_asset_path.exists():
    per_asset = pd.read_csv(per_asset_path)
    spy_per = per_asset[per_asset['Asset'] == 'SPY']

    if len(spy_per) > 0:
        spy_per_acc = spy_per.iloc[0]['Ensemble_Accuracy']
        print(f"📊 Per-Asset SPY:      {spy_per_acc:.1%} accuracy")

        print("\nPer-Asset Results (All Assets):")
        print(per_asset[['Asset', 'Samples', 'Ensemble_Accuracy']].to_string(index=False))
    else:
        spy_per_acc = None
        print("⚠️ No SPY results in per-asset training")
else:
    spy_per_acc = None
    print("⚠️ Per-asset results not found. Run: python train_per_asset.py")

# Load multi-asset results
multi_asset_path = MODELS_DIR / "multi_asset_results.csv"
if multi_asset_path.exists():
    multi_asset = pd.read_csv(multi_asset_path)
    ensemble = multi_asset[multi_asset['Model'] == 'Ensemble']

    if len(ensemble) > 0:
        multi_acc = ensemble.iloc[0]['Accuracy']
        print(f"\n📊 Multi-Asset:        {multi_acc:.1%} accuracy")

        print("\nMulti-Asset Results (All Models):")
        print(multi_asset[['Model', 'Accuracy', 'Precision', 'Recall']].to_string(index=False))

        # Show asset composition
        print(f"\nTraining samples: {ensemble.iloc[0].get('Features', 'N/A')}")
    else:
        multi_acc = None
        print("⚠️ No ensemble results in multi-asset training")
else:
    multi_acc = None
    print("⚠️ Multi-asset results not found. Run: python train_multi_asset.py")

# Load backtest results (if available)
print("\n" + "=" * 80)
print("BACKTEST PERFORMANCE")
print("=" * 80)

try:
    import glob
    backtest_runs = sorted(glob.glob('logs/run_*.json'))
    if backtest_runs:
        with open(backtest_runs[-1]) as f:
            backtest = json.load(f)

        print(f"\nLatest Backtest (from {backtest_runs[-1]}):")
        print(f"  Total Return:  {backtest['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio:  {backtest['sharpe']:.2f}")
        print(f"  Max Drawdown:  {backtest['max_drawdown']*100:.2f}%")
        print(f"  Win Rate:      {backtest['win_rate']*100:.2f}%")
        print(f"  Trades:        {backtest['trades']}")
except Exception as e:
    print(f"⚠️ Could not load backtest results: {e}")

# Comparison and recommendation
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if spy_per_acc and multi_acc:
    # Both available, compare
    diff = multi_acc - spy_per_acc
    pct_change = (multi_acc / spy_per_acc - 1) * 100

    print(f"\nSPY Performance Comparison:")
    print(f"  Baseline (before):     {baseline_acc:.1%}")
    print(f"  Per-Asset (after):     {spy_per_acc:.1%}  ({(spy_per_acc - baseline_acc)*100:+.1f} pp)")
    print(f"  Multi-Asset (after):   {multi_acc:.1%}  ({(multi_acc - baseline_acc)*100:+.1f} pp)")
    print(f"\n  Difference: {diff*100:+.2f} pp ({pct_change:+.1f}%)")

    # Decision criteria
    if multi_acc >= 0.71 and multi_acc >= spy_per_acc:
        print("\n✅ RECOMMENDATION: Use Multi-Asset approach")
        print("   Reasons:")
        if multi_acc > spy_per_acc:
            print(f"   - Better accuracy than per-asset ({multi_acc:.1%} vs {spy_per_acc:.1%})")
        if multi_acc >= baseline_acc * 1.02:
            print(f"   - Significant improvement over baseline (>{baseline_acc*1.02:.1%})")
        print("   - More training data = better generalization")
        print("\n   Next steps:")
        print("   1. Run backtest: python backtest.py")
        print("   2. Verify Sharpe >= 1.60 and drawdown <= -22%")
        print("   3. If backtest good, keep multi-asset models")

    elif spy_per_acc and spy_per_acc >= 0.71:
        print("\n✅ RECOMMENDATION: Use Per-Asset approach")
        print("   Reasons:")
        if spy_per_acc > multi_acc:
            print(f"   - Better accuracy than multi-asset ({spy_per_acc:.1%} vs {multi_acc:.1%})")
        print("   - Asset-specific patterns may be more relevant")
        print("\n   Next steps:")
        print("   1. Update predict script to use spy_*.pkl models")
        print("   2. Run backtest: python backtest.py")
        print("   3. Compare with multi-asset backtest")

    else:
        print("\n⚠️ RECOMMENDATION: Revert to Baseline (SPY-only)")
        print("   Reasons:")
        print(f"   - Both approaches performed worse than baseline ({baseline_acc:.1%})")
        print("   - Adding assets introduced negative transfer learning")
        print("\n   Next steps:")
        print("   1. Use baseline SPY-only models")
        print("   2. Consider extending SPY history to 1993 instead")
        print("   3. Focus on feature engineering rather than more assets")

elif multi_acc:
    # Only multi-asset available
    if multi_acc >= baseline_acc:
        print(f"\n✅ Multi-Asset: {multi_acc:.1%} (improved by {(multi_acc - baseline_acc)*100:+.1f} pp)")
        print("   Run backtest to verify performance")
    else:
        print(f"\n⚠️ Multi-Asset: {multi_acc:.1%} (worse by {(baseline_acc - multi_acc)*100:.1f} pp)")
        print("   Consider reverting to baseline")

elif spy_per_acc:
    # Only per-asset available
    if spy_per_acc >= baseline_acc:
        print(f"\n✅ Per-Asset: {spy_per_acc:.1%} (improved by {(spy_per_acc - baseline_acc)*100:+.1f} pp)")
        print("   Run backtest to verify performance")
    else:
        print(f"\n⚠️ Per-Asset: {spy_per_acc:.1%} (worse by {(baseline_acc - spy_per_acc)*100:.1f} pp)")
        print("   Consider reverting to baseline")

else:
    print("\n⚠️ No new training results found")
    print("   Run: python train_multi_asset.py or python train_per_asset.py")

# Success criteria reminder
print("\n" + "=" * 80)
print("SUCCESS CRITERIA (for any approach)")
print("=" * 80)
print("\nAccept new models if:")
print("  ✓ Test accuracy:  >= 71.0%")
print("  ✓ Sharpe ratio:   >= 1.60")
print("  ✓ Max drawdown:   <= -22%")
print("  ✓ Model agreement: >= 85%")
print("\nReject if:")
print("  ✗ Accuracy drops > 2%")
print("  ✗ Sharpe drops significantly")
print("  ✗ Training acc >> test acc (overfitting)")
