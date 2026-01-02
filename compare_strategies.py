#!/usr/bin/env python3
"""
Strategy Comparison Tool

Compares performance across:
1. Per-asset predictions (asset-specific models)
2. Multi-asset ensemble predictions
3. Portfolio combinations

Usage:
    python3 compare_strategies.py
"""

import sys
import shutil
from pathlib import Path
import pandas as pd

# Import backtest
from backtest import run_backtest

LOGS_DIR = Path("logs")
PREDICTIONS_DIR = LOGS_DIR / "predictions"
COMPARISON_DIR = LOGS_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# Backup original predictions
ORIGINAL_PRED = LOGS_DIR / "daily_predictions.csv"
BACKUP_PRED = LOGS_DIR / "daily_predictions_multiasset.csv"

print(f"\n{'=' * 80}")
print("STRATEGY COMPARISON: PER-ASSET vs MULTI-ASSET ENSEMBLE")
print(f"{'=' * 80}\n")

# 1. Save multi-asset ensemble predictions
if ORIGINAL_PRED.exists():
    shutil.copy(ORIGINAL_PRED, BACKUP_PRED)
    print(f"✓ Backed up multi-asset ensemble predictions")

# Assets to compare
assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

results = []

for asset in assets:
    asset_file = asset.replace('/', '_')

    print(f"\n{'=' * 80}")
    print(f"TESTING: {asset}")
    print(f"{'=' * 80}\n")

    # Test 1: Per-asset predictions
    per_asset_pred = PREDICTIONS_DIR / f"{asset_file}_predictions.csv"

    if per_asset_pred.exists():
        print(f"[1/2] Running per-asset backtest...")
        shutil.copy(per_asset_pred, ORIGINAL_PRED)

        try:
            trades_per, metrics_per, _ = run_backtest(asset=asset)

            result = {
                'Asset': asset,
                'Strategy': 'Per-Asset',
                'Total Return': metrics_per['total_return'],
                'Ann Return': metrics_per['annualized_return'],
                'Sharpe': metrics_per['sharpe'],
                'Max DD': metrics_per['max_drawdown'],
                'Trades': metrics_per['trades'],
                'Win Rate': metrics_per.get('win_rate', 0),
            }
            results.append(result)

            print(f"   ✓ Per-Asset: {metrics_per['total_return']:.2%} return, {metrics_per['sharpe']:.2f} Sharpe")

        except Exception as e:
            print(f"   ✗ Per-asset backtest failed: {e}")

    # Test 2: Multi-asset ensemble
    if BACKUP_PRED.exists():
        print(f"[2/2] Running multi-asset ensemble backtest...")
        shutil.copy(BACKUP_PRED, ORIGINAL_PRED)

        try:
            trades_multi, metrics_multi, _ = run_backtest(asset=asset)

            result = {
                'Asset': asset,
                'Strategy': 'Multi-Asset Ensemble',
                'Total Return': metrics_multi['total_return'],
                'Ann Return': metrics_multi['annualized_return'],
                'Sharpe': metrics_multi['sharpe'],
                'Max DD': metrics_multi['max_drawdown'],
                'Trades': metrics_multi['trades'],
                'Win Rate': metrics_multi.get('win_rate', 0),
            }
            results.append(result)

            print(f"   ✓ Multi-Asset: {metrics_multi['total_return']:.2%} return, {metrics_multi['sharpe']:.2f} Sharpe")

        except Exception as e:
            print(f"   ✗ Multi-asset backtest failed: {e}")

# Restore original predictions
if BACKUP_PRED.exists():
    shutil.copy(BACKUP_PRED, ORIGINAL_PRED)

# Print comparison
print(f"\n{'=' * 80}")
print("COMPARISON RESULTS")
print(f"{'=' * 80}\n")

if results:
    df_results = pd.DataFrame(results)

    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1 else f'{x:.2f}')

    print(df_results.to_string(index=False))

    # Save results
    output_file = COMPARISON_DIR / "per_asset_vs_ensemble.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved comparison: {output_file}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    for asset in assets:
        asset_results = df_results[df_results['Asset'] == asset]

        if len(asset_results) == 2:
            per_asset = asset_results[asset_results['Strategy'] == 'Per-Asset'].iloc[0]
            multi_asset = asset_results[asset_results['Strategy'] == 'Multi-Asset Ensemble'].iloc[0]

            return_diff = per_asset['Total Return'] - multi_asset['Total Return']
            sharpe_diff = per_asset['Sharpe'] - multi_asset['Sharpe']

            print(f"\n{asset}:")
            print(f"  Return improvement: {return_diff:+.2%}")
            print(f"  Sharpe improvement: {sharpe_diff:+.2f}")

            if return_diff > 0 and sharpe_diff > 0:
                print(f"  Winner: ✅ Per-Asset (better on both metrics)")
            elif return_diff > 0 or sharpe_diff > 0:
                print(f"  Winner: ⚠️  Mixed (per-asset better on {'return' if return_diff > 0 else 'sharpe'})")
            else:
                print(f"  Winner: Multi-Asset Ensemble")

    print(f"\n{'=' * 80}\n")

else:
    print("❌ No results to compare")

print("Done!")
