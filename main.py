#!/usr/bin/env python3
"""
NeuroVest Main Entry Point

Run this to start using the system. It will guide you through
the available options.

Usage:
    python3 main.py
"""

import subprocess
import sys

def main():
    print("=" * 60)
    print("NeuroVest - Economic Forecasting System")
    print("=" * 60)

    print("\nWhat would you like to do?\n")
    print("  TRAINING")
    print("  1. Train multi-asset models")
    print("  2. Train with hyperparameter tuning")
    print("  3. Train multi-horizon models (1d, 3d, 5d)")
    print("")
    print("  PREDICTION & BACKTEST")
    print("  4. Generate predictions")
    print("  5. Run backtest")
    print("  6. Run backtest (optimized config)")
    print("")
    print("  DIAGNOSTICS")
    print("  7. System diagnostics")
    print("  8. Evaluate model metrics")
    print("")
    print("  0. Exit")

    try:
        choice = input("\nSelect option: ").strip()

        commands = {
            "1": ["python3", "train_multi_asset.py"],
            "2": ["python3", "train_multi_asset.py", "--tune"],
            "3": ["python3", "train_multi_horizon_signals.py"],
            "4": ["python3", "predict_multi_asset_ensemble.py"],
            "5": ["python3", "backtest.py"],
            "6": ["python3", "backtest.py", "--config", "configs/backtest_optimized.json"],
            "7": ["python3", "diagnose_system.py"],
            "8": ["python3", "evaluate.py"],
        }

        if choice in commands:
            cmd = commands[choice]
            print(f"\nRunning: {' '.join(cmd)}\n")
            print("-" * 60)
            subprocess.run(cmd)
        elif choice == "0":
            print("\nDone.")
        else:
            print("\nInvalid option.")

    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
        sys.exit(0)

if __name__ == "__main__":
    main()
