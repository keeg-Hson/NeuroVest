#!/usr/bin/env python3
"""
NeuroVest Main Entry Point

Interactive menu for all system operations.

Usage:
    python3 main.py
"""

import subprocess
import sys
from pathlib import Path

def clear_screen():
    print("\033[2J\033[H", end="")

def print_header(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_available_assets():
    """Get list of available assets from data_cache"""
    data_cache = Path("data_cache")
    assets = []

    if data_cache.exists():
        for f in data_cache.glob("*_1d.csv"):
            name = f.stem.replace("_1d", "")
            if not name.startswith("fred_"):
                # Convert BTC_USDT back to BTC/USDT format for crypto
                if "_USDT" in name:
                    name = name.replace("_USDT", "/USDT")
                assets.append(name)

    # Always include SPY from main data dir
    if Path("data/SPY.csv").exists() and "SPY" not in assets:
        assets.append("SPY")

    return sorted(assets)

def show_main_menu():
    print_header("NeuroVest - Economic Forecasting System")
    print()
    print("  1. Training")
    print("  2. Predictions")
    print("  3. Backtesting")
    print("  4. Diagnostics")
    print("  5. Data Management")
    print()
    print("  ?. Quick Start Guide")
    print("  0. Exit")
    print()

def show_help():
    clear_screen()
    print_header("Quick Start Guide")
    print()
    print("  TYPICAL WORKFLOW")
    print("  " + "-" * 56)
    print()
    print("  1. DOWNLOAD DATA (first time only)")
    print("     → Go to: 5. Data Management")
    print("     → Select: 1. Update SPY data")
    print("     → Select: 2. Download crypto data")
    print()
    print("  2. TRAIN MODELS")
    print("     → Go to: 1. Training")
    print("     → Select: 1. Standard training (5-10 min)")
    print("     → Or:     4. With optimized weights (better accuracy)")
    print()
    print("  3. GENERATE PREDICTIONS")
    print("     → Go to: 2. Predictions")
    print("     → Select: 1. Generate predictions (SPY)")
    print()
    print("  4. RUN BACKTEST")
    print("     → Go to: 3. Backtesting")
    print("     → Select: 2. Optimized config (balanced)")
    print("     → Or:     3. High profit (330% return)")
    print("     → Or:     4. Aggressive (378% return)")
    print()
    print("  " + "-" * 56)
    print("  BACKTEST CONFIGURATIONS")
    print("  " + "-" * 56)
    print()
    print("  Config           TP ATR   Return   Sharpe   Max DD")
    print("  ─────────────────────────────────────────────────────")
    print("  Optimized        1.25x    191%     2.55     -5.4%")
    print("  High Profit      1.75x    330%     2.30     -7.4%")
    print("  Aggressive       2.5x     378%     2.03     -12.8%")
    print()
    print("  Higher TP ATR = more profit but more risk")
    print()
    input("  Press Enter to return to main menu...")

def show_training_menu():
    print_header("Training Options")
    print()
    print("  MULTI-ASSET (trains on SPY + crypto combined)")
    print("  1. Standard training")
    print("  2. With hyperparameter tuning (15-30 min)")
    print("  3. Quick hyperparameter tuning (5-10 min)")
    print("  4. With optimized ensemble weights")
    print("  5. With feature selection + weight optimization")
    print()
    print("  MULTI-HORIZON (1-day, 3-day, 5-day)")
    print("  6. Train all horizons")
    print("  7. Train specific horizons")
    print()
    print("  PER-ASSET (individual asset models)")
    print("  8. Train all per-asset models (SPY + crypto)")
    print()
    print("  0. Back")
    print()

def show_prediction_menu():
    print_header("Prediction Options")
    print()
    print("  MULTI-ASSET ENSEMBLE")
    print("  1. Generate predictions (SPY)")
    print()
    print("  PER-ASSET")
    print("  2. Predict single asset")
    print("  3. Predict asset group")
    print("  4. Predict all assets")
    print()
    print("  0. Back")
    print()

def show_backtest_menu():
    print_header("Backtesting Options")
    print()
    print("  CONFIGURATIONS")
    print("  1. Backtest (default config)")
    print("  2. Backtest (optimized config)")
    print("  3. Backtest (high profit - 1.75x ATR TP)")
    print("  4. Backtest (aggressive - 2.5x ATR TP)")
    print()
    print("  ASSET SELECTION")
    print("  5. Backtest specific asset")
    print("  6. Compare asset group")
    print("  7. Portfolio backtest")
    print()
    print("  0. Back")
    print()

def show_diagnostics_menu():
    print_header("Diagnostics & Analysis")
    print()
    print("  1. System diagnostics")
    print("  2. Evaluate model metrics")
    print("  3. Analyze correlations")
    print("  4. Compare strategies")
    print()
    print("  0. Back")
    print()

def show_data_menu():
    print_header("Data Management")
    print()
    print("  1. Update SPY data")
    print("  2. Download crypto data")
    print("  3. Download all framework assets")
    print("  4. List available assets")
    print()
    print("  0. Back")
    print()

def select_asset():
    """Interactive asset selection"""
    assets = get_available_assets()

    print("\nAvailable assets:")
    print("-" * 40)

    # Group by type
    crypto = [a for a in assets if "/USDT" in a]
    etfs = [a for a in assets if "/USDT" not in a]

    if etfs:
        print(f"  ETFs: {', '.join(etfs)}")
    if crypto:
        print(f"  Crypto: {', '.join(crypto)}")

    print()
    asset = input("Enter asset ticker (e.g., SPY, BTC/USDT): ").strip().upper()

    # Handle common variations
    if asset == "BTC":
        asset = "BTC/USDT"
    elif asset == "ETH":
        asset = "ETH/USDT"
    elif asset == "SOL":
        asset = "SOL/USDT"

    return asset

def select_asset_group():
    """Interactive asset group selection"""
    print("\nAsset groups:")
    print("-" * 40)
    print("  1. crypto - BTC, ETH, SOL, etc.")
    print("  2. equity - SPY, QQQ, etc.")
    print("  3. sector - XLF, XLK, XLE, etc.")
    print("  4. bond - TLT, IEF, etc.")
    print("  5. commodity - GLD, SLV, etc.")
    print()

    choice = input("Select group (1-5) or type name: ").strip()

    groups = {
        "1": "crypto",
        "2": "equity",
        "3": "sector",
        "4": "bond",
        "5": "commodity"
    }

    return groups.get(choice, choice)

def run_command(cmd, desc=None):
    """Run a command with optional description"""
    if desc:
        print(f"\n{desc}")
    print(f"Running: {' '.join(cmd)}\n")
    print("-" * 60)
    subprocess.run(cmd)
    print("-" * 60)
    input("\nPress Enter to continue...")

def handle_training():
    while True:
        clear_screen()
        show_training_menu()

        choice = input("Select option: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_command(["python3", "train_multi_asset.py"])
        elif choice == "2":
            run_command(["python3", "train_multi_asset.py", "--tune"])
        elif choice == "3":
            run_command(["python3", "train_multi_asset.py", "--tune-fast"])
        elif choice == "4":
            run_command(["python3", "train_multi_asset.py", "--optimize-weights"])
        elif choice == "5":
            run_command(["python3", "train_multi_asset.py", "--optimize-weights", "--feature-select"])
        elif choice == "6":
            run_command(["python3", "train_multi_horizon_signals.py"])
        elif choice == "7":
            horizons = input("Enter horizons (e.g., 1 3 5): ").strip()
            run_command(["python3", "train_multi_horizon_signals.py", "--horizons"] + horizons.split())
        elif choice == "8":
            run_command(["python3", "train_per_asset.py"])

def handle_predictions():
    while True:
        clear_screen()
        show_prediction_menu()

        choice = input("Select option: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_command(["python3", "predict_multi_asset_ensemble.py"])
        elif choice == "2":
            asset = select_asset()
            run_command(["python3", "predict_per_asset.py", "--asset", asset])
        elif choice == "3":
            group = select_asset_group()
            run_command(["python3", "predict_per_asset.py", "--asset-group", group])
        elif choice == "4":
            run_command(["python3", "predict_per_asset.py", "--all"])

def handle_backtesting():
    while True:
        clear_screen()
        show_backtest_menu()

        choice = input("Select option: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_command(["python3", "backtest.py"])
        elif choice == "2":
            run_command(["python3", "backtest.py", "--config", "configs/backtest_optimized.json"])
        elif choice == "3":
            run_command(["python3", "backtest.py", "--config", "configs/backtest_high_profit.json"])
        elif choice == "4":
            run_command(["python3", "backtest.py", "--config", "configs/backtest_aggressive.json"])
        elif choice == "5":
            asset = select_asset()
            run_command(["python3", "backtest.py", "--asset", asset])
        elif choice == "6":
            group = select_asset_group()
            run_command(["python3", "backtest.py", "--asset-group", group, "--compare"])
        elif choice == "7":
            print("\nPortfolio Backtest Setup")
            print("-" * 40)
            assets = input("Assets (comma-separated, e.g., SPY,GLD,BTC/USDT): ").strip()
            weights = input("Weights (comma-separated, e.g., 0.5,0.3,0.2): ").strip()
            rebalance = input("Rebalance (daily/weekly/monthly/quarterly) [monthly]: ").strip() or "monthly"
            run_command([
                "python3", "backtest_portfolio.py",
                "--assets", assets,
                "--weights", weights,
                "--rebalance", rebalance
            ])

def handle_diagnostics():
    while True:
        clear_screen()
        show_diagnostics_menu()

        choice = input("Select option: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_command(["python3", "diagnose_system.py"])
        elif choice == "2":
            run_command(["python3", "evaluate.py"])
        elif choice == "3":
            group = select_asset_group()
            run_command(["python3", "analyze_correlations.py", "--asset-group", group])
        elif choice == "4":
            run_command(["python3", "compare_strategies.py"])

def handle_data():
    while True:
        clear_screen()
        show_data_menu()

        choice = input("Select option: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_command(["python3", "update_spy_data.py"])
        elif choice == "2":
            run_command(["python3", "download_crypto_data.py"])
        elif choice == "3":
            run_command(["python3", "framework/download_all_assets.py"])
        elif choice == "4":
            assets = get_available_assets()
            print("\nAvailable assets:")
            print("-" * 40)
            crypto = [a for a in assets if "/USDT" in a]
            etfs = [a for a in assets if "/USDT" not in a]
            if etfs:
                print(f"ETFs ({len(etfs)}): {', '.join(etfs)}")
            if crypto:
                print(f"Crypto ({len(crypto)}): {', '.join(crypto)}")
            print(f"\nTotal: {len(assets)} assets")
            input("\nPress Enter to continue...")

def main():
    try:
        while True:
            clear_screen()
            show_main_menu()

            choice = input("Select option: ").strip()

            if choice == "0":
                print("\nDone.")
                break
            elif choice == "1":
                handle_training()
            elif choice == "2":
                handle_predictions()
            elif choice == "3":
                handle_backtesting()
            elif choice == "4":
                handle_diagnostics()
            elif choice == "5":
                handle_data()
            elif choice == "?":
                show_help()
            else:
                print("\nInvalid option.")
                input("Press Enter to continue...")

    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
        sys.exit(0)

if __name__ == "__main__":
    main()
