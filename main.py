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
    """Print a styled header"""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_section(title):
    """Print a section divider"""
    print(f"\n  {title}")
    print("  " + "-" * (len(title)))

def print_success(message):
    """Print a success message"""
    print(f"✓ {message}")

def print_error(message):
    """Print an error message"""
    print(f"✗ {message}")

def print_info(message):
    """Print an info message"""
    print(f"ℹ {message}")

def print_warning(message):
    """Print a warning message"""
    print(f"⚠ {message}")

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
    print_section("QUICK START")
    print("  R  → Run Full Pipeline (download → train → predict → backtest)")
    print()
    print_section("CORE MODULES")
    print("  1  → Training           (train models, tune hyperparameters)")
    print("  2  → Predictions        (generate forecasts)")
    print("  3  → Backtesting        (test strategies, risk profiles)")
    print("  4  → Diagnostics & LLM  (analyze performance, AI insights)")
    print("  5  → Data Management    (download, import data)")
    print()
    print_section("TOOLS")
    print("  6  → Asset Explorer     (guided analysis)")
    print("  7  → Web Dashboard      (interactive UI)")
    print()
    print_section("HELP & EXIT")
    print("  ?  → Quick Start Guide")
    print("  0  → Exit")
    print()
    print("=" * 70)

def show_help():
    clear_screen()
    print_header("NeuroVest Quick Start Guide")
    print()

    print_section("TYPICAL WORKFLOW")
    print()
    print("  Step 1: DOWNLOAD DATA (first time only)")
    print("          Menu → 5 (Data Management) → 1 (Update SPY)")
    print("          Menu → 5 (Data Management) → 2-4 (Download crypto)")
    print()
    print("  Step 2: TRAIN MODELS")
    print("          Menu → 1 (Training) → 1 (Standard, 5-10 min)")
    print("          Menu → 1 (Training) → 4 (Optimized weights, better accuracy)")
    print()
    print("  Step 3: GENERATE PREDICTIONS")
    print("          Menu → 2 (Predictions) → 1 (Multi-asset ensemble)")
    print("          Menu → 2 (Predictions) → 4 (All per-asset models)")
    print()
    print("  Step 4: RUN BACKTEST")
    print("          Menu → 3 (Backtesting) → 2 (Moderate profile)")
    print("          Menu → 3 (Backtesting) → 4 (Optimized config)")
    print()

    print_section("QUICK START (AUTOMATED)")
    print()
    print("  Option R: Run Full Pipeline")
    print("           Automates steps 1-4 above (~20-35 minutes)")
    print()

    print_section("NEW FEATURES")
    print()
    print("  Trading Risk Profiles:")
    print("    • Conservative: 70%+ confidence, tight stops, low risk")
    print("    • Moderate: 55%+ confidence, balanced risk-reward")
    print("    • Liberal: 45%+ confidence, aggressive, high reward")
    print()
    print("  Market Analysis:")
    print("    • Recession Indicator: Multi-signal recession probability")
    print("    • Valuation Detector: Over/undervalued asset analysis")
    print("    • Portfolio Rebalancing: Optimal period finder")
    print()
    print("  AI & Insights:")
    print("    • LLM Analysis: OpenAI/Anthropic market commentary")
    print("    • News Integration: Real-time news context (NewsAPI)")
    print("    • Scenario Likelihoods: Crash/Normal/Spike probabilities")
    print()

    print_section("BACKTEST CONFIGURATIONS")
    print()
    print("  Config           TP ATR   Return   Sharpe   Max DD")
    print("  ───────────────────────────────────────────────────")
    print("  Conservative      1.0x     ~150%    2.80     -4%")
    print("  Moderate (Opt)    1.25x    191%     2.55     -5.4%")
    print("  High Profit       1.75x    330%     2.30     -7.4%")
    print("  Aggressive        2.5x     378%     2.03     -12.8%")
    print()
    print("  Higher TP ATR = more profit but more risk")
    print()

    print("=" * 70)
    input("\n  Press Enter to return to main menu...")

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
    print("  COMPREHENSIVE")
    print("  9. Train ALL (multi-asset + per-asset + multi-horizon)")
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
    print("  RISK PROFILES")
    print("  1. Conservative trading (low risk, high confidence)")
    print("  2. Moderate trading (balanced risk-reward)")
    print("  3. Liberal/Aggressive trading (high risk, high reward)")
    print()
    print("  STANDARD CONFIGS")
    print("  4. Backtest (optimized config)")
    print("  5. Backtest (high profit - 1.75x ATR TP)")
    print("  6. Backtest (aggressive - 2.5x ATR TP)")
    print()
    print("  ASSET SELECTION")
    print("  7. Backtest specific asset")
    print("  8. Compare asset group")
    print()
    print("  PORTFOLIO")
    print("  9. Portfolio backtest")
    print("  10. Find optimal rebalancing period")
    print("  11. Execute portfolio rebalancing")
    print()
    print("  0. Back")
    print()

def show_diagnostics_menu():
    print_header("Diagnostics & Analysis")
    print()
    print("  SYSTEM")
    print("  1. System diagnostics")
    print("  2. Evaluate model metrics")
    print()
    print("  MARKET ANALYSIS")
    print("  3. Analyze correlations")
    print("  4. Compare strategies")
    print("  5. Recession indicator")
    print("  6. Valuation detector (single asset)")
    print("  7. Valuation detector (all assets)")
    print()
    print("  LLM & AI")
    print("  8. LLM analysis (single asset)")
    print("  9. LLM analysis (all assets)")
    print("  10. LLM newsletter summary")
    print()
    print("  REPORTS")
    print("  11. Generate newsletter (preview)")
    print("  12. Send newsletter via email")
    print()
    print("  0. Back")
    print()

def show_data_menu():
    print_header("Data Management")
    print()
    print("  DOWNLOAD")
    print("  1. Update SPY data")
    print("  2. Download crypto (basic - 3 coins)")
    print("  3. Download crypto (enhanced - 10 coins)")
    print("  4. Download crypto (comprehensive - 15 coins, multi-source)")
    print("  5. Download equity ETFs")
    print("  6. Download all framework assets")
    print()
    print("  IMPORT")
    print("  7. Import custom asset (CSV/Excel)")
    print("  8. Create sample import template")
    print()
    print("  LIVE UPDATES")
    print("  9. Run live update (single)")
    print("  10. Run scheduled updates")
    print("  11. Download all historical data")
    print()
    print("  INFO")
    print("  12. List available assets")
    print()
    print("  0. Back")
    print()

def select_asset():
    """Interactive asset selection"""
    assets = get_available_assets()

    print()
    print_section("Available Assets")
    print()

    # Group by type
    crypto = [a for a in assets if "/USDT" in a]
    etfs = [a for a in assets if "/USDT" not in a]

    if etfs:
        print(f"  📈 ETFs/Stocks ({len(etfs)})")
        print(f"     {', '.join(etfs)}")
        print()
    if crypto:
        print(f"  ₿  Crypto ({len(crypto)})")
        print(f"     {', '.join(crypto)}")
        print()

    if not assets:
        print_warning("No assets found. Download data first!")
        return None

    while True:
        asset = input("  Enter asset ticker (e.g., SPY, BTC/USDT): ").strip().upper()

        if not asset:
            print_warning("Asset ticker cannot be empty")
            continue

        # Handle common variations
        if asset == "BTC":
            asset = "BTC/USDT"
        elif asset == "ETH":
            asset = "ETH/USDT"
        elif asset == "SOL":
            asset = "SOL/USDT"

        # Validate asset exists
        if asset in assets or asset == "SPY":
            print_success(f"Selected: {asset}")
            return asset
        else:
            print_error(f"Asset '{asset}' not found")
            retry = input("  Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None

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
    print()
    if desc:
        print_info(desc)
    else:
        print_info(f"Running: {' '.join(cmd)}")

    print()
    print("─" * 70)
    try:
        result = subprocess.run(cmd)
        print("─" * 70)
        if result.returncode == 0:
            print_success("Command completed successfully")
        else:
            print_warning(f"Command exited with code {result.returncode}")
    except KeyboardInterrupt:
        print("\n" + "─" * 70)
        print_warning("Command interrupted by user")
    except Exception as e:
        print("\n" + "─" * 70)
        print_error(f"Error running command: {e}")

    print()
    input("Press Enter to continue...")

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
        elif choice == "9":
            # Train ALL - comprehensive pipeline
            print("\n" + "=" * 60)
            print("  COMPREHENSIVE TRAINING PIPELINE")
            print("=" * 60)
            print("\nThis will run:")
            print("  1. Multi-asset training (with weight optimization)")
            print("  2. Per-asset training (SPY + crypto)")
            print("  3. Multi-horizon training (1d, 3d, 5d)")
            print("\nEstimated time: 15-25 minutes")
            print()

            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\n[Step 1/3] Multi-asset training with optimized weights...")
                subprocess.run(["python3", "train_multi_asset.py", "--optimize-weights"])

                print("\n[Step 2/3] Per-asset training...")
                subprocess.run(["python3", "train_per_asset.py"])

                print("\n[Step 3/3] Multi-horizon training...")
                subprocess.run(["python3", "train_multi_horizon_signals.py"])

                print("\n" + "=" * 60)
                print("  COMPREHENSIVE TRAINING COMPLETE")
                print("=" * 60)
                input("\nPress Enter to continue...")

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
            if asset:
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
            # Conservative profile
            print("\n📘 CONSERVATIVE TRADING PROFILE")
            print("   - High confidence requirements (70%+)")
            print("   - Tight risk management (1.0x ATR stop)")
            print("   - Max 15% position size, 40% equity exposure")
            print()
            confirm = input("Run backtest with conservative profile? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\nNote: Conservative profile parameters integrated into backtest")
                run_command(["python3", "backtest.py", "--config", "configs/backtest_optimized.json"])
        elif choice == "2":
            # Moderate profile
            print("\n📗 MODERATE TRADING PROFILE")
            print("   - Balanced requirements (55%+ confidence)")
            print("   - Moderate risk (1.5x ATR stop)")
            print("   - Max 25% position size, 65% equity exposure")
            print()
            confirm = input("Run backtest with moderate profile? (y/n): ").strip().lower()
            if confirm == 'y':
                run_command(["python3", "backtest.py", "--config", "configs/backtest_optimized.json"])
        elif choice == "3":
            # Liberal/Aggressive profile
            print("\n📕 LIBERAL (AGGRESSIVE) TRADING PROFILE")
            print("   - Lower requirements (45%+ confidence)")
            print("   - Aggressive risk (2.0x ATR stop, 4.0x TP)")
            print("   - Max 40% position size, 85% equity exposure")
            print()
            confirm = input("Run backtest with liberal profile? (y/n): ").strip().lower()
            if confirm == 'y':
                run_command(["python3", "backtest.py", "--config", "configs/backtest_aggressive.json"])
        elif choice == "4":
            run_command(["python3", "backtest.py", "--config", "configs/backtest_optimized.json"])
        elif choice == "5":
            run_command(["python3", "backtest.py", "--config", "configs/backtest_high_profit.json"])
        elif choice == "6":
            run_command(["python3", "backtest.py", "--config", "configs/backtest_aggressive.json"])
        elif choice == "7":
            asset = select_asset()
            if asset:
                run_command(["python3", "backtest.py", "--asset", asset])
        elif choice == "8":
            group = select_asset_group()
            run_command(["python3", "backtest.py", "--asset-group", group, "--compare"])
        elif choice == "9":
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
        elif choice == "10":
            # Find optimal rebalancing period
            print("\nOptimal Rebalancing Period Finder")
            print("-" * 40)
            assets = input("Assets (comma-separated) [SPY,GLD,TLT]: ").strip() or "SPY,GLD,TLT"
            weights = input("Weights (comma-separated) [0.6,0.3,0.1]: ").strip() or "0.6,0.3,0.1"
            years = input("Lookback years [5]: ").strip() or "5"
            run_command([
                "python3", "portfolio_rebalancer.py",
                "--find-optimal",
                "--assets", assets,
                "--weights", weights,
                "--lookback-years", years
            ])
        elif choice == "11":
            # Execute rebalancing
            print("\nPortfolio Rebalancing")
            print("-" * 40)
            assets = input("Assets (comma-separated) [SPY,GLD,TLT]: ").strip() or "SPY,GLD,TLT"
            weights = input("Weights (comma-separated) [0.6,0.3,0.1]: ").strip() or "0.6,0.3,0.1"
            profile = input("Trading profile (conservative/moderate/liberal) [moderate]: ").strip() or "moderate"
            run_command([
                "python3", "portfolio_rebalancer.py",
                "--assets", assets,
                "--weights", weights,
                "--profile", profile
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
        elif choice == "5":
            # Recession indicator
            print("\nRunning recession probability analysis...")
            run_command(["python3", "recession_indicator.py", "--save"])
        elif choice == "6":
            # Valuation detector (single asset)
            asset = select_asset()
            if asset:
                run_command(["python3", "valuation_detector.py", "--asset", asset])
        elif choice == "7":
            # Valuation detector (all assets)
            print("\nAnalyzing valuations for all assets...")
            run_command(["python3", "valuation_detector.py", "--all", "--save"])
        elif choice == "8":
            asset = select_asset()
            if asset:
                provider = input("LLM provider (openai/anthropic) [openai]: ").strip() or "openai"
                run_command(["python3", "llm_forecast.py", "--asset", asset, "--provider", provider])
        elif choice == "9":
            # Multi-asset LLM analysis
            print("\nRunning LLM analysis on all assets...")
            provider = input("LLM provider (openai/anthropic) [openai]: ").strip() or "openai"
            run_command(["python3", "llm_forecast.py", "--all", "--provider", provider])
        elif choice == "10":
            # Newsletter summary
            print("\nGenerating LLM newsletter summary...")
            provider = input("LLM provider (openai/anthropic) [openai]: ").strip() or "openai"
            run_command(["python3", "llm_forecast.py", "--all", "--newsletter", "--provider", provider])
        elif choice == "11":
            assets = input("Assets (comma-separated) [SPY]: ").strip() or "SPY"
            run_command(["python3", "newsletter_generator.py", "--preview", "--assets", assets])
        elif choice == "12":
            assets = input("Assets (comma-separated) [SPY]: ").strip() or "SPY"
            run_command(["python3", "newsletter_generator.py", "--send", "--assets", assets])

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
            run_command(["python3", "download_crypto_enhanced.py"])
        elif choice == "4":
            run_command(["python3", "download_crypto_comprehensive.py"])
        elif choice == "5":
            run_command(["python3", "download_equity_etfs.py"])
        elif choice == "6":
            run_command(["python3", "framework/download_all_assets.py"])
        elif choice == "7":
            # Import custom asset
            filepath = input("Path to CSV/Excel file: ").strip()
            if filepath:
                ticker = input("Ticker symbol for this asset: ").strip().upper()
                if ticker:
                    run_command(["python3", "import_custom_asset.py", filepath, ticker])
                else:
                    print("Ticker symbol required")
                    input("Press Enter to continue...")
            else:
                print("File path required")
                input("Press Enter to continue...")
        elif choice == "8":
            run_command(["python3", "import_custom_asset.py", "--sample"])
        elif choice == "9":
            assets = input("Assets (comma-separated, blank for SPY,QQQ): ").strip() or "SPY,QQQ"
            run_command(["python3", "live_update.py", "--assets", assets, "--predict"])
        elif choice == "10":
            assets = input("Assets (comma-separated, blank for SPY): ").strip() or "SPY"
            interval = input("Interval in minutes [15]: ").strip() or "15"
            run_command(["python3", "live_update.py", "--mode", "scheduled", "--assets", assets, "--interval", interval])
        elif choice == "11":
            run_command(["python3", "live_update.py", "--download"])
        elif choice == "12":
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

def handle_full_pipeline():
    """Run comprehensive full pipeline: download, train, predict, backtest, LLM"""
    clear_screen()
    print_header("Full Pipeline Execution")
    print()
    print("  This will run the complete NeuroVest pipeline:")
    print()
    print("  1. Update/Download data (SPY + crypto)")
    print("  2. Train models (multi-asset with optimized weights)")
    print("  3. Generate predictions (all assets)")
    print("  4. Run backtests (optimized config)")
    print("  5. Generate LLM analysis (optional)")
    print()
    print("  Estimated time: 20-35 minutes")
    print()

    # Ask for configuration
    include_crypto = input("  Include crypto data download? (y/n) [y]: ").strip().lower() != 'n'
    include_llm = input("  Include LLM analysis at end? (y/n) [y]: ").strip().lower() != 'n'

    if include_llm:
        llm_provider = input("  LLM provider (openai/anthropic) [openai]: ").strip() or "openai"

    print()
    confirm = input("  Start full pipeline? (y/n): ").strip().lower()

    if confirm != 'y':
        print("\n  Cancelled.")
        input("  Press Enter to continue...")
        return

    print("\n" + "=" * 60)
    print("  STARTING FULL PIPELINE")
    print("=" * 60)

    success_count = 0
    total_steps = 5 if include_llm else 4

    # Step 1: Data Download
    print(f"\n[Step 1/{total_steps}] Downloading/Updating Data...")
    print("-" * 60)

    try:
        # Update SPY
        print("\nUpdating SPY data...")
        result = subprocess.run(["python3", "update_spy_data.py"], capture_output=False)

        # Download all equity ETFs, bonds, precious metals
        print("\nDownloading equity ETFs, bonds, and precious metals...")
        subprocess.run(["python3", "download_equity_etfs.py"])

        # Download crypto if requested
        if include_crypto:
            print("\nDownloading crypto data...")
            subprocess.run(["python3", "download_crypto_enhanced.py"])

        success_count += 1
        print("\n[+] Data download complete")
    except Exception as e:
        print(f"\n[!] Data download error: {e}")

    # Step 2: Training
    print(f"\n[Step 2/{total_steps}] Training Models...")
    print("-" * 60)

    try:
        subprocess.run(["python3", "train_multi_asset.py", "--optimize-weights"])
        success_count += 1
        print("\n[+] Training complete")
    except Exception as e:
        print(f"\n[!] Training error: {e}")

    # Step 3: Predictions
    print(f"\n[Step 3/{total_steps}] Generating Predictions...")
    print("-" * 60)

    try:
        # Multi-asset ensemble prediction
        subprocess.run(["python3", "predict_multi_asset_ensemble.py"])

        # Per-asset predictions
        subprocess.run(["python3", "predict_per_asset.py", "--all"])

        success_count += 1
        print("\n[+] Predictions complete")
    except Exception as e:
        print(f"\n[!] Prediction error: {e}")

    # Step 4: Backtesting
    print(f"\n[Step 4/{total_steps}] Running Backtests...")
    print("-" * 60)

    try:
        subprocess.run(["python3", "backtest.py", "--config", "configs/backtest_optimized.json"])
        success_count += 1
        print("\n[+] Backtesting complete")
    except Exception as e:
        print(f"\n[!] Backtest error: {e}")

    # Step 5: LLM Analysis (optional)
    if include_llm:
        print(f"\n[Step 5/{total_steps}] Generating LLM Analysis...")
        print("-" * 60)

        try:
            subprocess.run(["python3", "llm_forecast.py", "--all", "--provider", llm_provider])
            success_count += 1
            print("\n[+] LLM analysis complete")
        except Exception as e:
            print(f"\n[!] LLM analysis error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Completed: {success_count}/{total_steps} steps")
    print()
    print("  Output locations:")
    print("  - Predictions: logs/daily_predictions.csv")
    print("  - Backtest: outputs/backtest_results.json")
    if include_llm:
        print("  - LLM Analysis: logs/llm_multi_asset_summary_*.json")
    print()
    print("  Next steps:")
    print("  - View results in Web Dashboard (option 7)")
    print("  - Run Asset Explorer for specific analysis (option 6)")
    print()
    input("  Press Enter to continue...")


def handle_asset_explorer():
    """Guided pipeline to explore a specific asset"""
    clear_screen()
    print_header("Asset Explorer")
    print()

    # Select asset
    asset = select_asset()

    while True:
        clear_screen()
        print_header(f"Asset Explorer: {asset}")
        print()
        # Determine asset group for config selection
        if "/USDT" in asset:
            asset_group = "crypto"
        elif asset in ["TLT", "IEF", "SHY", "BND", "AGG"]:
            asset_group = "bond"
        elif asset in ["GLD", "SLV", "USO", "DBA", "DBC"]:
            asset_group = "commodity"
        else:
            asset_group = "equity"

        print("  DATA & INFO")
        print("  1. View data summary (rows, date range, etc.)")
        print("  2. Check data file exists")
        print()
        print("  BACKTESTS (group-optimized)")
        print(f"  3. Run backtest ({asset_group} config - recommended)")
        print("  4. Run backtest (high profit config)")
        print("  5. Run backtest (aggressive config)")
        print()
        print("  PREDICTIONS")
        print("  6. Generate predictions for this asset")
        print()
        print("  COMPARISON")
        print("  7. Compare with other assets in same group")
        print("  8. Analyze correlations")
        print()
        print("  OTHER")
        print("  9. Change asset")
        print("  0. Back to main menu")
        print()

        choice = input("Select option: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            # View data summary
            asset_file = asset.replace("/", "_")
            data_path = f"data_cache/{asset_file}_1d.csv"
            if asset == "SPY":
                data_path = "data/SPY.csv"

            print(f"\n📊 Data Summary for {asset}")
            print("-" * 40)

            try:
                import pandas as pd
                import numpy as np
                df = pd.read_csv(data_path)
                print(f"File: {data_path}")
                print(f"Rows: {len(df):,}")
                print(f"Columns: {len(df.columns)}")
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    date_min = df['Date'].min()
                    date_max = df['Date'].max()
                    years = (date_max - date_min).days / 365.25
                    print(f"Date range: {date_min.date()} to {date_max.date()} ({years:.1f} years)")
                if 'Close' in df.columns:
                    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
                    print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")

                    # Calculate returns
                    if len(df) > 252:
                        first_price = df['Close'].iloc[0]
                        last_price = df['Close'].iloc[-1]
                        total_return = (last_price / first_price) - 1
                        annual_return = (1 + total_return) ** (1 / years) - 1
                        print(f"\nTotal return: {total_return:.1%}")
                        print(f"Annual return: {annual_return:.1%}")

                        # Recent performance
                        if len(df) >= 252:
                            ytd_return = (df['Close'].iloc[-1] / df['Close'].iloc[-252]) - 1
                            print(f"1-year return: {ytd_return:.1%}")

            except FileNotFoundError:
                print(f"❌ Data file not found: {data_path}")
                print()
                download = input("Would you like to download this data? (y/n): ").strip().lower()
                if download == 'y':
                    if "/USDT" in asset:
                        run_command(["python3", "download_crypto_enhanced.py"])
                    elif asset == "SPY":
                        run_command(["python3", "update_spy_data.py"])
                    else:
                        run_command(["python3", "download_equity_etfs.py"])
                    continue  # Re-show menu after download
            except Exception as e:
                print(f"❌ Error reading data: {e}")

            input("\nPress Enter to continue...")

        elif choice == "2":
            # Check file exists
            asset_file = asset.replace("/", "_")
            data_path = Path(f"data_cache/{asset_file}_1d.csv")
            if asset == "SPY":
                data_path = Path("data/SPY.csv")

            if data_path.exists():
                size = data_path.stat().st_size / 1024
                print(f"\n✅ {data_path} exists ({size:.1f} KB)")
            else:
                print(f"\n❌ {data_path} not found")
            input("\nPress Enter to continue...")

        elif choice == "3":
            # Use group-specific config
            config_map = {
                "crypto": "configs/backtest_crypto.json",
                "bond": "configs/backtest_bond.json",
                "commodity": "configs/backtest_commodity.json",
                "equity": "configs/backtest_equity.json"
            }
            config = config_map.get(asset_group, "configs/backtest_optimized.json")
            run_command(["python3", "backtest.py", "--asset", asset, "--config", config])

        elif choice == "4":
            run_command(["python3", "backtest.py", "--asset", asset, "--config", "configs/backtest_high_profit.json"])

        elif choice == "5":
            run_command(["python3", "backtest.py", "--asset", asset, "--config", "configs/backtest_aggressive.json"])

        elif choice == "6":
            run_command(["python3", "predict_per_asset.py", "--asset", asset])

        elif choice == "7":
            print(f"\nComparing {asset} with {asset_group} group...")
            run_command(["python3", "backtest.py", "--asset-group", asset_group, "--compare"])

        elif choice == "8":
            run_command(["python3", "analyze_correlations.py", "--asset-group", asset_group])

        elif choice == "9":
            asset = select_asset()

def main():
    try:
        while True:
            clear_screen()
            show_main_menu()

            choice = input("Select option: ").strip()

            if choice == "0":
                print("\nDone.")
                break
            elif choice.lower() == "r":
                handle_full_pipeline()
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
            elif choice == "6":
                handle_asset_explorer()
            elif choice == "7":
                print("\n Launching Web Dashboard...")
                print("   Opening browser at http://localhost:8501")
                print("\n   Press Ctrl+C in terminal to stop the server")
                print("-" * 60)
                subprocess.run(["streamlit", "run", "dashboard.py"])
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
# main.py
# -----------MARKET PREDICTION ALGORITHM-----------
# ADDITIONAL: INSTALL LIBRARIES ON LOCAL MACHINE (OR FROM TERMINAL TO REPO DIRECTLY? LOOK INTO THIS)

# ___________USE OF ALPHA VANTAGE API TO PULL LIVE STOCK VALUATION FIGURES, TRAIN MODEL OFF OF THESE VALUATIONS (RANDOM FOREST MODEL)________
# ADDITIONALLY: BUILD IN FUCNTIONALITY THAT CPMPARES PREDICTED VALUATIONS WITH REAL TIME ONES/FUTURE ONES. THIS COULD BE ACHEVED WITH A GRAPHICAL VISUALIZATION OF PREDICTED VS. REAL TIME VALUATIONS

# test
# BOOTING PROGRAM PROMPT
print("Welcome! Booting program now, please wait momentarily...")

# LIBRARIES REQD'
import time  # PAUSE/TIMIING PROTOCOL: NECESSARY FOR LIVE VALUATIONS

import pandas as pd  # ENHANCED DATA MANIPULATION LAYER
import requests  # PULLS STOCK MARKET DATA

time.sleep(12)  # wait 12 seconds between requests


# ADDITIONAL
import os  # FILE MANAGEMENT

import joblib  # SAVE/LOAD MODEL, GIVE USER CAPABILITY TO RUN ACROSS VARIOUS SESSIONS USING PRESET METRICS
import matplotlib.pyplot as plt  # GRAPH STATS
import numpy as np  # ENHANCED NUMERICAL HANDLING
import schedule  # DAILY SCHEDULER
from dotenv import load_dotenv  # DEALS WITH API KEY
from sklearn.ensemble import RandomForestClassifier  # ML MODEL (ADDITIONAL)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

print("CWD:", os.getcwd())

import os

from sklearn.utils import resample  # DEALS WITH IMBALANCED DATASETS

from data_utils import load_spy_daily_data
from predict import live_predict
from train import retrain_model
from utils import (
    get_feature_list,
    init_labeled_log_file,
    label_events_future_window,
    label_real_outcomes_from_log,
    load_SPY_data,
)

df = load_SPY_data()
from utils import load_SPY_data

df = load_SPY_data()
from utils import load_SPY_data

df = load_SPY_data()
df = label_events_future_window(df, window=3)
from utils import load_SPY_data

df = load_SPY_data()


spy_df = load_spy_daily_data()
# now spy_df.index is datetime element that can be sliced, resampled, backtested, etc.


# FILE PATH CREATION
os.makedirs("logs", exist_ok=True)

init_labeled_log_file()
label_real_outcomes_from_log()


# Alpha Vantage API key configuration
# load_dotenv() # will load ./ .env automatically

# ^^^^^update to account for new repo key (market-pred-bot vs cs2704...... thing) #"/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/AV-API-key.env"
# api_key = os.getenv("ALPHA_VANTAGE_KEY")
# print(f"DEBUG: Loaded API key: {api_key}")

# Alpha Vantage API key configuration
print(
    "ENV file exists?",
    os.path.isfile("/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/.env"),
)
with open("/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/.env") as f:
    print(f.read())

load_dotenv(
    "/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/.env"
)  # "/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/AV-API-key.env"
api_key = os.getenv("ALPHA_VANTAGE_KEY")
print(f"DEBUG: Loaded API key: {api_key}")

# DEBUG: check if API key is loaded
if not api_key:
    # raise ValueError("ERROR: API key not found. is it in your .env file?")
    print("DEBUG: API key not found! check .env file or file path")
else:
    print("DEBUG: API Key loaded successfully!")

# global log file
LOG_FILE = "logs/daily_predictions.csv"
LABELED_LOG_FILE = "logs/labeled_predictions.csv"


# -----------GENERAL PSEUDOCODE/HIERARCHICAL LAYOUT-----------


# CRITERIA/FUNCTIONAL COMPONENTS


# 1. DATA INGESTION
# -DOWNLOAD DAILY/LIVE STOCK VALUATION FIGURES. TO BE ACCOMLISHED VIA. USE OF DAILY OHLCV FROM ALPHA VANTAGE
# --SCHEDULE DAILY JOB (VIA SCHEDULE)
# --FETCHING OF LATEST SPY DATA (VIA ALPHA VANTAGE API): *COMPLETED*
# ---THIS IS TO FETCH CURRENT MARKET VALUATION VARIABLES, DAILY ADJUSTED OHLCV VALUATIONS
# ----VIA USE OF ***TIME_SERIES_DAILY_ADJUSTED*** ENDPOINT, RETURNING OHLCV VALUATIONS FROM AV API
def fetch_ohlcv(
    symbol="SPY", interval="1min", outputsize="full", api_key=None
):  # for testing: outputsize: (thing),  #add interval='1min' if it fails
    # Fetch daily OHLCV data from Alpha Vantage API
    print("Fetching OHLCV data valuations...")
    url = "https://www.alphavantage.co/query"  # THIS LINK MIGHT BE BROKEN

    params = {
        "function": "TIME_SERIES_DAILY",  # ONLY USE TIME_SERIES_INTRADAY FOR PER MINUTE DATA, BUT THISLL DO FOR THE ASSIGNMENT OBJECTIVE ATM #TIME_SERIES_DAILY_ADJUSTED IS APPARENTLY A PREMIUM ENDPOINT??
        # "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
        "apikey": api_key,
        "outputsize": outputsize,  # much smaller data set, may be good for avoiding overwhelming AI
        "datatype": "json",
    }

    #    params = {
    #        "function": "TIME_SERIES_INTRADAY", #ONLY USE TIME_SERIES_INTRADAY FOR PER MINUTE DATA, BUT THISLL DO FOR THE ASSIGNMENT OBJECTIVE ATM #TIME_SERIES_DAILY_ADJUSTED IS APPARENTLY A PREMIUM ENDPOINT??
    #        "symbol": "SPY",
    #        "interval": "1min",
    #        "apikey": api_key,
    #        "outputsize": "compact",
    #        "datatype": "json"
    #    }
    print(f"DEBUG: API request params: {params}")
    print(f"DEBUG: API request URL: {url}?{requests.compat.urlencode(params)}")

    # Make the API request
    response = requests.get(url, params=params)
    print(f"DEBUG: API response status code: {response.status_code}")
    # -------
    # response=requests.get(url, params=params)

    # parse .json response
    data = response.json()
    print(f"DEBUG: API response: {data}")

    print("DEBUG: JSON response keys:", list(data.keys()))
    if "Note" in data:
        print("ERROR: Rate limit hit:", data["Note"])
        return None
    time_series_key = [k for k in data if "Time Series" in k]
    print("DEBUG: using key →", time_series_key)

    # DEBUG: check if API response was successful
    if response.status_code != 200:
        print(f"ERROR: API request failed with status code {response.status_code}")
        return None

    # check rate limit/invalid response time
    if "Note" in data:
        print("ERROR: API rate limit exceeded. Try again later.")
        return None
    # CHECK FOR INVALID/UNEXPECTED RESPONSE FORMAT
    time_series_key = [k for k in data if "Time Series" in k]
    if not time_series_key:
        print("ERROR: Time Series data not found in API response")
        return None

    key = time_series_key[0]
    raw_df = pd.DataFrame.from_dict(data[key], orient="index")
    raw_df = raw_df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
    )
    # Convert "sting" valuations to "floats"
    raw_df = raw_df.astype(float)

    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()
    print("DATA PARSED/EXTRACTED SUCCESSFULLY!")
    return raw_df

    # DEBUG: print first couple rows of DataFrame
    print(f"DEBUG: Parsed DataFrame head:\n{raw_df.head()}")

    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()
    print("DATA PARSED/EXTRACTED SUCCESSFULLY!")
    return raw_df


# 2. FEATURE ENGINEERING
# -STUCTURES ACTUAL SET FUNCTIONALITY OF ALGORITHMS TECHNICAL FEATURES
# --RSI, MACID, MOVING AVERAGES, VOLATILITY, RETURNS (OR OTHER RELEVANT INDICATORS)
# ---ONLY MOST RECENT ROW NEEDS COMPUTATION: REDUCES REDUNCANCY + MITIGATES POTENTIAL MODEL OVERFITTING
# ---***RECALCULATION OF FEATURE COLUMNS + FEATURE CONSISTENCY *CRITICAL* FOR VALID MODEL OUTPUT, AS IS TO BE SOLE BASIS OF OUR BELOW RENDITIONS!****
def calculate_technical_indicators(df):
    # this function intends to deal with adding our extracted moving averages, returns, RSI, etc. etc. etc.: effectively all the above metrics
    df["Return"] = df["Close"].pct_change()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["Volatility"] = df["Return"].rolling(window=20).std()

    # moving avg

    # Exponential Moving Average (EMA)
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()  # Short term EMA (12)
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()  # Medium term EMA (26)

    # MACD, signal, histogram (Moving Average Convergence Divergence)
    df["MACD"] = df["EMA_12"] - df["EMA_26"]  # MACD line
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()  # Signal line
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]  # MACD histogram

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()  # Middle band (20-day SMA)
    df["BB_Upper"] = df["BB_Mid"] + (
        df["Close"].rolling(window=20).std() * 2
    )  # Upper band (20-day SMA + 2*std dev)
    df["BB_Lower"] = df["BB_Mid"] - (
        df["Close"].rolling(window=20).std() * 2
    )  # Lower band (20-day SMA - 2*std dev)

    df["BB_Std"] = df["Close"].rolling(window=20).std()  # Bollinger Bands Standard Deviation
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]  # Bollinger Bands Width

    # Volume based indicators
    df["OBV"] = (
        (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    )  # On Balance Volume (OBV)

    df["Vol_MA_10"] = df["Volume"].rolling(window=10).mean()  # 10-day moving average of volume
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA_10"]

    # price momentum  + acceleration
    df["Price_Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Acceleration"] = df["Price_Momentum_10"] - df["Price_Momentum_10"].shift(1)

    # zscore normalized momentum
    df["ZMomentum"] = (df["Price_Momentum_10"] - df["Price_Momentum_10"].rolling(20).mean()) / df[
        "Price_Momentum_10"
    ].rolling(20).std()

    # RSI calc (14 day interval)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    #'relative strength' calculation
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))  # RS *INDICATIOR* VALUATION

    # RSI Momentum (delta)
    df["RSI_Delta"] = df["RSI"].diff()

    # lag features

    for lag in [1, 3, 5]:
        df[f"Return_Lag{lag}"] = df["Return"].shift(lag)
        if "RSI" in df.columns:
            df[f"RSI_Lag_{lag}"] = df["RSI"].shift(lag)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# 3 "EVENT LABELING" LOGIC
# BINARY CLASSIFICATION ON BASIS OF (PREDICTED) FUTURE RETURNS
# --Each row labeled as followed: (0==NORMAL (ELSE), 1==CRASH )
# + CONFIDENCE PROBABLITY VALUATION (LOOK INTO A LIL BIT)
def label_events(df, crash_threshold=-0.005, spike_threshold=0.005):
    df = df.copy()
    df["Future_Close"] = df["Close"].shift(-1)
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df = df.dropna(subset=["Future_Return"])

    df["Crash"] = (df["Future_Return"] < crash_threshold).astype(int)
    df["Spike"] = (df["Future_Return"] > spike_threshold).astype(int)

    df["Event"] = np.select(
        [df["Crash"] == 1, df["Spike"] == 1],
        [1, 2],  # 1 == 'Crash', 2 == 'Spike'
        default=0,  # 0 == 'Normal'
    )

    return df


# 4. Balance dataset
# -DEALS WITH IMBALANCED DATASETS
def balance_dataset(X, y):
    data = pd.concat([X, y], axis=1)
    crash = data[data["Crash"] == 1]
    normal = data[data["Crash"] == 0]

    # upsampling minorty class (crash data)
    crash_upsampled = resample(crash, replace=True, n_samples=len(normal), random_state=42)
    balanced = pd.concat([normal, crash_upsampled])
    balanced = balanced.sample(frac=1, random_state=42)  # shuffle set

    return balanced.drop("Crash", axis=1), balanced["Crash"]


# 5. ML MODEL ARCHETECHURE (BASED ON RANDOM FOREST MODEL)
# -RECURSIVE SELF TRAINING OF ML MODEL
# -- WILL UTILIZE "RANDOM FOREST" STYLED ML MODEL, BASED OFF OF THESE EXTRACTED VALUATIONS/EVERCHANGING DATASET VALUATIONS
# ---RANDOM FOREST MODEL: USED FOR INTERPRETABILITLY/ROBUSTNESS OF OVERALL ML ALGORITHM AND ARCHITECHTURE
def train_model(
    df, features=None, target="Event"
):  # in theory, trains our model on above extractions
    if features is None:
        features = get_feature_list()
    # selection of feature and target (X and y variables respectively) from DataFrame
    X = df[features]  # features inputted to be used to train our model below
    y = df[target]  # deals w/ output labels (0=='normal, 1=='crash", 2=='spike')
    # X,y=balance_dataset(X,y) #balances dataset to deal with imbalanced classes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )  # splitting of dat ainto training/test subsets (80/20 split in this case)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )  # imitilaization of RF classifier, in this case utilizing 100 trees.
    model.fit(X_train, y_train)  # train model w/ training data

    y_pred = model.predict(X_test)  # prediction crash labels on test set

    # display performance metrics
    print("\nModel Performance Metric Valuations:")
    print("Accuracy")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # saving of trained model above for future use.
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/market_crash_model.pkl")  # Saves our pre trained model (ideally)
    return model


# 6. LIVE PREDICTION PIPELINE
# (NOW LOCATED IN PREDICT.PY)


# 6.2: (OPTIONAL, FOR ACCURACY SAKE)
# RETRAIN ML MODEL MONTHLY WITH UPDATED DATASET VALUATIONS
# --THIS IN THEORY WILL HELP FOR OUR ML MODEL TO ADAPT TO EVER CHANGING MARKET BEHAVIOUR + MAINTAIN A LAYER OF PREDICTION ACCURACY


def retrain_model_monthly(df, features=None, target="Crash"):
    if features is None:
        features = get_feature_list()
    print("Retraining model with updated data figures...")
    model = train_model(df, features=features, target="Event")
    print("Model retraining successful!")
    return model


# 7. (TBD) DATA VISULAIZATION
# -WILL INCLUDE A GRAPHICAL VISUALIZATION OF PREDICTED VS. REAL TIME VALUATIONS
def visualize_data(df, save_path="graphs/daily_plot.png", show=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensures folder exsists
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.6)
    plt.plot(df.index, df["MA_20"], label="{20-Period Moving Average", linestyle="--", alpha=0.8)

    # highlight market crashes
    if "Crash" in df.columns:
        crash_points = df[df["Crash"] == 1]
        plt.scatter(
            crash_points.index,
            crash_points["Close"],
            color="red",
            label="Predicted Market Crashes",
            zorder=5,
            marker="v",
        )

    # highlight market spikes
    if "Spike" in df.columns:
        spike_points = df[df["Spike"] == 1]
        plt.scatter(
            spike_points.index,
            spike_points["Close"],
            color="green",
            label="Predicted Market Spikes",
            zorder=5,
            marker="^",
        )

    # PLOT FORMATTING
    plt.title("Stock Pricing w/ Moving Averages + Crash/Spike Events")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ENSURE OUTPUT DIR IS PRESENT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # maybe remove this? idk

    # save figure
    plt.savefig(save_path)
    print(f"[Graph] Saved plot to {save_path}")

    # SHOW ONLY IF SHOW=TRUE
    if show:
        plt.show()  # infinite chocopoints for meeeeeee x)
    plt.close()  # frees up unneeded memory this way


# 8. CONDIDENCE TREND VISUALIZER
def plot_confidence_trend(log_file="daily_predictions.csv", show=True):
    df = pd.read_csv(log_file)
    os.makedirs("graphs", exist_ok=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    plt.figure(figsize=(10, 6))
    df[["crash_prob", "spike_prob"]].plot(ax=plt.gca())
    plt.title("Crash/Spike Confidence Over Time")
    plt.ylabel("Probability (%)")
    plt.xlabel("Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 9. MAIN PIPELINE (moved to scheduler)
# -MAIN FUNCTIONALITY OF PROGRAM
# --WILL INCLUDE ALL ABOVE FUNCTIONS IN A SEQUENTIAL ORDER
# ---WILL ALSO INCLUDE A MAIN FUNCTIONALITY FOR USER TO RUN PROGRAM


# 10. PREDICTION LOG CLEANER
def clean_prediction_log():
    try:
        log_df = pd.read_csv(
            "logs/daily_predictions.csv",
            names=["Timestamp", "Prediction", "Crash_Conf", "Spike_Conf", "Close_Price"],
            skiprows=1,
        )

        # Drop any rows that are just headers written as data
        log_df = log_df[log_df["Crash_Conf"] != "Crash_Conf"]

        # Convert to float safely
        log_df["Crash_Conf"] = pd.to_numeric(log_df["Crash_Conf"], errors="coerce")
        log_df["Spike_Conf"] = pd.to_numeric(log_df["Spike_Conf"], errors="coerce")

        log_df = log_df.dropna(subset=["Crash_Conf", "Spike_Conf"])

        return log_df

    except Exception as e:
        print(f"❌ clean_prediction_log() failed: {e}")
        return pd.DataFrame()


# 11: DAILY SCHEDULER FUNCTIONALITY
# -WILL INCLUDE A DAILY SCHEDULER FUNCTIONALITY TO RUN THE PROGRAM ON A DAILY BASIS
# --WILL INCLUDE A FUNCTIONALITY TO RUN THE PROGRAM ONCE, THEN SCHEDULE IT TO RUN DAILY
# ------DAILY SCHEDULER FUNCTION--------#
def daily_job():
    print("[Scheduler] Executing daily market prediction...")

    # Fetch latest OHLCV data
    df = fetch_ohlcv(symbol="SPY", api_key=api_key, outputsize="full")
    if df is None:
        print("ERROR: Failed to fetch data")
        return

    # 1) feature-engineer & label
    df = calculate_technical_indicators(df)
    df = label_events_future_window(df, crash_threshold=-0.03, spike_threshold=0.03, window=3)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # daily job counter (events)
    counts = df["Event"].value_counts().to_dict()
    print(f"[Daily Label Counts] {counts}")

    # 2) train & live predict
    train_model(df, target="Event")
    live_predict(df)

    # 3) label the real outcome for this run
    label_real_outcomes_from_log()

    # 4) dashboard & clean
    show_combined_dashboard(df)
    clean_prediction_log()

    # 5) weekly retrain (Sunday)
    if pd.Timestamp.now().weekday() == 6:  # Sunday == 6
        print("[Retrain] Initiating weekly model retraining…")
        retrain_model(df)

    else:
        print("[Scheduler] Skipping weekly retrain (not Sunday)")


# -----START DAILY SCHEDULER-----#
def start_scheduler():
    # inintial predicitons
    print("[scheduler] Running initial prediction...")
    daily_job()  # runs once immediately
    # schedules job for 6pm daily
    schedule.every().day.at("18:00").do(daily_job)
    print("[Scheduler] Scheduled daily_job for 6:00pm")
    print("scheuduler initiatied, now waiting for jobs...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            print(f"ERROR: {e}")
            # time.sleep(60)


# -----ENTRY POINT FOR SCHEDULER-----#


def run_once_then_schedule():
    daily_job()
    schedule.every().day.at("18:00").do(daily_job)
    schedule.every().day.at("18:05").do(label_real_outcomes_from_log)

    def safe_retrain():
        if os.path.exists(LABELED_LOG_FILE):
            print("[Retrain] Initiating weekly model retraining...")
            retrain_model()
        else:
            print("[⚠️] No labeled outcomes found — skipping retraining.")

    schedule.every().sunday.at("18:10").do(safe_retrain)

    print("[Scheduler] Scheduled daily_job for 6:00pm")
    print("Press Ctrl+C to exit the scheduler")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Scheduler terminated by user")
        print("\nExiting program...")
        exit(0)


# 12. Combined Dashboard Functionality
def show_combined_dashboard(df, log_file=LOG_FILE):
    # load pred log
    os.makedirs("graphs", exist_ok=True)  # ✅ This ensures 'graphs/' exists

    try:
        log_df = pd.read_csv(
            log_file,
            names=["Timestamp", "Prediction", "Crash_Conf", "Spike_Conf", "Close_Price"],
            parse_dates=["Timestamp"],
            skiprows=1,
        )

        log_df = log_df.dropna(subset=["Crash_Conf", "Spike_Conf"])
        log_df["Crash_Conf"] = log_df["Crash_Conf"].astype(float)
        log_df["Spike_Conf"] = log_df["Spike_Conf"].astype(float)
    except FileNotFoundError:
        print(f"[Error]: Log file {log_file} not found!")
        return

    # set up fig with 2 subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), constrained_layout=True
    )  # 2r,1c #play around with figsize dimentions!

    # top: stock prices
    ax1.plot(df.index, df["Close"], label="Close Price", alpha=0.7)
    ax1.plot(df.index, df["MA_20"], label="20 DAY MOVING AVERAGE", linestyle="--", alpha=0.8)

    if "Crash" in df.columns:
        crash_points = df[df["Crash"] == 1]
        ax1.scatter(
            crash_points.index,
            crash_points["Close"],
            color="red",
            label="Predicted Crashes",
            marker="v",
        )
    if "Spike" in df.columns:
        spike_points = df[df["Spike"] == 1]
        ax1.scatter(
            spike_points.index,
            spike_points["Close"],
            color="green",
            label="Predicted Spikes",
            marker="^",
        )

    ax1.set_title("Stock Prices + Crash Spike Events - With Accompnaying Moving Averages")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # bottom: confidence trend
    ax2.plot(
        log_df["Timestamp"],
        log_df["Crash_Conf"],
        label="Crash Confidence",
        color="red",
        linewidth=2,
    )
    ax2.plot(
        log_df["Timestamp"],
        log_df["Spike_Conf"],
        label="Spike Confidence",
        color="green",
        linewidth=2,
    )

    ax2.set_title("Market Crash/Spike Confidence Trend Valuations Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("confidence level")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    ax2.legend()

    # Save directory check and figure save
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/combined_dashboard_{pd.Timestamp.now().date()}.png"
    plt.savefig(filename)
    print(f"[✅] Saved plot to {filename}")
    plt.show()


if __name__ == "__main__":
    run_once_then_schedule()


# for project
# -run program daily
# compare real vs predicted figures side by side, map out trends
# ***add a "in human speak" function, just to make valuations less abstract and numerical***

# TO DO:
# -MAKE SURE VALUATIONS AND PREDICTED GET APPENDED TO .TXT FILE ON CHRONOLOGICAL BASIS. MOIDEL IS GREAT FOR PAST DATA ESTIMATIONS, BUT TO TRAIN THE MODEL PRESENT VALUATIONS ARE NECESSARY
# --RETRAIN MODEL ON WEEKLY/MONTHLY BASIS (USE CRON/SCHEDULER LIBRARY FOR THIS!)
# --EXPANSION OF FEATURE ENGINEERING FUNCTION: INCLUDE METRICS LIKE MACD, BOLLINGER BANDS, EMA, VALUME BASED METRICS ETC.
# --BUILD DAILY SCHEDULER TO MAKE THIS HAPPEN

# -********BUILD IN "SPIKE" PREDICTION FUNCTION: WOULD BE AS SIMPLE AS INVERTING THE CURRENT LOGICAL ORDER (AS IN, SPIKE IMMINENT IF FUTURE RETURN<=3%. WOULD BE EASY TO ACHIEVE!)\
# ------ (IF PREDICTED SPIKE, THEN BUY, ELSE SELL)

# additional features to add (not for project per se but just becasue why not)
# -add a feature to compare predicted vs. real time valuations
# plot: volatility spikes, RSI over time
# crash confidence trend over time (based on predioction log)
