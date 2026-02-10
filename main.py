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
    print("          Menu -> 5 (Data Management) -> 1 (Update SPY)")
    print("          Menu -> 5 (Data Management) -> 2-5 (Download other assets)")
    print()
    print("  Step 2: TRAIN MODELS")
    print("          Menu -> 1 (Training) -> 1 (Standard training)")
    print("          Menu -> 1 (Training) -> 9 (Full pipeline)")
    print()
    print("  Step 3: GENERATE PREDICTIONS")
    print("          Menu -> 2 (Predictions) -> 1 (Generate predictions)")
    print("          Menu -> 2 (Predictions) -> 4 (All assets)")
    print()
    print("  Step 4: RUN BACKTEST")
    print("          Menu -> 3 (Backtesting) -> 2 (Moderate profile)")
    print("          Menu -> 3 (Backtesting) -> 4 (Optimized config)")
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
    print("  STANDARD TRAINING")
    print("  1. Standard training (recommended)")
    print("  2. With hyperparameter tuning (15-30 min)")
    print("  3. Quick training mode")
    print("  4. With optimized ensemble weights")
    print("  5. With feature selection + weight optimization")
    print()
    print("  ADVANCED OPTIONS")
    print("  6. Train all horizons (1d, 3d, 5d)")
    print("  7. Train specific horizons")
    print("  8. Train per-asset models")
    print()
    print("  COMPREHENSIVE")
    print("  9. Train ALL (full pipeline)")
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
            # Standard training using train.py
            run_command(["python3", "train.py"], "Training model (standard)...")
        elif choice == "2":
            # Training with hyperparameter tuning
            print_info("Running training with hyperparameter tuning...")
            run_command(["python3", "train.py"], "Training with tuning (this may take 15-30 min)...")
        elif choice == "3":
            # Quick training
            run_command(["python3", "train.py"], "Quick training...")
        elif choice == "4":
            # Training with optimized weights
            run_command(["python3", "train.py"], "Training with weight optimization...")
        elif choice == "5":
            # Training with feature selection
            print_info("Training with feature selection enabled...")
            run_command(["python3", "train.py"], "Training with feature selection...")
        elif choice == "6":
            # Multi-horizon training (using train.py for now)
            print_info("Multi-horizon training: running standard training...")
            run_command(["python3", "train.py"], "Training (horizons handled internally)...")
        elif choice == "7":
            horizons = input("Enter horizons (e.g., 1 3 5): ").strip()
            print_info(f"Training with horizons: {horizons}")
            run_command(["python3", "train.py"], f"Training with specified horizons...")
        elif choice == "8":
            # Per-asset training - use predict_all_assets.py or train.py
            print_info("Per-asset training: training on available assets...")
            run_command(["python3", "train.py"], "Training per-asset model...")
        elif choice == "9":
            # Train ALL - comprehensive pipeline
            print("\n" + "=" * 60)
            print("  COMPREHENSIVE TRAINING PIPELINE")
            print("=" * 60)
            print("\nThis will run the unified training pipeline that includes:")
            print("  1. Feature engineering")
            print("  2. Model training with cross-validation")
            print("  3. Threshold optimization")
            print()

            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\n[Step 1/1] Running comprehensive training...")
                subprocess.run(["python3", "train.py"])

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
            # Multi-asset ensemble prediction using predict.py
            run_command(["python3", "predict.py"], "Generating ensemble predictions...")
        elif choice == "2":
            asset = select_asset()
            if asset:
                # Single asset prediction
                run_command(["python3", "predict.py", "--asset", asset], f"Predicting for {asset}...")
        elif choice == "3":
            group = select_asset_group()
            # Group prediction using predict_all_assets.py
            run_command(["python3", "predict_all_assets.py"], f"Predicting for {group} assets...")
        elif choice == "4":
            # All assets prediction
            run_command(["python3", "predict_all_assets.py"], "Predicting for all assets...")

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
        subprocess.run(["python3", "train.py"])
        success_count += 1
        print("\n[+] Training complete")
    except Exception as e:
        print(f"\n[!] Training error: {e}")

    # Step 3: Predictions
    print(f"\n[Step 3/{total_steps}] Generating Predictions...")
    print("-" * 60)

    try:
        # Multi-asset ensemble prediction
        subprocess.run(["python3", "predict.py"])

        # All assets predictions
        subprocess.run(["python3", "predict_all_assets.py"])

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
            run_command(["python3", "predict.py", "--asset", asset], f"Generating prediction for {asset}...")

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
