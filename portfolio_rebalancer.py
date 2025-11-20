#!/usr/bin/env python3
"""
Portfolio Rebalancing System

Supports multiple rebalancing strategies:
- Fixed period (daily, weekly, monthly, quarterly)
- Threshold-based (rebalance when allocations drift)
- Optimal period detection via backtest

Usage:
    python3 portfolio_rebalancer.py --assets SPY,GLD,TLT --weights 0.6,0.3,0.1
    python3 portfolio_rebalancer.py --find-optimal
    python3 portfolio_rebalancer.py --profile conservative
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def load_trading_profile(profile_name="moderate"):
    """Load trading profile configuration"""
    profile_path = Path(f"configs/trading_profile_{profile_name}.json")

    if not profile_path.exists():
        print(f"⚠️  Profile not found: {profile_name}, using moderate")
        profile_path = Path("configs/trading_profile_moderate.json")

    with open(profile_path) as f:
        return json.load(f)


def load_asset_data(asset, start_date=None, end_date=None):
    """Load asset price data"""
    if asset == "SPY":
        filepath = Path("data/SPY.csv")
    else:
        asset_file = asset.replace("/", "_") + "_1d.csv"
        filepath = Path(f"data_cache/{asset_file}")

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]

    return df[['Date', 'Close']].rename(columns={'Close': asset})


def calculate_portfolio_value(prices, weights, rebalance_dates=None, method='buy_and_hold'):
    """
    Calculate portfolio performance with different rebalancing strategies.

    Args:
        prices: DataFrame with asset prices
        weights: Target allocation weights
        rebalance_dates: List of dates to rebalance (None = buy and hold)
        method: 'buy_and_hold', 'periodic', or 'threshold'

    Returns:
        DataFrame with portfolio values and metrics
    """
    assets = prices.columns[1:]  # First column is Date
    n_assets = len(assets)

    # Initialize portfolio
    initial_capital = 100000
    portfolio = pd.DataFrame(index=prices.index)
    portfolio['Date'] = prices['Date'].values

    # Track shares and values
    shares = {asset: 0 for asset in assets}
    cash = initial_capital

    # Initial allocation
    for i, asset in enumerate(assets):
        initial_price = prices[asset].iloc[0]
        target_value = initial_capital * weights[i]
        shares[asset] = target_value / initial_price
        cash -= target_value

    # Track portfolio value over time
    portfolio_values = []
    cash_values = []

    for idx in range(len(prices)):
        date = prices['Date'].iloc[idx]

        # Calculate current portfolio value
        total_value = cash
        for asset in assets:
            price = prices[asset].iloc[idx]
            total_value += shares[asset] * price

        portfolio_values.append(total_value)
        cash_values.append(cash)

        # Check if rebalancing needed
        should_rebalance = False

        if rebalance_dates is not None and date in rebalance_dates:
            should_rebalance = True
        elif method == 'threshold':
            # Calculate current allocations
            current_values = {asset: shares[asset] * prices[asset].iloc[idx] for asset in assets}
            current_allocs = {asset: val / total_value for asset, val in current_values.items()}

            # Check drift
            max_drift = max(abs(current_allocs[asset] - weights[i]) for i, asset in enumerate(assets))
            if max_drift > 0.10:  # 10% threshold
                should_rebalance = True

        # Rebalance if needed
        if should_rebalance and idx < len(prices) - 1:
            # Sell all positions
            cash = 0
            for asset in assets:
                price = prices[asset].iloc[idx]
                cash += shares[asset] * price
                shares[asset] = 0

            # Buy back at target weights
            for i, asset in enumerate(assets):
                price = prices[asset].iloc[idx]
                target_value = cash * weights[i]
                shares[asset] = target_value / price

            cash = 0  # Fully invested (ignoring transaction costs)

    portfolio['Value'] = portfolio_values
    portfolio['Cash'] = cash_values

    # Calculate returns
    portfolio['Returns'] = portfolio['Value'].pct_change()
    portfolio['Cumulative_Return'] = (portfolio['Value'] / initial_capital - 1) * 100

    return portfolio


def find_optimal_rebalancing_period(assets, weights, lookback_years=5):
    """
    Test different rebalancing periods to find optimal frequency.

    Args:
        assets: List of asset tickers
        weights: Target allocation weights
        lookback_years: Years of historical data to test

    Returns:
        dict: Results for each rebalancing frequency
    """
    print(f"\n{'=' * 70}")
    print("OPTIMAL REBALANCING PERIOD ANALYSIS")
    print(f"{'=' * 70}\n")

    # Load data for all assets
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)

    print(f"Loading data for {len(assets)} assets...")

    # Build combined price dataframe
    prices_df = None

    for asset in assets:
        asset_data = load_asset_data(asset, start_date, end_date)
        if asset_data is None:
            print(f"  ⚠️  {asset}: Data not available")
            continue

        if prices_df is None:
            prices_df = asset_data
        else:
            prices_df = prices_df.merge(asset_data, on='Date', how='outer')

    if prices_df is None:
        print("❌ No data available")
        return None

    # Forward fill missing values
    prices_df = prices_df.sort_values('Date').ffill()

    print(f"✓ Loaded {len(prices_df)} days of data\n")

    # Test different rebalancing frequencies
    strategies = {
        'Buy & Hold': None,
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Semi-Annual': '6M',
        'Annual': 'Y'
    }

    results = {}

    for strategy_name, freq in strategies.items():
        if freq is None:
            # Buy and hold
            portfolio = calculate_portfolio_value(prices_df, weights, None, 'buy_and_hold')
        else:
            # Generate rebalancing dates
            rebal_dates = pd.date_range(
                start=prices_df['Date'].iloc[0],
                end=prices_df['Date'].iloc[-1],
                freq=freq
            )
            portfolio = calculate_portfolio_value(prices_df, weights, rebal_dates, 'periodic')

        # Calculate metrics
        total_return = portfolio['Cumulative_Return'].iloc[-1]
        daily_returns = portfolio['Returns'].dropna()

        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Estimate transaction costs (0.1% per rebalance, both sides)
        if freq:
            n_rebalances = len([d for d in rebal_dates if d in prices_df['Date'].values])
            transaction_cost = n_rebalances * 0.002 * 100  # 0.2% per rebalance
        else:
            n_rebalances = 0
            transaction_cost = 0

        net_return = total_return - transaction_cost

        results[strategy_name] = {
            'total_return': total_return,
            'net_return': net_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_rebalances': n_rebalances,
            'transaction_cost': transaction_cost
        }

        print(f"{strategy_name:15s}: Return: {net_return:>7.2f}%  Sharpe: {sharpe:>5.2f}  MaxDD: {max_drawdown:>6.2f}%  Rebalances: {n_rebalances:>3d}")

    # Find optimal (highest Sharpe-adjusted return)
    best_strategy = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])

    print(f"\n{'=' * 70}")
    print(f"OPTIMAL STRATEGY: {best_strategy[0]}")
    print(f"  Net Return: {best_strategy[1]['net_return']:.2f}%")
    print(f"  Sharpe Ratio: {best_strategy[1]['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_strategy[1]['max_drawdown']:.2f}%")
    print(f"{'=' * 70}\n")

    return results


def execute_rebalancing(assets, target_weights, profile_name="moderate"):
    """
    Execute portfolio rebalancing based on trading profile.

    Args:
        assets: List of asset tickers
        target_weights: Target allocation weights
        profile_name: Trading profile to use
    """
    profile = load_trading_profile(profile_name)

    print(f"\n{'=' * 70}")
    print(f"PORTFOLIO REBALANCING - {profile['profile_name']} Profile")
    print(f"{'=' * 70}\n")

    # Load current prices
    current_prices = {}
    for asset in assets:
        asset_data = load_asset_data(asset)
        if asset_data is None:
            print(f"❌ {asset}: Data not available")
            return None

        current_prices[asset] = asset_data[asset].iloc[-1]

    # Simulate current portfolio (for demonstration)
    portfolio_value = 100000
    current_positions = {}

    # Display current allocation
    print("TARGET ALLOCATION:")
    for i, asset in enumerate(assets):
        print(f"  {asset:15s}: {target_weights[i]:>6.1%}")

    print(f"\nREBALANCING REQUIRED:")

    # Calculate trades needed
    for i, asset in enumerate(assets):
        target_value = portfolio_value * target_weights[i]
        target_shares = target_value / current_prices[asset]
        print(f"  {asset:15s}: ${target_value:>12,.2f}  ({target_shares:>10.4f} shares)")

    # Check profile constraints
    print(f"\nPROFILE CONSTRAINTS:")
    print(f"  Max Equity Exposure: {profile['portfolio_allocation']['max_equity_exposure']:.0%}")
    print(f"  Min Cash Reserve: {profile['portfolio_allocation']['min_cash_reserve']:.0%}")
    print(f"  Max Single Asset: {profile['portfolio_allocation']['max_single_asset']:.0%}")
    print(f"  Rebalance Frequency: {profile['rebalancing']['frequency']}")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Rebalancing System")
    parser.add_argument("--assets", help="Comma-separated asset list (e.g., SPY,GLD,TLT)")
    parser.add_argument("--weights", help="Comma-separated weights (e.g., 0.6,0.3,0.1)")
    parser.add_argument("--profile", default="moderate",
                        help="Trading profile: conservative, moderate, liberal")
    parser.add_argument("--find-optimal", action="store_true",
                        help="Find optimal rebalancing period via backtest")
    parser.add_argument("--lookback-years", type=int, default=5,
                        help="Years of data for optimal period search")

    args = parser.parse_args()

    if args.find_optimal:
        # Default portfolio for testing
        assets = args.assets.split(',') if args.assets else ['SPY', 'GLD', 'TLT']
        weights_str = args.weights.split(',') if args.weights else ['0.6', '0.3', '0.1']
        weights = [float(w) for w in weights_str]

        find_optimal_rebalancing_period(assets, weights, args.lookback_years)

    elif args.assets and args.weights:
        assets = args.assets.split(',')
        weights = [float(w) for w in args.weights.split(',')]

        if len(assets) != len(weights):
            print("❌ Number of assets must match number of weights")
            return

        if abs(sum(weights) - 1.0) > 0.01:
            print("❌ Weights must sum to 1.0")
            return

        execute_rebalancing(assets, weights, args.profile)

    else:
        print("\n💡 Usage examples:")
        print("  Find optimal period:     python3 portfolio_rebalancer.py --find-optimal")
        print("  Rebalance portfolio:     python3 portfolio_rebalancer.py --assets SPY,GLD --weights 0.7,0.3")
        print("  Use liberal profile:     python3 portfolio_rebalancer.py --assets SPY,GLD --weights 0.7,0.3 --profile liberal")


if __name__ == "__main__":
    main()
