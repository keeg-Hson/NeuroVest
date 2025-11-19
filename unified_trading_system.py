"""
Unified Trading System - Blends stocks and crypto into one portfolio

User inputs total capital, system automatically:
1. Determines optimal allocation between stocks/crypto
2. Executes trades on both markets
3. Tracks combined performance
4. Rebalances periodically
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from stocks.backtest import MultiAssetBacktester
from crypto.backtest import CryptoBacktester
from portfolio_allocator import PortfolioAllocator


class UnifiedTradingSystem:
    """
    Unified system that manages stocks and crypto in one portfolio
    """

    def __init__(
        self,
        total_capital=100000,
        risk_profile='moderate',
        rebalance_frequency_days=30
    ):
        """
        Initialize unified trading system

        Args:
            total_capital: Total capital to deploy
            risk_profile: 'conservative', 'moderate', or 'aggressive'
            rebalance_frequency_days: How often to rebalance (30 = monthly)
        """
        self.total_capital = total_capital
        self.risk_profile = risk_profile
        self.rebalance_frequency = rebalance_frequency_days

        # Portfolio allocator
        self.allocator = PortfolioAllocator(total_capital)

        # Performance tracking
        self.equity_curve = []
        self.all_trades = []
        self.rebalance_events = []

    def run_unified_backtest(
        self,
        stock_assets,
        crypto_assets,
        start_date=None,
        allocation_override=None
    ):
        """
        Run unified backtest across stocks and crypto

        Args:
            stock_assets: Dict mapping stock ticker to DataFrame
            crypto_assets: Dict mapping crypto symbol to DataFrame
            start_date: Start date for backtest
            allocation_override: Optional dict with stock_pct and crypto_pct

        Returns:
            Combined performance metrics
        """
        print("=" * 70)
        print("UNIFIED TRADING SYSTEM BACKTEST")
        print("=" * 70)
        print(f"Total Capital: ${self.total_capital:,.0f}")
        print(f"Risk Profile: {self.risk_profile.capitalize()}")
        print(f"Rebalance Frequency: Every {self.rebalance_frequency} days")
        print("=" * 70)

        # Determine allocation
        if allocation_override:
            stock_pct = allocation_override['stock_pct']
            crypto_pct = allocation_override['crypto_pct']
            print(f"\nUsing manual allocation: {stock_pct*100:.0f}% stocks, {crypto_pct*100:.0f}% crypto")
        else:
            # Use portfolio allocator with estimated metrics
            stock_metrics = {
                'annualized_return': 0.26,
                'sharpe_ratio': 1.80,
                'max_drawdown': 0.17
            }
            crypto_metrics = {
                'annualized_return': 0.45,
                'sharpe_ratio': 0.88,
                'max_drawdown': 0.85
            }

            allocation = self.allocator.calculate_optimal_allocation(
                stock_return=stock_metrics['annualized_return'],
                stock_sharpe=stock_metrics['sharpe_ratio'],
                stock_drawdown=stock_metrics['max_drawdown'],
                crypto_return=crypto_metrics['annualized_return'],
                crypto_sharpe=crypto_metrics['sharpe_ratio'],
                crypto_drawdown=crypto_metrics['max_drawdown'],
                risk_tolerance=self.risk_profile
            )

            stock_pct = allocation['stock_allocation_pct']
            crypto_pct = allocation['crypto_allocation_pct']

            print(f"\nOptimal allocation ({self.risk_profile}):")
            print(f"  Stocks: {stock_pct*100:.1f}% (${self.total_capital * stock_pct:,.0f})")
            print(f"  Crypto: {crypto_pct*100:.1f}% (${self.total_capital * crypto_pct:,.0f})")

        # Calculate capital allocation
        stock_capital = self.total_capital * stock_pct
        crypto_capital = self.total_capital * crypto_pct

        # Run stock backtest
        print("\n" + "=" * 70)
        print("RUNNING STOCK BACKTEST")
        print("=" * 70)

        stock_backtester = MultiAssetBacktester(
            initial_capital=stock_capital,
            max_positions=3
        )

        stock_signals = stock_backtester.generate_signals(
            stock_assets,
            min_probability=0.52,
            voting_threshold=0.5
        )

        stock_metrics, stock_equity, stock_trades = stock_backtester.run_backtest(
            stock_assets,
            stock_signals,
            holding_period=10,
            stop_loss_pct=0.04,
            max_position_size_pct=0.40,
            use_leverage=True,
            max_leverage=2.0
        )

        # Run crypto backtest
        print("\n" + "=" * 70)
        print("RUNNING CRYPTO BACKTEST")
        print("=" * 70)

        crypto_backtester = CryptoBacktester(
            initial_capital=crypto_capital,
            max_positions=3
        )

        crypto_signals = crypto_backtester.generate_signals(
            crypto_assets,
            min_probability=0.55,
            voting_threshold=0.5
        )

        crypto_metrics, crypto_equity, crypto_trades = crypto_backtester.run_backtest(
            crypto_assets,
            crypto_signals,
            holding_period=7,
            stop_loss_pct=0.08,
            max_position_size_pct=0.30,
            use_leverage=True,
            max_leverage=3.0
        )

        # Combine results
        combined_metrics = self.combine_results(
            stock_metrics, stock_equity, stock_trades,
            crypto_metrics, crypto_equity, crypto_trades,
            stock_pct, crypto_pct
        )

        return combined_metrics

    def combine_results(
        self,
        stock_metrics, stock_equity, stock_trades,
        crypto_metrics, crypto_equity, crypto_trades,
        stock_pct, crypto_pct
    ):
        """
        Combine stock and crypto results into unified metrics
        """
        print("\n" + "=" * 70)
        print("COMBINED PORTFOLIO RESULTS")
        print("=" * 70)

        # Align equity curves by date
        stock_equity['source'] = 'stocks'
        crypto_equity['source'] = 'crypto'

        # Merge equity curves
        stock_equity_indexed = stock_equity.set_index('date')[['portfolio_value']]
        crypto_equity_indexed = crypto_equity.set_index('date')[['portfolio_value']]

        # Combine (outer join to get all dates)
        combined_equity = stock_equity_indexed.add(
            crypto_equity_indexed, fill_value=0
        ).reset_index()
        combined_equity.columns = ['date', 'portfolio_value']
        combined_equity = combined_equity.sort_values('date')

        # Fill forward to handle days when only one market traded
        combined_equity['portfolio_value'] = combined_equity['portfolio_value'].fillna(method='ffill')

        # Calculate combined metrics
        initial_value = stock_metrics['initial_capital'] + crypto_metrics['initial_capital']
        final_value = stock_metrics['final_value'] + crypto_metrics['final_value']

        # FIX: Division-by-zero protection for total return
        if initial_value > 0:
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = 0.0

        days = (combined_equity['date'].max() - combined_equity['date'].min()).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Combine trades
        stock_trades['source'] = 'stocks'
        crypto_trades['source'] = 'crypto'
        all_trades = pd.concat([stock_trades, crypto_trades], ignore_index=True)

        # Overall metrics
        total_trades = len(all_trades)
        wins = (all_trades['profit'] > 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0

        # FIX: Drawdown calculation with division-by-zero protection
        combined_equity['peak'] = combined_equity['portfolio_value'].cummax()
        combined_equity['drawdown'] = np.where(
            combined_equity['peak'] > 0,
            (combined_equity['portfolio_value'] - combined_equity['peak']) / combined_equity['peak'],
            0.0
        )
        max_drawdown = combined_equity['drawdown'].min()

        # FIX: Sharpe ratio with protection against zero standard deviation
        combined_equity['daily_return'] = combined_equity['portfolio_value'].pct_change()
        std_return = combined_equity['daily_return'].std()
        if len(combined_equity) > 1 and std_return > 0:
            sharpe = combined_equity['daily_return'].mean() / std_return * np.sqrt(365)
        else:
            sharpe = 0.0

        # Print results
        print(f"\nCapital Allocation:")
        print(f"  Stocks: ${stock_metrics['initial_capital']:,.0f} ({stock_pct*100:.1f}%)")
        print(f"  Crypto: ${crypto_metrics['initial_capital']:,.0f} ({crypto_pct*100:.1f}%)")
        print(f"  Total: ${initial_value:,.0f}")

        print(f"\nFinal Values:")
        print(f"  Stocks: ${stock_metrics['final_value']:,.0f} "
              f"({stock_metrics['total_return']*100:+.2f}%)")
        print(f"  Crypto: ${crypto_metrics['final_value']:,.0f} "
              f"({crypto_metrics['total_return']*100:+.2f}%)")
        print(f"  Combined: ${final_value:,.0f} ({total_return*100:+.2f}%)")

        print(f"\nCombined Performance:")
        print(f"  Annualized Return: {annualized_return*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")

        print(f"\nTrading Activity:")
        print(f"  Stock Trades: {stock_metrics['num_trades']} "
              f"({stock_metrics['win_rate']*100:.1f}% win rate)")
        print(f"  Crypto Trades: {crypto_metrics['num_trades']} "
              f"({crypto_metrics['win_rate']*100:.1f}% win rate)")
        print(f"  Total Trades: {total_trades} ({win_rate*100:.1f}% win rate)")

        # Performance by source
        print(f"\nContribution to Returns:")
        stock_contribution = (stock_metrics['final_value'] - stock_metrics['initial_capital'])
        crypto_contribution = (crypto_metrics['final_value'] - crypto_metrics['initial_capital'])
        total_profit = stock_contribution + crypto_contribution

        if total_profit != 0:
            print(f"  Stock Profit: ${stock_contribution:,.0f} "
                  f"({stock_contribution/total_profit*100:.1f}% of total)")
            print(f"  Crypto Profit: ${crypto_contribution:,.0f} "
                  f"({crypto_contribution/total_profit*100:.1f}% of total)")

        print("\n" + "=" * 70)

        combined_metrics = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'stock_metrics': stock_metrics,
            'crypto_metrics': crypto_metrics,
            'stock_allocation': stock_pct,
            'crypto_allocation': crypto_pct
        }

        return combined_metrics, combined_equity, all_trades


def main():
    """
    Main unified system demo
    """
    from core.data_loader import DataLoader, get_default_multi_asset_config

    print("=" * 70)
    print("UNIFIED STOCK + CRYPTO TRADING SYSTEM")
    print("=" * 70)

    # Load stock data
    print("\nLoading stock assets...")
    stock_loader = DataLoader()
    stock_config = get_default_multi_asset_config()
    stock_assets = stock_loader.load_multi_asset(
        stock_config,
        start_date='2018-01-01'
    )
    print(f"✓ Loaded {len(stock_assets)} stock assets")

    # Load crypto data
    print("\nLoading crypto assets...")
    cache_dir = Path('data_cache')
    crypto_files = {
        'BTC/USDT': 'BTC_USDT_1d.csv',
        'ETH/USDT': 'ETH_USDT_1d.csv',
        'SOL/USDT': 'SOL_USDT_1d.csv',
        'AVAX/USDT': 'AVAX_USDT_1d.csv',
        'MATIC/USDT': 'MATIC_USDT_1d.csv'
    }

    crypto_assets = {}
    for symbol, filename in crypto_files.items():
        filepath = cache_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            crypto_assets[symbol] = df

    print(f"✓ Loaded {len(crypto_assets)} crypto assets")

    # Test different risk profiles
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT RISK PROFILES")
    print("=" * 70)

    profiles = ['conservative', 'moderate', 'aggressive']
    results = {}

    for profile in profiles:
        print(f"\n{'='*70}")
        print(f"PROFILE: {profile.upper()}")
        print(f"{'='*70}")

        system = UnifiedTradingSystem(
            total_capital=100000,
            risk_profile=profile,
            rebalance_frequency_days=30
        )

        metrics, equity, trades = system.run_unified_backtest(
            stock_assets,
            crypto_assets
        )

        results[profile] = metrics

    # Compare profiles
    print("\n" + "=" * 70)
    print("RISK PROFILE COMPARISON")
    print("=" * 70)

    comparison = []
    for profile, metrics in results.items():
        comparison.append({
            'Profile': profile.capitalize(),
            'Stock %': f"{metrics['stock_allocation']*100:.0f}%",
            'Crypto %': f"{metrics['crypto_allocation']*100:.0f}%",
            'Ann. Return': f"{metrics['annualized_return']*100:.2f}%",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Max DD': f"{metrics['max_drawdown']*100:.1f}%",
            'Total Trades': metrics['total_trades'],
            'Win Rate': f"{metrics['win_rate']*100:.1f}%"
        })

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    comparison_df.to_csv(output_dir / 'unified_system_comparison.csv', index=False)
    print(f"\n✓ Results saved to {output_dir}")

    print("\n" + "=" * 70)
    print("✓ Unified system backtest complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
