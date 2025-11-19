#!/usr/bin/env python3
"""
Portfolio Backtest

Backtests a portfolio of multiple assets with:
- Multi-asset allocation
- Periodic rebalancing
- Portfolio-level metrics (Sharpe, diversification)
- Correlation-aware position sizing

Usage:
    python3 backtest_portfolio.py --assets SPY,QQQ,BTC/USDT --weights 0.4,0.4,0.2
    python3 backtest_portfolio.py --asset-group crypto
    python3 backtest_portfolio.py --config portfolio_config.json

Output:
    - Portfolio equity curve
    - Per-asset performance
    - Rebalancing history
    - Combined metrics
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent / "framework"))
from framework.asset_manager import AssetManager

from utils import load_asset_data
from config import LOGS_DIR


class PortfolioBacktest:
    """Multi-asset portfolio backtester"""

    def __init__(
        self,
        assets: list,
        weights: list = None,
        initial_capital: float = 100_000,
        rebalance_freq: str = 'monthly',  # 'daily', 'weekly', 'monthly', 'quarterly'
        rebalance_threshold: float = 0.05,  # Rebalance if drift > 5%
        fee_bps: float = 2.0,
        slip_bps: float = 3.0,
    ):
        self.assets = assets
        self.n_assets = len(assets)

        # Default to equal weights
        if weights is None:
            self.target_weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.target_weights = np.array(weights)
            # Normalize to sum to 1
            self.target_weights = self.target_weights / self.target_weights.sum()

        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.rebalance_threshold = rebalance_threshold
        self.fee_bps = fee_bps / 10000  # Convert to decimal
        self.slip_bps = slip_bps / 10000

        self.predictions_dir = LOGS_DIR / "predictions"
        self.output_dir = LOGS_DIR / "portfolio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_asset_predictions(self, ticker: str) -> pd.DataFrame:
        """Load predictions for an asset"""

        pred_file = self.predictions_dir / f"{ticker.replace('/', '_')}_predictions.csv"

        if pred_file.exists():
            df = pd.read_csv(pred_file, parse_dates=['Date'])
            print(f"   ✓ {ticker}: Loaded {len(df):,} predictions from {pred_file.name}")
            return df
        else:
            # Fall back to asset data without predictions
            print(f"   ⚠️  {ticker}: No predictions, loading price data only")
            df = load_asset_data(ticker)
            df = df.reset_index()
            df['Prediction'] = 1  # Neutral/hold
            df['Confidence'] = 0.5
            df['Spike_Conf'] = 0.0
            df['Crash_Conf'] = 0.0
            return df

    def should_rebalance(self, date: pd.Timestamp, last_rebalance: pd.Timestamp) -> bool:
        """Check if it's time to rebalance"""

        if last_rebalance is None:
            return True

        days_since = (date - last_rebalance).days

        if self.rebalance_freq == 'daily':
            return days_since >= 1
        elif self.rebalance_freq == 'weekly':
            return days_since >= 7
        elif self.rebalance_freq == 'monthly':
            return days_since >= 30
        elif self.rebalance_freq == 'quarterly':
            return days_since >= 90
        else:
            return False

    def run(self):
        """Execute portfolio backtest"""

        print(f"\n{'=' * 80}")
        print("PORTFOLIO BACKTEST")
        print(f"{'=' * 80}")
        print(f"Assets: {len(self.assets)}")
        print(f"Weights: {[f'{w:.1%}' for w in self.target_weights]}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Rebalance: {self.rebalance_freq}")
        print(f"{'=' * 80}\n")

        # Load all asset data
        print("📥 Loading asset data...")
        asset_data = {}

        for ticker in self.assets:
            try:
                asset_data[ticker] = self.load_asset_predictions(ticker)
            except Exception as e:
                print(f"   ✗ {ticker}: {e}")
                return None

        if not asset_data:
            print("❌ No asset data loaded")
            return None

        # Find common date range
        all_dates = [df['Date'].values for df in asset_data.values()]
        common_start = max(df['Date'].min() for df in asset_data.values())
        common_end = min(df['Date'].max() for df in asset_data.values())

        print(f"\n📅 Date range: {common_start} to {common_end}")

        # Align all assets to common dates
        print("\n🔧 Aligning asset data to common dates...")
        aligned_data = {}

        for ticker, df in asset_data.items():
            df_filtered = df[(df['Date'] >= common_start) & (df['Date'] <= common_end)].copy()
            df_filtered = df_filtered.set_index('Date').sort_index()
            aligned_data[ticker] = df_filtered
            print(f"   ✓ {ticker}: {len(df_filtered):,} rows")

        # Get all unique dates
        all_dates = sorted(set().union(*[df.index for df in aligned_data.values()]))
        print(f"\n📊 Total trading days: {len(all_dates):,}")

        # Initialize portfolio tracking
        portfolio_history = []
        rebalance_history = []

        # Current holdings (in shares)
        holdings = {ticker: 0.0 for ticker in self.assets}
        cash = self.initial_capital
        last_rebalance = None

        # Track per-asset returns
        asset_returns = {ticker: [] for ticker in self.assets}

        print("\n🤖 Running backtest...")

        for i, date in enumerate(all_dates):

            # Get prices for all assets on this date
            prices = {}
            predictions = {}
            valid_assets = []

            for ticker in self.assets:
                if date in aligned_data[ticker].index:
                    row = aligned_data[ticker].loc[date]
                    if pd.notna(row['Close']) and row['Close'] > 0:
                        prices[ticker] = row['Close']
                        predictions[ticker] = row.get('Prediction', 1)
                        valid_assets.append(ticker)

            if not prices:
                continue

            # Calculate current portfolio value
            portfolio_value = cash
            asset_values = {}

            for ticker in valid_assets:
                if ticker in holdings:
                    asset_value = holdings[ticker] * prices[ticker]
                    portfolio_value += asset_value
                    asset_values[ticker] = asset_value

            # Check if we should rebalance
            needs_rebalance = self.should_rebalance(date, last_rebalance)

            # Also check drift threshold
            if not needs_rebalance and last_rebalance is not None and portfolio_value > 0:
                current_weights = np.array([asset_values.get(ticker, 0) / portfolio_value for ticker in self.assets])
                weight_drift = np.abs(current_weights - self.target_weights).max()

                if weight_drift > self.rebalance_threshold:
                    needs_rebalance = True

            # Rebalance if needed
            if needs_rebalance:
                # Sell all holdings back to cash
                for ticker in valid_assets:
                    if holdings[ticker] > 0:
                        sell_value = holdings[ticker] * prices[ticker]
                        # Apply fees and slippage
                        cost = sell_value * (self.fee_bps + self.slip_bps)
                        cash += sell_value - cost
                        holdings[ticker] = 0

                # Allocate cash according to target weights
                for j, ticker in enumerate(self.assets):
                    if ticker in valid_assets:
                        target_value = portfolio_value * self.target_weights[j]
                        # Buy shares
                        cost = target_value * (self.fee_bps + self.slip_bps)
                        shares = (target_value - cost) / prices[ticker]
                        holdings[ticker] = shares
                        cash -= target_value

                rebalance_history.append({
                    'Date': date,
                    'Portfolio_Value': portfolio_value,
                    'Weights': self.target_weights.copy()
                })

                last_rebalance = date

            # Record portfolio state
            current_weights = np.array([asset_values.get(ticker, 0) / portfolio_value for ticker in self.assets])

            portfolio_history.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': cash,
                **{f'{ticker}_value': asset_values.get(ticker, 0) for ticker in self.assets},
                **{f'{ticker}_weight': current_weights[j] for j, ticker in enumerate(self.assets)},
                **{f'{ticker}_price': prices.get(ticker, np.nan) for ticker in self.assets},
            })

            # Progress indicator
            if (i + 1) % 500 == 0:
                pct = (i + 1) / len(all_dates) * 100
                print(f"   [{pct:.0f}%] Processed {i+1:,}/{len(all_dates):,} days", end='\r')

        print(f"\n   [100%] Completed {len(all_dates):,} days")

        # Convert to dataframe
        df_portfolio = pd.DataFrame(portfolio_history)
        df_rebalances = pd.DataFrame(rebalance_history)

        # Calculate metrics
        print("\n📊 Calculating metrics...")

        returns = df_portfolio['Portfolio_Value'].pct_change().dropna()
        total_return = (df_portfolio['Portfolio_Value'].iloc[-1] / self.initial_capital) - 1
        cum_returns = (1 + returns).cumprod() - 1

        # Annualized metrics
        n_days = len(returns)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe ratio
        if len(returns) > 1:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        running_max = df_portfolio['Portfolio_Value'].expanding().max()
        drawdown = (df_portfolio['Portfolio_Value'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Volatility
        annual_vol = returns.std() * np.sqrt(252)

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': df_portfolio['Portfolio_Value'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_vol,
            'win_rate': win_rate,
            'n_days': n_days,
            'n_rebalances': len(df_rebalances),
            'n_assets': self.n_assets,
            'assets': self.assets,
            'target_weights': self.target_weights.tolist(),
        }

        # Per-asset statistics
        asset_stats = []
        for ticker in self.assets:
            col = f'{ticker}_value'
            if col in df_portfolio.columns:
                values = df_portfolio[col]
                asset_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0] if values.iloc[0] > 0 else 0
                asset_stats.append({
                    'Asset': ticker,
                    'Final Value': values.iloc[-1],
                    'Return': asset_return,
                    'Avg Weight': df_portfolio[f'{ticker}_weight'].mean(),
                })

        df_asset_stats = pd.DataFrame(asset_stats)

        # Print results
        print(f"\n{'=' * 80}")
        print("PORTFOLIO BACKTEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Initial Capital:      ${metrics['initial_capital']:,.2f}")
        print(f"Final Value:          ${metrics['final_value']:,.2f}")
        print(f"Total Return:         {metrics['total_return']:.2%}")
        print(f"Annualized Return:    {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio:         {metrics['sharpe']:.2f}")
        print(f"Max Drawdown:         {metrics['max_drawdown']:.2%}")
        print(f"Annual Volatility:    {metrics['annual_volatility']:.2%}")
        print(f"Win Rate:             {metrics['win_rate']:.2%}")
        print(f"Trading Days:         {metrics['n_days']:,}")
        print(f"Rebalances:           {metrics['n_rebalances']}")
        print(f"{'=' * 80}\n")

        print("Per-Asset Statistics:")
        print(df_asset_stats.to_string(index=False))
        print()

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        portfolio_file = self.output_dir / f"portfolio_history_{timestamp}.csv"
        df_portfolio.to_csv(portfolio_file, index=False)
        print(f"💾 Saved portfolio history: {portfolio_file}")

        rebalance_file = self.output_dir / f"rebalances_{timestamp}.csv"
        df_rebalances.to_csv(rebalance_file, index=False)
        print(f"💾 Saved rebalance history: {rebalance_file}")

        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"💾 Saved metrics: {metrics_file}")

        # Plot equity curve
        self.plot_results(df_portfolio, df_asset_stats, timestamp)

        return df_portfolio, metrics

    def plot_results(self, df_portfolio, df_asset_stats, timestamp):
        """Plot portfolio equity curve and asset weights"""

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Portfolio value
        axes[0].plot(df_portfolio['Date'], df_portfolio['Portfolio_Value'], linewidth=2)
        axes[0].set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].ticklabel_format(style='plain', axis='y')

        # Asset weights over time
        for ticker in self.assets:
            col = f'{ticker}_weight'
            if col in df_portfolio.columns:
                axes[1].plot(df_portfolio['Date'], df_portfolio[col], label=ticker, alpha=0.7)

        axes[1].set_title('Asset Weights Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Weight', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend(loc='upper left', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        plt.tight_layout()

        plot_file = self.output_dir / f"portfolio_chart_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved chart: {plot_file}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Portfolio backtest")

    parser.add_argument('--assets', type=str,
                        help="Comma-separated list of assets (e.g., SPY,QQQ,BTC/USDT)")
    parser.add_argument('--weights', type=str,
                        help="Comma-separated weights (e.g., 0.4,0.4,0.2). Default: equal weights")
    parser.add_argument('--asset-group', choices=['equity', 'crypto', 'bond', 'commodity'],
                        help="Use all assets from a group")
    parser.add_argument('--config', type=str,
                        help="JSON config file with portfolio settings")

    parser.add_argument('--capital', type=float, default=100_000,
                        help="Initial capital (default: 100,000)")
    parser.add_argument('--rebalance', type=str, default='monthly',
                        choices=['daily', 'weekly', 'monthly', 'quarterly'],
                        help="Rebalance frequency (default: monthly)")
    parser.add_argument('--rebalance-threshold', type=float, default=0.05,
                        help="Rebalance if weight drifts > threshold (default: 0.05)")
    parser.add_argument('--fee-bps', type=float, default=2.0,
                        help="Trading fee in basis points (default: 2.0)")
    parser.add_argument('--slip-bps', type=float, default=3.0,
                        help="Slippage in basis points (default: 3.0)")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        assets = config.get('assets')
        weights = config.get('weights')
        capital = config.get('capital', args.capital)
        rebalance = config.get('rebalance', args.rebalance)
        rebalance_threshold = config.get('rebalance_threshold', args.rebalance_threshold)
        fee_bps = config.get('fee_bps', args.fee_bps)
        slip_bps = config.get('slip_bps', args.slip_bps)

    elif args.assets:
        assets = [a.strip() for a in args.assets.split(',')]
        weights = None
        if args.weights:
            weights = [float(w) for w in args.weights.split(',')]
            if len(weights) != len(assets):
                print(f"❌ Number of weights ({len(weights)}) must match assets ({len(assets)})")
                return

        capital = args.capital
        rebalance = args.rebalance
        rebalance_threshold = args.rebalance_threshold
        fee_bps = args.fee_bps
        slip_bps = args.slip_bps

    elif args.asset_group:
        # Get assets from group
        mgr = AssetManager()
        group_assets = [a.ticker for a in mgr.get_all_assets() if a.asset_type == args.asset_group]

        # Filter to only assets with data
        data_dir = Path('data_cache')
        assets = []
        for ticker in group_assets:
            filename = ticker.replace('/', '_') + '_1d.csv'
            if (data_dir / filename).exists():
                assets.append(ticker)

        if not assets:
            print(f"❌ No assets with data in group: {args.asset_group}")
            return

        print(f"Using {len(assets)} assets from {args.asset_group}: {', '.join(assets)}")

        weights = None
        capital = args.capital
        rebalance = args.rebalance
        rebalance_threshold = args.rebalance_threshold
        fee_bps = args.fee_bps
        slip_bps = args.slip_bps

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 backtest_portfolio.py --assets SPY,QQQ,BTC/USDT --weights 0.4,0.4,0.2")
        print("  python3 backtest_portfolio.py --asset-group crypto")
        print("  python3 backtest_portfolio.py --config my_portfolio.json")
        return

    # Run backtest
    backtest = PortfolioBacktest(
        assets=assets,
        weights=weights,
        initial_capital=capital,
        rebalance_freq=rebalance,
        rebalance_threshold=rebalance_threshold,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
    )

    backtest.run()


if __name__ == "__main__":
    main()
