"""
Crypto backtesting system

Tests crypto trading strategy with appropriate risk parameters
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.train_models import ModelTrainer


class CryptoBacktester:
    """
    Backtest crypto trading strategy
    """

    def __init__(self, initial_capital=50000, max_positions=3):
        """
        Initialize crypto backtester

        Args:
            initial_capital: Starting capital
            max_positions: Maximum simultaneous positions
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions

        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Load trained crypto models
        models_dir = Path(__file__).parent.parent / 'models'
        self.trainer = ModelTrainer(output_dir=str(models_dir))
        self.trainer.load_models(prefix='crypto')

    def prepare_features(self, df):
        """
        Prepare features for crypto asset

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        from utils import add_features

        result = add_features(df)
        if isinstance(result, tuple):
            df, _ = result
        else:
            df = result

        # FIX: Validate feature_columns attribute exists
        if not hasattr(self.trainer, 'feature_columns'):
            raise ValueError("Trainer has no feature_columns attribute. Models may not be loaded correctly.")

        # FIX: Validate all required columns are present
        missing_cols = set(self.trainer.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        # Select feature columns
        feature_df = df[self.trainer.feature_columns].copy()

        # Handle NaN values
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                feature_df[col].fillna(feature_df[col].median(), inplace=True)

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)

        return feature_df

    def generate_signals(self, assets_data, min_probability=0.55, voting_threshold=0.5):
        """
        Generate trading signals for crypto assets

        Args:
            assets_data: Dict mapping symbol to DataFrame
            min_probability: Minimum probability (higher for crypto volatility)
            voting_threshold: Fraction of models that must agree

        Returns:
            Dict mapping symbol to signals DataFrame
        """
        signals = {}

        for symbol, df in assets_data.items():
            print(f"\nGenerating signals for {symbol}...")

            features_df = self.prepare_features(df)

            X = features_df.values
            pred, proba = self.trainer.predict_ensemble(X, voting_threshold=voting_threshold)

            signals_df = pd.DataFrame(index=df.index)
            signals_df['Prediction'] = pred
            signals_df['Probability'] = proba
            signals_df['Signal'] = ((pred == 1) & (proba >= min_probability)).astype(int)
            signals_df['Close'] = df['Close']

            signal_count = signals_df['Signal'].sum()
            avg_prob = signals_df[signals_df['Signal'] == 1]['Probability'].mean() if signal_count > 0 else 0

            print(f"  {symbol}: {signal_count} signals, avg probability: {avg_prob:.3f}")

            signals[symbol] = signals_df

        return signals

    def run_backtest(
        self,
        assets_data,
        signals,
        holding_period=7,         # Crypto: shorter holding
        stop_loss_pct=0.08,       # Crypto: wider stops (8% vs 4% stocks)
        max_position_size_pct=0.30,  # Crypto: smaller positions
        use_leverage=True,
        max_leverage=3.0          # Crypto: can use more leverage
    ):
        """
        Run crypto backtest

        Args:
            assets_data: Dict mapping symbol to DataFrame
            signals: Dict mapping symbol to signals DataFrame
            holding_period: Days to hold (7 for crypto vs 10 for stocks)
            stop_loss_pct: Stop loss (8% for crypto vs 4% for stocks)
            max_position_size_pct: Max position size
            use_leverage: Whether to use leverage
            max_leverage: Maximum leverage

        Returns:
            Backtest results
        """
        print("\n" + "=" * 70)
        print("CRYPTO BACKTEST")
        print("=" * 70)
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Max positions: {self.max_positions}")
        print(f"Holding period: {holding_period} days")
        print(f"Stop loss: {stop_loss_pct*100:.1f}%")
        print(f"Max leverage: {max_leverage}x")
        print("=" * 70)

        all_dates = sorted(set().union(*[set(df.index) for df in assets_data.values()]))

        for date in all_dates:
            # Calculate current portfolio value (cash + position values)
            portfolio_value = self.cash

            # Add value of open positions to portfolio value
            for symbol, position in self.positions.items():
                if date in assets_data[symbol].index:
                    current_price = assets_data[symbol].loc[date, 'Close']
                    position_value = position['shares'] * current_price
                    portfolio_value += position_value

            # Protect against NaN
            if portfolio_value <= 0 or np.isnan(portfolio_value):
                portfolio_value = self.cash if self.cash > 0 else self.initial_capital * 0.01

            # Update existing positions
            positions_to_close = []

            for symbol, position in self.positions.items():
                if date not in assets_data[symbol].index:
                    continue

                current_price = assets_data[symbol].loc[date, 'Close']
                position['current_price'] = current_price
                position['days_held'] += 1

                # FIX: Calculate P&L with division-by-zero protection
                if position['entry_price'] > 0:
                    position['pnl'] = (current_price - position['entry_price']) / position['entry_price']
                else:
                    position['pnl'] = 0.0

                position_value = position['shares'] * current_price

                portfolio_value += position_value

                exit_reason = None

                # Stop loss (wider for crypto)
                if position['pnl'] < -stop_loss_pct:
                    exit_reason = 'stop_loss'

                # Holding period
                elif position['days_held'] >= holding_period:
                    exit_reason = 'holding_period'

                if exit_reason:
                    exit_value = position['shares'] * current_price
                    profit = exit_value - position['cost_basis']

                    self.cash += exit_value

                    self.trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'cost_basis': position['cost_basis'],
                        'exit_value': exit_value,
                        'profit': profit,
                        'return': position['pnl'],
                        'days_held': position['days_held'],
                        'exit_reason': exit_reason
                    })

                    positions_to_close.append(symbol)

            for symbol in positions_to_close:
                del self.positions[symbol]

            # Check for new signals
            if len(self.positions) < self.max_positions:
                for symbol in assets_data.keys():
                    if symbol in self.positions:
                        continue

                    if date not in signals[symbol].index:
                        continue

                    signal_row = signals[symbol].loc[date]

                    if signal_row['Signal'] == 1:
                        base_size = portfolio_value * max_position_size_pct

                        # Adjust for confidence
                        leverage = 1.0
                        if use_leverage:
                            prob = signal_row['Probability']
                            if prob > 0.75:
                                leverage = min(2.5, max_leverage)
                            elif prob > 0.65:
                                leverage = min(2.0, max_leverage)
                            elif prob > 0.55:
                                leverage = min(1.5, max_leverage)

                        position_size = base_size * leverage
                        position_size = min(position_size, portfolio_value)

                        entry_price = signal_row['Close']
                        shares = position_size / entry_price

                        if shares > 0 and position_size <= self.cash:
                            cost_basis = shares * entry_price
                            self.cash -= cost_basis

                            self.positions[symbol] = {
                                'entry_date': date,
                                'entry_price': entry_price,
                                'shares': shares,
                                'cost_basis': cost_basis,
                                'current_price': entry_price,
                                'days_held': 0,
                                'pnl': 0,
                                'probability': signal_row['Probability']
                            }

                            if len(self.positions) >= self.max_positions:
                                break

            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'num_positions': len(self.positions)
            })

        # Final portfolio value
        final_value = self.cash
        for symbol, position in self.positions.items():
            final_value += position['shares'] * position['current_price']

        return self.calculate_metrics(final_value)

    def calculate_metrics(self, final_value):
        """
        Calculate performance metrics
        """
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        total_return = (final_value - self.initial_capital) / self.initial_capital

        days = (equity_df['date'].max() - equity_df['date'].min()).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        if len(trades_df) > 0:
            wins = (trades_df['profit'] > 0).sum()
            win_rate = wins / len(trades_df)
            avg_profit = trades_df[trades_df['profit'] > 0]['return'].mean() if wins > 0 else 0
            avg_loss = trades_df[trades_df['profit'] < 0]['return'].mean() if wins < len(trades_df) else 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0

        # FIX: Drawdown calculation with division-by-zero protection
        equity_df['peak'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = np.where(
            equity_df['peak'] > 0,
            (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak'],
            0.0
        )
        max_drawdown = equity_df['drawdown'].min()

        # FIX: Sharpe ratio with protection against zero standard deviation
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        std_return = equity_df['daily_return'].std()
        if len(equity_df) > 1 and std_return > 0:
            sharpe = equity_df['daily_return'].mean() / std_return * np.sqrt(365)
        else:
            sharpe = 0.0

        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'trading_days': len(equity_df)
        }

        return metrics, equity_df, trades_df

    def print_results(self, metrics, trades_df):
        """
        Print backtest results
        """
        print("\n" + "=" * 70)
        print("CRYPTO BACKTEST RESULTS")
        print("=" * 70)

        print(f"\nCapital:")
        print(f"  Initial: ${metrics['initial_capital']:,.0f}")
        print(f"  Final: ${metrics['final_value']:,.0f}")
        print(f"  Profit: ${metrics['final_value'] - metrics['initial_capital']:,.0f}")

        print(f"\nReturns:")
        print(f"  Total: {metrics['total_return']:.2%}")
        print(f"  Annualized: {metrics['annualized_return']:.2%}")

        print(f"\nRisk:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")

        print(f"\nTrading:")
        print(f"  Total Trades: {metrics['num_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Avg Profit: {metrics['avg_profit']:.2%}")
        print(f"  Avg Loss: {metrics['avg_loss']:.2%}")

        if len(trades_df) > 0:
            print(f"\nTrades by Asset:")
            asset_summary = trades_df.groupby('symbol').agg({
                'profit': ['count', 'mean', 'sum'],
                'return': 'mean'
            }).round(4)
            print(asset_summary)

        print("\n" + "=" * 70)


def main():
    """
    Main crypto backtest function
    """
    print("=" * 70)
    print("MULTI-ASSET CRYPTO BACKTEST")
    print("=" * 70)

    # Load crypto data
    cache_dir = Path('../data_cache')
    crypto_files = {
        'BTC/USDT': 'BTC_USDT_1d.csv',
        'ETH/USDT': 'ETH_USDT_1d.csv',
        'SOL/USDT': 'SOL_USDT_1d.csv',
        'AVAX/USDT': 'AVAX_USDT_1d.csv',
        'MATIC/USDT': 'MATIC_USDT_1d.csv'
    }

    print("\nLoading crypto data...")
    assets_data = {}

    for symbol, filename in crypto_files.items():
        filepath = cache_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            assets_data[symbol] = df
            print(f"  ✓ Loaded {symbol}: {len(df)} days")
        else:
            print(f"  ✗ Missing {symbol}: {filepath}")

    if len(assets_data) == 0:
        print("\n✗ No crypto data found. Run crypto/generate_synthetic_data.py first")
        return

    # Initialize backtester
    backtester = CryptoBacktester(initial_capital=50000, max_positions=3)

    # Generate signals (higher threshold for crypto)
    signals = backtester.generate_signals(
        assets_data,
        min_probability=0.55,  # Higher than stocks (0.52)
        voting_threshold=0.5
    )

    # Run backtest with crypto parameters
    metrics, equity_df, trades_df = backtester.run_backtest(
        assets_data,
        signals,
        holding_period=7,       # Shorter than stocks (10)
        stop_loss_pct=0.08,     # Wider than stocks (0.04)
        max_position_size_pct=0.30,  # Smaller than stocks (0.40)
        use_leverage=True,
        max_leverage=3.0        # Higher than stocks (2.0)
    )

    # Print results
    backtester.print_results(metrics, trades_df)

    # Save results
    output_dir = Path('../outputs')
    output_dir.mkdir(exist_ok=True)

    trades_df.to_csv(output_dir / 'crypto_trades.csv', index=False)
    equity_df.to_csv(output_dir / 'crypto_equity.csv', index=False)

    print(f"\n✓ Results saved to {output_dir}")

    print("\n" + "=" * 70)
    print(f"FINAL ANNUALIZED RETURN: {metrics['annualized_return']:.2%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
