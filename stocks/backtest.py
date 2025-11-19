"""
Multi-asset backtesting system

Tests trading strategy across multiple uncorrelated assets
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader, get_default_multi_asset_config
from core.train_models import ModelTrainer


class MultiAssetBacktester:
    """
    Backtest trading strategy across multiple assets
    """

    def __init__(self, initial_capital=100000, max_positions=3):
        """
        Initialize backtester

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

        # Load trained models
        models_dir = Path(__file__).parent.parent / 'models'
        self.trainer = ModelTrainer(output_dir=str(models_dir))
        self.trainer.load_models(prefix='stock')

    def prepare_features(self, df):
        """
        Prepare features for a single asset

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        # Add features using utils (same as training)
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

        # Select only the feature columns used in training
        feature_df = df[self.trainer.feature_columns].copy()

        # Handle NaN values same as training
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                feature_df[col].fillna(feature_df[col].median(), inplace=True)

        # Replace infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)

        return feature_df

    def generate_signals(self, assets_data, min_probability=0.52, voting_threshold=0.5):
        """
        Generate trading signals for all assets

        Args:
            assets_data: Dict mapping ticker to DataFrame
            min_probability: Minimum probability to trade
            voting_threshold: Fraction of models that must agree

        Returns:
            Dict mapping ticker to signals DataFrame
        """
        signals = {}

        for ticker, df in assets_data.items():
            print(f"\nGenerating signals for {ticker}...")

            # Prepare features
            features_df = self.prepare_features(df)

            # Get predictions
            X = features_df.values
            pred, proba = self.trainer.predict_ensemble(X, voting_threshold=voting_threshold)

            # Create signals DataFrame
            signals_df = pd.DataFrame(index=df.index)
            signals_df['Prediction'] = pred
            signals_df['Probability'] = proba
            signals_df['Signal'] = ((pred == 1) & (proba >= min_probability)).astype(int)
            signals_df['Close'] = df['Close']

            signal_count = signals_df['Signal'].sum()
            avg_prob = signals_df[signals_df['Signal'] == 1]['Probability'].mean() if signal_count > 0 else 0

            print(f"  {ticker}: {signal_count} signals, avg probability: {avg_prob:.3f}")

            signals[ticker] = signals_df

        return signals

    def run_backtest(
        self,
        assets_data,
        signals,
        holding_period=10,
        stop_loss_pct=0.04,
        max_position_size_pct=0.40,
        use_leverage=True,
        max_leverage=2.0
    ):
        """
        Run backtest across multiple assets

        Args:
            assets_data: Dict mapping ticker to DataFrame
            signals: Dict mapping ticker to signals DataFrame
            holding_period: Days to hold position
            stop_loss_pct: Stop loss percentage (e.g., 0.04 = 4%)
            max_position_size_pct: Max % of capital per position
            use_leverage: Whether to use leverage
            max_leverage: Maximum leverage multiplier

        Returns:
            Backtest results
        """
        print("\n" + "=" * 70)
        print("MULTI-ASSET BACKTEST")
        print("=" * 70)
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Max positions: {self.max_positions}")
        print(f"Holding period: {holding_period} days")
        print(f"Stop loss: {stop_loss_pct*100:.1f}%")
        print(f"Max leverage: {max_leverage}x")
        print("=" * 70)

        # Get all dates across all assets
        all_dates = sorted(set().union(*[set(df.index) for df in assets_data.values()]))

        for date in all_dates:
            # Calculate current portfolio value (cash + position values)
            portfolio_value = self.cash

            # Add value of open positions to portfolio value
            for ticker, position in self.positions.items():
                if date in assets_data[ticker].index:
                    current_price = assets_data[ticker].loc[date, 'Close']
                    position_value = position['shares'] * current_price
                    portfolio_value += position_value

            # Protect against NaN
            if portfolio_value <= 0 or np.isnan(portfolio_value):
                portfolio_value = self.cash if self.cash > 0 else self.initial_capital * 0.01  # Emergency fallback

            # Update existing positions
            positions_to_close = []

            for ticker, position in self.positions.items():
                if date not in assets_data[ticker].index:
                    continue

                current_price = assets_data[ticker].loc[date, 'Close']
                position['current_price'] = current_price
                position['days_held'] += 1

                # FIX: Calculate P&L with division-by-zero protection
                if position['entry_price'] > 0:
                    position['pnl'] = (current_price - position['entry_price']) / position['entry_price']
                else:
                    position['pnl'] = 0.0
                    # Invalid entry price - shouldn't happen but protect against bad data

                position_value = position['shares'] * current_price

                portfolio_value += position_value

                # Check exit conditions
                exit_reason = None

                # Stop loss
                if position['pnl'] < -stop_loss_pct:
                    exit_reason = 'stop_loss'

                # Holding period
                elif position['days_held'] >= holding_period:
                    exit_reason = 'holding_period'

                if exit_reason:
                    # Close position
                    exit_value = position['shares'] * current_price
                    profit = exit_value - position['cost_basis']

                    self.cash += exit_value

                    self.trades.append({
                        'ticker': ticker,
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

                    positions_to_close.append(ticker)

            # Remove closed positions
            for ticker in positions_to_close:
                del self.positions[ticker]

            # Check for new signals
            if len(self.positions) < self.max_positions:
                for ticker in assets_data.keys():
                    if ticker in self.positions:
                        continue  # Already have position

                    if date not in signals[ticker].index:
                        continue

                    signal_row = signals[ticker].loc[date]

                    if signal_row['Signal'] == 1:
                        # Calculate position size
                        base_size = portfolio_value * max_position_size_pct

                        # Adjust for probability/confidence
                        leverage = 1.0
                        if use_leverage:
                            prob = signal_row['Probability']
                            if prob > 0.70:
                                leverage = min(1.8, max_leverage)
                            elif prob > 0.60:
                                leverage = min(1.5, max_leverage)

                        position_size = base_size * leverage
                        position_size = min(position_size, portfolio_value)  # Don't exceed total capital

                        entry_price = signal_row['Close']
                        shares = int(position_size / entry_price)

                        if shares > 0 and position_size <= self.cash:
                            # Open position
                            cost_basis = shares * entry_price
                            self.cash -= cost_basis

                            self.positions[ticker] = {
                                'entry_date': date,
                                'entry_price': entry_price,
                                'shares': shares,
                                'cost_basis': cost_basis,
                                'current_price': entry_price,
                                'days_held': 0,
                                'pnl': 0,
                                'probability': signal_row['Probability']
                            }

                            # Stop after opening one position per day
                            if len(self.positions) >= self.max_positions:
                                break

            # Record equity curve
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'num_positions': len(self.positions)
            })

        # Final portfolio value
        final_value = self.cash
        for ticker, position in self.positions.items():
            final_value += position['shares'] * position['current_price']

        return self.calculate_metrics(final_value)

    def calculate_metrics(self, final_value):
        """
        Calculate performance metrics

        Args:
            final_value: Final portfolio value

        Returns:
            Dict of metrics
        """
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (equity_df['date'].max() - equity_df['date'].min()).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Win rate
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
            sharpe = equity_df['daily_return'].mean() / std_return * np.sqrt(252)
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

        Args:
            metrics: Dict of performance metrics
            trades_df: DataFrame of trades
        """
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
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
            asset_summary = trades_df.groupby('ticker').agg({
                'profit': ['count', 'mean', 'sum'],
                'return': 'mean'
            }).round(4)
            print(asset_summary)

        print("\n" + "=" * 70)


def main():
    """
    Main backtest function
    """
    print("=" * 70)
    print("MULTI-ASSET STOCK BACKTEST")
    print("=" * 70)

    # Load data
    loader = DataLoader(data_dir='..')
    config = get_default_multi_asset_config()

    print("\nLoading multi-asset data...")
    assets_data = loader.load_multi_asset(config, start_date='2018-01-01')

    print(f"\n✓ Loaded {len(assets_data)} assets")

    # Initialize backtester
    backtester = MultiAssetBacktester(initial_capital=100000, max_positions=3)

    # Generate signals
    signals = backtester.generate_signals(
        assets_data,
        min_probability=0.52,
        voting_threshold=0.5  # 2/4 models must agree (50%)
    )

    # Run backtest
    metrics, equity_df, trades_df = backtester.run_backtest(
        assets_data,
        signals,
        holding_period=10,
        stop_loss_pct=0.04,
        max_position_size_pct=0.40,
        use_leverage=True,
        max_leverage=2.0
    )

    # Print results
    backtester.print_results(metrics, trades_df)

    # Save results
    output_dir = Path('../outputs')
    output_dir.mkdir(exist_ok=True)

    trades_df.to_csv(output_dir / 'multi_asset_trades.csv', index=False)
    equity_df.to_csv(output_dir / 'multi_asset_equity.csv', index=False)

    print(f"\n✓ Results saved to {output_dir}")

    print("\n" + "=" * 70)
    print(f"FINAL ANNUALIZED RETURN: {metrics['annualized_return']:.2%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
