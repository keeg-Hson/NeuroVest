"""
Production Trading Bot
Automated trading system with real-time signal generation and execution

Features:
- Real-time data fetching
- ML-based signal generation
- Automated order execution
- Position and risk management
- Paper trading mode for testing
- Comprehensive logging and alerts
"""

import pandas as pd
import numpy as np
import pickle
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from real_data_loader import load_multi_asset_real_data
from utils import add_features, finalize_features


# ============================================================================
# Configuration and Data Classes
# ============================================================================

class TradingMode(Enum):
    """Trading mode: paper or live"""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class TradingConfig:
    """Configuration for the trading bot"""
    # Assets to trade
    tickers: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'])

    # Capital and position sizing
    initial_capital: float = 100000
    max_positions: int = 3
    max_position_size_pct: float = 0.40  # Max 40% per position

    # Leverage
    use_leverage: bool = True
    max_leverage: float = 2.0

    # Signal thresholds
    min_ensemble_agreement: float = 0.50  # At least 2/4 models agree
    min_signal_probability: float = 0.52

    # Holding period
    holding_period_days: int = 10

    # Risk management
    stop_loss_pct: float = 0.04  # 4% stop loss
    max_daily_loss_pct: float = 0.02  # 2% max daily loss

    # Trading mode
    mode: TradingMode = TradingMode.PAPER

    # Model paths
    model_dir: str = 'models'
    model_suffix: str = 'ultimate'  # Try: ultimate, max_perf, regime

    # Logging
    log_file: str = 'trading_bot.log'
    log_level: str = 'INFO'


@dataclass
class Position:
    """Represents an open trading position"""
    ticker: str
    shares: float
    entry_price: float
    entry_date: datetime
    entry_value: float
    leverage: float = 1.0
    signal_prob: float = 0.0
    stop_loss: float = 0.0

    def current_value(self, current_price: float) -> float:
        """Calculate current position value"""
        return self.shares * current_price

    def pnl(self, current_price: float) -> float:
        """Calculate profit/loss"""
        return self.current_value(current_price) - self.entry_value

    def pnl_pct(self, current_price: float) -> float:
        """Calculate profit/loss percentage"""
        return (self.pnl(current_price) / self.entry_value) * 100

    def days_held(self, current_date: datetime) -> int:
        """Calculate days held"""
        return (current_date - self.entry_date).days


# ============================================================================
# Trading Bot
# ============================================================================

class TradingBot:
    """
    Production trading bot with automated signal generation and execution
    """

    def __init__(self, config: TradingConfig):
        """Initialize the trading bot"""
        self.config = config
        self.setup_logging()

        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.daily_pnl = []

        # Models
        self.models = None
        self.load_models()

        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}

        self.logger.info("="*70)
        self.logger.info("PRODUCTION TRADING BOT INITIALIZED")
        self.logger.info("="*70)
        self.logger.info(f"Mode: {self.config.mode.value.upper()}")
        self.logger.info(f"Initial Capital: ${self.config.initial_capital:,.0f}")
        self.logger.info(f"Assets: {', '.join(self.config.tickers)}")
        self.logger.info(f"Max Positions: {self.config.max_positions}")
        self.logger.info(f"Leverage: {'Enabled (max ' + str(self.config.max_leverage) + 'x)' if self.config.use_leverage else 'Disabled'}")
        self.logger.info("="*70)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingBot')

    def load_models(self):
        """Load pre-trained ML models"""
        try:
            model_dir = self.config.model_dir
            suffix = self.config.model_suffix

            with open(f'{model_dir}/xgboost_{suffix}.pkl', 'rb') as f:
                xgb_model = pickle.load(f)
            with open(f'{model_dir}/lightgbm_{suffix}.pkl', 'rb') as f:
                lgb_model = pickle.load(f)
            with open(f'{model_dir}/random_forest_{suffix}.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            with open(f'{model_dir}/neural_net_{suffix}.pkl', 'rb') as f:
                nn_model = pickle.load(f)
            with open(f'{model_dir}/scaler_{suffix}.pkl', 'rb') as f:
                scaler = pickle.load(f)

            self.models = (xgb_model, lgb_model, rf_model, nn_model, scaler)
            self.logger.info(f"✓ Loaded {suffix} models successfully")

        except Exception as e:
            self.logger.error(f"✗ Failed to load models: {e}")
            self.logger.error("⚠️  Bot cannot run without models")
            raise

    def fetch_market_data(self, lookback_days: int = 365):
        """
        Fetch latest market data for all tickers

        Args:
            lookback_days: Days of historical data to fetch
        """
        self.logger.info("📊 Fetching market data...")

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        assets = load_multi_asset_real_data(
            tickers=self.config.tickers,
            start_date=start_date
        )

        # Add features
        for ticker, df in assets.items():
            df_with_features, features = add_features(df.copy())
            df_final = finalize_features(df_with_features, features)
            self.market_data[ticker] = {
                'data': df_final,
                'features': features
            }
            self.logger.info(f"✓ {ticker}: {len(df_final)} days with {len(features)} features")

    def generate_signals(self) -> Dict[str, Dict]:
        """
        Generate trading signals for all tickers

        Returns:
            Dictionary of {ticker: signal_info}
        """
        signals = {}
        xgb_model, lgb_model, rf_model, nn_model, scaler = self.models

        for ticker in self.config.tickers:
            # Skip if already in position
            if ticker in self.positions:
                continue

            try:
                asset_data = self.market_data[ticker]['data']
                features = self.market_data[ticker]['features']

                # Get latest features
                X = asset_data[features].iloc[-1:].fillna(0).values

                # Get predictions from all models
                xgb_prob = xgb_model.predict_proba(X)[0, 1]
                lgb_prob = lgb_model.predict_proba(X)[0, 1]
                rf_prob = rf_model.predict_proba(X)[0, 1]

                X_scaled = scaler.transform(X)
                nn_prob = nn_model.predict_proba(X_scaled)[0, 1]

                # Ensemble voting
                predictions = [xgb_prob, lgb_prob, rf_prob, nn_prob]
                votes = sum([1 for p in predictions if p > 0.5])
                avg_prob = np.mean(predictions)
                ensemble_agreement = votes / 4.0

                # Check if signal meets thresholds
                if (votes >= 2 and  # At least 2/4 models agree
                    avg_prob >= self.config.min_signal_probability):

                    signals[ticker] = {
                        'prob': avg_prob,
                        'agreement': ensemble_agreement,
                        'votes': votes,
                        'price': asset_data['Close'].iloc[-1],
                        'predictions': {
                            'xgb': xgb_prob,
                            'lgb': lgb_prob,
                            'rf': rf_prob,
                            'nn': nn_prob
                        }
                    }

                    self.logger.info(f"📈 {ticker} signal: prob={avg_prob:.3f}, agreement={ensemble_agreement:.2f}, votes={votes}/4")

            except Exception as e:
                self.logger.error(f"✗ Error generating signal for {ticker}: {e}")
                continue

        return signals

    def calculate_position_size(self, signal: Dict) -> Tuple[float, float]:
        """
        Calculate position size and leverage based on signal strength

        Args:
            signal: Signal information

        Returns:
            (position_value, leverage)
        """
        # Base position size
        base_size = self.cash * self.config.max_position_size_pct

        # Dynamic leverage based on signal strength
        leverage = 1.0

        if self.config.use_leverage:
            # Increase leverage based on model agreement
            if signal['agreement'] >= 0.75:  # 3/4 models
                leverage = 1.5
            if signal['agreement'] >= 1.0:  # All 4 models
                leverage = 1.8

            # Bonus for high probability
            if signal['prob'] >= 0.60:
                leverage += 0.2

            # Cap at max leverage
            leverage = min(leverage, self.config.max_leverage)

        position_value = base_size * leverage

        # Limit to available cash
        position_value = min(position_value, self.cash)

        return position_value, leverage

    def execute_buy_order(self, ticker: str, signal: Dict):
        """
        Execute a buy order

        Args:
            ticker: Ticker symbol
            signal: Signal information
        """
        if self.config.mode == TradingMode.PAPER:
            self._execute_paper_buy(ticker, signal)
        else:
            self._execute_live_buy(ticker, signal)

    def _execute_paper_buy(self, ticker: str, signal: Dict):
        """Execute paper trading buy order"""
        position_value, leverage = self.calculate_position_size(signal)

        if position_value < 1000:  # Minimum position size
            self.logger.info(f"⚠️  {ticker}: Position too small (${position_value:.0f}), skipping")
            return

        price = signal['price']
        shares = position_value / price

        # Calculate stop loss
        stop_loss = price * (1 - self.config.stop_loss_pct)

        # Create position
        position = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            entry_date=datetime.now(),
            entry_value=position_value,
            leverage=leverage,
            signal_prob=signal['prob'],
            stop_loss=stop_loss
        )

        self.positions[ticker] = position
        self.cash -= position_value

        self.logger.info("="*70)
        self.logger.info(f"✅ BUY ORDER EXECUTED: {ticker}")
        self.logger.info(f"   Price: ${price:.2f}")
        self.logger.info(f"   Shares: {shares:.2f}")
        self.logger.info(f"   Value: ${position_value:,.0f}")
        self.logger.info(f"   Leverage: {leverage:.2f}x")
        self.logger.info(f"   Signal Prob: {signal['prob']:.3f}")
        self.logger.info(f"   Agreement: {signal['agreement']:.2f} ({signal['votes']}/4)")
        self.logger.info(f"   Stop Loss: ${stop_loss:.2f}")
        self.logger.info(f"   Cash Remaining: ${self.cash:,.0f}")
        self.logger.info("="*70)

    def _execute_live_buy(self, ticker: str, signal: Dict):
        """Execute live trading buy order (placeholder for broker integration)"""
        self.logger.warning("⚠️  Live trading not yet implemented")
        self.logger.warning("⚠️  Use TradingMode.PAPER for paper trading")
        # TODO: Integrate with Interactive Brokers / Alpaca / etc.

    def execute_sell_order(self, ticker: str, reason: str):
        """
        Execute a sell order

        Args:
            ticker: Ticker symbol
            reason: Reason for selling
        """
        if self.config.mode == TradingMode.PAPER:
            self._execute_paper_sell(ticker, reason)
        else:
            self._execute_live_sell(ticker, reason)

    def _execute_paper_sell(self, ticker: str, reason: str):
        """Execute paper trading sell order"""
        position = self.positions[ticker]
        exit_price = self.market_data[ticker]['data']['Close'].iloc[-1]
        exit_value = position.current_value(exit_price)

        pnl = position.pnl(exit_price)
        pnl_pct = position.pnl_pct(exit_price)

        # Update cash
        self.cash += exit_value

        # Record trade
        trade = {
            'ticker': ticker,
            'entry_date': position.entry_date,
            'exit_date': datetime.now(),
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'shares': position.shares,
            'entry_value': position.entry_value,
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': position.days_held(datetime.now()),
            'leverage': position.leverage,
            'signal_prob': position.signal_prob,
            'exit_reason': reason
        }

        self.trade_history.append(trade)

        # Remove position
        del self.positions[ticker]

        self.logger.info("="*70)
        self.logger.info(f"✅ SELL ORDER EXECUTED: {ticker}")
        self.logger.info(f"   Exit Price: ${exit_price:.2f}")
        self.logger.info(f"   Exit Value: ${exit_value:,.0f}")
        self.logger.info(f"   P&L: ${pnl:,.0f} ({pnl_pct:+.2f}%)")
        self.logger.info(f"   Days Held: {trade['days_held']}")
        self.logger.info(f"   Reason: {reason}")
        self.logger.info(f"   Cash Now: ${self.cash:,.0f}")
        self.logger.info("="*70)

    def _execute_live_sell(self, ticker: str, reason: str):
        """Execute live trading sell order (placeholder for broker integration)"""
        self.logger.warning("⚠️  Live trading not yet implemented")
        # TODO: Integrate with Interactive Brokers / Alpaca / etc.

    def check_exit_conditions(self):
        """Check if any positions should be exited"""
        current_date = datetime.now()

        for ticker, position in list(self.positions.items()):
            current_price = self.market_data[ticker]['data']['Close'].iloc[-1]

            # Check stop loss
            if current_price <= position.stop_loss:
                self.execute_sell_order(ticker, f"Stop loss hit (${position.stop_loss:.2f})")
                continue

            # Check holding period
            if position.days_held(current_date) >= self.config.holding_period_days:
                self.execute_sell_order(ticker, f"Holding period reached ({self.config.holding_period_days} days)")
                continue

    def run_trading_cycle(self):
        """Execute one trading cycle"""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"🔄 TRADING CYCLE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)

        # 1. Fetch latest market data
        self.fetch_market_data()

        # 2. Check exit conditions for existing positions
        self.check_exit_conditions()

        # 3. Calculate portfolio value
        portfolio_value = self.cash
        for ticker, position in self.positions.items():
            current_price = self.market_data[ticker]['data']['Close'].iloc[-1]
            portfolio_value += position.current_value(current_price)

        self.logger.info(f"\n📊 PORTFOLIO STATUS:")
        self.logger.info(f"   Cash: ${self.cash:,.0f}")
        self.logger.info(f"   Positions: {len(self.positions)}")
        self.logger.info(f"   Total Value: ${portfolio_value:,.0f}")

        # 4. Generate new signals if capacity available for more positions
        if len(self.positions) < self.config.max_positions:
            signals = self.generate_signals()

            if signals:
                # Sort by probability (best first)
                sorted_signals = sorted(signals.items(), key=lambda x: x[1]['prob'], reverse=True)

                # Execute orders for best signals
                positions_to_open = self.config.max_positions - len(self.positions)

                for ticker, signal in sorted_signals[:positions_to_open]:
                    self.execute_buy_order(ticker, signal)
            else:
                self.logger.info("⚠️  No signals generated")

        self.logger.info("="*70)
        self.logger.info("✓ Trading cycle complete")
        self.logger.info("="*70 + "\n")

    def run_live(self, check_interval_minutes: int = 60):
        """
        Run bot in live mode with periodic checks

        Args:
            check_interval_minutes: Minutes between trading cycles
        """
        self.logger.info("🚀 Starting live trading bot...")
        self.logger.info(f"Check interval: {check_interval_minutes} minutes")

        try:
            while True:
                self.run_trading_cycle()

                # Wait for next cycle
                self.logger.info(f"💤 Sleeping for {check_interval_minutes} minutes...")
                time.sleep(check_interval_minutes * 60)

        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Bot stopped by user")
            self.print_summary()

        except Exception as e:
            self.logger.error(f"✗ Fatal error: {e}")
            raise

    def print_summary(self):
        """Print trading summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("TRADING SUMMARY")
        self.logger.info("="*70)

        # Portfolio value
        portfolio_value = self.cash
        for ticker, position in self.positions.items():
            current_price = self.market_data[ticker]['data']['Close'].iloc[-1]
            portfolio_value += position.current_value(current_price)

        total_pnl = portfolio_value - self.config.initial_capital
        total_return_pct = (total_pnl / self.config.initial_capital) * 100

        self.logger.info(f"\n💰 PERFORMANCE:")
        self.logger.info(f"   Initial Capital: ${self.config.initial_capital:,.0f}")
        self.logger.info(f"   Current Value:   ${portfolio_value:,.0f}")
        self.logger.info(f"   P&L:            ${total_pnl:,.0f} ({total_return_pct:+.2f}%)")

        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            avg_pnl = trades_df['pnl'].mean()
            avg_pnl_pct = trades_df['pnl_pct'].mean()

            self.logger.info(f"\n📈 TRADING STATS:")
            self.logger.info(f"   Total Trades: {len(trades_df)}")
            self.logger.info(f"   Win Rate:     {win_rate:.2f}%")
            self.logger.info(f"   Avg P&L:      ${avg_pnl:,.0f} ({avg_pnl_pct:+.2f}%)")

        self.logger.info("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function to run the trading bot"""

    # Create configuration
    config = TradingConfig(
        tickers=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
        initial_capital=100000,
        max_positions=3,
        use_leverage=True,
        max_leverage=2.0,
        holding_period_days=10,
        mode=TradingMode.PAPER,  # Start with paper trading
        model_suffix='ultimate',  # Try: ultimate, max_perf, regime
    )

    # Create and run bot
    bot = TradingBot(config)

    # Run single cycle (for testing)
    print("\n🎯 Running single trading cycle (test mode)...")
    bot.run_trading_cycle()

    # Print summary
    bot.print_summary()

    # To run continuously:
    # bot.run_live(check_interval_minutes=60)  # Check every hour


if __name__ == '__main__':
    main()
