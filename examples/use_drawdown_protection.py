"""
Example: Integrating Drawdown Protection with Trading System
Shows how to add comprehensive drawdown management to trading strategies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.drawdown_protection import DrawdownProtection, TradingStatus
import pandas as pd
import numpy as np


def example_1_basic_protection():
    """Example 1: Basic drawdown protection"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Drawdown Protection")
    print("="*70)

    # Initialize with conservative settings
    protection = DrawdownProtection(
        initial_capital=100000,
        max_drawdown_pct=0.10,      # Reduce size at -10%
        halt_drawdown_pct=0.15,     # Stop trading at -15%
        emergency_drawdown_pct=0.20,  # Exit all at -20%
        daily_loss_limit_pct=0.03   # Max -3% per day
    )

    # Simulate trading day
    portfolio_values = [100000, 102000, 98000, 95000, 92000, 88000]

    for value in portfolio_values:
        state = protection.update(current_value=value)
        can_trade, reason = protection.can_trade()

        print(f"\nValue: ${value:,} ({(value/100000-1)*100:+.1f}%)")
        print(f"  Drawdown: {state.drawdown_pct*100:.1f}%")
        print(f"  Status: {state.status.value}")
        print(f"  Can trade: {can_trade}")
        if not can_trade or state.status == TradingStatus.REDUCED:
            print(f"  Reason: {reason}")


def example_2_dynamic_position_sizing():
    """Example 2: Dynamic position sizing based on drawdown"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Dynamic Position Sizing")
    print("="*70)

    protection = DrawdownProtection(
        initial_capital=100000,
        max_drawdown_pct=0.10,
        halt_drawdown_pct=0.15
    )

    # Base position size
    base_size = 10000

    # Simulate different drawdown levels
    scenarios = [
        (100000, "Normal (no drawdown)"),
        (95000, "Small drawdown (-5%)"),
        (92000, "Medium drawdown (-8%)"),
        (88000, "Large drawdown (-12%)"),
        (85000, "Critical drawdown (-15%)")
    ]

    for value, description in scenarios:
        protection.update(current_value=value)
        size, reason = protection.calculate_position_size(
            base_size=base_size,
            risk_per_trade=0.04  # 4% risk per trade
        )

        print(f"\n{description}:")
        print(f"  Portfolio: ${value:,}")
        print(f"  Base size: ${base_size:,}")
        print(f"  Adjusted size: ${size:,.0f} ({size/base_size*100:.0f}% of base)")
        print(f"  Reason: {reason}")


def example_3_portfolio_heat_limit():
    """Example 3: Portfolio heat limit"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Portfolio Heat Limit")
    print("="*70)

    protection = DrawdownProtection(
        initial_capital=100000,
        portfolio_heat_limit_pct=0.06  # Max 6% total risk
    )

    # Add multiple positions
    positions = [
        ("SPY", 10000, 450, 445),   # $5 risk = 1.1%
        ("QQQ", 8000, 380, 375),     # $5 risk = 1.05%
        ("BTC", 5000, 65000, 60000)  # $5000 risk = 3.85%
    ]

    print("\nAdding positions (6% heat limit):")
    print(f"Starting heat: {protection.current_heat*100:.2f}%\n")

    for ticker, size_dollars, entry, stop in positions:
        # Calculate shares
        shares = size_dollars / entry

        # Add position
        success = protection.add_position(ticker, shares, entry, stop)

        # Try to add another position
        new_size, reason = protection.calculate_position_size(
            base_size=10000,
            risk_per_trade=abs(entry - stop)
        )

        print(f"\n{ticker} Position:")
        print(f"  Entry: ${entry}, Stop: ${stop}")
        print(f"  Risk: {abs(entry-stop)/entry*100:.2f}%")
        print(f"  Current heat: {protection.current_heat*100:.2f}%")
        print(f"  Can add more: ${new_size:,.0f}")
        if new_size < 10000:
            print(f"  Reason: {reason}")


def example_4_trailing_stops():
    """Example 4: Trailing stops to protect profits"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Trailing Stops")
    print("="*70)

    protection = DrawdownProtection(initial_capital=100000)

    # Position details
    entry_price = 450
    initial_stop = 445

    # Simulate price increases
    prices = [450, 455, 460, 465, 470, 465, 460]

    print(f"\nPosition: Entry at ${entry_price}, Initial stop at ${initial_stop}\n")

    current_stop = initial_stop
    for price in prices:
        # Update trailing stop (protect 50% of profit)
        new_stop = protection.get_trailing_stop(
            entry_price=entry_price,
            current_price=price,
            current_stop=current_stop,
            trail_pct=0.50  # Protect 50% of profit
        )

        profit = price - entry_price
        protected_profit = new_stop - entry_price

        print(f"Price: ${price}")
        print(f"  Profit: ${profit:+.2f} ({profit/entry_price*100:+.1f}%)")
        print(f"  Stop: ${new_stop:.2f}")
        print(f"  Protected profit: ${protected_profit:+.2f}\n")

        current_stop = new_stop


def example_5_complete_trading_day():
    """Example 5: Complete trading day with protection"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Trading Day Simulation")
    print("="*70)

    protection = DrawdownProtection(
        initial_capital=100000,
        max_drawdown_pct=0.10,
        halt_drawdown_pct=0.15,
        daily_loss_limit_pct=0.03,
        portfolio_heat_limit_pct=0.06
    )

    # Start of day
    print("\nMORNING (9:30 AM) - Portfolio: $100,000")
    state = protection.update(100000)

    # Trade 1: Win
    print("\n[10:00 AM] Signal: Buy SPY")
    size, reason = protection.calculate_position_size(base_size=10000, risk_per_trade=0.04)
    print(f"  Position size: ${size:,.0f}")
    if size > 0:
        protection.add_position("SPY", 100, 450, 445)
        print("  Position opened")

    # Trade 2: Win
    print("\n[11:00 AM] Signal: Buy QQQ")
    state = protection.update(101000)
    size, reason = protection.calculate_position_size(base_size=10000, risk_per_trade=0.04)
    print(f"  Portfolio: ${state.current_value:,} (+1.0%)")
    print(f"  Position size: ${size:,.0f}")

    # Market moves against us
    print("\n[1:00 PM] Market selloff")
    state = protection.update(97000)
    print(f"  Portfolio: ${state.current_value:,} (-3.0%)")
    print(f"  Daily loss: {protection.today_loss_pct*100:.2f}%")

    # Check if can still trade
    can_trade, reason = protection.can_trade()
    print(f"  Can trade: {can_trade}")
    if not can_trade:
        print(f"  Reason: {reason}")

    # End of day report
    print(protection.get_status_report())


def example_6_integration_with_backtest():
    """Example 6: Integration with backtesting"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Backtest with Drawdown Protection")
    print("="*70)

    # Simulate backtest equity curve
    np.random.seed(42)
    days = 252
    returns = np.random.randn(days) * 0.02 + 0.0008  # ~20% annualized

    # Initialize protection
    protection = DrawdownProtection(
        initial_capital=100000,
        max_drawdown_pct=0.10,
        halt_drawdown_pct=0.15,
        emergency_drawdown_pct=0.20
    )

    # Run backtest
    equity = [100000]
    max_dd_without_protection = 0
    max_dd_with_protection = 0
    trades_skipped = 0
    trades_executed = 0

    for day_return in returns:
        current_value = equity[-1]

        # Update protection
        state = protection.update(current_value)
        can_trade, reason = protection.can_trade()

        # Simulate trade
        if can_trade and state.status != TradingStatus.HALTED:
            # Execute trade
            new_value = current_value * (1 + day_return)
            trades_executed += 1
        else:
            # Skip trade (protection active)
            new_value = current_value
            trades_skipped += 1

        equity.append(new_value)

        # Track drawdowns
        peak = max(equity)
        dd_without = (peak - equity[-1]) / peak
        max_dd_without_protection = max(max_dd_without_protection, dd_without)
        max_dd_with_protection = max(max_dd_with_protection, state.drawdown_pct)

    # Results
    equity = np.array(equity)
    total_return = (equity[-1] / equity[0] - 1) * 100
    annualized = ((equity[-1] / equity[0]) ** (252/days) - 1) * 100

    print(f"\nBacktest Results ({days} days):")
    print(f"\n  Performance:")
    print(f"    Final value: ${equity[-1]:,.2f}")
    print(f"    Total return: {total_return:+.2f}%")
    print(f"    Annualized: {annualized:.2f}%")

    print(f"\n  Risk Management:")
    print(f"    Max DD (no protection): {max_dd_without_protection*100:.2f}%")
    print(f"    Max DD (with protection): {max_dd_with_protection*100:.2f}%")
    print(f"    Improvement: {(max_dd_without_protection - max_dd_with_protection)*100:.2f}pp")

    print(f"\n  Trading Activity:")
    print(f"    Trades executed: {trades_executed}")
    print(f"    Trades skipped: {trades_skipped}")
    print(f"    Protection rate: {trades_skipped/(trades_executed+trades_skipped)*100:.1f}%")


if __name__ == '__main__':
    # Run all examples
    example_1_basic_protection()
    example_2_dynamic_position_sizing()
    example_3_portfolio_heat_limit()
    example_4_trailing_stops()
    example_5_complete_trading_day()
    example_6_integration_with_backtest()

    print("\n" + "="*70)
    print("✓ All drawdown protection examples complete")
    print("="*70)
