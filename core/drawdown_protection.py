"""
Drawdown Protection System
Implements comprehensive drawdown management and circuit breakers

Features:
- Real-time drawdown monitoring
- Automatic circuit breakers
- Dynamic position sizing based on drawdown
- Portfolio heat limits
- Trailing stops
- Kill switches for extreme losses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class TradingStatus(Enum):
    """Trading system status"""
    ACTIVE = "active"
    REDUCED = "reduced"  # Reduced position sizing
    HALTED = "halted"    # No new positions
    EMERGENCY_EXIT = "emergency_exit"  # Close all positions


@dataclass
class DrawdownState:
    """Current drawdown state"""
    current_value: float
    peak_value: float
    drawdown_pct: float
    underwater_days: int
    status: TradingStatus
    last_update: datetime


class DrawdownProtection:
    """
    Comprehensive drawdown protection system

    Protections:
    1. Circuit Breakers - Stop trading at specific drawdown thresholds
    2. Dynamic Position Sizing - Reduce size during drawdowns
    3. Portfolio Heat Limit - Max % of portfolio at risk
    4. Trailing Stops - Protect profits
    5. Daily Loss Limits - Stop loss per day
    6. Emergency Exit - Close all at extreme loss
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 max_drawdown_pct: float = 0.15,
                 halt_drawdown_pct: float = 0.10,
                 emergency_drawdown_pct: float = 0.20,
                 daily_loss_limit_pct: float = 0.03,
                 portfolio_heat_limit_pct: float = 0.06):
        """
        Initialize drawdown protection

        Args:
            initial_capital: Starting capital
            max_drawdown_pct: Max acceptable drawdown before reducing size (10-15%)
            halt_drawdown_pct: Drawdown level to halt new trades (8-12%)
            emergency_drawdown_pct: Emergency exit threshold (15-25%)
            daily_loss_limit_pct: Max loss per day (2-5%)
            portfolio_heat_limit_pct: Max % of portfolio at risk (4-8%)
        """
        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.halt_drawdown_pct = halt_drawdown_pct
        self.emergency_drawdown_pct = emergency_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.portfolio_heat_limit_pct = portfolio_heat_limit_pct

        # State tracking
        self.peak_value = initial_capital
        self.current_value = initial_capital
        self.drawdown_pct = 0.0
        self.status = TradingStatus.ACTIVE
        self.underwater_since = None

        # Daily tracking
        self.today_start_value = initial_capital
        self.today_loss_pct = 0.0
        self.last_daily_reset = datetime.now().date()

        # Position tracking
        self.open_positions: List[Dict] = []
        self.current_heat = 0.0  # Total risk as % of portfolio

        print(f"✓ Drawdown Protection initialized")
        print(f"  Max drawdown: {max_drawdown_pct*100:.1f}%")
        print(f"  Halt threshold: {halt_drawdown_pct*100:.1f}%")
        print(f"  Emergency exit: {emergency_drawdown_pct*100:.1f}%")
        print(f"  Daily loss limit: {daily_loss_limit_pct*100:.1f}%")

    def update(self, current_value: float) -> DrawdownState:
        """
        Update portfolio value and check protections

        Args:
            current_value: Current portfolio value

        Returns:
            Current drawdown state
        """
        self.current_value = current_value

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.underwater_since = None

        # Calculate drawdown
        self.drawdown_pct = (self.peak_value - current_value) / self.peak_value

        # Update underwater duration
        if self.drawdown_pct > 0:
            if self.underwater_since is None:
                self.underwater_since = datetime.now()
            underwater_days = (datetime.now() - self.underwater_since).days
        else:
            underwater_days = 0

        # Reset daily tracking if new day
        if datetime.now().date() > self.last_daily_reset:
            self.today_start_value = current_value
            self.last_daily_reset = datetime.now().date()

        # Calculate daily loss
        self.today_loss_pct = (self.today_start_value - current_value) / self.today_start_value

        # Determine status based on protections
        previous_status = self.status
        self.status = self._determine_status()

        # Log status changes
        if self.status != previous_status:
            self._log_status_change(previous_status, self.status)

        return DrawdownState(
            current_value=current_value,
            peak_value=self.peak_value,
            drawdown_pct=self.drawdown_pct,
            underwater_days=underwater_days,
            status=self.status,
            last_update=datetime.now()
        )

    def _determine_status(self) -> TradingStatus:
        """Determine trading status based on current conditions"""

        # Emergency exit - close everything
        if self.drawdown_pct >= self.emergency_drawdown_pct:
            return TradingStatus.EMERGENCY_EXIT

        # Daily loss limit exceeded - halt for today
        if self.today_loss_pct >= self.daily_loss_limit_pct:
            return TradingStatus.HALTED

        # Drawdown halt threshold - no new positions
        if self.drawdown_pct >= self.halt_drawdown_pct:
            return TradingStatus.HALTED

        # Drawdown warning - reduce position sizes
        if self.drawdown_pct >= self.max_drawdown_pct:
            return TradingStatus.REDUCED

        # All clear
        return TradingStatus.ACTIVE

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed

        Returns:
            (can_trade: bool, reason: str)
        """
        if self.status == TradingStatus.EMERGENCY_EXIT:
            return False, "EMERGENCY EXIT MODE - Close all positions immediately"

        if self.status == TradingStatus.HALTED:
            if self.today_loss_pct >= self.daily_loss_limit_pct:
                return False, f"Daily loss limit reached ({self.today_loss_pct*100:.2f}%)"
            return False, f"Trading halted due to {self.drawdown_pct*100:.2f}% drawdown"

        if self.status == TradingStatus.REDUCED:
            return True, f"Trading with reduced size due to {self.drawdown_pct*100:.2f}% drawdown"

        return True, "Normal trading"

    def calculate_position_size(self, base_size: float, risk_per_trade: float) -> Tuple[float, str]:
        """
        Calculate position size with drawdown-adjusted scaling

        Args:
            base_size: Base position size
            risk_per_trade: Risk per trade (e.g., distance to stop loss)

        Returns:
            (adjusted_size, adjustment_reason)
        """
        adjusted_size = base_size

        # Check if we can trade
        can_trade, reason = self.can_trade()
        if not can_trade and self.status != TradingStatus.REDUCED:
            return 0.0, reason

        # Reduce size based on drawdown level
        if self.status == TradingStatus.REDUCED:
            # Scale down linearly from max_drawdown to halt_drawdown
            scale_factor = 1.0 - (self.drawdown_pct - self.max_drawdown_pct) / \
                          (self.halt_drawdown_pct - self.max_drawdown_pct)
            scale_factor = max(0.25, min(1.0, scale_factor))  # 25% minimum
            adjusted_size *= scale_factor
            reason = f"Size reduced to {scale_factor*100:.0f}% due to {self.drawdown_pct*100:.1f}% drawdown"

        # Check portfolio heat limit
        new_heat = self.current_heat + (risk_per_trade * adjusted_size / self.current_value)

        if new_heat > self.portfolio_heat_limit_pct:
            # Reduce size to stay within heat limit
            available_heat = self.portfolio_heat_limit_pct - self.current_heat
            if available_heat <= 0:
                return 0.0, f"Portfolio heat limit reached ({self.current_heat*100:.2f}%)"

            max_size_for_heat = (available_heat * self.current_value) / risk_per_trade
            if max_size_for_heat < adjusted_size:
                adjusted_size = max_size_for_heat
                reason = f"Size limited by portfolio heat ({new_heat*100:.2f}%)"

        return adjusted_size, reason

    def add_position(self, ticker: str, size: float, entry_price: float,
                    stop_loss: float) -> bool:
        """
        Add new position to tracking

        Args:
            ticker: Asset ticker
            size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Success boolean
        """
        # Calculate risk
        risk_per_unit = abs(entry_price - stop_loss)
        total_risk = risk_per_unit * size
        risk_pct = total_risk / self.current_value

        # Update heat
        self.current_heat += risk_pct

        # Track position
        self.open_positions.append({
            'ticker': ticker,
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_pct': risk_pct,
            'entry_time': datetime.now()
        })

        print(f"  Position added: {ticker}, Heat: {self.current_heat*100:.2f}%")
        return True

    def remove_position(self, ticker: str):
        """Remove closed position"""
        for i, pos in enumerate(self.open_positions):
            if pos['ticker'] == ticker:
                self.current_heat -= pos['risk_pct']
                del self.open_positions[i]
                print(f"  Position removed: {ticker}, Heat: {self.current_heat*100:.2f}%")
                return

    def should_close_all_positions(self) -> bool:
        """Check if all positions should be emergency closed"""
        return self.status == TradingStatus.EMERGENCY_EXIT

    def get_trailing_stop(self, entry_price: float, current_price: float,
                         current_stop: float, trail_pct: float = 0.50) -> float:
        """
        Calculate trailing stop to protect profits

        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_stop: Current stop loss
            trail_pct: Percent of profit to trail (0.5 = protect 50% of profit)

        Returns:
            New stop loss price
        """
        profit = current_price - entry_price

        if profit > 0:
            # Trail stop to protect percentage of profit
            new_stop = entry_price + (profit * trail_pct)
            return max(new_stop, current_stop)

        return current_stop

    def get_status_report(self) -> str:
        """Get detailed status report"""
        underwater_str = ""
        if self.underwater_since:
            days = (datetime.now() - self.underwater_since).days
            underwater_str = f"\n  Underwater for: {days} days"

        return f"""
DRAWDOWN PROTECTION STATUS
{'='*60}
Current Value: ${self.current_value:,.2f}
Peak Value: ${self.peak_value:,.2f}
Drawdown: {self.drawdown_pct*100:.2f}%{underwater_str}

Daily Performance:
  Start: ${self.today_start_value:,.2f}
  Loss: {self.today_loss_pct*100:.2f}%

Portfolio Risk:
  Current Heat: {self.current_heat*100:.2f}%
  Heat Limit: {self.portfolio_heat_limit_pct*100:.2f}%
  Open Positions: {len(self.open_positions)}

Trading Status: {self.status.value.upper()}
{'='*60}

Thresholds:
  Reduce Size: {self.max_drawdown_pct*100:.1f}%
  Halt Trading: {self.halt_drawdown_pct*100:.1f}%
  Emergency Exit: {self.emergency_drawdown_pct*100:.1f}%
  Daily Loss Limit: {self.daily_loss_limit_pct*100:.1f}%
"""

    def _log_status_change(self, old_status: TradingStatus, new_status: TradingStatus):
        """Log status changes"""
        print(f"\n⚠️  STATUS CHANGE: {old_status.value} → {new_status.value}")
        print(f"   Drawdown: {self.drawdown_pct*100:.2f}%")
        print(f"   Daily Loss: {self.today_loss_pct*100:.2f}%")

        if new_status == TradingStatus.EMERGENCY_EXIT:
            print("   🚨 EMERGENCY EXIT - CLOSE ALL POSITIONS 🚨")
        elif new_status == TradingStatus.HALTED:
            print("   ⏸️  TRADING HALTED - No new positions")
        elif new_status == TradingStatus.REDUCED:
            print("   ⚡ REDUCED MODE - Smaller position sizes")


def demo_drawdown_protection():
    """Demonstrate drawdown protection system"""
    print("\n" + "="*70)
    print("DRAWDOWN PROTECTION DEMO")
    print("="*70)

    # Initialize with $100k
    protection = DrawdownProtection(
        initial_capital=100000,
        max_drawdown_pct=0.10,     # Reduce size at -10%
        halt_drawdown_pct=0.15,    # Stop trading at -15%
        emergency_drawdown_pct=0.20,  # Exit all at -20%
        daily_loss_limit_pct=0.03,    # Max -3% per day
        portfolio_heat_limit_pct=0.06  # Max 6% total risk
    )

    # Simulate drawdown scenario
    print("\n" + "-"*70)
    print("Scenario 1: Normal Trading")
    print("-"*70)

    state = protection.update(current_value=102000)
    can_trade, reason = protection.can_trade()
    print(f"Value: $102,000 (+2.0%)")
    print(f"Can trade: {can_trade} - {reason}")

    # Add position
    size, reason = protection.calculate_position_size(base_size=10000, risk_per_trade=0.04)
    print(f"Position size: ${size:,.0f} - {reason}")

    print("\n" + "-"*70)
    print("Scenario 2: -8% Drawdown (approaching limit)")
    print("-"*70)

    state = protection.update(current_value=92000)
    can_trade, reason = protection.can_trade()
    print(f"Value: $92,000 (-8.0% from peak)")
    print(f"Status: {state.status.value}")
    print(f"Can trade: {can_trade} - {reason}")

    print("\n" + "-"*70)
    print("Scenario 3: -12% Drawdown (reduce size)")
    print("-"*70)

    state = protection.update(current_value=88000)
    can_trade, reason = protection.can_trade()
    size, reason = protection.calculate_position_size(base_size=10000, risk_per_trade=0.04)
    print(f"Value: $88,000 (-12.0% from peak)")
    print(f"Status: {state.status.value}")
    print(f"Can trade: {can_trade}")
    print(f"Position size: ${size:,.0f} (reduced from $10,000)")

    print("\n" + "-"*70)
    print("Scenario 4: -16% Drawdown (halt trading)")
    print("-"*70)

    state = protection.update(current_value=84000)
    can_trade, reason = protection.can_trade()
    print(f"Value: $84,000 (-16.0% from peak)")
    print(f"Status: {state.status.value}")
    print(f"Can trade: {can_trade} - {reason}")

    print("\n" + "-"*70)
    print("Scenario 5: -22% Drawdown (emergency exit)")
    print("-"*70)

    state = protection.update(current_value=78000)
    can_trade, reason = protection.can_trade()
    emergency = protection.should_close_all_positions()
    print(f"Value: $78,000 (-22.0% from peak)")
    print(f"Status: {state.status.value}")
    print(f"Emergency exit: {emergency}")
    print(f"Action: {reason}")

    # Print full report
    print(protection.get_status_report())

    print("\n✓ Drawdown protection demo complete")


if __name__ == '__main__':
    demo_drawdown_protection()
