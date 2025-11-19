"""
Transaction Cost Modeling

This module implements realistic transaction cost modeling for both stocks and cryptocurrency.
Proper cost modeling is critical - many "profitable" backtests become unprofitable after costs.

Cost Components:
1. Commissions (broker fees)
2. Bid-Ask Spread
3. Slippage (market impact)
4. Exchange/Regulatory Fees
5. Financing Costs (for leveraged positions)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class AssetClass(Enum):
    """Asset class for different cost structures"""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"


@dataclass
class TradeCost:
    """Breakdown of costs for a single trade"""
    commission: float
    spread_cost: float
    slippage: float
    exchange_fees: float
    total_cost: float
    cost_pct: float  # As percentage of trade value


@dataclass
class PositionCost:
    """Total cost for opening and closing a position"""
    entry_cost: float
    exit_cost: float
    financing_cost: float  # For leveraged positions
    total_cost: float
    cost_pct: float  # As percentage of position value


class TransactionCostModel:
    """
    Models realistic transaction costs for different asset classes

    Usage:
        model = TransactionCostModel(asset_class=AssetClass.STOCK)
        cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.02
        )
    """

    def __init__(self, asset_class: AssetClass = AssetClass.STOCK,
                 commission_pct: float = 0.0,
                 min_commission: float = 0.0):
        """
        Initialize transaction cost model

        Args:
            asset_class: Type of asset (STOCK, CRYPTO, ETF)
            commission_pct: Commission as percentage (default 0 for zero-commission brokers)
            min_commission: Minimum commission per trade (default $0)
        """
        self.asset_class = asset_class
        self.commission_pct = commission_pct
        self.min_commission = min_commission

        # Set default parameters by asset class
        self._set_defaults()

    def _set_defaults(self):
        """Set default cost parameters by asset class"""
        if self.asset_class == AssetClass.STOCK:
            # Stock trading costs (US markets)
            self.spread_pct_normal = 0.05  # 5 basis points for liquid stocks
            self.spread_pct_volatile = 0.15  # 15 bps during high volatility
            self.slippage_base = 0.02  # 2 bps base slippage
            self.slippage_volatility_factor = 0.5  # Slippage increases with volatility
            self.sec_fee_pct = 0.000278  # SEC Section 31 fee (on sales only)
            self.finra_taf = 0.000166  # FINRA Trading Activity Fee (on sales only)

        elif self.asset_class == AssetClass.CRYPTO:
            # Cryptocurrency trading costs
            self.spread_pct_normal = 0.10  # 10 bps for major crypto pairs
            self.spread_pct_volatile = 0.30  # 30 bps during high volatility
            self.slippage_base = 0.05  # 5 bps base slippage
            self.slippage_volatility_factor = 1.0  # Higher volatility impact
            self.exchange_fee_maker = 0.10  # 10 bps maker fee
            self.exchange_fee_taker = 0.15  # 15 bps taker fee (more common)
            self.withdrawal_fee = 0.05  # 5 bps average withdrawal fee
            self.network_fee_btc = 0.02  # 2 bps average network fee

        elif self.asset_class == AssetClass.ETF:
            # ETF trading costs
            self.spread_pct_normal = 0.03  # 3 bps for liquid ETFs
            self.spread_pct_volatile = 0.10  # 10 bps during volatility
            self.slippage_base = 0.01  # 1 bp base slippage
            self.slippage_volatility_factor = 0.3
            self.sec_fee_pct = 0.000278  # SEC fees apply
            self.finra_taf = 0.000166

    def calculate_trade_cost(self, trade_value: float, is_buy: bool,
                            volatility: float = 0.01,
                            is_market_order: bool = True,
                            liquidity_tier: str = "high") -> TradeCost:
        """
        Calculate total cost for a single trade

        Args:
            trade_value: Dollar value of trade
            is_buy: True for buy, False for sell
            volatility: Current volatility (default 1%)
            is_market_order: True for market order, False for limit (affects slippage)
            liquidity_tier: 'high', 'medium', or 'low' (affects spread)

        Returns:
            TradeCost object with breakdown
        """
        # 1. Commission
        commission = max(trade_value * self.commission_pct, self.min_commission)

        # 2. Spread cost
        spread_cost = self._calculate_spread_cost(trade_value, volatility, liquidity_tier)

        # 3. Slippage
        if is_market_order:
            slippage = self._calculate_slippage(trade_value, volatility, liquidity_tier)
        else:
            slippage = 0  # Limit orders don't have slippage (but may not fill)

        # 4. Exchange/regulatory fees
        exchange_fees = self._calculate_exchange_fees(trade_value, is_buy)

        # Total cost
        total_cost = commission + spread_cost + slippage + exchange_fees
        cost_pct = total_cost / trade_value

        return TradeCost(
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            exchange_fees=exchange_fees,
            total_cost=total_cost,
            cost_pct=cost_pct
        )

    def _calculate_spread_cost(self, trade_value: float, volatility: float,
                               liquidity_tier: str) -> float:
        """Calculate bid-ask spread cost"""
        # Base spread depends on volatility
        if volatility > 0.03:  # High volatility (>3% daily)
            spread_pct = self.spread_pct_volatile
        else:
            spread_pct = self.spread_pct_normal

        # Adjust for liquidity
        liquidity_multipliers = {
            'high': 1.0,
            'medium': 1.5,
            'low': 3.0
        }
        multiplier = liquidity_multipliers.get(liquidity_tier, 1.0)

        # You pay half the spread on each trade
        return trade_value * (spread_pct / 100) * multiplier * 0.5

    def _calculate_slippage(self, trade_value: float, volatility: float,
                           liquidity_tier: str) -> float:
        """Calculate slippage (market impact)"""
        # Base slippage
        slippage_pct = self.slippage_base

        # Increase with volatility
        slippage_pct += volatility * 100 * self.slippage_volatility_factor

        # Adjust for liquidity
        liquidity_multipliers = {
            'high': 1.0,
            'medium': 2.0,
            'low': 5.0
        }
        multiplier = liquidity_multipliers.get(liquidity_tier, 1.0)

        return trade_value * (slippage_pct / 100) * multiplier

    def _calculate_exchange_fees(self, trade_value: float, is_buy: bool) -> float:
        """Calculate exchange and regulatory fees"""
        if self.asset_class == AssetClass.STOCK or self.asset_class == AssetClass.ETF:
            # SEC and FINRA fees only on sells
            if not is_buy:
                sec_fee = trade_value * self.sec_fee_pct
                finra_fee = trade_value * self.finra_taf
                return sec_fee + finra_fee
            return 0.0

        elif self.asset_class == AssetClass.CRYPTO:
            # Crypto exchange fees (assume taker, more common)
            exchange_fee = trade_value * (self.exchange_fee_taker / 100)
            # Network fees (small, approximate)
            network_fee = trade_value * (self.network_fee_btc / 100)
            return exchange_fee + network_fee

        return 0.0

    def calculate_round_trip_cost(self, position_value: float, holding_days: int = 1,
                                  volatility: float = 0.01,
                                  use_leverage: bool = False,
                                  leverage_ratio: float = 1.0) -> PositionCost:
        """
        Calculate total cost for opening and closing a position (round trip)

        Args:
            position_value: Dollar value of position
            holding_days: Number of days position is held
            volatility: Current volatility
            use_leverage: Whether position uses leverage
            leverage_ratio: Leverage multiplier (e.g., 2.0 for 2x)

        Returns:
            PositionCost object with breakdown
        """
        # Entry cost
        entry = self.calculate_trade_cost(position_value, is_buy=True, volatility=volatility)

        # Exit cost (may be different due to regulatory fees)
        exit_trade = self.calculate_trade_cost(position_value, is_buy=False, volatility=volatility)

        # Financing cost (for leveraged positions)
        financing_cost = 0.0
        if use_leverage and leverage_ratio > 1.0:
            financing_cost = self._calculate_financing_cost(
                position_value, holding_days, leverage_ratio
            )

        total_cost = entry.total_cost + exit_trade.total_cost + financing_cost
        cost_pct = total_cost / position_value

        return PositionCost(
            entry_cost=entry.total_cost,
            exit_cost=exit_trade.total_cost,
            financing_cost=financing_cost,
            total_cost=total_cost,
            cost_pct=cost_pct
        )

    def _calculate_financing_cost(self, position_value: float, holding_days: int,
                                  leverage_ratio: float) -> float:
        """
        Calculate financing cost for leveraged positions

        Args:
            position_value: Position value
            holding_days: Days held
            leverage_ratio: Leverage multiplier

        Returns:
            Total financing cost
        """
        if self.asset_class == AssetClass.STOCK:
            # Margin interest (assume 8% annual for retail)
            annual_rate = 0.08
            borrowed_amount = position_value * (leverage_ratio - 1) / leverage_ratio
            daily_rate = annual_rate / 365
            return borrowed_amount * daily_rate * holding_days

        elif self.asset_class == AssetClass.CRYPTO:
            # Crypto funding rates (more expensive, ~0.01% per 8 hours = ~10% annual)
            annual_rate = 0.10
            borrowed_amount = position_value * (leverage_ratio - 1) / leverage_ratio
            funding_periods = holding_days * 3  # 3 funding periods per day
            per_period_rate = annual_rate / 365 / 3
            return borrowed_amount * per_period_rate * funding_periods

        return 0.0


class ConservativeCostModel(TransactionCostModel):
    """
    Conservative (pessimistic) cost model for robust backtesting

    Uses higher cost estimates to ensure strategy remains profitable
    even in worst-case execution scenarios
    """

    def _set_defaults(self):
        """Set conservative (higher) cost estimates"""
        super()._set_defaults()

        # Increase all costs by 50% for conservatism
        if hasattr(self, 'spread_pct_normal'):
            self.spread_pct_normal *= 1.5
            self.spread_pct_volatile *= 1.5
            self.slippage_base *= 1.5
            self.slippage_volatility_factor *= 1.5


def estimate_total_costs(total_trades: int, avg_trade_value: float,
                        asset_class: AssetClass = AssetClass.STOCK,
                        avg_holding_days: int = 5,
                        use_leverage: bool = False) -> Dict:
    """
    Estimate total transaction costs for a trading strategy

    Args:
        total_trades: Number of trades in backtest
        avg_trade_value: Average trade size in dollars
        asset_class: Type of asset
        avg_holding_days: Average days per position
        use_leverage: Whether strategy uses leverage

    Returns:
        Dictionary with cost breakdown and impact analysis
    """
    model = TransactionCostModel(asset_class=asset_class)

    # Calculate cost per round trip
    round_trip = model.calculate_round_trip_cost(
        position_value=avg_trade_value,
        holding_days=avg_holding_days,
        use_leverage=use_leverage,
        leverage_ratio=2.0 if use_leverage else 1.0
    )

    # Total costs
    num_round_trips = total_trades / 2  # Buy + sell = 1 round trip
    total_cost = round_trip.total_cost * num_round_trips
    total_cost_pct = round_trip.cost_pct * num_round_trips

    print(f"\n{'='*60}")
    print(f"Transaction Cost Analysis")
    print(f"{'='*60}")
    print(f"Asset Class: {asset_class.value}")
    print(f"Total Trades: {total_trades}")
    print(f"Round Trips: {num_round_trips:.0f}")
    print(f"Average Trade Value: ${avg_trade_value:,.0f}")
    print(f"\nCost Per Round Trip:")
    print(f"  Entry Cost: ${round_trip.entry_cost:.2f} ({round_trip.entry_cost/avg_trade_value*100:.3f}%)")
    print(f"  Exit Cost: ${round_trip.exit_cost:.2f} ({round_trip.exit_cost/avg_trade_value*100:.3f}%)")
    if round_trip.financing_cost > 0:
        print(f"  Financing Cost: ${round_trip.financing_cost:.2f} ({round_trip.financing_cost/avg_trade_value*100:.3f}%)")
    print(f"  Total: ${round_trip.total_cost:.2f} ({round_trip.cost_pct*100:.3f}%)")
    print(f"\nTotal Strategy Costs:")
    print(f"  Dollar Cost: ${total_cost:,.2f}")
    print(f"  Percentage Impact: {total_cost_pct*100:.2f}%")
    print(f"{'='*60}")

    if total_cost_pct > 0.20:  # > 20% of capital
        print(f"\n⚠️  WARNING: Transaction costs exceed 20% of capital!")
        print(f"   Strategy is likely unprofitable after costs")

    return {
        'round_trip_cost': round_trip.total_cost,
        'round_trip_cost_pct': round_trip.cost_pct,
        'total_cost': total_cost,
        'total_cost_pct': total_cost_pct,
        'num_round_trips': num_round_trips
    }


# Example usage
if __name__ == "__main__":
    print("Transaction Cost Model Examples\n")

    # Example 1: Stock trade
    print("Example 1: Stock Trade")
    stock_model = TransactionCostModel(asset_class=AssetClass.STOCK)
    cost = stock_model.calculate_trade_cost(
        trade_value=10000,
        is_buy=True,
        volatility=0.015  # 1.5% daily volatility
    )
    print(f"Trade Value: $10,000")
    print(f"Commission: ${cost.commission:.2f}")
    print(f"Spread Cost: ${cost.spread_cost:.2f}")
    print(f"Slippage: ${cost.slippage:.2f}")
    print(f"Exchange Fees: ${cost.exchange_fees:.2f}")
    print(f"Total Cost: ${cost.total_cost:.2f} ({cost.cost_pct*100:.3f}%)")

    # Example 2: Crypto trade
    print("\n" + "="*60)
    print("Example 2: Cryptocurrency Trade")
    crypto_model = TransactionCostModel(asset_class=AssetClass.CRYPTO)
    cost = crypto_model.calculate_trade_cost(
        trade_value=10000,
        is_buy=True,
        volatility=0.05  # 5% daily volatility (typical for crypto)
    )
    print(f"Trade Value: $10,000")
    print(f"Commission: ${cost.commission:.2f}")
    print(f"Spread Cost: ${cost.spread_cost:.2f}")
    print(f"Slippage: ${cost.slippage:.2f}")
    print(f"Exchange Fees: ${cost.exchange_fees:.2f}")
    print(f"Total Cost: ${cost.total_cost:.2f} ({cost.cost_pct*100:.3f}%)")

    # Example 3: Strategy impact analysis
    print("\n" + "="*60)
    print("Example 3: Full Strategy Cost Analysis")
    estimate_total_costs(
        total_trades=475,  # NeuroVest's trade count
        avg_trade_value=20000,
        asset_class=AssetClass.STOCK,
        avg_holding_days=5,
        use_leverage=True
    )

    print("\n" + "="*60)
    print("Example 4: Crypto Strategy Cost Analysis")
    estimate_total_costs(
        total_trades=200,
        avg_trade_value=15000,
        asset_class=AssetClass.CRYPTO,
        avg_holding_days=3,
        use_leverage=True
    )
