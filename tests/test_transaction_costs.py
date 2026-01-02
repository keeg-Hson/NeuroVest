"""
Unit Tests for Transaction Cost Model

These tests verify that transaction costs are calculated correctly.

Run with: pytest tests/test_transaction_costs.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.transaction_costs import (
    TransactionCostModel,
    ConservativeCostModel,
    AssetClass,
    estimate_total_costs
)


class TestTransactionCostModel:
    """Tests for TransactionCostModel"""

    def test_stock_trade_cost(self):
        """Test stock trading cost calculation"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.01
        )

        # Should have some cost
        assert cost.total_cost > 0
        assert cost.cost_pct > 0

        # Cost should be reasonable (0.05% - 0.5% for stocks)
        assert cost.cost_pct < 0.005  # Less than 0.5%
        assert cost.cost_pct > 0.0001  # More than 0.01%

        # Buy orders should not have SEC fees
        assert cost.exchange_fees == 0.0

    def test_stock_sell_fees(self):
        """Test that stock sells include SEC fees"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=False,  # Sell
            volatility=0.01
        )

        # Sell should have exchange fees (SEC + FINRA)
        assert cost.exchange_fees > 0

    def test_crypto_trade_cost(self):
        """Test crypto trading cost calculation"""
        model = TransactionCostModel(asset_class=AssetClass.CRYPTO)

        cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.05  # Higher volatility for crypto
        )

        # Crypto should have higher costs than stocks
        assert cost.total_cost > 0
        assert cost.cost_pct > 0.001  # At least 0.1%

        # Should include exchange fees
        assert cost.exchange_fees > 0

    def test_volatility_increases_costs(self):
        """Test that higher volatility increases costs"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        # Low volatility trade
        low_vol_cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.01  # 1% volatility
        )

        # High volatility trade
        high_vol_cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.05  # 5% volatility
        )

        # High volatility should cost more (spread and slippage)
        assert high_vol_cost.total_cost > low_vol_cost.total_cost
        assert high_vol_cost.spread_cost > low_vol_cost.spread_cost
        assert high_vol_cost.slippage > low_vol_cost.slippage

    def test_liquidity_affects_costs(self):
        """Test that liquidity tier affects costs"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        # High liquidity
        high_liq_cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.01,
            liquidity_tier="high"
        )

        # Low liquidity
        low_liq_cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.01,
            liquidity_tier="low"
        )

        # Low liquidity should cost significantly more
        assert low_liq_cost.total_cost > high_liq_cost.total_cost
        assert low_liq_cost.total_cost > high_liq_cost.total_cost * 2  # At least 2x

    def test_market_order_has_slippage(self):
        """Test that market orders include slippage"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        market_cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.01,
            is_market_order=True
        )

        # Market order should have slippage
        assert market_cost.slippage > 0

    def test_limit_order_no_slippage(self):
        """Test that limit orders don't include slippage"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        limit_cost = model.calculate_trade_cost(
            trade_value=10000,
            is_buy=True,
            volatility=0.01,
            is_market_order=False  # Limit order
        )

        # Limit order should have no slippage
        assert limit_cost.slippage == 0.0

    def test_round_trip_cost(self):
        """Test round trip (entry + exit) cost calculation"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        round_trip = model.calculate_round_trip_cost(
            position_value=10000,
            holding_days=5,
            volatility=0.01,
            use_leverage=False
        )

        # Round trip should cost more than single trade
        single_trade = model.calculate_trade_cost(10000, True, 0.01)
        assert round_trip.total_cost > single_trade.total_cost

        # Should include entry and exit
        assert round_trip.entry_cost > 0
        assert round_trip.exit_cost > 0

        # No leverage = no financing cost
        assert round_trip.financing_cost == 0.0

    def test_leverage_financing_cost(self):
        """Test that leverage adds financing costs"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        # With leverage
        leveraged = model.calculate_round_trip_cost(
            position_value=10000,
            holding_days=30,  # Hold for 30 days
            volatility=0.01,
            use_leverage=True,
            leverage_ratio=2.0
        )

        # Without leverage
        unleveraged = model.calculate_round_trip_cost(
            position_value=10000,
            holding_days=30,
            volatility=0.01,
            use_leverage=False
        )

        # Leveraged should have financing cost
        assert leveraged.financing_cost > 0

        # Leveraged should cost more total
        assert leveraged.total_cost > unleveraged.total_cost

    def test_crypto_financing_more_expensive(self):
        """Test that crypto leverage is more expensive than stocks"""
        stock_model = TransactionCostModel(asset_class=AssetClass.STOCK)
        crypto_model = TransactionCostModel(asset_class=AssetClass.CRYPTO)

        stock_cost = stock_model.calculate_round_trip_cost(
            position_value=10000,
            holding_days=30,
            use_leverage=True,
            leverage_ratio=2.0
        )

        crypto_cost = crypto_model.calculate_round_trip_cost(
            position_value=10000,
            holding_days=30,
            use_leverage=True,
            leverage_ratio=2.0
        )

        # Crypto financing should be more expensive
        assert crypto_cost.financing_cost > stock_cost.financing_cost


class TestConservativeCostModel:
    """Tests for ConservativeCostModel"""

    def test_conservative_costs_higher(self):
        """Test that conservative model has higher costs"""
        normal_model = TransactionCostModel(asset_class=AssetClass.STOCK)
        conservative_model = ConservativeCostModel(asset_class=AssetClass.STOCK)

        normal_cost = normal_model.calculate_trade_cost(10000, True, 0.01)
        conservative_cost = conservative_model.calculate_trade_cost(10000, True, 0.01)

        # Conservative should be higher
        assert conservative_cost.total_cost > normal_cost.total_cost

        # Should be roughly 1.5x higher (50% pessimism buffer)
        ratio = conservative_cost.total_cost / normal_cost.total_cost
        assert ratio > 1.3  # At least 30% higher
        assert ratio < 2.0  # Not more than 2x


class TestCostEstimation:
    """Tests for strategy cost estimation"""

    def test_cost_estimation_stocks(self):
        """Test cost estimation for stock strategy"""
        result = estimate_total_costs(
            total_trades=100,
            avg_trade_value=10000,
            asset_class=AssetClass.STOCK,
            avg_holding_days=5,
            use_leverage=False
        )

        # Should return valid results
        assert result['num_round_trips'] == 50  # 100 trades = 50 round trips
        assert result['total_cost'] > 0
        assert result['total_cost_pct'] > 0
        assert result['round_trip_cost'] > 0

    def test_cost_estimation_crypto(self):
        """Test cost estimation for crypto strategy"""
        result = estimate_total_costs(
            total_trades=100,
            avg_trade_value=10000,
            asset_class=AssetClass.CRYPTO,
            avg_holding_days=5,
            use_leverage=False
        )

        # Crypto should have higher costs than stocks
        stock_result = estimate_total_costs(
            total_trades=100,
            avg_trade_value=10000,
            asset_class=AssetClass.STOCK,
            avg_holding_days=5,
            use_leverage=False
        )

        assert result['total_cost'] > stock_result['total_cost']

    def test_neurovest_actual_costs(self):
        """Test costs for NeuroVest's actual trading (475 trades)"""
        result = estimate_total_costs(
            total_trades=475,
            avg_trade_value=20000,  # Approximate average
            asset_class=AssetClass.STOCK,
            avg_holding_days=5,
            use_leverage=True
        )

        # With 475 trades, costs should be significant
        # Print for visibility
        print(f"\nNeuroVest Actual Costs:")
        print(f"  Total Cost: ${result['total_cost']:,.2f}")
        print(f"  Cost %: {result['total_cost_pct']*100:.2f}%")

        # Cost percentage should be meaningful (likely 5-20% of capital)
        assert result['total_cost_pct'] > 0.05  # At least 5%


class TestCostComponents:
    """Test individual cost components"""

    def test_spread_cost_calculation(self):
        """Test spread cost is calculated"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        cost = model.calculate_trade_cost(10000, True, 0.01)

        # Should have spread cost
        assert cost.spread_cost > 0

        # Spread should be a significant portion of total cost
        assert cost.spread_cost > cost.total_cost * 0.2  # At least 20% of total

    def test_slippage_calculation(self):
        """Test slippage is calculated"""
        model = TransactionCostModel(asset_class=AssetClass.STOCK)

        cost = model.calculate_trade_cost(
            10000, True, 0.01, is_market_order=True
        )

        # Should have slippage for market orders
        assert cost.slippage > 0

    def test_commission_calculation(self):
        """Test commission calculation with custom settings"""
        model = TransactionCostModel(
            asset_class=AssetClass.STOCK,
            commission_pct=0.001,  # 0.1% commission
            min_commission=5.00  # $5 minimum
        )

        # Large trade (commission % applies)
        large_cost = model.calculate_trade_cost(10000, True, 0.01)
        assert large_cost.commission == 10.00  # 0.1% of $10,000

        # Small trade (minimum applies)
        small_cost = model.calculate_trade_cost(1000, True, 0.01)
        assert small_cost.commission == 5.00  # Minimum


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
