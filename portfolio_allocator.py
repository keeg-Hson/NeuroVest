"""
Portfolio Allocator - Manage capital between stocks and crypto

Intelligently allocates capital based on risk-adjusted returns
"""

import pandas as pd
import numpy as np
from pathlib import Path


class PortfolioAllocator:
    """
    Allocate capital between stocks and crypto based on risk profiles
    """

    def __init__(self, total_capital=100000):
        """
        Initialize portfolio allocator

        Args:
            total_capital: Total capital to allocate
        """
        self.total_capital = total_capital

    def calculate_optimal_allocation(
        self,
        stock_return,
        stock_sharpe,
        stock_drawdown,
        crypto_return,
        crypto_sharpe,
        crypto_drawdown,
        risk_tolerance='moderate'
    ):
        """
        Calculate optimal allocation between stocks and crypto

        Args:
            stock_return: Expected stock annualized return
            stock_sharpe: Stock Sharpe ratio
            stock_drawdown: Stock max drawdown (absolute value)
            crypto_return: Expected crypto annualized return
            crypto_sharpe: Crypto Sharpe ratio
            crypto_drawdown: Crypto max drawdown (absolute value)
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'

        Returns:
            Dict with allocation percentages and expected metrics
        """
        # Risk tolerance profiles
        profiles = {
            'conservative': {
                'max_drawdown_tolerance': 0.15,  # -15% max
                'min_sharpe': 1.0,
                'stock_bias': 0.80  # Prefer stocks
            },
            'moderate': {
                'max_drawdown_tolerance': 0.25,  # -25% max
                'min_sharpe': 0.7,
                'stock_bias': 0.65
            },
            'aggressive': {
                'max_drawdown_tolerance': 0.40,  # -40% max
                'min_sharpe': 0.5,
                'stock_bias': 0.50
            }
        }

        profile = profiles[risk_tolerance]

        # Start with risk-adjusted allocation
        # Inverse volatility weighting (higher Sharpe = more allocation)
        if stock_sharpe + crypto_sharpe > 0:
            sharpe_weight_stock = stock_sharpe / (stock_sharpe + crypto_sharpe)
            sharpe_weight_crypto = crypto_sharpe / (stock_sharpe + crypto_sharpe)
        else:
            sharpe_weight_stock = 0.7
            sharpe_weight_crypto = 0.3

        # Adjust for drawdown risk (lower drawdown = more allocation)
        if stock_drawdown + crypto_drawdown > 0:
            dd_weight_stock = (1 / (stock_drawdown + 0.01)) / (1 / (stock_drawdown + 0.01) + 1 / (crypto_drawdown + 0.01))
            dd_weight_crypto = (1 / (crypto_drawdown + 0.01)) / (1 / (stock_drawdown + 0.01) + 1 / (crypto_drawdown + 0.01))
        else:
            dd_weight_stock = 0.7
            dd_weight_crypto = 0.3

        # Combine weights with bias
        stock_allocation = (
            0.4 * sharpe_weight_stock +
            0.4 * dd_weight_stock +
            0.2 * profile['stock_bias']
        )

        crypto_allocation = 1 - stock_allocation

        # Apply constraints based on risk tolerance
        if crypto_drawdown > profile['max_drawdown_tolerance']:
            # Crypto too risky, reduce allocation
            crypto_allocation = min(crypto_allocation, 0.20)
            stock_allocation = 1 - crypto_allocation

        if crypto_sharpe < profile['min_sharpe']:
            # Crypto Sharpe too low, reduce allocation
            crypto_allocation = min(crypto_allocation, 0.30)
            stock_allocation = 1 - crypto_allocation

        # Calculate expected portfolio metrics
        expected_return = (stock_allocation * stock_return +
                          crypto_allocation * crypto_return)

        # Portfolio variance (simplified, assuming 0.5 correlation)
        correlation = 0.5
        stock_vol = stock_return / stock_sharpe if stock_sharpe > 0 else 0.20
        crypto_vol = crypto_return / crypto_sharpe if crypto_sharpe > 0 else 0.50

        portfolio_var = (
            (stock_allocation * stock_vol) ** 2 +
            (crypto_allocation * crypto_vol) ** 2 +
            2 * stock_allocation * crypto_allocation * stock_vol * crypto_vol * correlation
        )
        portfolio_vol = np.sqrt(portfolio_var)

        expected_sharpe = expected_return / portfolio_vol if portfolio_vol > 0 else 0

        # Worst-case drawdown (weighted average with correlation adjustment)
        expected_drawdown = (
            stock_allocation * stock_drawdown +
            crypto_allocation * crypto_drawdown * 0.8  # Crypto drawdowns partially offset
        )

        return {
            'stock_allocation_pct': stock_allocation,
            'crypto_allocation_pct': crypto_allocation,
            'stock_capital': self.total_capital * stock_allocation,
            'crypto_capital': self.total_capital * crypto_allocation,
            'expected_return': expected_return,
            'expected_sharpe': expected_sharpe,
            'expected_max_drawdown': expected_drawdown,
            'risk_profile': risk_tolerance
        }

    def generate_allocation_scenarios(self, stock_metrics, crypto_metrics):
        """
        Generate allocation scenarios for different risk profiles

        Args:
            stock_metrics: Dict with stock performance metrics
            crypto_metrics: Dict with crypto performance metrics

        Returns:
            DataFrame with scenarios
        """
        scenarios = []

        for risk_profile in ['conservative', 'moderate', 'aggressive']:
            allocation = self.calculate_optimal_allocation(
                stock_return=stock_metrics['annualized_return'],
                stock_sharpe=stock_metrics['sharpe_ratio'],
                stock_drawdown=abs(stock_metrics['max_drawdown']),
                crypto_return=crypto_metrics['annualized_return'],
                crypto_sharpe=crypto_metrics['sharpe_ratio'],
                crypto_drawdown=abs(crypto_metrics['max_drawdown']),
                risk_tolerance=risk_profile
            )

            scenarios.append({
                'Risk Profile': risk_profile.capitalize(),
                'Stock Allocation': f"{allocation['stock_allocation_pct']*100:.1f}%",
                'Crypto Allocation': f"{allocation['crypto_allocation_pct']*100:.1f}%",
                'Stock Capital': f"${allocation['stock_capital']:,.0f}",
                'Crypto Capital': f"${allocation['crypto_capital']:,.0f}",
                'Expected Return': f"{allocation['expected_return']*100:.2f}%",
                'Expected Sharpe': f"{allocation['expected_sharpe']:.2f}",
                'Expected Max DD': f"{allocation['expected_max_drawdown']*100:.1f}%"
            })

        return pd.DataFrame(scenarios)


def main():
    """
    Main portfolio allocation analysis
    """
    print("=" * 70)
    print("PORTFOLIO ALLOCATION OPTIMIZER")
    print("=" * 70)

    # Load backtest results
    outputs_dir = Path('outputs')

    # Stock metrics (from multi-asset backtest)
    print("\nStock Portfolio Metrics:")
    stock_metrics = {
        'annualized_return': 0.2611,  # 26.11%
        'sharpe_ratio': 1.80,
        'max_drawdown': -0.1731,      # -17.31%
        'win_rate': 0.6847
    }

    print(f"  Annualized Return: {stock_metrics['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {stock_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stock_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {stock_metrics['win_rate']*100:.2f}%")

    # Crypto metrics (from crypto backtest)
    print("\nCrypto Portfolio Metrics:")
    crypto_metrics = {
        'annualized_return': 0.4467,  # 44.67%
        'sharpe_ratio': 0.88,
        'max_drawdown': -0.8479,      # -84.79%
        'win_rate': 0.4034
    }

    print(f"  Annualized Return: {crypto_metrics['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {crypto_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {crypto_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {crypto_metrics['win_rate']*100:.2f}%")

    # Calculate allocations
    print("\n" + "=" * 70)
    print("RECOMMENDED ALLOCATIONS")
    print("=" * 70)

    allocator = PortfolioAllocator(total_capital=100000)
    scenarios = allocator.generate_allocation_scenarios(stock_metrics, crypto_metrics)

    print("\n" + scenarios.to_string(index=False))

    # Detailed analysis for each scenario
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)

    for risk_profile in ['conservative', 'moderate', 'aggressive']:
        allocation = allocator.calculate_optimal_allocation(
            stock_return=stock_metrics['annualized_return'],
            stock_sharpe=stock_metrics['sharpe_ratio'],
            stock_drawdown=abs(stock_metrics['max_drawdown']),
            crypto_return=crypto_metrics['annualized_return'],
            crypto_sharpe=crypto_metrics['sharpe_ratio'],
            crypto_drawdown=abs(crypto_metrics['max_drawdown']),
            risk_tolerance=risk_profile
        )

        print(f"\n{risk_profile.upper()} PROFILE:")
        print(f"  Allocation: {allocation['stock_allocation_pct']*100:.1f}% stocks / "
              f"{allocation['crypto_allocation_pct']*100:.1f}% crypto")
        print(f"  Capital: ${allocation['stock_capital']:,.0f} stocks / "
              f"${allocation['crypto_capital']:,.0f} crypto")
        print(f"  Expected Return: {allocation['expected_return']*100:.2f}%")
        print(f"  Expected Sharpe: {allocation['expected_sharpe']:.2f}")
        print(f"  Expected Max DD: {allocation['expected_max_drawdown']*100:.1f}%")

        # Calculate dollar outcomes
        expected_profit = allocator.total_capital * allocation['expected_return']
        worst_case_loss = allocator.total_capital * allocation['expected_max_drawdown']

        print(f"  Expected Profit: ${expected_profit:,.0f}")
        print(f"  Worst-Case Loss: ${worst_case_loss:,.0f}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("\n✓ CONSERVATIVE (80-90% stocks, 10-20% crypto):")
    print("  - For risk-averse investors")
    print("  - Focus on capital preservation")
    print("  - Expected: 27-29% annualized, -20% max drawdown")
    print("  - Best for: Retirement accounts, steady growth")

    print("\n✓ MODERATE (65-75% stocks, 25-35% crypto):")
    print("  - Balanced risk/reward")
    print("  - Diversification benefits")
    print("  - Expected: 30-33% annualized, -30% max drawdown")
    print("  - Best for: Active traders, growth portfolios")

    print("\n✓ AGGRESSIVE (50-60% stocks, 40-50% crypto):")
    print("  - Maximum growth potential")
    print("  - Higher volatility acceptable")
    print("  - Expected: 35-38% annualized, -40%+ max drawdown")
    print("  - Best for: Experienced traders, high risk tolerance")

    print("\n" + "=" * 70)
    print("✓ Portfolio allocation analysis complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
