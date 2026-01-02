"""
Example API Client for NeuroVest Trading API
Demonstrates how to interact with the API programmatically
"""

import requests
from typing import Optional, List, Dict
import time


class NeuroVestClient:
    """
    Python client for NeuroVest Trading API

    Features:
    - Portfolio management
    - Trading signal retrieval
    - Trade execution
    - Risk profile configuration
    - Market data access
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 token: str = "demo_token_replace_in_production"):
        """
        Initialize API client

        Args:
            base_url: API base URL
            token: Authentication token
        """
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    # Portfolio methods
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        response = requests.get(
            f"{self.base_url}/portfolio/status",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        response = requests.get(
            f"{self.base_url}/portfolio/positions",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_performance(self) -> Dict:
        """Get portfolio performance metrics"""
        response = requests.get(
            f"{self.base_url}/portfolio/performance",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    # Trading methods
    def get_signals(self, asset_type: Optional[str] = None,
                   min_confidence: float = 0.60) -> List[Dict]:
        """
        Get trading signals

        Args:
            asset_type: Filter by 'stock' or 'crypto'
            min_confidence: Minimum confidence threshold

        Returns:
            List of signals
        """
        params = {"min_confidence": min_confidence}
        if asset_type:
            params["asset_type"] = asset_type

        response = requests.get(
            f"{self.base_url}/signals",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def execute_trade(self, ticker: str, action: str,
                     quantity: float, mode: str = "paper") -> Dict:
        """
        Execute a trade

        Args:
            ticker: Asset ticker symbol
            action: 'buy' or 'sell'
            quantity: Number of shares/units
            mode: 'paper' or 'live'

        Returns:
            Trade execution details
        """
        params = {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "mode": mode
        }

        response = requests.post(
            f"{self.base_url}/trade",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    # Risk management methods
    def list_risk_profiles(self) -> Dict:
        """List all available risk profiles"""
        response = requests.get(
            f"{self.base_url}/risk/profiles",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_risk_profile(self, profile_name: str) -> Dict:
        """Get specific risk profile details"""
        response = requests.get(
            f"{self.base_url}/risk/profile/{profile_name}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def set_risk_profile(self, profile_name: str) -> Dict:
        """Set active risk profile"""
        response = requests.post(
            f"{self.base_url}/risk/profile/active",
            headers=self.headers,
            params={"profile_name": profile_name}
        )
        response.raise_for_status()
        return response.json()

    def create_custom_risk_profile(self, name: str, **kwargs) -> Dict:
        """
        Create custom risk profile

        Args:
            name: Profile name
            **kwargs: Risk parameters (stock_allocation, crypto_allocation, etc.)

        Returns:
            Created profile details
        """
        data = {"name": name, **kwargs}

        response = requests.post(
            f"{self.base_url}/risk/profile/create",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()

    # Data methods
    def get_market_data(self, ticker: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict:
        """
        Get market data for a ticker

        Args:
            ticker: Asset ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Market data
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        response = requests.get(
            f"{self.base_url}/data/{ticker}",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_data_stats(self) -> Dict:
        """Get database statistics"""
        response = requests.get(
            f"{self.base_url}/data/stats",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def trigger_data_update(self, ticker: Optional[str] = None) -> Dict:
        """Trigger data update"""
        params = {}
        if ticker:
            params["ticker"] = ticker

        response = requests.post(
            f"{self.base_url}/data/update",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()


# ==============================================================================
# Example Usage
# ==============================================================================

def example_1_portfolio_status():
    """Example: Get portfolio status"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Portfolio Status")
    print("="*70)

    client = NeuroVestClient()

    # Get portfolio status
    portfolio = client.get_portfolio_status()

    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${portfolio['total_value']:,.2f}")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    print(f"  Stock Value: ${portfolio['stock_value']:,.2f}")
    print(f"  Crypto Value: ${portfolio['crypto_value']:,.2f}")
    print(f"  Positions: {portfolio['positions']}")
    print(f"  Daily P&L: ${portfolio['daily_pnl']:,.2f} ({portfolio['daily_pnl_pct']:.2f}%)")
    print(f"  Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:.2f}%)")


def example_2_get_signals():
    """Example: Get trading signals"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Trading Signals")
    print("="*70)

    client = NeuroVestClient()

    # Get high-confidence stock signals
    signals = client.get_signals(asset_type="stock", min_confidence=0.75)

    print(f"\nFound {len(signals)} high-confidence stock signals:")

    for signal in signals:
        print(f"\n  {signal['ticker']} ({signal['asset_type']})")
        print(f"    Signal: {signal['signal'].upper()}")
        print(f"    Confidence: {signal['confidence']*100:.0f}%")
        print(f"    Entry: ${signal['entry_price']:.2f}")
        if signal['target_price']:
            print(f"    Target: ${signal['target_price']:.2f}")
        if signal['stop_loss']:
            print(f"    Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"    Position Size: ${signal['position_size']:,.2f}")
        print(f"    Reason: {signal['reason']}")


def example_3_execute_trades():
    """Example: Execute paper trades"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Execute Paper Trades")
    print("="*70)

    client = NeuroVestClient()

    # Get signals
    signals = client.get_signals(min_confidence=0.80)

    print(f"\nExecuting paper trades for {len(signals)} signals:")

    for signal in signals[:3]:  # Limit to 3 trades
        if signal['signal'] == 'buy':
            # Calculate quantity based on position size
            quantity = signal['position_size'] / signal['entry_price']

            # Execute paper trade
            trade = client.execute_trade(
                ticker=signal['ticker'],
                action='buy',
                quantity=quantity,
                mode='paper'
            )

            print(f"\n  {trade['ticker']}: {trade['action'].upper()}")
            print(f"    Quantity: {trade['quantity']:.2f}")
            print(f"    Price: ${trade['price']:.2f}")
            print(f"    Total: ${trade['total_value']:,.2f}")
            print(f"    Status: {trade['status']}")


def example_4_risk_profiles():
    """Example: Manage risk profiles"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Risk Profile Management")
    print("="*70)

    client = NeuroVestClient()

    # List available profiles
    profiles = client.list_risk_profiles()
    print(f"\nAvailable Profiles: {', '.join(profiles['profiles'])}")
    print(f"Active Profile: {profiles['active']}")

    # Get details of moderate profile
    moderate = client.get_risk_profile('moderate')
    print(f"\nModerate Profile:")
    print(f"  Stock Allocation: {moderate['stock_allocation']*100:.0f}%")
    print(f"  Crypto Allocation: {moderate['crypto_allocation']*100:.0f}%")
    print(f"  Max Leverage (Stocks): {moderate['max_leverage_stocks']}x")
    print(f"  Max Leverage (Crypto): {moderate['max_leverage_crypto']}x")
    print(f"  Stop Loss (Stocks): {moderate['stop_loss_pct_stocks']*100:.1f}%")
    print(f"  Stop Loss (Crypto): {moderate['stop_loss_pct_crypto']*100:.1f}%")

    # Switch to aggressive profile
    result = client.set_risk_profile('aggressive')
    print(f"\n{result['message']}")


def example_5_custom_risk_profile():
    """Example: Create custom risk profile"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Create Custom Risk Profile")
    print("="*70)

    client = NeuroVestClient()

    # Create custom profile
    result = client.create_custom_risk_profile(
        name='my_balanced',
        stock_allocation=0.65,
        crypto_allocation=0.35,
        max_leverage_stocks=1.5,
        max_leverage_crypto=2.5,
        stop_loss_pct_crypto=0.09
    )

    print(f"\n{result['message']}")
    print(f"\nCustom Profile Details:")
    profile = result['profile']
    print(f"  Stock Allocation: {profile['stock_allocation']*100:.0f}%")
    print(f"  Crypto Allocation: {profile['crypto_allocation']*100:.0f}%")
    print(f"  Max Leverage (Crypto): {profile['max_leverage_crypto']}x")


def example_6_market_data():
    """Example: Access market data"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Market Data Access")
    print("="*70)

    client = NeuroVestClient()

    # Get SPY data
    data = client.get_market_data('SPY', start_date='2024-01-01')

    print(f"\nSPY Market Data:")
    print(f"  Total Records: {data['records']}")
    print(f"  Date Range: {data['start_date'][:10]} to {data['end_date'][:10]}")
    print(f"\nLatest Data Point:")
    latest = data['data'][-1]
    print(f"  Date: {latest['timestamp'][:10]}")
    print(f"  Close: ${latest['Close']:.2f}")
    print(f"  Volume: {latest['Volume']:,}")


def example_7_performance_metrics():
    """Example: Get performance metrics"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Performance Metrics")
    print("="*70)

    client = NeuroVestClient()

    # Get performance
    perf = client.get_performance()

    print(f"\nPortfolio Performance:")
    print(f"  Total Return: {perf['total_return']*100:+.2f}%")
    print(f"  Annualized Return: {perf['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {perf['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {perf['win_rate']*100:.1f}%")
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {perf['total_trades']}")
    print(f"  Avg Profit per Trade: {perf['avg_profit_per_trade']*100:+.2f}%")
    print(f"  Best Trade: {perf['best_trade']*100:+.2f}%")
    print(f"  Worst Trade: {perf['worst_trade']*100:+.2f}%")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("NEUROVEST API CLIENT EXAMPLES")
    print("="*70)

    try:
        # Check API health
        client = NeuroVestClient()
        health = client.health_check()
        print(f"\nAPI Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")

        # Run examples
        example_1_portfolio_status()
        time.sleep(1)

        example_2_get_signals()
        time.sleep(1)

        example_3_execute_trades()
        time.sleep(1)

        example_4_risk_profiles()
        time.sleep(1)

        example_5_custom_risk_profile()
        time.sleep(1)

        example_6_market_data()
        time.sleep(1)

        example_7_performance_metrics()

        print("\n" + "="*70)
        print("✓ All examples completed successfully")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("  Make sure the API server is running:")
        print("  python api/trading_api.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == '__main__':
    run_all_examples()
