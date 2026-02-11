"""
Example: Using DataManager in Existing Code

Shows how to integrate the new data management system with existing
trading strategies and backtests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_manager_postgres import DataManager
import pandas as pd


def example_1_basic_usage():
    """Basic data retrieval"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)

    with DataManager('data/market_data.db') as dm:
        # Get SPY data for last year
        spy_data = dm.get_data('SPY', start_date='2024-01-01')

        print(f"\nRetrieved {len(spy_data)} days of SPY data")
        print(f"Latest close: ${spy_data['Close'].iloc[-1]:.2f}")
        print(f"\nFirst 3 rows:")
        print(spy_data.head(3))


def example_2_multi_asset():
    """Multi-asset portfolio"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Asset Portfolio")
    print("="*70)

    tickers = ['SPY', 'QQQ', 'TLT', 'GLD']

    with DataManager('data/market_data.db') as dm:
        # Get data for all assets
        assets = dm.get_multi_asset(tickers, start_date='2024-01-01')

        print(f"\nLoaded {len(assets)} assets:")
        for ticker, df in assets.items():
            latest = df['Close'].iloc[-1]
            change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            print(f"  {ticker}: ${latest:.2f} ({change:+.2f}% YTD)")


def example_3_replace_csv_loading():
    """Replace CSV loading in existing code"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Replace CSV Loading")
    print("="*70)

    # OLD WAY (CSV):
    # df = pd.read_csv('SPY.csv', index_col=0, parse_dates=True)

    # NEW WAY (Database):
    with DataManager('data/market_data.db') as dm:
        df = dm.get_data('SPY')

    print(f"\nLoaded {len(df)} records from database")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")


def example_4_integrate_with_backtest():
    """Integration with backtesting"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Backtest Integration")
    print("="*70)

    def run_simple_backtest(ticker, start_date='2024-01-01'):
        """Simple moving average crossover backtest"""
        with DataManager('data/market_data.db') as dm:
            # Get data
            df = dm.get_data(ticker, start_date=start_date)

            if df.empty:
                print(f"No data for {ticker}")
                return

            # Calculate moving averages
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()

            # Generate signals
            df['Signal'] = 0
            df.loc[df['MA_20'] > df['MA_50'], 'Signal'] = 1  # Bullish
            df.loc[df['MA_20'] < df['MA_50'], 'Signal'] = -1  # Bearish

            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']

            # Performance
            total_return = (1 + df['Strategy_Returns']).prod() - 1
            buy_hold = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

            print(f"\n{ticker} Backtest Results:")
            print(f"  Strategy Return: {total_return*100:+.2f}%")
            print(f"  Buy & Hold: {buy_hold*100:+.2f}%")
            print(f"  Outperformance: {(total_return - buy_hold)*100:+.2f}%")

    # Run backtest
    run_simple_backtest('SPY')
    run_simple_backtest('QQQ')


def example_5_crypto_data():
    """Working with crypto data"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Crypto Data")
    print("="*70)

    crypto_tickers = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

    with DataManager('data/market_data.db') as dm:
        for ticker in crypto_tickers:
            df = dm.get_data(ticker, start_date='2024-01-01')

            if not df.empty:
                latest = df['Close'].iloc[-1]
                vol = df['Close'].pct_change().std() * 100
                print(f"\n{ticker}:")
                print(f"  Latest: ${latest:,.2f}")
                print(f"  Volatility: {vol:.2f}%")
                print(f"  Records: {len(df)}")


def example_6_caching_performance():
    """Demonstrate caching performance"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Cache Performance")
    print("="*70)

    import time

    with DataManager('data/market_data.db') as dm:
        # First query (cache miss)
        start = time.time()
        df1 = dm.get_data('SPY')
        time1 = time.time() - start

        # Second query (cache hit)
        start = time.time()
        df2 = dm.get_data('SPY')
        time2 = time.time() - start

        # Third query (cache hit)
        start = time.time()
        df3 = dm.get_data('SPY')
        time3 = time.time() - start

        print(f"\nQuery Performance:")
        print(f"  First query (no cache): {time1*1000:.2f}ms")
        print(f"  Second query (cached): {time2*1000:.2f}ms ({time1/time2:.1f}x faster)")
        print(f"  Third query (cached): {time3*1000:.2f}ms ({time1/time3:.1f}x faster)")

        # Show cache stats
        stats = dm.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Hit rate: {stats['cache_hit_rate']}%")
        print(f"  Hits: {stats['cache_hits']}")
        print(f"  Misses: {stats['cache_misses']}")


if __name__ == '__main__':
    # Run all examples
    example_1_basic_usage()
    example_2_multi_asset()
    example_3_replace_csv_loading()
    example_4_integrate_with_backtest()
    example_5_crypto_data()
    example_6_caching_performance()

    print("\n" + "="*70)
    print("✓ All examples complete")
    print("="*70)
