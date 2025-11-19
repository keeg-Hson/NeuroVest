"""
Crypto data loader using CCXT

Downloads historical cryptocurrency data for backtesting
"""

import ccxt
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta


class CryptoDataLoader:
    """
    Load crypto data from exchanges using CCXT
    """

    def __init__(self, exchange_name='binance', cache_dir='../data_cache'):
        """
        Initialize crypto data loader

        Args:
            exchange_name: Exchange to use (binance, coinbase, kraken)
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,  # Required to avoid rate limits
        })

    def download_ohlcv(self, symbol, timeframe='1d', since=None, limit=1000):
        """
        Download OHLCV data from exchange

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1d', '1h', etc.)
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        all_candles = []

        # If since not provided, fetch last 1000 candles
        if since is None:
            # Calculate start time for last `limit` candles
            ms_per_candle = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
            }[timeframe]

            since = int((datetime.now().timestamp() - (limit * ms_per_candle / 1000)) * 1000)

        print(f"Downloading {symbol} {timeframe} data...")

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

                if len(candles) == 0:
                    break

                all_candles.extend(candles)

                # Update since to last candle timestamp + 1ms
                since = candles[-1][0] + 1

                # Check if end of data reached
                if len(candles) < limit:
                    break

                print(f"  Downloaded {len(all_candles)} candles...")

                # Rate limit protection
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"  Error downloading {symbol}: {e}")
                break

        # Convert to DataFrame
        if len(all_candles) > 0:
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            )

            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)

            # Add Adj_Close (same as Close for crypto)
            df['Adj_Close'] = df['Close']

            print(f"  ✓ Downloaded {len(df)} candles from {df.index.min()} to {df.index.max()}")

            return df
        else:
            return None

    def load_crypto(self, symbol, days_back=365 * 3, use_cache=True):
        """
        Load crypto data with caching

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            days_back: Number of days of history to load
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_1d.csv"

        if use_cache and cache_file.exists():
            print(f"Loading {symbol} from cache...")
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            print(f"  ✓ Loaded {len(df)} days from cache")
            return df

        # Download fresh data
        since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        df = self.download_ohlcv(symbol, timeframe='1d', since=since, limit=1000)

        # Save to cache
        if df is not None:
            df.to_csv(cache_file)
            print(f"  ✓ Saved to cache: {cache_file}")

        return df

    def load_multi_crypto(self, symbols, days_back=365 * 3, use_cache=True):
        """
        Load multiple crypto assets

        Args:
            symbols: List of trading pairs
            days_back: Number of days of history
            use_cache: Whether to use cached data

        Returns:
            Dict mapping symbol to DataFrame
        """
        assets = {}

        for symbol in symbols:
            try:
                df = self.load_crypto(symbol, days_back=days_back, use_cache=use_cache)
                if df is not None:
                    assets[symbol] = df
            except Exception as e:
                print(f"✗ Error loading {symbol}: {e}")

        return assets


def get_default_crypto_config():
    """
    Get default crypto asset configuration

    Returns:
        List of trading pairs
    """
    return [
        'BTC/USDT',   # Bitcoin - most liquid
        'ETH/USDT',   # Ethereum - #2
        'SOL/USDT',   # Solana - high performance L1
        'AVAX/USDT',  # Avalanche - alt L1
        'MATIC/USDT', # Polygon - L2 scaling
    ]


if __name__ == '__main__':
    # Test crypto data loader
    print("=" * 70)
    print("CRYPTO DATA LOADER TEST")
    print("=" * 70)

    loader = CryptoDataLoader()

    # Test single asset
    print("\n1. Loading single asset (BTC/USDT)...")
    btc = loader.load_crypto('BTC/USDT', days_back=365 * 2, use_cache=False)

    if btc is not None:
        print(f"\n   BTC/USDT data:")
        print(f"   - Days: {len(btc)}")
        print(f"   - Date range: {btc.index.min()} to {btc.index.max()}")
        print(f"   - Current price: ${btc['Close'].iloc[-1]:,.2f}")

    # Test multi-asset
    print("\n2. Loading multiple assets...")
    crypto_assets = get_default_crypto_config()
    assets = loader.load_multi_crypto(crypto_assets, days_back=365 * 2, use_cache=True)

    print(f"\n✓ Loaded {len(assets)} crypto assets:")
    for symbol, df in assets.items():
        print(f"   - {symbol}: {len(df)} days, ${df['Close'].iloc[-1]:,.2f}")

    # Calculate correlations
    if len(assets) > 1:
        print("\n3. Correlation matrix:")
        close_prices = pd.DataFrame({
            symbol: df['Close'] for symbol, df in assets.items()
        })
        corr_matrix = close_prices.pct_change().corr()
        print(corr_matrix.round(2))

    print("\n" + "=" * 70)
    print("✓ Crypto data loader test complete")
