#!/usr/bin/env python3
"""
Enhanced crypto data download with maximum history and multiple sources

Sources:
- Binance (2017+): Most accurate OHLCV
- CoinGecko (2013+): Longer history for BTC/ETH

Downloads all available history and updates to current date.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

DATA_CACHE = Path('data_cache')
DATA_CACHE.mkdir(exist_ok=True)

# Crypto assets with their CoinGecko IDs
CRYPTO_ASSETS = {
    'BTC/USDT': 'bitcoin',
    'ETH/USDT': 'ethereum',
    'SOL/USDT': 'solana',
    'BNB/USDT': 'binancecoin',
    'XRP/USDT': 'ripple',
    'ADA/USDT': 'cardano',
    'DOGE/USDT': 'dogecoin',
    'AVAX/USDT': 'avalanche-2',
    'MATIC/USDT': 'matic-network',
    'LINK/USDT': 'chainlink',
}

def download_from_coingecko(coin_id, symbol):
    """Download max history from CoinGecko (free, no API key, 2013+ for BTC)"""
    import requests

    print(f"   Trying CoinGecko for {coin_id}...")

    # Get max history (up to ~10 years)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': 'max',
        'interval': 'daily'
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse prices
        prices = data.get('prices', [])
        if not prices:
            return None

        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df['Date'] = pd.to_datetime(df['Date'])

        # CoinGecko only gives Close, so we'll use it for OHLC
        # This is approximate but gives us the long history
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']

        # Get volume if available
        volumes = data.get('total_volumes', [])
        if volumes:
            vol_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
            vol_df['Date'] = pd.to_datetime(vol_df['timestamp'], unit='ms').dt.date
            vol_df['Date'] = pd.to_datetime(vol_df['Date'])
            df = df.merge(vol_df[['Date', 'Volume']], on='Date', how='left')
        else:
            df['Volume'] = 0

        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates('Date')
        return df

    except Exception as e:
        print(f"   CoinGecko error: {e}")
        return None

def download_from_binance(symbol):
    """Download from Binance via ccxt (2017+, accurate OHLCV)"""
    try:
        import ccxt

        print(f"   Trying Binance for {symbol}...")

        exchange = ccxt.binance({'enableRateLimit': True})

        # Start from 2017 (Binance launch)
        since = exchange.parse8601('2017-01-01T00:00:00Z')

        all_ohlcv = []
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000  # Next day
            if since > exchange.milliseconds():
                break
            time.sleep(0.1)  # Rate limit

        if not all_ohlcv:
            return None

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates('Date')
        return df

    except Exception as e:
        print(f"   Binance error: {e}")
        return None

def merge_data(coingecko_df, binance_df):
    """Merge CoinGecko (long history) with Binance (accurate recent)"""
    if coingecko_df is None:
        return binance_df
    if binance_df is None:
        return coingecko_df

    # Use Binance for recent data (more accurate OHLCV)
    # Use CoinGecko for older data
    binance_start = binance_df['Date'].min()

    old_data = coingecko_df[coingecko_df['Date'] < binance_start]

    if len(old_data) > 0:
        combined = pd.concat([old_data, binance_df], ignore_index=True)
        combined = combined.drop_duplicates('Date').sort_values('Date')
        return combined
    else:
        return binance_df

def main():
    print("=" * 80)
    print("ENHANCED CRYPTO DATA DOWNLOAD")
    print("=" * 80)
    print(f"\nSources: CoinGecko (2013+) + Binance (2017+)")
    print(f"Assets: {len(CRYPTO_ASSETS)}")
    print("=" * 80)

    successful = []
    failed = []

    for symbol, coin_id in CRYPTO_ASSETS.items():
        filename = symbol.replace('/', '_') + '_1d.csv'
        filepath = DATA_CACHE / filename

        print(f"\n⬇ {symbol}")

        # Try both sources
        cg_df = download_from_coingecko(coin_id, symbol)
        time.sleep(1.5)  # CoinGecko rate limit

        bn_df = download_from_binance(symbol)

        # Merge
        df = merge_data(cg_df, bn_df)

        if df is not None and len(df) > 0:
            df.to_csv(filepath, index=False)

            years = (df['Date'].max() - df['Date'].min()).days / 365.25
            print(f"   ✓ {len(df):,} rows ({years:.1f} years)")
            print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

            # Calculate annual return
            if len(df) > 252:
                first_price = df['Close'].iloc[0]
                last_price = df['Close'].iloc[-1]
                total_return = (last_price / first_price) - 1
                annual_return = (1 + total_return) ** (1 / years) - 1
                print(f"   Annual return: {annual_return:.1%}")

            successful.append(symbol)
        else:
            print(f"   ✗ No data retrieved")
            failed.append(symbol)

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(successful)}/{len(CRYPTO_ASSETS)}")
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")

    print(f"\n📁 Data saved to: {DATA_CACHE}/")

if __name__ == "__main__":
    main()
