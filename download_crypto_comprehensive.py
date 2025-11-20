#!/usr/bin/env python3
"""
Comprehensive Crypto Data Downloader

Multiple data sources for maximum historical coverage:
1. CryptoCompare - Daily data back to 2010 for BTC (free API)
2. Binance - Accurate recent data (2017+)
3. CoinGecko - Good backup source
4. Alternative.me - Fear/Greed Index

Usage:
    python3 download_crypto_comprehensive.py                    # Download all
    python3 download_crypto_comprehensive.py --coins BTC,ETH    # Specific coins
    python3 download_crypto_comprehensive.py --source cryptocompare  # Specific source
"""

import os
import sys
import time
import json
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Create data directory
DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

# Comprehensive crypto list with CryptoCompare IDs
CRYPTO_ASSETS = {
    # Major cryptocurrencies
    'BTC/USDT': {'cc_symbol': 'BTC', 'name': 'Bitcoin', 'launch': '2010-07-17'},
    'ETH/USDT': {'cc_symbol': 'ETH', 'name': 'Ethereum', 'launch': '2015-08-07'},
    'BNB/USDT': {'cc_symbol': 'BNB', 'name': 'Binance Coin', 'launch': '2017-07-25'},
    'XRP/USDT': {'cc_symbol': 'XRP', 'name': 'Ripple', 'launch': '2013-08-04'},
    'ADA/USDT': {'cc_symbol': 'ADA', 'name': 'Cardano', 'launch': '2017-10-01'},
    'SOL/USDT': {'cc_symbol': 'SOL', 'name': 'Solana', 'launch': '2020-04-10'},
    'DOGE/USDT': {'cc_symbol': 'DOGE', 'name': 'Dogecoin', 'launch': '2013-12-15'},
    'DOT/USDT': {'cc_symbol': 'DOT', 'name': 'Polkadot', 'launch': '2020-08-22'},
    'MATIC/USDT': {'cc_symbol': 'MATIC', 'name': 'Polygon', 'launch': '2019-04-26'},
    'LTC/USDT': {'cc_symbol': 'LTC', 'name': 'Litecoin', 'launch': '2011-10-07'},
    'AVAX/USDT': {'cc_symbol': 'AVAX', 'name': 'Avalanche', 'launch': '2020-09-21'},
    'LINK/USDT': {'cc_symbol': 'LINK', 'name': 'Chainlink', 'launch': '2017-09-19'},
    'UNI/USDT': {'cc_symbol': 'UNI', 'name': 'Uniswap', 'launch': '2020-09-17'},
    'ATOM/USDT': {'cc_symbol': 'ATOM', 'name': 'Cosmos', 'launch': '2019-03-14'},
    'XLM/USDT': {'cc_symbol': 'XLM', 'name': 'Stellar', 'launch': '2014-08-01'},
}


def download_from_cryptocompare(symbol, to_symbol='USD', limit=2000, all_data=True):
    """
    Download historical data from CryptoCompare API.

    CryptoCompare has data going back to 2010 for BTC - much longer than Binance.
    Free tier: 100,000 calls/month

    Args:
        symbol: Crypto symbol (e.g., 'BTC')
        to_symbol: Quote currency (USD or USDT)
        limit: Max records per request (2000)
        all_data: If True, get all available history
    """
    print(f"   Downloading {symbol} from CryptoCompare...")

    api_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')

    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"

    all_data_list = []
    to_ts = None

    # Keep fetching until we get all data
    while True:
        params = {
            'fsym': symbol,
            'tsym': to_symbol,
            'limit': limit,
        }

        if api_key:
            params['api_key'] = api_key

        if to_ts:
            params['toTs'] = to_ts

        try:
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code != 200:
                print(f"   Error: HTTP {response.status_code}")
                break

            data = response.json()

            if data.get('Response') == 'Error':
                print(f"   API Error: {data.get('Message', 'Unknown error')}")
                break

            history = data.get('Data', {}).get('Data', [])

            if not history:
                break

            all_data_list = history + all_data_list

            # Check if we should continue
            if not all_data:
                break

            # Get timestamp of oldest record for next request
            oldest_ts = history[0]['time']

            # If oldest record is the same as our previous request, we're done
            if to_ts and oldest_ts >= to_ts:
                break

            to_ts = oldest_ts - 1

            # Rate limiting
            time.sleep(0.5)

            # Check if we have enough data or reached the beginning
            if len(all_data_list) > 5000:  # Max ~14 years of daily data
                break

        except Exception as e:
            print(f"   Request error: {e}")
            break

    if not all_data_list:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data_list)

    # Rename columns to match our format
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volumefrom': 'Volume'
    })

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Adj Close'] = df['Close']

    # Remove rows with zero/invalid data
    df = df[(df['Open'] > 0) & (df['Close'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]

    # Sort by date
    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

    print(f"   Got {len(df)} days of data from CryptoCompare")

    return df


def download_from_binance(symbol, interval='1d', limit=1000):
    """Download data from Binance (2017+ only)"""
    print(f"   Downloading {symbol} from Binance...")

    binance_symbol = symbol.replace('/', '')

    url = "https://api.binance.com/api/v3/klines"

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)

    # Go back as far as possible
    for _ in range(20):  # Max 20 iterations
        params = {
            'symbol': binance_symbol,
            'interval': interval,
            'endTime': end_time,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                break

            klines = response.json()

            if not klines:
                break

            all_data = klines + all_data

            # Update end time for next batch
            end_time = klines[0][0] - 1

            time.sleep(0.2)

        except Exception as e:
            print(f"   Binance error: {e}")
            break

    if not all_data:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base',
        'Taker_buy_quote', 'Ignore'
    ])

    df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Adj Close'] = df['Close']
    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

    print(f"   Got {len(df)} days from Binance")

    return df


def download_fear_greed_index():
    """Download Bitcoin Fear & Greed Index from Alternative.me"""
    print("\n📊 Downloading Fear & Greed Index...")

    url = "https://api.alternative.me/fng/?limit=0&format=json"

    try:
        response = requests.get(url, timeout=30)
        data = response.json()

        if 'data' not in data:
            print("   No data returned")
            return None

        records = []
        for item in data['data']:
            records.append({
                'Date': pd.to_datetime(int(item['timestamp']), unit='s'),
                'Fear_Greed_Value': int(item['value']),
                'Fear_Greed_Class': item['value_classification']
            })

        df = pd.DataFrame(records)
        df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

        # Save to CSV
        save_path = DATA_DIR / "fear_greed_index.csv"
        df.to_csv(save_path, index=False)
        print(f"   Saved {len(df)} days of Fear & Greed data to {save_path}")

        return df

    except Exception as e:
        print(f"   Error downloading Fear & Greed: {e}")
        return None


def merge_data_sources(cc_df, binance_df):
    """
    Merge CryptoCompare (long history) with Binance (accurate recent).

    Strategy:
    - Use CryptoCompare for older data (pre-2017)
    - Use Binance for recent data (more accurate OHLCV)
    """
    if cc_df is None and binance_df is None:
        return None

    if cc_df is None:
        return binance_df

    if binance_df is None:
        return cc_df

    # Use Binance data for overlapping period (more accurate)
    binance_start = binance_df['Date'].min()

    # Keep CryptoCompare data before Binance starts
    old_data = cc_df[cc_df['Date'] < binance_start]

    # Combine
    merged = pd.concat([old_data, binance_df], ignore_index=True)
    merged = merged.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

    return merged


def download_asset(ticker, asset_info):
    """Download data for a single asset from all sources"""
    print(f"\n{'='*60}")
    print(f"📥 {ticker} ({asset_info['name']})")
    print(f"   Launch date: {asset_info['launch']}")
    print(f"{'='*60}")

    cc_symbol = asset_info['cc_symbol']

    # Download from CryptoCompare (longer history)
    cc_df = download_from_cryptocompare(cc_symbol, to_symbol='USD')

    # Download from Binance (accurate recent data)
    binance_df = download_from_binance(ticker)

    # Merge sources
    final_df = merge_data_sources(cc_df, binance_df)

    if final_df is None or final_df.empty:
        print(f"   ❌ No data available for {ticker}")
        return False

    # Calculate some stats
    days = len(final_df)
    years = days / 365.25

    if days > 252:
        first_price = final_df['Close'].iloc[0]
        last_price = final_df['Close'].iloc[-1]
        total_return = (last_price / first_price - 1) * 100
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    else:
        total_return = 0
        annual_return = 0

    # Save to file
    filename = ticker.replace('/', '_') + '_1d.csv'
    save_path = DATA_DIR / filename
    final_df.to_csv(save_path, index=False)

    print(f"\n   ✅ Saved: {save_path}")
    print(f"   📈 {days} days ({years:.1f} years) of data")
    print(f"   📊 Date range: {final_df['Date'].min().date()} to {final_df['Date'].max().date()}")
    print(f"   💰 Total return: {total_return:,.1f}% ({annual_return:.1f}% annual)")

    return True


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Crypto Data Downloader')
    parser.add_argument('--coins', help='Comma-separated list of coins (e.g., BTC,ETH)')
    parser.add_argument('--source', choices=['all', 'cryptocompare', 'binance'],
                       default='all', help='Data source to use')
    parser.add_argument('--no-fng', action='store_true', help='Skip Fear & Greed index')
    args = parser.parse_args()

    print("=" * 60)
    print("  COMPREHENSIVE CRYPTO DATA DOWNLOADER")
    print("=" * 60)
    print("\nData sources:")
    print("  - CryptoCompare: Historical data back to 2010")
    print("  - Binance: Accurate recent data (2017+)")
    print("  - Alternative.me: Fear & Greed Index")
    print()

    # Determine which coins to download
    if args.coins:
        selected = [c.strip().upper() for c in args.coins.split(',')]
        assets_to_download = {}
        for ticker, info in CRYPTO_ASSETS.items():
            if info['cc_symbol'] in selected or ticker.split('/')[0] in selected:
                assets_to_download[ticker] = info
    else:
        assets_to_download = CRYPTO_ASSETS

    print(f"Downloading {len(assets_to_download)} cryptocurrencies...\n")

    # Download each asset
    results = {'success': [], 'failed': []}

    for ticker, info in assets_to_download.items():
        try:
            if download_asset(ticker, info):
                results['success'].append(ticker)
            else:
                results['failed'].append(ticker)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results['failed'].append(ticker)

        # Rate limiting between assets
        time.sleep(1)

    # Download Fear & Greed Index
    if not args.no_fng:
        download_fear_greed_index()

    # Summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"\n✅ Success: {len(results['success'])} assets")
    if results['success']:
        for ticker in results['success']:
            print(f"   - {ticker}")

    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])} assets")
        for ticker in results['failed']:
            print(f"   - {ticker}")

    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("1. Run training: python3 main.py -> Training")
    print("2. Run backtest: python3 main.py -> Backtesting")
    print("3. Check data: python3 main.py -> Data Management -> List assets")

    # Compare with previous data
    print("\n📊 Data Coverage Comparison:")
    for ticker in results['success'][:5]:  # Show first 5
        filepath = DATA_DIR / (ticker.replace('/', '_') + '_1d.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"   {ticker}: {len(df)} days from {df['Date'].min()[:10]}")


if __name__ == "__main__":
    main()
