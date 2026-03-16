#!/usr/bin/env python3
"""
Download equity ETF data for multi-asset training.

Uses Yahoo Finance v8 chart API (unauthenticated, full history from inception).
Saves to data_cache/ as {TICKER}_1d.csv.

Usage:
    python download_equity_etfs.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import requests

DATA_CACHE = Path('data_cache')
DATA_CACHE.mkdir(exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# All equity/ETF assets matching config/assets.yaml
EQUITY_ETFS = {
    # Major indexes
    'SPY':  'S&P 500',
    'QQQ':  'Nasdaq 100',
    'IWM':  'Russell 2000',
    'DIA':  'Dow Jones',
    'VTI':  'Total Stock Market',
    'EFA':  'Developed Markets EAFE',
    'EEM':  'Emerging Markets',
    'VEA':  'Vanguard Developed Markets',
    'VWO':  'Vanguard Emerging Markets',
    'IEFA': 'iShares MSCI EAFE',

    # Sector ETFs
    'XLF':  'Financials',
    'XLK':  'Technology',
    'XLE':  'Energy',
    'XLV':  'Healthcare',
    'XLI':  'Industrials',
    'XLY':  'Consumer Discretionary',
    'XLP':  'Consumer Staples',
    'XLU':  'Utilities',
    'XLB':  'Materials',
    'XLRE': 'Real Estate',
    'XLC':  'Communications',

    # Style/Size
    'VUG':  'Vanguard Growth',
    'VTV':  'Vanguard Value',
    'VO':   'Vanguard Mid-Cap',
    'VB':   'Vanguard Small-Cap',
    'IJH':  'iShares Mid-Cap',
    'IJR':  'iShares Small-Cap',

    # Thematic
    'ARKK': 'ARK Innovation',
    'SOXX': 'Semiconductor',
    'SMH':  'VanEck Semiconductor',
    'XBI':  'Biotech',
    'TAN':  'Solar Energy',
    'ICLN': 'Clean Energy',

    # Bonds
    'AGG':  'Total Bond Market',
    'BND':  'Total Bond Market Vanguard',
    'TLT':  '20+ Year Treasuries',
    'IEF':  '7-10 Year Treasuries',
    'SHY':  '1-3 Year Treasuries',
    'LQD':  'Investment Grade Bonds',
    'HYG':  'High Yield Bonds',
    'JNK':  'SPDR High Yield Bonds',
    'MUB':  'Municipal Bonds',
    'EMB':  'Emerging Markets Bonds',

    # Precious metals
    'GLD':  'Gold Trust',
    'SLV':  'Silver Trust',
    'GDX':  'Gold Miners',
    'GDXJ': 'Junior Gold Miners',
    'IAU':  'iShares Gold',
    'PPLT': 'Platinum',
    'PALL': 'Palladium',

    # Broad commodities / energy
    'USO':  'US Oil Fund',
    'UNG':  'Natural Gas Fund',
    'DBC':  'Commodity Tracking',
}

START_DATE = '2000-01-01'
start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp())
end_ts = int(datetime.now().timestamp())

print("=" * 80)
print("MULTI-ASSET DATA DOWNLOAD")
print(f"Assets: {len(EQUITY_ETFS)} | Source: Yahoo Finance v8 Chart API")
print("=" * 80)


def _parse_yf_v8(data: dict) -> pd.DataFrame:
    """Parse Yahoo Finance v8 chart JSON into a standard OHLCV DataFrame."""
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quotes = result['indicators']['quote'][0]
    adjclose_list = result['indicators'].get('adjclose', [{}])
    adjclose = adjclose_list[0].get('adjclose', [None] * len(timestamps)) if adjclose_list else [None] * len(timestamps)

    df = pd.DataFrame({
        'Date':      pd.to_datetime(timestamps, unit='s').normalize(),
        'Open':      quotes.get('open'),
        'High':      quotes.get('high'),
        'Low':       quotes.get('low'),
        'Close':     quotes.get('close'),
        'Adj Close': adjclose,
        'Volume':    quotes.get('volume'),
    })
    df = df.dropna(subset=['Close'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


successful = []
failed = []
total_samples = 0

for ticker, name in EQUITY_ETFS.items():
    filename = f"{ticker}_1d.csv"
    filepath = DATA_CACHE / filename

    try:
        print(f"\n⬇ {ticker:6s} ({name:35s})", end='', flush=True)

        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={start_ts}&period2={end_ts}"
        )

        df = None
        last_error = None

        for attempt in range(3):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                raw = resp.json()

                if raw.get('chart', {}).get('error'):
                    raise ValueError(raw['chart']['error'])

                if not raw.get('chart', {}).get('result'):
                    raise ValueError("Empty result")

                df = _parse_yf_v8(raw)
                if len(df) == 0:
                    raise ValueError("No rows after parsing")
                break

            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    time.sleep(2 ** attempt)

        # Fallback: try query2 subdomain
        if df is None:
            url2 = url.replace('query1', 'query2')
            for attempt in range(2):
                try:
                    resp = requests.get(url2, headers=HEADERS, timeout=30)
                    resp.raise_for_status()
                    raw = resp.json()
                    if raw.get('chart', {}).get('result'):
                        df = _parse_yf_v8(raw)
                        if len(df) > 0:
                            break
                except Exception as e:
                    last_error = str(e)
                    if attempt < 1:
                        time.sleep(2)

        if df is None or len(df) == 0:
            raise ValueError(f"All attempts failed: {last_error}")

        df.to_csv(filepath, index=False)
        print(f" ✓  {len(df):,} rows  ({df['Date'].min().date()} → {df['Date'].max().date()})")

        successful.append(ticker)
        total_samples += len(df)
        time.sleep(0.5)

    except Exception as e:
        print(f" ✗  {e}")
        failed.append(ticker)
        time.sleep(1)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"✓ Successful: {len(successful)}/{len(EQUITY_ETFS)}")
print(f"✗ Failed:     {len(failed)}/{len(EQUITY_ETFS)}")
if failed:
    print(f"  Failed tickers: {', '.join(failed)}")
print(f"\nTotal rows downloaded: {total_samples:,}")
print(f"Data saved to: {DATA_CACHE}/")
