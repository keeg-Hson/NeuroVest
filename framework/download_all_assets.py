#!/usr/bin/env python3
"""
Unified Asset Downloader

Automatically downloads all enabled assets from config/assets.yaml
Supports:  - Equity/Bond/Commodity ETFs via multiple sources (Alpha Vantage, pandas_datareader, manual)
- Cryptocurrencies via CCXT

Usage:
    python framework/download_all_assets.py
    python framework/download_all_assets.py --api-key YOUR_ALPHA_VANTAGE_KEY
    python framework/download_all_assets.py --type equity  # Download only equities
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not required, can use shell env vars

# Optional imports with fallbacks
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

try:
    from alpha_vantage.timeseries import TimeSeries
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

# Handle both direct execution and module import
try:
    from .asset_manager import AssetManager
except ImportError:
    # Running directly, add parent dir to path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from framework.asset_manager import AssetManager


class UnifiedDownloader:
    """Downloads all assets from configuration"""

    def __init__(self, data_dir: str = "data_cache", alpha_vantage_key: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.manager = AssetManager()
        # Try both ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_KEY (for .env.example compatibility)
        self.alpha_vantage_key = (
            alpha_vantage_key or
            os.getenv('ALPHA_VANTAGE_API_KEY') or
            os.getenv('ALPHA_VANTAGE_KEY')
        )
        self.settings = self.manager.get_settings()

        self.successful = []
        self.failed = []
        self.skipped = []

    def download_all(self, asset_type: Optional[str] = None, skip_existing: bool = True):
        """Download all enabled assets"""
        if asset_type:
            assets = self.manager.get_assets_by_type(asset_type)
            print(f"\n📥 Downloading {asset_type} assets...")
        else:
            assets = self.manager.get_all_assets()
            print(f"\n📥 Downloading all enabled assets...")

        print(f"Total to download: {len(assets)}")

        # Debug: Show if Alpha Vantage key is set
        if self.alpha_vantage_key:
            print(f"✓ Alpha Vantage API key: {self.alpha_vantage_key[:8]}..." if len(self.alpha_vantage_key) > 8 else "✓ Alpha Vantage API key set")
        else:
            print(f"⚠️  No Alpha Vantage API key found in environment")
            print(f"   Set ALPHA_VANTAGE_KEY in .env or export ALPHA_VANTAGE_API_KEY")

        print("=" * 80)

        for asset in assets:
            if asset.asset_type == 'crypto':
                self._download_crypto(asset, skip_existing)
            else:
                self._download_etf(asset, skip_existing)

        self._print_summary()

    def _download_crypto(self, asset, skip_existing: bool):
        """Download crypto via CCXT"""
        if not CCXT_AVAILABLE:
            print(f"⚠️  {asset.ticker}: CCXT not installed, skipping")
            self.failed.append(asset.ticker)
            return

        # Convert ticker format: BTC/USDT -> BTC_USDT for filename
        filename = asset.ticker.replace('/', '_') + '_1d.csv'
        filepath = self.data_dir / filename

        if skip_existing and filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✓ {asset.ticker:15s} - Already downloaded ({len(df):,} rows)")
            self.skipped.append(asset.ticker)
            return

        try:
            print(f"⬇ {asset.ticker:15s} - Downloading from {asset.exchange}...")

            exchange = getattr(ccxt, asset.exchange)({'enableRateLimit': True})

            start_date = self.settings.get('start_date', '2000-01-01')
            since = exchange.parse8601(f'{start_date}T00:00:00Z')

            # Fetch all historical data in chunks (CCXT limits to ~500-1000 per request)
            all_ohlcv = []
            limit = 1000  # Max candles per request

            while True:
                ohlcv = exchange.fetch_ohlcv(asset.ticker, '1d', since=since, limit=limit)

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Move since to last timestamp + 1 day
                since = ohlcv[-1][0] + 86400000  # +1 day in milliseconds

                # If we got less than limit, we've reached the end
                if len(ohlcv) < limit:
                    break

                time.sleep(1)  # Rate limiting between requests

            if not all_ohlcv:
                raise ValueError("No data returned")

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Remove duplicates (can happen at boundaries)
            df = df.drop_duplicates(subset=['Date'], keep='first')

            df.to_csv(filepath, index=False)

            print(f"   ✓ {len(df):,} rows | {df['Date'].min()} to {df['Date'].max()}")
            self.successful.append(asset.ticker)

        except Exception as e:
            print(f"   ✗ Error: {e}")
            self.failed.append(asset.ticker)

    def _download_etf(self, asset, skip_existing: bool):
        """Download ETF via multiple fallback sources"""
        filename = f"{asset.ticker}_1d.csv"
        filepath = self.data_dir / filename

        if skip_existing and filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✓ {asset.ticker:15s} - Already downloaded ({len(df):,} rows)")
            self.skipped.append(asset.ticker)
            return

        print(f"⬇ {asset.ticker:15s} - Downloading...")

        # Try Method 1: pandas_datareader
        df = self._try_pandas_datareader(asset)

        # Try Method 2: Alpha Vantage
        if df is None:
            df = self._try_alpha_vantage(asset)

        # Method 3: Manual instructions
        if df is None:
            self._provide_manual_instructions(asset)
            self.failed.append(asset.ticker)
            return

        # Save successful download
        try:
            df.to_csv(filepath, index=False)
            print(f"   ✓ {len(df):,} rows | {df['Date'].min()} to {df['Date'].max()}")
            self.successful.append(asset.ticker)
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"   ✗ Error saving: {e}")
            self.failed.append(asset.ticker)

    def _try_pandas_datareader(self, asset) -> Optional[pd.DataFrame]:
        """Try downloading via pandas_datareader"""
        if not PDR_AVAILABLE:
            print(f"      ℹ️  pandas_datareader not installed")
            return None

        try:
            print(f"   [1] Trying pandas_datareader (Yahoo Finance)...")
            start_date = self.settings.get('start_date', '2000-01-01')
            end_date = datetime.now().strftime('%Y-%m-%d')

            df = pdr.get_data_yahoo(asset.ticker, start=start_date, end=end_date)
            df.reset_index(inplace=True)
            print(f"      ✓ pandas_datareader success!")
            return df
        except Exception as e:
            print(f"      ⚠️  pandas_datareader failed: {str(e)[:60]}")
            return None

    def _try_alpha_vantage(self, asset) -> Optional[pd.DataFrame]:
        """Try downloading via Alpha Vantage"""
        if not AV_AVAILABLE:
            print(f"      ℹ️  alpha_vantage library not installed")
            return None

        if not self.alpha_vantage_key:
            print(f"      ℹ️  No Alpha Vantage API key set")
            return None

        try:
            print(f"      [2] Trying Alpha Vantage API...")
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            df, _ = ts.get_daily(symbol=asset.ticker, outputsize='full')

            # Rename columns
            df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)

            df.reset_index(inplace=True)
            df.rename(columns={'date': 'Date'}, inplace=True)

            # Filter date range
            start_date = self.settings.get('start_date', '2000-01-01')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] >= start_date]

            print(f"      ✓ Alpha Vantage success!")
            time.sleep(12)  # Alpha Vantage: 5 calls/min

            return df
        except Exception as e:
            print(f"      ⚠️  Alpha Vantage failed: {e}")
            return None

    def _provide_manual_instructions(self, asset):
        """Provide manual download instructions"""
        print(f"   ℹ️  Automatic download failed")
        print(f"   📋 Manual Download:")
        print(f"      1. Visit: https://finance.yahoo.com/quote/{asset.ticker}/history")
        print(f"      2. Download CSV and save to: {self.data_dir}/{asset.ticker}_1d.csv")

    def _print_summary(self):
        """Print download summary"""
        total = len(self.successful) + len(self.failed) + len(self.skipped)

        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"✓ Successful:  {len(self.successful):3d} / {total}")
        print(f"⊘ Skipped:     {len(self.skipped):3d} / {total} (already downloaded)")
        print(f"✗ Failed:      {len(self.failed):3d} / {total}")

        if self.failed:
            print(f"\n⚠️  Failed: {', '.join(self.failed[:10])}")
            if len(self.failed) > 10:
                print(f"   ... and {len(self.failed) - 10} more")

            if not self.alpha_vantage_key:
                print(f"\n💡 Tip: Get free Alpha Vantage API key:")
                print(f"   https://www.alphavantage.co/support/#api-key")
                print(f"   export ALPHA_VANTAGE_API_KEY='your-key'")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Download all configured assets")
    parser.add_argument('--type', choices=['equity', 'bond', 'commodity', 'crypto'],
                        help="Download only specific asset type")
    parser.add_argument('--asset-group', choices=['equity', 'bond', 'commodity', 'crypto'],
                        help="Download only specific asset group (alias for --type)")
    parser.add_argument('--asset', type=str,
                        help="Download single asset by ticker (e.g., GLD, SPY, BTC/USDT)")
    parser.add_argument('--api-key', help="Alpha Vantage API key")
    parser.add_argument('--force', action='store_true',
                        help="Re-download even if files exist")

    args = parser.parse_args()

    print("=" * 80)
    print("UNIFIED ASSET DOWNLOADER")
    print("=" * 80)

    downloader = UnifiedDownloader(alpha_vantage_key=args.api_key)

    # Handle single asset download
    if args.asset:
        manager = downloader.manager
        asset_obj = None
        for asset in manager.get_all_assets(enabled_only=False):
            if asset.ticker.upper() == args.asset.upper():
                asset_obj = asset
                break

        if not asset_obj:
            print(f"❌ Asset '{args.asset}' not found in configuration")
            print(f"\nAvailable assets:")
            for asset in sorted(manager.get_all_assets(enabled_only=False), key=lambda a: a.ticker):
                status = "✓" if asset.enabled else "✗"
                print(f"   {status} {asset.ticker:15s} - {asset.name} ({asset.asset_type})")
            return

        print(f"\n📥 Downloading single asset: {asset_obj.ticker}")
        print("=" * 80)

        if asset_obj.asset_type == 'crypto':
            downloader._download_crypto(asset_obj, skip_existing=not args.force)
        else:
            downloader._download_etf(asset_obj, skip_existing=not args.force)

        downloader._print_summary()
    else:
        # Handle group/type download (--asset-group takes precedence over --type)
        asset_type = args.asset_group or args.type
        downloader.download_all(asset_type=asset_type, skip_existing=not args.force)


if __name__ == "__main__":
    main()
