#!/usr/bin/env python3
"""
Live Data Update Mechanism

Provides continuous data updates for all asset types with scheduled refresh
and real-time prediction capabilities.

Usage:
    python3 live_update.py --mode scheduled    # Run scheduled updates
    python3 live_update.py --mode continuous   # Run continuous monitoring
    python3 live_update.py --assets SPY,QQQ    # Update specific assets
    python3 live_update.py --download          # Download all data

Environment variables:
    UPDATE_INTERVAL - Minutes between updates (default: 15)
    NOTIFY_ON_SIGNAL - Send notifications on high-confidence signals (default: true)
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/live_update.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("data_cache", exist_ok=True)

# Asset configurations
ASSET_CONFIGS = {
    'equity': {
        'assets': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
        'source': 'yfinance',
        'interval': '1d',
        'update_hours': [9, 12, 16],  # Market hours EST
    },
    'crypto': {
        'assets': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'],
        'source': 'binance',
        'interval': '1d',
        'update_hours': list(range(0, 24, 6)),  # Every 6 hours (24/7 market)
    },
    'bond': {
        'assets': ['TLT', 'IEF', 'SHY', 'BND', 'AGG'],
        'source': 'yfinance',
        'interval': '1d',
        'update_hours': [9, 16],
    },
    'commodity': {
        'assets': ['GLD', 'SLV', 'USO', 'UNG'],
        'source': 'yfinance',
        'interval': '1d',
        'update_hours': [9, 16],
    }
}


def download_yfinance_data(ticker, days=365*5):
    """Download data from Yahoo Finance"""
    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        # Reset index to get Date column
        data = data.reset_index()

        # Ensure proper column names
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})

        # Select and rename columns
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_cols = [c for c in columns if c in data.columns]
        data = data[available_cols]

        return data

    except Exception as e:
        logger.error(f"Error downloading {ticker} from yfinance: {e}")
        return None


def download_binance_data(symbol, days=365*5):
    """Download data from Binance"""
    try:
        import requests

        # Convert symbol format
        binance_symbol = symbol.replace('/', '')

        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        all_data = []
        current_start = start_time

        while current_start < end_time:
            params = {
                'symbol': binance_symbol,
                'interval': '1d',
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Binance API error: {response.status_code}")
                break

            klines = response.json()
            if not klines:
                break

            all_data.extend(klines)
            current_start = klines[-1][0] + 1

            # Rate limiting
            time.sleep(0.1)

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

        return df

    except Exception as e:
        logger.error(f"Error downloading {symbol} from Binance: {e}")
        return None


def update_asset_data(ticker, source='yfinance', days=365*5):
    """Update data for a single asset"""
    logger.info(f"Updating {ticker}...")

    # Download data
    if source == 'binance':
        df = download_binance_data(ticker, days)
    else:
        df = download_yfinance_data(ticker, days)

    if df is None or df.empty:
        logger.warning(f"No data retrieved for {ticker}")
        return False

    # Determine save path
    if '/' in ticker:
        filename = ticker.replace('/', '_') + '_1d.csv'
    else:
        filename = f"{ticker}_1d.csv"

    save_path = Path("data_cache") / filename

    # Also save to data/ for SPY
    if ticker == 'SPY':
        legacy_path = Path("data") / "SPY.csv"
        df.to_csv(legacy_path, index=False)
        logger.info(f"Saved {ticker} to {legacy_path}")

    # Save to data_cache
    df.to_csv(save_path, index=False)
    logger.info(f"Saved {ticker} to {save_path} ({len(df)} rows)")

    return True


def update_all_assets(asset_types=None):
    """Update data for all configured assets"""
    if asset_types is None:
        asset_types = list(ASSET_CONFIGS.keys())

    results = {'success': [], 'failed': []}

    for asset_type in asset_types:
        config = ASSET_CONFIGS.get(asset_type)
        if not config:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Updating {asset_type.upper()} assets")
        logger.info(f"{'='*50}")

        for ticker in config['assets']:
            if update_asset_data(ticker, config['source']):
                results['success'].append(ticker)
            else:
                results['failed'].append(ticker)

            # Rate limiting between assets
            time.sleep(1)

    return results


def run_predictions(assets):
    """Run predictions for specified assets"""
    from utils import add_features, load_asset_data
    from predict import live_predict

    predictions = {}

    for asset in assets:
        try:
            # Load and process data
            df = load_asset_data(asset)
            df, feature_cols = add_features(df)

            # Run prediction
            prediction, crash_conf, spike_conf = live_predict(df)

            predictions[asset] = {
                'prediction': int(prediction),
                'crash_conf': float(crash_conf),
                'spike_conf': float(spike_conf),
                'timestamp': datetime.now().isoformat(),
                'latest_price': float(df['Close'].iloc[-1])
            }

            logger.info(f"{asset}: Pred={prediction}, Crash={crash_conf:.2f}, Spike={spike_conf:.2f}")

        except Exception as e:
            logger.error(f"Prediction error for {asset}: {e}")
            predictions[asset] = {'error': str(e)}

    return predictions


def check_signals(predictions, thresholds=None):
    """Check for high-confidence signals"""
    if thresholds is None:
        thresholds = {
            'crash': 0.7,
            'spike': 0.7
        }

    signals = []

    for asset, pred in predictions.items():
        if 'error' in pred:
            continue

        crash_conf = pred.get('crash_conf', 0)
        spike_conf = pred.get('spike_conf', 0)

        if pred.get('prediction') == 0 and crash_conf > thresholds['crash']:
            signals.append({
                'asset': asset,
                'type': 'CRASH',
                'confidence': crash_conf,
                'price': pred.get('latest_price')
            })
        elif pred.get('prediction') == 2 and spike_conf > thresholds['spike']:
            signals.append({
                'asset': asset,
                'type': 'SPIKE',
                'confidence': spike_conf,
                'price': pred.get('latest_price')
            })

    return signals


def send_notification(signals):
    """Send notification for high-confidence signals"""
    if not signals:
        return

    from utils import send_telegram_alert

    message = "NeuroVest Alert!\n\n"
    for sig in signals:
        emoji = "" if sig['type'] == 'CRASH' else ""
        message += f"{emoji} {sig['asset']}: {sig['type']}\n"
        message += f"   Confidence: {sig['confidence']:.1%}\n"
        message += f"   Price: ${sig['price']:.2f}\n\n"

    send_telegram_alert(message)
    logger.info(f"Sent notification for {len(signals)} signal(s)")


def save_state(predictions, signals):
    """Save current state to file"""
    state = {
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions,
        'signals': signals
    }

    state_path = Path("logs/live_state.json")
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def run_scheduled_updates(assets=None, interval_minutes=15):
    """Run scheduled updates at regular intervals"""
    logger.info(f"Starting scheduled updates every {interval_minutes} minutes")

    if assets is None:
        # Default to equity assets
        assets = ASSET_CONFIGS['equity']['assets']

    while True:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Scheduled update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")

            # Update data
            for asset in assets:
                # Determine source
                if '/USDT' in asset:
                    source = 'binance'
                else:
                    source = 'yfinance'

                update_asset_data(asset, source, days=30)  # Just recent data for updates

            # Run predictions
            predictions = run_predictions(assets)

            # Check for signals
            signals = check_signals(predictions)

            # Save state
            save_state(predictions, signals)

            # Send notifications if enabled
            if os.getenv('NOTIFY_ON_SIGNAL', 'true').lower() == 'true' and signals:
                send_notification(signals)

            # Log signals
            if signals:
                logger.info(f"Found {len(signals)} high-confidence signal(s):")
                for sig in signals:
                    logger.info(f"  {sig['asset']}: {sig['type']} ({sig['confidence']:.1%})")

        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")

        # Wait for next interval
        logger.info(f"Next update in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def run_continuous_monitoring(assets=None):
    """Run continuous monitoring with adaptive intervals"""
    logger.info("Starting continuous monitoring mode")

    if assets is None:
        assets = ['SPY']

    # Check if market is open (simplified)
    def is_market_hours():
        now = datetime.now()
        # US market hours: 9:30 AM - 4:00 PM EST (adjust for timezone)
        hour = now.hour
        return 9 <= hour < 16  # Simplified check

    while True:
        try:
            # Adaptive interval based on market hours
            if is_market_hours():
                interval = 5  # More frequent during market hours
            else:
                interval = 30  # Less frequent outside market hours

            # Run update cycle
            for asset in assets:
                if '/USDT' in asset:
                    source = 'binance'
                else:
                    source = 'yfinance'

                update_asset_data(asset, source, days=7)

            # Run predictions
            predictions = run_predictions(assets)

            # Check signals
            signals = check_signals(predictions)

            # Save state
            save_state(predictions, signals)

            # Notify on signals
            if signals:
                send_notification(signals)

            logger.info(f"Monitoring cycle complete. Next in {interval} minutes.")
            time.sleep(interval * 60)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Live data update mechanism")
    parser.add_argument("--mode", choices=['scheduled', 'continuous', 'once'],
                       default='once', help="Update mode")
    parser.add_argument("--assets", help="Comma-separated list of assets")
    parser.add_argument("--download", action="store_true",
                       help="Download all historical data")
    parser.add_argument("--interval", type=int, default=15,
                       help="Update interval in minutes")
    parser.add_argument("--predict", action="store_true",
                       help="Run predictions after update")
    args = parser.parse_args()

    # Parse assets
    if args.assets:
        assets = [a.strip() for a in args.assets.split(',')]
    else:
        assets = None

    print("=" * 60)
    print("  NEUROVEST LIVE UPDATE MECHANISM")
    print("=" * 60)

    if args.download:
        # Download all historical data
        logger.info("Downloading all historical data...")
        results = update_all_assets()
        print(f"\nSuccess: {len(results['success'])} assets")
        print(f"Failed: {len(results['failed'])} assets")
        if results['failed']:
            print(f"Failed assets: {', '.join(results['failed'])}")

    elif args.mode == 'scheduled':
        # Run scheduled updates
        run_scheduled_updates(assets, args.interval)

    elif args.mode == 'continuous':
        # Run continuous monitoring
        run_continuous_monitoring(assets)

    else:
        # Single update
        if assets:
            for asset in assets:
                source = 'binance' if '/USDT' in asset else 'yfinance'
                update_asset_data(asset, source)
        else:
            # Update default assets
            for asset in ['SPY', 'QQQ']:
                update_asset_data(asset, 'yfinance')

        if args.predict:
            predictions = run_predictions(assets or ['SPY'])
            signals = check_signals(predictions)
            if signals:
                print("\nHigh-confidence signals detected:")
                for sig in signals:
                    print(f"  {sig['asset']}: {sig['type']} ({sig['confidence']:.1%})")


if __name__ == "__main__":
    main()
