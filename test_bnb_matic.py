#!/usr/bin/env python3
"""
Test BNB and MATIC loading from various exchanges
"""
import ccxt
import pandas as pd
from datetime import datetime

def test_exchange(exchange_name, symbol):
    """Test if we can fetch data from an exchange"""
    print(f"\n{'='*70}")
    print(f"Testing {symbol} on {exchange_name}")
    print('='*70)

    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        print(f"✓ Exchange initialized: {exchange_name}")

        # Check if symbol is available
        markets = exchange.load_markets()
        if symbol in markets:
            print(f"✓ Symbol found: {symbol}")
        else:
            print(f"✗ Symbol NOT found: {symbol}")
            print(f"  Similar symbols: {[s for s in markets.keys() if symbol.split('/')[0] in s][:5]}")
            return False

        # Try to fetch recent data
        print(f"  Fetching OHLCV data...")
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=10)

        if ohlcv and len(ohlcv) > 0:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"✓ Fetched {len(df)} rows")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
            return True
        else:
            print(f"✗ No data returned")
            return False

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

# Test configurations
tests = [
    # BNB tests
    ('binance', 'BNB/USDT'),
    ('binanceus', 'BNB/USDT'),
    ('kucoin', 'BNB/USDT'),
    ('okx', 'BNB/USDT'),
    ('bybit', 'BNB/USDT'),

    # MATIC tests
    ('binance', 'MATIC/USDT'),
    ('binanceus', 'MATIC/USDT'),
    ('kucoin', 'MATIC/USDT'),
    ('okx', 'MATIC/USDT'),
    ('coinbase', 'MATIC/USDT'),
]

print("\n" + "="*70)
print("🔍 TESTING BNB AND MATIC ON MULTIPLE EXCHANGES")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

results = {}
for exchange_name, symbol in tests:
    success = test_exchange(exchange_name, symbol)
    key = f"{exchange_name}:{symbol}"
    results[key] = success

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

bnb_working = []
matic_working = []

for key, success in results.items():
    exchange, symbol = key.split(':')
    if success:
        print(f"✅ {key}")
        if 'BNB' in symbol:
            bnb_working.append(exchange)
        else:
            matic_working.append(exchange)
    else:
        print(f"❌ {key}")

print("\n" + "="*70)
if bnb_working:
    print(f"✅ BNB available on: {', '.join(bnb_working)}")
else:
    print("❌ BNB not available on any tested exchange")

if matic_working:
    print(f"✅ MATIC available on: {', '.join(matic_working)}")
else:
    print("❌ MATIC not available on any tested exchange")

print("="*70)

# Recommendations
print("\n💡 RECOMMENDED CONFIGURATION:")
if bnb_working:
    print(f"   BNB/USDT: Use '{bnb_working[0]}' exchange")
if matic_working:
    print(f"   MATIC/USDT: Use '{matic_working[0]}' exchange")
print()
