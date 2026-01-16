#!/usr/bin/env python3
"""
Infrastructure Health Check
Verifies crypto asset synchronization and data integrity
"""
import sys
sys.path.insert(0, '.')

print("="*80)
print("NEUROVEST INFRASTRUCTURE HEALTH CHECK")
print("="*80)

# Check 1: Asset list synchronization
print("\n1. CHECKING ASSET LIST SYNCHRONIZATION")
print("-"*80)

# Expected crypto list (from reload_crypto_max_history.py)
expected_cryptos = [
    'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT',
    'XRP_USDT', 'ADA_USDT', 'DOGE_USDT', 'AVAX_USDT',
    'MATIC_USDT', 'LINK_USDT'
]

print(f"Expected crypto assets (reload script): {len(expected_cryptos)}")
print("  " + ", ".join(expected_cryptos))

# Check worker script
try:
    with open('worker_data_scheduler.py', 'r') as f:
        worker_content = f.read()
        worker_cryptos = []
        for crypto in expected_cryptos:
            if crypto in worker_content:
                worker_cryptos.append(crypto)

    print(f"\nFound in worker script: {len(worker_cryptos)}/{len(expected_cryptos)}")

    missing_from_worker = set(expected_cryptos) - set(worker_cryptos)
    if missing_from_worker:
        print(f"❌ MISSING FROM WORKER: {', '.join(missing_from_worker)}")
    else:
        print("✅ All crypto assets present in worker script")

except Exception as e:
    print(f"⚠️  Error checking worker script: {e}")

# Check 2: Database state
print("\n\n2. CHECKING DATABASE STATE")
print("-"*80)

try:
    from core.data_manager_postgres import DataManager
    from sqlalchemy import text

    dm = DataManager()

    # Check crypto data
    with dm.engine.connect() as conn:
        # Count crypto rows
        crypto_query = text("""
            SELECT ticker, COUNT(*) as row_count
            FROM price_data
            WHERE ticker IN :tickers
            GROUP BY ticker
            ORDER BY ticker
        """)

        # Convert list to tuple for SQL IN clause
        result = conn.execute(
            text("""
                SELECT ticker, COUNT(*) as row_count
                FROM price_data
                WHERE ticker LIKE '%_USDT'
                GROUP BY ticker
                ORDER BY ticker
            """)
        )

        db_cryptos = {row[0]: row[1] for row in result}

        print(f"Crypto assets with data: {len(db_cryptos)}/{len(expected_cryptos)}")
        print(f"\n{'Ticker':<15} {'Rows':<10} {'Status':<15}")
        print("-"*80)

        for ticker in expected_cryptos:
            rows = db_cryptos.get(ticker, 0)
            status = "✅ Has data" if rows > 0 else "❌ NO DATA"
            print(f"{ticker:<15} {rows:<10,} {status:<15}")

        missing_data = [t for t in expected_cryptos if db_cryptos.get(t, 0) == 0]
        if missing_data:
            print(f"\n❌ {len(missing_data)} cryptos missing data: {', '.join(missing_data)}")
        else:
            print("\n✅ All cryptos have data")

        # Check predictions
        pred_result = conn.execute(text("""
            SELECT COUNT(DISTINCT ticker) FROM predictions
            WHERE ticker LIKE '%_USDT'
        """))
        crypto_preds = pred_result.scalar()

        print(f"\nCrypto predictions: {crypto_preds}/{len(expected_cryptos)}")

        if crypto_preds < len(expected_cryptos):
            print(f"❌ Missing predictions for {len(expected_cryptos) - crypto_preds} cryptos")
        else:
            print("✅ All cryptos have predictions")

    dm.close()

except Exception as e:
    print(f"⚠️  Error checking database: {e}")
    import traceback
    traceback.print_exc()

# Check 3: File synchronization
print("\n\n3. CHECKING FILE SYNCHRONIZATION")
print("-"*80)

files_to_check = [
    ('worker_data_scheduler.py', 'Background worker configuration'),
    ('reload_crypto_max_history.py', 'Crypto reload script'),
    ('bootstrap_all.sh', 'Bootstrap script'),
    ('predict_all_assets.py', 'Prediction script'),
    ('train_multi_asset.py', 'Training script')
]

for filename, description in files_to_check:
    try:
        with open(filename, 'r') as f:
            content = f.read()
            if 'predict_all_assets.py' in content or 'LINK_USDT' in content:
                print(f"✅ {filename:<35} ({description})")
            else:
                print(f"⚠️  {filename:<35} (may need update)")
    except FileNotFoundError:
        print(f"❌ {filename:<35} NOT FOUND")

print("\n" + "="*80)
print("HEALTH CHECK COMPLETE")
print("="*80)
