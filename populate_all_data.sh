#!/bin/bash
# Complete data population script
# Downloads all missing data and generates predictions

echo "======================================================================="
echo "🔧 NEUROVEST - COMPLETE DATA POPULATION"
echo "======================================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "This will:"
echo "  1. Download full crypto history (~3000 days per asset)"
echo "  2. Download missing stock data"
echo "  3. Generate predictions for all 40 assets"
echo ""
echo "Estimated time: 30-60 minutes"
echo "======================================================================="
echo ""

# Step 1: Reload crypto with maximum history
echo "STEP 1/3: Loading crypto data (3000 days)..."
echo "-----------------------------------------------------------------------"
python3 reload_crypto_max_history.py
CRYPTO_EXIT=$?

if [ $CRYPTO_EXIT -ne 0 ]; then
    echo "⚠️  Crypto reload had issues (exit code $CRYPTO_EXIT)"
    echo "Some crypto assets may have limited data"
fi

echo ""
echo "======================================================================="
echo ""

# Step 2: Verify data and show counts
echo "STEP 2/3: Verifying database..."
echo "-----------------------------------------------------------------------"
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from core.data_manager_postgres import DataManager
from sqlalchemy import text

dm = DataManager()

# Get asset counts by type
with dm.engine.connect() as conn:
    result = conn.execute(text("""
        SELECT asset_type, COUNT(DISTINCT ticker) as count
        FROM assets
        GROUP BY asset_type
        ORDER BY asset_type
    """))

    print("📊 Assets by type:")
    for row in result:
        print(f"  {row[0]}: {row[1]} assets")

    # Get assets with data
    result = conn.execute(text("""
        SELECT a.ticker, a.asset_type, COUNT(p.id) as rows
        FROM assets a
        LEFT JOIN price_data p ON a.ticker = p.ticker
        GROUP BY a.ticker, a.asset_type
        ORDER BY a.asset_type, COUNT(p.id) DESC
    """))

    print("\n📈 Data status:")
    stocks_with_data = 0
    crypto_with_data = 0
    total_rows = 0

    for row in result:
        ticker, asset_type, rows = row
        total_rows += rows

        if rows > 0:
            if asset_type == 'stock':
                stocks_with_data += 1
            elif asset_type == 'crypto':
                crypto_with_data += 1

            status = "✅" if rows > 500 else "⚠️ "
            print(f"  {status} {ticker:15s} ({asset_type:6s}): {rows:6,} rows")
        else:
            print(f"  ❌ {ticker:15s} ({asset_type:6s}): NO DATA")

    print(f"\n📊 Summary:")
    print(f"  Stocks with data: {stocks_with_data}")
    print(f"  Crypto with data: {crypto_with_data}")
    print(f"  Total data rows: {total_rows:,}")

dm.close()
PYTHON_SCRIPT

echo ""
echo "======================================================================="
echo ""

# Step 3: Generate predictions
echo "STEP 3/3: Generating predictions for all assets..."
echo "-----------------------------------------------------------------------"

if [ -f "predict_multi_asset_ensemble.py" ]; then
    python3 predict_multi_asset_ensemble.py
    PRED_EXIT=$?

    if [ $PRED_EXIT -ne 0 ]; then
        echo "⚠️  Prediction generation failed (exit code $PRED_EXIT)"
        echo "Models may need training first"
    fi
else
    echo "⚠️  predict_multi_asset_ensemble.py not found"
    echo "Skipping prediction generation"
    PRED_EXIT=1
fi

echo ""
echo "======================================================================="
echo "✅ DATA POPULATION COMPLETE"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Refresh the dashboard to see updated data"
echo "  2. Check that all 40 assets show 'Downloaded' status"
echo "  3. Verify predictions show for all assets"
echo ""
echo "======================================================================="
