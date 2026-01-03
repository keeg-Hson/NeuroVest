#!/bin/bash
# DataWorker2 Start Script
# Handles both initial bootstrap and ongoing worker operations

echo "======================================================================="
echo "🚀 NEUROVEST DATAWORKER2 STARTING"
echo "======================================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if DATABASE_URL is set (PostgreSQL)
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL not set"
    echo "   This worker requires PostgreSQL database"
    exit 1
fi

echo "✅ Database: PostgreSQL configured"
echo ""

# Check if this is first run (database empty/needs bootstrap)
echo "🔍 Checking if bootstrap is needed..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.data_manager_postgres import DataManager
dm = DataManager()
assets = dm.get_all_assets()
dm.close()

if len(assets) < 10:
    print(f'   Found {len(assets)} assets - BOOTSTRAP NEEDED')
    sys.exit(1)  # Need bootstrap
else:
    print(f'   Found {len(assets)} assets - Bootstrap already done')
    sys.exit(0)  # Skip bootstrap
"

NEEDS_BOOTSTRAP=$?

if [ $NEEDS_BOOTSTRAP -ne 0 ]; then
    echo ""
    echo "======================================================================="
    echo "🔧 RUNNING BOOTSTRAP (First-time setup)"
    echo "======================================================================="
    echo ""

    # Run full bootstrap
    bash bootstrap_all.sh
    BOOTSTRAP_EXIT=$?

    if [ $BOOTSTRAP_EXIT -ne 0 ]; then
        echo ""
        echo "⚠️  Bootstrap had issues but continuing to worker..."
    fi
fi

echo ""
echo "======================================================================="
echo "🔄 STARTING CONTINUOUS WORKER"
echo "======================================================================="
echo ""
echo "This worker will:"
echo "  - Update data every hour"
echo "  - Generate predictions daily at 4:30 PM EST"
echo "  - Retrain models weekly on Sundays at 2:00 AM EST"
echo ""
echo "Worker starting..."
echo "======================================================================="
echo ""

# Start the continuous worker
exec python3 worker_data_scheduler.py
