#!/bin/bash
# Quick Fix: Generate Fresh Predictions NOW
# Run this manually if predictions are stale

echo "======================================================================="
echo "🔮 GENERATING FRESH PREDICTIONS"
echo "======================================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================="
echo ""

# Check if we have models (need training first)
echo "STEP 1/3: Checking if models exist..."
echo "-----------------------------------------------------------------------"
python3 -c "
import sys
sys.path.insert(0, '.')
from core.data_manager_postgres import DataManager
dm = DataManager()
models = dm.get_all_models()
dm.close()

if len(models) == 0:
    print('   ⚠️  No models found - training needed first')
    sys.exit(1)
else:
    print(f'   ✅ Found {len(models)} trained models')
    sys.exit(0)
"

HAS_MODELS=$?

if [ $HAS_MODELS -ne 0 ]; then
    echo ""
    echo "STEP 2/3: Training models (first time)..."
    echo "-----------------------------------------------------------------------"

    if [ -f "train_multi_asset.py" ]; then
        python3 train_multi_asset.py
        TRAIN_EXIT=$?

        if [ $TRAIN_EXIT -ne 0 ]; then
            echo ""
            echo "❌ Training failed - cannot generate predictions"
            exit 1
        fi
    else
        echo "❌ train_multi_asset.py not found"
        exit 1
    fi
else
    echo ""
    echo "STEP 2/3: Models already trained, skipping..."
    echo "-----------------------------------------------------------------------"
fi

echo ""
echo "STEP 3/3: Generating predictions..."
echo "-----------------------------------------------------------------------"

if [ -f "predict_multi_asset_ensemble.py" ]; then
    python3 predict_multi_asset_ensemble.py
    PRED_EXIT=$?

    if [ $PRED_EXIT -ne 0 ]; then
        echo ""
        echo "❌ Prediction generation failed"
        exit 1
    fi
else
    echo "❌ predict_multi_asset_ensemble.py not found"
    exit 1
fi

echo ""
echo "======================================================================="
echo "✅ FRESH PREDICTIONS GENERATED SUCCESSFULLY!"
echo "======================================================================="
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Refresh your dashboard to see the updated predictions."
echo "======================================================================="
echo ""

exit 0
