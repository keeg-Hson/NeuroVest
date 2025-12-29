#!/bin/bash
# Bootstrap Production - One-time setup
# Run this ONCE to initialize the production system

echo "======================================================================="
echo "🚀 NEUROVEST PRODUCTION BOOTSTRAP"
echo "======================================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "This script will:"
echo "  1. Load all historical data (~5-10 min)"
echo "  2. Train all models (~15-30 min)"
echo "  3. Generate predictions (~5 min)"
echo ""
echo "Total time: ~30-60 minutes"
echo "======================================================================="
echo ""

# Step 1: Load data
echo "STEP 1/3: Loading historical data..."
echo "-----------------------------------------------------------------------"
python3 bootstrap_data_load.py
DATA_EXIT=$?

if [ $DATA_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  Data loading had issues (exit code $DATA_EXIT)"
    echo "Continuing anyway..."
fi

echo ""
echo "======================================================================="
echo ""

# Step 2: Train models
if [ -f "train_multi_asset.py" ]; then
    echo "STEP 2/3: Training models..."
    echo "-----------------------------------------------------------------------"
    python3 train_multi_asset.py
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo ""
        echo "⚠️  Model training failed (exit code $TRAIN_EXIT)"
        echo "You may need to debug this manually"
    fi
else
    echo "STEP 2/3: SKIPPED (train_multi_asset.py not found)"
    TRAIN_EXIT=1
fi

echo ""
echo "======================================================================="
echo ""

# Step 3: Generate predictions
if [ -f "predict_multi_asset_ensemble.py" ]; then
    echo "STEP 3/3: Generating predictions..."
    echo "-----------------------------------------------------------------------"
    python3 predict_multi_asset_ensemble.py
    PRED_EXIT=$?

    if [ $PRED_EXIT -ne 0 ]; then
        echo ""
        echo "⚠️  Prediction generation failed (exit code $PRED_EXIT)"
        echo "You may need to debug this manually"
    fi
else
    echo "STEP 3/3: SKIPPED (predict_multi_asset_ensemble.py not found)"
    PRED_EXIT=1
fi

echo ""
echo "======================================================================="
echo "📊 BOOTSTRAP COMPLETE"
echo "======================================================================="
echo "  Data Load:  Exit code $DATA_EXIT"
echo "  Training:   Exit code $TRAIN_EXIT"
echo "  Predictions: Exit code $PRED_EXIT"
echo "======================================================================="
echo ""

if [ $DATA_EXIT -eq 0 ] && [ $TRAIN_EXIT -eq 0 ] && [ $PRED_EXIT -eq 0 ]; then
    echo "✅ ALL STEPS SUCCESSFUL - PRODUCTION READY!"
    echo ""
    echo "Your worker can now handle incremental updates."
    echo "Start the worker with: bash start_combined.sh"
    exit 0
else
    echo "⚠️  SOME STEPS FAILED - CHECK LOGS ABOVE"
    echo ""
    echo "You can run individual steps manually:"
    echo "  python3 bootstrap_data_load.py"
    echo "  python3 train_multi_asset.py"
    echo "  python3 predict_multi_asset_ensemble.py"
    exit 1
fi
