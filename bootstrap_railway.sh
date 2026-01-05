#!/bin/bash
# Railway Bootstrap - Train models and generate predictions
# Use this for Railway deployment's first run

echo "======================================================================="
echo "🚀 RAILWAY BOOTSTRAP - NEUROVEST API"
echo "======================================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 1: Train models (required for predictions)
echo "STEP 1/2: Training multi-asset models..."
echo "-----------------------------------------------------------------------"
python3 train_multi_asset.py
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Model training failed (exit code $TRAIN_EXIT)"
    echo "Cannot proceed without trained models"
    exit 1
fi

echo ""
echo "======================================================================="
echo ""

# Step 2: Generate predictions for all assets
echo "STEP 2/2: Generating predictions for all assets..."
echo "-----------------------------------------------------------------------"
python3 predict_all_assets.py
PRED_EXIT=$?

if [ $PRED_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Prediction generation failed (exit code $PRED_EXIT)"
    exit 1
fi

echo ""
echo "======================================================================="
echo "✅ RAILWAY BOOTSTRAP COMPLETE!"
echo "======================================================================="
echo "  Training:    Exit code $TRAIN_EXIT"
echo "  Predictions: Exit code $PRED_EXIT"
echo "======================================================================="
echo ""
echo "✅ Models trained and predictions generated!"
echo "Now starting worker scheduler..."
echo ""

# Step 3: Start the ongoing worker
exec python3 worker_data_scheduler.py
