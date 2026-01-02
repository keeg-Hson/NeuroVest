#!/bin/bash
# Quick status check for NeuroVest
# Run this anytime to check system health

echo "======================================================================"
echo "  NEUROVEST SYSTEM STATUS"
echo "  $(date)"
echo "======================================================================"

echo ""
echo "📁 DATA FILES"
echo "----------------------------------------------------------------------"

# Check SPY data
if [ -f "data/SPY.csv" ]; then
    lines=$(wc -l < data/SPY.csv)
    echo "✓ SPY.csv: $lines rows"
else
    echo "✗ SPY.csv: NOT FOUND (run: python3 update_spy_data.py)"
fi

# Check crypto data
crypto_count=0
for crypto in BTC_USDT ETH_USDT SOL_USDT; do
    if [ -f "data_cache/${crypto}_1d.csv" ]; then
        lines=$(wc -l < "data_cache/${crypto}_1d.csv")
        echo "✓ ${crypto}_1d.csv: $lines rows"
        ((crypto_count++))
    fi
done

if [ $crypto_count -eq 0 ]; then
    echo "  No crypto data (run: python3 download_crypto_enhanced.py)"
fi

echo ""
echo "🤖 MODELS"
echo "----------------------------------------------------------------------"

# Check models
model_count=0
for model in xgboost_multi_asset.pkl lightgbm_multi_asset.pkl catboost_multi_asset.pkl; do
    if [ -f "models/$model" ]; then
        size=$(du -h "models/$model" | cut -f1)
        echo "✓ $model: $size"
        ((model_count++))
    else
        echo "✗ $model: NOT FOUND"
    fi
done

if [ $model_count -eq 0 ]; then
    echo "  No models (run: python3 train_multi_asset.py)"
fi

echo ""
echo "📊 PREDICTIONS"
echo "----------------------------------------------------------------------"

if [ -f "logs/labeled_predictions.csv" ]; then
    lines=$(wc -l < logs/labeled_predictions.csv)
    echo "✓ Predictions: $lines rows"

    # Show last prediction date (if file has Date column)
    last_line=$(tail -1 logs/labeled_predictions.csv)
    echo "  Latest entry: $last_line"
else
    echo "✗ No predictions (run: python3 predict_multi_asset_ensemble.py)"
fi

echo ""
echo "📈 BACKTESTS"
echo "----------------------------------------------------------------------"

if [ -f "logs/latest.json" ]; then
    echo "✓ Backtest results found (logs/latest.json)"
    # Show key metrics if jq is available
    if command -v jq &> /dev/null; then
        total_return=$(jq -r '.total_return // "N/A"' logs/latest.json)
        sharpe=$(jq -r '.sharpe_ratio // "N/A"' logs/latest.json)
        echo "  Total Return: $total_return%"
        echo "  Sharpe Ratio: $sharpe"
    fi
else
    echo "  No backtest results (run: python3 backtest.py)"
fi

echo ""
echo "⚙️  CONFIGURATION"
echo "----------------------------------------------------------------------"

if [ -f "config.py" ]; then
    echo "✓ config.py found"
else
    echo "✗ config.py NOT FOUND"
fi

if [ -f ".env" ]; then
    echo "✓ .env found"
    if grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
        echo "  ✓ OpenAI API key configured"
    fi
    if grep -q "ANTHROPIC_API_KEY" .env 2>/dev/null; then
        echo "  ✓ Anthropic API key configured"
    fi
else
    echo "  No .env file (optional for LLM features)"
fi

echo ""
echo "💾 DISK USAGE"
echo "----------------------------------------------------------------------"

if command -v du &> /dev/null; then
    data_size=$(du -sh data 2>/dev/null | cut -f1)
    cache_size=$(du -sh data_cache 2>/dev/null | cut -f1)
    models_size=$(du -sh models 2>/dev/null | cut -f1)
    logs_size=$(du -sh logs 2>/dev/null | cut -f1)

    echo "data/:       $data_size"
    echo "data_cache/: $cache_size"
    echo "models/:     $models_size"
    echo "logs/:       $logs_size"
fi

echo ""
echo "======================================================================"
echo "  QUICK COMMANDS"
echo "======================================================================"
echo ""
echo "Setup:   python3 update_spy_data.py"
echo "Train:   python3 train_multi_asset.py"
echo "Predict: python3 predict_multi_asset_ensemble.py"
echo "Test:    python3 backtest.py"
echo "Dashboard: streamlit run dashboard_comprehensive.py"
echo ""
echo "======================================================================"
