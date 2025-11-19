# NeuroVest Quick Start Guide

**Get started with economic modeling in 5 minutes**

---

## Prerequisites

```bash
# Python 3.8 or higher
python --version

# Git (to clone repository)
git --version
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pandas`, `numpy` (data handling)
- `xgboost`, `lightgbm`, `catboost` (ensemble models)
- `scikit-learn` (ML utilities)
- `yfinance` (stock data)
- `ccxt` (crypto data)
- `ta-lib` (technical indicators)

**Note**: Installation typically takes 2-3 minutes.

---

## Basic Usage

### 1. Train Models (Required First Time)

Models are not included in the repository. Train them first:

```bash
# Download data
python download_spy_data.py
python download_multi_asset_data.py

# Train ensemble (10-15 minutes)
python train_multi_asset.py
```

**Note**: Models are in `.gitignore` and must be trained locally before predictions.

### 2. Generate Economic Predictions

After training, generate predictions:

```bash
python predict_multi_asset_ensemble.py
```

**Output**: `logs/daily_predictions.csv`

```csv
Date,Prediction,Probability,Asset
2025-11-15,1,0.523,SPY
2025-11-14,0,0.312,SPY
2025-11-13,1,0.478,SPY
```

**Interpretation**:
- `Prediction=1`: Model forecasts positive economic opportunity
- `Probability`: Model confidence (0.45 threshold)
- Higher probability = stronger signal

---

### 3. Validate Model Performance (Backtest)

Test the economic predictions against historical data:

```bash
python backtest.py
```

**Output** (example):
```
📈 Backtest Report
  Period:             2000-2025 (6,501 days)
  Signals:            454 opportunities (7.0%)
  Total Return:       636%
  Sharpe Ratio:       1.44
  Max Drawdown:       -16.99%
  Win Rate:           52.3%
  Avg Return/Trade:   0.187%

✅ Backtest complete. Results saved to logs/
```

**Files created**:
- `logs/backtest_trades.csv` - Trade log
- `logs/backtest_metrics.csv` - Performance metrics
- `logs/backtest_equity_curve.csv` - Portfolio value over time

---

### 4. Retrain Models (Optional)

To update models with fresh data (after initial training):

```bash
# Step 1: Download latest data
python download_spy_data.py
python download_multi_asset_data.py

# Step 2: Train ensemble
python train_multi_asset.py
```

**Training time**: ~10-15 minutes (depends on hardware)

**Output**:
```
================================================================================
MULTI-ASSET MODEL RESULTS
================================================================================
   Model  Accuracy  Precision   Recall       F1  Features
 Xgboost  0.606030   0.575816 0.352941 0.437637       137
Lightgbm  0.604497   0.568345 0.371765 0.449502       137
Catboost  0.593766   0.565632 0.278824 0.373522       137
Ensemble  0.595810   0.559838 0.324706 0.411020       137

✅ Models saved to models/ directory
```

---

## Understanding the Outputs

### Probability Scores

| Range | Interpretation | Action |
|-------|----------------|--------|
| < 0.30 | No economic signal | Monitor only |
| 0.30 - 0.40 | Weak signal | Watch regime changes |
| 0.40 - 0.50 | Moderate signal | Near threshold |
| **> 0.50** | **High conviction** | **Positive forecast** |
| > 0.60 | Very high conviction | Strong regime signal |

### Model Agreement (Ensemble Analysis)

Check `logs/ensemble_analysis.csv` for consensus across models:

```csv
Date,XGB_Prob,LGB_Prob,CatBoost_Prob,Ensemble_Prob,Agreement
2025-11-15,0.54,0.51,0.52,0.523,True
```

**Agreement=True**: All 3 models agree (current average: 89.8%)

---

## Automated Daily Pipeline

Run the full pipeline automatically (data update + predictions):

```bash
python run_daily_pipeline.py
```

**This executes**:
1. Download SPY data (Yahoo Finance)
2. Download crypto data (CCXT)
3. Compute 126 features
4. Generate ensemble predictions
5. Save to `logs/daily_predictions.csv`

**Recommended schedule**: Daily at 16:30 ET (after market close)

---

## Configuration

### Change Prediction Threshold

Edit `config.py`:

```python
PREDICT_CFG = {
    "p_min": 0.45,  # Change this (0.35 = more signals, 0.55 = fewer)
}
```

| Threshold | Signal Rate | Precision | Recall | Use Case |
|-----------|-------------|-----------|--------|----------|
| 0.35 | ~12% | ~78% | ~31% | Aggressive |
| **0.45** | **~7%** | **~92%** | **~20%** | **Balanced (default)** |
| 0.55 | ~3% | ~99% | ~7% | Ultra-conservative |

### Modify Training Horizon

Edit `config.py`:

```python
TRAIN_CFG = {
    "horizon": 1,  # Forward return horizon (days)
    # 1 = next-day prediction
    # 3 = 3-day forward prediction
    # 5 = weekly prediction
}
```

---

## Common Tasks

### Task 1: Check Latest Prediction

```bash
tail -1 logs/daily_predictions.csv
```

### Task 2: View Model Feature Importance

```bash
# View top features from recent analysis
cat ACCURACY_IMPROVEMENT_ANALYSIS.md | grep -A 15 "Top 20 Features"
```

### Task 3: Analyze Backtest Performance

```bash
python -c "
import pandas as pd
trades = pd.read_csv('logs/backtest_trades.csv')
print(f'Total trades: {len(trades)}')
print(f'Winners: {(trades.ret > 0).sum()} ({(trades.ret > 0).mean():.1%})')
print(f'Avg return: {trades.ret.mean():.3%}')
print(f'Best trade: {trades.ret.max():.3%}')
print(f'Worst trade: {trades.ret.min():.3%}')
"
```

### Task 4: Export Predictions for Analysis

```bash
# Export to Excel
python -c "
import pandas as pd
df = pd.read_csv('logs/daily_predictions.csv')
df.to_excel('predictions_export.xlsx', index=False)
print('✅ Exported to predictions_export.xlsx')
"
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'xgboost'"

**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: models/xgboost_multi_asset.pkl"

**Solution**: Models not trained yet. Run training:
```bash
python train_multi_asset.py
```

### Error: "No data available for SPY"

**Solution**: Download data first:
```bash
python download_spy_data.py
```

### Error: "AttributeError: _ARRAY_API not found" or NumPy/pyarrow compatibility

**Cause**: Outdated pyarrow version incompatible with your NumPy version

**Solution**: Upgrade dependencies (works with both NumPy 1.x and 2.x)
```bash
pip install -r requirements.txt --upgrade
```

This upgrades pyarrow to 17.0+, which supports both NumPy 1.x and 2.x

### Predictions are all 0 (no signals)

**Possible causes**:
1. Threshold too high → Lower `PREDICT_CFG["p_min"]` in `config.py`
2. Market regime change → Retrain models with fresh data
3. Data issue → Check `data/SPY.csv` has recent data

---

## Next Steps

### For Research Use

1. **Explore Feature Importance**
   ```bash
   # View top features driving predictions
   cat ACCURACY_IMPROVEMENT_ANALYSIS.md | grep -A 20 "Feature Importance"
   ```

2. **Analyze Prediction History**
   ```bash
   # Compare predictions vs actual outcomes
   python -c "
   import pandas as pd
   labeled = pd.read_csv('logs/labeled_predictions.csv')
   print('Accuracy:', (labeled.Prediction == labeled.Label).mean())
   print('Precision:', labeled[labeled.Prediction==1].Label.mean())
   "
   ```

3. **Test Different Configurations**
   - Modify `config.py` parameters
   - Re-run `train_multi_asset.py`
   - Compare backtest results

### For Development

1. **Read Technical Documentation**
   - `IMPLEMENTATION_SUMMARY.md` - Recent improvements
   - `ACCURACY_IMPROVEMENT_ANALYSIS.md` - Feature analysis
   - `README.md` - Full system documentation

2. **Understand the Code**
   - `utils.py` - Feature engineering (126 features)
   - `train_multi_asset.py` - Model training logic
   - `predict_multi_asset_ensemble.py` - Prediction pipeline

3. **Contribute**
   - Fork repository
   - Create feature branch
   - Submit pull request

---

## File Locations

| File | Purpose |
|------|---------|
| `config.py` | Global configuration (thresholds, parameters) |
| `data/SPY.csv` | Stock price data |
| `data/crypto/*.csv` | Cryptocurrency price data |
| `models/*.pkl` | Trained ensemble models |
| `logs/daily_predictions.csv` | Latest predictions |
| `logs/backtest_*.csv` | Backtest results |

---

## Key Concepts

**This is an economic modeling system**, not a trading bot:
- Focus: Regime analysis and economic forecasting
- Horizon: 3-day forward predictions
- Features: 126 indicators across 5 economic domains
- Validation: Multi-year backtesting framework

**The backtest is a validation tool**, not the primary purpose. It tests whether economic signals correspond to profitable market movements.

---

## Support

**Questions?**
- Check `README.md` for full documentation
- Review `IMPLEMENTATION_SUMMARY.md` for recent changes
- Open GitHub issue for bugs or feature requests

**Remember**: This is educational/research software. Do not use with real money without extensive validation.

---

## Summary Commands

```bash
# 1. Clone and install
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest
pip install -r requirements.txt

# 2. Train models (required first time - models not in repo)
python download_spy_data.py
python download_multi_asset_data.py
python train_multi_asset.py

# 3. Generate predictions
python predict_multi_asset_ensemble.py

# 4. Validate with backtest
python backtest.py

# 5. Automated daily pipeline
python run_daily_pipeline.py
```

**You're ready to start exploring economic modeling!**
