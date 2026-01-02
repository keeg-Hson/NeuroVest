# Manual Setup Commands for Asset Download and Training

**IMPORTANT:** These commands must be run manually due to environment configuration requirements.

---

## Environment Issue Resolution

The automated downloads are encountering Python environment issues. Run these commands manually in your terminal:

### Step 1: Verify Python Environment

```bash
# Check if you're in the project directory
pwd
# Should show: /path/to/NeuroVest

# Check Python version
python3 --version
# Should be Python 3.8 or higher
```

### Step 2: Install Dependencies (if needed)

```bash
# Option A: System Python
pip3 install -r requirements.txt

# Option B: Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

---

## No-Code Improvements (30 minutes)

Run these commands in sequence to dramatically improve asset coverage:

### Download All Assets

```bash
# 1. Download SPY (S&P 500) - REQUIRED
python3 update_spy_data.py

# 2. Download Stock ETFs and Bonds (~35 assets)
python3 download_equity_etfs.py

# 3. Download Cryptocurrency Data (~10 assets)
python3 download_crypto_enhanced.py

# 4. Verify downloads
ls -lh data/*.csv
ls -lh data_cache/*.csv
```

Expected result: 40+ asset files downloaded

### Train Key Models

```bash
# 1. Train Multi-Asset Ensemble (SPY + Crypto)
python3 train_multi_asset.py --optimize-weights

# 2. Train QQQ (Nasdaq 100)
python3 train_per_asset.py --asset QQQ

# 3. Train GLD (Gold)
python3 train_per_asset.py --asset GLD

# 4. Train TLT (20+ Year Bonds)
python3 train_per_asset.py --asset TLT

# 5. Verify models
ls -lh models/*.pkl
```

Expected result: 10+ model files created

### Generate Predictions

```bash
# 1. Generate ensemble predictions
python3 predict_multi_asset_ensemble.py

# 2. Generate per-asset predictions for all trained assets
python3 predict_per_asset.py --all

# 3. Verify predictions
ls -lh logs/predictions/*.csv
```

Expected result: Prediction files for each asset

---

## Result Verification

After running the above commands, verify the system status:

```bash
# Check data coverage
python3 -c "
import os
from pathlib import Path

data_files = list(Path('data').glob('*.csv')) + list(Path('data_cache').glob('*.csv'))
print(f'✅ Data files: {len(data_files)}')

model_files = list(Path('models').glob('*.pkl'))
print(f'✅ Model files: {len(model_files)}')

pred_files = list(Path('logs/predictions').glob('*.csv')) if Path('logs/predictions').exists() else []
print(f'✅ Prediction files: {len(pred_files)}')
"
```

Expected output:
```
✅ Data files: 40+
✅ Model files: 10+
✅ Prediction files: 5+
```

---

## Alternative: Use Main Menu

If you prefer a guided approach:

```bash
python3 main.py
```

Then follow the menu:
1. Select **5** (Data Management)
2. Download assets:
   - **1** - Update SPY data
   - **2** - Download crypto (top 10)
   - **3** - Download all equity ETFs

3. Exit back to main menu

4. Select **1** (Training)
5. Choose:
   - **4** - Train with optimized weights

6. Exit back to main menu

7. Select **2** (Predictions)
8. Generate predictions

9. Select **7** - Launch Web Dashboard

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Install dependencies
```bash
pip3 install -r requirements.txt
```

### "Permission denied" errors

**Solution:** Check file permissions
```bash
chmod +x *.py
```

### "No data available for SPY"

**Solution:** Download SPY data
```bash
python3 update_spy_data.py
```

### Downloads fail with network errors

**Solution:** Try with longer timeout
```bash
# Edit download script timeout if needed
# Or run downloads one asset at a time
```

---

## Quick Start (Minimum Viable Setup)

If you just want to get the dashboard running with minimal setup:

```bash
# 1. Download SPY only (required)
python3 update_spy_data.py

# 2. Train multi-asset model
python3 train_multi_asset.py

# 3. Generate predictions
python3 predict_multi_asset_ensemble.py

# 4. Launch dashboard
streamlit run dashboard_comprehensive.py
```

This gives you:
- Working recession indicator
- Working valuation detector
- SPY forecasts
- Functional dashboard

Time required: ~15 minutes

---

## Full Setup (Complete Coverage)

For full system with all features:

```bash
# Download all data (10-15 minutes)
python3 update_spy_data.py
python3 download_equity_etfs.py
python3 download_crypto_enhanced.py

# Train all key models (30-45 minutes)
python3 train_multi_asset.py --optimize-weights
python3 train_per_asset.py --asset QQQ
python3 train_per_asset.py --asset GLD
python3 train_per_asset.py --asset BTC/USDT
python3 train_per_asset.py --asset ETH/USDT

# Generate all predictions (5 minutes)
python3 predict_multi_asset_ensemble.py
python3 predict_per_asset.py --all

# Launch dashboard
streamlit run dashboard_comprehensive.py
```

Time required: ~50-70 minutes total

---

## System Status Check

Run this to check what's ready:

```bash
python3 diagnose_system.py
```

Or quick check:

```bash
python3 -c "
from pathlib import Path

print('=== NeuroVest System Status ===')
print()
print('DATA:')
print(f'  SPY.csv: {'✅' if Path('data/SPY.csv').exists() else '❌'}')
print(f'  Data cache files: {len(list(Path('data_cache').glob('*.csv')))}')
print()
print('MODELS:')
print(f'  xgboost_multi_asset.pkl: {'✅' if Path('models/xgboost_multi_asset.pkl').exists() else '❌'}')
print(f'  lightgbm_multi_asset.pkl: {'✅' if Path('models/lightgbm_multi_asset.pkl').exists() else '❌'}')
print(f'  catboost_multi_asset.pkl: {'✅' if Path('models/catboost_multi_asset.pkl').exists() else '❌'}')
print()
print('PREDICTIONS:')
pred_file = Path('logs/labeled_predictions.csv')
if pred_file.exists():
    import pandas as pd
    df = pd.read_csv(pred_file)
    print(f'  labeled_predictions.csv: ✅ ({len(df):,} rows)')
else:
    print(f'  labeled_predictions.csv: ❌')
"
```

---

## Next Steps After Setup

Once data is downloaded and models are trained:

1. **Launch Dashboard:**
   ```bash
   streamlit run dashboard_comprehensive.py
   ```

2. **Run Backtest:**
   ```bash
   python3 backtest.py
   ```

3. **Generate LLM Analysis** (if API keys configured):
   ```bash
   python3 llm_forecast.py --asset SPY --provider openai
   ```

4. **Explore Features:**
   - Navigate through all 9 dashboard pages
   - Test recession indicator
   - Test valuation detector
   - Try custom data imports

---

## Summary

**Minimum Setup (15 min):**
- SPY data
- Multi-asset model
- Basic predictions
- Dashboard functional

**Recommended Setup (60 min):**
- All stock/ETF data
- All crypto data
- Key models (QQQ, GLD, BTC)
- All predictions
- Full dashboard features

**Complete Setup (90 min):**
- All configured assets
- All per-asset models
- LLM integration
- Portfolio analysis
- Production ready

Choose based on your time and requirements.
