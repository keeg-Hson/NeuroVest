# NeuroVest Unified Framework Guide

## 🎯 Overview

The NeuroVest framework is a **plug-and-play system** for multi-asset machine learning trading predictions. It supports:

- **80+ assets** across equities, bonds, commodities, and cryptocurrencies
- **Per-asset models**: Separate models for each asset (SPY, BTC, QQQ, etc.)
- **Macro models**: Combined models for asset groups (all equities, all crypto, etc.)
- **REST API**: Easy access to predictions and results
- **Automated refresh**: Scheduled data updates and retraining
- **Configuration-driven**: Add/remove assets by editing one YAML file

---

## 🚀 Quick Start (5 Minutes)

### 1. Configure Assets

Edit `config/assets.yaml` to enable/disable assets:

```yaml
equity_major_indices:
  SPY:
    name: "S&P 500"
    threshold: 0.005
    enabled: true  # Change to false to disable
  QQQ:
    enabled: true
```

Currently configured: **80+ assets** (equities, bonds, commodities, crypto)

### 2. Download Data

```bash
# Download all enabled assets
python framework/download_all_assets.py

# Or download specific type
python framework/download_all_assets.py --type equity
python framework/download_all_assets.py --type crypto

# With Alpha Vantage API key (recommended)
export ALPHA_VANTAGE_API_KEY='your-key'
python framework/download_all_assets.py
```

### 3. Train Models

```bash
# Train everything (per-asset + macro models)
python framework/train_unified.py --all

# Or train specific types
python framework/train_unified.py --per-asset   # Per-asset models only
python framework/train_unified.py --macro       # Macro models only
python framework/train_unified.py --asset SPY   # Single asset
```

### 4. View Results

```bash
# Interactive dashboard
python framework/results_dashboard.py

# Filter by type
python framework/results_dashboard.py --type equity
python framework/results_dashboard.py --top 10

# Export to HTML
python framework/results_dashboard.py --export html
# Open: results/dashboard.html
```

### 5. Start API Server

```bash
python framework/api_server.py

# API runs on http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## 📋 Framework Components

### 1. Asset Manager (`framework/asset_manager.py`)

Central hub for asset configuration.

```python
from framework.asset_manager import AssetManager

manager = AssetManager()

# Get all assets
assets = manager.get_all_assets()

# Get specific types
equities = manager.get_assets_by_type('equity')
cryptos = manager.get_assets_by_type('crypto')

# Get macro groups
all_eq = manager.get_macro_group('all_equities')

# Get asset metadata
spy = manager.get_asset('SPY')
print(spy.threshold)  # 0.005 (0.5%)
```

### 2. Unified Downloader (`framework/download_all_assets.py`)

Automatically downloads all enabled assets.

**Features:**
- Multi-source fallback (pandas_datareader → Alpha Vantage → manual)
- Supports ETFs (Yahoo Finance) and crypto (CCXT)
- Skip existing files
- Progress reporting

```bash
# Download all
python framework/download_all_assets.py

# Force re-download
python framework/download_all_assets.py --force

# With API key
python framework/download_all_assets.py --api-key YOUR_KEY
```

### 3. Unified Trainer (`framework/train_unified.py`)

Trains both per-asset and macro models.

**Per-Asset Models:**
- Separate model for each asset (SPY, QQQ, BTC, etc.)
- Uses asset-specific threshold
- Best for assets with unique patterns

**Macro Models:**
- Combined model for asset groups
- More training data
- Better for assets with similar behavior

```bash
# Train all models
python framework/train_unified.py --all

# Results saved to:
# - results/per_asset_results_[timestamp].csv
# - results/macro_results_[timestamp].csv
# - results/training_summary_[timestamp].json
```

### 4. API Server (`framework/api_server.py`)

RESTful API for predictions and results.

**Endpoints:**
```
GET  /health                    - Health check
GET  /assets                    - List all assets
GET  /models                    - List trained models
GET  /predict/{asset}           - Get prediction for asset
GET  /results/per-asset         - Per-asset results
GET  /results/macro              - Macro results
GET  /results/summary            - Training summary
```

**Example Usage:**
```bash
# List all assets
curl http://localhost:8000/assets

# Get SPY prediction
curl http://localhost:8000/predict/SPY

# Get crypto prediction using macro model
curl "http://localhost:8000/predict/BTC/USDT?model_type=macro"

# Get all results
curl http://localhost:8000/results/per-asset
```

**Interactive Docs:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 5. Results Dashboard (`framework/results_dashboard.py`)

Aggregates and displays all results.

**Features:**
- Sortable tables
- Performance comparisons
- Best/worst performers
- Per-asset vs macro recommendations
- HTML export

```bash
# Show all results
python framework/results_dashboard.py

# Top 10 performers
python framework/results_dashboard.py --top 10

# Filter by type
python framework/results_dashboard.py --type equity

# Export to HTML
python framework/results_dashboard.py --export html

# Get best model for asset
python framework/results_dashboard.py --asset SPY
```

### 6. Auto Refresh (`framework/auto_refresh.py`)

Automated data refresh and retraining.

**Features:**
- Downloads latest data
- Retrains models
- Updates dashboard
- Runs on schedule

```bash
# Run once
python framework/auto_refresh.py

# Run as daemon (uses schedule from config)
python framework/auto_refresh.py --daemon

# Run on schedule
python framework/auto_refresh.py --schedule daily   # Daily at 2 AM
python framework/auto_refresh.py --schedule weekly  # Sunday 2 AM
python framework/auto_refresh.py --schedule hourly  # Every hour
```

**Cron Example:**
```cron
# Run daily at 2 AM
0 2 * * * cd /path/to/NeuroVest && python framework/auto_refresh.py
```

---

## ⚙️ Configuration

### Asset Configuration (`config/assets.yaml`)

**Structure:**
```yaml
# Asset groups
equity_major_indices:
  SPY:
    name: "S&P 500"
    category: "Large Cap Blend"
    threshold: 0.005  # 0.5% prediction threshold
    enabled: true

# Macro groups (combine assets)
macro_groups:
  all_equities:
    name: "All Equity ETFs Combined"
    includes: ["equity_major_indices", "equity_sectors"]
    enabled: true

# Global settings
settings:
  start_date: "2000-01-01"
  refresh_schedule: "daily"
  train_test_split: 0.8
  api_enabled: true
  api_port: 8000
```

**Asset Types:**
- `equity_major_indices`: SPY, QQQ, IWM, DIA, VTI
- `equity_international`: EFA, EEM, VEA, VWO, IEFA
- `equity_sectors`: XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC
- `equity_style`: VUG, VTV, VO, VB, IJH, IJR
- `equity_thematic`: ARKK, SOXX, SMH, XBI, TAN, ICLN
- `bonds`: AGG, BND, TLT, IEF, SHY, LQD, HYG, JNK, MUB, EMB
- `commodities`: GLD, SLV, USO, UNG, DBC, GDX
- `crypto`: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, XRP/USDT, etc.

**Adding New Assets:**
```yaml
equity_major_indices:
  VOO:  # Add Vanguard S&P 500
    name: "Vanguard S&P 500"
    category: "Large Cap Blend"
    threshold: 0.005
    enabled: true
```

---

## 📊 Model Performance Comparison

### Per-Asset vs Macro Models

**Per-Asset Models:**
- ✅ Best for unique assets (ARKK, commodities, individual stocks)
- ✅ Captures asset-specific patterns
- ❌ Requires sufficient data (1000+ samples)
- ❌ More models to maintain

**Macro Models:**
- ✅ Better for similar assets (sector ETFs, large-cap equities)
- ✅ More training data (7-8x increase)
- ✅ Fewer models to maintain
- ❌ May dilute signal for unique assets

**Recommendations:**
- **Use Per-Asset**: SPY, ARKK, BTC, individual stocks
- **Use Macro**: Sector ETFs (XLF, XLK), emerging markets
- **Dashboard shows best choice** for each asset

---

## 🔄 Typical Workflows

### Workflow 1: Initial Setup
```bash
# 1. Configure assets
vim config/assets.yaml

# 2. Download data
python framework/download_all_assets.py

# 3. Train models
python framework/train_unified.py --all

# 4. View results
python framework/results_dashboard.py --export html

# 5. Start API
python framework/api_server.py
```

### Workflow 2: Daily Updates
```bash
# Run automated refresh (downloads + retrains + updates dashboard)
python framework/auto_refresh.py
```

### Workflow 3: Add New Asset
```bash
# 1. Add to config/assets.yaml
vim config/assets.yaml

# 2. Download just that asset
python framework/download_all_assets.py --force

# 3. Train just that asset
python framework/train_unified.py --asset YOUR_ASSET

# 4. View updated results
python framework/results_dashboard.py
```

### Workflow 4: Production Deployment
```bash
# 1. Set up automated refresh
python framework/auto_refresh.py --daemon &

# 2. Start API server
python framework/api_server.py &

# 3. Access predictions via API
curl http://localhost:8000/predict/SPY
```

---

## 📁 Directory Structure

```
NeuroVest/
├── config/
│   └── assets.yaml              # Asset configuration
├── framework/
│   ├── asset_manager.py         # Asset management
│   ├── download_all_assets.py   # Data downloader
│   ├── train_unified.py         # Model trainer
│   ├── api_server.py            # REST API
│   ├── results_dashboard.py     # Results viewer
│   └── auto_refresh.py          # Automated refresh
├── data_cache/                   # Downloaded data
│   ├── SPY_1d.csv
│   ├── BTC_USDT_1d.csv
│   └── ...
├── models/                       # Trained models
│   ├── spy_xgboost.pkl          # Per-asset models
│   ├── macro_all_equities_xgboost.pkl  # Macro models
│   └── ...
└── results/                      # Training results
    ├── per_asset_results_[timestamp].csv
    ├── macro_results_[timestamp].csv
    ├── training_summary_[timestamp].json
    └── dashboard.html
```

---

## 🔧 Troubleshooting

### Data Download Issues

**Problem:** Yahoo Finance 401/429 errors
**Solution:**
1. Get free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
2. Set environment variable: `export ALPHA_VANTAGE_API_KEY='your-key'`
3. Re-run download

**Problem:** CCXT crypto errors
**Solution:**
```bash
pip install --upgrade ccxt
```

### Training Issues

**Problem:** Insufficient data
**Solution:** Assets need 1000+ samples. Disable or use macro model.

**Problem:** Out of memory
**Solution:** Train in batches:
```bash
python framework/train_unified.py --per-asset  # First batch
python framework/train_unified.py --macro      # Second batch
```

### API Issues

**Problem:** Port 8000 already in use
**Solution:** Change port in `config/assets.yaml`:
```yaml
settings:
  api_port: 8080
```

---

## 🎯 Best Practices

1. **Start Small**: Enable 5-10 assets first, then expand
2. **Use Alpha Vantage**: Free tier handles all equity/bond downloads
3. **Check Results**: Use dashboard to verify models before trading
4. **Automate Refresh**: Set up `auto_refresh.py` as cron job
5. **Monitor API**: Use `/health` endpoint for uptime monitoring
6. **Version Results**: Results are timestamped - compare over time
7. **Test First**: Use `--asset SPY` to test before training all

---

## 📞 Support

- **Framework Issues**: Check this guide
- **Model Performance**: See `results_dashboard.py`
- **API Docs**: http://localhost:8000/docs
- **Asset Config**: See `config/assets.yaml` comments

---

## 🚀 Next Steps

1. ✅ Configure your assets
2. ✅ Download data
3. ✅ Train models
4. ✅ View results dashboard
5. ✅ Start API server
6. 🔄 Set up automated refresh
7. 📈 Start trading!
