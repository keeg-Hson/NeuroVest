# NeuroVest Framework

Plug-and-play multi-asset ML trading system with 80+ configured assets.

## Quick Reference

### Download Data
```bash
# All assets
python download_all_assets.py

# Specific type
python download_all_assets.py --type equity
python download_all_assets.py --type crypto

# With API key (recommended)
export ALPHA_VANTAGE_API_KEY='your-key'
python download_all_assets.py
```

### Train Models
```bash
# Train everything
python train_unified.py --all

# Per-asset only
python train_unified.py --per-asset

# Macro models only
python train_unified.py --macro

# Single asset
python train_unified.py --asset SPY
```

### View Results
```bash
# CLI dashboard
python results_dashboard.py

# Top 10 performers
python results_dashboard.py --top 10

# Filter by type
python results_dashboard.py --type equity

# Export HTML
python results_dashboard.py --export html

# Best model for asset
python results_dashboard.py --asset SPY
```

### API Server
```bash
# Start server
python api_server.py

# Visit http://localhost:8000/docs for interactive docs
```

### Automated Refresh
```bash
# Run once
python auto_refresh.py

# Daily schedule (2 AM)
python auto_refresh.py --schedule daily

# Daemon mode
python auto_refresh.py --daemon
```

## Asset Configuration

Edit `../config/assets.yaml` to add/remove assets:

```yaml
equity_major_indices:
  SPY:
    name: "S&P 500"
    threshold: 0.005  # 0.5%
    enabled: true     # Change to false to disable
```

Currently configured:
- 33 Equity ETFs
- 10 Bond ETFs
- 6 Commodity ETFs
- 10 Cryptocurrencies
- 6 Macro groups

## Documentation

See `../FRAMEWORK_GUIDE.md` for complete documentation.

## Test Framework

```bash
cd ..
python test_framework.py
```

## Troubleshooting

**Yahoo Finance errors?**
Get free Alpha Vantage key: https://www.alphavantage.co/support/#api-key

**Missing dependencies?**
```bash
pip install -r ../requirements.txt
```

**Port 8000 in use?**
Change `api_port` in `../config/assets.yaml`
