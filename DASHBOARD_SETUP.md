# Dashboard Setup and Usage Guide

**Last Updated:** December 24, 2024

---

## Dashboard Options

NeuroVest provides two dashboard interfaces:

### 1. **dashboard_comprehensive.py** (RECOMMENDED - Primary Dashboard)

**Purpose:** Full-featured production dashboard showcasing ALL NeuroVest capabilities

**Features:**
- 9 comprehensive pages
- Real recession indicator integration
- Real valuation detector integration
- LLM analysis interface
- Portfolio rebalancing optimizer
- Asset download manager
- REST API documentation
- Custom data imports
- Production-ready homepage

**Best For:**
- Production deployment
- Client-facing demonstrations
- Full feature showcase
- API integration testing

**Launch:**
```bash
streamlit run dashboard_comprehensive.py
# Access at: http://localhost:8501
```

### 2. **dashboard.py** (Basic/Minimal Dashboard)

**Purpose:** Lightweight dashboard focused on data display and basic analysis

**Features:**
- 6 simple pages
- Asset analysis
- Basic forecasts view
- Data import
- Minimal UI

**Best For:**
- Quick data review
- Development testing
- Lightweight deployments
- Simple forecast viewing

**Launch:**
```bash
streamlit run dashboard.py
# Access at: http://localhost:8501
```

---

## When to Use Which Dashboard

| Scenario | Recommended Dashboard |
|----------|---------------------|
| Production deployment | `dashboard_comprehensive.py` |
| Client demonstrations | `dashboard_comprehensive.py` |
| Full feature testing | `dashboard_comprehensive.py` |
| API integration | `dashboard_comprehensive.py` |
| Quick data checks | `dashboard.py` |
| Development testing | `dashboard.py` |
| Minimal resource usage | `dashboard.py` |

---

## Setup Instructions

### First-Time Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download Initial Data:**
```bash
# SPY (required for most features)
python3 update_spy_data.py

# Optional: Download additional assets
python3 download_equity_etfs.py      # ETFs, bonds
python3 download_crypto_enhanced.py   # Crypto
```

3. **Train Models (Required):**
```bash
# Train multi-asset ensemble
python3 train_multi_asset.py --optimize-weights

# Optional: Train additional assets
python3 train_per_asset.py --asset QQQ
python3 train_per_asset.py --asset GLD
```

4. **Generate Predictions:**
```bash
python3 predict_multi_asset_ensemble.py
```

5. **Launch Dashboard:**
```bash
streamlit run dashboard_comprehensive.py
```

---

## Feature Requirements

Some dashboard features require specific data/setup:

### Recession Indicator
**Requires:**
- SPY data (200+ days)
- TNX data (optional, for yield curve)

**Setup:**
```bash
python3 update_spy_data.py
```

### Valuation Detector
**Requires:**
- Asset data for analysis
- Minimum 200 days of history

**Setup:**
```bash
# For SPY
python3 update_spy_data.py

# For crypto
python3 download_crypto_enhanced.py

# For metals
python3 framework/download_all_assets.py --asset GLD
```

### LLM Analysis
**Requires:**
- API keys in `.env` file
- Prediction data

**Setup:**
```bash
# Create .env file
cp .env.example .env

# Add your API keys
nano .env
# Add: OPENAI_API_KEY=sk-...
# Or: ANTHROPIC_API_KEY=sk-ant-...
# Optional: NEWS_API_KEY=...

# Generate predictions first
python3 predict_multi_asset_ensemble.py
```

### Portfolio Rebalancing
**Requires:**
- Multiple assets downloaded
- Historical data (1+ year recommended)

**Setup:**
```bash
# Download assets
python3 download_equity_etfs.py

# Assets needed: SPY, GLD, TLT (minimum)
```

---

## Deployment Options

### Local Development
```bash
streamlit run dashboard_comprehensive.py
```

### Production (Streamlit Cloud)
1. Push code to GitHub
2. Connect to https://streamlit.io/cloud
3. Deploy from repository
4. Set environment variables in dashboard settings

### Production (Custom Server)
```bash
# With custom port
streamlit run dashboard_comprehensive.py --server.port 8080

# Headless mode (no browser auto-open)
streamlit run dashboard_comprehensive.py --server.headless true

# External access
streamlit run dashboard_comprehensive.py --server.address 0.0.0.0
```

### Docker Deployment
```bash
# Build image
docker build -t neurovest-dashboard .

# Run container
docker run -p 8501:8501 neurovest-dashboard
```

---

## Troubleshooting

### "No assets found"
**Solution:** Download data first
```bash
python3 update_spy_data.py
python3 download_equity_etfs.py
```

### "No predictions available"
**Solution:** Train models and generate predictions
```bash
python3 train_multi_asset.py
python3 predict_multi_asset_ensemble.py
```

### "Models not trained"
**Solution:** Run training
```bash
python3 train_multi_asset.py --optimize-weights
```

### Recession/Valuation features not working
**Solution:** Ensure SPY data is downloaded
```bash
python3 update_spy_data.py
# Refresh dashboard in browser
```

### LLM Analysis shows errors
**Solution:** Check API keys in `.env`
```bash
# Verify .env file exists and has keys
cat .env | grep API_KEY
```

---

## Performance Optimization

### For Large Datasets
- Enable caching in dashboard code (already implemented)
- Limit date ranges for chart display
- Use `@st.cache_data` decorators (already in place)

### For Multiple Users
- Deploy on server with adequate resources
- Use Streamlit Cloud for automatic scaling
- Consider load balancer for high traffic

### Memory Usage
- Dashboard comprehensive: ~500MB RAM
- Dashboard basic: ~200MB RAM
- Add asset data as needed (don't load all at once)

---

## Security Considerations

### API Keys
- Never commit `.env` file to git (already in `.gitignore`)
- Use environment variables in production
- Rotate keys periodically

### Data Privacy
- Dashboard runs locally by default
- No data sent to external services (except LLM APIs if configured)
- Prediction data stored locally in `logs/`

### Access Control
- For production, add authentication layer
- Use Streamlit Cloud built-in auth
- Or implement custom auth with st.secrets

---

## Support

**Issues:**
- Check logs in terminal where Streamlit is running
- Refresh browser page
- Clear Streamlit cache (sidebar button)
- Restart Streamlit server

**Documentation:**
- README.md - Main documentation
- DEPLOYMENT.md - Deployment guide
- FRAMEWORK_GUIDE.md - Framework documentation
- This file (DASHBOARD_SETUP.md) - Dashboard setup

**Getting Help:**
- Open GitHub issue
- Check documentation files
- Run diagnostic: `python3 diagnose_system.py`

---

## Quick Reference Commands

```bash
# Launch primary dashboard
streamlit run dashboard_comprehensive.py

# Launch minimal dashboard
streamlit run dashboard.py

# Download all data
python3 update_spy_data.py && python3 download_equity_etfs.py && python3 download_crypto_enhanced.py

# Train models
python3 train_multi_asset.py --optimize-weights

# Generate predictions
python3 predict_multi_asset_ensemble.py

# Run backtest
python3 backtest.py

# Full pipeline (automated)
python3 main.py
# Select: R (Run Full Pipeline)
```

---

## Summary

**Primary Dashboard:** `dashboard_comprehensive.py`
- Full features
- Production-ready
- Recommended for all uses

**Basic Dashboard:** `dashboard.py`
- Simple interface
- Quick data viewing
- Development testing

Both dashboards share the same backend data and models. Choose based on your needs.
