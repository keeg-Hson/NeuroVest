# NeuroVest System Design

> **Single Source of Truth Document**
> Last Updated: 2026-03-01

This document defines the canonical architecture, data sources, and entry points for NeuroVest. All development should align with these specifications.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Railway PostgreSQL                         │
│              (PRIMARY DATA STORE)                            │
│         DATABASE_URL environment variable                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐    ┌─────────▼─────────┐
    │   DataManager     │    │   SQLite Fallback │
    │   (PostgreSQL)    │    │   (Local Dev Only)│
    └─────────┬─────────┘    └───────────────────┘
              │
    ┌─────────▼─────────────────────────────────┐
    │         Unified Data Pipeline              │
    │         core/data_pipeline.py              │
    └─────────┬─────────────────────────────────┘
              │
    ┌─────────▼─────────────────────────────────┐
    │         Feature Engineering                │
    │         build_feature_table.py             │
    └─────────┬─────────────────────────────────┘
              │
    ┌─────────┴─────────────────────────────────┐
    │                                           │
    ▼                                           ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────────────┐
│ train.py  │  │predict.py │  │backtest.py│  │ dashboard_     │
│           │  │           │  │           │  │ comprehensive  │
└───────────┘  └───────────┘  └───────────┘  └────────────────┘
```

---

## Primary Data Store

| Environment | Database | Configuration |
|-------------|----------|---------------|
| **Production** | PostgreSQL on Railway | `DATABASE_URL` env var |
| **Development** | SQLite fallback | `data/market_data.db` |

**Important**: Railway PostgreSQL is the primary source of truth. Local SQLite is for development/testing only and will show empty data if not synchronized.

---

## Canonical Files

### Configuration
| Component | Canonical File | Purpose |
|-----------|---------------|---------|
| Global Config | `config.py` | Paths, thresholds, training params |
| Asset Definitions | `config/assets.yaml` | 59 assets (equities, crypto, bonds, commodities) |
| Trading Profiles | `configs/*.json` | Backtest configurations per strategy |

### Core Modules
| Component | Canonical File | Purpose |
|-----------|---------------|---------|
| Data Management | `core/data_manager_postgres.py` | Dual-mode PostgreSQL/SQLite |
| Data Pipeline | `core/data_pipeline.py` | Unified training data pipeline |
| Prediction Engine | `core/prediction_engine.py` | Unified prediction system |
| Model Architectures | `core/models/base_models.py` | XGBoost, LightGBM, CatBoost |
| Feature Engineering | `build_feature_table.py` | Technical, macro, sentiment features |

### Entry Points
| Function | Canonical Script | Usage |
|----------|-----------------|-------|
| Training | `train.py` | `python train.py [--asset SPY]` |
| Prediction | `predict.py` | `python predict.py [--asset SPY]` |
| Backtesting | `backtest.py` | `python backtest.py [--asset SPY]` |
| Dashboard | `dashboard_comprehensive.py` | `streamlit run dashboard_comprehensive.py` |
| CLI | `neurovest_cli.py` | `python neurovest_cli.py` |
| Menu | `main.py` | `python main.py` |

---

## Model Configuration

### Training Parameters (Locked Feb 2026)
```python
TRAIN_CFG = {
    "horizon": 1,              # 1-day forward returns
    "pos_threshold": 0.005,    # 0.5% binary threshold
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
}
```

### Prediction Threshold
```python
PREDICTION_THRESHOLD = 0.45   # Precision-focused (~42% precision target)
```

### Risk Management
```python
RISK_CFG = {
    "max_position_pct": 0.05,      # 5% max per trade
    "risk_per_trade_pct": 0.01,    # 1% portfolio risk
    "kelly_fraction": 0.25,        # 25% Kelly scaling
    "max_daily_drawdown_pct": 0.03 # 3% max daily loss
}
```

---

## Asset Coverage

| Category | Count | Examples |
|----------|-------|----------|
| Equity ETFs | 33 | SPY, QQQ, IWM, DIA, XLF, XLK |
| Cryptocurrencies | 10 | BTC, ETH, SOL, BNB, XRP, ADA |
| Bond ETFs | 10 | TLT, IEF, LQD, HYG, EMB |
| Commodity ETFs | 6 | GLD, SLV, USO, UNG, GDX |

---

## Deployment

### Railway Services
1. **API Server**: FastAPI on port 8000 (`api_server.py`)
2. **Dashboard**: Streamlit (`dashboard_comprehensive.py`)
3. **Background Worker**: Data updates + scheduled training
4. **Database**: PostgreSQL (shared)

### Environment Variables Required
```bash
DATABASE_URL=postgresql://...    # Railway PostgreSQL connection
NEWSAPI_KEY=...                  # Optional: News sentiment
REDDIT_CLIENT_ID=...             # Optional: Reddit sentiment
REDDIT_CLIENT_SECRET=...
```

---

## Directory Structure

```
NeuroVest/
├── config.py                    # Global configuration
├── train.py                     # Training entry point
├── predict.py                   # Prediction entry point
├── backtest.py                  # Backtesting entry point
├── main.py                      # Interactive menu
├── neurovest_cli.py             # CLI interface
├── build_feature_table.py       # Feature engineering
├── dashboard_comprehensive.py   # Streamlit dashboard
│
├── core/                        # Core modules
│   ├── data_manager_postgres.py # Data management (canonical)
│   ├── data_pipeline.py         # Training pipeline
│   ├── prediction_engine.py     # Prediction system
│   └── models/
│       └── base_models.py       # Model architectures
│
├── config/
│   └── assets.yaml              # Asset definitions
│
├── configs/                     # Trading profiles
│   ├── backtest_optimized.json
│   ├── backtest_conservative.json
│   └── backtest_aggressive.json
│
├── data/                        # Local data (dev only)
├── data_cache/                  # Asset cache
├── models/                      # Trained models
├── logs/                        # Predictions & metrics
│
└── archive/                     # Deprecated code (DO NOT USE)
    ├── legacy_scripts/
    ├── train_scripts/
    ├── predict_scripts/
    └── download_scripts/
```

---

## Anti-Patterns (Avoid)

1. **Direct SQLite access** - Always use `core/data_manager_postgres.py`
2. **Hardcoded model paths** - Use `config.py` constants
3. **Legacy model references** - `market_crash_model_fwd_improved.pkl` does not exist
4. **Archived scripts** - Everything in `archive/` is deprecated
5. **Local diagnostics without DATABASE_URL** - Will show empty data

---

## Quick Reference

```bash
# Check database connection
export DATABASE_URL="postgresql://..."
python -c "from core.data_manager_postgres import DataManager; DataManager()"

# Training
python train.py --asset SPY

# Prediction
python predict.py --asset SPY

# Backtest
python backtest.py --asset SPY

# Dashboard
streamlit run dashboard_comprehensive.py

# Full pipeline
python main.py
```

---

*Document generated from CONSOLIDATION_PLAN.md and codebase analysis*
