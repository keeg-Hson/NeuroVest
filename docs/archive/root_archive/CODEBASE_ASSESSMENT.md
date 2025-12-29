# NeuroVest Codebase Assessment
**Date:** 2024-12-23
**Purpose:** Complete inventory of actual capabilities (no hallucinations)

---

## 📊 AVAILABLE ASSETS

### Stocks/ETFs (data/ directory)
Currently Downloaded:
- **SPY** - S&P 500 ETF (primary)
- **DXY** - US Dollar Index
- **HYG** - High Yield Corporate Bonds
- **LQD** - Investment Grade Corporate Bonds
- **TNX** - 10-Year Treasury Note Yield
- **UUP** - US Dollar Bullish Fund

Supported (via download_equity_etfs.py):
- **QQQ** - Nasdaq 100 (Tech-heavy)
- **IWM** - Russell 2000 (Small caps)
- **DIA** - Dow Jones (Blue chips)
- **VTI** - Total Stock Market
- **EEM** - Emerging Markets
- **XLF** - Financials Sector
- **XLK** - Technology Sector
- **XLE** - Energy Sector

### Crypto Assets (via download_crypto_enhanced.py)
- **BTC/USDT** - Bitcoin
- **ETH/USDT** - Ethereum
- **SOL/USDT** - Solana
- **BNB/USDT** - Binance Coin
- **XRP/USDT** - Ripple
- **ADA/USDT** - Cardano
- **DOGE/USDT** - Dogecoin
- **AVAX/USDT** - Avalanche
- **MATIC/USDT** - Polygon
- **LINK/USDT** - Chainlink

**Total Supported Assets:** 24

---

## 🎯 ACTUAL FEATURES (Verified from codebase)

### 1. **Market Forecasting API Core**
**Files:** `predict_multi_asset_ensemble.py`, `predict_per_asset.py`

**Capabilities:**
- Ensemble predictions (XGBoost + LightGBM + CatBoost)
- Multi-asset support (stocks, ETFs, crypto)
- 3-class forecasting: CRASH (0) / NORMAL (1) / SPIKE (2)
- Confidence scores per prediction
- Per-asset individual models
- Multi-horizon predictions (1d, 3d, 5d)

**What it does:**
Generates market movement forecasts using ensemble ML models trained on 126+ features

### 2. **Recession Probability Indicator**
**File:** `recession_indicator.py`

**Actual Functions:**
```python
- load_treasury_data()
- calculate_yield_curve_spread()      # 10Y-2Y Treasury spread
- calculate_unemployment_trend()      # Labor market analysis
- calculate_market_stress()           # Volatility + drawdown metrics
- calculate_technical_signals()       # Death cross, MA deviations
- calculate_recession_probability()   # Combined risk score
- format_recession_report()
- save_recession_analysis()
```

**What it does:**
Multi-signal recession risk analysis combining yield curves, unemployment, market stress, and technical signals

### 3. **Valuation Detector**
**File:** `valuation_detector.py`

**Actual Functions:**
```python
- calculate_rsi(prices, period=14)           # Overbought/oversold
- calculate_zscore(prices, period=252)       # Statistical deviation
- calculate_bollinger_position(prices)       # BB band position
- analyze_valuation(asset="SPY")             # Full analysis
- analyze_all_assets()                       # Batch analysis
- format_valuation_report()
```

**What it does:**
Identifies over/undervalued assets using RSI, Z-Score, Bollinger Bands, MA deviations

### 4. **LLM Market Analysis**
**File:** `llm_forecast.py`

**Actual Functions:**
```python
- get_available_assets()
- load_latest_predictions(asset)
- load_asset_data(asset)
- load_sentiment_data()
- get_market_news_summary(assets)            # NewsAPI integration
- build_context(asset, prediction, ...)      # Prepare LLM prompt
- get_llm_analysis(context, provider)        # OpenAI/Anthropic
- generate_forecast(asset, provider)         # Single asset analysis
- generate_multi_asset_summary(assets)       # Multi-asset overview
```

**Providers Supported:**
- OpenAI (GPT-4)
- Anthropic (Claude)

**What it does:**
AI-powered market commentary combining predictions, price data, sentiment, and news

### 5. **Portfolio Rebalancing Optimizer**
**File:** `portfolio_rebalancer.py`

**Actual Functions:**
```python
- load_trading_profile(profile_name)
- load_asset_data(asset, start_date, end_date)
- calculate_portfolio_value(prices, weights, rebalance_dates, method)
- find_optimal_rebalancing_period(assets, weights, lookback_years)
- execute_rebalancing(assets, target_weights, profile_name)
```

**What it does:**
Tests different rebalancing frequencies (daily, weekly, monthly, quarterly, etc.) and finds optimal strategy

### 6. **Model Training & Evaluation**
**Files:** `train_multi_asset.py`, `train_per_asset.py`, `evaluate.py`

**Training Options:**
- Standard training
- Hyperparameter tuning (full / fast)
- Ensemble weight optimization
- Feature selection
- Multi-horizon training
- Per-asset specialized models

**Models:**
- XGBoost
- LightGBM
- CatBoost
- Ensemble (weighted average)

### 7. **Feature Engineering**
**Files:** `add_macro_features.py`, `add_sentiment_features.py`, `external_signals.py`

**Features (126+ total):**
- Technical indicators (RSI, MACD, Bollinger, etc.)
- Cross-asset correlations
- Macro indicators (DXY, TNX, etc.)
- Sentiment features
- Sector rotations
- Options flow (if available)
- News-based signals

### 8. **Data Management**
**Download Scripts:**
- `update_spy_data.py` - SPY OHLCV data
- `download_equity_etfs.py` - Stock ETFs
- `download_crypto_enhanced.py` - Crypto from Binance/CoinGecko
- `import_custom_asset.py` - Custom CSV imports

### 9. **Performance Analysis**
**Files:** `extract_metrics.py`, `validate_signals.py`, `backtest.py`

**Capabilities:**
- Extract model accuracy, AUC, precision/recall
- Validate signal distribution
- Detect false positives
- Backtest with different risk profiles
- Performance attribution

### 10. **Risk Profiles**
**File:** `backtest.py`

**Profiles:**
- **Conservative:** 70%+ confidence, 1.0x ATR, 5-15% position
- **Moderate:** 55%+ confidence, 1.5x ATR, 10-25% position
- **Liberal/Aggressive:** 45%+ confidence, 2.0x ATR, 15-40% position

---

## 📁 DASHBOARD FILES (Historical Analysis)

### Git History (Oldest → Newest):
1. **framework/results_dashboard.py** (CLI, not Streamlit)
2. **dashboard.py** (Original Streamlit dashboard)
3. **dashboard_comprehensive.py** (Added recently)
4. **dashboard_demo.py** (Added recently)

### What Each Dashboard Should Show:

#### dashboard.py (Main API Dashboard)
**Purpose:** Primary interface for API users
**Should Show:**
- All 24 assets (stocks, ETFs, crypto)
- API forecast results
- Prediction accuracy metrics
- Asset price charts
- Custom data import

#### dashboard_comprehensive.py (Feature Showcase)
**Purpose:** Demonstrate ALL NeuroVest capabilities
**Should Show:**
- Recession indicator UI
- Valuation detector UI
- LLM analysis examples
- Portfolio rebalancing
- Multi-asset forecasts
- Risk profiles
- Signal analytics

#### dashboard_demo.py (Quick Testing)
**Purpose:** Developer testing interface
**Should Show:**
- System health checks
- Model status
- Quick forecast results
- Data validation

---

## 🔥 WHAT'S MISSING FROM CURRENT DASHBOARDS

### Missing from dashboard.py:
- ❌ Only showing assets in data/ directory
- ❌ Not showing all 24 supported assets
- ❌ Not showing crypto assets (need to download first)
- ❌ Not showing equity ETF options

### Missing from dashboard_comprehensive.py:
- ❌ Recession indicator UI (function exists, UI missing)
- ❌ Valuation detector UI (function exists, UI missing)
- ❌ Multi-asset comparison
- ❌ Full feature showcase

### Missing from dashboard_demo.py:
- ❌ Asset coverage status
- ❌ Feature availability checks

---

## ✅ RECOMMENDED FIXES

### 1. Asset Display Strategy
Show assets in tiers:
- **Tier 1 (Downloaded):** Assets with data files
- **Tier 2 (Available):** Assets that can be downloaded
- **Tier 3 (Custom):** User-imported assets

### 2. Feature Integration
Connect existing features to dashboards:
- Recession indicator → Interactive UI with charts
- Valuation detector → Per-asset valuation display
- LLM analysis → Show AI commentary in dashboard
- Portfolio rebalancing → Interactive optimizer

### 3. Asset Management UI
Add "Download Assets" page:
- List all 24 supported assets
- Show which are downloaded
- One-click download buttons
- Progress indicators

---

## 📊 USAGE STATISTICS (From Codebase)

**Features Implemented:** 10 major systems
**Total Python Files:** 120+
**Supported Assets:** 24
**ML Models:** 3 (XGBoost, LightGBM, CatBoost)
**API Providers:** 3 (NewsAPI, OpenAI, Anthropic)
**Risk Profiles:** 3
**Forecast Horizons:** 3 (1d, 3d, 5d)

---

## 🎯 WHAT NEUROVEST ACTUALLY DOES

**Primary Function:**
Market forecasting API that predicts price movements (CRASH/NORMAL/SPIKE) for stocks, ETFs, and cryptocurrencies using ensemble machine learning.

**Secondary Functions:**
1. Recession probability analysis
2. Asset valuation assessment
3. AI-powered market commentary
4. Portfolio rebalancing optimization
5. Multi-asset correlation analysis
6. Signal validation and backtesting

**What It's NOT:**
- NOT a trade execution system
- NOT an automated trading bot
- NOT a portfolio management service
- It's a **FORECASTING API** that provides predictions and analysis

---

## 🔧 ACTUAL API ENDPOINTS (from api/trading_api.py)

**Note:** FastAPI server exists but is for backtesting results, not live trading

**Data Models:**
- AssetType (stock, crypto)
- SignalType (buy, sell, hold)
- Position (historical positions from backtest)
- PortfolioStatus (backtest portfolio state)

---

## 📝 CONCLUSIONS

1. **NeuroVest has 10 major features** but dashboards only show ~30% of capabilities
2. **24 assets supported** but dashboards only show currently downloaded (6)
3. **Rich functionality exists** (recession, valuation, LLM) but not exposed in UI
4. **Clear mission:** Market forecasting API, not trading execution
5. **Key gap:** UI doesn't showcase what the code can actually do

**Next Steps:**
- Update dashboards to show all 24 assets
- Add asset download management UI
- Integrate recession/valuation/LLM features into Streamlit
- Clear "Forecasting API" branding throughout
