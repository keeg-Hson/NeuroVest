# NeuroVest

**AI-Powered Economic Forecasting & Trading Strategy System**

Advanced ensemble ML system for predicting market opportunities through regime analysis, cross-asset dynamics, and macro indicators. Features risk-managed backtesting, LLM-powered insights, and automated portfolio rebalancing.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Quick Start

**New to NeuroVest?** Launch the interactive menu:

```bash
python3 main.py
```

Then select **Option R** to run the complete pipeline automatically (20-35 minutes).

### First-Time Setup

```bash
# 1. Clone repository
git clone https://github.com/keeg-Hson/NeuroVest.git
cd NeuroVest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (optional)
cp .env.example .env
# Edit .env with your API keys

# 4. Launch main menu
python3 main.py
```

### Typical Workflow

1. **Download Data** → Menu: 5 → 1 (SPY) & 2-4 (Crypto)
2. **Train Models** → Menu: 1 → 4 (Optimized weights)
3. **Generate Predictions** → Menu: 2 → 1 (Ensemble) & 4 (Per-asset)
4. **Run Backtest** → Menu: 3 → 2 (Moderate profile)
5. **View Results** → Menu: 7 (Web Dashboard)

---

## 🎯 What This Does

NeuroVest trains **XGBoost, LightGBM, and CatBoost** ensemble models on **126+ features** to predict multi-day price movements with:

- **Accuracy**: ~59% (realistic, no data leakage)
- **AUC**: ~0.6
- **Signal Distribution**: 30% CRASH / 40% NORMAL / 30% SPIKE
- **Backtest Returns**: 191% (optimized) to 378% (aggressive) over 25 years
- **Sharpe Ratio**: 2.03 to 2.80
- **Max Drawdown**: -4% to -12.8%

### Core Capabilities

✅ **Multi-Asset Support** - SPY, crypto (BTC/ETH/SOL), ETFs, custom imports
✅ **Ensemble Learning** - 3 models with optimized weights
✅ **Risk Profiles** - Conservative / Moderate / Liberal trading strategies
✅ **Portfolio Management** - Rebalancing optimization, multi-asset backtesting
✅ **Market Analysis** - Recession indicator, valuation detector
✅ **AI Insights** - LLM-powered analysis (OpenAI/Anthropic)
✅ **Real-Time News** - NewsAPI integration for market context
✅ **Web Dashboard** - Interactive Streamlit interface

---

## 📊 Key Features

### 1. **Trading Risk Profiles**

Choose your risk tolerance:

| Profile | Confidence | Stop Loss | Position Size | Max Equity | Best For |
|---------|-----------|-----------|---------------|------------|----------|
| **Conservative** | 70%+ | 1.0x ATR | 5-15% | 40% | Risk-averse, steady growth |
| **Moderate** | 55%+ | 1.5x ATR | 10-25% | 65% | Balanced risk-reward |
| **Liberal** | 45%+ | 2.0x ATR | 15-40% | 85% | Aggressive, high returns |

Access via: **Menu → 3 (Backtesting) → 1-3**

### 2. **Recession Probability Indicator**

Multi-signal recession analysis:

- **Yield Curve** - 10Y-2Y Treasury spread inversion detection
- **Market Stress** - Volatility, drawdown, performance metrics
- **Technical Signals** - Price vs MAs, death cross patterns
- **Risk Levels** - LOW (0-25%), MODERATE (25-40%), ELEVATED (40-60%), HIGH (60%+)

Access via: **Menu → 4 (Diagnostics) → 5**

### 3. **Valuation Detector**

Over/undervalued asset analysis using:

- RSI (overbought/oversold)
- Z-Score (statistical deviation)
- Bollinger Bands position
- MA deviation (50-day, 200-day)
- 30-day momentum

**Valuation Score**: -1.0 (deeply undervalued) to +1.0 (overvalued)

Access via: **Menu → 4 (Diagnostics) → 6-7**

### 4. **Portfolio Rebalancing**

Optimize rebalancing frequency:

- Tests: Daily, Weekly, Monthly, Quarterly, Semi-annual, Annual
- Includes transaction costs
- Calculates Sharpe, returns, max drawdown
- Finds optimal strategy automatically

Access via: **Menu → 3 (Backtesting) → 10-11**

### 5. **LLM Market Analysis**

AI-powered insights with scenario likelihoods:

```
SCENARIO LIKELIHOODS:
- CRASH (Bearish):  15% - Significant downward movement
- NORMAL (Neutral): 25% - Sideways/mixed price action
- SPIKE (Bullish):  60% - Significant upward movement
```

Supports OpenAI GPT-4 and Anthropic Claude.

Access via: **Menu → 4 (Diagnostics) → 8-10**

### 6. **Real-Time News Integration**

Fetches financial news from:
- Bloomberg, Reuters, WSJ, CNBC, Financial Times
- Asset-specific news queries
- Integrated into LLM analysis

Requires `NEWS_API_KEY` in `.env`

---

## 🎓 Training Options

### Standard Training (5-10 min)
```bash
python3 train_multi_asset.py
```

### Hyperparameter Tuning
```bash
python3 train_multi_asset.py --tune        # Full search (15-30 min)
python3 train_multi_asset.py --tune-fast   # Quick search (5-10 min)
```

### Accuracy Improvements
```bash
python3 train_multi_asset.py --optimize-weights                    # Optimal ensemble weights
python3 train_multi_asset.py --optimize-weights --feature-select   # + feature selection
```

### Multi-Horizon Training
```bash
python3 train_multi_horizon_signals.py                    # 1d, 3d, 5d horizons
python3 train_multi_horizon_signals.py --horizons 1 3     # Specific horizons
```

### Per-Asset Training
```bash
python3 train_per_asset.py                  # All assets
python3 train_per_asset.py --asset SPY      # Single asset
```

Trained models saved to `models/`, hyperparameters to `models/best_hyperparameters.json`.

---

## 📈 Backtesting

### Risk Profile Backtests
```bash
# Menu → 3 → 1-3 for guided profile selection
```

### Configuration-Based Backtests
```bash
python3 backtest.py --config configs/backtest_optimized.json    # 191% return, 2.55 Sharpe
python3 backtest.py --config configs/backtest_high_profit.json  # 330% return, 2.30 Sharpe
python3 backtest.py --config configs/backtest_aggressive.json   # 378% return, 2.03 Sharpe
```

### Asset-Specific Backtests
```bash
python3 backtest.py --asset BTC/USDT
python3 backtest.py --asset-group crypto --compare
```

### Portfolio Backtests
```bash
python3 backtest_portfolio.py --assets SPY,GLD,TLT --weights 0.6,0.3,0.1 --rebalance monthly
```

### Backtest Results

| Config | TP ATR | Return | Sharpe | Max DD | Win Rate |
|--------|--------|--------|--------|--------|----------|
| Conservative | 1.0x | ~150% | 2.80 | -4.0% | 62% |
| Optimized | 1.25x | 191% | 2.55 | -5.4% | 58% |
| High Profit | 1.75x | 330% | 2.30 | -7.4% | 56% |
| Aggressive | 2.5x | 378% | 2.03 | -12.8% | 54% |

---

## 🌐 Web Dashboard

Interactive UI with real-time insights:

```bash
streamlit run dashboard.py
# Or: Menu → 7
```

**Features:**
- 📊 Asset overview with charts (RSI, volume, MAs)
- 🎯 Prediction viewer
- 📉 Backtest results visualization
- 📥 Custom data import (CSV/Excel)
- 🔄 Live data refresh

Accessible at `http://localhost:8501`

---

## 🤖 AI & LLM Features

### Single Asset Analysis
```bash
python3 llm_forecast.py --asset SPY --provider openai
```

### Multi-Asset Summary
```bash
python3 llm_forecast.py --all --provider anthropic
```

### Newsletter Generation
```bash
python3 newsletter_generator.py --preview --assets SPY,BTC/USDT
python3 newsletter_generator.py --send --assets SPY
```

**Required in `.env`:**
```bash
OPENAI_API_KEY=your-key-here
# Or
ANTHROPIC_API_KEY=your-key-here

# For news integration
NEWS_API_KEY=your-newsapi-key

# For newsletter email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
NEWSLETTER_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

---

## 📥 Data Management

### Download Market Data
```bash
python3 update_spy_data.py                          # SPY (S&P 500)
python3 download_crypto_enhanced.py                 # Top 10 crypto
python3 download_crypto_comprehensive.py            # 15 crypto, multi-source
python3 download_equity_etfs.py                     # Various ETFs
```

### Import Custom Data
```bash
python3 import_custom_asset.py mydata.csv TICKER    # Import CSV/Excel
python3 import_custom_asset.py --sample             # Generate template
python3 import_custom_asset.py --list               # List imported assets
```

**Required columns:** Date, Close (or Price)
**Optional:** Open, High, Low, Volume

### Live Updates
```bash
python3 live_update.py --mode scheduled --assets SPY,QQQ --interval 15
python3 live_update.py --download    # Download all historical data
```

---

## 🔧 Advanced Features

### Find Optimal Rebalancing Period
```bash
python3 portfolio_rebalancer.py --find-optimal --assets SPY,GLD,TLT --weights 0.6,0.3,0.1
```

Tests all frequencies, outputs best strategy based on Sharpe ratio.

### Recession Analysis
```bash
python3 recession_indicator.py --save
```

Generates comprehensive recession probability report.

### Valuation Analysis
```bash
python3 valuation_detector.py --asset SPY
python3 valuation_detector.py --all --save
```

Analyzes over/undervaluation using multiple technical indicators.

### Fetch Market News
```bash
python3 fetch_news.py --asset BTC/USDT --days 7
python3 fetch_news.py --query "federal reserve" --save
```

---

## 📐 How Predictions Work

**Pipeline:**

1. **Feature Engineering** - 126+ features across 5 categories:
   - Technical (40): RSI, MACD, Bollinger Bands, ATR, MAs
   - Cross-Asset (24): Credit spreads, yields, dollar strength, crypto vol
   - Macro (18): 10Y yield, yield curve, rate changes
   - Regime (32): VIX regimes, trend strength, volatility regimes
   - Interactions (12): Non-linear combinations

2. **Model Training** - XGBoost, LightGBM, CatBoost with:
   - TimeSeriesSplit cross-validation
   - Hyperparameter tuning (optional)
   - Ensemble weight optimization

3. **Prediction Generation** - Ensemble averaging:
   - Each model generates probability scores
   - Scores averaged for ensemble probability
   - Percentile-based thresholds (30th/70th):
     - Bottom 30% → CRASH (short signal)
     - Middle 40% → NORMAL (hold)
     - Top 30% → SPIKE (long signal)

4. **Confidence Calculation** - Percentile-relative:
   - SPIKE predictions: confidence based on distance above 70th percentile
   - CRASH predictions: confidence based on distance below 30th percentile
   - Higher confidence → larger position sizes

**No data leakage** - All features lagged ≥1 day, rigorous validation.

---

## 📁 Project Structure

```
NeuroVest/
├── main.py                          # Main menu interface
├── train_multi_asset.py             # Multi-asset training
├── train_per_asset.py               # Per-asset training
├── train_multi_horizon_signals.py   # Multi-horizon training
├── predict_multi_asset_ensemble.py  # Ensemble predictions
├── predict_per_asset.py             # Per-asset predictions
├── backtest.py                      # Backtesting engine
├── backtest_portfolio.py            # Portfolio backtesting
├── portfolio_rebalancer.py          # Rebalancing optimizer
├── recession_indicator.py           # Recession analysis
├── valuation_detector.py            # Valuation analysis
├── llm_forecast.py                  # LLM market analysis
├── fetch_news.py                    # News API integration
├── dashboard.py                     # Streamlit web interface
├── newsletter_generator.py          # Email newsletter
├── diagnose_system.py               # System diagnostics
├── utils.py                         # Feature engineering
├── config.py                        # Configuration
│
├── configs/                         # Configuration files
│   ├── backtest_*.json              # Backtest configs
│   └── trading_profile_*.json       # Risk profiles
│
├── models/                          # Trained models (.pkl)
├── logs/                            # Predictions, analysis
│   └── predictions/                 # Per-asset predictions
├── data/                            # Market data (SPY, etc.)
└── data_cache/                      # Downloaded assets
```

---

## ⚙️ Configuration

### Training Parameters (`config.py`)

```python
TRAIN_CFG = {
    "horizon": 1,              # Days forward for prediction
    "pos_threshold": 0.005,    # 0.5% min return for positive label
    "fee_bps": 1.5,            # Transaction fees (basis points)
    "slippage_bps": 2.0,       # Slippage assumption
    "weight_power": 1.75,      # Sample weighting exponent
}
```

### Asset-Specific Thresholds

Calibrated to ~0.5x daily volatility:

- **SPY**: 0.6% (daily vol ~1.2%)
- **BTC**: 2.2% (daily vol ~4.4%)
- **ETH**: 2.8% (daily vol ~5.7%)
- **SOL**: 4.0% (daily vol ~8.0%)

### Backtest Parameters

Key settings in `configs/backtest_*.json`:

- `target_ann_vol`: Target annualized volatility (0.15-0.18)
- `conf_size_bounds`: [min, max] position size based on confidence
- `sl_atr`, `tp_atr`: Stop loss and take profit (ATR multiples)
- `use_regime_filter`: Only long in uptrends, short in downtrends

---

## 🛠️ System Diagnostics

If experiencing issues:

```bash
python3 diagnose_system.py
```

**Checks:**
- ✓ Data files exist
- ✓ Models load correctly
- ✓ Prediction distribution (should be ~30/40/30)
- ✓ Backtest signal generation
- ✓ Feature calculation

**Common Issues:**

| Issue | Likely Cause | Fix |
|-------|-------------|------|
| 99% accuracy | Data leakage | Check features don't include forward data |
| 0% NORMAL predictions | Threshold issue | System uses percentile-based thresholds (fixed) |
| Few trades | Thresholds too strict | Lower confidence requirements in profile |
| Poor backtest | Overfitting | Use cross-validation, reduce features |

---

## 🎯 Performance Metrics

**Expected Performance:**

- **Test Accuracy**: ~59% (realistic, sustainable)
- **AUC-ROC**: ~0.60
- **Precision (SPIKE)**: ~60-65%
- **Recall (SPIKE)**: ~55-60%
- **Signal Distribution**: 30% crash / 40% normal / 30% spike

**Backtest Performance (25 years, SPY):**

- **Conservative**: 150% return, 2.80 Sharpe, -4% max DD
- **Optimized**: 191% return, 2.55 Sharpe, -5.4% max DD
- **High Profit**: 330% return, 2.30 Sharpe, -7.4% max DD
- **Aggressive**: 378% return, 2.03 Sharpe, -12.8% max DD

---

## 🌟 Recent Updates

**New in Latest Release:**

✨ **Trading Risk Profiles** - Conservative/Moderate/Liberal with preset parameters
✨ **Recession Indicator** - Multi-signal recession probability analysis
✨ **Valuation Detector** - Over/undervalued asset identification
✨ **Portfolio Rebalancing** - Automated optimal period finder
✨ **News Integration** - Real-time news from NewsAPI
✨ **Scenario Likelihoods** - Crash/Normal/Spike probability distribution
✨ **Enhanced UX** - Improved menu formatting, error handling, validation
✨ **Per-Asset Predictions** - Individual models for each asset

---

## ⚠️ Important Disclaimers

**This is a research/educational project. NOT financial advice.**

- **Test accuracy ~59%** means ~41% of signals will be wrong
- Backtest shows modest returns but **past performance ≠ future results**
- **Do NOT use with real money** without extensive paper trading (6-12 months minimum)
- Use proper **position sizing** and **risk management**
- Start with amounts **you can afford to lose entirely**

**For educational and research purposes only.**

---

## 📚 Documentation

- [FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md) - Full framework documentation
- [TRAINING_SYSTEMS_GUIDE.md](TRAINING_SYSTEMS_GUIDE.md) - Training approaches
- [ACCURACY_OPTIMIZATION_GUIDE.md](ACCURACY_OPTIMIZATION_GUIDE.md) - Threshold tuning
- [CRASH_PREDICTION_ANALYSIS.md](CRASH_PREDICTION_ANALYSIS.md) - Crash detection analysis
- [MULTI_ASSET_ANALYSIS_SUMMARY.md](MULTI_ASSET_ANALYSIS_SUMMARY.md) - Portfolio tools

---

## 📦 Requirements

```
Python 3.8+
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
yfinance>=0.2.0
ccxt>=3.0.0
streamlit>=1.29.0
plotly>=5.18.0
openai>=1.0.0
anthropic>=0.7.0
openpyxl>=3.1.2
python-dotenv>=1.0.0
requests>=2.31.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**keeg-Hson**

- GitHub: [@keeg-Hson](https://github.com/keeg-Hson)

---

## 🙏 Acknowledgments

Built with:
- XGBoost, LightGBM, CatBoost
- Streamlit
- OpenAI GPT-4, Anthropic Claude
- yfinance, CCXT
- NewsAPI

---

**Last Updated**: December 2024

---

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on [GitHub](https://github.com/keeg-Hson/NeuroVest/issues)
- Check existing documentation in `/docs`
- Review diagnostic output from `diagnose_system.py`

---

**Happy Trading! 📈**

*Remember: This is for educational purposes. Always paper trade first.*
