# 📈 Market-Trading-Bot

## 👨‍💻 Author

- Keegan Hutchinson [@keeg-Hson](https://www.github.com/keeg-Hson)

- # 📈 Market Prediction Bot

Market-Trading-Bot
==================

Overview
--------
Market-Trading-Bot is an AI-powered market prediction and trading assistant that analyzes historical and live market data to detect potential market spikes and crashes. 
It uses an XGBoost classifier alongside technical indicators, sentiment analysis, and macroeconomic signals to generate actionable predictions.

The system supports:
- Historical backtesting
- Threshold optimization sweeps
- Live prediction scheduling (cron-compatible)
- Telegram alerts for trade signals
- Integration with Reddit, NewsAPI, and FRED economic indicators

Repository Link
---------------
https://github.com/keeg-Hson/Market-Trading-Bot

Current Status
--------------
As of August 2025, the bot:
1. Loads and updates SPY (S&P 500 ETF) data automatically via `update_spy_data.py` (yfinance).
2. Generates predictions using a trained **XGBoost** model from `market_crash_model.pkl`.
3. Enhances predictions with:
   - Technical indicators (RSI, momentum, moving averages)
   - Reddit sentiment (PRAW + TextBlob)
   - News sentiment (NewsAPI)
   - Macroeconomic indicators (FRED)
4. Logs all predictions in `logs/daily_predictions.csv` with timestamps, class labels, and confidence scores.
5. Supports backtesting with capital tracking (`backtest.py`, `backtest_module.py`).
6. Runs full parameter sweeps to identify optimal thresholds (`sweep_runner.py`, `threshold_sweep.py`).
7. Sends Telegram alerts with trade signal details and optional graphs.

Planned Features
----------------
- Trade execution integration with broker APIs (Alpaca, IBKR)
- Automated daily run with enriched Telegram reporting (graphs, trade history)
- Improved model accuracy with expanded feature engineering and alternative ML models (LightGBM, ensembles)
- Archiving & versioning of prediction logs
- Leaderboard visualization for sweep results

Repository Structure
--------------------
.
├── configs/                    # Config files for parameters and sweep setups
├── data/                        # Historical market data (SPY, etc.)
├── graphs/                      # Generated prediction and sweep graphs
├── logs/                        # Prediction and sweep result logs
├── models/                      # Trained model files (XGBoost)
├── analyze_signals.py           # Analyzes combined external and technical signals
├── backtest.py                  # Full backtesting script with capital tracking
├── backtest_module.py           # Backtesting functions (importable)
├── config.py                    # Script-wide configuration constants
├── crontab.txt                  # Example crontab entries for automation
├── data_utils.py                # Data handling utilities
├── download_spy_data.py         # Download historical SPY data
├── evaluate.py                  # Model evaluation and metrics
├── external_signals.py          # Reddit, NewsAPI, and FRED integration
├── fetch_spy_history.py         # Fetch historical SPY data
├── fetch_spy.py                 # Fetch latest SPY data
├── generate_labels.py           # Generate training labels from market events
├── live_loop.py                 # Continuous market monitoring loop
├── main.py                      # Main entry point for orchestrating all components
├── market_crash_model.pkl       # Trained XGBoost model
├── predict.py                   # Generate live/historical predictions
├── run_all.py                   # Run full data update → prediction → logging pipeline
├── run_daily_pipeline.py        # Scheduled daily prediction run
├── select_top_signals.py        # Filter and rank the best trading signals
├── signal_logger.py             # Log trade/prediction signals
├── sweep_optimizer.py           # Optimize sweep runs
├── sweep_runner.py              # Run threshold sweeps and save results
├── threshold_sweep.py           # Threshold-based performance testing
├── top_config_runner.py         # Run top-performing sweep configurations
├── trade_executor.py            # Execute trades via broker API (future)
├── trade_simulator.py           # Simulate trades on historical data
├── train.py                     # Train XGBoost model
├── train_from_labels.py         # Train from pre-labeled datasets
├── update_spy_data.py           # Update SPY data from yfinance
├── utils.py                     # Shared helper functions (feature lists, logging, formatting)
├── viz.py                       # Visualization scripts for predictions/sweeps
└── .env                         # Environment variables (API keys, tokens)

Setup
-----
1. Clone this repository:
   git clone https://github.com/keeg-Hson/Market-Trading-Bot.git
   cd Market-Trading-Bot

2. Install dependencies:
   pip install -r requirements.txt

3. Create a `.env` file with the following:
   TELEGRAM_TOKEN=your_telegram_token
   TELEGRAM_CHAT_ID=your_chat_id
   REDDIT_CLIENT_ID=your_reddit_id
   REDDIT_CLIENT_SECRET=your_reddit_secret
   REDDIT_USER_AGENT=your_agent
   NEWSAPI_KEY=your_newsapi_key
   FRED_API_KEY=your_fred_api_key

4. Ensure you have Python 3.10+ installed.

Usage
-----
- Update SPY data:
    python3 update_spy_data.py

- Generate predictions:
    python3 predict.py

- Run backtest:
    python3 backtest.py

- Run parameter sweeps:
    python3 sweep_runner.py

- Schedule daily predictions (example crontab entry for 9:30 AM):
    30 9 * * 1-5 cd /path/to/Market-Trading-Bot && /usr/local/bin/python3 run_daily_pipeline.py

Development Notes
-----------------
- `utils.py` contains `get_feature_list()`, `log_prediction_to_file()`, and `in_human_speak()` for consistency.
- CSV logging is timestamped and validated to avoid malformed rows.
- External signals are merged directly into the SPY dataframe before prediction.
- Model confidence thresholds and crash/spike probability cutoffs are adjustable via CLI args.

Next Steps
----------
- Integrate automated broker trade execution.
- Add rolling model retraining and outcome labeling.
- Expand Telegram alerts to include graphs and trade summaries.
- Improve parameter sweep visualizations.

---

## 📜 License

MIT License. Free for public and commercial use. Attribution appreciated.

---

## 👤 Author

Built with ❤️ by [Keegan Hutchinson](https://github.com/keeg-Hson)  
Contributions, feedback, and improvements welcome!




