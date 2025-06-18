# 📈 Market-Trading-Bot

## 👨‍💻 Author

- Keegan Hutchinson [@keeg-Hson](https://www.github.com/keeg-Hson)

- # 📈 Market Prediction Bot

A fully automated AI-powered market analysis system that fetches live stock data, predicts short-term market events (Crash, Spike, or Normal), logs daily predictions, and generates visual dashboards — all powered by machine learning and technical indicators.

---

## 🧠 What It Does

This bot uses a trained Random Forest model to classify the market's short-term future into three categories:

- **Normal**: No significant change expected.
- **Crash**: A drop of more than 3% is predicted.
- **Spike**: A gain of more than 3% is predicted.

It processes technical indicators from SPY (S&P 500 ETF), runs daily predictions, logs results, and visualizes everything into a dashboard saved to `/graphs`.

---

## 🚀 Key Features

- 🔍 **Live data ingestion** via Alpha Vantage API (OHLCV)
- 🧠 **Machine learning classification** using Random Forest
- ⚙️ **Feature engineering** including:
  - Moving Averages (SMA, EMA)
  - RSI, MACD, Bollinger Bands
  - Momentum, Volatility, OBV, Volume Ratios
- 🕒 **Automated daily scheduler** (runs every day at 6:00pm)
- 📊 **Dual visualization**: market chart + confidence trend
- 📝 **Prediction logs** with confidence values, event labels, and real-time price tracking
- 🧾 **Human-readable output** for easier interpretation

---

## 📂 Directory Structure

```
├── main.py                  # Main script with scheduler
├── utils.py                # Core functions and helpers
├── predict.py              # Live prediction module
├── models/                 # Stores trained ML model
├── logs/                   # CSV log of predictions
├── graphs/                 # Daily dashboard visualizations
├── .env                    # API key (not included)
```

---

## ⚙️ How to Run

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/market-prediction-bot.git
cd market-prediction-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add your Alpha Vantage API key**

Create a `.env` file in the project root:
```
ALPHA_VANTAGE_KEY=your_api_key_here
```

4. **Run the bot**
```bash
python main.py
```

It will make one prediction immediately, then schedule itself to run daily at 6:00 PM. Logs and visualizations are auto-generated.

---

## 📊 Example Output

```
🔮 Prediction: 0 (Normal)
📊 Class Probabilities: [1. 0. 0.]
✅ MARKET APPEARS STABLE: No crash or spike predicted.
[✅] Saved plot to graphs/combined_dashboard_2025-06-18.png
```

---

## ✅ To-Do / Upcoming

- [ ] Support for multiple tickers
- [ ] Email or Telegram alert integration
- [ ] Weekly retraining from live data
- [ ] Web-based dashboard
- [ ] Integration with portfolio simulation

---

## ⚠️ Notes

- Alpha Vantage free tier has strict rate limits (5 requests/min, 500/day)
- Ensure logs follow the expected format to avoid tokenization errors
- If using on a server or VPS, consider running via `cron` or background daemon

---

## 📜 License

MIT License. Free for public and commercial use. Attribution appreciated.

---

## 👤 Author

Built with ❤️ by [Keegan Hutchinson](https://github.com/keeg-Hson)  
Contributions, feedback, and improvements welcome!




## Quick Start:
1. **Clone the repo**
   ```bash
   git clone https://github.com/keeg-Hson/Market-Trading-Bot
   cd Market-Trading-Bot

2. **Set up enviornment**
    **A) Install dependencies**
       
        pip install -r requirements.txt
    
    **B) Create .env file, add [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) here**
    
        ALPHA_VANTAGE_KEY=your_api_key_here

3. **Run program maually**
    
        python main.py

4. **Run with daily scheduler**
    
        python main.py  #Scheduler initializes automatically at 6:00 PM

-----

## Example Output:
### Graphs:
Saved to /graphs/
### Model: 
Stored in Market_Crash_Model.pkl
### Logs:
 Confidence prediction valuations stored in /prediction_log.txt
-----




