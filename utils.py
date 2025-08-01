# utils.py
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import os, shutil, datetime
import requests
from dotenv import load_dotenv

load_dotenv()



# --- Ensure folders exist ---
os.makedirs("logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# --- Global log file paths ---
LOG_FILE = "logs/daily_predictions.csv"
LABELED_LOG_FILE = "logs/labeled_predictions.csv"

# --- Ensure labeled_predictions.csv exists ---
def init_labeled_log_file():
    if not os.path.exists(LABELED_LOG_FILE):
        print("[Init] Creating blank labeled_predictions.csv")
        with open(LABELED_LOG_FILE, "w") as f:
            f.write("Timestamp,Prediction,Crash_Conf,Spike_Conf,Close_Price,Actual_Event\n")

# --- Feature List ---
def get_feature_list():
    return [
        "MA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Width", "OBV", "Vol_Ratio", "Price_Momentum_10", "Acceleration",
        "RSI", "RSI_Delta", "ZMomentum",
        "Return_Lag1", "Return_Lag3", "Return_Lag5",
        "RSI_Lag_1", "RSI_Lag_3", "RSI_Lag_5"
    ]

# --- Log prediction to file ---
def log_prediction_to_file(timestamp, prediction, crash_conf, spike_conf, close_price, open_price=None, high=None, low=None, log_path=LOG_FILE):
    print("[DEBUG] Entering log_prediction_to_file")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    headers = "Date,Timestamp,Prediction,Crash_Conf,Spike_Conf,Close_Price,Open,High,Low\n"
    date_str = timestamp.date()

    entry = f"{date_str},{timestamp},{prediction},{crash_conf:.4f},{spike_conf:.4f},{close_price:.2f},{open_price},{high},{low}\n"

    with open(log_path, "a") as f:
        if not file_exists or os.path.getsize(log_path) == 0:
            f.write(headers)
        f.write(entry)
        print(f"[DEBUG] Wrote entry to {log_path}: {entry.strip()}")

# --- Human-readable prediction output ---
def in_human_speak(prediction, crash_conf, spike_conf):
    if prediction == 1:
        if crash_conf < 0.2:
            return f"✅ MARKET APPEARS STABLE! Very low crash confidence ({crash_conf*100:.1f}%)"
        elif crash_conf < 0.5:
            return f"⚠️ CAUTION: Moderate crash risk detected ({crash_conf*100:.1f}%)"
        else:
            return f"🚨 HIGH CRASH RISK! Confidence: {crash_conf*100:.1f}%"
    
    elif prediction == 2:
        if spike_conf > 0.8:
            return f"📈 STRONG BUY SIGNAL! Spike confidence: {spike_conf*100:.1f}%"
        elif spike_conf > 0.5:
            return f"✅ POSSIBLE RALLY: Spike confidence at {spike_conf*100:.1f}%"
        else:
            return f"🔍 Mild spike likelihood ({spike_conf*100:.1f}%)"
    
    else:
        if crash_conf > 0.4:
            return f"⚠️ NEUTRAL TREND, but some crash signals ({crash_conf*100:.1f}%)"
        elif spike_conf > 0.4:
            return f"⚠️ NEUTRAL TREND, with possible upward signals ({spike_conf*100:.1f}%)"
        else:
            return f"✅ MARKET APPEARS STABLE: no significant crash/spike behaviour detected! ({crash_conf*100:.1f}% crash confidence)"

# --- Label real outcomes using future close data ---
def label_real_outcomes_from_log(crash_thresh=-0.005, spike_thresh=0.005):
    if not os.path.exists(LOG_FILE):
        print("[⚠️] daily_predictions.csv not found — skipping outcome labeling.")
        return

    df = pd.read_csv(LOG_FILE, parse_dates=["Timestamp"])
    print(f"[DEBUG] read {len(df)} rows from {LOG_FILE}")
    print(df.tail())

    df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)

    if len(df) < 2:
        print("[⏭] Not enough data to label real outcomes — skipping for now.")
        return

    df["Next_Close"] = df["Close_Price"].shift(-1)
    df["Future_Return"] = (df["Next_Close"] - df["Close_Price"]) / df["Close_Price"]
    df["Actual_Event"] = np.select(
        [df["Future_Return"] < -0.005, df["Future_Return"] > 0.005],
        [1, 2],
        default=0
    )

    df.dropna(subset=["Future_Return"], inplace=True)
    df.to_csv(LABELED_LOG_FILE, index=False)
    print(f"[DEBUG] wrote {len(df)} rows to {LABELED_LOG_FILE}")
    #print(f"[Labeling] Return: {future_return:.4f} → Event: {actual_event}")

    print("[✅] Labeled outcomes written to logs/labeled_predictions.csv")

    #backup_logs()
    df.to_csv(LABELED_LOG_FILE, index=False)
    print("[✅] Labeled outcomes written to logs/labeled_predictions.csv")

    backup_logs()


def backup_logs():
    """Make timestamped copies of your two main CSVs into ./backups/"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("backups", exist_ok=True)
    shutil.copy(LOG_FILE, f"backups/daily_predictions_{ts}.csv")
    shutil.copy(LABELED_LOG_FILE, f"backups/labeled_predictions_{ts}.csv")

def summarize_trades(trades, initial_balance=10000, save_plot_path=None):
    """
    Summarizes trade results: final balance, win rate, trade count.
    Compatible with trades from both simulate_trades() and run_backtest().

    Returns dict with balance, win rate, and equity curve.
    """
    if not trades:
        return {
            'final_balance': initial_balance,
            'total_trades': 0,
            'win_rate': 0.0,
            'equity_curve': [initial_balance]
        }

    equity = [initial_balance]
    wins = 0
    total_trades = 0

    for trade in trades:
        if 'ROI' in trade:
            roi = trade['ROI']
        elif 'Entry_Price' in trade and 'Exit_Price' in trade:
            roi = (trade['Exit_Price'] - trade['Entry_Price']) / trade['Entry_Price']
        else:
            print("⚠️ Skipping trade — missing ROI or price data:", trade)
            continue

        total_trades += 1
        if roi > 0:
            wins += 1
        equity.append(equity[-1] * (1 + roi))

    final_balance = equity[-1]
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    if save_plot_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(equity, linewidth=2)
        plt.title("Equity Curve")
        plt.xlabel("Trade #")
        plt.ylabel("Account Balance ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()

    return {
        'final_balance': final_balance,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'equity_curve': equity
    }

import pandas as pd
import os

def load_SPY_data():
    spy_path = "data/SPY.csv"
    if not os.path.exists(spy_path):
        raise FileNotFoundError(f"[❌] Could not find SPY data at {spy_path}")

    df = pd.read_csv(spy_path, skiprows=[1], index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    return df



def add_features(df):
    df = df.copy()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["Close"]

    if "Volume" in df.columns:
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    else:
        df["OBV"] = 0


    if "Volume" in df.columns:
        df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(window=10).mean()
    else:
        df["Vol_Ratio"] = 1  # Fallback value or skip the feature entirely


    df["Price_Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Acceleration"] = df["Price_Momentum_10"] - df["Price_Momentum_10"].shift(5)

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))
    df["RSI_Delta"] = df["RSI"].diff()

    df["ZMomentum"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).std()

    df["Return_Lag1"] = df["Close"].pct_change(1)
    df["Return_Lag3"] = df["Close"].pct_change(3)
    df["Return_Lag5"] = df["Close"].pct_change(5)

    df["RSI_Lag_1"] = df["RSI"].shift(1)
    df["RSI_Lag_3"] = df["RSI"].shift(3)
    df["RSI_Lag_5"] = df["RSI"].shift(5)

    df.dropna(inplace=True)
    return df

def save_predictions_dataframe(df, path="logs/latest_predictions.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[💾] Predictions saved to {path}")

#notification helper
def notify_user(prediction, crash_conf, spike_conf):
    """
    Triggers alert if crash or spike is highly probable.
    """
    should_alert = False
    label = "NORMAL"

    if prediction == 1 and crash_conf >= 0.7:
        should_alert = True
        label = "CRASH"
    elif prediction == 2 and spike_conf >= 0.7:
        should_alert = True
        label = "SPIKE"

    if should_alert:
        print("\n🚨🚨🚨 MARKET ALERT 🚨🚨🚨")
        print(f"⚠️  High-confidence {label} detected!")
        print(f"📉 Crash: {crash_conf:.2f} | 📈 Spike: {spike_conf:.2f}\n")

        # Make a system beep
        try:
            os.system('play -nq -t alsa synth 0.3 sine 880')  # Linux
        except:
            os.system('printf "\a"')  # Fallback for macOS/Linux beep

#telegram bot
def send_telegram_alert(message, token=None, chat_id=None):
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram credentials missing.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"⚠️ Telegram error: {response.text}")
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")




