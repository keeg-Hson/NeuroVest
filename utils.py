# utils.py
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import os, shutil, datetime
import requests
import pandas_ta as ta

from dotenv import load_dotenv
from external_signals import add_external_signals #EXTERNAL SIGNALS MODULE

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
        "BB_Width", "Volatility", "OBV", "Vol_Ratio", "Price_Momentum_10", "Acceleration",
        "RSI", "RSI_Delta", "ZMomentum",
        "Return_Lag1", "Return_Lag3", "Return_Lag5",
        "RSI_Lag_1", "RSI_Lag_3", "RSI_Lag_5",
        "Rolling_STD_5", "Daily_Return",
        "MACD_x_RSI", "Volume_per_ATR",
        "Stoch_K", "Stoch_D"
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

    #simplified signal log for simulation
    signals_row = {
        "Date": date_str,
        "Signal": "BUY" if prediction == 2 else "SELL" if prediction == 1 else "HOLD",
        "Confidence": max(crash_conf, spike_conf),
        "Price": close_price,
        "Spike_Conf": spike_conf,
        "Crash_Conf": crash_conf
    }

    signals_path = "logs/signals.csv"
    signals_df = pd.DataFrame([signals_row])
    if os.path.exists(signals_path):
        signals_df.to_csv(signals_path, mode="a", header=False, index=False)
    else:
        signals_df.to_csv(signals_path, mode="w", header=True, index=False)

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

    df = pd.read_csv(spy_path, index_col=0, parse_dates=True)  # , skiprows=[1] if needed later
    df.sort_index(inplace=True)

    # Drop last row if critical columns are missing
    if df.iloc[-1][["High", "Low", "Volume"]].isna().any():
        df = df.iloc[:-1]

    return df





def add_features(df):
    df = df.copy()

    # --- Price-based Indicators ---
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD Indicators ---
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # --- RSI & Momentum ---
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    RS = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + RS))
    df["RSI_Delta"] = df["RSI"].diff()
    df["ZMomentum"] = (df["Close"] - df["Close"].rolling(10).mean()) / (df["Close"].rolling(10).std() + 1e-9)
    df["Price_Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Acceleration"] = df["Price_Momentum_10"].diff()

    # --- Lagged Features ---
    df["Return_Lag1"] = df["Close"].pct_change(1)
    df["Return_Lag3"] = df["Close"].pct_change(3)
    df["Return_Lag5"] = df["Close"].pct_change(5)
    df["RSI_Lag_1"] = df["RSI"].shift(1)
    df["RSI_Lag_3"] = df["RSI"].shift(3)
    df["RSI_Lag_5"] = df["RSI"].shift(5)

    # --- Bollinger Band Width ---
    rolling_std = df["Close"].rolling(20).std()
    upper_band = df["MA_20"] + (2 * rolling_std)
    lower_band = df["MA_20"] - (2 * rolling_std)
    df["BB_Width"] = (upper_band - lower_band) / (df["MA_20"] + 1e-9)

    # --- Volatility (define vol_window first!) ---
    vol_window = 20  # Set your default volatility window here
    df["Volatility"] = df["Close"].rolling(window=vol_window).std(ddof=0)
    #df["Volatility"] = df["Close"].rolling(window=vol_window).std()

    # --- On-Balance Volume (OBV) ---
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # --- Volume Ratio ---
    df["Vol_Ratio"] = df["Volume"] / (df["Volume"].rolling(10).mean() + 1e-9)

    # --- Rolling Std Dev ---
    df["Rolling_STD_5"] = df["Close"].rolling(window=5).std()

    # --- Daily Return ---
    df["Daily_Return"] = df["Close"].pct_change()

    # --- MACD × RSI Interaction ---
    df["MACD_x_RSI"] = df["MACD"] * df["RSI"]

    # --- Volume per ATR ---
    atr = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    df["Volume_per_ATR"] = df["Volume"] / (atr + 1e-9)

    # --- Stochastic Oscillator ---
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14 + 1e-9))
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # --- External Macro + Sentiment Signals ---
    df = add_external_signals(df)

    # --- Final clean-up: drop NaNs from anywhere ---
    #df.dropna(inplace=True)
    # Only return warning here — let train.py handle NaN rows later
    if df.isna().sum().sum() > 0:
        print("⚠️ Warning: NaNs remain in feature set — will be handled after labeling.")


    # --- Collect feature columns (only numeric & known) ---
    feature_cols = [
        "MA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Histogram",
        "RSI", "RSI_Delta", "ZMomentum", "Price_Momentum_10", "Acceleration",
        "Return_Lag1", "Return_Lag3", "Return_Lag5", "RSI_Lag_1", "RSI_Lag_3", "RSI_Lag_5",
        "BB_Width", "Volatility", "OBV", "Vol_Ratio", "Rolling_STD_5", "Daily_Return",
        "MACD_x_RSI", "Volume_per_ATR", "Stoch_K", "Stoch_D",
        "CPI", "Unemployment", "InterestRate", "YieldCurve",
        "ConsumerSentiment", "IndustrialProduction", "VIX",
        "News_Sentiment", "Reddit_Sentiment"
    ]

    return df, feature_cols




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

def label_events_volatility_adjusted(df, window=3, vol_window=10, multiplier=0.2):
    df = df.copy()
    df["Event"] = 0  # Default: no event

    df["Volatility"] = df["Close"].rolling(window=vol_window).std()

    for i in range(window, len(df)):
        window_slice = df.iloc[i - window:i + 1]
        current_close = df.iloc[i]["Close"]
        current_vol = df.iloc[i]["Volatility"]

        if pd.isna(current_vol) or current_vol == 0:
            continue

        spike_threshold = window_slice["Close"].mean() + multiplier * current_vol
        crash_threshold = window_slice["Close"].mean() - multiplier * current_vol

        if current_close > spike_threshold:
            df.at[df.index[i], "Event"] = 2
        elif current_close < crash_threshold:
            df.at[df.index[i], "Event"] = 1

    return df


def label_events_simple(df, window=3, pct_threshold=0.01):
    """
    Labels spikes and crashes based on simple forward percentage movement over 'window' days.
    """
    df = df.copy()
    df["Future_Return"] = (df["Close"].shift(-window) - df["Close"]) / df["Close"]

    conditions = [
        df["Future_Return"] <= -pct_threshold,  # Crash
        df["Future_Return"] >= pct_threshold    # Spike
    ]

    choices = [1, 2]  # 1 = crash, 2 = spike
    df["Event"] = np.select(conditions, choices, default=0)

    return df

def finalize_features(df, feature_cols):
    """
    Make feature matrix model-safe:
    - Intersect with existing columns
    - Ensure numeric dtype
    - Interpolate time-wise if a DatetimeIndex is present; otherwise fallback to ffill/bfill
    - Return df with original columns preserved
    """
    df = df.copy()

    # Keep only features that exist now
    cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) or c in df.columns]
    # Force numeric where possible
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If there's a Date column but no DatetimeIndex, temporarily set it
    had_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    reset_back = False
    if not had_datetime_index:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
            reset_back = True

    # Interpolate if time index; else skip to ffill/bfill
    if isinstance(df.index, pd.DatetimeIndex):
        df[cols] = df[cols].interpolate(method="time", limit_direction="both")

    # Always do safety fills
    df[cols] = df[cols].ffill().bfill()

    # Restore original index/Date column if changed
    if reset_back:
        df = df.reset_index().rename(columns={"index": "Date"})

    return df







