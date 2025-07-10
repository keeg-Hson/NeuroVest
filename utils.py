# utils.py
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import os, shutil, datetime



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
def log_prediction_to_file(timestamp, prediction, crash_conf, spike_conf, close_price, log_path=LOG_FILE):
    print("[DEBUG] Entering log_prediction_to_file")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    entry = f"{timestamp},{prediction},{crash_conf:.4f},{spike_conf:.4f},{close_price:.2f}\n"

    with open(log_path, "a") as f:
        if not file_exists or os.path.getsize(log_path) == 0:
            f.write("Timestamp,Prediction,Crash_Conf,Spike_Conf,Close_Price\n")  # ✅ FIXED
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








