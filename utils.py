#utils.py
import os
import csv
from datetime import datetime

#ensures repective folders exist
os.makedirs("logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

#global log file
LOG_FILE = "logs/daily_predictions.csv"



#Features
def get_feature_list():
    return [
        "MA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Width", "OBV", "Vol_Ratio", "Price_Momentum_10", "Acceleration",
        "RSI", "RSI_Delta", "ZMomentum",
        "Return_Lag1", "Return_Lag3", "Return_Lag5",
        "RSI_Lag_1", "RSI_Lag_3", "RSI_Lag_5"
    ]

#logging function to log predictions to a CSV file
#import csv
#from datetime import datetime
#def log_prediction_to_file(timestamp,prediction,Crash_Conf,Spike_Conf, file_path=LOG_FILE):
    
def log_prediction_to_file(timestamp, prediction, crash_conf, spike_conf, close_price, log_path='logs/daily_predictions.csv'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    entry = f"{timestamp},{prediction},{crash_conf:.4f},{spike_conf:.4f},{close_price:.2f}\n"

    with open(log_path, "a") as f:
        if not file_exists or os.path.getsize(log_path) == 0:
            f.write("Timestamp,Prediction,Crash_Conf,Spike_Conf\n")
        f.write(entry)


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



