# predict.py

import joblib
import os
import pandas as pd
from utils import get_feature_list, log_prediction_to_file, in_human_speak
from datetime import datetime
import requests
from dotenv import load_dotenv
import joblib
from dotenv import load_dotenv
from utils import (
    get_feature_list,
    log_prediction_to_file,
    in_human_speak,
    load_SPY_data as load_data,
    add_features,
    send_telegram_alert,
    notify_user
)

#load .env variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Model path
MODEL_PATH = "models/rf_trained.pkl"



spy_df = load_data()
feature_df = add_features(spy_df)



MODEL_PATH = "models/rf_trained.pkl"
model = joblib.load(MODEL_PATH)



def live_predict(df, model_path='models/market_crash_model.pkl'):
    if not os.path.exists(model_path):
        print("❌ Model file not found.")
        return None, None, None

    model = joblib.load(model_path)
    df_raw = df.copy()


    try:
        features = get_feature_list()
        latest_row = df.iloc[-1]  # get last full row with all columns
        latest_features = latest_row[features].to_frame().T  # for model input

        prediction = model.predict(latest_features)[0]
        class_probs = model.predict_proba(latest_features)[0]

        if max(class_probs) < 0.6:
            print("⚠️ Low-confidence prediction — consider ignoring this signal.")

            


        crash_confidence = class_probs[list(model.classes_).index(1)] if 1 in model.classes_ else 0
        spike_confidence = class_probs[list(model.classes_).index(2)] if 2 in model.classes_ else 0

        close_price = latest_row["Close"]





        timestamp = datetime.now()
        log_prediction_to_file(timestamp, prediction, crash_confidence, spike_confidence, close_price)

    


        print(f"🔮 Prediction: {prediction}")
        print(f"📊 Class Probabilities: {class_probs}")
        print(f"Normal: {class_probs[0]*100:.2f}%")
        print(f"Crash: {class_probs[1]*100:.2f}%")
        print(f"Spike: {class_probs[2]*100:.2f}%")

        #should work now
        print(f"Prediction Forecast: {in_human_speak(prediction, crash_confidence, spike_confidence)}")
        
        # Notify via terminal & system beep
        notify_user(prediction, crash_confidence, spike_confidence)

        # Optionally send Telegram alert if confidence is high
        label = "NORMAL"
        if prediction == 1 and crash_confidence >= 0.7:
            label = "CRASH"
        elif prediction == 2 and spike_confidence >= 0.7:
            label = "SPIKE"

        if label != "NORMAL":
            msg = f"🚨 *Market Alert* — {label} signal detected!\n\n📉 Crash: `{crash_confidence:.2f}`\n📈 Spike: `{spike_confidence:.2f}`\n🔎 *Prediction*: `{label}`"
            send_telegram_alert(msg, token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)

        return prediction, crash_confidence, spike_confidence

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, None
    
def run_predictions(confidence_threshold=0.80):
    spy_df = load_SPY_data()
    feature_df = add_features(spy_df)
    model = joblib.load("MODEL_PATH")
    X = feature_df[model.feature_names_in_]  # auto-align columns
    preds = model.predict(X)

    feature_df["Prediction"] = preds
    feature_df["Timestamp"] = feature_df.index  # for backtest compatibility


    feature_df.to_csv("logs/daily_predictions.csv", index=False)
    return feature_df


if __name__ == "__main__":
    from utils import load_SPY_data

    print("📥 Loading SPY data...")
    df = load_SPY_data()
    df = add_features(df)


    print("🔮 Running prediction on latest row...")
    live_predict(df)
