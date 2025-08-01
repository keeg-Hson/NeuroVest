# predict.py

import os
import joblib
import pandas as pd #says "pandas" is not used, but it is used in the function load_SPY_data
from datetime import datetime
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

# Load .env variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Model path
MODEL_PATH = "models/market_crash_model.pkl"

# --- LIVE PREDICT ---
def live_predict(feature_df, raw_df, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print("❌ Model file not found.")
        return None, None, None

    model = joblib.load(model_path)

    try:
        features = get_feature_list()
        latest_row = feature_df.iloc[-1]
        print("[DEBUG] raw_df columns:", raw_df.columns.tolist())

        raw_latest = raw_df.iloc[-1]

        latest_features = latest_row[features].to_frame().T

        prediction = model.predict(latest_features)[0]
        class_probs = model.predict_proba(latest_features)[0]

        if max(class_probs) < 0.6:
            print("⚠️ Low-confidence prediction — consider ignoring this signal.")

        crash_confidence = class_probs[list(model.classes_).index(1)] if 1 in model.classes_ else 0
        spike_confidence = class_probs[list(model.classes_).index(2)] if 2 in model.classes_ else 0

        close_price = raw_latest["Close"]
        open_price = raw_latest["Open"]
        high = raw_latest["High"]
        low = raw_latest["Low"]

        timestamp = datetime.now()
        log_prediction_to_file(
            timestamp,
            prediction,
            crash_confidence,
            spike_confidence,
            close_price,
            open_price,
            high,
            low
        )

        print(f"🔮 Prediction: {prediction}")
        print(f"📊 Class Probabilities: {class_probs}")
        print(f"Normal: {class_probs[0]*100:.2f}%")
        print(f"Crash: {class_probs[1]*100:.2f}%")
        print(f"Spike: {class_probs[2]*100:.2f}%")
        print(f"Prediction Forecast: {in_human_speak(prediction, crash_confidence, spike_confidence)}")

        notify_user(prediction, crash_confidence, spike_confidence)

        label = "NORMAL"
        if prediction == 1 and crash_confidence >= 0.7:
            label = "CRASH"
        elif prediction == 2 and spike_confidence >= 0.7:
            label = "SPIKE"

        if label != "NORMAL":
            msg = (
                f"🚨 *Market Alert* — {label} signal detected!\n\n"
                f"📉 Crash: `{crash_confidence:.2f}`\n"
                f"📈 Spike: `{spike_confidence:.2f}`\n"
                f"🔎 *Prediction*: `{label}`"
            )
            send_telegram_alert(msg, token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)

        return prediction, crash_confidence, spike_confidence

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, None


# --- BATCH PREDICTION ---
def run_predictions(confidence_threshold=0.80):
    raw_df = load_data()
    feature_df = add_features(raw_df)
    model = joblib.load(MODEL_PATH)
    X = feature_df[model.feature_names_in_]
    preds = model.predict(X)

    feature_df["Prediction"] = preds
    feature_df["Timestamp"] = feature_df.index

    feature_df.to_csv("logs/daily_predictions.csv", index=False)
    return feature_df


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("📥 Loading SPY data...")
    spy_df = load_data() #line 118
    feature_df = add_features(spy_df)

    print("🔮 Running prediction on latest row...")
    live_predict(feature_df, spy_df)
