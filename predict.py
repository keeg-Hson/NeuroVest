# predict.py

import os
import joblib
import pandas as pd #says "pandas" is not used, but it is used in the function load_SPY_data
from datetime import datetime

from dotenv import load_dotenv
import pandas_ta as ta
from utils import (
    get_feature_list,
    log_prediction_to_file,
    in_human_speak,
    load_SPY_data as load_data,
    add_features,
    send_telegram_alert,
    notify_user,
    finalize_features
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
        #high = raw_latest["High"]
        #low = raw_latest["Low"]
        high = raw_latest.get("High", -1) if not pd.isna(raw_latest.get("High")) else -1
        low = raw_latest.get("Low", -1) if not pd.isna(raw_latest.get("Low")) else -1


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
    feature_df, feature_cols = add_features(raw_df)

    # Clean feature matrix for inference
    feature_df = finalize_features(feature_df, feature_cols)

    required = [c for c in feature_cols if c in feature_df.columns]
    feature_df.dropna(subset=required, inplace=True)




    model = joblib.load(MODEL_PATH)

    try:
        X = feature_df[model.feature_names_in_]
    except:
        X = feature_df[feature_cols]



    # Predict class labels and probabilities
    preds = model.predict(X)
    probs = model.predict_proba(X)

    # Map classes to columns
    class_indices = {label: idx for idx, label in enumerate(model.classes_)}

    # Extract confidences for spike (2) and crash (1)
    crash_conf = probs[:, class_indices.get(1, 0)]  # default to 0 if not found

    #if 2 not in class_indices:
    #    print("❌ Class 2 (spike) not found in model — aborting prediction.")
    #    return

    spike_conf = probs[:, class_indices.get(2, 0)]

    # Attach predictions
    
    feature_df["Prediction"] = preds
    feature_df["Crash_Conf"] = crash_conf
    feature_df["Spike_Conf"] = spike_conf
    feature_df["Confidence"] = probs.max(axis=1)

    # --- Robust Timestamp construction (works for Index or a 'Date' column) ---
    if isinstance(feature_df.index, pd.DatetimeIndex):
        ts = feature_df.index
    elif "Date" in feature_df.columns:
        ts = pd.to_datetime(feature_df["Date"], errors="coerce")
    else:
        ts = pd.to_datetime(feature_df.index, errors="coerce")

    ts = pd.to_datetime(ts, errors="coerce")
    # If ts is a Series, use .dt.tz_localize; if it's a DatetimeIndex, use .tz_localize
    if isinstance(ts, pd.Series):
        ts = ts.dt.tz_localize(None)
    else:
        ts = ts.tz_localize(None)

    feature_df["Timestamp"] = ts

    # Save and return
    feature_df.to_csv("logs/daily_predictions.csv", index=False)


    return feature_df



# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("📥 Loading SPY data...")
    spy_df = load_data()

    # 👇 add_features returns (df, feature_cols)
    feature_df, feature_cols = add_features(spy_df)

    # Clean feature matrix before live prediction
    feature_df = finalize_features(feature_df, feature_cols)

    required = [c for c in feature_cols if c in feature_df.columns]
    feature_df.dropna(subset=required, inplace=True)

    # Ensure Timestamp exists and is timezone-naive datetime
    if isinstance(feature_df.index, pd.DatetimeIndex):
        ts = feature_df.index
    elif "Date" in feature_df.columns:
        ts = pd.to_datetime(feature_df["Date"], errors="coerce")
    else:
        ts = pd.to_datetime(feature_df.index, errors="coerce")

    ts = pd.to_datetime(ts, errors="coerce")
    if isinstance(ts, pd.Series):
        ts = ts.dt.tz_localize(None)
    else:
        ts = ts.tz_localize(None)
    feature_df["Timestamp"] = ts

    print("🔮 Running prediction on latest row...")
    live_predict(feature_df, spy_df)
