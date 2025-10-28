# live_loop.py
# Continuously runs live predictions and updates every few minutes

# RUN IN BACKGROUND: BASH -> "nohup python3 live_loop.py &"


import time
from datetime import datetime

from predict import live_predict
from utils import add_features, load_SPY_data, notify_user

INTERVAL_MINUTES = 5  # Change to however often you want

print(f"🔁 Starting live prediction loop every {INTERVAL_MINUTES} minutes...")

while True:
    try:
        # Load and feature-engineer SPY data
        df = load_SPY_data()
        df = add_features(df)

        print("\n⏱️ Running live prediction...")
        prediction, crash_conf, spike_conf = live_predict(df)

        prediction, crash_conf, spike_conf = live_predict(df)
        notify_user(prediction, crash_conf, spike_conf)

        # Write to rolling daily log file
        log_entry = f"{datetime.now()} | Prediction: {prediction}, Crash Conf: {crash_conf:.2f}, Spike Conf: {spike_conf:.2f}\n"
        log_filename = f"logs/prediction_log_{datetime.now().date()}.txt"
        with open(log_filename, "a") as f:
            f.write(log_entry)

    except Exception as e:
        error_log = f"{datetime.now()} | ❌ Error: {e}\n"
        print(error_log)
        with open(f"logs/prediction_errors_{datetime.now().date()}.txt", "a") as f:
            f.write(error_log)

    time.sleep(INTERVAL_MINUTES * 60)
