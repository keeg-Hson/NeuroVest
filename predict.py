# predict.py

import joblib
import os
import pandas as pd
from utils import get_feature_list, log_prediction_to_file, in_human_speak

def live_predict(df, model_path='models/market_crash_model.pkl'):
    if not os.path.exists(model_path):
        print("❌ Model file not found.")
        return None, None, None

    model = joblib.load(model_path)

    try:
        features=get_feature_list()
        latest_row = df[features].iloc[-5:].mean().to_frame().T


        prediction=model.predict(latest_row[features])[0]
        class_probs=model.predict_proba(latest_row[features])[0]

        log_prediction_to_file(prediction, class_probs)

        print(f"🔮 Prediction: {prediction}")
        print(f"📊 Class Probabilities: {class_probs}")

        print(f"Normal: {class_probs[0]*100:.2f}%")
        print(f"Crash: {class_probs[1]*100:.2f}%")
        print(f"Spike: {class_probs[2]*100:.2f}%")
        print(f"Prediction in human speak: {in_human_speak(prediction)}")

        crash_confidence = class_probs[list(model.classes_).index(1)] if 1 in model.classes_ else 0 
        spike_confidence = class_probs[list(model.classes_).index(2)] if 2 in model.classes_ else 0

        label = "CRASH" if prediction == 1 else "SPIKE" if prediction == 2 else "NORMAL"
        print(f"Live Prediction: {label} | Crash Confidence: {crash_confidence:.2f} | Spike Confidence: {spike_confidence:.2f}")

        return prediction, crash_confidence, spike_confidence

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, None