# predict.py

import joblib
import os
import pandas as pd
from utils import get_feature_list, log_prediction_to_file, in_human_speak
from datetime import datetime
from utils import load_SPY_data as load_data, get_feature_list, add_features




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

        crash_confidence = class_probs[list(model.classes_).index(1)] if 1 in model.classes_ else 0
        spike_confidence = class_probs[list(model.classes_).index(2)] if 2 in model.classes_ else 0

        close_price = latest_row["Close"]





        timestamp = datetime.now()
        log_prediction_to_file(timestamp, prediction, crash_confidence, spike_confidence, close_price)

        timestamp = pd.Timestamp.now()


        print(f"🔮 Prediction: {prediction}")
        print(f"📊 Class Probabilities: {class_probs}")
        print(f"Normal: {class_probs[0]*100:.2f}%")
        print(f"Crash: {class_probs[1]*100:.2f}%")
        print(f"Spike: {class_probs[2]*100:.2f}%")

        #should work now
        print(f"Prediction in human speak: {in_human_speak(prediction, crash_confidence, spike_confidence)}")


        label = "CRASH" if prediction == 1 else "SPIKE" if prediction == 2 else "NORMAL"
        print(f"Live Prediction: {label} | Crash Confidence: {crash_confidence:.2f} | Spike Confidence: {spike_confidence:.2f}")

        return prediction, crash_confidence, spike_confidence

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, None
    
def run_predictions(confidence_threshold=0.80):
    import pandas as pd
    from utils import load_SPY_data as load_data, get_feature_list
    import joblib
    import os
    import datetime

    model_path = "models/market_crash_model.pkl"
    if not os.path.exists(model_path):
        print("🚫 Model file not found.")
        return None

    df = add_features(load_data())

    X = df[get_feature_list()]
    model = joblib.load(model_path)
    probs = model.predict_proba(X)

    predictions = []
    for i, prob in enumerate(probs):
        max_class = prob.argmax()
        confidence = prob[max_class]
        if confidence >= confidence_threshold:
            predictions.append({
                "Date": df.index[i].date(),
                "Timestamp": datetime.datetime.now(),
                "Prediction": max_class,
                "Crash_Conf": prob[1],
                "Spike_Conf": prob[2],
                "Close_Price": df["Close"].iloc[i],
                "Open": df["Open"].iloc[i],
                "Close": df["Close"].iloc[i]
            })

    if not predictions:
        return pd.DataFrame()  # return empty DataFrame

    return pd.DataFrame(predictions)

if __name__ == "__main__":
    from utils import load_SPY_data

    print("📥 Loading SPY data...")
    df = load_SPY_data()
    df = add_features(df)


    print("🔮 Running prediction on latest row...")
    live_predict(df)
