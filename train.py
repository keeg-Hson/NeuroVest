# train.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import get_feature_list

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

LABELED_LOG_FILE = "logs/labeled_predictions.csv"

def train_model(df, features=None, target="Event"):
    if features is None:
        features = get_feature_list()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('\n📊 Model Performance:')
    print(classification_report(y_test, y_pred))
    print(f"Features used for training: {features}")

    joblib.dump(model, "models/market_crash_model.pkl")
    print("✅ Model trained and saved as 'models/market_crash_model.pkl'")
    return model


def build_retraining_dataset(df):
    labeled_log_path = "logs/labeled_predictions.csv"
    
    if not os.path.exists(labeled_log_path):
        print("[⚠️] Labeled predictions file not found. Skipping retraining dataset build.")
        return None, None

    labeled_df = pd.read_csv(labeled_log_path, parse_dates=["Timestamp"])
    labeled_df.set_index("Timestamp", inplace=True)

    df_features = df.copy()
    df_features.index = pd.to_datetime(df_features.index, errors='coerce')

    merged = df_features.join(labeled_df["Actual_Event"], how="inner")
    merged.dropna(subset=["Actual_Event"], inplace=True)

    if merged.empty:
        print("[⚠️] No overlapping timestamps between features and labeled log — skipping retraining.")
        return None, None

    X = merged[get_feature_list()]
    y = merged["Actual_Event"].astype(int)

    return X, y



def retrain_model(df, model_path='models/market_crash_model.pkl'):
    X, y = build_retraining_dataset(df)
    if X is None or X.empty:
        print("[⚠️] Skipping model retraining due to insufficient data.")
        return

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X, y)

    joblib.dump(clf, model_path)
    print(f"✅ Model retrained and saved to {model_path}")
