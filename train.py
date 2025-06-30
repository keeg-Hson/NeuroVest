# train.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import get_feature_list
from datetime import datetime

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

LABELED_LOG_FILE = "logs/labeled_predictions.csv"

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import get_feature_list
from viz import plot_model_performance


def train_model(df, features=None, target="Event"):
    if features is None:
        features = get_feature_list()

    X = df[features]
    y = df[target]

    # Sanity check
    for col in features:
        if col not in df.columns:
            raise ValueError(f"[❌] Missing expected feature column: {col}")
    
    if df[features].isnull().any().any():
        raise ValueError("[❌] NaNs found in input features — check preprocessing pipeline.")


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

    if X is None or y is None or len(X) == 0:
        print("[⚠️] Skipping model retraining due to insufficient data.")
        return

    # Input integrity check
    if X.isnull().any().any():
        raise ValueError("[❌] NaNs detected in retraining features — aborting.")

    missing_cols = [col for col in get_feature_list() if col not in X.columns]
    if missing_cols:
        raise ValueError(f"[❌] Missing required columns in retraining dataset: {missing_cols}")
    
    # Load existing model if it exists
    if X is None or y is None or len(X) == 0:
        print("[⚠️] Skipping model retraining due to insufficient data.")
        return

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X, y)

    # Evaluate
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    # Log performance
    performance_log_path = "logs/model_performance.csv"
    os.makedirs("logs", exist_ok=True)
    is_new = not os.path.exists(performance_log_path)
    with open(performance_log_path, "a") as f:
        if is_new:
            f.write("Date,Accuracy,Precision,Recall,F1\n")
        f.write(f"{datetime.now().date()},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")

    print(f"[📊] Retraining metrics — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Save and plot
    joblib.dump(clf, model_path)
    print(f"✅ Model retrained and saved to {model_path}")
    plot_model_performance()
