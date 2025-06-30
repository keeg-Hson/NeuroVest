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

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE  # Only if using SMOTE
import numpy as np  # Required for np.select

def generate_more_labels(df):
    df["Future_Close"] = df["Close"].shift(-1)
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df["Actual_Event"] = np.select(
        [df["Future_Return"] < -0.03, df["Future_Return"] > 0.03],
        [1, 2],
        default=0
    )
    return df


def cleanse_labeled_data(df):
    df = df.copy()

    # Drop rows where Future_Return is exactly 0
    df = df[df["Future_Return"] != 0]

    # Optionally: downsample 'Normal' class if needed
    df_normal = df[df["Actual_Event"] == 0]
    df_events = df[df["Actual_Event"] != 0]

    if len(df_events) > 0:
        df_normal_downsampled = df_normal.sample(n=len(df_events)*2, random_state=42)
        df = pd.concat([df_events, df_normal_downsampled])

    return df.sort_values("Timestamp")



def balance_dataset(X, y):
    df = X.copy()
    df['label'] = y

    df_minority = df[df['label'] != 0]
    df_majority = df[df['label'] == 0]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced.drop("label", axis=1), df_balanced["label"]



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
    
    # CLEANSE the merged training data
    merged = cleanse_labeled_data(merged)

    X = merged[get_feature_list()]
    y = merged["Actual_Event"].astype(int)

    return X, y



def retrain_model(df, model_path='models/market_crash_model.pkl'):
    X, y = build_retraining_dataset(df)

    # Load existing model if it exists
    if X is None or y is None or len(X) == 0:
        print("[⚠️] Skipping model retraining due to insufficient data.")
        return

    # Input integrity check
    if X.isnull().any().any():
        raise ValueError("[❌] NaNs detected in retraining features — aborting.")

    missing_cols = [col for col in get_feature_list() if col not in X.columns]
    if missing_cols:
        raise ValueError(f"[❌] Missing required columns in retraining dataset: {missing_cols}")
    

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced") #if x in

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    clf.fit(X_resampled, y_resampled)


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

    print("\n📊 Class distribution in retraining set:")
    print(y.value_counts())


    # Save and plot
    joblib.dump(clf, model_path)
    print(f"✅ Model retrained and saved to {model_path}")
    plot_model_performance()
