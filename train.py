# train.py

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from utils import (
    load_SPY_data, 
    add_features, 
    finalize_features, 
    label_events_volatility_adjusted, 
    get_feature_list, 
    label_events_simple
)

import warnings
warnings.filterwarnings("ignore", message=r"\[.*\] WARNING: .*Parameters: { \"use_label_encoder\" } are not used\.")


# Create output folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def train_best_xgboost_model(df):
    print("\n📊 Generating features...")
    df, all_feature_cols = add_features(df)

    # ── Load Top Signals (with basic sanity filter) ─────────────────────────────
    try:
        with open("logs/top_signals.txt", "r") as f:
            top_lines = f.readlines()
            top_signals = [
                line.strip().split(",")[0]
                for line in top_lines
                if line.strip() and not line.startswith("Top")
            ]
        print(f"✅ Loaded top signals: {top_signals}")

        MIN_VALID_ROWS = 100  # tune if needed
        top_signals = [
            sig for sig in top_signals
            if sig in df.columns and df[sig].notna().sum() > MIN_VALID_ROWS
        ]
        print(f"✅ Filtered top signals with data: {top_signals}")
    except FileNotFoundError:
        print("⚠️ logs/top_signals.txt not found. Using all available features instead.")
        top_signals = all_feature_cols

    # ── Pick feature set, then CLEAN features BEFORE labeling ──────────────────
    feature_cols = [c for c in all_feature_cols if c in top_signals]
    print(f"🧪 Using top correlated signals only: {feature_cols}")

    # ✅ Ensure no NaNs remain in these features BEFORE labeling/splitting
    df = finalize_features(df, feature_cols)

    # ── Volatility (not required to be in feature_cols) ────────────────────────
    vol_window = 20
    df["Volatility"] = df["Close"].rolling(window=vol_window).std()
    print("\n🧪 Sample volatility values (after rolling std applied):")
    print(df["Volatility"].dropna().tail(10))

    print("✅ Type after add_features:", type(df))

    # ── Label AFTER features are cleaned ───────────────────────────────────────
    df = label_events_volatility_adjusted(df, window=3, vol_window=10, multiplier=0.2)

    # ── Basic label sanity checks ──────────────────────────────────────────────
    print(df[["Date", "Close", "Event"]].tail(15))
    print("\n📊 Distribution of Event labels (incl. NaNs):")
    print(df["Event"].value_counts(dropna=False))
    print("\n📊 Number of unique Events:")
    print(df["Event"].nunique(dropna=False))

    if df["Event"].nunique() <= 1:
        print("❌ Not enough class diversity in Event labels — training aborted.")
        return False

    # ── Drop rows missing Event or all features ────────────────────────────────
    print("\n🧪 Missing values per feature column:")
    print(df[feature_cols].isna().sum())
    print(f"\n🧪 Total rows before dropna: {len(df)}")

    valid_feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]
    required_cols = ["Event"] + valid_feature_cols
    df = df.dropna(subset=required_cols)

    print(f"\n🧹 Rows remaining after dropna: {len(df)}")
    if len(df) == 0:
        print("❌ No data left after dropping NaNs. Check signal columns or event labeling.")
        return False

    X = df[valid_feature_cols]
    y = df["Event"]

    print("\n📊 Original class distribution:")
    print(y.value_counts())

    # ── Balance classes ────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("\n⚖️ After SMOTE:")
    print(pd.Series(y_resampled).value_counts())

    # ── Model + Grid ──────────────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", use_label_encoder=False)

    print("\n🔍 Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_resampled, y_resampled)

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"🎯 Best Score (F1 Weighted): {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_

    # ── Training-set sanity metrics ────────────────────────────────────────────
    y_pred = best_model.predict(X)
    print("\n📈 Training Evaluation Metrics:")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, average='weighted'):.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path = "models/market_crash_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n💾 Best model saved to {model_path}")

    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv("logs/gridsearch_xgb_results.csv", index=False)
    print("📊 Grid search results saved to logs/gridsearch_xgb_results.csv")

    return True



if __name__ == "__main__":
    print("📥 Loading SPY data...")
    df = load_SPY_data()

    success = train_best_xgboost_model(df)

    if success:
        from predict import run_predictions
        run_predictions()




