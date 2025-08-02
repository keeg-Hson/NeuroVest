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

from utils import load_SPY_data, add_features, label_events_volatility_adjusted, get_feature_list

import warnings
warnings.filterwarnings("ignore", message=r"\[.*\] WARNING: .*Parameters: { \"use_label_encoder\" } are not used\.")


# Create output folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def train_best_xgboost_model(df):
    print("\n📊 Generating features...")


    df = add_features(df)
    from utils import label_events_volatility_adjusted
    
    print("✅ Type after add_features:", type(df))
    df = label_events_volatility_adjusted(df, window=3, vol_window=5, multiplier=1.5)


  

    

    #clean NaNs before SMOTE
    before = len(df)
    df = df.dropna(subset=get_feature_list() + ["Event"])
    after = len(df)
    print(f"🧹 Dropped {before - after} rows with NaNs before SMOTE.")

    # Check if we have enough data after cleaning
    X = df[get_feature_list()]
    y = df["Event"]

    print("\n📊 Original class distribution:")
    print(y.value_counts())


    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\n⚖️ After SMOTE:")
    print(pd.Series(y_resampled).value_counts())

    # TimeSeriesSplit for time-aware CV
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", use_label_encoder=False)

    print("\n🔍 Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_resampled, y_resampled)

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"🎯 Best Score (F1 Weighted): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # Evaluate on training data (optional sanity check)
    y_pred = best_model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted")
    rec = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    print("\n📈 Training Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save model
    model_path = "models/market_crash_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n💾 Best model saved to {model_path}")

    # Save grid results to CSV
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv("logs/gridsearch_xgb_results.csv", index=False)
    print(f"📊 Grid search results saved to logs/gridsearch_xgb_results.csv")

if __name__ == "__main__":
    print("📥 Loading SPY data...")
    df = load_SPY_data()
    train_best_xgboost_model(df)
    from predict import run_predictions
    run_predictions() #RUN BATCH PREDICTIONS AFTER TRAINING



