# train.py

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    label_events_volatility_adjusted,
)

# Quiet some noisy warnings
warnings.filterwarnings(
    "ignore",
    message=r"\[.*\] WARNING: .*Parameters: { \"use_label_encoder\" } are not used\."
)
xgb.set_config(verbosity=0)

# Create output folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def _build_adaptive_cv(n_rows: int) -> TimeSeriesSplit:
    """Small-data-safe TimeSeriesSplit."""
    # ~3 folds on tiny sets, up to 5 on larger sets; keep a small gap.
    n_splits = min(5, max(2, n_rows // 200))
    gap = min(5, max(1, n_rows // 150))
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def _min_minority_per_fold(y: pd.Series, cv: TimeSeriesSplit) -> int:
    """Return the smallest minority-class count across all CV train folds."""
    min_count = np.inf
    idx = np.arange(len(y))
    for tr, _ in cv.split(idx):
        vc = y.iloc[tr].value_counts()
        if len(vc) < 2:
            # Only one class in this train fold → no SMOTE possible
            return 0
        min_count = min(min_count, int(vc.min()))
    return int(min_count if min_count != np.inf else 0)


def _safe_smote_from_fold(y: pd.Series, cv: TimeSeriesSplit):
    """
    Decide whether to use SMOTE based on minority size in folds.
    Returns (use_smote: bool, smote_step_or_passthrough).
    """
    m = _min_minority_per_fold(y, cv)
    if m < 2:
        print(f"ℹ️ SMOTE disabled (min minority per fold = {m}).")
        return False, "passthrough"
    # SMOTE requires k_neighbors <= minority_count - 1
    k = max(1, min(5, m - 1))
    print(f"ℹ️ SMOTE enabled with k_neighbors={k} (min minority per fold={m}).")
    return True, SMOTE(random_state=42, k_neighbors=k)


def train_best_xgboost_model(df: pd.DataFrame) -> bool:
    print("\n📊 Generating features...")
    df, all_feature_cols = add_features(df)

    # --- Optional: read top_signals.txt produced by your selector ---
    try:
        with open("logs/top_signals.txt", "r") as f:
            raw_top = [
                line.strip().split(",")[0]
                for line in f.readlines()
                if line.strip() and not line.startswith("Top")
            ]
        print(f"✅ Loaded top signals: {raw_top}")
    except FileNotFoundError:
        print("⚠️ logs/top_signals.txt not found. Will fall back to generated feature list.")
        raw_top = []

    # --- Dynamic coverage thresholds (scale with data size) ---
    N = len(df)
    MIN_VALID_ROWS = max(5, min(60, int(N * 0.40)))
    FORCE_MIN_ROWS = max(5, min(50, int(N * 0.30)))

    top_signals = [
        s for s in raw_top
        if s in df.columns and df[s].notna().sum() >= MIN_VALID_ROWS
    ]
    print(f"✅ Filtered top signals with data: {top_signals}")

    feature_cols = [c for c in all_feature_cols if c in top_signals]
    print(f"🧪 Using top correlated signals only: {feature_cols}")

    # ---- Force-include a few external signals (if sufficiently covered) ----
    forced_externals = [
        "Sector_MedianRet_20", "Sector_Dispersion_20",
        "Credit_Spread_20", "TNX_Change_20", "DXY_Change_20",
        "News_Sent_Z20", "Reddit_Sent_Z20",
    ]
    forced_available = [
        c for c in forced_externals
        if c in df.columns and df[c].notna().sum() >= FORCE_MIN_ROWS
    ]

    if not feature_cols:
        print("⚠️ No top_signals survived coverage filter — fallback to all_feature_cols + forced.")
        feature_cols = [
            c for c in all_feature_cols
            if c in df.columns and df[c].notna().sum() >= MIN_VALID_ROWS
        ]

    feature_cols = feature_cols + [c for c in forced_available if c not in feature_cols]

    if not feature_cols:
        minimal_base = [
            "Daily_Return", "Return_Lag1", "Return_Lag3", "Return_Lag5",
            "ZMomentum", "RSI", "MACD", "MACD_Signal", "Stoch_K", "Stoch_D",
            "Gap_Pct", "Acceleration",
        ]
        feature_cols = [
            c for c in minimal_base
            if c in df.columns and df[c].notna().sum() >= 5
        ]

    if not feature_cols:
        raise RuntimeError(
            "No features available after dynamic fallback. "
            "Refresh data to increase history, or loosen coverage thresholds."
        )

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"➕ Force-available externals: {forced_available}")
    print(f"🧪 Final candidate features ({len(feature_cols)}): "
          f"{feature_cols[:20]}{'...' if len(feature_cols) > 20 else ''}")

    # --- CLEAN features BEFORE labeling/splitting ---
    df = finalize_features(df, feature_cols)

    # Volatility (for logging only)
    df["Volatility"] = df["Close"].rolling(window=20).std()
    print("\n🧪 Sample volatility (tail):")
    print(df["Volatility"].dropna().tail(10))

    # Label AFTER features are cleaned
    df = label_events_volatility_adjusted(df, window=3, vol_window=10, multiplier=0.2)

    # Label sanity checks
    print(df[["Date", "Close", "Event"]].tail(15))
    print("\n📊 Distribution of Event labels (incl. NaNs):")
    print(df["Event"].value_counts(dropna=False))
    print("\n📊 Number of unique Events:")
    print(df["Event"].nunique(dropna=False))

    if df["Event"].nunique() <= 1:
        print("❌ Not enough class diversity in Event labels — training aborted.")
        return False

    # Drop rows missing Event or required features
    print("\n🧪 Missing values per feature column (post-clean):")
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

    if X.shape[0] == 0:
        raise RuntimeError("Empty training matrix after cleaning.")
    if X.shape[1] == 0:
        raise RuntimeError("No features selected after filtering.")

    print("\n📊 Original class distribution:")
    print(y.value_counts())

    # ---- Adaptive CV & SMOTE safety ----
    tscv_local = _build_adaptive_cv(len(X))
    use_smote, smote_step = _safe_smote_from_fold(y, tscv_local)

    # ---- Pipeline (skip KBest when too few features) ----
    use_kbest = X.shape[1] >= 2
    if use_kbest:
        max_k = X.shape[1]
        k_choices = sorted(set([1, 2, 3, 5, 8, 10, 12, max(1, max_k // 2)]))
        k_choices = [k for k in k_choices if 1 <= k <= max_k]

        steps = [
            ("smote", smote_step),
            ("kbest", SelectKBest(score_func=f_classif)),
            ("clf", XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )),
        ]
        pipe = Pipeline(steps=steps)

        param_grid = {
            "kbest__k": k_choices,
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    else:
        steps = [
            ("smote", smote_step),
            ("clf", XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )),
        ]
        pipe = Pipeline(steps=steps)

        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }

    

    print("\n🔍 Starting Grid Search (time-series CV)...")
    
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=tscv_local,
        n_jobs=-1,
        verbose=1,
        error_score=0,  # ignore fold errors instead of crashing
    )
    grid_search.fit(X, y)

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"🎯 Best Score (F1 Weighted): {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_

    # ---- ALWAYS define selected_cols safely (after best_model exists) ----
    selected_cols = list(X.columns)  # default fallback
    try:
        kb = None
        if hasattr(best_model, "named_steps"):
            kb = best_model.named_steps.get("kbest", None)
        if kb is not None and hasattr(kb, "get_support"):
            mask = kb.get_support()
            if hasattr(mask, "__len__") and len(mask) == X.shape[1]:
                selected_cols = list(X.columns[mask])
            else:
                print("⚠️ KBest mask shape mismatch; using all input columns.")
    except Exception as e:
        print(f"⚠️ Could not extract KBest-selected columns ({e}); using all input columns.")

    # Save *only* the selected features list (no index/header to avoid stray '0')
    os.makedirs("models", exist_ok=True)
    pd.Series(selected_cols, dtype=str).to_csv(
        "models/selected_features.txt", index=False, header=False
    )
    print(f"💾 Selected feature columns saved to models/selected_features.txt ({len(selected_cols)} cols)")

    # ---- Training-set metrics (sanity only) ----
    y_pred = best_model.predict(X)
    print("\n📈 Training Evaluation Metrics:")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, average='weighted'):.4f}")

    # ---- Persist artifacts ----
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
    try:
        success = train_best_xgboost_model(df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(f"[train] hard failure: {e}")

    if success:
        from predict import run_predictions
        run_predictions()
