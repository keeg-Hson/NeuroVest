#!/usr/bin/env python3
"""
train_from_labels.py

Trains a lightweight classifier on SPY using simple price-derived features:
ret_1, ret_5, sma_5, sma_20, vol_10, rsi14.

Design goals
- Silence dtype/fill FutureWarnings (low_memory=False, pct_change(fill_method=None)).
- Strict X/y alignment: drop rows with NaNs across features + label.
- Time-based split (75% train / 25% holdout, shuffle=False).
- Persist both model and its feature list for consistent downstream usage.
- Save split metadata so backtests can evaluate OOS only.

Outputs
- models/market_crash_model.pkl    -> {"model": sklearn estimator, "features": [...]}
- models/split_meta.json            -> {"split_index": int, "split_date": "YYYY-MM-DD"}
- stdout prints feature set, shapes, and holdout metrics.

Notes
- You can swap RandomForest for XGBoost (xgboost.XGBClassifier) later.
"""

from __future__ import annotations

import json
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from config import MODELS_DIR, SPY_DAILY_CSV, TRAIN_CFG


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    px = df["Close"].astype(float)

    feat = {}
    feat["ret_1"] = px.pct_change(1, fill_method=None)
    feat["ret_5"] = px.pct_change(5, fill_method=None)
    feat["sma_5"] = px.rolling(5, min_periods=5).mean() / px - 1.0
    feat["sma_20"] = px.rolling(20, min_periods=20).mean() / px - 1.0
    feat["vol_10"] = px.pct_change(fill_method=None).rolling(10, min_periods=10).std()

    # RSI(14)
    d = px.diff()
    up = d.clip(lower=0.0).rolling(14, min_periods=14).mean()
    down = (-d.clip(upper=0.0)).rolling(14, min_periods=14).mean().replace(0, 1e-12)
    rs = up / down
    feat["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    out = df[["Date", "Close"]].copy()
    for k, v in feat.items():
        out[k] = v
    return out


def _build_labels(feat_df: pd.DataFrame) -> pd.Series:
    """Binary label: 1 if next h-day return (after friction) > 0 else 0."""
    h = int(TRAIN_CFG.get("horizon", 5))
    fee_bps = float(TRAIN_CFG.get("fee_bps", 1.5))
    slip_bps = float(TRAIN_CFG.get("slippage_bps", 2.0))
    friction = (fee_bps + slip_bps) / 10_000.0

    px = feat_df["Close"].astype(float)
    fwd_ret = px.shift(-h) / px - 1.0 - friction
    return (fwd_ret > 0).astype(int)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_model() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Read with low_memory=False to avoid dtype fragmentation + warnings
    df = pd.read_csv(SPY_DAILY_CSV, low_memory=False)

    feat_df = _build_features(df)
    label = _build_labels(feat_df)

    feature_cols = ["ret_1", "ret_5", "sma_5", "sma_20", "vol_10", "rsi14"]

    # Align X and y
    full = feat_df[["Date"] + feature_cols].copy()
    full["Label"] = label
    full = full.dropna(subset=feature_cols + ["Label"]).reset_index(drop=True)

    X = full[feature_cols].astype(float)
    y = full["Label"].astype(int)

    print("\n[train] features:", feature_cols, "\n")
    print(f"[train] aligned shapes → X:{X.shape}, y:{y.shape}\n")

    # Time-based split: first 75% train, last 25% holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Simple RF baseline (fast & robust)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Holdout report
    y_pred = clf.predict(X_test)
    print("[train] holdout:\n", classification_report(y_test, y_pred, digits=3))

    # Split meta (start of test fold)
    split_idx = len(X_train)
    split_date = full.loc[split_idx, "Date"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "split_meta.json").write_text(
        json.dumps(
            {"split_index": int(split_idx), "split_date": str(pd.to_datetime(split_date).date())}
        )
    )
    print(f"[train] split_date (start of test): {split_date}")

    # Save model + feature schema for downstream code reuse
    out_path = MODELS_DIR / "market_crash_model.pkl"
    joblib.dump({"model": clf, "features": feature_cols}, out_path)
    print(f"[train] saved → {out_path}")


if __name__ == "__main__":
    train_model()
