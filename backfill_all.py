"""
backfill_all.py — build full-history logs/labeled_predictions.csv from SPY.csv
using the currently trained model (either a plain sklearn estimator, or a dict
with {"model": clf, "features": [...]}).

It reconstructs a simple, consistent feature set:
ret_1, ret_5, sma_5, sma_20, vol_10, rsi14
unless the model provides its own feature list.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LOGS_DIR, MODELS_DIR, PREDICT_CFG, SPY_DAILY_CSV, TRAIN_CFG


def load_model_any(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        clf = obj.get("model", obj.get("clf"))
        feats = obj.get("features")
        if clf is None:
            raise ValueError(f"{path} dict missing 'model'/'clf'.")
        return clf, feats
    return obj, None


def build_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    px = df["Close"].astype(float)

    feat = {}
    feat["ret_1"] = px.pct_change(1, fill_method=None)
    feat["ret_5"] = px.pct_change(5, fill_method=None)
    feat["sma_5"] = px.rolling(5, min_periods=5).mean() / px - 1.0
    feat["sma_20"] = px.rolling(20, min_periods=20).mean() / px - 1.0
    feat["vol_10"] = px.pct_change(fill_method=None).rolling(10, min_periods=10).std()

    d = px.diff()
    up = (d.clip(lower=0)).rolling(14, min_periods=14).mean()
    down = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / (down.replace(0, 1e-12))
    feat["rsi14"] = 100 - (100 / (1 + rs))

    out = df[["Date", "Close"]].copy()
    for k, v in feat.items():
        out[k] = v
    return out


def build_labels(df_feat: pd.DataFrame) -> pd.Series:
    h = int(TRAIN_CFG.get("horizon", 5))
    friction = (
        float(TRAIN_CFG.get("fee_bps", 1.5)) + float(TRAIN_CFG.get("slippage_bps", 2.0))
    ) / 10000.0
    px = df_feat["Close"].astype(float)
    fwd_ret = px.shift(-h) / px - 1.0 - friction
    return (fwd_ret > 0).astype(float)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)

    feat_df = build_simple_features(raw)
    label = build_labels(feat_df)

    model_path = MODELS_DIR / "market_crash_model.pkl"
    model, trained_feats = load_model_any(model_path)
    use_cols = trained_feats or ["ret_1", "ret_5", "sma_5", "sma_20", "vol_10", "rsi14"]

    full = feat_df[["Date"] + use_cols].copy()
    full["Label"] = label

    # align X with y
    full = full.dropna(subset=use_cols).reset_index(drop=True)
    X = full[use_cols].astype(float)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        dfc = model.decision_function(X).astype(float)
        lo, hi = float(np.min(dfc)), float(np.max(dfc))
        proba = (dfc - lo) / (hi - lo + 1e-12)
    else:
        pred_plain = model.predict(X).astype(int)
        proba = pred_plain * 0.5 + 0.5 * (1 - pred_plain)

    p_min = float(PREDICT_CFG.get("p_min", 0.55))
    pred = (proba >= p_min).astype(int)

    out = (
        pd.DataFrame(
            {
                "Date": full["Date"],
                "Label": full["Label"],
                "Prediction": pred,
                "Proba": proba,
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    out_path = LOGS_DIR / "labeled_predictions.csv"
    out.to_csv(out_path, index=False)
    print(
        f"[backfill] wrote → {out_path}  rows={len(out)}  positives={int(out['Prediction'].sum())}"
    )


if __name__ == "__main__":
    main()
