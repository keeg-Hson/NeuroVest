#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LOGS_DIR, MODELS_DIR, PREDICT_CFG, SPY_DAILY_CSV


# -----------------------------------------------------------------------------
# Feature engineering (MUST match train_from_labels.py)
# -----------------------------------------------------------------------------
def _nv_simple_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal daily features matching the trainer:
    ret_1, ret_5, sma_5, sma_20, vol_10, rsi14 (0-100 scale).
    """
    df = raw_df.copy()

    # Date
    if "Date" not in df.columns:
        for c in list(df.columns)[:5]:
            if str(c).lower() == "date":
                df = df.rename(columns={c: "Date"})
                break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Price column — prefer Close/AdjClose (exact names first)
    cands = [c for c in df.columns if str(c).lower() in ("close", "adjclose", "adj close")]
    if not cands:
        cands = [c for c in df.columns if "close" in str(c).lower()]
    if not cands:
        raise SystemExit("No price column found (Close/AdjClose).")
    px = df[cands[0]].astype(float)

    out = pd.DataFrame({"Date": df["Date"]})

    # Match trainer's formulas & warning-safe fill_method=None
    out["ret_1"] = px.pct_change(1, fill_method=None)
    out["ret_5"] = px.pct_change(5, fill_method=None)
    out["sma_5"] = px.rolling(5, min_periods=5).mean() / px - 1.0
    out["sma_20"] = px.rolling(20, min_periods=20).mean() / px - 1.0
    out["vol_10"] = px.pct_change(fill_method=None).rolling(10, min_periods=10).std()

    d = px.diff()
    up = d.clip(lower=0.0).rolling(14, min_periods=14).mean()
    down = (-d.clip(upper=0.0)).rolling(14, min_periods=14).mean().replace(0, 1e-12)
    rs = up / down
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))  # 0–100, like trainer

    return out


def _load_model_and_features(path: Path):
    obj = joblib.load(path)
    # Newer save format: dict with model + features
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], list(obj.get("features", []))
    # Legacy: bare estimator
    return obj, []


def _align_features(feat_df: pd.DataFrame, saved_feats: list[str]) -> pd.DataFrame:
    """
    Ensure columns/ordering match the training schema.
    Missing features get 0.0; extras are dropped.
    Keeps Date column if present.
    """
    df = feat_df.copy()
    if not saved_feats:
        # Fallback: keep numeric cols only (drop Date)
        return df.drop(columns=["Date"], errors="ignore").select_dtypes(include=[np.number])

    for c in saved_feats:
        if c not in df.columns:
            df[c] = 0.0

    cols = (["Date"] + saved_feats) if "Date" in df.columns else saved_feats
    return df[cols]


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def _score_latest(model, X: pd.DataFrame) -> tuple[int, float, pd.Timestamp]:
    # Grab last row, drop Date, cast to float
    X_last = X.tail(1).drop(columns=["Date"], errors="ignore").astype(float)

    # Predict probability for class 1 (long)
    if hasattr(model, "predict_proba"):
        class_probs = model.predict_proba(X_last)[0]
        classes_enc = list(getattr(model, "classes_", [0, 1]))
        proba_map = {int(k): float(v) for k, v in zip(classes_enc, class_probs, strict=False)}
        p1 = proba_map.get(1, class_probs[-1])
    else:
        # if no proba, degrade gracefully using class label as a pseudo-prob
        pred_plain = int(model.predict(X_last)[0])
        p1 = 0.7 if pred_plain == 1 else 0.3

    thresholds = _load_thresholds()
    decision = 1 if p1 >= float(thresholds.get("p_min", 0.55)) else 0

    # Timestamp
    ts = None
    if "Date" in X.columns and X["Date"].notna().any():
        ts = pd.to_datetime(X["Date"].dropna().iloc[-1])
    else:
        ts = pd.Timestamp("now").normalize()

    return decision, p1, ts


def _load_thresholds() -> dict:
    try:
        with open(MODELS_DIR / "thresholds.json") as f:
            return json.load(f)
    except Exception:
        return {
            "p_min": float(PREDICT_CFG.get("p_min", 0.55)),
            "ev_min": float(PREDICT_CFG.get("ev_min", 0.0005)),
        }


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _append_single(decision: int, p1: float, when: pd.Timestamp) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "labeled_predictions.csv"

    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
    else:
        df = pd.DataFrame(columns=["Date", "Label", "Pred", "Prediction", "Proba"])

    # Update/append
    row_mask = df["Date"] == when
    if row_mask.any():
        df.loc[row_mask, ["Pred", "Prediction", "Proba"]] = [decision, decision, p1]
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Date": when,
                            "Label": pd.NA,
                            "Pred": decision,  # legacy field some tools expect
                            "Prediction": decision,  # canonical name going forward
                            "Proba": p1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[predict] wrote → {path}")


def _backfill_full(model, saved_feats: list[str]) -> None:
    """Score all dates and (re)write labeled_predictions.csv, preserving any existing Label values."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "labeled_predictions.csv"

    # Load raw
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Build + align
    feat = _nv_simple_features(raw)
    X = _align_features(feat, saved_feats)

    # Drop rows with any NaNs in model features (avoid training-time NaN issues)
    if "Date" in X.columns:
        full = X.dropna(subset=[c for c in X.columns if c != "Date"]).copy()
    else:
        full = X.dropna(axis=0).copy()

    # Score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(full.drop(columns=["Date"], errors="ignore").astype(float))
        # Map to class=1
        classes_enc = list(getattr(model, "classes_", [0, 1]))
        idx1 = classes_enc.index(1) if 1 in classes_enc else (len(classes_enc) - 1)
        p1 = probs[:, idx1]
    else:
        preds = model.predict(full.drop(columns=["Date"], errors="ignore").astype(float)).astype(
            int
        )
        p1 = preds * 0.7 + (1 - preds) * 0.3

    thresholds = _load_thresholds()
    pred = (p1 >= float(thresholds.get("p_min", 0.55))).astype(int)

    out = (
        pd.DataFrame(
            {
                "Date": full["Date"],
                "Prediction": pred,
                "Pred": pred,  # legacy
                "Proba": p1,
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Preserve existing Label if present
    if out_path.exists():
        prev = pd.read_csv(out_path, parse_dates=["Date"])
        out = out.merge(prev[["Date", "Label"]], on="Date", how="left")

    out.to_csv(out_path, index=False)
    print(
        f"[backfill] wrote → {out_path}  rows={len(out)}  positives={int(out['Prediction'].sum())}"
    )


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------
def live_predict() -> tuple[int, float, pd.Timestamp]:
    """Return (decision, p(long=1), timestamp)."""
    # Load raw SPY prices
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Build features and align to training schema
    feat = _nv_simple_features(raw)

    model_path = MODELS_DIR / "market_crash_model.pkl"
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    model, saved_feats = _load_model_and_features(model_path)
    X = _align_features(feat, saved_feats)

    return _score_latest(model, X)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroVest live prediction / backfill")
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Score full history and rewrite labeled_predictions.csv",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    model_path = MODELS_DIR / "market_crash_model.pkl"
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    model, saved_feats = _load_model_and_features(model_path)

    if args.backfill:
        _backfill_full(model, saved_feats)
        return 0

    # Single live prediction
    pred, prob, when = live_predict()
    print(f"[predict] {when.date()}  p(long=1)={prob:.4f}  decision={pred}")
    _append_single(pred, prob, when)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
