#!/usr/bin/env python3
"""
NeuroVest — prediction runner (backfill + live)

Purpose
-------
Loads a trained model, rebuilds the same features used during training, scores
probabilities, applies a decision threshold, and writes predictions to logs.

Label convention
----------------
Unified 3-class convention:
    0 = SPIKE
    1 = NORMAL
    2 = CRASH

The active forward-returns model is binary internally ({0,1} = no-trade/trade).
To remain compatible with 3-class outputs, binary predictions are mapped to:
    0 → 1 (NORMAL)
    1 → 0 (SPIKE)
CRASH (2) is reserved for explicit crash labels and is not emitted by the binary model.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LOGS_DIR, MODELS_DIR, PREDICT_CFG, SPY_DAILY_CSV
from utils import add_features, finalize_features  # must match training pipeline


# =============================================================================
# Public API
# =============================================================================
def run_predictions(backfill: bool = True) -> None:
    """
    When backfill=True, scores full history and rewrites logs/labeled_predictions.csv.
    Otherwise, appends a single live row.
    """
    model_path, variant = _resolve_model_path()
    model, saved_feats = _load_model_and_features(model_path, variant)
    _assert_binary_model(model)
    if backfill:
        _backfill_full(model, saved_feats, variant)
    else:
        pred_bin, prob, when = live_predict()
        pred_012 = _binary_to_legacy(pred_bin)
        _append_single(pred_012, prob, when)


# =============================================================================
# Resolve variant/model/thresholds
# =============================================================================
def _resolve_model_path() -> tuple[Path, str]:
    variant = os.getenv("PREDICT_VARIANT", "forward_returns").strip().lower()
    if variant.startswith("forward"):
        return (MODELS_DIR / "market_crash_model_fwd.pkl"), "forward"
    return (MODELS_DIR / "market_crash_model.pkl"), "generic"


def _load_model_and_features(path: Path, variant: str) -> tuple[object, list[str]]:
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    obj = joblib.load(path)
    # Newer save format: dict with model + features
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], list(obj.get("features", []))
    # Legacy: bare estimator; fallback to saved feature list file
    ftxt = MODELS_DIR / ("input_features_fwd.txt" if variant == "forward" else "input_features.txt")
    saved_feats = []
    if ftxt.exists():
        saved_feats = [ln.strip() for ln in ftxt.read_text().splitlines() if ln.strip()]
    return obj, saved_feats


def _load_thresholds(variant: str) -> dict[str, float]:
    """
    Threshold precedence (first found wins):
      THRESH_PATH (env), models/thresholds_fwd.json, configs/best_thresholds.json, models/thresholds.json.
    """
    env_path = os.getenv("THRESH_PATH")
    if env_path and os.path.exists(env_path):
        try:
            with open(env_path) as f:
                obj = json.load(f)
            t = obj.get("threshold") or obj.get("spike_thresh") or obj.get("p_min")
            inv = bool(obj.get("invert_proba", False))
            if t is not None:
                t = float(t)
                t = max(0.10, min(0.90, t))
                print(f"[debug] using threshold {t:.6f} from THRESH_PATH={env_path}")
                print(f"[debug] invert_proba={inv}")
                return {"p_min": t, "invert_proba": inv}
        except Exception as e:
            print(f"[warn] failed to read THRESH_PATH={env_path}: {e}")

    fallbacks = [
        MODELS_DIR / "thresholds_fwd.json",
        Path("configs/best_thresholds.json"),
        MODELS_DIR / "thresholds.json",
    ]
    for p in fallbacks:
        try:
            with open(p) as f:
                obj = json.load(f)
            t = obj.get("threshold") or obj.get("spike_thresh") or obj.get("p_min")
            inv = bool(obj.get("invert_proba", False))
            if t is not None:
                t = float(t)
                t = max(0.10, min(0.90, t))
                print(f"[debug] using threshold {t:.6f} from {p}")
                print(f"[debug] invert_proba={inv}")
                return {"p_min": t, "invert_proba": inv}
        except Exception:
            pass

    t = float(PREDICT_CFG.get("p_min", 0.55))
    t = max(0.10, min(0.90, t))
    print(f"[debug] using default threshold {t:.6f} (PREDICT_CFG/default)")
    print("[debug] invert_proba=False")
    return {"p_min": t, "invert_proba": False}


# =============================================================================
# Feature engineering — identical to training
# =============================================================================
def _build_inference_features_from_prices(
    raw_df: pd.DataFrame, saved_feats: list[str]
) -> pd.DataFrame:
    """
    Uses utils.add_features + utils.finalize_features, then aligns to the saved schema.
    Carries 'Date' through for logging.
    """
    df_feat, all_cols = add_features(raw_df)
    cols = saved_feats if saved_feats else all_cols
    df_feat = finalize_features(df_feat, cols)
    if "Date" not in df_feat.columns:
        df_feat = df_feat.reset_index().rename(columns={"index": "Date"})
    return df_feat


def _align_features(feat_df: pd.DataFrame, saved_feats: list[str]) -> pd.DataFrame:
    """
    Ensures columns/ordering match the training schema.
    Missing features get 0.0; extras are dropped. Preserves 'Date' if present.
    """
    df = feat_df.copy()
    if not saved_feats:
        return df.drop(columns=["Date"], errors="ignore").select_dtypes(include=[np.number])
    for c in saved_feats:
        if c not in df.columns:
            df[c] = 0.0
    cols = (["Date"] + saved_feats) if "Date" in df.columns else saved_feats
    return df[cols]


def _feature_coverage_guard(X: pd.DataFrame, saved_feats: list[str], min_coverage: float = 0.8):
    """
    Aborts if too many expected features are missing (or have zero variance).
    Coverage = fraction of saved_feats present with non-zero variance.
    """
    if not saved_feats:
        return
    present = [c for c in saved_feats if c in X.columns]
    if not present:
        raise SystemExit("No overlap between saved training features and current feature set.")
    var = X[present].astype(float).var(numeric_only=True)
    good = (var > 0).sum()
    coverage = good / max(1, len(saved_feats))
    if coverage < min_coverage:
        raise SystemExit(
            f"Feature coverage too low: {coverage:.1%} of required features have variance. "
            "Prediction aborted — rebuild inference features to match training."
        )


# =============================================================================
# Mapping & scoring helpers
# =============================================================================
def _binary_to_legacy(pred_bin: int) -> int:
    """
    Map binary forward-returns decision to 3-class 0/1/2:

      binary 1 → 0 (SPIKE)
      binary 0 → 1 (NORMAL)

    CRASH (2) is not emitted by this mapping.
    """
    return 0 if int(pred_bin) == 1 else 1


def _assert_binary_model(model) -> None:
    classes = list(getattr(model, "classes_", [0, 1]))
    if not set(classes) <= {0, 1}:
        raise SystemExit(f"Expected binary model classes {{0,1}}, got {classes}")


def _map_proba_to_p1(model, X_last: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_last)[0]
        classes_enc = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes_enc:
            return float(probs[classes_enc.index(1)])
        return float(probs[-1])
    pred_plain = int(model.predict(X_last)[0])
    return 0.7 if pred_plain == 1 else 0.3


def _score_latest(model, X: pd.DataFrame, variant: str) -> tuple[int, float, pd.Timestamp]:
    """
    Takes the last row, computes p(long=1), applies the threshold, and returns
    (binary_decision, p1, timestamp).
    """
    X_last = X.tail(1).drop(columns=["Date"], errors="ignore").astype(float)
    if X_last.isna().any(axis=None):
        X_valid = X.dropna().tail(1).drop(columns=["Date"], errors="ignore").astype(float)
        if X_valid.empty:
            raise SystemExit("No valid feature row to score (NaNs present in latest rows).")
        X_last = X_valid

    p1_raw = _map_proba_to_p1(model, X_last)
    cfg = _load_thresholds(variant)
    p1 = 1.0 - p1_raw if cfg.get("invert_proba", False) else p1_raw
    decision = int(p1 >= float(cfg.get("p_min", 0.55)))

    if "Date" in X.columns and X["Date"].notna().any():
        ts = pd.to_datetime(X["Date"].dropna().iloc[-1])
    else:
        ts = pd.Timestamp("now").normalize()

    return decision, p1, ts


# =============================================================================
# Logging helpers
# =============================================================================
_REQUIRED_COLS = [
    "Date",
    "Label",
    "Pred",
    "Prediction",
    "Proba",
    "Spike_Conf",
    "Crash_Conf",
    "Confidence",
]


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def _append_single(pred_012: int, p1: float, when: pd.Timestamp) -> None:
    """
    Appends/updates a single day in logs/labeled_predictions.csv (0=SPIKE,1=NORMAL,2=CRASH).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "labeled_predictions.csv"

    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
    else:
        df = pd.DataFrame(columns=_REQUIRED_COLS)

    df = _ensure_columns(df, _REQUIRED_COLS)

    row_mask = df["Date"] == pd.to_datetime(when)
    row = {
        "Date": pd.to_datetime(when),
        "Label": df.loc[row_mask, "Label"].iloc[0] if row_mask.any() else pd.NA,
        "Pred": int(pred_012),
        "Prediction": int(pred_012),
        "Proba": float(p1),
        "Spike_Conf": float(p1),
        "Crash_Conf": float(1.0 - p1),
        "Confidence": float(abs(p1 - 0.5) * 2.0),
    }

    if row_mask.any():
        for k, v in row.items():
            df.loc[row_mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values("Date").reset_index(drop=True)
    print(f"[debug] writing preds to {path.resolve()}")
    df.to_csv(path, index=False)
    print(f"[predict] wrote → {path}")


# =============================================================================
# Backfill
# =============================================================================
def _backfill_full(model, saved_feats: list[str], variant: str) -> None:
    """
    Scores all valid dates and (re)writes logs/labeled_predictions.csv.
    Preserves existing Label. Outputs 0=SPIKE,1=NORMAL,2=CRASH Prediction.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "labeled_predictions.csv"

    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    feat = _build_inference_features_from_prices(raw, saved_feats)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    feature_cols = [c for c in X.columns if c != "Date"]
    full = X.dropna(subset=feature_cols).copy()
    if full.empty:
        raise SystemExit("No rows to score after dropping NaN feature rows.")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(full[feature_cols].astype(float))
        classes_enc = list(getattr(model, "classes_", [0, 1]))
        if not set(classes_enc) <= {0, 1}:
            raise SystemExit(f"Expected binary model classes {{0,1}}, got {classes_enc}")
        idx1 = classes_enc.index(1) if 1 in classes_enc else (len(classes_enc) - 1)
        p1 = probs[:, idx1].astype(float)
    else:
        preds = model.predict(full[feature_cols].astype(float)).astype(int)
        p1 = preds * 0.7 + (1 - preds) * 0.3

    print(
        f"[debug] p1 mean={float(np.mean(p1)):.4f} sd={float(np.std(p1)):.4f} "
        f"min={float(np.min(p1)):.4f} max={float(np.max(p1)):.4f}"
    )

    thresholds = _load_thresholds(variant)
    if thresholds.get("invert_proba", False):
        p1 = 1.0 - p1
        print("[debug] applied invert_proba: using (1 - p1) for decisions")

    pred_bin = (p1 >= float(thresholds.get("p_min", 0.55))).astype(int)
    pred_012 = np.where(pred_bin == 1, 0, 1).astype(int)

    out = (
        pd.DataFrame(
            {
                "Date": full["Date"],
                "Prediction": pred_012,
                "Pred": pred_012,
                "Proba": p1.astype(float),
                "Spike_Conf": p1.astype(float),
                "Crash_Conf": (1.0 - p1).astype(float),
                "Confidence": np.abs(p1 - 0.5) * 2.0,
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if out_path.exists():
        prev = pd.read_csv(out_path, parse_dates=["Date"])
        prev = _ensure_columns(prev, _REQUIRED_COLS)
        out = out.merge(prev[["Date", "Label"]], on="Date", how="left")

    out = _ensure_columns(out, _REQUIRED_COLS)

    n_spike = int((out["Prediction"] == 0).sum())
    n_norm = int((out["Prediction"] == 1).sum())
    n_crash = int((out["Prediction"] == 2).sum())
    print(f"[debug] preds — spike=0:{n_spike} normal=1:{n_norm} crash=2:{n_crash}")
    print(f"[debug] writing preds to {out_path.resolve()}")

    out.to_csv(out_path, index=False)
    print(f"[backfill] wrote → {out_path}  rows={len(out)}")


# =============================================================================
# Single live prediction
# =============================================================================
def live_predict() -> tuple[int, float, pd.Timestamp]:
    """
    Returns (binary_decision, p(long=1), timestamp) using the latest row in SPY_DAILY_CSV.
    Caller maps to 0=SPIKE,1=NORMAL,2=CRASH if needed.
    """
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    model_path, variant = _resolve_model_path()
    model, saved_feats = _load_model_and_features(model_path, variant)
    _assert_binary_model(model)

    feat = _build_inference_features_from_prices(raw, saved_feats)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    return _score_latest(model, X, variant)


# =============================================================================
# CLI
# =============================================================================
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
    model_path, variant = _resolve_model_path()
    model, saved_feats = _load_model_and_features(model_path, variant)
    _assert_binary_model(model)

    if args.backfill:
        _backfill_full(model, saved_feats, variant)
        return 0

    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    feat = _build_inference_features_from_prices(raw, saved_feats)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    pred_bin, prob, when = _score_latest(model, X, variant)
    pred_012 = _binary_to_legacy(pred_bin)
    label_human = {0: "SPIKE", 1: "NORMAL", 2: "CRASH"}[pred_012]
    print(
        f"[predict] {pd.to_datetime(when).date()}  p(long=1)={prob:.4f}  "
        f"binary={pred_bin}  legacy={pred_012} ({label_human})"
    )

    _append_single(pred_012, prob, when)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
