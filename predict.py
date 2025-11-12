#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LOGS_DIR, MODELS_DIR, PREDICT_CFG, SPY_DAILY_CSV


# =============================================================================
# Public API (used by sweep scripts / CLI)
# =============================================================================
def run_predictions(backfill: bool = True) -> None:
    """
    Entry point for external callers (e.g., sweep_optimizer.py).
    If backfill=True, scores full history and rewrites logs/labeled_predictions.csv.
    Otherwise, appends a single live row.
    """
    model_path = MODELS_DIR / "market_crash_model.pkl"
    model, saved_feats = _load_model_and_features(model_path)
    if backfill:
        _backfill_full(model, saved_feats)
    else:
        pred, prob, when = live_predict()
        _append_single(pred, prob, when)


# =============================================================================
# Feature engineering (keep aligned with training)
# =============================================================================
def _nv_simple_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal daily features matching the trainer:
    ret_1, ret_5, sma_5, sma_20, vol_10, rsi14 (0-100 scale).
    (Kept for compatibility; production uses _rich_price_features below.)
    """
    df = raw_df.copy()

    # Normalize Date
    if "Date" not in df.columns:
        # best-effort alias
        for c in df.columns:
            if str(c).strip().lower() == "date":
                df = df.rename(columns={c: "Date"})
                break
    if "Date" not in df.columns:
        raise SystemExit("Input price table is missing a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Price column — prefer exact, then fuzzy
    candidates = [c for c in df.columns if str(c).lower() in ("adj close", "adjclose", "close")]
    if not candidates:
        candidates = [c for c in df.columns if "close" in str(c).lower()]
    if not candidates:
        raise SystemExit(
            "No price column found (expected one of: 'Adj Close', 'AdjClose', 'Close')."
        )
    px = pd.to_numeric(df[candidates[0]], errors="coerce")

    out = pd.DataFrame({"Date": df["Date"]})

    # Returns / overlays (use fill_method=None to avoid pandas FutureWarnings)
    out["ret_1"] = px.pct_change(1, fill_method=None)
    out["ret_5"] = px.pct_change(5, fill_method=None)
    out["sma_5"] = px.rolling(5, min_periods=5).mean() / px - 1.0
    out["sma_20"] = px.rolling(20, min_periods=20).mean() / px - 1.0
    out["vol_10"] = px.pct_change(fill_method=None).rolling(10, min_periods=10).std()

    # RSI(14)
    d = px.diff()
    gain = d.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0.0)).rolling(14, min_periods=14).mean().replace(0, 1e-12)
    rs = gain / loss
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    return out


def _rich_price_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    More complete daily feature set (ret_N, sma_N pct devs, vol_N, RSI(2/14), MACD(12,26,9), %b).
    This dramatically reduces zero-filled columns vs training.
    """
    df = raw_df.copy()
    if "Date" not in df.columns:
        raise SystemExit("Missing 'Date' in price table.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # pick price: prefer Adj Close -> Close
    price_col = None
    for c in ["Adj Close", "AdjClose", "Close"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        cands = [c for c in df.columns if "close" in str(c).lower()]
        if not cands:
            raise SystemExit("No price column found.")
        price_col = cands[0]

    px = pd.to_numeric(df[price_col], errors="coerce")
    out = pd.DataFrame({"Date": df["Date"]})

    # returns
    for n in [1, 2, 3, 5, 10, 20]:
        out[f"ret_{n}"] = px.pct_change(n, fill_method=None)

    # moving-average deviations
    for n in [5, 10, 20, 50, 100, 200]:
        ma = px.rolling(n, min_periods=n).mean()
        out[f"sma_{n}"] = ma / px - 1.0

    # volatility
    for n in [5, 10, 20]:
        out[f"vol_{n}"] = px.pct_change(fill_method=None).rolling(n, min_periods=n).std()

    # RSI(2) and RSI(14)
    d = px.diff()
    for n in [2, 14]:
        up = d.clip(lower=0.0).rolling(n, min_periods=n).mean()
        down = (-d.clip(upper=0.0)).rolling(n, min_periods=n).mean().replace(0, 1e-12)
        rs = up / down
        out[f"rsi{n}"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD(12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # Bollinger %b (20,2)
    ma20 = px.rolling(20, min_periods=20).mean()
    sd20 = px.rolling(20, min_periods=20).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    out["pct_b"] = (px - lower) / (upper - lower)

    return out


def _load_model_and_features(path: Path) -> tuple[object, list[str]]:
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    obj = joblib.load(path)
    # Newer save format: dict with model + features
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], list(obj.get("features", []))
    # Legacy: bare estimator
    return obj, []


def _align_features(feat_df: pd.DataFrame, saved_feats: list[str]) -> pd.DataFrame:
    """
    Ensure columns/ordering match the training schema.
    Missing features get 0.0; extras are dropped. Preserves 'Date' if present.
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


def _feature_coverage_guard(X: pd.DataFrame, saved_feats: list[str], min_coverage: float = 0.8):
    """
    Abort if too many expected features are missing (and got zero-filled) or NaN.
    Coverage = fraction of saved_feats that are present and have non-zero variance.
    """
    if not saved_feats:
        return
    present = [c for c in saved_feats if c in X.columns]
    if not present:
        raise SystemExit("No overlap between saved training features and current feature set.")
    # zero-variance = suspicious (likely zero-filled)
    var = X[present].astype(float).var(numeric_only=True)
    good = (var > 0).sum()
    coverage = good / max(1, len(saved_feats))
    if coverage < min_coverage:
        raise SystemExit(
            f"Feature coverage too low: {coverage:.1%} of required features have variance. "
            "Prediction aborted — rebuild inference features to match training."
        )


# =============================================================================
# Thresholds / Scoring
# =============================================================================
def _load_thresholds() -> dict[str, float]:
    """
    Prefer tuned sweep → generic model thresholds → forward-returns thresholds.
    Also read optional 'invert_proba' flag if present in configs/best_thresholds.json.
    Clamp threshold to [0.10, 0.90] to avoid pathological all-ones/all-zeros.
    """
    candidates = [
        Path(
            "configs/best_thresholds.json"
        ),  # may contain {"spike_thresh": ..., "invert_proba": true/false}
        MODELS_DIR / "thresholds.json",  # {"threshold": ...}
        MODELS_DIR / "thresholds_fwd.json",  # {"p_min": ...} legacy
    ]
    t = None
    inv = False
    src = "PREDICT_CFG"
    for p in candidates:
        try:
            with open(p) as f:
                obj = json.load(f)
            # threshold
            if "spike_thresh" in obj and obj["spike_thresh"] is not None:
                t = float(obj["spike_thresh"])
                src = str(p)
            elif "threshold" in obj:
                t = float(obj["threshold"])
                src = str(p)
            elif "p_min" in obj:
                t = float(obj["p_min"])
                src = str(p)
            # inversion flag (only meaningful in configs/best_thresholds.json)
            if "invert_proba" in obj and obj["invert_proba"] is not None:
                inv = bool(obj["invert_proba"])
            if t is not None:
                break
        except Exception:
            pass

    if t is None:
        t = float(PREDICT_CFG.get("p_min", 0.55))

    t_clamped = max(0.10, min(0.90, t))
    if abs(t_clamped - t) > 1e-9:
        print(f"[debug] threshold {t:.6f} from {src} clamped → {t_clamped:.6f}")
    else:
        print(f"[debug] using threshold {t_clamped:.6f} from {src}")

    print(f"[debug] invert_proba={inv}")
    return {"p_min": t_clamped, "invert_proba": inv}


def _map_proba_to_p1(model, X_last: pd.DataFrame) -> float:
    """
    Return probability for class=1 (long). If model has no predict_proba,
    degrade to a pseudo-probability.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_last)[0]
        classes_enc = list(getattr(model, "classes_", [0, 1]))
        # robust mapping in case class order isn't [0,1]
        if 1 in classes_enc:
            p1 = float(probs[classes_enc.index(1)])
        else:
            p1 = float(probs[-1])
        return p1
    # Fallback for margin-only estimators
    pred_plain = int(model.predict(X_last)[0])
    return 0.7 if pred_plain == 1 else 0.3


def _score_latest(model, X: pd.DataFrame) -> tuple[int, float, pd.Timestamp]:
    """
    Grab last row, compute p(long=1), optionally invert, apply threshold,
    and return (decision, p1_used_for_decision, timestamp).
    """
    X_last = X.tail(1).drop(columns=["Date"], errors="ignore").astype(float)
    if X_last.isna().any(axis=None):
        X_valid = X.dropna().tail(1).drop(columns=["Date"], errors="ignore").astype(float)
        if X_valid.empty:
            raise SystemExit("No valid feature row to score (NaNs present in latest rows).")
        X_last = X_valid

    p1_raw = _map_proba_to_p1(model, X_last)
    cfg = _load_thresholds()
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


def _append_single(decision: int, p1: float, when: pd.Timestamp) -> None:
    """
    Append/update a single day’s prediction into logs/labeled_predictions.csv.
    Ensures all expected columns exist.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "labeled_predictions.csv"

    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
    else:
        df = pd.DataFrame(columns=_REQUIRED_COLS)

    df = _ensure_columns(df, _REQUIRED_COLS)

    # Update or append row for 'when'
    row_mask = df["Date"] == pd.to_datetime(when)
    row = {
        "Date": pd.to_datetime(when),
        "Label": df.loc[row_mask, "Label"].iloc[0] if row_mask.any() else pd.NA,
        "Pred": int(decision),
        "Prediction": int(decision),
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

    # NEW: absolute path visibility
    print(f"[debug] writing preds to {path.resolve()}")

    df.to_csv(path, index=False)
    print(f"[predict] wrote → {path}")


def _backfill_full(model, saved_feats: list[str]) -> None:
    """
    Score all valid dates and (re)write labeled_predictions.csv.
    Preserves any existing Label values.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "labeled_predictions.csv"

    # Load raw SPY prices
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Build + align (use rich features)
    feat = _rich_price_features(raw)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    # Keep only rows without NaNs in model features
    feature_cols = [c for c in X.columns if c != "Date"]
    full = X.dropna(subset=feature_cols).copy()
    if full.empty:
        raise SystemExit("No rows to score after dropping NaN feature rows.")

    # Score probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(full[feature_cols].astype(float))
        classes_enc = list(getattr(model, "classes_", [0, 1]))
        idx1 = classes_enc.index(1) if 1 in classes_enc else (len(classes_enc) - 1)
        p1 = probs[:, idx1].astype(float)
    else:
        preds = model.predict(full[feature_cols].astype(float)).astype(int)
        p1 = preds * 0.7 + (1 - preds) * 0.3

    # Debug distribution of probabilities
    print(
        f"[debug] p1 mean={float(np.mean(p1)):.4f} sd={float(np.std(p1)):.4f} min={float(np.min(p1)):.4f} max={float(np.max(p1)):.4f}"
    )

    thresholds = _load_thresholds()
    if thresholds.get("invert_proba", False):
        p1 = 1.0 - p1
        print("[debug] applied invert_proba: using (1 - p1) for decisions")

    pred = (p1 >= float(thresholds.get("p_min", 0.55))).astype(int)

    out = (
        pd.DataFrame(
            {
                "Date": full["Date"],
                "Prediction": pred.astype(int),
                "Pred": pred.astype(int),
                "Proba": p1.astype(float),
                "Spike_Conf": p1.astype(float),
                "Crash_Conf": (1.0 - p1).astype(float),
                "Confidence": np.abs(p1 - 0.5) * 2.0,
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Preserve existing Label if present
    if out_path.exists():
        prev = pd.read_csv(out_path, parse_dates=["Date"])
        prev = _ensure_columns(prev, _REQUIRED_COLS)
        out = out.merge(prev[["Date", "Label"]], on="Date", how="left")

    # Ensure all required columns present
    out = _ensure_columns(out, _REQUIRED_COLS)

    # NEW: final counts + absolute path
    final_pos = int(out["Prediction"].sum())
    print(f"[debug] final positives={final_pos} zeros={len(out) - final_pos}")
    print(f"[debug] writing preds to {out_path.resolve()}")

    out.to_csv(out_path, index=False)
    print(f"[backfill] wrote → {out_path}  rows={len(out)}  positives={final_pos}")


# =============================================================================
# Single live prediction (convenience)
# =============================================================================
def live_predict() -> tuple[int, float, pd.Timestamp]:
    """
    Return (decision, p(long=1), timestamp) using latest available row in SPY_DAILY_CSV.
    """
    # Load raw SPY prices
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Build features and align to training schema (use rich features)
    feat = _rich_price_features(raw)

    model_path = MODELS_DIR / "market_crash_model.pkl"
    model, saved_feats = _load_model_and_features(model_path)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    return _score_latest(model, X)


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

    model_path = MODELS_DIR / "market_crash_model.pkl"
    model, saved_feats = _load_model_and_features(model_path)

    if args.backfill:
        _backfill_full(model, saved_feats)
        return 0

    # Single live prediction
    # read raw once (avoid double IO)
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    feat = _rich_price_features(raw)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    pred, prob, when = _score_latest(model, X)
    print(f"[predict] {pd.to_datetime(when).date()}  p(long=1)={prob:.4f}  decision={pred}")
    _append_single(pred, prob, when)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
