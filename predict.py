#!/usr/bin/env python3
# predict.py
"""
NeuroVest — prediction runner (backfill + live)

Purpose
-------
Loads a trained model, rebuilds the same features used during training, scores
probabilities, applies a decision threshold, and writes predictions to logs.

Label convention
----------------
Unified 3-class convention:
    0 = CRASH
    1 = NORMAL
    2 = SPIKE

The active forward-returns model is binary internally ({0,1} = no-trade/trade).
To remain compatible with 3-class outputs, binary predictions are mapped to:
    0 → 1 (NORMAL)
    1 → 2 (SPIKE)

CRASH (0) is reserved for explicit crash labels and is not emitted by the
current binary forward-returns model.
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

# Regime-adaptive thresholds (Feb 2026)
# Use different thresholds based on current market regime
USE_REGIME_ADAPTIVE_THRESHOLDS = os.getenv("USE_REGIME_THRESHOLDS", "1").lower() in ("1", "true", "yes")


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

    # Check for multi-asset model first (60% accuracy vs 58% single-asset)
    multi_asset_path = MODELS_DIR / "xgboost_multi_asset.pkl"
    if multi_asset_path.exists():
        return multi_asset_path, "multi_asset"

    # Fallback to single-asset models
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
    if variant == "multi_asset":
        ftxt = MODELS_DIR / "multi_asset_features.txt"
    elif variant == "forward":
        ftxt = MODELS_DIR / "input_features_fwd.txt"
    else:
        ftxt = MODELS_DIR / "input_features.txt"
    saved_feats = []
    if ftxt.exists():
        saved_feats = [ln.strip() for ln in ftxt.read_text().splitlines() if ln.strip()]
    return obj, saved_feats


def _load_thresholds(variant: str) -> dict[str, float]:
    """
    Threshold precedence (first found wins):

      1) THRESH_PATH (env override; optional)
      2) configs/best_thresholds.json
      3) models/thresholds_fwd.json
      4) models/thresholds.json
      5) PREDICT_CFG["p_min"] (default)

    All thresholds are clamped into [0.10, 0.90]. If a file also includes
    "invert_proba", it is respected.
    """
    # 1) Explicit env override
    env_path = os.getenv("THRESH_PATH")
    if env_path and os.path.exists(env_path):
        try:
            with open(env_path) as f:
                obj = json.load(f)
            t = obj.get("spike_thresh") or obj.get("p_min") or obj.get("threshold")
            inv = bool(obj.get("invert_proba", False))
            if t is not None:
                t = float(t)
                t = max(0.10, min(0.90, t))
                print(f"[debug] using threshold {t:.6f} from THRESH_PATH={env_path}")
                print(f"[debug] invert_proba={inv}")
                return {"p_min": t, "invert_proba": inv}
        except Exception as e:
            print(f"[warn] failed to read THRESH_PATH={env_path}: {e}")

    # 2–4) Standard precedence chain
    fallbacks = [
        Path("configs/best_thresholds.json"),
        MODELS_DIR / "thresholds_fwd.json",
        MODELS_DIR / "thresholds.json",
    ]
    for p in fallbacks:
        try:
            with open(p) as f:
                obj = json.load(f)
            t = obj.get("spike_thresh") or obj.get("p_min") or obj.get("threshold")
            inv = bool(obj.get("invert_proba", False))
            if t is not None:
                t = float(t)
                t = max(0.10, min(0.90, t))
                print(f"[debug] using threshold {t:.6f} from {p}")
                print(f"[debug] invert_proba={inv}")
                return {"p_min": t, "invert_proba": inv}
        except Exception:
            pass

    # 5) Fall back to PREDICT_CFG default (single source of truth)
    from config import PREDICTION_THRESHOLD
    t = float(PREDICT_CFG.get("p_min", PREDICTION_THRESHOLD))
    t = max(0.10, min(0.90, t))
    print(f"[debug] using default threshold {t:.6f} (PREDICT_CFG/default)")
    print("[debug] invert_proba=False")
    return {"p_min": t, "invert_proba": False}


def _get_regime_adaptive_threshold(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Get threshold based on current market regime.

    Args:
        df: DataFrame with price/feature data for regime detection

    Returns:
        (threshold, regime_info_dict)
    """
    try:
        from core.regime_adaptive_thresholds import RegimeAdaptiveThresholds

        rat = RegimeAdaptiveThresholds(df)
        rat.load()  # Load saved thresholds (or use defaults)

        regime = rat.detect_current_regime()
        threshold = rat.get_threshold(regime)

        print(f"[regime] Current: volatility={regime['volatility']}, "
              f"trend={regime['trend']}, risk={regime['risk_appetite']}")
        print(f"[regime] Using threshold: {threshold:.4f}")

        return threshold, regime

    except Exception as e:
        print(f"[warn] Regime detection failed: {e}")
        print("[warn] Falling back to static threshold")
        from config import PREDICTION_THRESHOLD
        return PREDICTION_THRESHOLD, {"error": str(e)}


# =============================================================================
# Feature engineering — identical to training
# =============================================================================
def _build_inference_features_from_prices(
    raw_df: pd.DataFrame, saved_feats: list[str], variant: str = "forward"
) -> pd.DataFrame:
    """
    Uses utils.add_features + utils.finalize_features, then aligns to the saved schema.
    Carries 'Date' through for logging.
    """
    df_feat, all_cols = add_features(raw_df)
    cols = saved_feats if saved_feats else all_cols
    df_feat = finalize_features(df_feat, cols)

    # Add asset_type features for multi-asset models (SPY = stock)
    if variant == "multi_asset":
        df_feat["asset_type_stock"] = 1
        df_feat["asset_type_crypto"] = 0

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
            "Prediction aborted — inference features do not match training schema."
        )


# =============================================================================
# Mapping & scoring helpers
# =============================================================================
def _binary_to_legacy(pred_bin: int) -> int:
    """
    Map binary forward-returns decision to 3-class 0/1/2:

      binary 1 → 2 (SPIKE)
      binary 0 → 1 (NORMAL)

    CRASH (0) is not emitted by this mapping for the current binary model.
    """
    return 2 if int(pred_bin) == 1 else 1


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


def _score_latest(
    model,
    X: pd.DataFrame,
    variant: str,
    raw_df: pd.DataFrame = None,
) -> tuple[int, float, pd.Timestamp]:
    """
    Takes the last row, computes p(long=1), applies the threshold, and returns
    (binary_decision, p1, timestamp).

    If USE_REGIME_ADAPTIVE_THRESHOLDS is enabled and raw_df is provided,
    uses regime-specific thresholds.
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

    # Use regime-adaptive threshold if enabled
    if USE_REGIME_ADAPTIVE_THRESHOLDS and raw_df is not None:
        threshold, regime = _get_regime_adaptive_threshold(raw_df)
    else:
        threshold = float(cfg.get("p_min", 0.55))

    decision = int(p1 >= threshold)

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
    # Signal contract fields (Day 4)
    "symbol",
    "horizon",
    "side",
    "ev",
    "model_version",
    "config_hash",
]


def _load_manifest() -> dict:
    """Load artifact_manifest.json to get model_version and config_hash."""
    try:
        with open(MODELS_DIR / "artifact_manifest.json") as f:
            return json.load(f)
    except Exception:
        return {}


def _compute_ev(prob: float, avg_gain: float = 0.004, avg_loss: float = 0.003) -> float:
    """Expected value = p*gain - (1-p)*loss (in return units)."""
    return float(prob * avg_gain - (1.0 - prob) * avg_loss)


def fill_realized_pnl(
    signal_path: Path | None = None,
    price_csv: Path | None = None,
    horizon_days: int = 1,
) -> None:
    """
    Annotate signals with realized PnL once the horizon has elapsed.
    Adds/updates column 'realized_ret' and 'label_realized' in labeled_predictions.csv.
    Only processes rows where horizon has elapsed and 'realized_ret' is NaN.
    """
    if signal_path is None:
        signal_path = LOGS_DIR / "labeled_predictions.csv"
    if price_csv is None:
        price_csv = Path(os.getenv("SPY_DAILY_CSV", str(SPY_DAILY_CSV)))
    if not signal_path.exists():
        return

    preds = pd.read_csv(signal_path, parse_dates=["Date"])
    if preds.empty:
        return

    prices = pd.read_csv(price_csv, parse_dates=["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    price_map = prices.set_index("Date")["Close"].to_dict()

    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=horizon_days)

    if "realized_ret" not in preds.columns:
        preds["realized_ret"] = float("nan")
    if "label_realized" not in preds.columns:
        preds["label_realized"] = pd.NA

    updated = 0
    for i, row in preds.iterrows():
        if pd.notna(preds.at[i, "realized_ret"]):
            continue  # already annotated
        signal_date = pd.to_datetime(row["Date"])
        if signal_date > cutoff:
            continue  # horizon has not elapsed yet
        exit_date = signal_date + pd.Timedelta(days=horizon_days)
        entry = price_map.get(signal_date)
        exit_ = price_map.get(exit_date)
        if entry is None or exit_ is None:
            # Try to find nearest exit price within ±3 days
            candidates = [
                price_map.get(exit_date + pd.Timedelta(days=d))
                for d in range(-3, 4)
                if price_map.get(exit_date + pd.Timedelta(days=d))
            ]
            exit_ = candidates[0] if candidates else None
        if entry and exit_:
            ret = float(exit_) / float(entry) - 1.0
            preds.at[i, "realized_ret"] = ret
            preds.at[i, "label_realized"] = 1 if ret > 0 else 0
            updated += 1

    if updated:
        preds.to_csv(signal_path, index=False)
        print(f"[realized_pnl] annotated {updated} signals with realized returns → {signal_path}")


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def _append_single(pred_012: int, p1: float, when: pd.Timestamp) -> None:
    """
    Appends or updates a single day in logs/labeled_predictions.csv
    using the convention 0=CRASH,1=NORMAL,2=SPIKE.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "labeled_predictions.csv"

    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
    else:
        df = pd.DataFrame(columns=_REQUIRED_COLS)

    df = _ensure_columns(df, _REQUIRED_COLS)

    manifest = _load_manifest()
    row_mask = df["Date"] == pd.to_datetime(when)
    side = "long" if pred_012 == 2 else "flat"
    row = {
        "Date": pd.to_datetime(when),
        "Label": df.loc[row_mask, "Label"].iloc[0] if row_mask.any() else pd.NA,
        "Pred": int(pred_012),
        "Prediction": int(pred_012),
        "Proba": float(p1),
        "Spike_Conf": float(p1),
        "Crash_Conf": float(1.0 - p1),
        "Confidence": float(abs(p1 - 0.5) * 2.0),
        # Signal contract fields
        "symbol": "SPY",
        "horizon": int(PREDICT_CFG.get("horizon", 1)),
        "side": side,
        "ev": _compute_ev(p1, PREDICT_CFG.get("avg_gain", 0.004), PREDICT_CFG.get("avg_loss", 0.003)),
        "model_version": manifest.get("code_git_sha", "unknown"),
        "config_hash": manifest.get("config_hash", "unknown"),
    }

    if row_mask.any():
        for k, v in row.items():
            df.loc[row_mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values("Date").reset_index(drop=True)

    unique_preds = set(df["Prediction"].dropna().astype(int))
    if not unique_preds.issubset({0, 1, 2}):
        raise SystemExit(
            f"Unexpected Prediction values in labeled_predictions.csv: {sorted(unique_preds)}"
        )

    n_crash = int((df["Prediction"] == 0).sum())
    n_norm = int((df["Prediction"] == 1).sum())
    n_spike = int((df["Prediction"] == 2).sum())

    print(f"[debug] writing preds to {path.resolve()}")
    print(
        "[debug] prediction class distribution (0=CRASH,1=NORMAL,2=SPIKE): "
        f"crash={n_crash} normal={n_norm} spike={n_spike}"
    )
    df.to_csv(path, index=False)
    print(f"[predict] wrote → {path}")


# =============================================================================
# Backfill
# =============================================================================
def _backfill_full(model, saved_feats: list[str], variant: str) -> None:
    """
    Scores all valid dates and (re)writes logs/labeled_predictions.csv.
    Preserves existing Label. Outputs 0=CRASH,1=NORMAL,2=SPIKE in Prediction.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "labeled_predictions.csv"

    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    feat = _build_inference_features_from_prices(raw, saved_feats, variant)
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

    # Use regime-adaptive thresholds if enabled
    if USE_REGIME_ADAPTIVE_THRESHOLDS:
        try:
            from core.regime_adaptive_thresholds import RegimeAdaptiveThresholds
            from core.regime_backtest import RegimeDetector

            print("[regime] Computing per-row regime-adaptive thresholds...")
            rat = RegimeAdaptiveThresholds(raw)
            rat.load()  # Load saved thresholds

            # Detect regimes for all dates
            detector = RegimeDetector()
            regimes = detector.detect_all_regimes(raw)

            # Align regimes with predictions
            # Use dates from full DataFrame
            pred_dates = pd.to_datetime(full["Date"])
            thresholds_per_row = np.full(len(full), 0.45)  # Default

            for i, date in enumerate(pred_dates):
                if date in regimes.index:
                    regime = {
                        "volatility": regimes.loc[date, "volatility"],
                        "trend": regimes.loc[date, "trend"],
                        "risk_appetite": regimes.loc[date, "risk_appetite"],
                    }
                    thresholds_per_row[i] = rat.get_threshold(regime)

            # Count threshold distribution
            unique_thresholds = np.unique(thresholds_per_row)
            print(f"[regime] Threshold distribution: {len(unique_thresholds)} unique values")
            for t in unique_thresholds[:5]:  # Show first 5
                count = (thresholds_per_row == t).sum()
                print(f"  - {t:.2f}: {count} rows ({count/len(thresholds_per_row):.1%})")

            pred_bin = (p1 >= thresholds_per_row).astype(int)

        except Exception as e:
            print(f"[warn] Regime-adaptive backfill failed: {e}")
            print("[warn] Using static threshold")
            pred_bin = (p1 >= float(thresholds.get("p_min", 0.55))).astype(int)
    else:
        pred_bin = (p1 >= float(thresholds.get("p_min", 0.55))).astype(int)

    # Binary 1 → 2 (SPIKE), binary 0 → 1 (NORMAL); CRASH (0) is not emitted here.
    pred_012 = np.where(pred_bin == 1, 2, 1).astype(int)

    manifest = _load_manifest()
    model_version = manifest.get("code_git_sha", "unknown")
    config_hash = manifest.get("config_hash", "unknown")
    ev_vals = np.array([
        _compute_ev(float(p), PREDICT_CFG.get("avg_gain", 0.004), PREDICT_CFG.get("avg_loss", 0.003))
        for p in p1
    ])

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
                # Signal contract fields
                "symbol": "SPY",
                "horizon": int(PREDICT_CFG.get("horizon", 1)),
                "side": np.where(pred_bin == 1, "long", "flat"),
                "ev": ev_vals,
                "model_version": model_version,
                "config_hash": config_hash,
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if out_path.exists():
        prev = pd.read_csv(out_path, parse_dates=["Date"])
        prev = _ensure_columns(prev, _REQUIRED_COLS)
        # Preserve Label and realized_ret from prior runs
        merge_cols = [c for c in ["Label", "realized_ret", "label_realized"] if c in prev.columns]
        if merge_cols:
            out = out.merge(prev[["Date"] + merge_cols], on="Date", how="left")

    out = _ensure_columns(out, _REQUIRED_COLS)

    unique_preds = set(out["Prediction"].dropna().astype(int))
    if not unique_preds.issubset({0, 1, 2}):
        raise SystemExit(f"Unexpected Prediction values in backfill output: {sorted(unique_preds)}")

    n_crash = int((out["Prediction"] == 0).sum())
    n_norm = int((out["Prediction"] == 1).sum())
    n_spike = int((out["Prediction"] == 2).sum())
    print(f"[debug] preds — crash=0:{n_crash} normal=1:{n_norm} spike=2:{n_spike}")
    print(f"[debug] writing preds to {out_path.resolve()}")

    out.to_csv(out_path, index=False)
    print(f"[backfill] wrote → {out_path}  rows={len(out)}")


# =============================================================================
# Single live prediction
# =============================================================================
def live_predict() -> tuple[int, float, pd.Timestamp]:
    """
    Returns (binary_decision, p(long=1), timestamp) using the latest row in SPY_DAILY_CSV.
    Caller maps to 0=CRASH,1=NORMAL,2=SPIKE if a 3-class label is needed.
    """
    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    model_path, variant = _resolve_model_path()
    model, saved_feats = _load_model_and_features(model_path, variant)
    _assert_binary_model(model)

    feat = _build_inference_features_from_prices(raw, saved_feats, variant)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    # Pass raw_df for regime-adaptive thresholds
    return _score_latest(model, X, variant, raw_df=raw)


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
    p.add_argument(
        "--asof",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only use SPY data up to this date (live or backfill). Defaults to today.",
    )
    p.add_argument(
        "--fill-realized",
        action="store_true",
        help="Annotate past signals with realized PnL where horizon has elapsed.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # --fill-realized: annotate past signals with actual PnL (no model needed)
    if args.fill_realized:
        fill_realized_pnl()
        return 0

    model_path, variant = _resolve_model_path()
    model, saved_feats = _load_model_and_features(model_path, variant)
    _assert_binary_model(model)

    raw = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # --asof: truncate data to a point-in-time date
    if args.asof is not None:
        cutoff = pd.to_datetime(args.asof)
        raw = raw[raw["Date"] <= cutoff].reset_index(drop=True)
        print(f"[predict] --asof {args.asof}: using {len(raw)} rows up to {cutoff.date()}")

    if args.backfill:
        _backfill_full(model, saved_feats, variant)
        return 0

    feat = _build_inference_features_from_prices(raw, saved_feats, variant)
    X = _align_features(feat, saved_feats)
    _feature_coverage_guard(X, saved_feats)

    # Pass raw_df for regime-adaptive thresholds
    pred_bin, prob, when = _score_latest(model, X, variant, raw_df=raw)
    pred_012 = _binary_to_legacy(pred_bin)
    label_human = {0: "CRASH", 1: "NORMAL", 2: "SPIKE"}[pred_012]

    if pred_012 not in (0, 1, 2):
        raise SystemExit(f"Unexpected legacy prediction value: {pred_012}")

    print(
        f"[predict] {pd.to_datetime(when).date()}  p(long=1)={prob:.4f}  "
        f"binary={pred_bin}  legacy={pred_012} ({label_human})"
    )

    _append_single(pred_012, prob, when)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
