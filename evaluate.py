#!/usr/bin/env python3
# evaluate.py
"""
evaluate.py - unified evaluation with Proba-based thresholding and correct hurdle.

Purpose
-------
Provides a unified evaluation pipeline that:
- Aligns Actual_Event with the training horizon and hurdle (pos_threshold + costs).
- Derives PredLong from the Proba column using the resolved threshold (NOT from
  stale Prediction column, so threshold changes take effect immediately).
- Falls back to legacy class labels only when Proba is unavailable.

Hurdle calculation
------------------
The hurdle for Actual_Event matches training labels exactly:
    hurdle = pos_threshold + (fee_bps + slippage_bps) / 10000
    (NOT min_edge_bps, which is for EV gating only)

PredLong derivation (priority order)
-------------------------------------
    1) Proba column + resolved threshold (PREFERRED - always re-thresholds)
    2) Legacy Prediction column (fallback: 2=SPIKE → PredLong=1)
    3) Spike_Conf/Crash_Conf (rare fallback)

Threshold resolution
--------------------
    1) configs/best_thresholds.json  -> keys: spike_thresh, invert_proba
    2) models/thresholds_fwd.json    -> keys: p_min or spike_thresh
    3) models/thresholds.json        -> keys: threshold or p_min
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score

from config import TRAIN_CFG

LABELED_LOG = Path("logs/labeled_predictions.csv")
METRIC_OUTPUT_CSV = Path("logs/model_performance.csv")
DB_PATH = Path("neurovest.duckdb")

# Where to look for a price file if actuals/forward returns are missing
PRICE_CANDIDATES = [
    Path("data/SPY.csv"),
    Path("data/spy.csv"),
    Path("data/prices/SPY.csv"),
    Path("data/price/SPY.csv"),
    Path("SPY.csv"),
]


# ---------------------------------------------------------------------------
# Threshold and hurdle helpers
# ---------------------------------------------------------------------------
def _train_hurdle() -> float:
    """
    Compute the hurdle that matches training labels exactly.

    Training labels (add_forward_returns_and_labels) use:
        fwd_ret_net = fwd_ret_raw - (fee_bps + slippage_bps) / 10000
        y = 1 if fwd_ret_net >= pos_threshold

    In raw-return terms the hurdle is:
        fwd_ret_raw >= pos_threshold + (fee_bps + slippage_bps) / 10000

    evaluate.py computes Actual_Event from raw forward returns, so the
    hurdle must include pos_threshold + cost (NOT min_edge_bps, which is
    only used for EV gating in predictions, not for label creation).
    """
    fee_bps = float(TRAIN_CFG.get("fee_bps", 1.5))
    slip_bps = float(TRAIN_CFG.get("slippage_bps", 2.0))
    pos_threshold = float(TRAIN_CFG.get("pos_threshold", 0.005))
    cost = (fee_bps + slip_bps) / 10_000.0
    return pos_threshold + cost


def _resolve_eval_threshold_and_inversion() -> tuple[float, bool, str]:
    """
    Resolves the evaluation probability threshold and inversion flag.

    Priority order:
      1) configs/best_thresholds.json        -> keys: spike_thresh, invert_proba
      2) models/thresholds_fwd.json (forward)-> keys: p_min or spike_thresh
      3) models/thresholds.json (generic)    -> keys: threshold or p_min

    Threshold is clamped into [0.10, 0.90].
    """
    candidates = [
        Path("configs/best_thresholds.json"),
        Path("models/thresholds_fwd.json"),
        Path("models/thresholds.json"),
    ]
    thr, invert, src = None, False, "fallback"

    for p in candidates:
        try:
            obj = json.loads(Path(p).read_text())
        except Exception:
            # Missing or malformed file: try the next candidate
            continue

        if "spike_thresh" in obj and obj["spike_thresh"] is not None:
            thr = float(obj["spike_thresh"])
            src = str(p)
        elif "p_min" in obj and obj["p_min"] is not None:
            thr = float(obj["p_min"])
            src = str(p)
        elif "threshold" in obj and obj["threshold"] is not None:
            thr = float(obj["threshold"])
            src = str(p)

        if "invert_proba" in obj and obj["invert_proba"] is not None:
            invert = bool(obj["invert_proba"])

        if thr is not None:
            break

    if thr is None:
        thr = 0.55

    thr = max(0.10, min(0.90, float(thr)))
    print(f"[debug] evaluate threshold = {thr:.3f} (source: {src})")
    print(f"[debug] probability inversion {'ON' if invert else 'OFF'}")
    return thr, invert, src


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _from_duckdb() -> pd.DataFrame:
    import duckdb

    if not DB_PATH.exists():
        raise FileNotFoundError("DuckDB file not found")
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM eval_join ORDER BY Date").df()
    con.close()
    if df.empty:
        raise ValueError("eval_join returned no rows")
    if "Actual_Event_1d" in df.columns and "Actual_Event" not in df.columns:
        df = df.rename(columns={"Actual_Event_1d": "Actual_Event"})
    return df


def _from_csv_pipeline() -> pd.DataFrame:
    if not LABELED_LOG.exists():
        raise FileNotFoundError(
            "No DuckDB and no labeled CSV; run the pipeline to create logs/labeled_predictions.csv first."
        )
    print(f"[debug] evaluating from {LABELED_LOG.resolve()}")
    return pd.read_csv(LABELED_LOG, low_memory=False)


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _coerce_binary_series(s: pd.Series) -> pd.Series:
    """
    Coerces a variety of representations into a binary 0/1 Series.
    """
    if s.dtype == bool:
        return s.astype(int)
    mapping = {
        "long": 1,
        "buy": 1,
        "up": 1,
        "bull": 1,
        "true": 1,
        "1": 1,
        "short": 0,
        "sell": 0,
        "down": 0,
        "bear": 0,
        "false": 0,
        "0": 0,
    }
    if s.dtype == object:
        return (
            s.astype(str).str.strip().str.lower().map(mapping).astype("Int64").fillna(0).astype(int)
        )
    if s.dtype.kind in "fc":
        return (pd.to_numeric(s, errors="coerce").fillna(0) >= 0.5).astype(int)
    return s.astype(int)


def _load_price_series() -> tuple[pd.DataFrame, str]:
    """
    Attempts to load a SPY price series from a set of candidate paths.
    """
    for p in PRICE_CANDIDATES:
        if p.exists():
            px = pd.read_csv(p)
            if "Date" not in px.columns:
                continue
            col = (
                "Adj Close"
                if "Adj Close" in px.columns
                else ("Close" if "Close" in px.columns else None)
            )
            if col is None:
                continue
            px["Date"] = pd.to_datetime(px["Date"]).dt.date
            return px[["Date", col]].rename(columns={col: "PX"}), str(p)
    raise FileNotFoundError(
        "Could not find a SPY price CSV. Looked for: " + ", ".join(map(str, PRICE_CANDIDATES))
    )


# ---------------------------------------------------------------------------
# Target and PredLong derivation
# ---------------------------------------------------------------------------
def _ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the evaluation frame has:
      - Date (normalized to date)
      - Actual_Event (binary target, aligned with training horizon and hurdle)
      - PredLong (binary prediction, 1=long/positive event, 0=no-trade)
      - Outcome (TP/FP/TN/FN classification for each row)
    """
    df = df.copy()
    if "Date" not in df.columns:
        raise SystemExit("Missing 'Date' column in labeled_predictions.csv.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # -------- Actual_Event: prefer forward-returns / price-derived (TRAIN_CFG-aligned) --------
    H = int(TRAIN_CFG.get("horizon", 1))
    hurdle = _train_hurdle()
    exact_col = f"FwdRet_{H}d"
    present_fwds = [c for c in df.columns if re.fullmatch(r"FwdRet_\d+d", str(c))]

    if exact_col in df.columns:
        fwd = pd.to_numeric(df[exact_col], errors="coerce")
        df["Actual_Event"] = (fwd > hurdle).astype(int)
        print(f"[debug] using existing {exact_col} with hurdle={hurdle:.5f} (H={H})")
    elif present_fwds:
        horizons = [(int(re.findall(r"\d+", c)[0]), c) for c in present_fwds]
        closest = min(horizons, key=lambda t: abs(t[0] - H))
        fwd = pd.to_numeric(df[closest[1]], errors="coerce")
        df["Actual_Event"] = (fwd > hurdle).astype(int)
        print(f"[debug] using closest {closest[1]} (H={closest[0]}) with hurdle={hurdle:.5f}")
    else:
        try:
            px, src = _load_price_series()
            px = px.sort_values("Date").reset_index(drop=True)
            px["PX_NEXT"] = px["PX"].shift(-H)
            col_name = f"FwdRet_{H}d"
            px[col_name] = px["PX_NEXT"] / px["PX"] - 1.0
            df = df.merge(px[["Date", col_name]], on="Date", how="left", validate="m:1")
            if df[col_name].isna().all():
                raise SystemExit(
                    "Could not derive 'Actual_Event': join with price file produced no matches.\n"
                    "- Ensure the prediction CSV 'Date' matches the dates in the price file.\n"
                    f"- Price source tried: {src}"
                )
            df["Actual_Event"] = (pd.to_numeric(df[col_name], errors="coerce") > hurdle).astype(int)
            print(f"[debug] derived Actual_Event with H={H}, hurdle={hurdle:.5f} (source: {src})")
        except Exception:
            actual_col = _first_present(
                df.columns, ["Actual_Event", "Actual_Event_1d", "Actual", "Target", "y"]
            )

            if actual_col is None:
                raise SystemExit(
                    "Could not derive 'Actual_Event' from forward returns or existing columns."
                ) from None

            df["Actual_Event"] = _coerce_binary_series(df[actual_col])
            print(f"[debug] fallback Actual_Event from existing column '{actual_col}'")

    # Final sanity: ensure Actual_Event is binary 0/1
    unique_actual = set(
        pd.to_numeric(df["Actual_Event"], errors="coerce").dropna().unique().tolist()
    )
    if not unique_actual.issubset({0, 1}):
        raise SystemExit(
            "Actual_Event contains unexpected values; "
            f"expected subset of {{0,1}}, found {sorted(unique_actual)}"
        )

    # -------- PredLong: ALWAYS derive from Proba + threshold for consistency --------
    # Previously, PredLong was derived from the stale Prediction column (which
    # baked in whatever threshold predict.py used at backfill time). This meant
    # changing the threshold in best_thresholds.json had no effect on evaluate.py
    # metrics unless predict.py --backfill was re-run first.
    #
    # Now: always re-threshold from raw Proba so evaluate.py metrics reflect
    # the current threshold setting without requiring a predict.py re-run.
    thr, invert, _ = _resolve_eval_threshold_and_inversion()
    used_proba = False

    if "Proba" in df.columns:
        p = pd.to_numeric(df["Proba"], errors="coerce")
        if p.notna().any():
            if invert:
                p = 1.0 - p
            df["PredLong"] = (p >= thr).astype(int)
            used_proba = True
            print(
                f"[debug] PredLong derived from Proba using threshold={thr:.3f}"
                f"{' (inverted)' if invert else ''}"
            )

    if not used_proba:
        # Fallback: derive from legacy class column
        pred_like = _first_present(df.columns, ["Prediction", "Pred", "Label"])
        if pred_like is not None:
            try:
                pred_int = pd.to_numeric(df[pred_like], errors="coerce").astype("Int64")
                uniques = set(pred_int.dropna().unique().tolist())
                if uniques.issubset({0, 1, 2}) and len(uniques) > 0:
                    df["PredLong"] = (pred_int == 2).astype(int)
                    print(f"[debug] PredLong fallback from legacy class '{pred_like}' (2=SPIKE → long)")
                else:
                    raise SystemExit(
                        "Missing columns for evaluation: could not find/derive 'PredLong'."
                    )
            except SystemExit:
                raise
            except Exception:
                raise SystemExit(
                    "Missing columns for evaluation: could not find/derive 'PredLong'."
                )
        elif {"Spike_Conf", "Crash_Conf"} <= set(df.columns):
            sc = pd.to_numeric(df["Spike_Conf"], errors="coerce")
            cc = pd.to_numeric(df["Crash_Conf"], errors="coerce")
            df["PredLong"] = ((sc >= thr) & (sc >= cc)).astype(int)
            print(f"[debug] PredLong fallback from Spike_Conf with threshold={thr:.3f}")
        else:
            raise SystemExit("Missing columns for evaluation: could not find/derive 'PredLong'.")

    # -------- Outcome labels --------
    if "Outcome" not in df.columns:

        def _outcome_row(r):
            p, a = int(r["PredLong"]), int(r["Actual_Event"])
            if p == 1 and a == 1:
                return "TP"
            if p == 1 and a == 0:
                return "FP"
            if p == 0 and a == 0:
                return "TN"
            if p == 0 and a == 1:
                return "FN"
            return "?"

        df["Outcome"] = df.apply(_outcome_row, axis=1)

    return df


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------
def main():
    print("[eval] Evaluating predictions...")

    # 1) Load labeled data (DuckDB if available, otherwise CSV)
    try:
        df = _from_duckdb()
        source = "duckdb"
    except Exception as e:
        print(f"[eval] DuckDB path unavailable: {e} → falling back to CSV")
        df = _from_csv_pipeline()
        source = "csv"

    # 2) Normalize and derive required columns
    df = _ensure_targets(df)

    # 3) Persist normalized labeled CSV
    LABELED_LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[debug] writing normalized labeled CSV to {LABELED_LOG.resolve()}")
    df.to_csv(LABELED_LOG, index=False)
    print(f"[eval] Labeled data source: {source} → {LABELED_LOG}")

    # 4) Metrics on rows with both prediction and actual
    sub = df.dropna(subset=["PredLong", "Actual_Event"]).copy()
    sub["PredLong"] = sub["PredLong"].astype(int)
    sub["Actual_Event"] = sub["Actual_Event"].astype(int)

    total = len(sub)
    acc = (sub["PredLong"] == sub["Actual_Event"]).mean() if total else float("nan")

    print("\n[eval] Summary")
    print(f"Rows evaluated: {total}")
    print(f"Accuracy:       {acc:.2%}")

    # Distribution diagnostics
    pred_counts = sub["PredLong"].value_counts().to_dict()
    actual_counts = sub["Actual_Event"].value_counts().to_dict()
    print(f"PredLong distribution: {pred_counts}")
    print(f"Actual_Event distribution: {actual_counts}")

    # AUC diagnostics (when Proba is present)
    try:
        thr, invert, _ = _resolve_eval_threshold_and_inversion()
        proba = pd.to_numeric(df.loc[sub.index, "Proba"], errors="coerce")
        if invert:
            proba = 1.0 - proba
            print("[debug] probability inversion applied for AUC calculation")
        auc = roc_auc_score(sub["Actual_Event"], proba)
        label_name = "Proba (inverted)" if invert else "Proba"
        print(f"AUC (using {label_name}): {auc:.4f}")
        if auc < 0.5:
            auc_flip = roc_auc_score(sub["Actual_Event"], 1.0 - proba)
            print(f"AUC if flipped (1-p): {auc_flip:.4f}  (diagnostic)")
    except Exception as e:
        print(f"[eval] AUC computation skipped ({e})")

    bal_acc = balanced_accuracy_score(sub["Actual_Event"], sub["PredLong"])
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # Full classification report
    report = classification_report(
        sub["Actual_Event"], sub["PredLong"], output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(report).transpose()
    print("\n[eval] Classification Report:")
    print(rep_df)

    METRIC_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(METRIC_OUTPUT_CSV)
    print(f"\n[eval] Saved detailed report → {METRIC_OUTPUT_CSV}")

    # Save machine-readable summary for update_metrics_docs.py
    import json as _json
    _summary = {
        "n_rows": len(sub),
        "accuracy": report.get("accuracy", 0.0),
        "auc": float(auc) if "auc" in dir() else None,
        "balanced_accuracy": float(bal_acc),
        "precision_1": report.get("1", {}).get("precision", 0.0),
        "recall_1": report.get("1", {}).get("recall", 0.0),
        "f1_1": report.get("1", {}).get("f1-score", 0.0),
        "support_0": int(report.get("0", {}).get("support", 0)),
        "support_1": int(report.get("1", {}).get("support", 0)),
        "pred_long_0": int((sub["PredLong"] == 0).sum()),
        "pred_long_1": int((sub["PredLong"] == 1).sum()),
    }
    _summary_path = METRIC_OUTPUT_CSV.parent / "evaluate_metrics.json"
    _summary_path.write_text(_json.dumps(_summary, indent=2))
    print(f"[eval] Saved metrics summary → {_summary_path}")


if __name__ == "__main__":
    main()
