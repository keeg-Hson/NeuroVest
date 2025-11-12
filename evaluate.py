#!/usr/bin/env python3
# evaluate.py - robust evaluation with auto-derivation of Actuals from SPY prices

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score

# Align actuals horizon with training
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


# -------- helpers (hurdle & invert/threshold) --------
def _train_hurdle() -> float:
    fee_bps = float(TRAIN_CFG.get("fee_bps", 1.5))
    slip_bps = float(TRAIN_CFG.get("slippage_bps", 2.0))
    edge_bps = float(TRAIN_CFG.get("min_edge_bps", 10.0))
    return (fee_bps + slip_bps + edge_bps) / 10_000.0


def _resolve_eval_threshold_and_inversion() -> tuple[float, bool, str]:
    candidates = [
        Path("configs/best_thresholds.json"),  # {"spike_thresh": 0.53, "invert_proba": true}
        Path("models/thresholds.json"),  # {"threshold": 0.55}
        Path("models/thresholds_fwd.json"),  # {"p_min": 0.60}
    ]
    thr, invert, src = None, False, "fallback"
    for p in candidates:
        try:
            obj = json.loads(Path(p).read_text())
            if p.name == "best_thresholds.json":
                if obj.get("spike_thresh") is not None:
                    thr = float(obj["spike_thresh"])
                    invert = bool(obj.get("invert_proba", False))
                    src = str(p)
                    break
            elif "threshold" in obj:
                thr = float(obj["threshold"])
                src = str(p)
                break
            elif "p_min" in obj:
                thr = float(obj["p_min"])
                src = str(p)
                break
        except Exception:
            pass
    if thr is None:
        thr = 0.55
    thr = max(0.10, min(0.90, float(thr)))
    print(f"[debug] evaluate threshold = {thr:.3f} (source: {src})")
    print(f"[debug] probability inversion {'ON' if invert else 'OFF'}")
    return thr, invert, src


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


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_binary_series(s: pd.Series) -> pd.Series:
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


def _ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" not in df.columns:
        raise SystemExit("Missing 'Date' column in labeled_predictions.csv.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # Actual_Event
    actual_col = _first_present(df, ["Actual_Event", "Actual_Event_1d", "Actual", "Target", "y"])
    if actual_col is not None:
        df["Actual_Event"] = _coerce_binary_series(df[actual_col])
    else:
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
            px, src = _load_price_series()
            px = px.sort_values("Date").reset_index(drop=True)
            px["PX_NEXT"] = px["PX"].shift(-H)
            col_name = f"FwdRet_{H}d"
            px[col_name] = px["PX_NEXT"] / px["PX"] - 1.0
            df = df.merge(px[["Date", col_name]], on="Date", how="left", validate="m:1")
            if df[col_name].isna().all():
                raise SystemExit(
                    "Could not derive 'Actual_Event': join with price file produced no matches.\n"
                    f"- Ensure your prediction CSV 'Date' matches the dates in the price file.\n"
                    f"- Price source tried: {src}"
                )
            df["Actual_Event"] = (pd.to_numeric(df[col_name], errors="coerce") > hurdle).astype(int)
            print(f"[debug] derived Actual_Event with H={H}, hurdle={hurdle:.5f} (source: {src})")

    # PredLong from Proba (with threshold + inversion)
    thr, invert, _ = _resolve_eval_threshold_and_inversion()
    if "Proba" in df.columns:
        p = pd.to_numeric(df["Proba"], errors="coerce")
        if invert:
            p = 1.0 - p
        df["PredLong"] = (p >= thr).astype(int)
        print(
            f"[debug] PredLong derived from Proba using threshold={thr:.3f}{' (inverted)' if invert else ''}"
        )
    else:
        pred_col = _first_present(df, ["PredLong", "Prediction", "Pred", "Signal", "Position"])
        if pred_col is None:
            if {"Spike_Conf", "Crash_Conf"} <= set(df.columns):
                sc = pd.to_numeric(df["Spike_Conf"], errors="coerce")
                cc = pd.to_numeric(df["Crash_Conf"], errors="coerce")
                df["PredLong"] = ((sc >= 0.5) & (sc >= cc)).astype(int)
            elif "ProbaLong" in df.columns:
                p = pd.to_numeric(df["ProbaLong"], errors="coerce")
                if invert:
                    p = 1.0 - p
                df["PredLong"] = (p >= thr).astype(int)
            else:
                raise SystemExit(
                    "Missing columns for evaluation: could not find/derive 'PredLong'."
                )
        else:
            df["PredLong"] = _coerce_binary_series(df[pred_col])

    # Outcome
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


def main():
    print("📊 Evaluating predictions...")
    # 1) load
    try:
        df = _from_duckdb()
        source = "duckdb"
    except Exception as e:
        print(f"[ℹ️] DuckDB path unavailable: {e} → falling back to CSV")
        df = _from_csv_pipeline()
        source = "csv"

    # 2) normalize/derive required columns
    df = _ensure_targets(df)

    # 3) persist normalized labeled CSV
    LABELED_LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[debug] writing normalized labeled CSV to {LABELED_LOG.resolve()}")
    df.to_csv(LABELED_LOG, index=False)
    print(f"✅ Labeled data ({source}) → {LABELED_LOG}")

    # 4) metrics
    sub = df.dropna(subset=["PredLong", "Actual_Event"]).copy()
    sub["PredLong"] = sub["PredLong"].astype(int)
    sub["Actual_Event"] = sub["Actual_Event"].astype(int)

    total = len(sub)
    acc = (sub["PredLong"] == sub["Actual_Event"]).mean() if total else float("nan")
    print("\n📋 Summary")
    print(f"Rows evaluated: {total}")
    print(f"Accuracy:       {acc:.2%}")

    try:
        thr, invert, _ = _resolve_eval_threshold_and_inversion()
        proba = pd.to_numeric(df.loc[sub.index, "Proba"], errors="coerce")
        if invert:
            proba = 1.0 - proba
            print("[debug] probability inversion ON (source: configs/best_thresholds.json)")
        auc = roc_auc_score(sub["Actual_Event"], proba)
        lab = "Proba (inverted)" if invert else "Proba"
        print(f"AUC (using {lab}): {auc:.4f}")
        if auc < 0.5:
            auc_flip = roc_auc_score(sub["Actual_Event"], 1.0 - proba)
            print(f"AUC if flipped (1-p): {auc_flip:.4f}  ← diagnostic")
    except Exception:
        pass

    print(f"Balanced Accuracy: {balanced_accuracy_score(sub['Actual_Event'], sub['PredLong']):.4f}")

    report = classification_report(
        sub["Actual_Event"], sub["PredLong"], output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(report).transpose()
    print("\n🔍 Classification Report:")
    print(rep_df)

    METRIC_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(METRIC_OUTPUT_CSV)
    print(f"\n💾 Saved detailed report → {METRIC_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
