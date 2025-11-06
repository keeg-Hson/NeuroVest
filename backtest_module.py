#!/usr/bin/env python3
"""
backtest_module.py

Evaluates and visualizes predictions against SPY. Robust to column names and
merge suffixes. Can restrict to out-of-sample (OOS) via training split metadata.

Env / CLI knobs (env has priority):
- NEUROVEST_PREDICTIONS_CSV   : path to predictions (default: logs/labeled_predictions.csv)
- NEUROVEST_BACKTEST_WINDOW_DAYS : int, only last N days (applied post OOS filter)
- NEUROVEST_OOS_ONLY          : "1" to restrict metrics to OOS (uses models/split_meta.json)

Outputs
- outputs/backtest_plot.png
- outputs/metrics.json (accuracy + classification report text)

Expected prediction columns
- "Prediction" (preferred), else falls back to "Pred" or "Label".
- Optional label column: "True_Label" or "Label" for metric computation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config import LOGS_DIR, MODELS_DIR, SPY_DAILY_CSV

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _pick_first(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load SPY
    spy = pd.read_csv(SPY_DAILY_CSV, low_memory=False, parse_dates=["Date"])
    spy = spy.sort_values("Date").reset_index(drop=True)

    # Load predictions
    default_preds = LOGS_DIR / "labeled_predictions.csv"
    preds_path = Path(os.getenv("NEUROVEST_PREDICTIONS_CSV", default_preds))
    if not preds_path.exists():
        raise SystemExit(f"Predictions file not found: {preds_path}")

    preds = pd.read_csv(preds_path, low_memory=False)
    if "Date" in preds.columns:
        preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce")
        preds = preds.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Ensure we have a prediction-like column
    pred_col = _pick_first(preds.columns.tolist(), ["Prediction", "Pred", "Label"])
    if pred_col is None:
        raise SystemExit(
            "No prediction-like column found in predictions (need one of Prediction/Pred/Label)."
        )
    if pred_col != "Prediction":
        preds["Prediction"] = preds[pred_col]

    # Optional label remap (idempotent if file missing)
    label_map_path = MODELS_DIR / "label_map_fwd.json"
    if label_map_path.exists():
        try:
            with open(label_map_path) as f:
                label_map = json.load(f)
            preds["Prediction"] = preds["Prediction"].map(label_map).fillna(preds["Prediction"])
        except Exception:
            pass

    return spy, preds


def _apply_oos_and_window(merged: pd.DataFrame) -> pd.DataFrame:
    """Apply OOS filter (if NEUROVEST_OOS_ONLY=1) and window filter (NEUROVEST_BACKTEST_WINDOW_DAYS)."""
    df = merged.copy()

    # OOS filter
    if os.getenv("NEUROVEST_OOS_ONLY", "0") == "1":
        split_meta = MODELS_DIR / "split_meta.json"
        if split_meta.exists():
            meta = json.loads(split_meta.read_text())
            split_date = pd.to_datetime(meta["split_date"], errors="coerce")
            if pd.notna(split_date):
                df = df[df["Date"] >= split_date]
                print(f"[eval] using OOS only from {split_date.date()}")
        else:
            print("[eval] split_meta.json not found; evaluating full history.")

    # Window filter
    win = os.getenv("NEUROVEST_BACKTEST_WINDOW_DAYS")
    if win:
        try:
            n = int(win)
            if n > 0:
                if not df.empty:
                    latest = df["Date"].max()
                    cutoff = latest - pd.Timedelta(days=n)
                    df = df[df["Date"] >= cutoff]
                    print(f"[eval] window: last {n} days (since {cutoff.date()})")
        except Exception:
            pass

    return df


def plot_predictions(merged: pd.DataFrame, savepath: Path) -> None:
    df = merged.copy()

    # Pick a price column robustly
    price_col = _pick_first(
        df.columns.tolist(),
        ["Close", "AdjClose", "Close_SPY", "AdjClose_SPY", "Close_y", "Close_x"],
    )
    if price_col is None:
        close_like = [c for c in df.columns if "close" in str(c).lower()]
        if not close_like:
            raise SystemExit("Could not find a Close/AdjClose column for plotting.")
        price_col = close_like[0]

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df[price_col], label=f"SPY ({price_col})", alpha=0.7)

    # Shade long periods where Prediction==1
    if "Prediction" in df.columns:
        sig = df["Prediction"].fillna(0).astype(float)
        ymin, ymax = df[price_col].min(), df[price_col].max()
        plt.fill_between(df["Date"], ymin, ymax, where=sig > 0, alpha=0.09, label="Long")

    # Optional event markers
    for col, color, marker, label in [
        ("Spike", "green", "^", "Spike"),
        ("Crash", "red", "v", "Crash"),
    ]:
        if col in df.columns:
            pts = df[df[col] == 1]
            if not pts.empty:
                plt.scatter(pts["Date"], pts[price_col], c=color, marker=marker, s=50, label=label)

    plt.title("SPY vs Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"[plot] saved → {savepath}")


def evaluate_predictions(df: pd.DataFrame) -> dict:
    """Return metrics dict; prints report if labels available."""
    label_col = _pick_first(df.columns.tolist(), ["True_Label", "Label"])
    if label_col is None:
        print("[eval] No ground-truth labels (True_Label/Label) found — skipping metrics.")
        return {"has_labels": False}

    sub = df[(df["Prediction"].notna()) & (df[label_col].notna())].copy()
    if sub.empty:
        print("[eval] No overlapping rows with both Prediction and labels — skipping.")
        return {"has_labels": False}

    y_true = sub[label_col].astype(int)
    y_pred = sub["Prediction"].astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    print(f"[eval] Accuracy: {acc:.3f}")
    try:
        report_txt = classification_report(y_true, y_pred, digits=3)
        print(report_txt)
    except Exception as e:
        report_txt = f"classification_report failed: {e}"
        print("[eval]", report_txt)

    return {
        "has_labels": True,
        "accuracy": acc,
        "n": int(len(sub)),
        "label_col": label_col,
        "report": report_txt,
    }


def run_backtest() -> None:
    spy_df, preds_df = load_data()

    # Normalize dates
    spy_df["Date"] = pd.to_datetime(spy_df["Date"], errors="coerce").dt.normalize()
    preds_df["Date"] = pd.to_datetime(preds_df["Date"], errors="coerce").dt.normalize()
    preds_df = preds_df.dropna(subset=["Date"])

    # Merge (inner keeps common dates only)
    merged = pd.merge(spy_df, preds_df, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Apply OOS + window filters
    merged = _apply_oos_and_window(merged)

    print("Merged Columns:", merged.columns.tolist())

    # Plot
    plot_predictions(merged, OUT_DIR / "backtest_plot.png")

    # Evaluate
    metrics = evaluate_predictions(merged)

    # Persist metrics
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[eval] metrics saved → {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    run_backtest()
