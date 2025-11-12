#!/usr/bin/env python3
"""
NeuroVest — tear sheet generator

Purpose
-------
Builds a lightweight performance tear sheet from SPY prices and model outputs.
Calculates a simple long/flat equity curve, reports summary metrics, and writes
an HTML file with metrics and an embedded chart.

Label convention (legacy)
-------------------------
Predictions and labels use a 3-class convention:
    0 = CRASH
    1 = NORMAL
    2 = SPIKE

Trading rule (forward-returns / legacy)
---------------------------------------
The active forward-returns model is binary internally ({0,1} = no-trade/trade)
and is written to logs using the legacy mapping:
    binary 0 → legacy 1 (NORMAL)
    binary 1 → legacy 2 (SPIKE)

The tear sheet interprets predictions as:
    go long next day if legacy Prediction == 2 (SPIKE), else stay flat.

Outputs
-------
- Cumulative strategy equity and SPY benchmark
- CAGR, Sharpe (annualized from daily), Max Drawdown
- Optional classification metrics if a Label column is available
- HTML saved to outputs/tearsheet_YYYY-MM-DD.html
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LOGS_DIR, OUTPUT_DIR, SPY_DAILY_CSV


# =============================================================================
# Metrics
# =============================================================================
def _metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    mu, sigma = rets.mean(), rets.std()
    sharpe = 0.0 if sigma == 0 else (mu / sigma) * np.sqrt(252)
    run_max = equity.cummax()
    dd = (equity / run_max - 1).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(dd)}


def _class_report(df_preds: pd.DataFrame) -> dict | None:
    """
    If Label is present alongside Prediction, compute simple class metrics
    under the legacy 0/1/2 convention. Returns None if insufficient data.
    """
    if "Label" not in df_preds.columns:
        return None
    tmp = df_preds.dropna(subset=["Label", "Prediction"]).copy()
    if tmp.empty:
        return None
    y_true = tmp["Label"].astype(int).values
    y_pred = tmp["Prediction"].astype(int).values

    # per-class precision/recall/f1 (micro-safe, no sklearn dependency here)
    report = {}
    classes = [0, 1, 2]
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        report[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(np.sum(y_true == c)),
        }
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    report["accuracy"] = acc
    return report


# =============================================================================
# Loaders
# =============================================================================
def _load_prices() -> pd.DataFrame:
    df = pd.read_csv(SPY_DAILY_CSV, low_memory=False)
    if "Date" not in df.columns:
        for c in list(df.columns)[:3]:
            if str(c).strip().lower() == "date":
                df = df.rename(columns={c: "Date"})
                break
    candidates = [c for c in df.columns if str(c).lower() in ("adjclose", "adj close", "close")]
    if not candidates:
        candidates = [c for c in df.columns if "close" in str(c).lower()]
    if not candidates:
        raise SystemExit("No price column found in SPY_DAILY_CSV.")
    px = candidates[0]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df[["Date", px]].rename(columns={px: "Close"})
    return df


def _load_preds(path: str | None = None) -> pd.DataFrame:
    """
    Loads predictions and preserves useful columns when available.
    Expected columns (best effort): Date, Prediction (legacy 0/1/2), Label, Proba, Spike_Conf, Crash_Conf.
    """
    path = path or (LOGS_DIR / "labeled_predictions.csv")
    df = pd.read_csv(path, low_memory=False)
    if "Date" not in df.columns:
        raise SystemExit("Predictions file needs a Date column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Normalize a single 'Prediction' column (prefer explicit, then fallback)
    pred_col = None
    for c in ("Prediction", "Pred", "Label"):
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        raise SystemExit("Could not find a prediction-like column (Prediction/Pred/Label).")

    keep = ["Date", pred_col]
    for extra in ("Label", "Proba", "Spike_Conf", "Crash_Conf", "Confidence"):
        if extra in df.columns:
            keep.append(extra)
    out = df[keep].rename(columns={pred_col: "Prediction"})
    # ensure int type where appropriate
    out["Prediction"] = pd.to_numeric(out["Prediction"], errors="coerce").astype("Int64")
    if "Label" in out.columns:
        out["Label"] = pd.to_numeric(out["Label"], errors="coerce").astype("Int64")
    return out


# =============================================================================
# Equity construction
# =============================================================================
def build_equity(spy: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Long/flat strategy:
      - Interpret legacy predictions: long next day if Prediction == 2 (SPIKE), else flat.
      - Close-to-close returns, signal is shifted by +1 day to avoid lookahead.
    """
    df = spy.merge(preds, on="Date", how="left").sort_values("Date")
    # Forward fill sparse predictions; default to NORMAL (1) if no signal yet
    df["Prediction"] = df["Prediction"].ffill().fillna(1).astype(int)

    # Legacy mapping → position: 1 if SPIKE(2), else 0
    pos = (df["Prediction"] == 2).astype(float)

    # Shift to apply on next day
    pos_shifted = pos.shift(1).fillna(0.0)

    # Daily close-to-close returns
    rets = df["Close"].pct_change(fill_method=None).fillna(0.0)

    # Equity and benchmark
    strat = (1 + pos_shifted * rets).cumprod()
    bench = (1 + rets).cumprod()

    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "Equity": strat,
            "Benchmark": bench,
            "Position": pos_shifted,
            "Prediction": df["Prediction"],
        }
    ).set_index("Date")
    return out


# =============================================================================
# Render
# =============================================================================
def render_tearsheet(equity_df: pd.DataFrame, preds_raw: pd.DataFrame, out_html: str) -> None:
    # Compute metrics
    m = _metrics(equity_df["Equity"])
    b = _metrics(equity_df["Benchmark"])

    # Optional classification report
    cls = _class_report(preds_raw)

    # Plot (single chart; no explicit colors)
    fig, ax = plt.subplots(figsize=(10, 5))
    equity_df["Equity"].plot(ax=ax, label="Strategy")
    equity_df["Benchmark"].plot(ax=ax, label="SPY")
    ax.set_title("Equity vs Benchmark (SPY)")
    ax.set_ylabel("Cumulative Value (normalized)")
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Optional classification metrics HTML
    def _cls_html(report: dict | None) -> str:
        if not report:
            return "<p><i>No labels available for classification metrics.</i></p>"
        rows = []
        for c in (0, 1, 2):
            r = report.get(c, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
            rows.append(
                f"<tr><td>{c}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
                f"<td>{r['f1']:.3f}</td><td>{r['support']}</td></tr>"
            )
        acc = report.get("accuracy", 0.0)
        table = f"""
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        <p><b>Accuracy:</b> {acc:.3f}</p>
        """
        return table

    # HTML
    html = f"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>NeuroVest Tear Sheet</title></head>
<body>
<h2>NeuroVest Tear Sheet</h2>
<p><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<h3>Strategy Metrics</h3>
<ul>
  <li><b>CAGR:</b> {m['CAGR']:.2%}</li>
  <li><b>Sharpe:</b> {m['Sharpe']:.2f}</li>
  <li><b>Max Drawdown:</b> {m['MaxDD']:.2%}</li>
</ul>

<h3>Benchmark (SPY) Metrics</h3>
<ul>
  <li><b>CAGR:</b> {b['CAGR']:.2%}</li>
  <li><b>Sharpe:</b> {b['Sharpe']:.2f}</li>
  <li><b>Max Drawdown:</b> {b['MaxDD']:.2%}</li>
</ul>

<h3>Classification (legacy 0/1/2) — if available</h3>
{_cls_html(cls)}

<img src="data:image/png;base64,{img_b64}" alt="Equity Chart" />
</body>
</html>
    """.strip()

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
# Main
# =============================================================================
def main(preds_path: str | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spy = _load_prices()
    preds = _load_preds(preds_path)
    eq = build_equity(spy, preds)
    out_file = OUTPUT_DIR / f"tearsheet_{datetime.now().strftime('%Y-%m-%d')}.html"
    render_tearsheet(eq, preds, str(out_file))
    print(f"[tearsheet] wrote → {out_file}")


if __name__ == "__main__":
    main()
