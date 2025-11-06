"""
tearsheet.py

Generates a lightweight performance tear sheet after a run.
- Loads SPY prices and a predictions file (default: logs/labeled_predictions.csv).
- Builds a simple long/flat equity curve: long next day when Prediction==1 (fallback to Label/Pred).
- Computes CAGR, Sharpe (annualized from daily), and Max Drawdown.
- Saves an HTML file with metrics and an embedded PNG chart to outputs/tearsheet_YYYY-MM-DD.html.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LOGS_DIR, OUTPUT_DIR, SPY_DAILY_CSV


def _metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    # CAGR
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    # Sharpe (daily → annualized)
    mu, sigma = rets.mean(), rets.std()
    sharpe = 0.0 if sigma == 0 else (mu / sigma) * np.sqrt(252)
    # Max Drawdown
    run_max = equity.cummax()
    dd = (equity / run_max - 1).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd}


def _load_prices() -> pd.DataFrame:
    df = pd.read_csv(SPY_DAILY_CSV)
    if "Date" not in df.columns:
        for c in list(df.columns)[:3]:
            if str(c).lower() == "date":
                df = df.rename(columns={c: "Date"})
                break
    # choose price
    cands = [c for c in df.columns if str(c).lower() in ("adjclose", "adj close", "close")]
    if not cands:
        cands = [c for c in df.columns if "close" in str(c).lower()]
    px = cands[0]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", px]].rename(columns={px: "Close"}).sort_values("Date").reset_index(drop=True)
    return df


def _load_preds(path: str | None = None) -> pd.DataFrame:
    path = path or (LOGS_DIR / "labeled_predictions.csv")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        raise SystemExit("Predictions file needs a Date column.")
    # unify prediction column
    pred_col = "Prediction"
    if pred_col not in df.columns:
        for c in ("Pred", "Label"):
            if c in df.columns:
                df[pred_col] = df[c]
                break
    if pred_col not in df.columns:
        raise SystemExit("Could not find a prediction-like column (Prediction/Pred/Label).")
    return df[["Date", pred_col]].rename(columns={pred_col: "Prediction"})


def build_equity(spy: pd.DataFrame, preds: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Simple long/flat: go long next day if Prediction==1, else flat."""
    df = spy.merge(preds, on="Date", how="left").sort_values("Date")
    df["Prediction"] = df["Prediction"].ffill().fillna(0)
    # Shift signal forward one day (trade on next open/close approx)
    sig = df["Prediction"].shift(1).fillna(0)
    # Daily returns from Close-to-Close
    rets = df["Close"].pct_change(fill_method=None).fillna(0.0)
    strat = (1 + sig * rets).cumprod()
    bench = (1 + rets).cumprod()
    out = pd.DataFrame({"Date": df["Date"], "Equity": strat, "Benchmark": bench}).set_index("Date")
    return out


def render_tearsheet(equity_df: pd.DataFrame, out_html: str) -> None:
    # Compute metrics
    m = _metrics(equity_df["Equity"])
    b = _metrics(equity_df["Benchmark"])

    # Plot (single chart; no explicit colors)
    fig, ax = plt.subplots(figsize=(10, 5))
    equity_df["Equity"].plot(ax=ax)
    equity_df["Benchmark"].plot(ax=ax)
    ax.set_title("Equity vs Benchmark (SPY)")
    ax.set_ylabel("Cumulative Value (normalized)")
    ax.grid(True, linestyle="--", alpha=0.4)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

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
<img src="data:image/png;base64,{img_b64}" alt="Equity Chart" />
</body>
</html>
    """.strip()

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main(preds_path: str | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spy = _load_prices()
    preds = _load_preds(preds_path)
    eq = build_equity(spy, preds)
    out_file = OUTPUT_DIR / f"tearsheet_{datetime.now().strftime('%Y-%m-%d')}.html"
    render_tearsheet(eq, str(out_file))
    print(f"[tearsheet] wrote → {out_file}")


if __name__ == "__main__":
    main()
