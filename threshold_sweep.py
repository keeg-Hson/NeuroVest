#!/usr/bin/env python3
"""
threshold_sweep.py - align thresholds to forward-returns predictions.

- Detects Spike_Conf range (DuckDB if available; else CSV).
- Sweeps only spike_thresh across that range (crash_thresh ignored).
- Optionally sweeps a couple of backtest knobs (lookahead/tp/sl) if you want.

Outputs:
  logs/threshold_sweep_results.csv
  logs/threshold_search.csv
  configs/best_thresholds.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 1) Make sure prices are fresh
subprocess.run([sys.executable, "update_spy_data.py"], check=False)

# 2) Import backtest AFTER potential env setup
from backtest import run_backtest

DB_PATH = Path("neurovest.duckdb")
PRED_CSV = Path("logs/daily_predictions.csv")
os.makedirs("configs", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def _detect_spike_range() -> tuple[float, float]:
    """Return (lo, hi) for Spike_Conf from either DuckDB or CSV; fallback to sane defaults."""
    lo, hi = 0.50, 0.75  # safe defaults
    try:
        if DB_PATH.exists():
            import duckdb

            con = duckdb.connect(str(DB_PATH))
            row = con.execute("SELECT MIN(Spike_Conf), MAX(Spike_Conf) FROM predictions").fetchone()
            con.close()
            if row and row[0] is not None and row[1] is not None:
                lo, hi = float(row[0]), float(row[1])
        elif PRED_CSV.exists():
            df = pd.read_csv(PRED_CSV)
            if "Spike_Conf" in df.columns:
                s = pd.to_numeric(df["Spike_Conf"], errors="coerce").dropna()
                if not s.empty:
                    lo, hi = float(s.min()), float(s.max())
    except Exception:
        pass
    # pad slightly and clamp to [0,1]
    pad = 0.02
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    # ensure a sensible width
    if hi - lo < 0.05:
        lo = max(0.0, lo - 0.02)
        hi = min(1.0, hi + 0.03)
    return round(lo, 3), round(hi, 3)


def _score_row(metrics: dict, trades_df: pd.DataFrame | None) -> float:
    """
    Composite objective — tweak to taste.
    Examples:
      - avg $ per trade: trades_df["dollar_return"].mean()
      - profit factor: metrics["profit_factor"]
      - Sharpe * win_rate (your original idea)
    """
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    wr = float(metrics.get("win_rate", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    # mildly reward PF and Sharpe, and not reward tiny-sample hacks
    return (pf if np.isfinite(pf) else 0.0) * (0.25 + 0.75 * wr) + 0.5 * sharpe


def sweep(
    spike_grid: list[float],
    min_trades: int = 20,
    window_days: int | None = None,
    lookahead: int = 5,
    tp_atr: float = 1.25,
    sl_atr: float = 0.75,
):
    rows = []
    for t in spike_grid:
        print(f"\n🚦 spike_thresh={t:.3f}")
        trades, metrics, _ = run_backtest(
            window_days=window_days,
            spike_thresh=t,
            crash_thresh=None,  # ignore crash;no crash_conf
            confidence_thresh=None,  # avoid double-gating; spike only
            lookahead=lookahead,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            allow_overlap=False,
            ambig_policy="close_dir",
            fee_bps=2.0,
            slip_bps=3.0,
            atr_len=14,
            margin=0.0,
            use_regime_filter=False,
            use_weekly_trend=False,
            use_atr_band=False,
            trend_len=50,
        )
        n = int(metrics.get("trades", 0) or 0)
        score = -1e9
        if n >= min_trades and trades is not None and not trades.empty:
            score = _score_row(metrics, trades)
        rows.append(
            {
                "spike_thresh": t,
                "trades": n,
                **{
                    k: metrics.get(k)
                    for k in [
                        "total_return",
                        "annualized_return",
                        "sharpe",
                        "win_rate",
                        "avg_return",
                        "median_return",
                        "max_drawdown",
                        "profit_factor",
                    ]
                },
                "score": score,
            }
        )
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df


if __name__ == "__main__":
    lo, hi = _detect_spike_range()
    print(f"🔎 Detected Spike_Conf range ≈ [{lo}, {hi}]")
    # build a grid across that real range
    spike_grid = np.round(np.linspace(lo, hi, 15), 3).tolist()

    df = sweep(
        spike_grid=spike_grid,
        min_trades=20,
        window_days=None,  # or 365*5 to focus on recent
        lookahead=5,
        tp_atr=1.25,
        sl_atr=0.75,
    )

    out_full = Path("logs/threshold_sweep_results.csv")
    out_search = Path("logs/threshold_search.csv")
    out_full.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_full, index=False)
    df.to_csv(out_search, index=False)
    print(f"\n✅ Saved:\n - {out_full}\n - {out_search}")

    if not df.empty and np.isfinite(df["score"].iloc[0]):
        best = df.iloc[0]
        best_json = {
            "crash_thresh": None,  # explicit None; unused
            "spike_thresh": float(best["spike_thresh"]),
            "confidence_thresh": None,  # avoid double gating
        }
        with open("configs/best_thresholds.json", "w") as f:
            json.dump(best_json, f, indent=2)
        print("\n💾 Best thresholds → configs/best_thresholds.json")
        print(best_json)
    else:
        print("\n⚠️ No viable combo met min_trades; try lowering min_trades or widening the grid.")
