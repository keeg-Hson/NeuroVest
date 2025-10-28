# sweep_optimizer.py
import json
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from backtest import CAPITAL_BASE, run_backtest
from predict import run_predictions

os.makedirs("logs", exist_ok=True)
os.makedirs("configs", exist_ok=True)

# Dense ranges (adjust as you like)
SPIKE_GRID: Iterable[float] = np.arange(0.50, 0.91, 0.05)
CRASH_GRID: Iterable[float] = np.arange(0.50, 0.91, 0.05)
CONF_GRID: Iterable[float | None] = [None, 0.50, 0.60, 0.70, 0.80]

OBJECTIVE = "final_balance"  # "avg_dollar_return" | "final_balance" | "total_return" | "win_rate" | "profit_factor"
BACKTEST_WINDOW_DAYS = None  # set to N (e.g., 365) to speed up


def main():
    # Generate predictions once
    print("▶️ Generating predictions once up-front...")
    pred_df = run_predictions()
    if pred_df is None or pred_df.empty:
        print("🚫 No predictions available. Aborting sweep.")
        return

    rows = []
    for spike in SPIKE_GRID:
        for crash in CRASH_GRID:
            for conf in CONF_GRID:
                print(
                    f"\n🔎 Testing thresholds — spike={spike:.2f}, crash={crash:.2f}, confidence={conf}"
                )
                trades, metrics, _ = run_backtest(
                    window_days=BACKTEST_WINDOW_DAYS,
                    crash_thresh=float(crash),
                    spike_thresh=float(spike),
                    confidence_thresh=None if conf is None else float(conf),
                    simulate_mode=False,
                )

                n = metrics.get("trades", 0) if isinstance(metrics, dict) else 0
                if n == 0:
                    print("ℹ️ No trades — skipping.")
                    continue

                final_balance = (1.0 + metrics.get("total_return", 0.0)) * CAPITAL_BASE
                avg_dollar_return = trades["dollar_return"].mean() if not trades.empty else 0.0

                if OBJECTIVE == "avg_dollar_return":
                    score = avg_dollar_return
                elif OBJECTIVE == "total_return":
                    score = metrics.get("total_return", 0.0)
                elif OBJECTIVE == "win_rate":
                    score = metrics.get("win_rate", 0.0)
                elif OBJECTIVE == "profit_factor":
                    score = metrics.get("profit_factor", 0.0)
                else:  # final_balance
                    score = final_balance

                rows.append(
                    {
                        "spike_thresh": float(spike),
                        "crash_thresh": float(crash),
                        "confidence_thresh": None if conf is None else float(conf),
                        "trades": n,
                        "win_rate": metrics.get("win_rate", 0.0),
                        "sharpe": metrics.get("sharpe", float("nan")),
                        "max_drawdown": metrics.get("max_drawdown", float("nan")),
                        "profit_factor": metrics.get("profit_factor", float("nan")),
                        "total_return": metrics.get("total_return", 0.0),
                        "avg_dollar_return": avg_dollar_return,
                        "final_balance": final_balance,
                        "score": score,
                    }
                )

    if not rows:
        print("\n🚫 No combinations produced trades — nothing to save.")
        return

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_csv = "logs/threshold_leaderboard.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n🏁 Sweep complete! Top results written to {out_csv}")
    print(df.head(10))

    best = df.iloc[0]
    best_cfg = {
        "confidence_thresh": best["confidence_thresh"],
        "crash_thresh": best["crash_thresh"],
        "spike_thresh": best["spike_thresh"],
        "objective": OBJECTIVE,
        "score": float(best["score"]),
    }
    with open("configs/best_thresholds.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print("💾 Best thresholds → configs/best_thresholds.json:", best_cfg)


if __name__ == "__main__":
    main()
