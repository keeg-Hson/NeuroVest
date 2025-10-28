# sweep_runner.py
import itertools
import json
import os

import pandas as pd

from backtest import CAPITAL_BASE, run_backtest
from predict import run_predictions

# ── Config ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("configs", exist_ok=True)

# Parameter ranges (tweak as desired)
CONFIDENCE_RANGE: list[float | None] = [None, 0.60, 0.65, 0.70, 0.75]
CRASH_RANGE: list[float | None] = [0.20, 0.25, 0.30, 0.35, 0.40]
SPIKE_RANGE: list[float | None] = [0.20, 0.25, 0.30, 0.35, 0.40]


# Choose optimization objective: "avg_dollar_return" | "final_balance" | "total_return" | "profit_factor" | "win_rate"
OBJECTIVE = "sharpe"  # was "avg_dollar_return"

# Optional: limit backtest to last N days (set to None for full history)
BACKTEST_WINDOW_DAYS = None


def main():
    # 1) Produce predictions once (writes logs/daily_predictions.csv)
    print("▶️ Generating predictions once up-front...")
    pred_df = run_predictions()
    if pred_df is None or pred_df.empty:
        print("🚫 No predictions available. Aborting sweep.")
        return

    # 2) Sweep thresholds
    results = []
    combos = itertools.product(CONFIDENCE_RANGE, CRASH_RANGE, SPIKE_RANGE)

    for conf, crash, spike in combos:
        print(
            f"\n🚦 Backtest with thresholds — " f"confidence={conf}, crash={crash}, spike={spike}"
        )

        trades, metrics, _ = run_backtest(
            window_days=BACKTEST_WINDOW_DAYS,
            crash_thresh=crash,
            spike_thresh=spike,
            confidence_thresh=conf,
            simulate_mode=False,
        )

        # If backtest returned nothing, skip
        if not isinstance(metrics, dict) or metrics.get("trades", 0) == 0:
            print("ℹ️ No trades for this combo — skipping.")
            continue

        # Derive helpful aggregates
        final_balance = (1.0 + metrics.get("total_return", 0.0)) * CAPITAL_BASE
        avg_dollar_return = trades["dollar_return"].mean() if not trades.empty else 0.0

        # Pick an objective
        if OBJECTIVE == "final_balance":
            score = final_balance
        elif OBJECTIVE == "total_return":
            score = metrics.get("total_return", 0.0)
        elif OBJECTIVE == "profit_factor":
            score = metrics.get("profit_factor", 0.0)
        elif OBJECTIVE == "win_rate":
            score = metrics.get("win_rate", 0.0)
        else:  # default
            score = avg_dollar_return

        row = {
            "confidence_thresh": conf,
            "crash_thresh": crash,
            "spike_thresh": spike,
            "trades": metrics.get("trades", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "sharpe": metrics.get("sharpe", float("nan")),
            "max_drawdown": metrics.get("max_drawdown", float("nan")),
            "profit_factor": metrics.get("profit_factor", float("nan")),
            "total_return": metrics.get("total_return", 0.0),
            "avg_dollar_return": avg_dollar_return,
            "final_balance": final_balance,
            "score": score,
        }
        results.append(row)

    if not results:
        print("\n🚫 Sweep finished — no results recorded (no combos produced trades).")
        return

    # 3) Save + report
    df = pd.DataFrame(results).sort_values("score", ascending=False)

    # drop exact duplicates of thresholds+score to clean leaderboard noise
    df = df.drop_duplicates(subset=["confidence_thresh", "crash_thresh", "spike_thresh", "score"])
    df.to_csv("logs/threshold_search.csv", index=False)

    out_csv = "logs/threshold_search.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n🏁 Sweep complete — wrote {out_csv}")
    print(df.head(10))

    # Save best thresholds for downstream use
    best = df.iloc[0]
    best_cfg = {
        "confidence_thresh": (
            None if pd.isna(best["confidence_thresh"]) else float(best["confidence_thresh"])
        ),
        "crash_thresh": None if pd.isna(best["crash_thresh"]) else float(best["crash_thresh"]),
        "spike_thresh": None if pd.isna(best["spike_thresh"]) else float(best["spike_thresh"]),
        "objective": OBJECTIVE,
        "score": float(best["score"]),
    }
    with open("configs/best_thresholds.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print("💾 Best thresholds → configs/best_thresholds.json:", best_cfg)


if __name__ == "__main__":
    main()
