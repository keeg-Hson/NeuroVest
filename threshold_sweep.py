# threshold_sweep.py
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backtest import run_backtest

# auto update SPY data before anything else
subprocess.run([sys.executable, "update_spy_data.py"], check=True)


# ensure configs folder exists
os.makedirs("configs", exist_ok=True)


# Define threshold sweep function
def sweep_thresholds(crash_thresh_list, spike_thresh_list, confidence_thresh_list):
    results = []

    for crash in crash_thresh_list:
        for spike in spike_thresh_list:
            for conf in confidence_thresh_list:
                print(
                    f"\n🚦 Testing thresholds — Crash: {crash}, Spike: {spike}, Confidence: {conf}"
                )

                trades, metrics, _ = run_backtest(
                    crash_thresh=crash,
                    spike_thresh=spike,
                    confidence_thresh=conf,
                )

                result = {
                    "crash_thresh": crash,
                    "spike_thresh": spike,
                    "confidence_thresh": conf,
                    **metrics,
                }
                results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Ranges (probabilities)
    crash_range = [0.6, 0.7, 0.8, 0.9]
    spike_range = [0.6, 0.7, 0.8, 0.9]
    confidence_range = [0.6, 0.7, 0.8]

    df_sweep = sweep_thresholds(crash_range, spike_range, confidence_range)

    # Save to CSV
    df_sweep.to_csv("logs/threshold_sweep_results.csv", index=False)
    print("\n✅ Saved sweep results to logs/threshold_sweep_results.csv")

    # Ensure needed columns exist (older backtest runs might omit some)
    for col in ["sharpe", "win_rate", "total_return", "profit_factor", "trades"]:
        if col not in df_sweep.columns:
            df_sweep[col] = 0.0

    # 🧠 Composite scoring: prioritize good Sharpe + win rate
    df_sweep["composite_score"] = df_sweep["sharpe"].fillna(0) * df_sweep["win_rate"].fillna(0)

    # Pick best
    best = df_sweep.sort_values("composite_score", ascending=False).head(1).iloc[0]

    # canonical file
    df_sweep["score"] = df_sweep.get("avg_return", 0.0)  # or compute avg_dollar like in runner
    df_sweep.to_csv("logs/threshold_search.csv", index=False)
    print("\n✅ Saved sweep to logs/threshold_search.csv")

    # best json
    best = df_sweep.sort_values("score", ascending=False).head(1).iloc[0]
    best_config = {
        "crash_thresh": float(best["crash_thresh"]),
        "spike_thresh": float(best["spike_thresh"]),
        "confidence_thresh": float(best["confidence_thresh"]),
    }
    with open("configs/best_thresholds.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("\n💾 Best thresholds saved to configs/best_thresholds.json:")
    print(best_config)

    # Print Top 5
    try:
        print("\n🏆 Top Composite Thresholds:")
        print(df_sweep.sort_values("composite_score", ascending=False).head(5))
    except KeyError:
        print(
            "\nℹ️ 'composite_score' missing — likely no valid rows. Check logs/threshold_sweep_results.csv"
        )

    # Optional: Heatmap Visuals (aggregated over confidence threshold)
    df_avg = (
        df_sweep.groupby(["crash_thresh", "spike_thresh"])
        .agg({"total_return": "mean", "win_rate": "mean", "profit_factor": "mean", "trades": "sum"})
        .reset_index()
    )

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 6))

    pivot_return = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="total_return")
    sns.heatmap(pivot_return, annot=True, fmt=".2%", cmap="viridis", ax=ax1, vmin=0, vmax=0.02)
    ax1.set_title("Avg Return by Crash/Spike")

    pivot_wr = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="win_rate")
    sns.heatmap(pivot_wr, annot=True, fmt=".0%", cmap="coolwarm", ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Avg Win Rate by Crash/Spike")

    df_avg["profit_factor_plot"] = df_avg["profit_factor"].replace(np.inf, 10)
    pivot_pf = df_avg.pivot(
        index="crash_thresh", columns="spike_thresh", values="profit_factor_plot"
    )
    sns.heatmap(pivot_pf, annot=True, fmt=".2f", cmap="magma", ax=ax3)
    ax3.set_title("Avg Profit Factor")

    pivot_trades = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="trades")
    sns.heatmap(pivot_trades, annot=True, fmt=".0f", cmap="Greens", ax=ax4)
    ax4.set_title("Trade Count")

    plt.tight_layout()
    plt.show()
