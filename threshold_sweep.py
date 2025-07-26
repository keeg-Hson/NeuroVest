#threshold_sweep.py
import pandas as pd
import numpy as np
import json
import os
from backtest import run_backtest
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

#auto update SPY data before anything else
subprocess.run(["python3", "update_spy_data.py"])


# ensure configs folder exists
os.makedirs("configs", exist_ok=True)

# Define threshold sweep function
def sweep_thresholds(crash_thresh_list, spike_thresh_list, confidence_thresh_list):
    results = []

    for crash in crash_thresh_list:
        for spike in spike_thresh_list:
            for conf in confidence_thresh_list:
                print(f"\n🚦 Testing thresholds — Crash: {crash}, Spike: {spike}, Confidence: {conf}")

                trades, metrics, _ = run_backtest(
                    crash_thresh=crash,
                    spike_thresh=spike,
                    confidence_thresh=conf,
                )

                result = {
                    "crash_thresh": crash,
                    "spike_thresh": spike,
                    "confidence_thresh": conf,
                    **metrics
                }
                results.append(result)

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Threshold Ranges (adjustable)
    crash_range = [1.5, 2.0, 2.5]
    spike_range = [1.5, 2.0, 2.5]
    confidence_range = [0.5, 0.6, 0.7, 0.8]  # 🔧 NEW sweep dimension

    # Run sweep
    df_sweep = sweep_thresholds(crash_range, spike_range, confidence_range)

    # Save to CSV
    df_sweep.to_csv("logs/threshold_sweep_results.csv", index=False)
    print("\n✅ Saved sweep results to logs/threshold_sweep_results.csv")

    # 🧠 Composite scoring: prioritize good Sharpe + win rate
    df_sweep["composite_score"] = df_sweep["sharpe"].fillna(0) * df_sweep["win_rate"].fillna(0)
    best = df_sweep.sort_values("composite_score", ascending=False).head(1).iloc[0]

    # Save best config
    best_config = {
        "crash_thresh": best["crash_thresh"],
        "spike_thresh": best["spike_thresh"],
        "confidence_thresh": best["confidence_thresh"]
    }

    with open("configs/best_thresholds.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("\n💾 Best thresholds saved to configs/best_thresholds.json:")
    print(best_config)

    # Print Top 5
    print("\n🏆 Top Composite Thresholds:")
    print(df_sweep.sort_values("composite_score", ascending=False).head(5))

    # Optional: Heatmap Visuals (aggregated over confidence threshold)
    df_avg = df_sweep.groupby(["crash_thresh", "spike_thresh"]).agg({
        "total_return": "mean",
        "win_rate": "mean",
        "profit_factor": "mean",
        "trades": "sum"
    }).reset_index()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 6))

    pivot_return = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="total_return")
    sns.heatmap(pivot_return, annot=True, fmt=".2%", cmap="viridis", ax=ax1, vmin=0, vmax=0.02)
    ax1.set_title("Avg Return by Crash/Spike")

    pivot_wr = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="win_rate")
    sns.heatmap(pivot_wr, annot=True, fmt=".0%", cmap="coolwarm", ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Avg Win Rate by Crash/Spike")

    df_avg["profit_factor_plot"] = df_avg["profit_factor"].replace(np.inf, 10)
    pivot_pf = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="profit_factor_plot")
    sns.heatmap(pivot_pf, annot=True, fmt=".2f", cmap="magma", ax=ax3)
    ax3.set_title("Avg Profit Factor")

    pivot_trades = df_avg.pivot(index="crash_thresh", columns="spike_thresh", values="trades")
    sns.heatmap(pivot_trades, annot=True, fmt=".0f", cmap="Greens", ax=ax4)
    ax4.set_title("Trade Count")

    plt.tight_layout()
    plt.show()