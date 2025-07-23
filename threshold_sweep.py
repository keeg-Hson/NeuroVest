#threshold_sweep.py
import pandas as pd
import numpy as np
from backtest import run_backtest
import seaborn as sns
import matplotlib.pyplot as plt

# Define threshold sweep function
def sweep_thresholds(crash_thresh_list, spike_thresh_list):
    results = []

    for crash in crash_thresh_list:
        for spike in spike_thresh_list:
            print(f"\n🚦 Testing thresholds — Crash: {crash}, Spike: {spike}")

            # NOTE: In full integration, you'd modify model thresholding here,
            # or load pre-labeled predictions according to these thresholds.
            # For now, assume prediction file already uses generic predictions.

            # Suggestion: Only use simulate_mode=True for single manual test runs, not sweeping
            trades, metrics, _ = run_backtest(
                crash_thresh=crash,
                spike_thresh=spike,
                #simulate_mode=False   # << Disable simulate_mode for sweep
            )

            result = {
                "crash_thresh": crash,
                "spike_thresh": spike,
                **metrics
            }
            results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example threshold ranges
    crash_range = [1.5, 2.0, 2.5]
    spike_range = [1.5, 2.0, 2.5]

    # Run sweep
    df_sweep = sweep_thresholds(crash_range, spike_range)

    # Save results to CSV
    df_sweep.to_csv("logs/threshold_sweep_results.csv", index=False)
    print("\n✅ Saved sweep results to logs/threshold_sweep_results.csv")

    # Show top Sharpe combos
    top = df_sweep.sort_values("sharpe", ascending=False).head()
    print("\n🏆 Top Threshold Combos by Sharpe:")
    print(top[["crash_thresh", "spike_thresh", "sharpe", "annualized_return", "profit_factor"]])


    df = pd.read_csv("logs/threshold_sweep_results.csv")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 6))


    # Total Return heatmap
    pivot_return = df_sweep.pivot(index="crash_thresh", columns="spike_thresh", values="total_return")
    sns.heatmap(pivot_return, annot=True, fmt=".2%", cmap="viridis", ax=ax1,
            vmin=0, vmax=0.02)  # Adjust range based on expectations

    ax1.set_title("Total Return by Threshold Combo")
    ax1.set_xlabel("Spike Threshold")
    ax1.set_ylabel("Crash Threshold")

    # Profit Factor heatmap (replace inf for clarity)
    df_plot = df_sweep.copy()
    df_plot["profit_factor_plot"] = df_plot["profit_factor"].replace(np.inf, 10)
    pivot_pf = df_plot.pivot(index="crash_thresh", columns="spike_thresh", values="profit_factor_plot")
    sns.heatmap(pivot_pf, annot=True, fmt=".2f", cmap="magma", ax=ax2)
    ax2.set_title("Profit Factor by Threshold Combo")
    ax2.set_xlabel("Spike Threshold")
    ax2.set_ylabel("Crash Threshold")

    # Win Rate heatmap
    pivot_wr = df_sweep.pivot(index="crash_thresh", columns="spike_thresh", values="win_rate")
    sns.heatmap(pivot_wr, annot=True, fmt=".0%", cmap="coolwarm", ax=ax3,
            vmin=0, vmax=1)  # Ensures 0–100% scale across all heatmaps

    ax3.set_title("Win Rate by Threshold Combo")
    ax3.set_xlabel("Spike Threshold")
    ax3.set_ylabel("Crash Threshold")

    # Trade Count heatmap (or swap for Sharpe if preferred)
    pivot_trades = df_sweep.pivot(index="crash_thresh", columns="spike_thresh", values="trades")
    sns.heatmap(pivot_trades, annot=True, fmt=".0f", cmap="Greens", ax=ax4)
    ax4.set_title("Trade Count by Threshold Combo")
    ax4.set_xlabel("Spike Threshold")
    ax4.set_ylabel("Crash Threshold")


    plt.tight_layout()
    plt.show()




    print("\n🔝 Top 5 Threshold Combos by Total Return:")
    print(df_sweep.sort_values("total_return", ascending=False).head())

    print("\n📈 Trades per Combo:")
    print(df_sweep.pivot(index="crash_thresh", columns="spike_thresh", values="trades"))




