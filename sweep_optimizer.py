# sweep_optimizer.py

from trade_executor import simulate_trade_execution
import numpy as np
import pandas as pd

results = []

for spike in np.arange(0.5, 0.9, 0.05):
    for crash in np.arange(0.5, 0.9, 0.05):
        for momentum in [True, False]:
            simulate_trade_execution(
                signal_log_path="logs/daily_predictions.csv",
                min_spike_conf=spike,
                min_crash_conf=crash,
                use_momentum=momentum
            )

            # Read the latest result
            log = pd.read_csv("logs/trade_log.csv")
            final = log.iloc[-1]
            results.append({
                "Spike_Conf": spike,
                "Crash_Conf": crash,
                "Momentum": momentum,
                "Final_Balance": final["Balance"],
                "Final_Position": final["Position"]
            })

# Save leaderboard
df = pd.DataFrame(results)
df.sort_values(by="Final_Balance", ascending=False).to_csv("logs/threshold_leaderboard.csv", index=False)

print("🏁 Sweep complete! Top results:")
print(df.sort_values(by="Final_Balance", ascending=False).head(10))
