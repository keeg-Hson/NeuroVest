# sweep_runner.py
import itertools
import pandas as pd
from predict import run_predictions
from backtest import run_backtest
from utils import log_prediction_to_file #, load_SPY_data as load_data
import os



# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Parameter ranges (wider + lower)
confidence_range = [0.55, 0.60, 0.65, 0.70]
crash_range = [0.20, 0.30, 0.40]
spike_range = [0.20, 0.30, 0.40]

# Store results
results = []

# Sweep through all combinations
for conf, crash, spike in itertools.product(confidence_range, crash_range, spike_range):
    print(f"\n▶️ Running backtest: Confidence={conf:.2f}, Crash={crash:.2f}, Spike={spike:.2f}")
    
    # Step 1: Run prediction
    prediction_df = run_predictions(confidence_threshold=conf)
    if prediction_df is None or prediction_df.empty:
        print("⚠️ No prediction data — skipping this combo.")
        continue

    #log predictions
    from utils import save_predictions_dataframe
    save_predictions_dataframe(prediction_df, path="logs/daily_predictions.csv")


    # Step 2: Run backtest
    _, summary, _ = run_backtest(
    crash_thresh=crash,
    spike_thresh=spike,
    confidence_thresh=conf
    )

    if summary is None or summary.get("total_trades", 0) == 0:
        print("⚠️ No trades executed — skipping result.")
        continue

    # Step 3: Log summary
    results.append({
        "Confidence": conf,
        "Crash_Threshold": crash,
        "Spike_Threshold": spike,
        "Final_Balance": summary["final_balance"],
        "Total_Trades": summary["total_trades"],
        "Win_Rate": summary["win_rate"],
        "Max_Drawdown": summary["max_drawdown"],
        "Avg_Return_Per_Trade": summary.get("avg_return_pct", 0)
    })

# Save leaderboard
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by="Final_Balance", ascending=False)
    df.to_csv("logs/sweep_results.csv", index=False)
    print("\n🏁 Sweep complete — results saved to logs/sweep_results.csv")
    print(df.head(5))
else:
    print("\n🚫 Sweep finished — but no results were recorded (no trades met thresholds).")
