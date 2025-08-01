# top_config_runner.py

import pandas as pd
from trade_executor import simulate_trade_execution

LEADERBOARD_PATH = "logs/threshold_leaderboard.csv"

def run_best_config():
    # Load leaderboard
    df = pd.read_csv(LEADERBOARD_PATH)
    if df.empty:
        print("❌ Leaderboard is empty or missing.")
        return

    # Get top config by balance
    top_row = df.sort_values(by="Final_Balance", ascending=False).iloc[0]
    best_spike = top_row["Spike_Conf"]
    best_crash = top_row["Crash_Conf"]
    use_momentum = bool(top_row["Momentum"])

    print("🔧 Running best configuration from leaderboard:")
    print(f"   📈 Spike Conf ≥ {best_spike}")
    print(f"   📉 Crash Conf ≥ {best_crash}")
    print(f"   🚀 Momentum Filter: {use_momentum}")

    # Run simulation with best config
    simulate_trade_execution(
        signal_log_path="logs/daily_predictions.csv",
        min_spike_conf=best_spike,
        min_crash_conf=best_crash,
        use_momentum=use_momentum
    )

if __name__ == "__main__":
    run_best_config()
