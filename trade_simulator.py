#trade_simulator.py
# This file is used to simulate trades based on the backtest results

# trade_simulator.py
# Simulates trades based on daily_predictions.csv

import pandas as pd
from utils import summarize_trades
#from simulate_trades import simulate_trades  # if simulate_trades is in a separate file, adjust import accordingly

def simulate_trades(predictions_df, initial_balance=10000, hold_days=3):
    """
    Simulates trades based on spike/crash predictions.
    - Buy on spike
    - Sell on crash or after hold_days

    Returns:
        final_balance (float), trades (list of dicts)
    """
    balance = initial_balance
    in_position = False
    entry_price = 0
    entry_date = None
    trades = []

    for idx, row in predictions_df.iterrows():
        date = pd.to_datetime(row['Timestamp'])
        prediction = row['Prediction']
        price = row['Close_Price']

        # Buy on spike
        if prediction == 2 and not in_position:
            in_position = True
            entry_price = price
            entry_date = date

        # Sell on crash or hold limit
        if in_position:
            if (date - entry_date).days >= hold_days or prediction == 1:
                roi = (price - entry_price) / entry_price
                balance *= (1 + roi)
                trades.append({
                    'Entry_Timestamp': entry_date,
                    'Exit_Timestamp': date,
                    'Entry_Price': entry_price,
                    'Exit_Price': price,
                    'ROI': roi
                })
                in_position = False

    return balance, trades


def save_trade_log(trades, output_path="logs/trade_log.csv"):
    if trades:
        df = pd.DataFrame(trades)
        df.to_csv(output_path, index=False)
        print(f"✅ Trade log saved to {output_path}")
    else:
        print("⚠️ No trades executed.")

# === Main runner ===
if __name__ == "__main__":
    initial_balance = 10000
    predictions = pd.read_csv("logs/daily_predictions.csv")
    
    # Run trade simulation
    final_balance, trade_list = simulate_trades(predictions, initial_balance=initial_balance)

    # Save trade log
    save_trade_log(trade_list)

    # Summarize and plot
    summary = summarize_trades(trade_list, initial_balance, save_plot_path="logs/equity_drawdown_plot.png")

    # Print summary
    print(f"\n💰 Final Balance: ${summary['final_balance']:.2f}")
    print(f"📈 Total trades: {summary['total_trades']}")
    print(f"✅ Win rate: {summary['win_rate'] * 100:.2f}%")
    print("📊 Equity curve saved to logs/equity_drawdown_plot.png")
