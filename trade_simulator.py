#trade_simulator.py
import pandas as pd
from utils import summarize_trades
import os

# === CONFIG ===
INITIAL_BALANCE = 10000.0
HOLD_DAYS = 3
TRADE_LOG_PATH = "logs/trade_log.csv"

# 🔒 New Constraints
MAX_TRADE_AMOUNT = 2000           # Hard cap per trade
MAX_SHARES = 1000                 # Cap number of shares bought
REINVEST_FRACTION = 0.2           # Use only 20% of available cash per trade
REQUIRE_PROFIT = True             # Only sell if profitable

def simulate_trades(df, initial_balance=INITIAL_BALANCE, hold_days=HOLD_DAYS):
    balance = initial_balance
    position = 0
    entry_price = None
    entry_date = None
    trades = []

    for i, row in df.iterrows():
        date = pd.to_datetime(row['Timestamp'])
        prediction = row['Prediction']
        price = row.get("Close_Price", row.get("Close"))

        if price is None or pd.isna(price):
            continue

        # BUY logic
        if prediction == 2 and position == 0:
            invest_amount = min(balance * REINVEST_FRACTION, MAX_TRADE_AMOUNT)
            shares_to_buy = int(min(invest_amount // price, MAX_SHARES))
            if shares_to_buy > 0:
                cost = price * shares_to_buy
                balance -= cost
                position = shares_to_buy
                entry_price = price
                entry_date = date

                trades.append({
                    "Date": date,
                    "Action": "BUY",
                    "Price": round(price, 2),
                    "Shares": shares_to_buy,
                    "Balance": round(balance, 2),
                    "Position": position
                })

        # SELL logic
        elif position > 0 and entry_price is not None and (
            prediction == 1 or (date - entry_date).days >= hold_days
        ):
            roi = (price - entry_price) / entry_price
            if not REQUIRE_PROFIT or roi > 0:
                proceeds = price * position
                profit = proceeds - (entry_price * position)
                balance += proceeds

                trades.append({
                    "Date": date,
                    "Action": "SELL",
                    "Price": round(price, 2),
                    "Shares": position,
                    "Balance": round(balance, 2),
                    "Position": 0,
                    "Profit": round(profit, 2),
                    "ROI": round(roi, 4)
                })

                position = 0
                entry_price = None
                entry_date = None

    # Optional: liquidate final position
    if position > 0:
        final_price = df.iloc[-1].get("Close_Price", df.iloc[-1].get("Close"))
        if final_price is not None and (not REQUIRE_PROFIT or final_price > entry_price):
            proceeds = final_price * position
            profit = proceeds - (entry_price * position)
            balance += proceeds
            trades.append({
                "Date": df.iloc[-1]['Timestamp'],
                "Action": "FINAL SELL",
                "Price": round(final_price, 2),
                "Shares": position,
                "Balance": round(balance, 2),
                "Position": 0,
                "Profit": round(profit, 2),
                "ROI": round((final_price - entry_price) / entry_price, 4)
            })

    return balance, trades

def save_trade_log(trades, path=TRADE_LOG_PATH):
    if trades:
        pd.DataFrame(trades).to_csv(path, index=False)
        print(f"✅ Trade log saved to {path}")
    else:
        print("⚠️ No trades executed.")

# === MAIN RUNNER ===
if __name__ == "__main__":
    predictions = pd.read_csv("logs/daily_predictions.csv")
    final_balance, trade_log = simulate_trades(predictions)

    save_trade_log(trade_log)

    summary = summarize_trades(
        trade_log,
        INITIAL_BALANCE,
        save_plot_path="logs/equity_drawdown_plot.png"
    )
    print(f"\n💰 Final Balance: ${summary['final_balance']:.2f}")
    print(f"📈 Total trades: {summary['total_trades']}")
    print(f"✅ Win rate: {summary['win_rate'] * 100:.2f}%")
