# trade_executor.py

import pandas as pd
import os

# Configurable starting balance
START_BALANCE = 10000.0
TRADE_QUANTITY = 10  # number of shares per trade
LOG_PATH = "logs/trade_log.csv"

def get_close_price(row):
    return row.get("Close_Price", row.get("Close"))


def get_account_balance():
    """Return dummy account balance and position (defaults to initial if not found)."""
    if os.path.exists(LOG_PATH):
        try:
            log = pd.read_csv(LOG_PATH)
            if "Balance" in log.columns and "Position" in log.columns and not log.empty:
                balance = log.iloc[-1]["Balance"]
                position = log.iloc[-1]["Position"]
                return balance, position
        except Exception as e:
            print(f"⚠️ Failed to parse trade log: {e}")
    
    # Default fallback
    return START_BALANCE, 0


def place_trade(signal, price, confidence, date, balance, position, total_cost,
                avg_buy_price, spike_conf, crash_conf, prev_price=None,
                min_spike_conf=0.6, min_crash_conf=0.6, use_momentum=True):
    
    if signal == "BUY":
        if spike_conf < min_spike_conf:
            status = "Skipped BUY (low spike confidence)"
        elif position > 0:
            status = "Skipped BUY (already holding)"
        elif use_momentum and prev_price is not None and price <= prev_price:
            status = "Skipped BUY (no upward momentum)"
        else:
            cost = price * TRADE_QUANTITY
            if balance >= cost:
                position += TRADE_QUANTITY
                balance -= cost
                total_cost += cost
                avg_buy_price = total_cost / position if position > 0 else 0
                status = "Executed BUY"
            else:
                status = "Insufficient funds"

    elif signal == "SELL":
        if position >= TRADE_QUANTITY:
            if price > avg_buy_price:
                position -= TRADE_QUANTITY
                balance += price * TRADE_QUANTITY
                total_cost -= avg_buy_price * TRADE_QUANTITY
                avg_buy_price = total_cost / position if position > 0 else 0
                status = "Executed SELL"
            else:
                status = "Skipped SELL (would be a loss)"
        else:
            status = "Insufficient holdings"

    else:
        status = "No action"

    return {
        "Date": date,
        "Signal": signal,
        "Confidence": round(confidence, 3),
        "Price": round(price, 2),
        "Status": status,
        "Balance": round(balance, 2),
        "Position": position,
        "Avg_Buy_Price": round(avg_buy_price, 2)
    }, balance, position, total_cost, avg_buy_price



def simulate_trade_execution(signal_log_path="logs/daily_predictions.csv",
                             min_spike_conf=0.6,
                             min_crash_conf=0.6,
                             use_momentum=True):
    
    df = pd.read_csv(signal_log_path, parse_dates=["Date"], on_bad_lines='skip')

    
    balance = START_BALANCE
    position = 0
    total_cost = 0.0
    avg_buy_price = 0.0
    trade_logs = []

    for idx, row in df.iterrows():
        prev_price = get_close_price(df.iloc[idx - 1]) if idx > 0 else None


        trade, balance, position, total_cost, avg_buy_price = place_trade(
            signal="BUY" if row["Prediction"] == 2 else "SELL" if row["Prediction"] == 1 else "HOLD",
            confidence=max(row["Crash_Conf"], row["Spike_Conf"]),
            price=get_close_price(row),

            date=row["Date"],
            balance=balance,
            position=position,
            total_cost=total_cost,
            avg_buy_price=avg_buy_price,
            spike_conf=row.get("Spike_Conf", 0),
            crash_conf=row.get("Crash_Conf", 0),
            prev_price=prev_price,
            min_spike_conf=min_spike_conf,
            min_crash_conf=min_crash_conf,
            use_momentum=use_momentum
        )
        trade_logs.append(trade)

    df_out = pd.DataFrame(trade_logs)
    df_out.to_csv(LOG_PATH, index=False)

    print(f"✅ Trade simulation complete. Final balance: ${round(balance, 2)}, Position: {position} shares")
    print(f"📄 Trades logged to: {LOG_PATH}")





if __name__ == "__main__":
    simulate_trade_execution()
