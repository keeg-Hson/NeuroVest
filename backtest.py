# backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_spy_daily_data

# ─── Trade rules ────────────────────────────────────────────────────────────────
ENTRY_SHIFT   = 1    # enter at next bar’s open
EXIT_SHIFT    = 2    # exit at the following bar’s close
POSITION_SIZE = 1.0  # 1x notional per trade

def run_backtest():
    # 1) load signals and history
    preds  = pd.read_csv("logs/daily_predictions.csv", parse_dates=["Timestamp"])
    spy_df = load_spy_daily_data()

    # Convert timestamps to midnight-of-that-day
    preds["Date"] = preds["Timestamp"].dt.floor("D")

    # Re-join on that Date index instead of the full datetime
    df = (
        preds
        .set_index("Date")
        .join(spy_df[["Open", "Close"]], how="inner")
        .sort_index()
    )

    # 2) join on timestamp -> quotes
    df = (
        preds
        .set_index("Timestamp")
        .join(spy_df[["Open", "Close"]], how="inner")
        .sort_index()
    )

    # 3) build trades
    trades_list = []
    for ts, row in df.iterrows():
        sig = row["Prediction"]
        if sig not in (1, 2):
            continue

        try:
            entry = df.iloc[df.index.get_loc(ts) + ENTRY_SHIFT]
            exit_ = df.iloc[df.index.get_loc(ts) + EXIT_SHIFT]
        except IndexError:
            break

        entry_price = entry["Open"]
        exit_price  = exit_["Close"]

        ret = (
            exit_price / entry_price - 1
            if sig == 2 else
            entry_price / exit_price - 1
        )

        trades_list.append({
            "signal_time": ts,
            "sig":          sig,
            "entry_time":   entry.name,
            "exit_time":    exit_.name,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "return_pct":   ret * POSITION_SIZE
        })

    # 3a) handle no‐trades case
    if not trades_list:
        print("⚠️  No trades generated in this backtest.")
        zero_metrics = {
            "trades":            0,
            "total_return":      0.0,
            "annualized_return": 0.0,
            "sharpe":            0.0
        }
        return pd.DataFrame(), zero_metrics

    # 4) convert to DataFrame & index
    trades = pd.DataFrame(trades_list).set_index("signal_time")

    # 5) performance metrics
    trades["equity_curve"] = (1 + trades["return_pct"]).cumprod()
    total_return      = trades["equity_curve"].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(trades)) - 1
    sharpe            = (
        trades["return_pct"].mean()
        / trades["return_pct"].std()
    ) * np.sqrt(252)

    metrics = {
        "trades":            len(trades),
        "total_return":      total_return,
        "annualized_return": annualized_return,
        "sharpe":            sharpe
    }

    return trades, metrics


if __name__ == "__main__":
    trades, m = run_backtest()

    # report
    print("\n📈 Backtest Report")
    print(f"  Trades taken:       {m['trades']}")
    print(f"  Total return:       {m['total_return']:.2%}")
    print(f"  Annualized return:  {m['annualized_return']:.2%}")
    print(f"  Sharpe ratio (252d):{m['sharpe']:.2f}\n")

    print("\nSample trades:")

    if trades.empty:
        print("  (no trades to show)")
    else:
        print(trades.head())

        # only plot if there's something to plot
        trades[["equity_curve"]].plot(title="Equity Curve", figsize=(8,4))
        plt.xlabel("Signal Time")
        plt.ylabel("Cumulative Return")
        plt.tight_layout()
        plt.show()
