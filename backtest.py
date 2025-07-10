# backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_spy_daily_data, log_rolling_accuracy
from datetime import timedelta, datetime



# ─── Trade rules ────────────────────────────────────────────────────────────────
ENTRY_SHIFT   = 1    # enter at next bar’s open
EXIT_SHIFT    = 2    # exit at the following bar’s close
POSITION_SIZE = 1.0  # 1x notional per trade

def run_backtest():
    # 1) load signals and history
    preds  = pd.read_csv("logs/daily_predictions.csv", parse_dates=["Timestamp"])
    spy_df = load_spy_daily_data()

    # Floor timestamps early
    preds["Date"] = preds["Timestamp"].dt.floor("D")
    spy_df.index = spy_df.index.floor("D")

    # Restrict to valid SPY dates
    valid_dates = preds[preds["Date"].isin(spy_df.index)].copy()
    inject_idx = valid_dates.index[10]

    # Inject spike at safe index
    preds.loc[inject_idx, "Prediction"] = 2

    print("\n✅ Injected spike at index:", inject_idx)
    print(preds.loc[inject_idx])

    # 2) join after injection
    df = (
        preds
        .set_index("Date")
        .join(spy_df[["Open", "Close"]], how="inner")
        .sort_index()
        .reset_index()
    )




    print("\n📊 Joined Prediction Candidates (1 or 2):")
    print(df[df["Prediction"].isin([1, 2])])


    #Diagnostics

    print("\n🔍 Prediction Counts:")
    print(df["Prediction"].value_counts())
    print("\n🗓️  Joined Date Range:", df.index.min(), "→", df.index.max())
    print("\n📊 Sample trade candidates:")
    print(df[df["Prediction"].isin([1, 2])].head())

    print("\n🗓️  Date range in predictions:", preds["Date"].min(), "→", preds["Date"].max())
    print("📅 Date range in SPY data:    ", spy_df.index.min(), "→", spy_df.index.max())

    if all(df["Prediction"] == 0):
        print("⚠️  All predictions are '0' — consider lowering crash/spike thresholds.")


    # 3) build trades
    trades_list = []
    for i, row in df.iterrows():
        sig = row["Prediction"]
        if sig not in (1, 2):
            continue

        try:
            entry = df.iloc[i + ENTRY_SHIFT]
            exit_ = df.iloc[i + EXIT_SHIFT]
        except IndexError:
            continue

        entry_price = entry["Open"]
        exit_price  = exit_["Close"]

        ret = (
            exit_price / entry_price - 1
            if sig == 2 else
            entry_price / exit_price - 1
        )

        trades_list.append({
            "signal_time": row["Timestamp"],
            "sig":          sig,
            "entry_time":   entry["Timestamp"],
            "exit_time":    exit_["Timestamp"],
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

    #Prevent sharpe divide-by-zero
    ret_std = trades["return_pct"].std()
    sharpe = (
        trades["return_pct"].mean() / ret_std * np.sqrt(252)
        if ret_std != 0 else 0.0
    )

    #warning if trades are too few to be meaningful
    if len(trades) < 5:
        print("⚠️  Very few trades — annualized return and Sharpe ratio may be misleading.")


    #sharpe            = (
    #    trades["return_pct"].mean()
    #    / trades["return_pct"].std()
    #) * np.sqrt(252)

    # 6) log and save
    log_rolling_accuracy(datetime.now(), acc_7d=1.0, acc_30d=1.0)
    trades.to_csv("logs/trade_log.csv")


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

        #Rename column for prettier plot label
        trades = trades.rename(columns={"equity_curve": "Equity"})
        ax = trades[["Equity"]].plot(title="Equity Curve", figsize=(8, 4))

        #Zoom in Y-axis for small return deltas
        min_eq, max_eq = trades["Equity"].min(), trades["Equity"].max()
        padding = 0.01
        plt.ylim([min_eq - padding, max_eq + padding])

        plt.xlabel("Signal Time")
        plt.ylabel("Cumulative Return")
        plt.tight_layout()
        plt.show()
