# backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_spy_daily_data, log_rolling_accuracy
from datetime import timedelta, datetime
import subprocess

#auto update SPY data before anything else
subprocess.run(["python3", "update_spy_data.py"])


#sets a reference portfolio size to simulate dollar returns from % returns.





# ─── Trade rules ────────────────────────────────────────────────────────────────
ENTRY_SHIFT   = 1    # enter at next bar’s open
EXIT_SHIFT    = 2    # exit at the following bar’s close
POSITION_SIZE = 1.0  # 1x notional per trade
CAPITAL_BASE  = 100_000  #initial capital base for dollar-return tracking

def run_backtest(crash_thresh=2.0, spike_thresh=2.0, confidence_thresh=0.5, simulate_mode=False): #comment out threshholds if running simulated trades
    # 1) load signals and history
    preds  = pd.read_csv("logs/daily_predictions.csv", parse_dates=["Timestamp"])
    spy_df = load_spy_daily_data()

    #apply confidence filtering to reduce noise
    
    preds = preds[
        (preds["Crash_Conf"] >= confidence_thresh) | 
        (preds["Spike_Conf"] >= confidence_thresh)
    ]



    simulate_mode = True  # <- Set to False for real runs
        
    if crash_thresh is not None or spike_thresh is not None:
        simulate_mode = False


    # ensure 'Date' is created early for all branches
    preds["Date"] = preds["Timestamp"].dt.floor("D")
    spy_df.index = spy_df.index.floor("D")

    if crash_thresh is not None or spike_thresh is not None:
        print(f"📊 Applying thresholds — Crash ≥ {crash_thresh}, Spike ≥ {spike_thresh}")
        
        preds["Prediction"] = 0  # Reset all predictions

        # Crash condition (if threshold provided)
        if crash_thresh is not None:
            preds.loc[preds["Crash_Conf"] >= crash_thresh, "Prediction"] = 1  # 1 = crash

        # Spike condition (if threshold provided)
        if spike_thresh is not None:
            preds.loc[preds["Spike_Conf"] >= spike_thresh, "Prediction"] = 2  # 2 = spike

    # Inject fake predictions if simulating
    if simulate_mode:
        print("🧪 Simulate mode ON — injecting fake spike predictions.")
        valid_idx = preds[preds["Date"].isin(spy_df.index)].index

        # Choose ~10% of rows evenly spaced
        inject_points = np.linspace(0, len(valid_idx) - 3, 10, dtype=int)

        for idx in inject_points:
            safe_idx = valid_idx[idx]
            preds.loc[safe_idx, "Prediction"] = 2  # simulate spike

        # Floor timestamps early
        preds["Date"] = preds["Timestamp"].dt.floor("D")
        spy_df.index = spy_df.index.floor("D")

    # Restrict to valid SPY dates
    valid_dates = preds[preds["Date"].isin(spy_df.index)].copy()
    if len(valid_dates) > 10:
        inject_idx = valid_dates.index[10]
        preds.loc[inject_idx, "Prediction"] = 2
        print("\n✅ Injected spike at index:", inject_idx)
        print(preds.loc[inject_idx])
    else:
        print("⚠️  Not enough valid prediction rows to inject spike at index 10.")
        inject_idx=None


    

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
        #final_balance = trades['cash_curve'].iloc[-1]  # final simulated capital
        print("⚠️  No trades generated in this backtest.")
        zero_metrics = {
            "trades":            0,
            "total_return":      0.0,
            "annualized_return": 0.0,
            "sharpe":            0.0,
            "avg_return":        0.0,
            "median_return":     0.0,
            "win_rate":          0.0,
            "avg_long":          0.0,
            "avg_short":         0.0,
            "max_drawdown":      0.0,
            "profit_factor":     0.0
        }
        return pd.DataFrame(), zero_metrics, simulate_mode
    

    # 4) convert to DataFrame & index
    trades = pd.DataFrame(trades_list).set_index("signal_time")

    #compute dollar return per trade based on notional capital
    trades["dollar_return"] = trades["return_pct"] * CAPITAL_BASE

    #track cumulative cash equity over time
    trades["cash_curve"] = trades["dollar_return"].cumsum() + CAPITAL_BASE


    # 5) performance metrics
    trades["equity_curve"] = (1 + trades["return_pct"]).cumprod()
    total_return      = trades["equity_curve"].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(trades)) - 1


    #Prevent sharpe divide-by-zero
    returns = trades["return_pct"]
    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)


    #warning if trades are too few to be meaningful
    if len(trades) < 5:
        print("⚠️  Very few trades — annualized return and Sharpe ratio may be misleading.")


    # 5a) Extended per-trade stats
    avg_return = trades["return_pct"].mean()
    median_return = trades["return_pct"].median()
    win_rate = (trades["return_pct"] > 0).mean()

    long_returns = trades[trades["sig"] == 2]["return_pct"]
    short_returns = trades[trades["sig"] == 1]["return_pct"]

    avg_long = long_returns.mean() if not long_returns.empty else 0.0
    avg_short = short_returns.mean() if not short_returns.empty else 0.0

    print("\n📊 Extended Trade Stats:")
    print(f"  Avg return/trade:     {avg_return:.2%}")
    print(f"  Median return/trade:  {median_return:.2%}")
    print(f"  Win rate:             {win_rate:.2%}")
    print(f"  Avg long return:      {avg_long:.2%}")
    print(f"  Avg short return:     {avg_short:.2%}")

    # 5b) Drawdown metrics — track equity high water mark and percent drops
    trades["peak"] = trades["equity_curve"].cummax()  # running peak
    trades["drawdown"] = trades["equity_curve"] / trades["peak"] - 1  # % drop from peak
    max_drawdown = trades["drawdown"].min()  # most negative drawdown

    # Optional: print drawdown summary
    print(f"\n📉 Max drawdown:         {max_drawdown:.2%}")
    




    #sharpe            = (
    #    trades["return_pct"].mean()
    #    / trades["return_pct"].std()
    #) * np.sqrt(252)

    # 6) log and save
    log_rolling_accuracy(datetime.now(), acc_7d=1.0, acc_30d=1.0)
    cols = [
        "sig", "entry_time", "exit_time",
        "entry_price", "exit_price", "return_pct", "dollar_return",
        "equity_curve", "peak", "drawdown"
    ]
    trades[cols].to_csv("logs/trade_log.csv", float_format="%.5f")

    # calculate profit factor
    gross_win = trades[trades["dollar_return"] > 0]["dollar_return"].sum()
    gross_loss = -trades[trades["dollar_return"] < 0]["dollar_return"].sum()
    profit_factor = gross_win / gross_loss if gross_loss != 0 else np.inf


    # collect and return metrics
    metrics = {
        "trades":            len(trades),
        "total_return":      total_return,
        "annualized_return": annualized_return,
        "sharpe":            sharpe,
        "avg_return":        avg_return,
        "median_return":     median_return,
        "win_rate":          win_rate,
        "avg_long":          avg_long,
        "avg_short":         avg_short,
        "max_drawdown":      max_drawdown,
        "profit_factor":     profit_factor,

    }

    metrics["max_drawdown"] = max_drawdown



    #metrics = {
    #    "trades":            len(trades),
    #    "total_return":      total_return,
    #    "annualized_return": annualized_return,
    #    "sharpe":            sharpe
    #}

    return trades, metrics, simulate_mode


if __name__ == "__main__":
    trades, m, simulate_mode = run_backtest()

    print("\n📈 Backtest Report")
    print(f"  Trades taken:       {m['trades']}")
    print(f"  Total return:       {m['total_return']:.2%}")
    print(f"  Annualized return:  {m['annualized_return']:.2%}")
    print(f"  Sharpe ratio (252d):{m['sharpe']:.2f}\n")
    print(f"  Max drawdown:        {m['max_drawdown']:.2%}")


    if not trades.empty:
        # print final balance + profit
        final_balance = trades["cash_curve"].iloc[-1]
        print(f"  Final capital:      ${final_balance:,.2f}")
        print(f"  Net profit:         ${final_balance - CAPITAL_BASE:,.2f}")


    print("\nSample trades:")
    if trades.empty:
        print("  (no trades to show)")
    else:
        print(trades.head())

        # Rename equity column for consistency
        trades = trades.rename(columns={"equity_curve": "Equity"})
        trades["Drawdown %"] = trades["drawdown"] * 100  # in percent terms

        # Ensure expected plot columns are present
        if "equity_curve" in trades.columns:
            trades["Equity"] = trades["equity_curve"]  # manually copy
        if "drawdown" in trades.columns:
            trades["Drawdown %"] = trades["drawdown"] * 100  # convert to percent

        # Combine both plots in one window using subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Equity Curve
        trades["Equity"].plot(ax=ax1, title="Equity Curve")
        ax1.set_ylabel("Cumulative Return")
        ax1.grid(True)

        # Drawdown Plot
        trades["Drawdown %"].plot(ax=ax2, title="Drawdown (%)", color='red', linestyle='--')
        ax2.set_ylabel("Drawdown")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.grid(True)

        plt.xlabel("Signal Time")
        plt.tight_layout()
        plt.show()

        # Save the equity + drawdown plot to file
        fig.savefig("logs/equity_drawdown_plot.png", dpi=300)
        print("📸 Saved equity and drawdown chart to logs/equity_drawdown_plot.png")



    if simulate_mode:  #callout
        print("\n⚠️ NOTE: This was a simulated run with injected predictions.")
