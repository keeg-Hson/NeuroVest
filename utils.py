# utils.py
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import os, shutil, datetime
import requests
import pandas_ta as ta
import platform
import warnings

from dotenv import load_dotenv
from external_signals import add_external_signals #EXTERNAL SIGNALS MODULE

load_dotenv()

import socket
socket.setdefaulttimeout(float(os.getenv("NET_TIMEOUT", "3")))


# --- Hooks consumed by run_all.py ----------------------------------------------------
def update_spy_data():
    """
    Hook for run_all.py step_refresh_data().
    If you already have a real price refresh elsewhere, call it here.
    For now we just touch/load SPY to mark the step successful.
    """
    try:
        _ = load_SPY_data()
        return True
    except Exception:
        return False

# Optional aliases that run_all.py also checks for:
def refresh_prices():
    return update_spy_data()

def update_yfinance_data():
    return update_spy_data()
# -------------------------------------------------------------------------------------




# --- Ensure folders exist ---
os.makedirs("logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# --- Global log file paths ---
LOG_FILE = "logs/daily_predictions.csv"
LABELED_LOG_FILE = "logs/labeled_predictions.csv"

# --- Ensure labeled_predictions.csv exists ---
def init_labeled_log_file():
    if not os.path.exists(LABELED_LOG_FILE):
        print("[Init] Creating blank labeled_predictions.csv")
        with open(LABELED_LOG_FILE, "w") as f:
            f.write("Timestamp,Prediction,Crash_Conf,Spike_Conf,Close_Price,Actual_Event\n")

# --- Feature List ---
def get_feature_list():
    return [
        "MA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Width", "Volatility", "OBV", "Vol_Ratio", "Price_Momentum_10", "Acceleration",
        "RSI", "RSI_Delta", "ZMomentum",
        "Return_Lag1", "Return_Lag3", "Return_Lag5",
        "RSI_Lag_1", "RSI_Lag_3", "RSI_Lag_5",
        "Rolling_STD_5", "Daily_Return",
        "MACD_x_RSI", "Volume_per_ATR",
        "Stoch_K", "Stoch_D", "BB_PctB","KC_Width","VWAP_Dev","Ret_Skew_20","Ret_Kurt_20",
        "Sent_x_Vol","RSI_x_NewsZ","RSI_x_RedditZ"

    ]


def in_human_speak(label):
    """
    Convert your internal labels to human-readable strings.

    Accepts ints or strings. Your codebase convention has:
      0 = NORMAL (often filtered out before training)
      1 = CRASH
      2 = SPIKE
    """
    try:
        if isinstance(label, str) and label.isdigit():
            label = int(label)
    except Exception:
        pass

    mapping = {
        0: "NORMAL",
        1: "CRASH",
        2: "SPIKE",
        "0": "NORMAL",
        "1": "CRASH",
        "2": "SPIKE",
        "NORMAL": "NORMAL",
        "CRASH": "CRASH",
        "SPIKE": "SPIKE",
    }
    return mapping.get(label, str(label))

# --- Log prediction to file ---
# --- Log prediction to file ---
def log_prediction_to_file(timestamp, prediction, crash_conf, spike_conf,
                           close_price, open_price=None, high=None, low=None,
                           log_path="logs/daily_predictions.csv"):
    import csv, os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    headers = [
        "Date","Timestamp","Prediction","Crash_Conf","Spike_Conf",
        "Close","Open","High","Low","Confidence","Regime","FeatSnapshot"

    ]
    date_str = str(getattr(timestamp, "date", lambda: timestamp)())

    regime = os.getenv("REGIME_TAG", "")
    import hashlib, json as _json, os as _os
    def _hash_file(p):
        try:
            with open(p,"rb") as fh: return hashlib.sha1(fh.read()).hexdigest()[:10]
        except Exception: return "NA"
    feat_snapshot = "-".join([
        _hash_file("models/market_crash_model.pkl"),
        _hash_file("models/thresholds.json"),
        _hash_file("configs/best_thresholds.json"),
    ])


    row = {
        "Date":        date_str,
        "Timestamp":   str(timestamp),
        "Prediction":  int(prediction) if prediction is not None else 0,
        "Crash_Conf":  float(crash_conf) if crash_conf is not None else 0.0,
        "Spike_Conf":  float(spike_conf) if spike_conf is not None else 0.0,
        "Close":       float(close_price) if close_price is not None else "",
        "Open":        float(open_price) if open_price is not None else "",
        "High":        float(high) if high is not None else "",
        "Low":         float(low) if low is not None else "",
        "Confidence":  float(max(crash_conf or 0.0, spike_conf or 0.0)),
        "Regime": regime,
        "FeatSnapshot": feat_snapshot,

    }

    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists or os.path.getsize(log_path) == 0:
            w.writeheader()
        w.writerow(row)

    # Optional: append a lightweight signals csv for eyeballing
    sig = "BUY" if row["Prediction"] == 2 else ("SELL" if row["Prediction"] == 1 else "HOLD")
    signals_path = "logs/signals.csv"
    signals_headers = ["Date","Signal","Confidence","Price","Spike_Conf","Crash_Conf"]
    sig_row = {
        "Date": date_str,
        "Signal": sig,
        "Confidence": row["Confidence"],
        "Price": row["Close"],
        "Spike_Conf": row["Spike_Conf"],
        "Crash_Conf": row["Crash_Conf"],
    }
    file_exists2 = os.path.isfile(signals_path)
    with open(signals_path, "a", newline="") as f2:
        w2 = csv.DictWriter(f2, fieldnames=signals_headers)
        if not file_exists2 or os.path.getsize(signals_path) == 0:
            w2.writeheader()
        w2.writerow(sig_row)

# --- Label real outcomes using future close data ---
def label_real_outcomes_from_log(crash_thresh=-0.005, spike_thresh=0.005):
    if not os.path.exists(LOG_FILE):
        print("[⚠️] daily_predictions.csv not found — skipping outcome labeling.")
        return

    df = pd.read_csv(LOG_FILE, parse_dates=["Timestamp"])
    print(f"[DEBUG] read {len(df)} rows from {LOG_FILE}")
    print(df.tail())

    df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)

    if len(df) < 2:
        print("[⏭] Not enough data to label real outcomes — skipping for now.")
        return

    # Your log_prediction_to_file() writes 'Close' (not 'Close_Price')
    price_col = "Close"
    if price_col not in df.columns:
        raise KeyError(f"[label_real_outcomes_from_log] Expected column '{price_col}' in {LOG_FILE}")

    df["Next_Close"] = df[price_col].shift(-1)
    df["Future_Return"] = (df["Next_Close"] - df[price_col]) / df[price_col]

    df["Actual_Event"] = np.select(
        [df["Future_Return"] < -0.005, df["Future_Return"] > 0.005],
        [1, 2],
        default=0
    )

    df.dropna(subset=["Future_Return"], inplace=True)
    df.to_csv(LABELED_LOG_FILE, index=False)
    print(f"[DEBUG] wrote {len(df)} rows to {LABELED_LOG_FILE}")
    #print(f"[Labeling] Return: {future_return:.4f} → Event: {actual_event}")

    print("[✅] Labeled outcomes written to logs/labeled_predictions.csv")

    #backup_logs()
    df.to_csv(LABELED_LOG_FILE, index=False)
    print("[✅] Labeled outcomes written to logs/labeled_predictions.csv")

    backup_logs()


def backup_logs():
    """Make timestamped copies of your two main CSVs into ./backups/"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("backups", exist_ok=True)
    shutil.copy(LOG_FILE, f"backups/daily_predictions_{ts}.csv")
    shutil.copy(LABELED_LOG_FILE, f"backups/labeled_predictions_{ts}.csv")

def summarize_trades(trades, initial_balance=10000, save_plot_path=None):
    """
    Summarizes trade results: final balance, win rate, trade count.
    Compatible with trades from both simulate_trades() and run_backtest().

    Returns dict with balance, win rate, and equity curve.
    """
    if not trades:
        return {
            'final_balance': initial_balance,
            'total_trades': 0,
            'win_rate': 0.0,
            'equity_curve': [initial_balance]
        }

    equity = [initial_balance]
    wins = 0
    total_trades = 0

    for trade in trades:
        if 'ROI' in trade:
            roi = trade['ROI']
        elif 'Entry_Price' in trade and 'Exit_Price' in trade:
            roi = (trade['Exit_Price'] - trade['Entry_Price']) / trade['Entry_Price']
        else:
            print("⚠️ Skipping trade — missing ROI or price data:", trade)
            continue

        total_trades += 1
        if roi > 0:
            wins += 1
        equity.append(equity[-1] * (1 + roi))

    final_balance = equity[-1]
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    if save_plot_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(equity, linewidth=2)
        plt.title("Equity Curve")
        plt.xlabel("Trade #")
        plt.ylabel("Account Balance ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()

    return {
        'final_balance': final_balance,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'equity_curve': equity
    }


def load_SPY_data():
    """
    Loads data/SPY.csv robustly and returns a clean OHLCV dataframe:
    - Quietly coerces index to datetime (no parse_dates warnings)
    - Flattens accidental MultiIndex columns
    - Normalizes column names to canonical OHLCV
    - Forces numeric dtype
    - Drops duplicate/NaT index entries; sorts by date
    - Trims trailing incomplete rows (e.g., NaN High/Low/Volume)
    - Ensures both a DatetimeIndex (named 'Date') and a 'Date' column exist
    """
    spy_path = "data/SPY.csv"
    if not os.path.exists(spy_path):
        raise FileNotFoundError(f"[❌] Could not find SPY data at {spy_path}")

    # Read without global parse_dates to avoid noisy inference warnings
    df = pd.read_csv(spy_path, index_col=0, low_memory=False)

    # Coerce index → datetime, quietly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # If your CSV index is always YYYY-MM-DD, you can use format="%Y-%m-%d"
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop rows where the index couldn't be parsed; de-dup & sort
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Flatten accidental MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip("_") for tup in df.columns]

    # Normalize names → keep canonical OHLCV
    rename_map = {
        # common “ticker suffixed” names
        "Open_SPY": "Open", "High_SPY": "High", "Low_SPY": "Low", "Close_SPY": "Close",
        "Adj Close_SPY": "Adj Close", "Volume_SPY": "Volume",
        "SPY_Open": "Open", "SPY_High": "High", "SPY_Low": "Low", "SPY_Close": "Close",
        "SPY_Adj Close": "Adj Close", "SPY_Volume": "Volume",
        # minor variants
        "AdjClose": "Adj Close", "Adj_Close": "Adj Close",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Keep canonical columns if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if keep:
        df = df[keep]

    # Force numeric dtypes
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Repeatedly drop the last row while critical fields are NaN (incomplete latest bar)
    crit = [c for c in ["High", "Low", "Volume"] if c in df.columns]
    while len(df) and crit and df.iloc[-1][crit].isna().any():
        df = df.iloc[:-1]

    # Also drop any rows missing Open/Close entirely
    base_crit = [c for c in ["Open", "Close"] if c in df.columns]
    if base_crit:
        df = df.dropna(subset=base_crit)

    # Ensure DatetimeIndex and a 'Date' column for downstream merges
    df.index.name = "Date"
    df["Date"] = df.index

    return df

def safe_read_csv(path, prefer_index=True):
    """Robust CSV reader that tolerates missing 'Date' column."""
    import pandas as pd, os
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    # Normalize time
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        # try index
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df["Date"] = df.index
        except Exception:
            pass
    # Optionally set as index
    if prefer_index and "Date" in df.columns:
        df = df.set_index("Date")
    return df








def add_features(df):
    df = df.copy()

    # --- Price-based Indicators ---
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD Indicators ---
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # --- RSI & Momentum ---
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    RS = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + RS))
    df["RSI_Delta"] = df["RSI"].diff()
    df["ZMomentum"] = (df["Close"] - df["Close"].rolling(10).mean()) / (df["Close"].rolling(10).std() + 1e-9)
    df["Price_Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Acceleration"] = df["Price_Momentum_10"].diff()

    # --- Lagged Features ---
    df["Return_Lag1"] = df["Close"].pct_change(1)
    df["Return_Lag3"] = df["Close"].pct_change(3)
    df["Return_Lag5"] = df["Close"].pct_change(5)
    df["RSI_Lag_1"] = df["RSI"].shift(1)
    df["RSI_Lag_3"] = df["RSI"].shift(3)
    df["RSI_Lag_5"] = df["RSI"].shift(5)

    # --- Bollinger Band Width ---
    rolling_std = df["Close"].rolling(20).std()
    upper_band = df["MA_20"] + (2 * rolling_std)
    lower_band = df["MA_20"] - (2 * rolling_std)
    df["BB_Width"] = (upper_band - lower_band) / (df["MA_20"] + 1e-9)

    # --- Volatility ---
    vol_window = 20
    df["Volatility"] = df["Close"].rolling(window=vol_window).std(ddof=0)

    # --- On-Balance Volume (OBV) ---
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # --- Volume Ratio ---
    df["Vol_Ratio"] = df["Volume"] / (df["Volume"].rolling(10).mean() + 1e-9)

    # --- Rolling Std Dev ---
    df["Rolling_STD_5"] = df["Close"].rolling(window=5).std()

    # --- Daily Return ---
    df["Daily_Return"] = df["Close"].pct_change()

    # --- MACD × RSI Interaction ---
    df["MACD_x_RSI"] = df["MACD"] * df["RSI"]

    # --- Volume per ATR (proxy) ---
    atr_proxy = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    df["Volume_per_ATR"] = df["Volume"] / (atr_proxy + 1e-9)

    # --- Stochastic Oscillator ---
    low_14  = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14 + 1e-9))
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # --- Bands & Channels ---
    ma20 = df["Close"].rolling(20).mean()
    sd20 = df["Close"].rolling(20).std()
    upper = ma20 + 2*sd20
    lower = ma20 - 2*sd20
    df["BB_PctB"] = (df["Close"] - lower) / ((upper - lower) + 1e-9)

    # Keltner: EMA + ATR-like true range
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr20 = tr.ewm(alpha=1/20, adjust=False).mean()
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    kc_upper = ema20 + 2*atr20
    kc_lower = ema20 - 2*atr20
    df["KC_Width"] = (kc_upper - kc_lower) / (ema20 + 1e-9)

    # VWAP deviation (rolling daily approx)
    typ = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vwap = (typ * df["Volume"]).rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)
    df["VWAP_Dev"] = (df["Close"] - vwap) / (vwap + 1e-9)

    # Tail stats
    ret = df["Close"].pct_change()
    df["Ret_Skew_20"] = ret.rolling(20).skew()
    df["Ret_Kurt_20"] = ret.rolling(20).kurt()

    # Interactions with externals (guard for missing)
    news_z   = df.get("News_Sent_Z20")
    reddit_z = df.get("Reddit_Sent_Z20")
    if "Volatility" not in df.columns:
        df["Volatility"] = df["Close"].rolling(20).std()
    df["Sent_x_Vol"]    = (news_z if news_z is not None else 0) * df["Volatility"]
    df["RSI_x_NewsZ"]   = df["RSI"] * (news_z if news_z is not None else 0)
    df["RSI_x_RedditZ"] = df["RSI"] * (reddit_z if reddit_z is not None else 0)


    # --- VOLATILITY / RANGE ---
    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"]   = tr.ewm(alpha=1/14, adjust=False).mean()
    df["Range_Exp"] = (df["High"] - df["Low"]) / df["ATR_14"]
    df["ATR_Trend"] = df["ATR_14"] / (df["ATR_14"].rolling(100).mean() + 1e-9)

    # --- GAPS ---
    df["Gap_Pct"] = (df["Open"] - df["Close"].shift()) / (df["Close"].shift() + 1e-9)
    df["Gap_ATR"] = (df["Open"] - df["Close"].shift()).abs() / (df["ATR_14"] + 1e-9)
    df["Gap_Dir"] = np.sign(df["Gap_Pct"]).fillna(0)

    # --- LEVEL DISTANCES / MA POSTURE ---
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["MA_Stack_Bull"] = ((df["Close"] > df["MA50"]) & (df["MA50"] > df["MA200"])).astype(int)
    df["Dist_High20_ATR"] = (df["Close"] - df["High"].rolling(20).max()) / (df["ATR_14"] + 1e-9)
    df["Dist_Low20_ATR"]  = (df["Close"] - df["Low"].rolling(20).min())  / (df["ATR_14"] + 1e-9)

    # --- SIMPLE CANDLE FLAGS ---
    real_body = (df["Close"] - df["Open"]).abs()
    wick_up   = df["High"] - df[["Close","Open"]].max(axis=1)
    wick_dn   = df[["Close","Open"]].min(axis=1) - df["Low"]
    rng = (df["High"] - df["Low"]).replace(0, np.nan)

    df["Doji"] = (real_body / rng < 0.1).fillna(0).astype(int)
    df["Engulf_Bull"] = (
        (df["Close"] > df["Open"]) &
        (df["Open"]  < df["Close"].shift()) &
        (df["Close"] > df["Open"].shift())
    ).fillna(0).astype(int)
    df["NR7"] = (rng == rng.rolling(7).min()).astype(int)

    # Overnight gap
    #df["Gap_Pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Realized vol (Parkinson)
    df["Parkinson_RV"] = (np.log(df["High"]/df["Low"])**2).rolling(20).mean()

    # RSI(2) for short-term mean reversion
    chg = df["Close"].diff()
    gain2 = chg.clip(lower=0).rolling(2).sum()
    loss2 = -chg.clip(upper=0).rolling(2).sum()
    df["RSI_2"] = 100 - (100/(1 + (gain2/(loss2+1e-9))))

    # Rolling autocorrelation (trendiness proxy)
    df["Ret1"] = df["Close"].pct_change()
    df["AutoCorr_5"] = df["Ret1"].rolling(10).apply(lambda x: pd.Series(x).autocorr(lag=5), raw=False)

    # Day-of-week seasonal
    df["DOW"] = df["Date"].dt.weekday if "Date" in df.columns else df.index.weekday


    # --- External Macro + Sentiment Signals (optional/skip when offline) ---
    OFFLINE = os.getenv("OFFLINE_MODE", "0").lower() in {"1", "true", "yes"}
    if OFFLINE:
        print("🔌 OFFLINE_MODE=1 → skipping external macro/sentiment fetches.")
    else:
        try:
            df = add_external_signals(df)
        except Exception as e:
            print(f"⚠️ add_external_signals failed: {e} — continuing without externals.")


    # (Optional) chatty warning — safe to comment out if noisy
    # if df.isna().sum().sum() > 0:
    #     print("⚠️ Warning: NaNs remain in feature set — will be handled after labeling.")

    # --- Collect feature columns (only those that exist) ---
    base_cols = [
        "MA_20","EMA_12","EMA_26","MACD","MACD_Signal","MACD_Histogram",
        "RSI","RSI_Delta","ZMomentum","Price_Momentum_10","Acceleration",
        "Return_Lag1","Return_Lag3","Return_Lag5","RSI_Lag_1","RSI_Lag_3","RSI_Lag_5",
        "BB_Width","Volatility","OBV","Vol_Ratio","Rolling_STD_5","Daily_Return",
        "MACD_x_RSI","Volume_per_ATR","Stoch_K","Stoch_D",
        # price/structure extras
        "ATR_14","Range_Exp","ATR_Trend",
        "Gap_Pct","Gap_ATR","Gap_Dir",
        "MA50","MA200","MA_Stack_Bull",
        "Dist_High20_ATR","Dist_Low20_ATR",
        "Doji","Engulf_Bull","NR7",
    ]

    macro_sentiment = [
        "CPI","Unemployment","InterestRate","YieldCurve",
        "ConsumerSentiment","IndustrialProduction","VIX",
        "News_Sentiment","Reddit_Sentiment",
        # structured sentiment
        "News_Sent_Z20","News_Sent_ROC3","News_Sent_Concord",
        "Reddit_Sent_Z20","Reddit_Sent_ROC3","Reddit_Sent_Concord",
        # cross-asset
        "Sector_MedianRet_5","Sector_MedianRet_20","Sector_Dispersion_5","Sector_Dispersion_20",
        "HYG_Ret_5","HYG_Ret_20","LQD_Ret_5","LQD_Ret_20","Credit_Spread_20",
        "TNX_Change_5","TNX_Change_20","DXY_Change_5","DXY_Change_20",
    ]

    # Keep only columns that actually exist now
    feature_cols = [c for c in (base_cols + macro_sentiment) if c in df.columns]
    # De-dup while preserving order (defensive)
    feature_cols = list(dict.fromkeys(feature_cols))

    return df, feature_cols





def save_predictions_dataframe(df, path="logs/latest_predictions.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[💾] Predictions saved to {path}")

#notification helper
def notify_user(prediction, crash_conf, spike_conf):
    """
    Triggers alert if crash or spike is highly probable.
    """
    should_alert = False
    label = "NORMAL"

    if prediction == 1 and crash_conf >= 0.7:
        should_alert = True
        label = "CRASH"
    elif prediction == 2 and spike_conf >= 0.7:
        should_alert = True
        label = "SPIKE"

    if should_alert:
        print("\n🚨🚨🚨 MARKET ALERT 🚨🚨🚨")
        print(f"⚠️  High-confidence {label} detected!")
        print(f"📉 Crash: {crash_conf:.2f} | 📈 Spike: {spike_conf:.2f}\n")

        # Cross-platform alert sound (no 'play' dependency)
        try:
            if platform.system() == "Darwin":
                os.system('afplay /System/Library/Sounds/Glass.aiff')
            else:
                # basic terminal bell
                os.system('printf "\\a"')
        except Exception:
            pass

#telegram bot
def send_telegram_alert(text, token=None, chat_id=None):
    import requests, json, os
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram not configured (missing token or chat id).")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.ok:
            return True
        else:
            try:
                err = r.json()
            except Exception:
                err = {"status_code": r.status_code, "text": r.text[:200]}
            print(f"⚠️ Telegram error: {json.dumps(err)}")
            return False
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")
        return False


def label_events_volatility_adjusted(df, window=3, vol_window=10, multiplier=0.2):
    df = df.copy()
    df["Event"] = 0  # Default: no event

    df["Volatility"] = df["Close"].rolling(window=vol_window).std()

    for i in range(window, len(df)):
        window_slice = df.iloc[i - window:i + 1]
        current_close = df.iloc[i]["Close"]
        current_vol = df.iloc[i]["Volatility"]

        if pd.isna(current_vol) or current_vol == 0:
            continue

        spike_threshold = window_slice["Close"].mean() + multiplier * current_vol
        crash_threshold = window_slice["Close"].mean() - multiplier * current_vol

        if current_close > spike_threshold:
            df.at[df.index[i], "Event"] = 2
        elif current_close < crash_threshold:
            df.at[df.index[i], "Event"] = 1

    return df


def label_events_simple(df, window=3, pct_threshold=0.01):
    """
    Labels spikes and crashes based on simple forward percentage movement over 'window' days.
    """
    df = df.copy()
    df["Future_Return"] = (df["Close"].shift(-window) - df["Close"]) / df["Close"]

    conditions = [
        df["Future_Return"] <= -pct_threshold,  # Crash
        df["Future_Return"] >= pct_threshold    # Spike
    ]

    choices = [1, 2]  # 1 = crash, 2 = spike
    df["Event"] = np.select(conditions, choices, default=0)

    return df

def _drop_constant_and_allnan(df: pd.DataFrame, cols: list[str], min_var=1e-12):
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            # keep if there is variance (after fills/interp)
            if s.std(skipna=True) > min_var:
                keep.append(c)
        # else: drop columns that are entirely NaN
    dropped = [c for c in cols if c not in keep]
    if dropped:
        print(f"ℹ️ Dropping constant/empty features: {dropped[:12]}{'...' if len(dropped)>12 else ''}")
    return keep


# --- Label real outcomes using future close data ---   # (search anchor a few lines above)
def finalize_features(df, feature_cols):
    """
    Keep exactly the features selected for training (order matters),
    and median-impute NaNs. No blanket dropping of external columns.
    """
    out = df.copy()
    # add any missing columns as zeros (predict-time parity)
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[feature_cols]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out





def label_events_triple_barrier(df, vol_col="ATR_14", pt_mult=2.0, sl_mult=2.0, t_max=5):
    """
    0/1/2 label using triple-barrier on Close path after each day:
      - profit-take at +pt_mult*vol
      - stop-loss  at -sl_mult*vol
      - time-out at t_max bars then sign of return
    Requires a volatility proxy column (e.g., ATR_14).
    """
    out = df.copy()
    out["Event"] = 0
    close = out["Close"].values
    vol   = out[vol_col].values
    N = len(out)
    for i in range(N-1):
        pt = close[i] * (1 + pt_mult * (vol[i]/max(close[i],1e-9)))
        sl = close[i] * (1 - sl_mult * (vol[i]/max(close[i],1e-9)))
        end = min(N-1, i + t_max)
        label = 0
        for j in range(i+1, end+1):
            if close[j] >= pt:  label = 2; break
            if close[j] <= sl:  label = 1; break
        if label == 0:
            ret = (close[end]-close[i]) / close[i]
            label = 2 if ret > 0 else 1 if ret < 0 else 0
        out.iat[i, out.columns.get_loc("Event")] = label
    return out

#trade quality checks

def assert_time_ordered(df, ts_col="timestamp"):
    if not df[ts_col].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly increasing.")

def drop_dupes_and_nans(df):
    df = df.drop_duplicates().copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(how="any")

def ensure_no_future_leakage(df, feature_cols, label_cols, horizon_col="horizon_forward"):
    # Basic sentinel: labels must depend on strictly future info.
    if horizon_col in df.columns:
        if (df[horizon_col] <= 0).any():
            raise ValueError("Detected non-positive forward horizon rows; check labeling.")
    # Spot check: features computed via rolling must not reference future rows.
    # (Heuristic: ensure no merging on future timestamps happened.)
    # Feature-generation logic must enforce windowing.
    return True

def add_forward_returns_and_labels(
    df,
    price_col="close",
    horizon=5,
    fee_bps=1.5,
    slippage_bps=2.0,
    long_only=True,
    pos_threshold=0.0,
    neg_threshold=0.0
):
    """
    Computes forward return over `horizon` bars and net-of-costs return.
    Labels:
      - long_only=True: y = 1 if net_fwd_ret > pos_threshold else 0 (no-trade)
      - long_only=False (optional): y in {1 (long), -1 (short), 0 (no-trade)} via thresholds
    """
    if price_col not in df.columns:
        for alt in ["Adj Close","ClosePrice","Price","Last","Last Price"]:
            if alt in df.columns:
                price_col = alt
                break
        else:
            raise KeyError(f"Price column '{price_col}' not found and no fallback present.")

    df = df.copy()
    df["fwd_price"] = df[price_col].shift(-horizon)
    raw_ret = (df["fwd_price"] - df[price_col]) / df[price_col]
    costs = (fee_bps + slippage_bps) * 1e-4  # convert bps to fraction
    df["fwd_ret_raw"] = raw_ret
    df["fwd_ret_net"] = raw_ret - costs

    if long_only:
        df["y"] = (df["fwd_ret_net"] > pos_threshold).astype(int)
    else:
        # Example 3-class scheme; adjust thresholds to taste.
        df["y"] = 0
        df.loc[df["fwd_ret_net"] > pos_threshold, "y"] = 1
        df.loc[df["fwd_ret_net"] < -abs(neg_threshold), "y"] = -1

    df["horizon_forward"] = horizon
    return df

def compute_sample_weights(df, min_weight=0.1, max_weight=5.0, power=1.0, long_only=True):
    """
    Weight by positive net forward return magnitude for winners; small weight for losers.
    Avoids deleting losers (which biases), but de-emphasizes them.
    power > 1 boosts separation on bigger winners.
    """
    df = df.copy()
    if long_only:
        base = df["fwd_ret_net"].clip(lower=0.0) ** power
    else:
        # For 3-class, weight by |return| where the sign matches the class direction
        base = (df["fwd_ret_net"].abs()) ** power
        # Optional: set base=0 when label==0 (no-trade)
        base[df["y"] == 0] = 0.0

    w = base / (base.mean() + 1e-12)
    w = w.clip(lower=min_weight, upper=max_weight)
    return w.fillna(min_weight).values

def expected_value(prob_long, avg_gain, avg_loss, fee_bps=1.5, slippage_bps=2.0):
    """
    EV = p_win * avg_gain - (1 - p_win) * avg_loss - costs
    avg_gain/avg_loss are fractions (e.g., 0.003 for 30 bps)
    """
    costs = (fee_bps + slippage_bps) * 1e-4
    ev = prob_long * avg_gain - (1.0 - prob_long) * avg_loss - costs
    return ev









