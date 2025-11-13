import csv
import os
import shutil
import socket
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

socket.setdefaulttimeout(float(os.getenv("NET_TIMEOUT", "3")))

# --- canonical data locations ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CSV_PATH = os.path.join(DATA_DIR, "SPY.csv")


# --- Hooks consumed by run_all.py ----------------------------------------------------
def update_spy_data():
    """
    Hook for run_all.py step_refresh_data().
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


# --- Ensure labeled_predictions.csv exists (schema aligned with predict.py) ---
def init_labeled_log_file():
    if not os.path.exists(LABELED_LOG_FILE):
        print("[Init] Creating blank labeled_predictions.csv (predict.py schema)")
        headers = [
            "Date",
            "Label",
            "Pred",
            "Prediction",
            "Proba",
            "Spike_Conf",
            "Crash_Conf",
            "Confidence",
        ]
        with open(LABELED_LOG_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)


# --- Feature List ---
def get_feature_list():
    return [
        "MA_20",
        "EMA_12",
        "EMA_26",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Width",
        "Volatility",
        "OBV",
        "Vol_Ratio",
        "Price_Momentum_10",
        "Acceleration",
        "RSI",
        "RSI_Delta",
        "ZMomentum",
        "Return_Lag1",
        "Return_Lag3",
        "Return_Lag5",
        "RSI_Lag_1",
        "RSI_Lag_3",
        "RSI_Lag_5",
        "Rolling_STD_5",
        "Daily_Return",
        "MACD_x_RSI",
        "Volume_per_ATR",
        "Stoch_K",
        "Stoch_D",
        "BB_PctB",
        "KC_Width",
        "VWAP_Dev",
        "Ret_Skew_20",
        "Ret_Kurt_20",
        "Sent_x_Vol",
        "RSI_x_NewsZ",
        "RSI_x_RedditZ",
    ]


def in_human_speak(label):
    """
    Convert internal labels to human-readable strings.
    0 = CRASH, 1 = NORMAL, 2 = SPIKE
    """
    try:
        if isinstance(label, str) and label.isdigit():
            label = int(label)
    except Exception:
        pass

    mapping = {
        0: "CRASH",
        1: "NORMAL",
        2: "SPIKE",
        "0": "CRASH",
        "1": "NORMAL",
        "2": "SPIKE",
        "NORMAL": "NORMAL",
        "CRASH": "CRASH",
        "SPIKE": "SPIKE",
    }
    return mapping.get(label, str(label))


# --- Log prediction to file ---
def log_prediction_to_file(
    timestamp,
    prediction,
    crash_conf,
    spike_conf,
    close_price,
    open_price=None,
    high=None,
    low=None,
    log_path="logs/daily_predictions.csv",
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    headers = [
        "Date",
        "Timestamp",
        "Prediction",
        "Crash_Conf",
        "Spike_Conf",
        "Close",
        "Open",
        "High",
        "Low",
        "Confidence",
        "Regime",
        "FeatSnapshot",
    ]

    # Safe date string
    try:
        date_str = str(getattr(timestamp, "date", lambda: timestamp)())
    except Exception:
        date_str = str(timestamp)

    regime = os.getenv("REGIME_TAG", "")

    def _hash_file(pth):
        try:
            with open(pth, "rb") as fh:
                import hashlib

                return hashlib.sha1(fh.read()).hexdigest()[:10]
        except Exception:
            return "NA"

    feat_snapshot = "-".join(
        [
            _hash_file("models/market_crash_model.pkl"),
            _hash_file("models/thresholds.json"),
            _hash_file("configs/best_thresholds.json"),
            _hash_file("models/market_crash_model_fwd.pkl"),
            _hash_file("models/thresholds_fwd.json"),
        ]
    )

    row = {
        "Date": date_str,
        "Timestamp": str(timestamp),
        "Prediction": int(prediction) if prediction is not None else 0,
        "Crash_Conf": float(crash_conf) if crash_conf is not None else 0.0,
        "Spike_Conf": float(spike_conf) if spike_conf is not None else 0.0,
        "Close": float(close_price) if close_price is not None else "",
        "Open": float(open_price) if open_price is not None else "",
        "High": float(high) if high is not None else "",
        "Low": float(low) if low is not None else "",
        "Confidence": float(max(crash_conf or 0.0, spike_conf or 0.0)),
        "Regime": regime,
        "FeatSnapshot": feat_snapshot,
    }

    # De-dupe: compare to last row's Date/Prediction/Close
    try:
        if os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
            with open(log_path, newline="") as _f:
                rdr = csv.DictReader(_f)
                last = None
                for r in rdr:
                    last = r
            if last:
                same_date = str(last.get("Date")) == str(row["Date"])
                same_pred = str(last.get("Prediction")) == str(row["Prediction"])
                same_close = str(last.get("Close")) == str(row["Close"])
                if same_date and same_pred and same_close:
                    print("[skip] duplicate daily_predictions row (same Date/Prediction/Close)")
                    return
    except Exception:
        pass

    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists or os.path.getsize(log_path) == 0:
            w.writeheader()
        w.writerow(row)

    # also append to logs/signals.csv with variant-aware labels
    VAR = os.getenv("PREDICT_VARIANT", "crash_spike").strip().lower()
    if VAR == "forward_returns":
        sig = "TRADE" if row["Prediction"] == 1 else "NO-TRADE"
    else:
        # 3-class convention: 0 = CRASH (sell), 1 = NORMAL (hold), 2 = SPIKE (buy)
        if row["Prediction"] == 2:
            sig = "BUY"
        elif row["Prediction"] == 0:
            sig = "SELL"
        else:
            sig = "HOLD"

    signals_path = "logs/signals.csv"
    signals_headers = ["Date", "Signal", "Confidence", "Price", "Spike_Conf", "Crash_Conf"]
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


def label_real_outcomes_from_log(crash_thresh=-0.005, spike_thresh=0.005):
    if not os.path.exists(LOG_FILE):
        print("[⚠️] daily_predictions.csv not found — skipping outcome labeling.")
        return

    df = pd.read_csv(LOG_FILE, parse_dates=["Timestamp"])
    print(f"[DEBUG] read {len(df)} rows from {LOG_FILE}")
    print(df.tail())

    # last write wins per Timestamp
    df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)

    if len(df) < 2:
        print("[⏭] Not enough data to label real outcomes — skipping for now.")
        return

    price_col = "Close"
    if price_col not in df.columns:
        raise KeyError(
            f"[label_real_outcomes_from_log] Expected column '{price_col}' in {LOG_FILE}"
        )

    df["Next_Close"] = df[price_col].shift(-1)
    df["Future_Return"] = (df["Next_Close"] - df[price_col]) / df[price_col]

    # 3-class convention: 0 = CRASH, 1 = NORMAL, 2 = SPIKE
    df["Actual_Event"] = np.select(
        [
            df["Future_Return"] <= crash_thresh,  # big down move
            df["Future_Return"] >= spike_thresh,  # big up move
        ],
        [0, 2],  # 0=CRASH, 2=SPIKE
        default=1,
    )

    df.dropna(subset=["Future_Return"], inplace=True)
    df.to_csv(LABELED_LOG_FILE, index=False)
    print(f"[DEBUG] wrote {len(df)} rows to {LABELED_LOG_FILE}")
    print("[✅] Labeled outcomes written to logs/labeled_predictions.csv")

    backup_logs()


def backup_logs():
    """Make timestamped copies of your two main CSVs into ./backups/"""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("backups", exist_ok=True)

    def _safe_copy(src, dst):
        try:
            if os.path.exists(src) and os.path.getsize(src) > 0:
                shutil.copy(src, dst)
        except Exception as e:
            print(f"[backup] skip {src}: {e}")

    _safe_copy(LOG_FILE, f"backups/daily_predictions_{ts}.csv")
    _safe_copy(LABELED_LOG_FILE, f"backups/labeled_predictions_{ts}.csv")


def summarize_trades(trades, initial_balance=10000, save_plot_path=None):
    """
    Summarizes trade results: final balance, win rate, trade count.
    Compatible with trades from both simulate_trades() and run_backtest().
    """
    if not trades:
        return {
            "final_balance": initial_balance,
            "total_trades": 0,
            "win_rate": 0.0,
            "equity_curve": [initial_balance],
        }

    equity = [initial_balance]
    wins = 0
    total_trades = 0

    for trade in trades:
        if "ROI" in trade:
            roi = trade["ROI"]
        elif "Entry_Price" in trade and "Exit_Price" in trade:
            roi = (trade["Exit_Price"] - trade["Entry_Price"]) / trade["Entry_Price"]
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
        "final_balance": final_balance,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "equity_curve": equity,
    }


def expected_value(prob_long, avg_gain, avg_loss, fee_bps=1.5, slippage_bps=2.0):
    costs = (fee_bps + slippage_bps) * 1e-4
    return prob_long * avg_gain - (1.0 - prob_long) * avg_loss - costs


# --- injected minimal helpers ---
def add_features(df):
    import pandas as pd

    d = df.copy()

    if not isinstance(d.index, pd.DatetimeIndex):
        if "Date" in d.columns:
            d.index = pd.to_datetime(d["Date"], errors="coerce")
        else:
            d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[d.index.notna()]
    d.index = d.index.tz_localize(None)
    d.index.name = "Date"

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Basic return structure
    d["Daily_Return"] = d["Close"].pct_change()
    d["Return_Lag1"] = d["Close"].pct_change(1)
    d["Return_Lag3"] = d["Close"].pct_change(3)
    d["Return_Lag5"] = d["Close"].pct_change(5)

    # Rolling volatility
    d["Rolling_STD_5"] = d["Daily_Return"].rolling(5).std()
    d["Volatility"] = d["Daily_Return"].rolling(20).std()

    # Moving averages and MACD family
    d["MA_20"] = d["Close"].rolling(20).mean()
    d["EMA_12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA_26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Histogram"] = d["MACD"] - d["MACD_Signal"]

    # Bollinger bands and related features
    bb_window = 20
    bb_mean = d["Close"].rolling(bb_window).mean()
    bb_std = d["Close"].rolling(bb_window).std()
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std
    denom_bb = bb_mean.replace(0, np.nan)
    d["BB_Width"] = (bb_upper - bb_lower) / denom_bb
    band_span = (bb_upper - bb_lower).replace(0, np.nan)
    d["BB_PctB"] = (d["Close"] - bb_lower) / band_span

    # ATR and derived features
    hl = (d["High"] - d["Low"]).abs()
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"] - d["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["Dist_High20_ATR"] = (d["Close"] - d["High"].rolling(20).max()) / (d["ATR_14"] + 1e-9)

    # Stochastics
    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * ((d["Close"] - low14) / ((high14 - low14) + 1e-9))
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    # Gap percentage
    d["Gap_Pct"] = (d["Open"] - d["Close"].shift(1)) / d["Close"].shift(1)

    # OBV and volume-based features
    price_diff = d["Close"].diff()
    direction = np.sign(price_diff).fillna(0.0)
    d["OBV"] = (direction * d["Volume"].fillna(0.0)).cumsum()
    vol_roll = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"] = d["Volume"] / vol_roll

    # Momentum and z-momentum
    d["Price_Momentum_10"] = d["Close"].pct_change(10)
    roll_mom_mean = d["Price_Momentum_10"].rolling(60).mean()
    roll_mom_std = d["Price_Momentum_10"].rolling(60).std()
    d["ZMomentum"] = (d["Price_Momentum_10"] - roll_mom_mean) / roll_mom_std

    # Acceleration
    d["Acceleration"] = d["Daily_Return"] - d["Daily_Return"].shift(1)

    # RSI and derived features
    delta = d["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    d["RSI_Delta"] = d["RSI"].diff()
    d["RSI_Lag_1"] = d["RSI"].shift(1)
    d["RSI_Lag_3"] = d["RSI"].shift(3)
    d["RSI_Lag_5"] = d["RSI"].shift(5)

    # Higher-order return stats
    d["Ret_Skew_20"] = d["Daily_Return"].rolling(20).skew()
    d["Ret_Kurt_20"] = d["Daily_Return"].rolling(20).kurt()

    # VWAP and deviation
    typical_price = (d["High"] + d["Low"] + d["Close"]) / 3.0
    cum_vol = d["Volume"].replace(0, np.nan).cumsum()
    d["VWAP"] = (typical_price * d["Volume"]).cumsum() / cum_vol
    d["VWAP_Dev"] = (d["Close"] - d["VWAP"]) / d["VWAP"]

    # Volume scaled by ATR
    d["Volume_per_ATR"] = d["Volume"] / (d["ATR_14"] + 1e-9)

    # KC_Width via Keltner-like channel around an EMA
    ema_price_20 = d["Close"].ewm(span=20, adjust=False).mean()
    kc_upper = ema_price_20 + 2.0 * d["ATR_14"]
    kc_lower = ema_price_20 - 2.0 * d["ATR_14"]
    kc_span = (kc_upper - kc_lower).replace(0, np.nan)
    d["KC_Width"] = kc_span / ema_price_20.replace(0, np.nan)

    # External signals merged after core price features
    try:
        from external_signals import add_external_signals as _add_ext

        d = _add_ext(d)
    except Exception:
        pass

    # Interaction features that depend on sentiment and volume/RSI
    if "News_Sent_Z20" in d.columns and "Vol_Ratio" in d.columns:
        d["Sent_x_Vol"] = d["News_Sent_Z20"] * d["Vol_Ratio"]
    if "RSI" in d.columns and "News_Sent_Z20" in d.columns:
        d["RSI_x_NewsZ"] = d["RSI"] * d["News_Sent_Z20"]
    if "RSI" in d.columns and "Reddit_Sent_Z20" in d.columns:
        d["RSI_x_RedditZ"] = d["RSI"] * d["Reddit_Sent_Z20"]

    # Second-order feature interactions
    if "MACD" in d.columns and "RSI" in d.columns:
        d["MACD_x_RSI"] = d["MACD"] * d["RSI"]

    # Build the rich feature set: technical core + macro/sentiment extras
    core_features = get_feature_list()
    extra_features = [
        "Gap_Pct",
        "Dist_High20_ATR",
        "ATR_14",
        "VIX",
        "Sector_MedianRet_20",
        "Sector_Dispersion_20",
        "Credit_Spread_20",
        "TNX_Change_20",
        "DXY_Change_20",
        "News_Sent_Z20",
        "Reddit_Sent_Z20",
    ]
    candidate_features = list(dict.fromkeys(core_features + extra_features))
    feature_cols = [c for c in candidate_features if c in d.columns]
    feature_cols = list(dict.fromkeys(feature_cols))
    return d, feature_cols


def finalize_features(df, feature_cols):
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[feature_cols]
    out = out.replace([np.inf, -np.inf], np.nan)
    # median impute per-column (scalar), safe for TS when used inside CV/pipelines
    out = out.fillna(out.median(numeric_only=True))
    return out


# --- injected missing helpers (safe) ---
def send_telegram_alert(text, token=None, chat_id=None):
    import json

    try:
        import requests
    except Exception:
        print("⚠️ Telegram: requests not installed; skipping.")
        return False
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
        try:
            err = r.json()
        except Exception:
            err = {"status_code": r.status_code, "text": r.text[:200]}
        print(f"⚠️ Telegram error: {json.dumps(err)}")
        return False
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")
        return False


def notify_user(prediction, crash_conf, spike_conf):
    try:
        label = "TRADE" if str(prediction) == "1" else "NO-TRADE"
        print(f"[notify] {label} — Crash={crash_conf:.3f}, Spike={spike_conf:.3f}")
    except Exception:
        print("[notify] event")


# === Forward-returns labeling & safety helpers =====================================
def add_forward_returns_and_labels(
    df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 5,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
    long_only: bool = True,
    pos_threshold: float = 0.0,
):
    d = df.copy()
    if price_col not in d.columns:
        raise KeyError(f"[add_forward_returns_and_labels] '{price_col}' not in df.")

    d[price_col] = pd.to_numeric(d[price_col], errors="coerce")
    cost = (float(fee_bps) + float(slippage_bps)) * 1e-4

    d["fwd_price"] = d[price_col].shift(-int(horizon))
    d["fwd_ret_raw"] = (d["fwd_price"] / d[price_col]) - 1.0
    d["fwd_ret_net"] = d["fwd_ret_raw"] - cost
    d["horizon_forward"] = int(horizon)

    d["y"] = (
        (d["fwd_ret_net"] >= float(pos_threshold)).astype(int)
        if long_only
        else (d["fwd_ret_net"] >= float(pos_threshold)).astype(int)
    )
    return d


def compute_sample_weights(
    df: pd.DataFrame,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
    power: float = 1.0,
    long_only: bool = True,
):
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    if "fwd_ret_net" not in df.columns:
        return np.ones(n, dtype=float)

    r = pd.to_numeric(df["fwd_ret_net"], errors="coerce").fillna(0.0).to_numpy()
    pos = np.maximum(r, 0.0)
    scale = 0.01
    w = 1.0 + (pos / scale) ** float(power)
    w = np.clip(w, float(min_weight), float(max_weight))
    return w


def ensure_no_future_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    horizon_col: str = "horizon_forward",
):
    blacklist = {
        "y",
        "fwd_price",
        "fwd_ret_raw",
        "fwd_ret_net",
        horizon_col,
        "horizon",
        "future_return",
        "future_price",
    }
    feats = set(map(str, feature_cols))
    leaks = feats & blacklist
    leaks |= {c for c in feats if c.lower().startswith(("fwd_", "future_"))}
    if leaks:
        raise RuntimeError(f"[ensure_no_future_leakage] Leaky features detected: {sorted(leaks)}")
    return True


# --- stub to avoid crashes if triple-barrier import is hit -------------
def label_events_triple_barrier(
    df: pd.DataFrame,
    vol_col: str = "ATR_14",
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    t_max: int = 10,
):
    out = df.copy()
    out["Event"] = 0
    return out


# ================================================================================


# === Canonical SPY loader =====================
def load_SPY_data() -> pd.DataFrame:
    header = pd.read_csv(CSV_PATH, nrows=1, low_memory=False)
    want = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    usecols = [c for c in want if c in header.columns]

    df = pd.read_csv(
        CSV_PATH,
        usecols=usecols,
        parse_dates=["Date"],
        low_memory=False,
    )
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").set_index("Date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise AssertionError("SPY loader: index is not DatetimeIndex")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if "Open" not in df.columns or "Close" not in df.columns:
        raise AssertionError("SPY loader: missing required OHLC columns")

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# =========================
# Canonical loader + helpers
# =========================
def safe_read_csv(path, prefer_index=None, **kwargs):
    """Read CSV; if prefer_index is provided, set it if present. Ignores unsupported kwargs."""
    df = pd.read_csv(path, **kwargs)
    if prefer_index and prefer_index in df.columns:
        df = df.set_index(prefer_index)
    return df


# v2 helpers kept for compatibility (not wired by default)
def _add_forward_returns_and_labels_v2(
    df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 1,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
    long_only: bool = True,
    pos_threshold: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    if price_col not in out.columns:
        raise RuntimeError(f"add_forward_returns_and_labels: missing price_col '{price_col}'")

    out["fwd_price"] = out[price_col].shift(-horizon)
    out["fwd_ret_raw"] = out["fwd_price"] / out[price_col] - 1.0
    c = (fee_bps + slippage_bps) / 1e4
    out["fwd_ret_net"] = (1.0 + out["fwd_ret_raw"]) * (1.0 - c) - 1.0

    if long_only:
        out["y"] = (out["fwd_ret_net"] >= float(pos_threshold)).astype(int)
    else:
        neg_thr = -abs(float(pos_threshold))
        out["y"] = 0
        out.loc[out["fwd_ret_net"] >= float(pos_threshold), "y"] = 1
        out.loc[out["fwd_ret_net"] <= neg_thr, "y"] = -1

    out["horizon_forward"] = int(horizon)
    return out


def _compute_sample_weights_v2(
    df_labeled: pd.DataFrame,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    power: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    dfx = df_labeled
    if "fwd_ret_net" not in dfx.columns:
        raise RuntimeError(
            "compute_sample_weights: missing fwd_ret_net (call add_forward_returns_and_labels first)"
        )
    mag = np.abs(np.asarray(dfx["fwd_ret_net"].fillna(0.0)))
    if power != 1.0:
        mag = mag ** float(power)
    if mag.max() > 0:
        mag = mag / mag.max()
    w = min_weight + (max_weight - min_weight) * mag
    if long_only and "y" in dfx.columns:
        w = np.where(dfx["y"].values == 0, np.maximum(min_weight, 0.8 * w), w)
    return w


def _ensure_no_future_leakage_v2(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    horizon_col: str = "horizon_forward",
) -> None:
    forbidden = {"y", "fwd_price", "fwd_ret_raw", "fwd_ret_net", horizon_col}
    leaky = sorted(forbidden.intersection(set(feature_cols)))
    if leaky:
        raise RuntimeError(f"Leakage: features contain forward-looking columns: {leaky}")
    if horizon_col in feature_cols:
        raise RuntimeError(f"Leakage: '{horizon_col}' is in features")
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")
