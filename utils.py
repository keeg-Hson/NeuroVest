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

# --- canonical data locations ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CSV_PATH = os.path.join(DATA_DIR, "SPY.csv")



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
    import csv, os, hashlib

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    headers = [
        "Date","Timestamp","Prediction","Crash_Conf","Spike_Conf",
        "Close","Open","High","Low","Confidence","Regime","FeatSnapshot"
    ]

    # Safe date string
    try:
        date_str = str(getattr(timestamp, "date", lambda: timestamp)())
    except Exception:
        date_str = str(timestamp)

    # regime / snapshot hashes
    regime = os.getenv("REGIME_TAG", "")
    def _hash_file(pth):
        try:
            with open(pth,"rb") as fh:
                import hashlib
                return hashlib.sha1(fh.read()).hexdigest()[:10]
        except Exception:
            return "NA"
    feat_snapshot = "-".join([
        _hash_file("models/market_crash_model.pkl"),
        _hash_file("models/thresholds.json"),
        _hash_file("configs/best_thresholds.json"),
        _hash_file("models/market_crash_model_fwd.pkl"),
        _hash_file("models/thresholds_fwd.json"),
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
        "Regime":      regime,
        "FeatSnapshot": feat_snapshot,
    }

    # de-dupe: avoid appending if the last row has same Date/Prediction/Close
    try:
        if os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
            with open(log_path, "r", newline="") as _f:
                rdr = csv.DictReader(_f)
                last = None
                for last in rdr:
                    pass
            if last:
                same_date  = str(last.get("Date")) == str(row["Date"])
                same_pred  = str(last.get("Prediction")) == str(row["Prediction"])
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
    VAR = os.getenv("PREDICT_VARIANT","crash_spike").strip().lower()
    if VAR == "forward_returns":
        sig = "TRADE" if row["Prediction"] == 1 else "NO-TRADE"
    else:
        sig = "BUY" if row["Prediction"] == 2 else ("SELL" if row["Prediction"] == 1 else "HOLD")

    signals_path = "logs/signals.csv"
    signals_headers = ["Date","Signal","Confidence","Price","Spike_Conf","Crash_Conf"]
    sig_row = {
        "Date":        date_str,
        "Signal":      sig,
        "Confidence":  row["Confidence"],
        "Price":       row["Close"],
        "Spike_Conf":  row["Spike_Conf"],
        "Crash_Conf":  row["Crash_Conf"],
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
    import pandas as pd
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"

    def _read_one(p: Path) -> pd.DataFrame:
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.map(str).str.strip()
        dcol = next((c for c in df.columns if c.lower().startswith("date")), None)
        if not dcol:
            return pd.DataFrame()

        dt = pd.to_datetime(df[dcol].astype(str).str.slice(0,10), errors="coerce", format="mixed")
        df = df.loc[dt.notna()].copy()
        df[dcol] = dt
        df = df.drop_duplicates(subset=[dcol]).set_index(dcol).sort_index()

        def pick(*names):
            for n in names:
                for c in df.columns:
                    if c == n or c.startswith(n + "."):
                        s = pd.to_numeric(df[c], errors="coerce")
                        if s.notna().any():
                            return s
            return pd.Series(index=df.index, dtype="float64")

        out = pd.DataFrame(index=df.index)
        out["Open"]      = pick("Open","1. open")
        out["High"]      = pick("High","2. high")
        out["Low"]       = pick("Low","3. low")
        out["Close"]     = pick("Close","4. close","Adj Close","adjclose","close")
        out["Adj Close"] = pick("Adj Close","adjclose","close")
        out["Volume"]    = pick("Volume","5. volume","volume")
        return out

    parts = []
    for pth in [DATA / "SPY.csv", DATA / "spy_daily.csv"]:
        d = _read_one(pth)
        if not d.empty:
            parts.append(d)

    if not parts:
        raise FileNotFoundError("No SPY data could be loaded from data/spy_daily.csv or data/SPY.csv")

    base = pd.concat(parts)
    base.index = pd.to_datetime(base.index, errors="coerce").tz_localize(None).normalize()
    base = base[base.index.notna()].sort_index()
    base = base.groupby(level=0).agg(lambda col: col.dropna().iloc[-1] if col.notna().any() else pd.NA)
    base = base[base.index >= pd.Timestamp("1993-01-29")]
    base.index.name = "Date"
    return base

def expected_value(prob_long, avg_gain, avg_loss, fee_bps=1.5, slippage_bps=2.0):
    costs = (fee_bps + slippage_bps) * 1e-4
    return prob_long * avg_gain - (1.0 - prob_long) * avg_loss - costs

# --- injected minimal helpers ---

def add_features(df):
    import numpy as np
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

    for col in ["Open","High","Low","Close","Volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d["Daily_Return"] = d["Close"].pct_change()
    d["Return_Lag1"] = d["Close"].pct_change(1)
    d["Return_Lag3"] = d["Close"].pct_change(3)
    d["Return_Lag5"] = d["Close"].pct_change(5)

    low14  = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * ((d["Close"] - low14) / ((high14 - low14) + 1e-9))

    hl = (d["High"] - d["Low"]).abs()
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"]  - d["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["Dist_High20_ATR"] = (d["Close"] - d["High"].rolling(20).max()) / (d["ATR_14"] + 1e-9)

    d["ZMomentum"] = (d["Close"] - d["Close"].rolling(10).mean()) / (d["Close"].rolling(10).std() + 1e-9)
    d["Acceleration"] = (d["Close"] - d["Close"].shift(10)).diff()
    d["Gap_Pct"] = (d["Open"] - d["Close"].shift()) / (d["Close"].shift() + 1e-9)

    try:
        from external_signals import add_external_signals as _add_ext
        d = _add_ext(d)
    except Exception:
        pass

    feature_cols = [
        "ZMomentum","Acceleration",
        "Return_Lag1","Return_Lag3","Return_Lag5","Daily_Return",
        "Stoch_K","Gap_Pct","Dist_High20_ATR","ATR_14",
        "VIX","Sector_MedianRet_20","Sector_Dispersion_20",
        "Credit_Spread_20","TNX_Change_20","DXY_Change_20",
        "News_Sent_Z20","Reddit_Sent_Z20",
    ]
    feature_cols = [c for c in feature_cols if c in d.columns]
    feature_cols = list(dict.fromkeys(feature_cols))
    return d, feature_cols

def finalize_features(df, feature_cols):
    import numpy as np
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[feature_cols]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out

# --- injected missing helpers (safe) ---

def send_telegram_alert(text, token=None, chat_id=None):
    import os, json
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
    """
    Adds:
      - 'fwd_price': price shifted -horizon
      - 'fwd_ret_raw': (fwd_price / Close) - 1
      - 'fwd_ret_net': fees/slippage adjusted forward return
      - 'horizon_forward': the lookahead horizon (int)
      - 'y': 1 if forward trade is positive (>= pos_threshold), else 0 (long_only)
             If long_only=False, you can extend to 3-class, but train.py expects {0,1}.
    """
    d = df.copy()
    if price_col not in d.columns:
        raise KeyError(f"[add_forward_returns_and_labels] '{price_col}' not in df.")

    d[price_col] = pd.to_numeric(d[price_col], errors="coerce")
    cost = (float(fee_bps) + float(slippage_bps)) * 1e-4  # per round-trip, conservative

    d["fwd_price"]   = d[price_col].shift(-int(horizon))
    d["fwd_ret_raw"] = (d["fwd_price"] / d[price_col]) - 1.0
    # Subtract costs on both sides (entry/exit) once; adjust if model fills separately
    d["fwd_ret_net"] = d["fwd_ret_raw"] - cost

    d["horizon_forward"] = int(horizon)

    # Binary label: trade (1) vs no-trade (0)
    if long_only:
        d["y"] = (d["fwd_ret_net"] >= float(pos_threshold)).astype(int)
    else:
        d["y"] = (d["fwd_ret_net"] >= float(pos_threshold)).astype(int)

    return d


def compute_sample_weights(
    df: pd.DataFrame,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
    power: float = 1.0,
    long_only: bool = True,
):
    """
    Profit-aware weights. More positive forward return ⇒ larger weight.
    - If 'fwd_ret_net' missing, returns ones.
    - Clipped to [min_weight, max_weight].
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)

    if "fwd_ret_net" not in df.columns:
        return np.ones(n, dtype=float)

    r = pd.to_numeric(df["fwd_ret_net"], errors="coerce").fillna(0.0).to_numpy()

    # Emphasize positive outcomes; non-positive stay closer to min_weight
    pos = np.maximum(r, 0.0)
    # Normalize by a small scale to avoid exploding weights; 1% is a decent default scale
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
    """
    Sanity checks that features don't include forward-looking columns.
    Will raise if obviously leaky columns are present.
    """
    blacklist = {
        "y", "fwd_price", "fwd_ret_raw", "fwd_ret_net",
        horizon_col, "horizon", "future_return", "future_price"
    }
    feats = set(map(str, feature_cols))
    leaks = feats & blacklist
    # Also catch any features that *look* forward-y by name
    leaks |= {c for c in feats if c.lower().startswith(("fwd_", "future_"))}

    if leaks:
        raise RuntimeError(f"[ensure_no_future_leakage] Leaky features detected: {sorted(leaks)}")

    # Optional: could check that all features use only information up to t (e.g., via lags).
    return True


# --- stub to avoid crashes if triple-barrier import is hit -------------
def label_events_triple_barrier(
    df: pd.DataFrame,
    vol_col: str = "ATR_14",
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    t_max: int = 10,
):
    """
    Minimal stub. Produces an Event column of zeros so training will fall back
    to forward-returns (train.py already handles 'not enough class diversity').
    Replace with your full triple-barrier labeling when ready.
    """
    out = df.copy()
    out["Event"] = 0
    return out
# ================================================================================

# --- SPY data loader ---
import pandas as pd, os
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "SPY.csv")

def load_SPY_data(parse_dates=True):
    if parse_dates:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    else:
        df = pd.read_csv(CSV_PATH)
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# =========================
# Canonical loader + helpers
# =========================

def safe_read_csv(path: str, **kwargs):
    """
    A tolerant CSV reader used everywhere we read our own data files.
    - Parses Date if present
    - Strips columns
    - Returns empty DataFrame on failure (with a warning)
    """
    import pandas as pd
    try:
        kw = dict(kwargs)
        if "parse_dates" not in kw and "Date" in pd.read_csv(path, nrows=1).columns:
            kw["parse_dates"] = ["Date"]
        df = pd.read_csv(path, **kw)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"⚠️ safe_read_csv: could not read {path} ({e})")
        return pd.DataFrame()


def add_forward_returns_and_labels(
    df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 1,
    fee_bps: float = 1.5,
    slippage_bps: float = 2.0,
    long_only: bool = True,
    pos_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build forward-returns labels:
      - fwd_price, fwd_ret_raw, fwd_ret_net
      - y ∈ {0,1} for forward_returns (1 = take-trade), or { -1,0,1 } if long_only=False
    """
    out = df.copy()
    if price_col not in out.columns:
        raise RuntimeError(f"add_forward_returns_and_labels: missing price_col '{price_col}'")

    out["fwd_price"]   = out[price_col].shift(-horizon)
    out["fwd_ret_raw"] = out["fwd_price"] / out[price_col] - 1.0

    # simple round-trip costs
    c = (fee_bps + slippage_bps) / 1e4
    out["fwd_ret_net"] = (1.0 + out["fwd_ret_raw"]) * (1.0 - c) - 1.0

    if long_only:
        # 1 = trade if forward net return >= threshold; else 0 = no-trade
        out["y"] = (out["fwd_ret_net"] >= float(pos_threshold)).astype(int)
    else:
        # ternary label: go short if sufficiently negative, otherwise long if >= pos_threshold
        neg_thr = -abs(float(pos_threshold))
        out["y"] = 0
        out.loc[out["fwd_ret_net"] >= float(pos_threshold), "y"] = 1
        out.loc[out["fwd_ret_net"] <= neg_thr, "y"] = -1

    # horizon stamp (helps leakage checks)
    out["horizon_forward"] = int(horizon)
    return out


def compute_sample_weights(
    df_labeled: pd.DataFrame,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    power: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Emphasize samples with larger |forward net return|.
    Bounded in [min_weight, max_weight].
    """
    import numpy as _np
    dfx = df_labeled
    if "fwd_ret_net" not in dfx.columns:
        raise RuntimeError("compute_sample_weights: missing fwd_ret_net (call add_forward_returns_and_labels first)")

    mag = _np.abs(_np.asarray(dfx["fwd_ret_net"].fillna(0.0)))
    if power != 1.0:
        mag = mag ** float(power)

    # normalize to [0,1] then scale into [min,max]
    if mag.max() > 0:
        mag = mag / mag.max()
    w = min_weight + (max_weight - min_weight) * mag

    # optional: downweight "no-trade" (y=0) slightly in long_only regime
    if long_only and "y" in dfx.columns:
        w = _np.where(dfx["y"].values == 0, _np.maximum(min_weight, 0.8 * w), w)

    return w


def ensure_no_future_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    horizon_col: str = "horizon_forward",
) -> None:
    """
    Very fast structural checks that your model inputs can't peek forward.
    - Forbids fwd/*future columns in features
    - Ensures horizon stamp exists if present in pipeline
    """
    forbidden = {"y", "fwd_price", "fwd_ret_raw", "fwd_ret_net", horizon_col}
    leaky = sorted(forbidden.intersection(set(feature_cols)))
    if leaky:
        raise RuntimeError(f"Leakage: features contain forward-looking columns: {leaky}")

    # If horizon stamp exists, it must *NOT* be in features
    if horizon_col in feature_cols:
        raise RuntimeError(f"Leakage: '{horizon_col}' is in features")

    # Targets should be present
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")
    
# utils.py (put near other IO helpers)
import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data")
CSV_PATH = os.path.join(DATA_DIR, "SPY.csv")

def load_SPY_data() -> pd.DataFrame:
    """
    Canonical SPY daily loader.
    - Always parse Date as datetime
    - Drop bad rows
    - De-duplicate by Date (keep last)
    - Return with DatetimeIndex (daily, sorted, unique)
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    # keep only columns we use
    keep = [c for c in ["Date","Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep].copy()

    # drop obvious bad rows
    df = df.dropna(subset=["Date"])
    # ensure monotonic, unique dates; keep the freshest duplicate if any
    df = (df.sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .set_index("Date"))

    # enforce numeric dtypes
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # final cleanup and sort
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df




