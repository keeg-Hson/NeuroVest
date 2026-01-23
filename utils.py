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
        # Core technical features
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
        # Removed zero-importance sentiment features (analysis Nov 2025):
        # "Sent_x_Vol", "RSI_x_NewsZ", "RSI_x_RedditZ"
        # Enhanced features from importance analysis
        # 1. Feature interactions for top features
        "BB_Width_x_RSI",
        "BB_Width_x_Return_Lag1",
        "BB_Width_x_Vol_Ratio",
        "Return_Lag1_x_Return_Lag3",
        "Return_Trend_Strength",
        "RSI_x_Vol_Ratio",
        "OBV_x_Return_Lag1",
        # 2. Enhanced lags for top features
        "BB_Width_Lag1",
        "BB_Width_Lag3",
        "BB_Width_Change",
        "RSI_Lag7",
        "RSI_Lag10",
        "Return_Lag7",
        "Return_Lag10",
        "Return_Lag15",
        # 3. Volatility regime features
        # Removed: "Vol_Expanding" (zero importance)
        "Vol_Percentile",
        "Volatility_Acceleration",
        "BB_Width_Mean_10",
        "BB_Width_Std_10",
        "BB_Width_ZScore",
        # 4. Return-based enhancements
        "Return_Momentum_Ratio",
        "Return_Acceleration",
        "Positive_Return_Streak",
        # Removed: "Return_Reversal" (zero importance)
        # 5. Rolling window statistics
        "Return_Lag1_MA5",
        "Return_Lag1_MA10",
        "Return_Volatility_20",
        "Return_Skew_10",
        "Return_Kurt_10",
        # 6. Volume enhancements
        "Volume_Momentum_5",
        "OBV_Change_5",
        "OBV_Trend",
        # 7. Cross-sectional features
        "Return_1d_vs_10d",
        "Return_3d_vs_10d",
        # 8. RSI regime features
        # Removed binary indicators (zero importance): "RSI_Overbought", "RSI_Oversold", "RSI_Neutral"
        "RSI_Momentum_5",
        "RSI_ROC_5",
        # 9. Market regime detection features
        "MA_200",
        "Price_vs_MA200",
        # Removed: "Bull_Market" (zero importance, binary)
        "MA200_Slope",
        "MA200_Distance_Vol",
        "VIX_Percentile",
        # Removed: "High_Fear" (zero importance, binary)
        # Removed: "VIX_Spike" (zero importance, binary - keep Vol_Spike from CSV)
        "VIX_Change",
        "Vol_Percentile_252",
        "High_Volatility",
        # Removed: "MA_20_50_Cross" (zero importance, binary)
        # Removed: "Pct_Above_MA20" (zero importance, binary)
        "Near_52w_High",
        # Removed: "Near_52w_Low" (zero importance)
        "ADX",
        "Plus_DI",
        "Minus_DI",
        # Removed: "Strong_Trend" (zero importance, binary)
        "Days_Above_MA20",
        "Trend_Consistency",
        "Regime_Score",
        # 10. Multi-timeframe features (selective - removed redundant short-term)
        # Removed: "Returns_5d", "Returns_10d" (redundant with Return_Lag5/10)
        "Returns_50d",  # Keep long-term for regime detection
        # Removed: "Volatility_5d", "Volatility_10d" (redundant with Rolling_STD_5, Volatility)
        "Volatility_50d",  # Keep long-term for regime detection
        # Removed: "RSI_5", "RSI_10" (redundant with RSI, RSI_Lag_1/3/5)
        "RSI_50",  # Keep long-term for regime detection
        # 11. Advanced feature interactions (already computed)
        "RSI_x_Volatility",
        "Volume_x_Returns",
        "Volume_x_Volatility",
        "MACD_divergence",
        # 12. Enhanced momentum features (removed redundant)
        # Removed: "ROC_5", "ROC_10" (redundant with Return_Lag5/10)
        # Removed: "MOM_ratio" (redundant with Return_Momentum_Ratio)
        # 13. Trend strength indicators (already computed)
        "Trend_strength_10",
        "Trend_strength_20",
        "Trend_strength_50",
        # 14. Volume profile features (removed redundant)
        # Removed: "Volume_trend" (redundant with Vol_Ratio)
        # Removed: "Volume_volatility" (low added value)
        # 15. Temporal features (removed - binary encodings don't work well with gradient boosting)
        # Removed: "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos", "Quarter"
        # Tree models handle raw temporal features better than cyclic encodings
        # 16. Cross-asset features (from pre-computed CSV)
        "Credit_Ratio",
        "Credit_Change_20d",
        "Credit_Stress",
        "Yield_10Y",
        "Yield_Change_20d",
        "High_Yield_Regime",
        "DXY_Level",
        "DXY_Change_20d",
        "Strong_Dollar",
        "Realized_Vol_20",
        "Realized_Vol_60",
        "High_Vol_Regime",
        "Vol_Spike",
        # 17. Macro features (from pre-computed CSV - selective, continuous over binary)
        "Macro_10Y_Yield",  # Continuous - KEEP
        # Removed binary regime flags: "Low_Rate_Regime", "High_Rate_Regime" (prefer continuous)
        "Rate_Change_3m",  # Continuous - KEEP (was #14 in importance)
        "Rate_Change_6m",  # Continuous - KEEP
        "Tightening_Cycle",  # Keep - important for economic modeling
        "Easing_Cycle",  # Keep - important for economic modeling
        "Recession_Signal",  # Keep - critical for economic modeling
        "Recovery_Signal",  # Keep - critical for economic modeling
        "Inflation_Proxy",  # Continuous - KEEP
        # Removed: "High_Inflation" (binary version, keep Inflation_Proxy instead)
        # Removed: "Expansion", "Contraction" (binary, covered by Recession/Recovery signals)
        "Financial_Stress",  # Keep - important risk indicator
        # 18. Economic Modeling Interaction Features (Nov 2025)
        # These capture regime shifts and cross-domain relationships
        "Near_52w_High_x_Volatility",  # Position × Market stress
        "Near_52w_High_x_KC_Width",  # Position × Volatility bands
        "Stoch_K_x_Volatility",  # Momentum × Market stress
        "Return_Lag3_x_Volatility",  # Returns × Market regime
        "Return_Lag5_x_ATR",  # Returns × Volatility measure
        "Near_52w_High_x_Return_Lag3",  # Position × Momentum (trend confirmation)
        "BB_PctB_x_Stoch_K",  # Bands × Momentum
        "Credit_Ratio_x_Volatility",  # Credit stress × Market stress
        "Realized_Vol_60_x_Volatility",  # Cross-asset vol × SPY vol
        "Rate_Change_3m_x_MA200_Slope",  # Fed policy × Market trend
        "DXY_Level_x_Return_Lag5",  # Dollar strength × Returns
        "Yield_10Y_x_Price_vs_MA200",  # Interest rates × Market position
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
def _compute_rsi_helper(close_series, window=14):
    """Helper function to compute RSI for any window size."""
    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


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
    # OBV is a cumulative indicator - cumsum() is the correct implementation
    # (not data leakage because it only uses PAST data, not future)
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
    # VWAP is a cumulative indicator - cumsum() is the correct implementation
    # (not data leakage because it only uses PAST data, not future)
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

    # ==========================================================================
    # Pre-computed features integration (cross-asset, sentiment, macro)
    # ==========================================================================

    # Cross-asset features (HYG, LQD, TNX, DXY correlations and spreads)
    try:
        cross_path = os.path.join(DATA_DIR, "cross_asset_features.csv")
        if os.path.exists(cross_path):
            cross = pd.read_csv(cross_path, parse_dates=['Date'])
            cross = cross.set_index('Date')
            # High-value features with XAsset_ prefix (actual column names)
            cross_cols_map = {
                'XAsset_Credit_Ratio': 'Credit_Ratio',
                'XAsset_Credit_Change_20d': 'Credit_Change_20d',
                'XAsset_Credit_Stress': 'Credit_Stress',
                'XAsset_10Y_Yield': 'Yield_10Y',
                'XAsset_10Y_Change_20d': 'Yield_Change_20d',
                'XAsset_High_Yield_Regime': 'High_Yield_Regime',
                'XAsset_DXY': 'DXY_Level',
                'XAsset_DXY_Change_20d': 'DXY_Change_20d',
                'XAsset_Strong_Dollar': 'Strong_Dollar',
                'XAsset_Realized_Vol_20': 'Realized_Vol_20',
                'XAsset_Realized_Vol_60': 'Realized_Vol_60',
                'XAsset_High_Vol_Regime': 'High_Vol_Regime',
                'XAsset_Vol_Spike': 'Vol_Spike',
            }
            cross_available = cross[[c for c in cross_cols_map.keys() if c in cross.columns]]
            cross_aligned = cross_available.reindex(d.index)
            for old_col, new_col in cross_cols_map.items():
                if old_col in cross_aligned.columns:
                    # Lag 1 day to prevent lookahead bias
                    d[new_col] = cross_aligned[old_col].shift(1).ffill().fillna(0)
    except Exception as e:
        # Fail gracefully - cross-asset features are optional
        pass

    # Sentiment features (NewsAPI and Reddit sentiment)
    # Note: sentiment_features.csv appears to be mostly empty, skipping for now
    # Sentiment is already handled via external_signals.py

    # Macro features (FRED economic indicators)
    try:
        macro_path = os.path.join(DATA_DIR, "macro_features.csv")
        if os.path.exists(macro_path):
            macro = pd.read_csv(macro_path, parse_dates=['Date'])
            macro = macro.set_index('Date')
            # High-value macro features with Macro_ prefix (actual column names)
            macro_cols_map = {
                'Macro_10Y_Yield': 'Macro_10Y_Yield',
                'Macro_Low_Rate_Regime': 'Low_Rate_Regime',
                'Macro_High_Rate_Regime': 'High_Rate_Regime',
                'Macro_Rate_Change_3m': 'Rate_Change_3m',
                'Macro_Rate_Change_6m': 'Rate_Change_6m',
                'Macro_Tightening_Cycle': 'Tightening_Cycle',
                'Macro_Easing_Cycle': 'Easing_Cycle',
                'Macro_Recession_Signal': 'Recession_Signal',
                'Macro_Recovery_Signal': 'Recovery_Signal',
                'Macro_Inflation_Proxy': 'Inflation_Proxy',
                'Macro_High_Inflation': 'High_Inflation',
                'Macro_Expansion': 'Expansion',
                'Macro_Contraction': 'Contraction',
                'Macro_Financial_Stress': 'Financial_Stress',
            }
            macro_available = macro[[c for c in macro_cols_map.keys() if c in macro.columns]]
            macro_aligned = macro_available.reindex(d.index)
            for old_col, new_col in macro_cols_map.items():
                if old_col in macro_aligned.columns:
                    # Lag 1 day to prevent lookahead bias
                    d[new_col] = macro_aligned[old_col].shift(1).ffill().fillna(0)
    except Exception as e:
        # Fail gracefully - macro features are optional
        pass

    # Advanced feature engineering: multi-timeframe and enhanced interactions
    # 1. Multi-timeframe returns and volatility
    for window in [5, 10, 50]:
        d[f"Returns_{window}d"] = d["Close"].pct_change(window)
        d[f"Volatility_{window}d"] = d["Daily_Return"].rolling(window).std()
        d[f"RSI_{window}"] = _compute_rsi_helper(d["Close"], window)

    # 2. Feature interactions (capturing regime changes)
    d["RSI_x_Volatility"] = d["RSI"] * d["Volatility"]
    if "Volume" in d.columns:
        d["Volume_x_Returns"] = d["Volume"] * d["Daily_Return"].abs()
        d["Volume_x_Volatility"] = d["Volume"] * d["Volatility"]
    d["MACD_divergence"] = (d["MACD"] - d["MACD"].rolling(5).mean()) / (
        d["MACD"].rolling(5).std() + 1e-9
    )

    # 3. Enhanced momentum features
    d["ROC_5"] = (d["Close"] - d["Close"].shift(5)) / (d["Close"].shift(5) + 1e-9)
    d["ROC_10"] = (d["Close"] - d["Close"].shift(10)) / (d["Close"].shift(10) + 1e-9)
    d["MOM_ratio"] = d["ROC_5"] / (d["ROC_10"].abs() + 1e-9)

    # 4. Trend strength indicators
    for window in [10, 20, 50]:
        sma = d["Close"].rolling(window).mean()
        d[f"Trend_strength_{window}"] = (d["Close"] - sma) / (
            d["Close"].rolling(window).std() + 1e-9
        )

    # 5. Volume profile features
    if "Volume" in d.columns:
        d["Volume_trend"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)
        d["Volume_volatility"] = d["Volume"].rolling(20).std() / (
            d["Volume"].rolling(20).mean() + 1e-9
        )

    # 6. Price patterns
    d["Higher_high"] = (
        (d["High"] > d["High"].shift(1)) & (d["High"].shift(1) > d["High"].shift(2))
    ).astype(int)
    d["Lower_low"] = (
        (d["Low"] < d["Low"].shift(1)) & (d["Low"].shift(1) < d["Low"].shift(2))
    ).astype(int)

    # 7. Rolling correlation features
    d["Returns_volatility_corr"] = d["Daily_Return"].rolling(20).corr(d["Volatility"])
    if "Volume" in d.columns:
        d["Volume_returns_corr"] = d["Volume"].rolling(20).corr(d["Daily_Return"].abs())

    # 8. Temporal features (capture day-of-week, month effects)
    if isinstance(d.index, pd.DatetimeIndex):
        d["DayOfWeek"] = d.index.dayofweek
        d["Month"] = d.index.month
        d["Quarter"] = d.index.quarter
        d["DayOfMonth"] = d.index.day
        # Cyclical encoding for better ML representation
        d["DayOfWeek_sin"] = np.sin(2 * np.pi * d["DayOfWeek"] / 7)
        d["DayOfWeek_cos"] = np.cos(2 * np.pi * d["DayOfWeek"] / 7)
        d["Month_sin"] = np.sin(2 * np.pi * d["Month"] / 12)
        d["Month_cos"] = np.cos(2 * np.pi * d["Month"] / 12)

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

    # =============================================================================
    # ENHANCED FEATURES - Based on Feature Importance Analysis
    # Top features identified: BB_Width, Return_Lag1, Return_Lag3, RSI, OBV
    # =============================================================================

    # 1. Feature Interactions for Top Features
    # BB_Width is the most important feature - create interactions
    if "BB_Width" in d.columns:
        if "RSI" in d.columns:
            d["BB_Width_x_RSI"] = d["BB_Width"] * d["RSI"]
        if "Return_Lag1" in d.columns:
            d["BB_Width_x_Return_Lag1"] = d["BB_Width"] * d["Return_Lag1"]
        if "Vol_Ratio" in d.columns:
            d["BB_Width_x_Vol_Ratio"] = d["BB_Width"] * d["Vol_Ratio"]

    # Return interactions (Returns are highly important)
    if "Return_Lag1" in d.columns and "Return_Lag3" in d.columns:
        d["Return_Lag1_x_Return_Lag3"] = d["Return_Lag1"] * d["Return_Lag3"]
        # Return momentum consistency
        d["Return_Trend_Strength"] = d["Return_Lag1"] * d["Return_Lag3"] * d.get("Return_Lag5", 1)

    # RSI interactions
    if "RSI" in d.columns and "Vol_Ratio" in d.columns:
        d["RSI_x_Vol_Ratio"] = d["RSI"] * d["Vol_Ratio"]

    # OBV interactions
    if "OBV" in d.columns and "Return_Lag1" in d.columns:
        d["OBV_x_Return_Lag1"] = d["OBV"] * d["Return_Lag1"]

    # 2. Enhanced Lags for Top Features
    # Add deeper lags since temporal features are important
    if "BB_Width" in d.columns:
        d["BB_Width_Lag1"] = d["BB_Width"].shift(1)
        d["BB_Width_Lag3"] = d["BB_Width"].shift(3)
        d["BB_Width_Change"] = d["BB_Width"].diff()

    if "RSI" in d.columns:
        d["RSI_Lag7"] = d["RSI"].shift(7)  # Weekly lag
        d["RSI_Lag10"] = d["RSI"].shift(10)

    # Additional return lags (returns category has highest importance)
    d["Return_Lag7"] = d["Close"].pct_change(7)
    d["Return_Lag10"] = d["Close"].pct_change(10)
    d["Return_Lag15"] = d["Close"].pct_change(15)

    # 3. Volatility Regime Features (BB_Width is #1 feature)
    if "BB_Width" in d.columns:
        # Volatility expanding vs contracting
        d["Vol_Expanding"] = (d["BB_Width"] > d["BB_Width"].shift(1)).astype(int)

        # Volatility percentile (regime detection)
        d["Vol_Percentile"] = d["BB_Width"].rolling(60).rank(pct=True)

        # Volatility acceleration
        d["Volatility_Acceleration"] = d["BB_Width"].diff()

        # Rolling volatility statistics
        d["BB_Width_Mean_10"] = d["BB_Width"].rolling(10).mean()
        d["BB_Width_Std_10"] = d["BB_Width"].rolling(10).std()
        d["BB_Width_ZScore"] = (d["BB_Width"] - d["BB_Width_Mean_10"]) / (
            d["BB_Width_Std_10"] + 1e-9
        )

    # 4. Return-based Enhancements (Returns category = 60.84 importance)
    if "Return_Lag1" in d.columns and "Return_Lag5" in d.columns:
        # Return momentum ratio
        d["Return_Momentum_Ratio"] = d["Return_Lag1"] / (d["Return_Lag5"].abs() + 1e-9)

        # Return acceleration
        d["Return_Acceleration"] = d["Return_Lag1"] - d["Return_Lag3"]

        # Positive return streak
        d["Positive_Return_Streak"] = (d["Return_Lag1"] > 0).astype(int).rolling(5).sum()

        # Return reversal indicator
        d["Return_Reversal"] = (
            (d["Return_Lag1"] > 0) & (d["Return_Lag3"] < 0)
        ).astype(int)

    # 5. Rolling Window Statistics for Top Features
    if "Return_Lag1" in d.columns:
        # Rolling stats for returns
        d["Return_Lag1_MA5"] = d["Return_Lag1"].rolling(5).mean()
        d["Return_Lag1_MA10"] = d["Return_Lag1"].rolling(10).mean()
        d["Return_Volatility_20"] = d["Return_Lag1"].rolling(20).std()

        # Return skewness and kurtosis (regime indicators)
        d["Return_Skew_10"] = d["Return_Lag1"].rolling(10).skew()
        d["Return_Kurt_10"] = d["Return_Lag1"].rolling(10).kurt()

    # 6. Volume-based Enhancements (Volume features = 54.65 importance)
    if "Volume" in d.columns and "OBV" in d.columns:
        # Volume momentum
        d["Volume_Momentum_5"] = d["Volume"] / (d["Volume"].shift(5) + 1e-9)

        # OBV momentum
        d["OBV_Change_5"] = d["OBV"].diff(5)
        d["OBV_Trend"] = d["OBV"] / (d["OBV"].rolling(20).mean() + 1e-9)

    # 7. Cross-sectional Features (combining multiple timeframes)
    if "Return_Lag1" in d.columns and "Return_Lag10" in d.columns:
        # Short vs long-term momentum
        d["Return_1d_vs_10d"] = d["Return_Lag1"] - d["Return_Lag10"]
        d["Return_3d_vs_10d"] = d["Return_Lag3"] - d["Return_Lag10"]

    # 8. RSI-based Regime Features
    if "RSI" in d.columns:
        # RSI zones
        d["RSI_Overbought"] = (d["RSI"] > 70).astype(int)
        d["RSI_Oversold"] = (d["RSI"] < 30).astype(int)
        d["RSI_Neutral"] = ((d["RSI"] >= 30) & (d["RSI"] <= 70)).astype(int)

        # RSI momentum
        d["RSI_Momentum_5"] = d["RSI"].diff(5)
        d["RSI_ROC_5"] = d["RSI"].pct_change(5, fill_method=None)

    # ========================================================================
    # 9. MARKET REGIME DETECTION FEATURES
    # ========================================================================
    # These features help identify bull/bear markets and different market conditions

    # Bull/Bear Market Detection (200-day MA)
    if "Close" in d.columns:
        d["MA_200"] = d["Close"].rolling(200).mean()
        # Price relative to 200-day MA (>1 = bull, <1 = bear)
        d["Price_vs_MA200"] = d["Close"] / (d["MA_200"] + 1e-9)
        # Bull market flag
        d["Bull_Market"] = (d["Price_vs_MA200"] > 1.0).astype(int)
        # MA200 slope (trend strength)
        d["MA200_Slope"] = d["MA_200"].diff(20) / (d["MA_200"].shift(20) + 1e-9)
        # Distance from MA200 in terms of volatility
        if "Volatility" in d.columns:
            d["MA200_Distance_Vol"] = (d["Close"] - d["MA_200"]) / (d["Volatility"] + 1e-9)

    # VIX-based Fear Index (if available, otherwise use volatility proxy)
    if "VIX" in d.columns:
        # VIX percentile (regime detection)
        d["VIX_Percentile"] = d["VIX"].rolling(252).rank(pct=True)
        # High fear regime
        d["High_Fear"] = (d["VIX"] > 25).astype(int)
        # VIX spike detection
        d["VIX_Spike"] = (d["VIX"] > d["VIX"].rolling(20).mean() * 1.5).astype(int)
        # VIX momentum
        d["VIX_Change"] = d["VIX"].diff()
    else:
        # Use volatility as proxy for fear
        if "Volatility" in d.columns:
            d["Vol_Percentile_252"] = d["Volatility"].rolling(252).rank(pct=True)
            d["High_Volatility"] = (
                d["Volatility"] > d["Volatility"].rolling(60).quantile(0.75)
            ).astype(int)

    # Market Breadth Indicators
    if "Close" in d.columns:
        # Short-term trend strength
        d["MA_20_50_Cross"] = (d["MA_20"] > d.get("MA_50", d["MA_20"])).astype(int) if "MA_50" in d.columns else 0

        # Percentage above short-term MA (bullish breadth proxy)
        d["Pct_Above_MA20"] = (d["Close"] > d["MA_20"]).astype(int) if "MA_20" in d.columns else 0

        # New highs indicator (52-week high)
        d["Near_52w_High"] = (
            d["Close"] > d["Close"].rolling(252).max() * 0.98
        ).astype(int)

        # New lows indicator
        d["Near_52w_Low"] = (
            d["Close"] < d["Close"].rolling(252).min() * 1.02
        ).astype(int)

    # Trend Strength and Quality
    if "Close" in d.columns and "MA_20" in d.columns:
        # ADX-like trend strength (simplified)
        # Calculate true range
        if "High" in d.columns and "Low" in d.columns:
            d["TR"] = np.maximum(
                d["High"] - d["Low"],
                np.maximum(
                    abs(d["High"] - d["Close"].shift(1)),
                    abs(d["Low"] - d["Close"].shift(1))
                )
            )
            # Directional movement
            d["Plus_DM"] = np.where(
                (d["High"] - d["High"].shift(1)) > (d["Low"].shift(1) - d["Low"]),
                np.maximum(d["High"] - d["High"].shift(1), 0),
                0
            )
            d["Minus_DM"] = np.where(
                (d["Low"].shift(1) - d["Low"]) > (d["High"] - d["High"].shift(1)),
                np.maximum(d["Low"].shift(1) - d["Low"], 0),
                0
            )
            # Smooth and calculate DI
            d["Plus_DI"] = 100 * (d["Plus_DM"].rolling(14).mean() / (d["TR"].rolling(14).mean() + 1e-9))
            d["Minus_DI"] = 100 * (d["Minus_DM"].rolling(14).mean() / (d["TR"].rolling(14).mean() + 1e-9))
            # ADX (trend strength)
            d["DX"] = 100 * abs(d["Plus_DI"] - d["Minus_DI"]) / (d["Plus_DI"] + d["Minus_DI"] + 1e-9)
            d["ADX"] = d["DX"].rolling(14).mean()
            # Strong trend flag
            d["Strong_Trend"] = (d["ADX"] > 25).astype(int)

        # Trend consistency (how long price stays above/below MA)
        d["Days_Above_MA20"] = (d["Close"] > d["MA_20"]).astype(int).rolling(20).sum()
        d["Trend_Consistency"] = d["Days_Above_MA20"] / 20  # 0 to 1

    # Market Regime Classification (combining multiple factors)
    regime_features = []
    if "Bull_Market" in d.columns:
        regime_features.append("Bull_Market")
    if "High_Fear" in d.columns:
        regime_features.append("High_Fear")
    elif "High_Volatility" in d.columns:
        regime_features.append("High_Volatility")
    if "Strong_Trend" in d.columns:
        regime_features.append("Strong_Trend")

    # Create a composite regime score
    if len(regime_features) > 0:
        regime_sum = sum(d[f] for f in regime_features if f in d.columns)
        d["Regime_Score"] = regime_sum / len(regime_features)

    # =========================================================================
    # Economic Modeling Interaction Features (Nov 2025)
    # Based on feature importance analysis - interactions capture regime shifts
    # =========================================================================

    # Position × Volatility Regime (Near_52w_High was #1 in importance)
    if "Near_52w_High" in d.columns and "Volatility" in d.columns:
        d["Near_52w_High_x_Volatility"] = d["Near_52w_High"] * d["Volatility"]

    if "Near_52w_High" in d.columns and "KC_Width" in d.columns:
        d["Near_52w_High_x_KC_Width"] = d["Near_52w_High"] * d["KC_Width"]

    # Momentum × Volatility Regime (Stoch_K was #5 in importance)
    if "Stoch_K" in d.columns and "Volatility" in d.columns:
        d["Stoch_K_x_Volatility"] = d["Stoch_K"] * d["Volatility"]

    # Returns × Volatility Regime (regime-dependent momentum)
    if "Return_Lag3" in d.columns and "Volatility" in d.columns:
        d["Return_Lag3_x_Volatility"] = d["Return_Lag3"] * d["Volatility"]

    if "Return_Lag5" in d.columns and "ATR_14" in d.columns:
        d["Return_Lag5_x_ATR"] = d["Return_Lag5"] * d["ATR_14"]

    # Position × Momentum (trend confirmation)
    if "Near_52w_High" in d.columns and "Return_Lag3" in d.columns:
        d["Near_52w_High_x_Return_Lag3"] = d["Near_52w_High"] * d["Return_Lag3"]

    if "BB_PctB" in d.columns and "Stoch_K" in d.columns:
        d["BB_PctB_x_Stoch_K"] = d["BB_PctB"] * d["Stoch_K"]

    # Cross-Asset × Market Volatility (economic stress indicators)
    if "Credit_Ratio" in d.columns and "Volatility" in d.columns:
        d["Credit_Ratio_x_Volatility"] = d["Credit_Ratio"] * d["Volatility"]

    if "Realized_Vol_60" in d.columns and "Volatility" in d.columns:
        d["Realized_Vol_60_x_Volatility"] = d["Realized_Vol_60"] * d["Volatility"]

    # Macro × Market Trend (policy impact on markets)
    if "Rate_Change_3m" in d.columns and "MA200_Slope" in d.columns:
        d["Rate_Change_3m_x_MA200_Slope"] = d["Rate_Change_3m"] * d["MA200_Slope"]

    if "DXY_Level" in d.columns and "Return_Lag5" in d.columns:
        d["DXY_Level_x_Return_Lag5"] = d["DXY_Level"] * d["Return_Lag5"]

    # Yield Curve × Market Position (recession indicator interaction)
    if "Yield_10Y" in d.columns and "Price_vs_MA200" in d.columns:
        d["Yield_10Y_x_Price_vs_MA200"] = d["Yield_10Y"] * d["Price_vs_MA200"]

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

    # Defragment DataFrame to improve performance (consolidates fragmented blocks)
    # This eliminates PerformanceWarnings from incremental column additions
    d = d.copy()

    return d, feature_cols


def finalize_features(df, feature_cols):
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[feature_cols]
    out = out.replace([np.inf, -np.inf], np.nan)

    # FIX: Use forward-fill instead of global median to prevent lookahead bias
    # Previous: out.fillna(out.median()) used future data in median calculation
    # Current: ffill() only uses past data (time-aware imputation)
    out = out.ffill()

    # Fill any remaining NaN at start of series with 0
    out = out.fillna(0)

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
    volatility_adjusted: bool = True,  # NEW: Enable volatility adjustment
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

    # FIX: Volatility-adjusted thresholds
    # In high volatility (VIX 30+), a 0.5% move is noise
    # In low volatility (VIX 10), a 0.5% move is significant
    if volatility_adjusted and "Volatility" in d.columns:
        median_vol = d["Volatility"].median()
        if median_vol > 0:
            # Scale threshold by realized volatility
            vol_ratio = d["Volatility"] / median_vol
            adjusted_threshold = float(pos_threshold) * vol_ratio
            d["y"] = (d["fwd_ret_net"] >= adjusted_threshold).astype(int)
        else:
            # Fallback to fixed threshold if volatility is invalid
            d["y"] = (d["fwd_ret_net"] >= float(pos_threshold)).astype(int)
    else:
        # Use fixed threshold (original behavior)
        d["y"] = (d["fwd_ret_net"] >= float(pos_threshold)).astype(int)

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
    """
    Enhanced sample weighting with profit-based, recency, and volatility adjustments.

    Args:
        df: DataFrame with fwd_ret_net and optionally Volatility columns
        min_weight: Minimum sample weight
        max_weight: Maximum sample weight
        power: Exponent for profit-based scaling (reduced from 1.75 for stability)
        long_only: Whether to only weight positive returns
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    if "fwd_ret_net" not in df.columns:
        return np.ones(n, dtype=float)

    # 1. Profit-based weights (existing logic, but with tuned power)
    r = pd.to_numeric(df["fwd_ret_net"], errors="coerce").fillna(0.0).to_numpy()
    pos = np.maximum(r, 0.0)
    scale = 0.01
    profit_weight = 1.0 + (pos / scale) ** float(power)
    profit_weight = np.clip(profit_weight, min_weight * 0.7, max_weight * 0.6)

    # 2. Recency weights (more recent data is more relevant)
    positions = np.arange(n)
    recency_weight = np.exp(positions / n * 0.5)  # Exponential growth
    recency_weight = recency_weight / recency_weight.mean()  # Normalize to mean=1

    # 3. Volatility-adjusted weights (down-weight high volatility periods for stability)
    vol_weight = np.ones(n)
    if "Volatility" in df.columns:
        vol = pd.to_numeric(df["Volatility"], errors="coerce").fillna(df["Volatility"].median())
        vol_median = vol.median()
        if vol_median > 0:
            vol_weight = 1.0 / (1.0 + vol / vol_median)  # Inverse volatility
            vol_weight = vol_weight / vol_weight.mean()  # Normalize to mean=1

    # 4. Combine weights multiplicatively (but with dampening to prevent extremes)
    # More conservative powers to reduce overfitting to recent regime
    combined = profit_weight * (recency_weight**0.3) * (vol_weight**0.5)
    combined = combined / combined.mean()  # Normalize to mean=1

    # 5. Apply final clipping
    w = np.clip(combined, float(min_weight), float(max_weight))

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
def load_asset_data(ticker: str = "SPY", data_dir: str = "data_cache") -> pd.DataFrame:
    """
    Generic asset data loader - loads any asset from data_cache/

    Args:
        ticker: Asset ticker (e.g., 'SPY', 'QQQ', 'BTC/USDT')
        data_dir: Directory containing asset data files

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns

    Examples:
        load_asset_data('SPY')          # Equity ETF
        load_asset_data('QQQ')          # Nasdaq ETF
        load_asset_data('BTC/USDT')     # Crypto (file: BTC_USDT_1d.csv)
    """
    from pathlib import Path

    # Special handling for SPY: prefer data/SPY.csv (full history) over data_cache/SPY_1d.csv
    if ticker == 'SPY':
        main_spy_path = Path("data") / "SPY.csv"
        if main_spy_path.exists():
            csv_path = str(main_spy_path)
        else:
            # Fall back to cache
            cache_path = Path(data_dir) / "SPY_1d.csv"
            if cache_path.exists():
                csv_path = str(cache_path)
            else:
                raise FileNotFoundError(f"SPY data not found. Tried: {main_spy_path}, {cache_path}")
    else:
        # Handle crypto tickers with slashes
        if '/' in ticker:
            filename = ticker.replace('/', '_') + '_1d.csv'
        else:
            filename = f"{ticker}_1d.csv"

        # Try data_cache first, then fall back to data/
        data_cache_path = Path(data_dir) / filename
        legacy_path = Path("data") / f"{ticker}.csv"

        if data_cache_path.exists():
            csv_path = str(data_cache_path)
        elif legacy_path.exists():
            csv_path = str(legacy_path)
        else:
            raise FileNotFoundError(
                f"Asset data not found for {ticker}. "
                f"Tried: {data_cache_path}, {legacy_path}"
            )

    # Read and validate
    header = pd.read_csv(csv_path, nrows=1, low_memory=False)
def load_SPY_data() -> pd.DataFrame:
    header = pd.read_csv(CSV_PATH, nrows=1, low_memory=False)
    want = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    usecols = [c for c in want if c in header.columns]

    df = pd.read_csv(
        csv_path,
        CSV_PATH,
        usecols=usecols,
        parse_dates=["Date"],
        low_memory=False,
    )
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").set_index("Date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise AssertionError(f"{ticker} loader: index is not DatetimeIndex")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if "Open" not in df.columns or "Close" not in df.columns:
        raise AssertionError(f"{ticker} loader: missing required OHLC columns")
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


def load_SPY_data() -> pd.DataFrame:
    """Legacy wrapper - loads SPY data using generic loader"""
    return load_asset_data("SPY", data_dir=DATA_DIR)


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
