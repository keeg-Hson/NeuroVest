#!/usr/bin/env python3
"""
Build a single, canonical feature table for NeuroVest (or any SPY-based pipeline).

Outputs (by default under data/features/):
  - neurovest_features_YYYYMMDD.parquet         # canonical training matrix (preferred)
  - neurovest_features_YYYYMMDD.csv             # optional CSV export
  - neurovest_features_YYYYMMDD.features.txt    # exact feature column list used
  - neurovest_features_YYYYMMDD.meta.json       # provenance + config
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- helpers ----------


def _ensure_datetime_index_any(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DatetimeIndex from any common date column name."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        for cand in ["Date", "DATE", "date", "observation_date", "timestamp", "ds"]:
            if cand in df.columns:
                idx = pd.to_datetime(df[cand], errors="coerce")
                break
        else:
            raise ValueError("Dataframe must have a DatetimeIndex or a date-like column.")
    idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.isna()]
    return df


def _sma(x, w):
    return x.rolling(w).mean()


def _ema(x, w):
    return x.ewm(span=w, adjust=False).mean()


def _rsi(series, window=14):
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1 / window, adjust=False).mean() / dn.ewm(
        alpha=1 / window, adjust=False
    ).mean().replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series, fast=12, slow=26, signal=9):
    f = _ema(series, fast)
    s = _ema(series, slow)
    m = f - s
    sig = _ema(m, signal)
    hist = m - sig
    return m, sig, hist


def _true_range(high, low, close):
    """True range using current high/low and previous close."""
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(high, low, close, w=14):
    """Average True Range over a rolling window."""
    return _true_range(high, low, close).ewm(alpha=1 / w, adjust=False).mean()


def _boll(series, w=20, n=2.0):
    ma = _sma(series, w)
    sd = series.rolling(w).std()
    return ma + n * sd, ma, ma - n * sd


def _stoch_kd(high, low, close, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    D = K.rolling(d).mean()
    return K, D


def _vol(series, w=20):
    return series.pct_change(fill_method=None).rolling(w).std()


def _merge_external_if_available(df: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    """Merge any numeric CSVs by as-of date (causal). Auto-detect date column names."""
    if not external_dir.exists():
        return df
    base = df.sort_index()
    for csv in sorted(external_dir.glob("*.csv")):
        try:
            x = pd.read_csv(csv, low_memory=False)
            x = _ensure_datetime_index_any(x)
            num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                continue
            x = x[num_cols].add_prefix(csv.stem + "_").sort_index()
            base = pd.merge_asof(base, x, left_index=True, right_index=True, direction="backward")
        except Exception as e:
            print(f"[warn] Failed to merge {csv.name}: {e}")
    return base


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical features and retain any external numeric signals from df.
    """
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    out = pd.DataFrame(index=df.index)

    # returns (explicitly disable fill to avoid FutureWarning)
    out["ret_1d"] = close.pct_change(1, fill_method=None)
    out["ret_5d"] = close.pct_change(5, fill_method=None)
    out["ret_10d"] = close.pct_change(10, fill_method=None)
    out["logret_1d"] = np.log(close).diff(1)

    # trend
    out["sma_20"] = _sma(close, 20)
    out["sma_50"] = _sma(close, 50)
    out["ema_12"] = _ema(close, 12)
    out["ema_26"] = _ema(close, 26)

    # momentum
    out["rsi_14"] = _rsi(close, 14)
    macd, sig, hist = _macd(close)
    out["macd"] = macd
    out["macd_signal"] = sig
    out["macd_hist"] = hist

    # bands
    up, mid, lo = _boll(close, 20, 2.0)
    out["bb_up_20_2"] = up
    out["bb_mid_20_2"] = mid
    out["bb_lo_20_2"] = lo
    out["bb_width_20_2"] = (up - lo) / mid

    # volatility / range
    out["atr_14"] = _atr(high, low, close, 14)
    out["vol_20"] = _vol(close, 20)

    # stochastic
    k, d = _stoch_kd(high, low, close, 14, 3)
    out["stoch_k_14_3"] = k
    out["stoch_d_14_3"] = d

    # volume
    out["vol_sma_20"] = _sma(vol, 20)
    out["vol_pct_20"] = vol / out["vol_sma_20"] - 1.0

    # price vs MAs
    out["px_over_sma20"] = close / out["sma_20"] - 1.0
    out["px_over_sma50"] = close / out["sma_50"] - 1.0

    # carry through external numeric signals (non-OHLCV, non-technical) as features
    ohlcv_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    existing_cols = set(df.columns)
    extra_cols = [
        c for c in existing_cols if c not in ohlcv_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if extra_cols:
        out = pd.concat([out, df[list(extra_cols)]], axis=1)

    return out


def make_label_forward_up(close: pd.Series, horizon: int, min_return: float = 0.0) -> pd.Series:
    fwd_ret = close.shift(-horizon) / close - 1.0
    y = (fwd_ret >= min_return).astype("Int64")
    y.iloc[-horizon:] = pd.NA  # drop unlabeled tail
    return y


def make_label_spike_normal_crash(
    close: pd.Series,
    horizon: int,
    spike_return: float = 0.005,
    crash_return: float = -0.005,
) -> pd.Series:
    """
    3-class label with unified convention:
      0 = SPIKE (forward return >= spike_return)
      1 = NORMAL (between thresholds)
      2 = CRASH (forward return <= crash_return)
    """
    fwd_ret = close.shift(-horizon) / close - 1.0
    y = pd.Series(index=close.index, dtype="Int64")
    y[fwd_ret >= spike_return] = 0
    y[(fwd_ret < spike_return) & (fwd_ret > crash_return)] = 1
    y[fwd_ret <= crash_return] = 2
    y.iloc[-horizon:] = pd.NA
    return y


def assign_time_splits(idx: pd.DatetimeIndex, train_ratio=0.7, val_ratio=0.15):
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    s = pd.Series(index=idx, dtype="string")
    s.iloc[:n_train] = "train"
    s.iloc[n_train : n_train + n_val] = "val"
    s.iloc[n_train + n_val :] = "test"
    return s


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price_csv", default="data/SPY.csv")
    ap.add_argument("--external_dir", default="data_cache")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min_return", type=float, default=0.0)
    ap.add_argument("--spike_return", type=float, default=0.005)
    ap.add_argument("--crash_return", type=float, default=-0.005)
    ap.add_argument("--out_dir", default="data/features")
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--ticker", default="SPY")
    args = ap.parse_args()

    price_path = Path(args.price_csv)
    ext_dir = Path(args.external_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read only OHLCV; avoid dtype warning
    wanted = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    price_df = pd.read_csv(price_path, usecols=lambda c: c in wanted, low_memory=False)
    price_df = _ensure_datetime_index_any(price_df)

    keep = [
        c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in price_df.columns
    ]
    price_df = price_df[keep]

    # Merge external (auto-detect date columns and drop NaT)
    base = price_df.copy()
    base = _merge_external_if_available(base, ext_dir)

    # Features: technicals + external numeric signals
    feats = build_features(pd.concat([base, price_df[["Open", "High", "Low", "Close"]]], axis=1))

    # Label series
    close = price_df["Adj Close"] if "Adj Close" in price_df.columns else price_df["Close"]
    y_up = make_label_forward_up(close, args.horizon, args.min_return)
    y_3class = make_label_spike_normal_crash(
        close,
        args.horizon,
        spike_return=args.spike_return,
        crash_return=args.crash_return,
    )

    # Assemble
    out = pd.DataFrame(index=feats.index)
    out["date"] = out.index
    out["ticker"] = args.ticker
    for c in keep:
        out[c.lower().replace(" ", "_")] = price_df[c]
    out = out.join(feats)

    out["y_up_fwd"] = y_up
    out["y_class_3"] = y_3class
    out["split"] = assign_time_splits(out.index)

    # Final clean: drop any rows with NaN features or unlabeled tail
    meta_cols = ["date", "ticker", "y_up_fwd", "y_class_3", "split"]
    feature_cols = [c for c in out.columns if c not in meta_cols]
    mask = out[feature_cols].notna().all(axis=1) & out["y_up_fwd"].notna()
    out = out.loc[mask].copy()

    # Stable column order
    out = out[["date", "ticker"] + feature_cols + ["y_up_fwd", "y_class_3", "split"]]

    # Write
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    base_name = f"neurovest_features_{stamp}"
    pq_path = out_dir / f"{base_name}.parquet"
    csv_path = out_dir / f"{base_name}.csv"
    feat_path = out_dir / f"{base_name}.features.txt"
    meta_path = out_dir / f"{base_name}.meta.json"

    wrote_parquet = False
    try:
        out.to_parquet(pq_path, index=False)
        wrote_parquet = True
    except Exception as e:
        print(f"[warn] Parquet write failed ({e}); writing CSV only.")

    if not args.no_csv:
        out.to_csv(csv_path, index=False)

    with open(feat_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    meta = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "price_csv": str(price_path),
        "external_dir": str(ext_dir),
        "horizon_days": args.horizon,
        "min_return": args.min_return,
        "spike_return": args.spike_return,
        "crash_return": args.crash_return,
        "rows": int(len(out)),
        "features": feature_cols,
        "label_col": "y_up_fwd",
        "label_3class_col": "y_class_3",
        "split_col": "split",
        "ticker": args.ticker,
        "notes": "Causal engineered features + auto-merged external numeric signals. 3-class labels use 0=SPIKE, 1=NORMAL, 2=CRASH.",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote feature table:")
    if wrote_parquet:
        print(" -", pq_path)
    if not args.no_csv:
        print(" -", csv_path)
    print("Artifacts:")
    print(" -", feat_path)
    print(" -", meta_path)


if __name__ == "__main__":
    main()
