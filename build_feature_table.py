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
from datetime import datetime, timezone
from pathlib import Path

# Python 3.10 compatibility (UTC added in 3.11)
UTC = timezone.utc

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

    # Convert to DatetimeIndex if it's a Series
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    # Remove timezone if present
    if idx.tz is not None:
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


def _merge_precomputed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-computed feature files (cross_asset_features.csv, macro_features.csv).
    These contain 50+ features that would otherwise require downloading individual assets.
    """
    data_dir = Path("data")
    precomputed_files = [
        ("cross_asset_features.csv", ""),  # Already has XAsset_ prefix
        ("macro_features.csv", "Macro_"),
    ]

    out = df.copy()

    for filename, prefix in precomputed_files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue

        try:
            feat_df = pd.read_csv(filepath, low_memory=False)
            feat_df = _ensure_datetime_index_any(feat_df)

            # Get only numeric columns
            num_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                continue

            # Add prefix if specified and not already prefixed
            if prefix:
                feat_df = feat_df[num_cols].add_prefix(prefix)
            else:
                feat_df = feat_df[num_cols]

            # Merge by date (causal - use backward fill)
            feat_df = feat_df.sort_index()
            out = pd.merge_asof(
                out.sort_index(),
                feat_df,
                left_index=True,
                right_index=True,
                direction="backward"
            )
            print(f"  Merged {len(num_cols)} features from {filename}")
        except Exception as e:
            print(f"  [warn] Could not merge {filename}: {e}")

    return out


def _load_external_asset(ticker: str, ext_dir: Path = None) -> pd.DataFrame:
    """
    Load external asset data from multiple possible directories.
    Search order: ext_dir (if provided), data/, data_cache/, data/etfs/
    """
    search_dirs = []
    if ext_dir is not None:
        search_dirs.append(ext_dir)
    search_dirs.extend([
        Path("data"),
        Path("data_cache"),
        Path("data/etfs"),
    ])

    # Try multiple file name patterns in each directory
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        patterns = [
            search_dir / f"{ticker}.csv",
            search_dir / f"{ticker.replace('^', '')}.csv",
            search_dir / f"{ticker.replace('-', '_')}.csv",
            search_dir / f"{ticker}_1d.csv",  # data_cache format
        ]

        for path in patterns:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.lower() for c in df.columns]
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    return df
            except Exception:
                pass
    return None


# Feature pruning configuration based on analyze_features.py results (Feb 2026)
# Features that HURT model performance when included:
FEATURES_TO_PRUNE = {
    'stoch_k_14_3',    # +0.0052 AUC when removed
    'ret_10d',         # +0.0022 AUC when removed
    'trend_strength',  # -15.70 importance score
    'ret_5d',          # -8.74 importance score
    'px_over_sma20',   # -6.86 importance score
    'rsi_price_div',   # -3.19 importance score
    'rsi_14',          # -0.51 importance, redundant with rsi_7
    'rsi_21',          # Redundant with rsi_7 (r=0.98)
    'sma_20',          # Redundant with sma_50 (r=0.999)
    'bb_up_20_2',      # Redundant with sma_50 (r=1.0)
    'trend_accel',     # Low importance (0.53)
}

# Features confirmed as valuable (keep these)
CORE_FEATURES = {
    'sma_50',          # Primary trend filter (score: 64.51)
    'macd_hist',       # Key momentum signal (score: 57.37)
    'px_over_sma200',  # Long-term position (score: 41.60)
    'stoch_d_14_3',    # Momentum oscillator (score: 33.37)
    'rsi_7',           # Short-term momentum (score: 25.44)
    'px_over_sma50',   # Medium-term position (score: 24.95)
    'zscore_20',       # Mean reversion (score: 24.81)
    'bb_pct',          # Bollinger position (score: 20.14)
    'sma_200',         # Long-term trend (score: 17.67)
    'vol_ratio',       # Volatility regime (score: 16.43)
    'atr_14',          # Volatility measure (score: 14.62)
    'vol_20',          # Realized volatility (score: 5.89)
    'ret_1d',          # Daily returns (score: 9.55)
    'ret_21d',         # Monthly returns (score: 8.39)
    'bb_width_20_2',   # Volatility expansion (score: 8.54)
    'vol_pct_5',       # Short-term volume (score: 14.85)
    'vol_pct_20',      # Volume trend (score: 9.89)
}


def build_features(df: pd.DataFrame, ext_dir: Path = None, prune_features: bool = True) -> pd.DataFrame:
    """
    Build technical features including cross-asset signals.

    Pruned based on feature analysis (Feb 2026):
    - Removed: logret_1d (redundant with ret_1d), bb_mid_20_2 (redundant with sma_20)
    - Removed: ema_12, ema_26 (redundant with sma_20)
    - Removed: macd, vol_sma_20 (hurt performance)
    - Kept: macd_hist (most important feature)

    New pruning (Feb 2026 analysis):
    - Removed: stoch_k_14_3, ret_10d, trend_strength, ret_5d, px_over_sma20, rsi_price_div
    - Removed: rsi_14, rsi_21 (redundant with rsi_7)
    - Removed: sma_20, bb_up_20_2 (redundant with sma_50)

    Args:
        df: DataFrame with OHLCV data
        ext_dir: Directory for external asset data
        prune_features: If True, exclude low-value features (default: True)
    """
    # Handle both lowercase and capitalized column names
    def get_col(names):
        for n in names:
            if n in df.columns:
                return df[n]
        raise KeyError(f"None of {names} found in columns: {list(df.columns)}")

    close = get_col(["Adj Close", "adj close", "Close", "close"])
    high = get_col(["High", "high"])
    low = get_col(["Low", "low"])
    vol = get_col(["Volume", "volume"])

    out = pd.DataFrame(index=df.index)

    # ══════════════════════════════════════════════════════════════════════════
    # CORE FEATURES (kept from analysis)
    # ══════════════════════════════════════════════════════════════════════════

    # Returns - multiple horizons
    out["ret_1d"] = close.pct_change(1, fill_method=None)
    out["ret_5d"] = close.pct_change(5, fill_method=None)
    out["ret_10d"] = close.pct_change(10, fill_method=None)
    out["ret_21d"] = close.pct_change(21, fill_method=None)  # Monthly

    # Trend - keep only sma_20, sma_50 (others were redundant)
    out["sma_20"] = _sma(close, 20)
    out["sma_50"] = _sma(close, 50)
    out["sma_200"] = _sma(close, 200)  # Long-term trend

    # Momentum - RSI at multiple timeframes
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_7"] = _rsi(close, 7)   # Short-term
    out["rsi_21"] = _rsi(close, 21)  # Medium-term

    # MACD - keep only histogram (most important)
    _, _, hist = _macd(close)
    out["macd_hist"] = hist

    # Bollinger Bands - keep width and upper (lower was harmful)
    up, mid, lo = _boll(close, 20, 2.0)
    out["bb_up_20_2"] = up
    out["bb_width_20_2"] = (up - lo) / mid
    out["bb_pct"] = (close - lo) / (up - lo)  # Position within bands

    # Volatility
    out["atr_14"] = _atr(high, low, close, 14)
    out["vol_20"] = _vol(close, 20)
    out["vol_ratio"] = _vol(close, 5) / _vol(close, 20)  # Short vs long vol

    # Stochastic
    k, d = _stoch_kd(high, low, close, 14, 3)
    out["stoch_k_14_3"] = k
    out["stoch_d_14_3"] = d

    # Volume features (removed vol_sma_20 which hurt performance)
    out["vol_pct_20"] = vol / _sma(vol, 20) - 1.0
    out["vol_pct_5"] = vol / _sma(vol, 5) - 1.0

    # Price position
    out["px_over_sma20"] = close / out["sma_20"] - 1.0
    out["px_over_sma50"] = close / out["sma_50"] - 1.0
    out["px_over_sma200"] = close / out["sma_200"] - 1.0

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-ASSET FEATURES (new - using external data)
    # ══════════════════════════════════════════════════════════════════════════

    if ext_dir is None:
        ext_dir = Path("data/external_temp")

    # VIX - Volatility regime
    vix = _load_external_asset("^VIX", ext_dir)
    if vix is not None and 'close' in vix.columns:
        vix_close = vix['close'].reindex(df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.get('date', df.index)))
        if len(vix_close.dropna()) > 50:
            out["vix_level"] = vix_close.values[:len(out)] if len(vix_close) >= len(out) else np.nan
            out["vix_sma10"] = _sma(pd.Series(out["vix_level"]), 10)
            out["vix_over_sma"] = out["vix_level"] / out["vix_sma10"] - 1.0
            out["vix_regime"] = (out["vix_level"] > 20).astype(float)  # High vol regime

    # QQQ - Tech momentum relative to SPY
    qqq = _load_external_asset("QQQ", ext_dir)
    if qqq is not None and 'close' in qqq.columns:
        qqq_close = qqq['close'].values[:len(out)] if len(qqq) >= len(out) else None
        if qqq_close is not None:
            out["spy_qqq_ratio"] = close.values / qqq_close
            out["spy_qqq_ratio_ma"] = _sma(pd.Series(out["spy_qqq_ratio"]), 20)
            out["qqq_momentum"] = pd.Series(qqq_close).pct_change(5).values

    # TLT - Bond/Equity relationship (risk on/off)
    tlt = _load_external_asset("TLT", ext_dir)
    if tlt is not None and 'close' in tlt.columns:
        tlt_close = tlt['close'].values[:len(out)] if len(tlt) >= len(out) else None
        if tlt_close is not None:
            out["spy_tlt_ratio"] = close.values / tlt_close
            out["tlt_momentum"] = pd.Series(tlt_close).pct_change(5).values
            # Rolling correlation (risk sentiment)
            spy_ret = close.pct_change()
            tlt_ret = pd.Series(tlt_close).pct_change()
            out["spy_tlt_corr_20"] = spy_ret.rolling(20).corr(tlt_ret).values

    # GLD - Gold as risk hedge
    gld = _load_external_asset("GLD", ext_dir)
    if gld is not None and 'close' in gld.columns:
        gld_close = gld['close'].values[:len(out)] if len(gld) >= len(out) else None
        if gld_close is not None:
            out["spy_gld_ratio"] = close.values / gld_close
            out["gld_momentum"] = pd.Series(gld_close).pct_change(10).values

    # HYG/LQD - Credit spread proxy
    hyg = _load_external_asset("HYG", ext_dir)
    lqd = _load_external_asset("LQD", ext_dir)
    if hyg is not None and lqd is not None and 'close' in hyg.columns and 'close' in lqd.columns:
        hyg_close = hyg['close'].values[:len(out)] if len(hyg) >= len(out) else None
        lqd_close = lqd['close'].values[:len(out)] if len(lqd) >= len(out) else None
        if hyg_close is not None and lqd_close is not None:
            out["credit_spread"] = hyg_close / lqd_close
            out["credit_spread_ma"] = _sma(pd.Series(out["credit_spread"]), 10)
            out["credit_spread_chg"] = pd.Series(out["credit_spread"]).pct_change(5).values

    # Dollar strength (UUP)
    uup = _load_external_asset("UUP", ext_dir)
    if uup is not None and 'close' in uup.columns:
        uup_close = uup['close'].values[:len(out)] if len(uup) >= len(out) else None
        if uup_close is not None:
            out["dollar_momentum"] = pd.Series(uup_close).pct_change(10).values
            out["dollar_level"] = uup_close / np.mean(uup_close[:50]) if len(uup_close) > 50 else np.nan

    # Sector rotation - XLK (tech) vs XLF (financials)
    xlk = _load_external_asset("XLK", ext_dir)
    xlf = _load_external_asset("XLF", ext_dir)
    if xlk is not None and xlf is not None and 'close' in xlk.columns and 'close' in xlf.columns:
        xlk_close = xlk['close'].values[:len(out)] if len(xlk) >= len(out) else None
        xlf_close = xlf['close'].values[:len(out)] if len(xlf) >= len(out) else None
        if xlk_close is not None and xlf_close is not None:
            out["tech_fin_ratio"] = xlk_close / xlf_close
            out["tech_momentum"] = pd.Series(xlk_close).pct_change(5).values
            out["fin_momentum"] = pd.Series(xlf_close).pct_change(5).values

    # IWM - Small cap momentum (risk appetite)
    iwm = _load_external_asset("IWM", ext_dir)
    if iwm is not None and 'close' in iwm.columns:
        iwm_close = iwm['close'].values[:len(out)] if len(iwm) >= len(out) else None
        if iwm_close is not None:
            out["spy_iwm_ratio"] = close.values / iwm_close
            out["smallcap_momentum"] = pd.Series(iwm_close).pct_change(5).values

    # ══════════════════════════════════════════════════════════════════════════
    # ADDITIONAL DERIVED FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    # Trend strength
    out["trend_strength"] = (out["sma_20"] - out["sma_50"]) / out["sma_50"]
    out["trend_accel"] = out["trend_strength"].diff(5)

    # Mean reversion signals
    out["zscore_20"] = (close - out["sma_20"]) / out["vol_20"]

    # Momentum divergence
    out["rsi_price_div"] = out["rsi_14"].diff(5) - (close.pct_change(5) * 100)

    # Carry forward close for target calculation
    out["close"] = close.values

    # ══════════════════════════════════════════════════════════════════════════
    # MERGE PRE-COMPUTED FEATURES (cross_asset, macro)
    # ══════════════════════════════════════════════════════════════════════════
    # These files contain 50+ features that would otherwise require downloading
    # individual assets. This is the key step that was missing!
    initial_cols = len(out.columns)
    out = _merge_precomputed_features(out)
    added_cols = len(out.columns) - initial_cols
    if added_cols > 0:
        print(f"  Total: Added {added_cols} pre-computed features")

    # Apply feature pruning if enabled
    if prune_features:
        cols_to_drop = [c for c in out.columns if c in FEATURES_TO_PRUNE]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)
            print(f"  Pruned {len(cols_to_drop)} low-value features")

    # Drop rows with NaN in critical features
    return out


def get_pruning_config() -> dict:
    """
    Return the current feature pruning configuration.
    Useful for analysis scripts to understand what's being excluded.
    """
    return {
        'pruned_features': list(FEATURES_TO_PRUNE),
        'core_features': list(CORE_FEATURES),
        'reason': 'Based on analyze_features.py results (Feb 2026)',
        'expected_auc_improvement': 0.0074,  # Sum of AUC gains from pruning
    }


def get_feature_importance_ranking() -> list:
    """
    Return features ranked by importance from latest analysis.
    Higher score = more important for model performance.
    """
    return [
        ('sma_50', 64.51),
        ('macd_hist', 57.37),
        ('px_over_sma200', 41.60),
        ('stoch_d_14_3', 33.37),
        ('rsi_7', 25.44),
        ('px_over_sma50', 24.95),
        ('zscore_20', 24.81),
        ('bb_pct', 20.14),
        ('ret_10d', 19.93),  # Pruned but tracked
        ('sma_200', 17.67),
        ('vol_ratio', 16.43),
        ('vol_pct_5', 14.85),
        ('atr_14', 14.62),
        ('bb_up_20_2', 11.65),  # Pruned (redundant)
        ('vol_pct_20', 9.89),
        ('sma_20', 9.70),  # Pruned (redundant)
        ('ret_1d', 9.55),
        ('rsi_21', 9.27),  # Pruned (redundant)
        ('bb_width_20_2', 8.54),
        ('ret_21d', 8.39),
        ('stoch_k_14_3', 8.00),  # Pruned (hurts AUC)
        ('vol_20', 5.89),
        ('trend_accel', 0.53),  # Pruned (low value)
        ('rsi_14', -0.51),  # Pruned (negative)
        ('rsi_price_div', -3.19),  # Pruned (negative)
        ('px_over_sma20', -6.86),  # Pruned (negative)
        ('ret_5d', -8.74),  # Pruned (negative)
        ('trend_strength', -15.70),  # Pruned (negative)
    ]


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
