# backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_SPY_data
from datetime import timedelta, datetime
import subprocess

#auto update SPY data before anything else
subprocess.run(["python3", "update_spy_data.py"])

import os, json
from pathlib import Path

import sys, subprocess, pathlib
subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("update_spy_data.py"))], check=False)


def _ensure_logs_dir(path: str = "logs") -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "NA"

def _to_jsonable(x):
    # make numpy/pandas types JSON-friendly
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if pd.isna(x):
        return None
    return x

def save_run_record(config: dict,
                    metrics: dict,
                    simulate_mode: bool,
                    trades_df: pd.DataFrame | None = None,
                    out_dir: str = "logs") -> str:
    """
    Saves a per-run JSON, appends to a JSONL history, and updates latest.json.
    Returns the path of the per-run JSON.
    """
    _ensure_logs_dir(out_dir)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {
        "timestamp_utc": ts,
        "git_sha": _git_sha(),
        "simulate_mode": bool(simulate_mode),
        "config": {k: _to_jsonable(v) for k, v in (config or {}).items()},
        "metrics": {k: _to_jsonable(v) for k, v in (metrics or {}).items()},
        "n_trades": int(metrics.get("trades", 0)) if metrics else 0,
    }

    if trades_df is not None and not trades_df.empty:
        rec["first_signal"] = str(pd.to_datetime(trades_df.index.min(), errors="coerce"))
        rec["last_exit"]    = str(pd.to_datetime(trades_df["exit_time"]).max())

    stamp = ts.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    run_file = Path(out_dir) / f"run_{stamp}__{rec['git_sha']}.json"
    with open(run_file, "w") as f: json.dump(rec, f, indent=2)
    with open(Path(out_dir) / "run_history.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    with open(Path(out_dir) / "latest.json", "w") as f: json.dump(rec, f, indent=2)

    print(f"📝 Saved run record → {run_file}")
    return str(run_file)

def print_run_summary(metrics: dict, config: dict) -> None:
    safe = {k: _to_jsonable(v) for k, v in (metrics or {}).items()}
    try:
        line = (
            f"trades={safe.get('trades', 0)} | "
            f"total={safe.get('total_return', 0.0):.2%} | "
            f"ann={safe.get('annualized_return', float('nan')):.2%} | "
            f"sharpe={safe.get('sharpe', float('nan')):.2f} | "
            f"maxDD={safe.get('max_drawdown', 0.0):.2%} | "
            f"PF={safe.get('profit_factor', float('inf')):.2f}"
        )
    except Exception:
        line = str(safe)

    key_cfg = {k: config.get(k) for k in [
        "lookahead","tp_atr","sl_atr","allow_overlap","ambig_policy",
        "confidence_thresh","crash_thresh","spike_thresh",
        "fee_bps","slip_bps","atr_len","trend_len",
        "cooldown_long_days","cooldown_short_days",
        "use_opposite_exit","use_conf_size","use_weekly_trend",
        "target_ann_vol","vol_lookback","size_cap","conf_size_bounds" 
    ] if k in config}

    print("\n🧾 Run config:", key_cfg)
    print("🔐 Git SHA: ", _git_sha())
    print("📊 Summary: ", line)


# ─── Trade exit resolution ────────────────────────────────────────────────
def _resolve_bar_exit_long(bar, tp_px, sl_px):
    hi = bar.get("High"); lo = bar.get("Low")
    hit_tp = pd.notna(hi) and hi >= tp_px
    hit_sl = pd.notna(lo) and lo <= sl_px
    if hit_tp and hit_sl:         # don't decide here
        return "AMBIG", None, True
    if hit_tp: return "TP", tp_px, False
    if hit_sl: return "SL", sl_px, False
    return None, None, False

# ─── Trade exit resolution for shorts ─────────────────────────────────────
def _resolve_bar_exit_short(bar, tp_px, sl_px):
    hi = bar.get("High"); lo = bar.get("Low")
    hit_tp = pd.notna(lo) and lo <= tp_px    # profit down
    hit_sl = pd.notna(hi) and hi >= sl_px    # stop up
    if hit_tp and hit_sl:
        return "AMBIG", None, True
    if hit_tp: return "TP", tp_px, False
    if hit_sl: return "SL", sl_px, False
    return None, None, False




def _load_predictions(prefer_full: bool = True) -> pd.DataFrame:
    """
    Load predictions, preferring the full dump if available.
    Returns a dataframe with Timestamp (naive) and a normalized Date column.
    De-duplicates by Date keeping the last row (most recent signal of the day).
    """
    full_path  = "logs/predictions_full.csv"
    daily_path = "logs/daily_predictions.csv"

    path = full_path if (prefer_full and os.path.exists(full_path)) else daily_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions file found at {path}")

    preds = pd.read_csv(path)

    # If we loaded the "full" file but confidences are all zeros, fall back to daily.
    if (path.endswith("predictions_full.csv")
        and os.path.exists("logs/daily_predictions.csv")):
        # probe whether confidences are non-informative (all zeros or NaN)
        def _conf_all_zero(df, cols):
            for c in cols:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    if s.abs().sum() > 0:
                        return False
            return True

        if _conf_all_zero(preds, ["Trade_Conf", "Spike_Conf", "Crash_Conf"]):
            print("ℹ️ predictions_full.csv has zero confidences — falling back to daily_predictions.csv")
            path = "logs/daily_predictions.csv"
            preds = pd.read_csv(path)

    # Timestamp → datetime (naive), then a daily Date key
    if "Timestamp" in preds.columns:
        preds["Timestamp"] = pd.to_datetime(preds["Timestamp"], errors="coerce").dt.tz_localize(None)
    else:
        # fall back to any Date column present
        preds["Timestamp"] = pd.to_datetime(preds.get("Date", pd.NaT), errors="coerce").dt.tz_localize(None)

    preds = preds.dropna(subset=["Timestamp"]).copy()
    preds["Date"] = preds["Timestamp"].dt.normalize()

    # If there are multiple rows per Date, keep the last one (often the most recent append)
    preds = preds.sort_values(["Date", "Timestamp"]).drop_duplicates(subset=["Date"], keep="last")

    # Avoid collisions later
    preds = preds.loc[:, ~preds.columns.duplicated(keep="first")]
    return preds


def _load_best_thresholds(
    csv_path="logs/threshold_search.csv",
    json_path="configs/best_thresholds.json",
    min_trades=10,
    objective_col="score",
):
    # Prefer JSON if present
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                cfg = json.load(f)
            return cfg.get("confidence_thresh"), cfg.get("crash_thresh"), cfg.get("spike_thresh")
        except Exception as e:
            print(f"⚠️ Could not read {json_path}: {e}")

    # Fallback to CSV leaderboard
    try:
        df = pd.read_csv(csv_path)
        if "trades" in df.columns:
            df = df[df["trades"] >= min_trades]
        if df.empty:
            return None, None, None
        row = df.sort_values(objective_col, ascending=False).iloc[0]
        return row.get("confidence_thresh"), row.get("crash_thresh"), row.get("spike_thresh")
    except FileNotFoundError:
        print(f"⚠️ {csv_path} not found — run a threshold sweep first.")
    except Exception as e:
        print(f"⚠️ Could not read {csv_path}: {e}")

    return None, None, None



#sets a reference portfolio size to simulate dollar returns from % returns.





# ─── Trade rules ────────────────────────────────────────────────────────────────
ENTRY_SHIFT   = 1    # enter at next bar’s open
#EXIT_SHIFT    = 1+3    # exit at day+3 close  <-- match label window
POSITION_SIZE = 1.0  # 1x notional per trade
CAPITAL_BASE  = 100_000  #initial capital base for dollar-return tracking

def run_backtest(window_days: int | None = None,
                 crash_thresh: float | None = None,
                 spike_thresh: float | None = None,
                 confidence_thresh: float | None = None,
                 simulate_mode: bool = False,
                 lookahead: int = 3,
                 tp_atr: float = 0.5,
                 sl_atr: float = 0.5,
                 allow_overlap: bool = True,
                 ambig_policy: str = "sl_first",   # 'sl_first' | 'tp_first' | 'skip' | 'close_dir' | 'random'
                 rng_seed: int = 7,
                 fee_bps: float = 0.5,   # 0.005% per fill
                 slip_bps: float = 1.0,  # 0.01% per fill
                 atr_len: int = 14,
                 trend_len: int = 50, 

                 target_ann_vol: float | None = None,  # e.g., 0.12 for 12%
                 vol_lookback: int = 20,
                 size_cap: float = 2.0,

                 conf_size_bounds: tuple[float,float] | None = (0.7, 1.3),

                 trail_trigger: float | None = 0.5,  # trigger at 0.5×TP_ATR move
                 trail_k: float = 0.5,  

                 cooldown_days: int = 0,

                 cooldown_long_days: int = 1,
                 cooldown_short_days: int = 2,

                 use_conf_size: bool = True,
                 use_opposite_exit: bool = True,

                 use_weekly_trend: bool = True,
                 use_atr_band: bool = False,
                 use_regime_filter: bool = True,

                 dyn_t: float | None = None,      
                 margin: float = 0.05 

            


                 ):
    """
    End-to-end backtest:
      - loads predictions (logs/daily_predictions.csv)
      - loads SPY history and normalizes duplicate/multiindex columns
      - optional pre-filter by confidence
      - optional explicit class thresholds (else use model-provided labels)
      - joins by Date and simulates ENTRY_SHIFT/EXIT_SHIFT trade rules
      - returns (trades_df, metrics_dict, simulate_mode)
    """
    # ── 1) Load predictions ─────────────────────────────────────────────────────
    preds = _load_predictions(prefer_full=True)

    # --- Adapt predictions for forward-returns variant ---
    VAR = os.getenv("PREDICT_VARIANT", "crash_spike").strip().lower()
    if VAR == "forward_returns":
        # Ensure expected columns exist for downstream sizing/filters
        if "Crash_Conf" not in preds.columns:
            preds["Crash_Conf"] = 0.0
        # predict.py (forward-returns) writes Trade_Conf; reuse as Spike_Conf for charts/logic
        if "Spike_Conf" not in preds.columns and "Trade_Conf" in preds.columns:
            preds["Spike_Conf"] = preds["Trade_Conf"]

        # Map {0,1} (No-Trade/Trade) -> {0,2} (Hold/Spike=long) to match backtest logic
        if "Prediction" in preds.columns:
            preds["Prediction"] = preds["Prediction"].replace({1: 2})



    # ── 2) Load & normalize SPY ─────────────────────────────────────────────────
    spy_df = load_SPY_data()

    spy_df = spy_df[~spy_df.index.duplicated(keep="last")].sort_index()


    # 2a) Flatten MultiIndex like ('Open','SPY') → 'Open'
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [c[0] if isinstance(c, tuple) else c for c in spy_df.columns]

    # 2b) Rename stringified tuple columns → canonical names (handles .1/.2 too)
    rename_map = {
        "('Open', 'SPY')": "Open",
        "('High', 'SPY')": "High",
        "('Low', 'SPY')": "Low",
        "('Close', 'SPY')": "Close",
        "('Adj Close', 'SPY')": "Adj Close",
        "('Volume', 'SPY')": "Volume",
    }
    new_cols = []
    for col in map(str, spy_df.columns):
        base = col.replace(".1", "").replace(".2", "").replace(".3", "")
        new_cols.append(rename_map.get(base, col))
    spy_df.columns = new_cols

    # 2c) Coalesce duplicates per name by picking the series with FEWEST NaNs
    def _pick_best(series_list):
        nan_counts = [s.isna().sum() for s in series_list]
        return series_list[int(np.argmin(nan_counts))]

    canonical = {}
    for name in set(spy_df.columns):
        idxs = [i for i, c in enumerate(spy_df.columns) if c == name]
        if len(idxs) == 1:
            canonical[name] = spy_df.iloc[:, idxs[0]]
        else:
            series_list = [spy_df.iloc[:, i] for i in idxs]
            canonical[name] = _pick_best(series_list)

    spy_df = pd.DataFrame(canonical, index=spy_df.index)

    # 2d) Keep only OHLCV we need and enforce daily DatetimeIndex
    needed = [c for c in ["Open", "Close", "High", "Low", "Volume"] if c in spy_df.columns]
    spy_df = spy_df[needed].copy()
    if not isinstance(spy_df.index, pd.DatetimeIndex):
        spy_df.index = pd.to_datetime(spy_df.index, errors="coerce")
    spy_df.index = spy_df.index.floor("D")

  

    

    # ── 3) Align predictions to available SPY dates + optional window ──────────
    preds = preds.dropna(subset=["Date"])
    preds = preds[preds["Date"].isin(spy_df.index)].copy()

    if window_days:
        cutoff = preds["Date"].max() - pd.Timedelta(days=window_days)
        preds = preds[preds["Date"] >= cutoff]
        spy_df = spy_df[spy_df.index >= cutoff]

    # ── 4) Remove any stray price columns from preds to avoid join collisions ──
    preds = preds.loc[:, ~preds.columns.duplicated(keep="first")]
    price_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    drop_cols = [c for c in preds.columns if c in price_cols]
    if drop_cols:
        print(f"ℹ️ Dropping price cols from preds to avoid join collision: {drop_cols}")
    preds = preds.drop(columns=drop_cols, errors="ignore")




    # ── 5) Optional pre-filter by probability confidence ───────────────────────
    # if confidence_thresh is not None:
    #     preds = preds[
    #         (preds["Crash_Conf"] >= confidence_thresh) |
    #         (preds["Spike_Conf"] >= confidence_thresh)
    #     ]

    # ── 6) Optional explicit thresholds (else use model labels already in file)
    if (crash_thresh is not None) or (spike_thresh is not None) or (confidence_thresh is not None):
        print(f"📊 Applying explicit thresholds — "
            f"Confidence ≥ {confidence_thresh if confidence_thresh is not None else '—'}, "
            f"Crash ≥ {crash_thresh if crash_thresh is not None else '—'}, "
            f"Spike ≥ {spike_thresh if spike_thresh is not None else '—'}")
        preds["Prediction"] = 0
        # optional overall confidence gate
        if confidence_thresh is not None:
            ok_conf = (preds["Crash_Conf"].fillna(0).clip(0,1).ge(confidence_thresh)) | \
                    (preds["Spike_Conf"].fillna(0).clip(0,1).ge(confidence_thresh))
        else:
            ok_conf = pd.Series(True, index=preds.index)

        # apply class thresholds
        if crash_thresh is not None:
            preds.loc[ok_conf & preds["Crash_Conf"].ge(crash_thresh), "Prediction"] = 1
        if spike_thresh is not None:
            preds.loc[ok_conf & preds["Spike_Conf"].ge(spike_thresh), "Prediction"] = 2
    else:
        print("ℹ️ Using model-provided class labels (no explicit crash/spike thresholds).")



    # ── 7) (Optional) Simulate mode: inject some spikes for plumbing tests ─────
    if simulate_mode:
        print("🧪 Simulate mode ON — injecting fake spike predictions.")
        valid_idx = preds.index  # already aligned to SPY dates
        if len(valid_idx) >= 10:
            inject_points = np.linspace(0, len(valid_idx) - 3, 10, dtype=int)
            for i in inject_points:
                preds.loc[valid_idx[i], "Prediction"] = 2

    # ── 8) Join on Date and drop rows with missing prices ──────────────────────
    df = (
        preds
        .set_index("Date")
        .join(spy_df[["Open", "High", "Low", "Close"]], how="inner")  #08/21: addition of High/Low
        .sort_index()
        .reset_index()
    )


    # Ensure numeric prices and drop any row that can’t be used
    df["Open"]  = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    bad = df[["Open", "Close"]].isna().any(axis=1)
    if bad.any():
        print(f"⚠️ Dropping {int(bad.sum())} rows with NaN/invalid Open/Close after join.")
        df = df[~bad]
    # Also coerce the rest so comparisons don’t silently fail
    for c in ("High", "Low", "Crash_Conf", "Spike_Conf", "Confidence"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Canonical filter: trust model labels; add simple margin winner check ----
    lead = (df["Spike_Conf"] - df["Crash_Conf"]).abs()
    min_margin = float(margin)
    base = len(df)
    df = df[(df["Prediction"].isin([1, 2])) & (lead >= min_margin)].copy()
    print(f"🧪 Margin gate kept {len(df)} rows of {base} (margin ≥ {min_margin:.2f}).")




    # Regime filter: longs only in up-trend, shorts only in down-trend
    if use_regime_filter:
        trend_ma = spy_df["Close"].rolling(trend_len).mean()
        df["Trend_MA"] = trend_ma.reindex(df["Date"]).to_numpy()
        df = df[df["Trend_MA"].notna()]
        df = df[
            ((df["Prediction"] == 2) & (df["Close"] >= df["Trend_MA"])) |
            ((df["Prediction"] == 1) & (df["Close"] <  df["Trend_MA"]))
        ].sort_values("Date").reset_index(drop=True)

        if use_weekly_trend:
            weekly_close = spy_df["Close"].resample("W-FRI").last()
            weekly_trend = weekly_close.rolling(26).mean().reindex(spy_df.index, method="ffill")
            df["WeeklyTrend"] = weekly_trend.reindex(df["Date"]).to_numpy()
            df = df[
                ((df["Prediction"] == 2) & (df["Close"] >= df["WeeklyTrend"])) |
                ((df["Prediction"] == 1) & (df["Close"] <= df["WeeklyTrend"]))
            ].sort_values("Date").reset_index(drop=True)



    before, after = len(preds), len(df)
    print(f"🧹 Filtered rows: kept {after} of {before}.")
    # ---- End single canonical filter ----

    n_trades = int((df["Prediction"].isin([1,2])).sum())
    print(f"✅ After filters: {len(df)} rows kept, {n_trades} trade signals in window.")








    # ── 9) Diagnostics ─────────────────────────────────────────────────────────
    print("\n🔎 Confidence Range Stats:")
    if not preds.empty:
        print("📈 Raw Spike confidence range:", preds["Spike_Conf"].min(), "→", preds["Spike_Conf"].max())
        print("📉 Raw Crash confidence range:", preds["Crash_Conf"].min(), "→", preds["Crash_Conf"].max())
    else:
        print("ℹ️ No prediction rows after filtering.")

    print("\n📊 Joined Prediction Candidates (1 or 2):")
    print(df[df["Prediction"].isin([1, 2])].head())

    print("\n🔍 Prediction Counts:")
    print(df["Prediction"].value_counts())

    print("\n🗓️  Date range in predictions:", preds["Date"].min() if not preds.empty else None,
          "→", preds["Date"].max() if not preds.empty else None)
    print("📅 Date range in SPY data:    ", spy_df.index.min(), "→", spy_df.index.max())

    if not df.empty and all(df["Prediction"] == 0):
        print("⚠️  All predictions are '0' — consider lowering thresholds.")

    # ── 10) Build trades (label-aligned exits, conservative TP/SL) ────────────────

    # --- exit config derived from function args ---
    LOOKAHEAD = max(1, int(lookahead))
    TP_ATR    = float(tp_atr)
    SL_ATR    = float(sl_atr)

    # Wilder-style ATR in DOLLARS with configurable length
    hl = (spy_df["High"] - spy_df["Low"]).abs()
    hc = (spy_df["High"] - spy_df["Close"].shift(1)).abs()
    lc = (spy_df["Low"]  - spy_df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()

    # Ensure df["Date"] is datetime and unique for reindexing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date","Timestamp" if "Timestamp" in df.columns else "Date"])
    # if multiple rows share the same Date, keep the last signal for that day
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # Also make sure ATR's index itself has no duplicates
    atr = atr[~atr.index.duplicated(keep="last")].sort_index()


   

    # align to joined df dates
    df["ATR_dol"] = atr.reindex(df["Date"]).to_numpy()

    # Ensure df["Date"] is datetime and unique for reindexing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date","Timestamp" if "Timestamp" in df.columns else "Date"])
    # if multiple rows share the same Date, keep the last signal for that day
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # Also make sure ATR's index itself has no duplicates
    atr = atr[~atr.index.duplicated(keep="last")].sort_index()

    # Now this reindex will be safe
    df["ATR_dol"] = atr.reindex(df["Date"]).to_numpy()


    df["ATR_dol"] = pd.to_numeric(df["ATR_dol"], errors="coerce").replace([np.inf,-np.inf], np.nan)
    # optional: trim ultra-low ATR tails to avoid microscopic bands
    df["ATR_dol"] = df["ATR_dol"].fillna(np.nanmedian(df["ATR_dol"]))
    df["ATR_dol"] = df["ATR_dol"].clip(lower=np.nanpercentile(df["ATR_dol"], 5))

    # ATR band filter (off by default)
    if use_atr_band:
        p = np.nanpercentile(df["ATR_dol"], [20, 80])
        df = df[((df["ATR_dol"] >= p[0]) & (df["ATR_dol"] <= p[1])) | (df["Prediction"] == 0)]



    # --- realized ann. volatility per date (for vol targeting) ---
    ret_d = spy_df["Close"].pct_change()
    vol = (ret_d.rolling(vol_lookback).std() * np.sqrt(252.0)).reindex(df["Date"]).to_numpy()
    df["ann_vol"] = pd.to_numeric(vol, errors="coerce")

    # --- learned decision threshold for sizing (loaded once) ---
    try:
        with open("models/thresholds.json", "r") as _f:
            _thr = json.load(_f)
        learned_t = float(_thr.get("threshold", 0.5))
    except Exception:
        learned_t = 0.5
    if dyn_t is not None:
        learned_t = float(dyn_t)

    # --- Build opposite-signal map on the full trading calendar (raw preds) ---
    pred_map = (
        preds.set_index("Date")["Prediction"]
            .reindex(spy_df.index)
            .fillna(0)
            .astype(int)
            .to_dict()
    )








    trades_list = []
    ambig_bars = 0
    i = 0
    N = len(df)
    rng = np.random.default_rng(rng_seed)

    while i < N:
        row = df.iloc[i]
        sig = int(row.get("Prediction", 0)) if pd.notna(row.get("Prediction", np.nan)) else 0
        if sig not in (1, 2):
            i += 1
            continue

        # === CALENDAR-BASED ENTRY: next trading day in SPY (not next signal row) ===
        signal_date = pd.to_datetime(row["Date"])
        idx = spy_df.index.searchsorted(signal_date, side="right")
        if idx >= len(spy_df):
            i += 1
            continue
        entry_date = spy_df.index[idx]
        entry_bar  = spy_df.iloc[idx]

        # --- sizing pieces (use SIGNAL-day info) ---
        # (1) confidence sizing (vs learned threshold; clamp to bounds)
        pos_conf = 1.0
        if use_conf_size:
            lo, hi = conf_size_bounds if conf_size_bounds else (0.7, 1.3)
            winner_prob = float(row["Spike_Conf"] if sig == 2 else row["Crash_Conf"])
            if np.isnan(winner_prob):
                pos_conf = 1.0
            else:
                u = max(0.0, min(1.0, (winner_prob - learned_t) / max(1e-6, 1.0 - learned_t)))
                pos_conf = lo + u * (hi - lo)

        # (2) volatility targeting (from signal day’s ann_vol)
        size_vol = 1.0
        if target_ann_vol is not None and pd.notna(row.get("ann_vol", np.nan)) and row["ann_vol"] > 0:
            size_vol = np.clip(target_ann_vol / float(row["ann_vol"]), 0.1, size_cap)

        # (3) optional confidence-bounds map
        size_conf = 1.0
        if conf_size_bounds is not None:
            lo, hi = conf_size_bounds
            if sig == 2:
                conf = float(row.get("Spike_Conf", np.nan))
                thr  = float(spike_thresh if spike_thresh is not None else 0.90)
            else:
                conf = float(row.get("Crash_Conf", np.nan))
                thr  = float(crash_thresh if crash_thresh is not None else 0.60)
            if not np.isnan(conf):
                t = max(0.0, min(1.0, (conf - thr) / max(1e-6, 1.0 - thr)))
                size_conf = lo + t * (hi - lo)

        # Edge-aware (Kelly-lite) from SIGNAL confidences
        sp = float(row.get("Spike_Conf", np.nan))
        cr = float(row.get("Crash_Conf", np.nan))
        if not np.isnan(sp) and not np.isnan(cr):
            edge = sp - cr                          # [-1, 1]
            kelly = np.clip(edge / 0.5, 0.0, 0.5)   # conservative; cap 50%
        else:
            kelly = 0.0

        pos_mult_final = pos_conf * size_vol * size_conf * (1.0 + kelly)

        # --- prices/targets on the actual entry trading day ---
        entry_px = float(entry_bar["Open"])
        _atr_on_entry = atr.reindex([entry_date]).iloc[0] if entry_date in atr.index else np.nan
        atr_dol = float(_atr_on_entry) if pd.notna(_atr_on_entry) else 0.0

        if sig == 2:  # Spike/long
            tp_px = entry_px + TP_ATR * atr_dol
            sl_px = entry_px - SL_ATR * atr_dol
        else:         # Crash/short
            tp_px = entry_px - TP_ATR * atr_dol
            sl_px = entry_px + SL_ATR * atr_dol

        # === calendar look window AFTER entry ===
        end_date  = entry_date + pd.Timedelta(days=LOOKAHEAD)
        look_bars = spy_df.loc[(spy_df.index > entry_date) & (spy_df.index <= end_date)].copy()

        exit_px, exit_ts = None, None

        for bar_date, bar in look_bars.iterrows():
            tag, px = None, None

            # 1) TP/SL on daily highs/lows
            if sig == 2:  # long
                hit_tp = pd.notna(bar.get("High")) and bar["High"] >= tp_px
                hit_sl = pd.notna(bar.get("Low"))  and bar["Low"]  <= sl_px
                if hit_tp and hit_sl:
                    if   ambig_policy == "sl_first": tag, px = "SL", sl_px
                    elif ambig_policy == "tp_first": tag, px = "TP", tp_px
                    elif ambig_policy == "close_dir":
                        upbar = pd.notna(bar.get("Open")) and pd.notna(bar.get("Close")) and bar["Close"] >= bar["Open"]
                        tag, px = ("TP", tp_px) if upbar else ("SL", sl_px)
                    elif ambig_policy == "random":
                        tag, px = (("TP", tp_px) if rng.random() < 0.5 else ("SL", sl_px))
                elif hit_tp: tag, px = "TP", tp_px
                elif hit_sl: tag, px = "SL", sl_px
            else:         # short
                hit_tp = pd.notna(bar.get("Low"))  and bar["Low"]  <= tp_px
                hit_sl = pd.notna(bar.get("High")) and bar["High"] >= sl_px
                if hit_tp and hit_sl:
                    if   ambig_policy == "sl_first": tag, px = "SL", sl_px
                    elif ambig_policy == "tp_first": tag, px = "TP", tp_px
                    elif ambig_policy == "close_dir":
                        upbar = pd.notna(bar.get("Open")) and pd.notna(bar.get("Close")) and bar["Close"] >= bar["Open"]
                        tag, px = ("SL", sl_px) if upbar else ("TP", tp_px)
                    elif ambig_policy == "random":
                        tag, px = (("TP", tp_px) if rng.random() < 0.5 else ("SL", sl_px))
                elif hit_tp: tag, px = "TP", tp_px
                elif hit_sl: tag, px = "SL", sl_px

            # 2) opposite-signal exit (if a signal actually fires on that calendar day)
            if tag is None and use_opposite_exit:
                opp = ((sig == 2 and pred_map.get(bar_date, 0) == 1) or
                    (sig == 1 and pred_map.get(bar_date, 0) == 2))
                if opp:
                    tag, px = "OPP", (float(bar["Open"]) if pd.notna(bar.get("Open")) else float(bar["Close"]))

            # 3) dynamic trail after progress
            if tag is None and trail_trigger is not None and atr_dol > 0:
                prog_long  = (sig == 2) and pd.notna(bar.get("High")) and (bar["High"] >= entry_px + trail_trigger * TP_ATR * atr_dol)
                prog_short = (sig == 1) and pd.notna(bar.get("Low"))  and (bar["Low"]  <= entry_px - trail_trigger * TP_ATR * atr_dol)
                if   prog_long:  sl_px = max(sl_px, entry_px + trail_k * atr_dol)
                elif prog_short: sl_px = min(sl_px, entry_px - trail_k * atr_dol)

            if tag in ("TP", "SL", "OPP") and pd.notna(px):
                exit_px, exit_ts = float(px), bar_date
                break

        # No exit hit inside window → exit at window end (or last available)
        if exit_px is None:
            tail = spy_df.loc[spy_df.index <= end_date]
            last_bar = tail.iloc[-1]
            exit_px, exit_ts = float(last_bar["Close"]), last_bar.name

        # P&L with costs
        c = (fee_bps + slip_bps) / 1e4
        cost_mult = (1 - c) / (1 + c)
        if sig == 2:
            net = (exit_px / entry_px) * cost_mult - 1.0
        else:
            net = (entry_px / exit_px) * cost_mult - 1.0
        ret = net

        # === record the trade (entry_time = entry_date) ===
        trades_list.append({
            "signal_time": row.get("Timestamp", pd.NaT),
            "sig":         sig,
            "entry_time":  entry_date,            # calendar entry time
            "exit_time":   exit_ts,
            "entry_price": entry_px,
            "exit_price":  exit_px,
            "return_pct":  ret * POSITION_SIZE * pos_mult_final,
        })

        # non-overlap jump + per-direction cooldown
        if not allow_overlap:
            cool = cooldown_long_days if sig == 2 else cooldown_short_days
            cut  = end_date + pd.Timedelta(days=cool)
            j = df.index[df["Date"] >= cut]
            i = int(j[0]) if len(j) else N
        else:
            i += 1







    if ambig_bars:
        print(f"ℹ️ Ambiguous TP/SL bars resolved conservatively: {ambig_bars}")

    if not trades_list:
        print("⚠️  No trades generated in this backtest.")
        zero = {
            "trades": 0, "total_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0,
            "avg_return": 0.0, "median_return": 0.0, "win_rate": 0.0,
            "avg_long": 0.0, "avg_short": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0
        }
        return pd.DataFrame(), zero, simulate_mode




    # ── 11) Compute metrics ────────────────────────────────────────────────────
    trades = pd.DataFrame(trades_list).set_index("signal_time")

    # equity assumes full notional each trade (compounded)
    trades["equity_curve"] = (1.0 + trades["return_pct"]).cumprod()
    trades["cash_curve"]   = CAPITAL_BASE * trades["equity_curve"]
    trades["dollar_return"] = trades["return_pct"] * CAPITAL_BASE

    total_return = trades["equity_curve"].iloc[-1] - 1.0

    # annualize by elapsed trading days between first signal and last exit
    if len(trades) > 1:
        start = pd.to_datetime(trades.index.min(), errors="coerce")
        end   = pd.to_datetime(trades["exit_time"].max(), errors="coerce")
        if pd.notna(start) and pd.notna(end) and end > start:
            elapsed_days = max(1, (end - start).days)
            annualized_return = (1.0 + total_return) ** (252.0 / elapsed_days) - 1.0
        else:
            annualized_return = np.nan
    else:
        annualized_return = np.nan

    # Calculate Sharpe ratio using average holding period
    returns = trades["return_pct"]
    hold_days = (
        pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
    ).dt.days.clip(lower=1)
    avg_hold = hold_days.mean() if not hold_days.empty else 1.0
    print(f"🕒 Avg hold (days): {avg_hold:.2f}")


    sigma = returns.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        sharpe = np.nan
    else:
        sharpe = (returns.mean() / sigma) * np.sqrt(252.0 / avg_hold)
    
    # Warn if too few trades
    if len(trades) < 5:
        print("⚠️  Very few trades — annualized return and Sharpe ratio may be misleading.")

    avg_return    = returns.mean()
    median_return = returns.median()
    win_rate      = (returns > 0).mean()

    avg_long  = trades.loc[trades["sig"] == 2, "return_pct"].mean() if (trades["sig"] == 2).any() else 0.0
    avg_short = trades.loc[trades["sig"] == 1, "return_pct"].mean() if (trades["sig"] == 1).any() else 0.0

    trades["peak"] = trades["equity_curve"].cummax()
    trades["drawdown"] = trades["equity_curve"] / trades["peak"] - 1.0
    max_drawdown = trades["drawdown"].min()


    # Save log
    cols = ["sig","entry_time","exit_time","entry_price","exit_price",
            "return_pct","dollar_return","equity_curve","peak","drawdown","cash_curve"]
    trades[cols].to_csv("logs/trade_log.csv", float_format="%.5f")

    gross_win  = trades.loc[trades["dollar_return"] > 0, "dollar_return"].sum()
    gross_loss = -trades.loc[trades["dollar_return"] < 0, "dollar_return"].sum()
    profit_factor = gross_win / gross_loss if gross_loss != 0 else np.inf

    metrics = {
        "trades": len(trades), "total_return": total_return,
        "annualized_return": annualized_return, "sharpe": sharpe,
        "avg_return": avg_return, "median_return": median_return, "win_rate": win_rate,
        "avg_long": avg_long, "avg_short": avg_short,
        "max_drawdown": max_drawdown, "profit_factor": profit_factor
    }

    # --- Optional top-K per week/month to force activity ---
    TOPK_MODE      = os.getenv("BT_TOPK_MODE", "").strip().lower()    # "week" or "month" or ""
    TOPK_PER_BUCKET= int(os.getenv("BT_TOPK_K", "0"))                  # e.g. 2
    TOPK_MIN_PROB  = float(os.getenv("BT_TOPK_MIN_PROB", "0.0"))       # low bar, e.g. 0.35

    if TOPK_MODE in {"week", "month"} and TOPK_PER_BUCKET > 0 and "Spike_Conf" in df.columns:
        df["bucket"] = df["Date"].dt.to_period("W" if TOPK_MODE=="week" else "M")
        # use Spike_Conf because forward-returns adapter maps trade→Spike
        def _topk(g):
            g = g.sort_values("Spike_Conf", ascending=False)
            g = g[g["Spike_Conf"] >= TOPK_MIN_PROB]
            return g.head(TOPK_PER_BUCKET)
        before = len(df)
        df = df.groupby("bucket", group_keys=False).apply(_topk)
        print(f"🎯 TOP-K filter kept {len(df)} of {before} rows "
            f"({TOPK_PER_BUCKET}/{TOPK_MODE}, min_prob={TOPK_MIN_PROB:.2f})")



    return trades, metrics, simulate_mode



# ─── Threshold optimizer ───────────────────────────────────────────────────────

import itertools
import math

def optimize_thresholds(
        
    window_days: int | None = None,
    grid: dict | None = None,
    min_trades: int = 20,
    objective: str = "avg_dollar_return",
    lookahead=5, tp_atr=1.25, sl_atr=1.0,
    allow_overlap=False, ambig_policy="close_dir",
    fee_bps=2.0, slip_bps=3.0, atr_len=14, trend_len=50

) -> tuple[float | None, float | None, float | None, dict]:
    """
    Grid-search thresholds to maximize chosen objective.
    Returns: (confidence_thresh, crash_thresh, spike_thresh, best_metrics)
    """
    # Default search space (tighten/loosen as you like)
    if grid is None:
        grid = {
            "confidence_thresh": [None, 0.60, 0.70, 0.80, 0.85, 0.90],
            "crash_thresh":      [None, 0.60, 0.70, 0.80, 0.85, 0.90],
            "spike_thresh":      [None, 0.60, 0.70, 0.80, 0.85, 0.90],
        }

    keys = ["confidence_thresh", "crash_thresh", "spike_thresh"]
    combos = list(itertools.product(*(grid[k] for k in keys)))

    rows = []
    best_val = -math.inf
    best_combo = (None, None, None)
    best_metrics = {}

    for conf, crash, spike in combos:
        trades, metrics, _ = run_backtest(
            window_days=window_days,
            crash_thresh=crash, 
            spike_thresh=spike, 
            confidence_thresh=conf,
            lookahead=lookahead, 
            tp_atr=tp_atr, 
            sl_atr=sl_atr,
            allow_overlap=allow_overlap, 
            ambig_policy=ambig_policy,
            fee_bps=fee_bps, 
            slip_bps=slip_bps, 
            atr_len=atr_len, 
            trend_len=trend_len)

        n = metrics.get("trades", 0)
        if n < min_trades:
            val = -math.inf  # avoid overfitting to tiny samples
        else:
            if objective == "avg_dollar_return":
                val = trades["dollar_return"].mean() if not trades.empty else -math.inf
            elif objective == "total_profit":
                val = trades["dollar_return"].sum() if not trades.empty else -math.inf
            elif objective == "win_rate":
                val = (trades["return_pct"] > 0).mean() if not trades.empty else -math.inf
            else:
                # fallback: profit factor
                val = metrics.get("profit_factor", 0.0)

        rows.append({
            "confidence_thresh": conf,
            "crash_thresh": crash,
            "spike_thresh": spike,
            "trades": n,
            "objective": objective,
            "score": val,
            **metrics
        })

        if val > best_val:
            best_val = val
            best_combo = (conf, crash, spike)
            best_metrics = metrics

    # Save the sweep for review
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(
        "logs/threshold_search.csv", index=False
    )
    print(f"🔎 Threshold search complete. Best score={best_val:.6f} "
          f"@ conf={best_combo[0]} crash={best_combo[1]} spike={best_combo[2]}")
    print("📄 Wrote: logs/threshold_search.csv")

    return (*best_combo, best_metrics)

def sweep_params_big():
    from itertools import product
    import pandas as pd
    grids = {
        "confidence_thresh": [0.80, 0.85],
        "crash_thresh":      [0.60, 0.70, 0.75],
        "spike_thresh":      [0.90, 0.95, 0.975],
        "lookahead":         [3, 5],
        "tp_atr":            [1.25, 1.5],
        "sl_atr":            [0.75, 1.0],
        "allow_overlap":     [False],
        "ambig_policy":      ["close_dir"],
    }
    keys = list(grids.keys())
    rows = []
    for vals in product(*[grids[k] for k in keys]):
        kw = dict(zip(keys, vals))
        _, m, _ = run_backtest(window_days=365*12, fee_bps=2.0, slip_bps=3.0, **kw)
        rows.append({**kw, **m})
    df = pd.DataFrame(rows)
    out = "logs/param_sweep.csv"
    df.sort_values(["sharpe","max_drawdown"], ascending=[False,True]).to_csv(out, index=False)
    print("Wrote", out)











# ─── Main entrypoint ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse, json, inspect
    from copy import deepcopy

    # ---- defaults mirroring your current hard-coded run ----
    base_cfg = dict(
        window_days=None,
        lookahead=5,
        tp_atr=1.25,
        sl_atr=0.75,
        allow_overlap=False,
        ambig_policy="close_dir",
        confidence_thresh=None,
        crash_thresh=None,
        spike_thresh=None,
        margin = 0.0,
        fee_bps=2.0,
        slip_bps=3.0,
        atr_len=14,
        trend_len=50,
        target_ann_vol=None,
        vol_lookback=20,
        size_cap=2.0,
        conf_size_bounds=(0.7, 1.3),
        trail_trigger=0.5,
        trail_k=0.5,
        cooldown_days=0,
        cooldown_long_days=1,
        cooldown_short_days=2,
        use_conf_size=True,
        use_opposite_exit=True,
        use_weekly_trend=False,
        use_regime_filter=False,
        use_atr_band=False,
    )

    def _load_json(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

        
    def _parse_grid_list(s: str):
        # e.g. "None,0.7,0.8,0.9" -> [None, 0.7, 0.8, 0.9]
        return [None if t.strip().lower()=="none" else float(t.strip())
                for t in s.split(",") if t.strip() != ""]

    def _save_best_thresholds(path, conf, crash, spike, meta=None):
        dirpath = os.path.dirname(path)
        if dirpath:  # only mkdir if there is a directory component
            os.makedirs(dirpath, exist_ok=True)
        payload = {
            "confidence_thresh": conf,
            "crash_thresh": crash,
            "spike_thresh": spike,
            "meta": (meta or {})
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved best thresholds → {path}")


        
    

    parser = argparse.ArgumentParser(description="Backtest runner")
    parser.add_argument("--config", help="JSON file to seed/override params")
    parser.add_argument("--save-config", help="Save the final merged config to this JSON")

    # numerics
    parser.add_argument("--window-days", type=int)
    parser.add_argument("--lookahead", type=int)
    parser.add_argument("--tp-atr", type=float)
    parser.add_argument("--sl-atr", type=float)
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--slip-bps", type=float)
    parser.add_argument("--atr-len", type=int)
    parser.add_argument("--trend-len", type=int)
    parser.add_argument("--confidence-thresh", type=float)
    parser.add_argument("--crash-thresh", type=float)
    parser.add_argument("--spike-thresh", type=float)
    parser.add_argument("--ambig-policy", choices=["sl_first","tp_first","skip","close_dir","random"])
    parser.add_argument("--cooldown-days", type=int)
    parser.add_argument("--cooldown-long-days", type=int)
    parser.add_argument("--cooldown-short-days", type=int)
    parser.add_argument("--target-ann-vol", type=float)
    parser.add_argument("--vol-lookback", type=int)
    parser.add_argument("--size-cap", type=float)
    parser.add_argument("--trail-trigger", type=float)
    parser.add_argument("--trail-k", type=float)
    parser.add_argument("--conf-size-bounds", type=str, help="lo,hi  e.g. 0.7,1.3")

    # IMPORTANT: do NOT set a default for margin here; let base_cfg control it
    parser.add_argument("--margin", type=float,
        help="Min |Spike_Conf - Crash_Conf| to accept (separation margin)")

    # booleans (tri-state so JSON/flags can override either way)
    parser.add_argument("--allow-overlap", dest="allow_overlap", action="store_true")
    parser.add_argument("--no-allow-overlap", dest="allow_overlap", action="store_false")
    parser.set_defaults(allow_overlap=None)

    parser.add_argument("--use-opposite-exit", dest="use_opposite_exit", action="store_true")
    parser.add_argument("--no-use-opposite-exit", dest="use_opposite_exit", action="store_false")
    parser.set_defaults(use_opposite_exit=None)

    parser.add_argument("--use-conf-size", dest="use_conf_size", action="store_true")
    parser.add_argument("--no-use-conf-size", dest="use_conf_size", action="store_false")
    parser.set_defaults(use_conf_size=None)

    parser.add_argument("--use-weekly-trend", dest="use_weekly_trend", action="store_true")
    parser.add_argument("--no-use-weekly-trend", dest="use_weekly_trend", action="store_false")
    parser.set_defaults(use_weekly_trend=None)

    # ---- optimizer switches ----
    parser.add_argument("--optimize", action="store_true",
                        help="Run threshold grid search instead of a plain backtest.")
    parser.add_argument("--opt-window-days", type=int,
                        help="Window (days) to use during optimization; defaults to --window-days.")
    parser.add_argument("--opt-min-trades", type=int, default=20,
                        help="Minimum trades required for a combo to be considered.")
    parser.add_argument("--opt-objective", default="avg_dollar_return",
                        choices=["avg_dollar_return","total_profit","win_rate","profit_factor"],
                        help="Metric to maximize during optimization.")
    parser.add_argument("--grid-confidence", type=str,
                        help='Comma list for confidence grid, e.g. "None,0.6,0.7,0.8,0.9"')
    parser.add_argument("--grid-crash", type=str,
                        help='Comma list for crash grid, e.g. "None,0.6,0.7,0.8,0.9"')
    parser.add_argument("--grid-spike", type=str,
                        help='Comma list for spike grid, e.g. "None,0.9,0.95,0.975"')
    parser.add_argument("--save-best", type=str, default="configs/best_thresholds.json",
                        help="Path to save best thresholds JSON.")
    parser.add_argument("--apply-best", action="store_true",
                        help="After optimizing, run a backtest using the best thresholds.")

    parser.add_argument("--dyn-t", type=float,
        help="Override learned threshold t for dynamic gating (0..1)")

    parser.add_argument("--use-regime-filter", dest="use_regime_filter", action="store_true")
    parser.add_argument("--no-use-regime-filter", dest="use_regime_filter", action="store_false")
    parser.set_defaults(use_regime_filter=None)

    parser.add_argument("--use-atr-band",    dest="use_atr_band", action="store_true")
    parser.add_argument("--no-use-atr-band", dest="use_atr_band", action="store_false")
    parser.set_defaults(use_atr_band=None)

    # ✅ ADD THIS BEFORE PARSING:
    parser.add_argument("--use-model-labels", action="store_true",
        help="Ignore explicit thresholds and use model-provided labels.")

    # ---- finally parse
    args = parser.parse_args()

    # ---- merge order: defaults → config file → CLI flags
    cfg = deepcopy(base_cfg)
    if args.config:
        cfg.update(_load_json(args.config))

    # If flag is set, clear thresholds so the model’s labels are used
    if args.use_model_labels:
        cfg["confidence_thresh"] = None
        cfg["crash_thresh"] = None
        cfg["spike_thresh"] = None



     

    # numeric/str updates
    for k in [
        "window_days","lookahead","tp_atr","sl_atr","fee_bps","slip_bps",
        "atr_len","trend_len","confidence_thresh","crash_thresh","spike_thresh",
        "ambig_policy","cooldown_days","cooldown_long_days","cooldown_short_days",
        "target_ann_vol","vol_lookback","size_cap","trail_trigger","trail_k", "dyn_t","margin"
    ]:
        v = getattr(args, k, None)
        if v is not None: cfg[k] = v

    # booleans
    for k in ["allow_overlap","use_conf_size","use_opposite_exit","use_weekly_trend","use_regime_filter","use_atr_band"]:
        v = getattr(args, k, None)
        if v is not None: cfg[k] = bool(v)

    # conf_size_bounds "lo,hi"
    if args.conf_size_bounds:
        lo, hi = (float(x) for x in args.conf_size_bounds.split(","))
        cfg["conf_size_bounds"] = (lo, hi)

    # keep only kwargs that run_backtest actually accepts
    allowed = set(inspect.signature(run_backtest).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in allowed}

    if args.optimize:
        # Build grid from flags or use defaults inside optimize_thresholds
        grid = None
        if any([args.grid_confidence, args.grid_crash, args.grid_spike]):
            grid = {
                "confidence_thresh": _parse_grid_list(args.grid_confidence) if args.grid_confidence else [None, 0.6, 0.7, 0.8, 0.9],
                "crash_thresh":      _parse_grid_list(args.grid_crash)      if args.grid_crash      else [None, 0.6, 0.7, 0.8, 0.9],
                "spike_thresh":      _parse_grid_list(args.grid_spike)      if args.grid_spike      else [None, 0.9, 0.95, 0.975],
            }

        # Use opt-window-days if provided, else fall back to cfg["window_days"]
        opt_window = args.opt_window_days if args.opt_window_days is not None else cfg.get("window_days", None)

        best_conf, best_crash, best_spike, best_metrics = optimize_thresholds(
            window_days=opt_window,
            grid=grid,
            min_trades=args.opt_min_trades,
            objective=args.opt_objective,
            # carry through core backtest mechanics that affect PnL
            lookahead=cfg["lookahead"],
            tp_atr=cfg["tp_atr"],
            sl_atr=cfg["sl_atr"],
            allow_overlap=cfg["allow_overlap"],
            ambig_policy=cfg["ambig_policy"],
            fee_bps=cfg["fee_bps"],
            slip_bps=cfg["slip_bps"],
            atr_len=cfg["atr_len"],
            trend_len=cfg["trend_len"],
        )

        print("\n🏁 Optimization result")
        print(f"  Best thresholds: confidence={best_conf}, crash={best_crash}, spike={best_spike}")
        for k,v in best_metrics.items():
            if isinstance(v, float):
                if "return" in k or "rate" in k or "drawdown" in k:
                    print(f"  {k:>20}: {v:.4f}")
                else:
                    print(f"  {k:>20}: {v:.6f}")
            else:
                print(f"  {k:>20}: {v}")

        # Save for future runs
        _save_best_thresholds(args.save_best, best_conf, best_crash, best_spike, meta={
            "objective": args.opt_objective,
            "min_trades": args.opt_min_trades,
            "window_days": opt_window,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })

        # Optionally apply and continue into a real backtest with the best thresholds
        if args.apply_best:
            cfg["confidence_thresh"] = best_conf
            cfg["crash_thresh"] = best_crash
            cfg["spike_thresh"] = best_spike
            print("\n▶️  Applying best thresholds and running a backtest...")
        else:
            # Exit early: do not run a backtest unless asked
            exit(0)


    trades, m, simulate_mode = run_backtest(**kwargs)

    # --- print your existing backtest report (unchanged) ---
    print("\n📈 Backtest Report")
    print(f"  Trades taken:       {m['trades']}")
    print(f"  Total return:       {m['total_return']:.2%}")
    print(f"  Annualized return:  {m['annualized_return']:.2%}")
    print(f"  Sharpe ratio (252d):{m['sharpe']:.2f}\n")
    print(f"  Max drawdown:        {m['max_drawdown']:.2%}")
    print(f"  Avg long:           {m['avg_long']:.5f}")
    print(f"  Avg short:          {m['avg_short']:.5f}")

    # optional summary/record (if you pasted the helpers earlier)
    try:
        print_run_summary(m, cfg)
        save_run_record(cfg, m, simulate_mode, trades)
    except NameError:
        pass

    if not trades.empty:
        final_balance = trades["cash_curve"].iloc[-1]
        print(f"  Final capital:      ${final_balance:,.2f}")
        print(f"  Net profit:         ${final_balance - CAPITAL_BASE:,.2f}")

    print("\nSample trades:")
    if trades.empty:
        print("  (no trades to show)")
    else:
        print(trades.head())

        trades = trades.rename(columns={"equity_curve": "Equity"})
        trades["Drawdown %"] = trades["drawdown"] * 100

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        trades["Equity"].plot(ax=ax1, title="Equity Curve"); ax1.set_ylabel("Cumulative Return"); ax1.grid(True)
        trades["Drawdown %"].plot(ax=ax2, title="Drawdown (%)", color='red', linestyle='--')
        ax2.set_ylabel("Drawdown"); ax2.axhline(0, color="black", linewidth=0.5); ax2.grid(True)
        plt.xlabel("Signal Time"); plt.tight_layout(); plt.show()
        fig.savefig("logs/equity_drawdown_plot.png", dpi=300)
        print("📸 Saved equity and drawdown chart to logs/equity_drawdown_plot.png")

    if simulate_mode:
        print("\n⚠️ NOTE: This was a simulated run with injected predictions.")
