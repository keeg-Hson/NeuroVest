#!/usr/bin/env python3
"""
NeuroVest CLI - one-file command center

Wraps runnable scripts into a single CLI with subcommands:
  data        → update SPY data (update_spy_data.py) or bootstrap downloads
  labels      → generate labels (generate_labels.py)
  train       → train model (train_from_labels.py)
  predict     → run predictions (predict.py) [FULL backfill with --backfill]
  predict-live→ append a single live row (predict.py without --backfill)
  thresh-sweep→ sweep balanced accuracy; write configs/best_thresholds.json (with invert_proba); re-run predict backfill
  backtest    → full trade sim (backtest.py), with pass-through flags
  backtest-opt→ backtest optimizer mode (backtest.py --optimize ...)
  eval        → evaluate predictions (evaluate.py)
  tearsheet   → build P&L/tearsheet (tearsheet.py)
  run-all     → end-to-end pipeline (run_all.py)
  schedule    → daily scheduler (run_daily_pipeline.py)
  live        → live loop (live_loop.py)
  sanitize    → clean dirty CSVs in data/
  features    → build a canonical feature table (single CSV/Parquet)
  all         → run EVERYTHING sequentially; optional --auto-threshold

Label convention (unified)
--------------------------
Predictions and evaluation use:
    0 = SPIKE
    1 = NORMAL
    2 = CRASH

Threshold sweeps treat the positive class as "SPIKE" under this convention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# ---------- helpers ----------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
CONFIGS = ROOT / "configs"

SCRIPTS = {
    "update_spy_data": ROOT / "update_spy_data.py",
    "generate_labels": ROOT / "generate_labels.py",
    "train_from_labels": ROOT / "train_from_labels.py",
    "predict": ROOT / "predict.py",
    "backtest": ROOT / "backtest.py",
    "backtest_module": ROOT / "backtest_module.py",
    "evaluate": ROOT / "evaluate.py",
    "tearsheet": ROOT / "tearsheet.py",
    "run_all": ROOT / "run_all.py",
    "run_daily_pipeline": ROOT / "run_daily_pipeline.py",
    "live_loop": ROOT / "live_loop.py",
    # optional utilities if present
    "bootstrap_downloads": ROOT / "data" / "bootstrap_downloads.py",
}


def _exists(p: Path) -> bool:
    return p.exists()


def _ensure_dirs():
    for d in (DATA, LOGS, MODELS, OUT, CONFIGS):
        d.mkdir(parents=True, exist_ok=True)


def _run_py(script: Path, extra_args: list[str] | None = None, check: bool = True) -> int:
    if not _exists(script):
        print(f"⚠️  script not found: {script.name} (expected at {script})")
        return 127
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"▶️  running: {' '.join(cmd)}")
    return subprocess.call(cmd) if not check else subprocess.check_call(cmd)


# ---------- subcommand implementations ----------


def cmd_data(args: argparse.Namespace) -> None:
    _ensure_dirs()
    if args.action == "update":
        _run_py(SCRIPTS["update_spy_data"], check=False)
    elif args.action == "bootstrap":
        if _exists(SCRIPTS["bootstrap_downloads"]):
            _run_py(SCRIPTS["bootstrap_downloads"], check=False)
        else:
            # fallback bootstrap via yfinance (no extra file needed)
            code = dedent(
                """
            import yfinance as yf
            from pathlib import Path
            OUT=Path("data"); OUT.mkdir(parents=True, exist_ok=True)
            START="2000-01-01"
            symbols={"SPY":"SPY","HYG":"HYG","LQD":"LQD","UUP":"UUP","TNX":"^TNX","DXY":"DX-Y.NYB"}
            for name, ysym in symbols.items():
                try:
                    df=yf.download(ysym, start=START, auto_adjust=False)
                    if df.empty:
                        print(f"[skip] {name} empty"); continue
                    df.index.name="Date"
                    df=df.reset_index()
                    cols=["Date","Open","High","Low","Close","Adj Close","Volume"]
                    for c in cols:
                        if c not in df.columns: df[c]=0
                    df=df[cols]
                    df.to_csv(OUT/f"{name}.csv", index=False)
                    print(f"✅ wrote data/{name}.csv ({len(df)})")
                except Exception as e:
                    print(f"⚠️ {name}: {e}")
            """
            ).strip()
            subprocess.check_call([sys.executable, "-c", code])


def cmd_labels(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["generate_labels"], check=False)


def cmd_train(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["train_from_labels"], check=False)


def cmd_predict(args: argparse.Namespace) -> None:
    _ensure_dirs()
    if getattr(args, "backfill", False):
        _run_py(SCRIPTS["predict"], extra_args=["--backfill"], check=False)
    else:
        _run_py(SCRIPTS["predict"], check=False)


def cmd_predict_live(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["predict"], check=False)


def _sweep_best_threshold(csv_path: Path) -> tuple[float, bool, float, float]:
    """
    Choose threshold that maximizes BALANCED ACCURACY on current labeled_predictions.csv.
    Uses the unified label convention (0=SPIKE,1=NORMAL,2=CRASH) and treats
    the positive class as "SPIKE".

    Also detects whether probabilities should be inverted (AUC < 0.5 and flipped AUC > 0.5).

    Returns: (best_threshold, invert_proba, best_bal_acc, plain_accuracy_at_best)
    """
    # normalize/derive Actuals if needed
    _ensure_dirs()
    _run_py(SCRIPTS["evaluate"], check=False)

    df = pd.read_csv(csv_path, low_memory=False)
    if "Actual_Event" not in df.columns:
        raise SystemExit("Actual_Event not present after evaluate; cannot sweep threshold.")
    if "Proba" not in df.columns:
        raise SystemExit("Proba column not present in predictions; cannot sweep threshold.")

    # Raw multi-class labels under unified convention: 0=SPIKE,1=NORMAL,2=CRASH
    y_raw = pd.to_numeric(df["Actual_Event"], errors="coerce")

    # Derive a binary target for thresholding.
    # Primary rule under unified mapping: positive class = SPIKE (0).
    # If labels are strictly binary {0,1} or {1,2}, fall back to "max label is positive" as a safety net.
    y_unique = set(int(v) for v in y_raw.dropna().unique())

    if y_unique.issuperset({0, 1, 2}):
        # Explicit 3-class case → SPIKE vs non-SPIKE
        y = (y_raw == 0).astype(int)
    elif y_unique == {0, 1}:
        # Binary case with 0/1 → treat 1 as positive
        y = (y_raw == 1).astype(int)
    elif y_unique == {1, 2}:
        # Binary case with 1/2 → treat 2 as positive
        y = (y_raw == 2).astype(int)
    else:
        # Fallback: treat the maximum label value as the positive class
        max_lab = max(y_unique) if y_unique else 1
        y = (y_raw == max_lab).astype(int)

    p = pd.to_numeric(df["Proba"], errors="coerce")
    mask = ~(y.isna() | p.isna())
    y, p = y[mask].to_numpy(), p[mask].to_numpy()

    if y.size == 0:
        raise SystemExit("No valid (y, p) pairs for threshold sweep after filtering NaNs.")

    # Decide inversion by AUC on the derived binary target
    inv = False
    try:
        auc = roc_auc_score(y, p)
        auc_flip = roc_auc_score(y, 1.0 - p)
        if auc < 0.5 and auc_flip > 0.5:
            inv = True
            p = 1.0 - p
            print(
                f"[thresh-sweep] probability inversion enabled (AUC={auc:.4f} → flipped={auc_flip:.4f})"
            )
        else:
            print(
                f"[thresh-sweep] probability inversion not used (AUC={auc:.4f}, flipped={auc_flip:.4f})"
            )
    except Exception as e:
        print(f"[thresh-sweep] AUC computation failed: {e}; proceeding without inversion")

    # Sweep thresholds for balanced accuracy
    grid = np.arange(0.25, 0.7500001, 0.005)
    best_thr, best_bal, best_acc = 0.55, -1.0, -1.0
    for thr in grid:
        pred = (p >= thr).astype(int)
        bal = balanced_accuracy_score(y, pred)
        acc = (pred == y).mean()
        if bal > best_bal:
            best_thr, best_bal, best_acc = float(thr), float(bal), float(acc)

    print(
        f"[thresh-sweep] best threshold={best_thr:.3f}  bal_acc={best_bal:.4f}  acc={best_acc:.4f}"
    )
    return best_thr, inv, best_bal, best_acc


def cmd_thresh_sweep(_: argparse.Namespace) -> None:
    """
    Sweep balanced accuracy over current predictions, possibly set invert_proba,
    write configs/best_thresholds.json, then re-run predict backfill.
    """
    _ensure_dirs()
    labeled = LOGS / "labeled_predictions.csv"
    if not labeled.exists():
        # Ensure there are predictions to sweep
        cmd_predict(argparse.Namespace(backfill=True))

    best_thr, inv, best_bal, best_acc = _sweep_best_threshold(labeled)

    CONFIGS.mkdir(parents=True, exist_ok=True)
    tgt = CONFIGS / "best_thresholds.json"
    with open(tgt, "w") as f:
        json.dump({"spike_thresh": best_thr, "invert_proba": bool(inv)}, f, indent=2)
    print(f"[thresh-sweep] wrote {tgt} with spike_thresh={best_thr:.3f}  invert_proba={bool(inv)}")

    # Re-apply by rebuilding predictions with the new threshold/inversion
    cmd_predict(argparse.Namespace(backfill=True))


# --- minimal guard so backtest/eval use a sane threshold without changing CLI UX ---
def _ensure_threshold_before_eval(auto_threshold: bool) -> None:
    """
    If --auto-threshold is set → sweep now.
    If not set but configs/best_thresholds.json is missing → sweep once to initialize.
    Otherwise → do nothing.
    """
    CONFIGS.mkdir(parents=True, exist_ok=True)
    tgt = CONFIGS / "best_thresholds.json"
    if auto_threshold or not tgt.exists():
        labeled = LOGS / "labeled_predictions.csv"
        if not labeled.exists():
            cmd_predict(argparse.Namespace(backfill=True))
        best_thr, inv, *_ = _sweep_best_threshold(labeled)
        with open(tgt, "w") as f:
            json.dump({"spike_thresh": best_thr, "invert_proba": bool(inv)}, f, indent=2)
        print(
            f"[thresh-sweep] wrote {tgt} with spike_thresh={best_thr:.3f}  invert_proba={bool(inv)}"
        )
        cmd_predict(argparse.Namespace(backfill=True))
    else:
        print("[thresh-sweep] using existing configs/best_thresholds.json")


def cmd_backtest(args: argparse.Namespace) -> None:
    _ensure_dirs()
    rest = args.rest or []
    rest = [x for x in rest if x != "--"]
    _run_py(SCRIPTS["backtest"], extra_args=rest, check=False)


def cmd_backtest_opt(args: argparse.Namespace) -> None:
    _ensure_dirs()
    rest = args.rest or []
    rest = [x for x in rest if x != "--"]
    if "--optimize" not in rest:
        rest = ["--optimize", "--apply-best"] + rest
    _run_py(SCRIPTS["backtest"], extra_args=rest, check=False)


def cmd_eval(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["evaluate"], check=False)


def cmd_tearsheet(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["tearsheet"], check=False)


def cmd_run_all(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["run_all"], check=False)


def cmd_schedule(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["run_daily_pipeline"], check=False)


def cmd_live(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["live_loop"], check=False)


# ---------- built-in sanitizer (repair messy CSVs) ----------

DATE_CANDIDATES = ["Date", "DATE", "date", "observation_date", "timestamp", "ds", "Ticker"]


def _load_any_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if (
        "Ticker" in df.columns
        and isinstance(df["Ticker"].iloc[0], str)
        and df["Ticker"].iloc[0].lower() == "date"
    ):
        df = df.iloc[1:].reset_index(drop=True)
    date_col = None  # detect a viable date column
    for cand in DATE_CANDIDATES:
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            if dt.notna().sum() > 3:
                df["Date"] = dt
                date_col = "Date"
                break
    if date_col is None:
        raise ValueError(f"{path.name}: no recognizable date column")
    df = df[~df["Date"].isna()].drop_duplicates(subset=["Date"]).sort_values("Date")
    return df


def _pick_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    exact = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if all(c in df.columns for c in exact):
        return df[["Date"] + exact].copy()
    numcols = [c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])]
    if len(numcols) >= 6:
        out = pd.concat([df[["Date"]], df[numcols[:6]]], axis=1)
        out.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        return out
    raise ValueError("No OHLCV columns found")


def _enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
    df["Volume"] = df["Volume"].fillna(0).astype(float)
    return df


def cmd_sanitize(_: argparse.Namespace) -> None:
    _ensure_dirs()
    repaired = []
    for p in sorted(DATA.glob("*.csv")):
        try:
            raw = _load_any_csv(p)
            clean = _enforce_types(_pick_ohlcv(raw))
            clean.to_csv(p, index=False)
            repaired.append((p.name, len(clean)))
            print(f"✅ cleaned {p.name}: {len(clean)} rows")
        except Exception as e:
            print(f"⚠️ {p.name}: {e}")
    if not repaired:
        print("ℹ️ nothing repaired; data/*.csv may already be clean.")


# ---------- built-in feature table builder ----------


def cmd_features(args: argparse.Namespace) -> None:
    _ensure_dirs()
    price_path = Path(args.price_csv)
    ext_dir = Path(args.external_dir) if args.external_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not price_path.exists():
        raise SystemExit(f"Price CSV not found: {price_path}")

    usecols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    price_df = pd.read_csv(price_path, usecols=lambda c: c in usecols, low_memory=False)
    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    price_df = price_df.dropna(subset=["Date"]).sort_values("Date")
    price_df = price_df.set_index("Date")
    keep = [
        c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in price_df.columns
    ]
    price_df = price_df[keep]

    base = price_df.copy()

    if ext_dir and ext_dir.exists():
        for csv in sorted(ext_dir.glob("*.csv")):
            try:
                x = pd.read_csv(csv, low_memory=False)
                for cand in DATE_CANDIDATES:
                    if cand in x.columns:
                        x[cand] = pd.to_datetime(x[cand], errors="coerce")
                        x = x.dropna(subset=[cand]).sort_values(cand).set_index(cand)
                        break
                else:
                    raise ValueError("no date column found")
                num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
                if not num_cols:
                    continue
                x = x[num_cols].add_prefix(csv.stem + "_")
                base = pd.merge_asof(
                    base.sort_index(),
                    x.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                )
            except Exception as e:
                print(f"[warn] external merge failed for {csv.name}: {e}")

    close = base["Adj Close"] if "Adj Close" in base.columns else base["Close"]
    high, low, vol = base["High"], base["Low"], base["Volume"]

    def sma(s, w):
        return s.rolling(w).mean()

    def ema(s, w):
        return s.ewm(span=w, adjust=False).mean()

    def rsi(s, w=14):
        d = s.diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        rs = up.ewm(alpha=1 / w, adjust=False).mean() / dn.ewm(
            alpha=1 / w, adjust=False
        ).mean().replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def macd(s):
        f, sl = ema(s, 12), ema(s, 26)
        m = f - sl
        sig = ema(m, 9)
        return m, sig, m - sig

    def trange(high, low, close):
        pc = close.shift(1)
        return pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    def atr(high, low, close, w=14):
        return trange(high, low, close).ewm(alpha=1 / w, adjust=False).mean()

    def boll(s, w=20, n=2.0):
        m = sma(s, w)
        sd = s.rolling(w).std()
        return m + n * sd, m, m - n * sd

    feats = pd.DataFrame(index=base.index)
    feats["ret_1d"] = close.pct_change(1, fill_method=None)
    feats["ret_5d"] = close.pct_change(5, fill_method=None)
    feats["ret_10d"] = close.pct_change(10, fill_method=None)
    feats["logret_1d"] = np.log(close).diff(1)
    feats["sma_20"] = sma(close, 20)
    feats["sma_50"] = sma(close, 50)
    feats["ema_12"] = ema(close, 12)
    feats["ema_26"] = ema(close, 26)
    feats["rsi_14"] = rsi(close, 14)
    m, s, hist = macd(close)
    feats["macd"] = m
    feats["macd_signal"] = s
    feats["macd_hist"] = hist
    up, mid, lo = boll(close, 20, 2.0)
    feats["bb_up_20_2"] = up
    feats["bb_mid_20_2"] = mid
    feats["bb_lo_20_2"] = lo
    feats["bb_width_20_2"] = (up - lo) / mid
    feats["atr_14"] = atr(high, low, close, 14)
    feats["vol_20"] = close.pct_change(fill_method=None).rolling(20).std()
    k_low = low.rolling(14).min()
    k_high = high.rolling(14).max()
    feats["stoch_k_14_3"] = 100 * (close - k_low) / (k_high - k_low)
    feats["stoch_d_14_3"] = feats["stoch_k_14_3"].rolling(3).mean()
    feats["vol_sma_20"] = sma(vol, 20)
    feats["vol_pct_20"] = vol / feats["vol_sma_20"] - 1.0
    feats["px_over_sma20"] = close / feats["sma_20"] - 1.0
    feats["px_over_sma50"] = close / feats["sma_50"] - 1.0

    H = int(args.horizon)
    fwd_ret = close.shift(-H) / close - 1.0
    y = (fwd_ret >= float(args.min_return)).astype("Int64")
    y.iloc[-H:] = pd.NA

    out = pd.DataFrame(index=feats.index)
    out["date"] = out.index
    out["ticker"] = args.ticker
    for c in keep:
        out[c.lower().replace(" ", "_")] = price_df[c] if c in price_df.columns else base[c]
    out = out.join(feats)
    out["y_up_fwd"] = y

    n = len(out)
    ntr = int(n * 0.7)
    nval = int(n * 0.15)
    split = pd.Series(index=out.index, dtype="string")
    split.iloc[:ntr] = "train"
    split.iloc[ntr : ntr + nval] = "val"
    split.iloc[ntr + nval :] = "test"
    out["split"] = split

    feat_cols = [c for c in out.columns if c not in ["date", "ticker", "y_up_fwd", "split"]]
    mask = out[feat_cols].notna().all(axis=1) & out["y_up_fwd"].notna()
    out = out.loc[mask].copy()
    out = out[["date", "ticker"] + feat_cols + ["y_up_fwd", "split"]]

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d")
    base_name = f"neurovest_features_{stamp}"
    pq = out_dir / f"{base_name}.parquet"
    csv = out_dir / f"{base_name}.csv"
    fl = out_dir / f"{base_name}.features.txt"
    meta = out_dir / f"{base_name}.meta.json"

    wrote_parquet = False
    try:
        out.to_parquet(pq, index=False)
        wrote_parquet = True
    except Exception as e:
        print(f"[warn] parquet write failed: {e}; writing CSV only")
    out.to_csv(csv, index=False)

    with open(fl, "w") as f:
        for c in feat_cols:
            f.write(c + "\n")
    with open(meta, "w") as f:
        json.dump(
            {
                "created_at_utc": pd.Timestamp.utcnow().isoformat(),
                "price_csv": str(price_path),
                "external_dir": str(ext_dir) if ext_dir else None,
                "rows": int(len(out)),
                "features": feat_cols,
                "label_col": "y_up_fwd",
                "split_col": "split",
                "horizon_days": H,
                "min_return": float(args.min_return),
                "ticker": args.ticker,
            },
            f,
            indent=2,
        )

    print("✅ feature table written:")
    if wrote_parquet:
        print(" -", pq)
    print(" -", csv)
    print(" -", fl)
    print(" -", meta)


# ---------- ALL-IN-ONE runner ----------


def cmd_all(args: argparse.Namespace) -> None:
    """
    Run: sanitize → (bootstrap|update) → labels → train → predict → [auto-threshold/init-threshold] → backtest → eval → tearsheet
    """
    _ensure_dirs()

    if not args.no_sanitize:
        cmd_sanitize(args)

    ns = argparse.Namespace(action="bootstrap" if args.bootstrap else "update")
    cmd_data(ns)
    cmd_labels(ns)
    cmd_train(ns)
    cmd_predict(argparse.Namespace(backfill=True))
    _ensure_threshold_before_eval(auto_threshold=args.auto_threshold)

    rest = args.rest or []
    if args.fast:
        default_bt = [
            "--use-regime-filter",
            "--lookahead",
            "5",
            "--tp-atr",
            "1.25",
            "--sl-atr",
            "0.75",
            "--fee-bps",
            "1",
            "--slip-bps",
            "1",
        ]
        cmd_backtest(argparse.Namespace(rest=(default_bt + rest)))
    else:
        default_opt = [
            "--optimize",
            "--apply-best",
            "--opt-min-trades",
            "30",
            "--opt-objective",
            "profit_factor",
            "--use-regime-filter",
            "--lookahead",
            "5",
            "--tp-atr",
            "1.25",
            "--sl-atr",
            "0.75",
            "--fee-bps",
            "1",
            "--slip-bps",
            "1",
        ]
        cmd_backtest(argparse.Namespace(rest=(default_opt + rest)))

    cmd_eval(ns)
    cmd_tearsheet(ns)


# ---------- parser ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NeuroVest one-file CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("data", help="data utilities")
    dp_sub = dp.add_subparsers(dest="action", required=True)
    dp_sub.add_parser("update", help="run update_spy_data.py")
    dp_sub.add_parser("bootstrap", help="download SPY/HYG/LQD/UUP/TNX/DXY via yfinance")
    dp.set_defaults(func=cmd_data)

    sub.add_parser("labels", help="run generate_labels.py").set_defaults(func=cmd_labels)
    sub.add_parser("train", help="run train_from_labels.py").set_defaults(func=cmd_train)

    sp_predict = sub.add_parser("predict", help="run predict.py (FULL backfill with --backfill)")
    sp_predict.add_argument(
        "--backfill",
        action="store_true",
        help="Score full history and rewrite logs/labeled_predictions.csv",
    )
    sp_predict.set_defaults(func=cmd_predict)

    sub.add_parser("predict-live", help="append a single live row").set_defaults(
        func=cmd_predict_live
    )

    sub.add_parser(
        "thresh-sweep",
        help="sweep balanced accuracy → configs/best_thresholds.json (with invert_proba) → backfill",
    ).set_defaults(func=cmd_thresh_sweep)

    bp = sub.add_parser("backtest", help="run backtest.py (pass-through flags with --)")
    bp.add_argument("rest", nargs=argparse.REMAINDER, help="flags after -- passed to backtest.py")
    bp.set_defaults(func=cmd_backtest)

    bop = sub.add_parser("backtest-opt", help="run backtest optimizer (--optimize ...)")
    bop.add_argument("rest", nargs=argparse.REMAINDER, help="flags after -- passed to backtest.py")
    bop.set_defaults(func=cmd_backtest_opt)

    sub.add_parser("eval", help="run evaluate.py").set_defaults(func=cmd_eval)
    sub.add_parser("tearsheet", help="run tearsheet.py").set_defaults(func=cmd_tearsheet)
    sub.add_parser("run-all", help="run run_all.py").set_defaults(func=cmd_run_all)
    sub.add_parser("schedule", help="run run_daily_pipeline.py").set_defaults(func=cmd_schedule)
    sub.add_parser("live", help="run live_loop.py").set_defaults(func=cmd_live)

    sub.add_parser("sanitize", help="repair messy data/*.csv").set_defaults(func=cmd_sanitize)

    fp = sub.add_parser("features", help="materialize single feature table")
    fp.add_argument("--price-csv", default="data/SPY.csv")
    fp.add_argument("--external-dir", default="data_cache")
    fp.add_argument("--horizon", type=int, default=5)
    fp.add_argument("--min-return", type=float, default=0.0)
    fp.add_argument("--out-dir", default="data/features")
    fp.add_argument("--ticker", default="SPY")
    fp.set_defaults(func=cmd_features)

    ap = sub.add_parser(
        "all",
        help="run the whole pipeline (sanitize → data → labels → train → predict → [auto-threshold/init-threshold] → backtest → eval → tearsheet)",
    )
    ap.add_argument(
        "--fast", action="store_true", help="skip optimizer; quick backtest with sane defaults"
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="bootstrap fresh downloads instead of update_spy_data.py",
    )
    ap.add_argument("--no-sanitize", action="store_true", help="skip the sanitize step")
    ap.add_argument(
        "--auto-threshold",
        action="store_true",
        help="sweep best threshold (balanced accuracy) and re-apply before backtest/eval",
    )
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="flags after -- passed to backtest.py")
    ap.set_defaults(func=cmd_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
