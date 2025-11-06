#!/usr/bin/env python3
"""
NeuroVest CLI - one-file command center

Wraps runnable scripts into a single CLI with subcommands:
  data        → update SPY data (update_spy_data.py) or bootstrap downloads
  labels      → generate labels (generate_labels.py)
  train       → train model (train_from_labels.py)
  predict     → run predictions (predict.py)
  backtest    → full trade sim (backtest.py), with pass-through flags
  backtest-opt→ backtest optimizer mode (backtest.py --optimize ...)
  eval        → evaluate predictions (evaluate.py)
  tearsheet   → build P&L/tearsheet (tearsheet.py)
  run-all     → end-to-end pipeline (run_all.py)
  schedule    → daily scheduler (run_daily_pipeline.py)
  live        → live loop (live_loop.py)
  sanitize    → clean dirty CSVs in data/
  features    → build a canonical feature table (single CSV/Parquet)
  all         → 🔥 run EVERYTHING sequentially (sanitize → data → labels → train → predict → backtest → eval → tearsheet)

Usage examples:
  python3 neurovest_cli.py data update
  python3 neurovest_cli.py labels
  python3 neurovest_cli.py train
  python3 neurovest_cli.py predict
  python3 neurovest_cli.py backtest -- --use-regime-filter --lookahead 5 --tp-atr 1.25 --sl-atr 0.75
  python3 neurovest_cli.py backtest-opt -- --optimize --apply-best --opt-min-trades 30
  python3 neurovest_cli.py eval
  python3 neurovest_cli.py tearsheet
  python3 neurovest_cli.py run-all
  python3 neurovest_cli.py features --price-csv data/SPY.csv --external-dir data_cache --horizon 5
  python3 neurovest_cli.py all
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

# ---------- helpers ----------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

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
    for d in (DATA, LOGS, MODELS, OUT):
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


def cmd_predict(_: argparse.Namespace) -> None:
    _ensure_dirs()
    _run_py(SCRIPTS["predict"], check=False)


def cmd_backtest(args: argparse.Namespace) -> None:
    _ensure_dirs()
    passthru = args.rest or []
    _run_py(SCRIPTS["backtest"], extra_args=passthru, check=False)


def cmd_backtest_opt(args: argparse.Namespace) -> None:
    _ensure_dirs()
    # ensure optimizer flags exist; pass the rest through as-is
    rest = args.rest or []
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
    # fix "Ticker == Date" header-as-row
    if (
        "Ticker" in df.columns
        and isinstance(df["Ticker"].iloc[0], str)
        and df["Ticker"].iloc[0].lower() == "date"
    ):
        df = df.iloc[1:].reset_index(drop=True)
    date_col = None
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
    # fallback: first 6 numeric columns
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

    # Load SPY (Yahoo schema)
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

    # Merge external signals (optional)
    if ext_dir and ext_dir.exists():
        for csv in sorted(ext_dir.glob("*.csv")):
            try:
                x = pd.read_csv(csv, low_memory=False)
                # detect date col
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

    # features
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

    def trange(h, l, c):
        pc = c.shift(1)
        return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

    def atr(h, l, c, w=14):
        return trange(h, l, c).ewm(alpha=1 / w, adjust=False).mean()

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

    # label
    H = int(args.horizon)
    fwd_ret = close.shift(-H) / close - 1.0
    y = (fwd_ret >= float(args.min_return)).astype("Int64")
    y.iloc[-H:] = pd.NA

    # assemble
    out = pd.DataFrame(index=feats.index)
    out["date"] = out.index
    out["ticker"] = args.ticker
    for c in keep:
        out[c.lower().replace(" ", "_")] = price_df[c] if c in price_df.columns else base[c]
    out = out.join(feats)
    out["y_up_fwd"] = y

    # time split
    n = len(out)
    ntr = int(n * 0.7)
    nval = int(n * 0.15)
    split = pd.Series(index=out.index, dtype="string")
    split.iloc[:ntr] = "train"
    split.iloc[ntr : ntr + nval] = "val"
    split.iloc[ntr + nval :] = "test"
    out["split"] = split

    # drop NaNs/unlabeled
    feat_cols = [c for c in out.columns if c not in ["date", "ticker", "y_up_fwd", "split"]]
    mask = out[feat_cols].notna().all(axis=1) & out["y_up_fwd"].notna()
    out = out.loc[mask].copy()
    out = out[["date", "ticker"] + feat_cols + ["y_up_fwd", "split"]]

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d")
    base = f"neurovest_features_{stamp}"
    pq = out_dir / f"{base}.parquet"
    csv = out_dir / f"{base}.csv"
    fl = out_dir / f"{base}.features.txt"
    meta = out_dir / f"{base}.meta.json"

    wrote_parquet = False
    try:
        out.to_parquet(pq, index=False)  # requires pyarrow/fastparquet
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
    Run: sanitize → (bootstrap|update) → labels → train → predict → (opt backtest or fast) → eval → tearsheet
    """
    _ensure_dirs()

    # 1) sanitize
    if not args.no_sanitize:
        cmd_sanitize(args)  # reuse sanitizer

    # 2) data (bootstrap or update)
    if args.bootstrap:
        ns = argparse.Namespace(action="bootstrap")
    else:
        ns = argparse.Namespace(action="update")
    cmd_data(ns)

    # 3) labels
    cmd_labels(ns)

    # 4) train
    cmd_train(ns)

    # 5) predict
    cmd_predict(ns)

    # 6) backtest
    rest = args.rest or []
    if args.fast:
        # quick run without optimizer; add some sane defaults unless user overrides
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
        # optimizer on + apply-best; user flags appended after --
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

    # 7) eval
    cmd_eval(ns)

    # 8) tearsheet
    cmd_tearsheet(ns)


# ---------- parser ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NeuroVest one-file CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # data
    dp = sub.add_parser("data", help="data utilities")
    dp_sub = dp.add_subparsers(dest="action", required=True)
    dp_sub.add_parser("update", help="run update_spy_data.py")
    dp_sub.add_parser("bootstrap", help="download SPY/HYG/LQD/UUP/TNX/DXY via yfinance")
    dp.set_defaults(func=cmd_data)

    # labels/train/predict
    sub.add_parser("labels", help="run generate_labels.py").set_defaults(func=cmd_labels)
    sub.add_parser("train", help="run train_from_labels.py").set_defaults(func=cmd_train)
    sub.add_parser("predict", help="run predict.py").set_defaults(func=cmd_predict)

    # backtest (pass-through)
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

    # sanitize
    sub.add_parser("sanitize", help="repair messy data/*.csv").set_defaults(func=cmd_sanitize)

    # features
    fp = sub.add_parser("features", help="materialize single feature table")
    fp.add_argument("--price-csv", default="data/SPY.csv")
    fp.add_argument("--external-dir", default="data_cache")
    fp.add_argument("--horizon", type=int, default=5)
    fp.add_argument("--min-return", type=float, default=0.0)
    fp.add_argument("--out-dir", default="data/features")
    fp.add_argument("--ticker", default="SPY")
    fp.set_defaults(func=cmd_features)

    # all-in-one
    ap = sub.add_parser(
        "all",
        help="run the whole pipeline (sanitize → data → labels → train → predict → backtest → eval → tearsheet)",
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
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="flags after -- passed to backtest.py")
    ap.set_defaults(func=cmd_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
