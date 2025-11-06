#!/usr/bin/env python3
"""
Unified pipeline runner for NeuroVest.

What it does (toggle-able via CLI flags):
  1) Refresh data sources (best-effort; optional modules)
  2) Select/rank signals (optional module)
  3) Train model(s) via train_from_labels.py (with X/y alignment)
     - Also writes models/split_meta.json (OOS start date)
  4) Backfill predictions (predict.py) → logs/labeled_predictions.csv
     - Ensures models/thresholds.json exists (seed from config if present)
  5) Backtest (backtest_module.py) with OOS-only option
  6) Generate HTML tearsheet (tearsheet.py)

Key repo expectations (with fallbacks):
  - config.py (optional) defining LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, SPY_DAILY_CSV, PREDICT_CFG
  - train_from_labels.py with train_model() or main()
  - predict.py with main() (current repo prints/writes labeled_predictions.csv)
  - backtest_module.py with run_backtest() or main()
  - tearsheet.py with main() that emits outputs/tearsheet_YYYY-MM-DD.html

Examples:
  python run_all.py --all
  python run_all.py --predict-only
  python run_all.py --skip-refresh --oos-only
  python run_all.py --backtest-window 365 --auto-thresholds (no-op unless your backtest supports it)

Notes:
- We import modules if possible for speed and cleaner logs; otherwise we fall back
  to running the script file with subprocess.
- We attempt to write OOS split metadata if the training script doesn’t.
- We silently create models/thresholds.json if it’s missing using PREDICT_CFG defaults or
  (p_min=0.55, ev_min=0.0005).

Outputs:
- Orchestrator log: logs/run_all_YYYYmmdd_HHMMSS.log
- Artifacts are produced by the respective modules as they do today.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Paths & logging
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Try to import config for canonical paths; otherwise, fall back to sensible defaults.
CONFIG = None
try:
    CONFIG = importlib.import_module("config")
except Exception:
    CONFIG = None

LOGS_DIR = Path(getattr(CONFIG, "LOGS_DIR", ROOT / "logs"))
MODELS_DIR = Path(getattr(CONFIG, "MODELS_DIR", ROOT / "models"))
OUTPUTS_DIR = Path(getattr(CONFIG, "OUTPUTS_DIR", ROOT / "outputs"))
SPY_DAILY_CSV = Path(getattr(CONFIG, "SPY_DAILY_CSV", ROOT / "data" / "SPY_daily.csv"))
PREDICT_CFG = getattr(CONFIG, "PREDICT_CFG", {"p_min": 0.55, "ev_min": 0.0005})

for d in (LOGS_DIR, MODELS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _make_logger() -> logging.Logger:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_all_{ts}.log"

    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logging to %s", log_path)
    return logger


LOGGER = _make_logger()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    extra: dict[str, Any] | None = None
    error: str | None = None


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        LOGGER.debug("Import fail for %s: %s", module_name, e)
        return None


def _find_callable(mod, candidates: list[str]) -> Callable | None:
    if mod is None:
        return None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _call_script(py_file: Path, args: list[str] | None = None) -> tuple[bool, str, str]:
    """Run a Python file via subprocess, return (ok, stdout, stderr)."""
    args = args or []
    cmd = [sys.executable, str(py_file), *args]
    try:
        LOGGER.info("Subprocess: %s", " ".join(cmd))
        out = subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
        if out.stdout:
            for line in out.stdout.rstrip().splitlines():
                LOGGER.info(line)
        if out.stderr:
            for line in out.stderr.rstrip().splitlines():
                LOGGER.warning(line)
        return True, out.stdout, out.stderr
    except subprocess.CalledProcessError as e:
        LOGGER.error("Script failed: %s", e)
        if e.stdout:
            LOGGER.error(e.stdout)
        if e.stderr:
            LOGGER.error(e.stderr)
        return False, e.stdout, e.stderr


def _ensure_thresholds_json() -> None:
    """Create models/thresholds.json if it doesn't exist, using PREDICT_CFG or defaults."""
    f = MODELS_DIR / "thresholds.json"
    if not f.exists():
        payload = {
            "p_min": float(PREDICT_CFG.get("p_min", 0.55)),
            "ev_min": float(PREDICT_CFG.get("ev_min", 0.0005)),
        }
        f.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Seeded %s with %s", f, payload)


def _write_split_meta_from_training(full_csv: Path, ratio: float = 0.75) -> Path | None:
    """
    Best-effort: if models/split_meta.json missing, derive OOS split from SPY CSV length.
    """
    meta = MODELS_DIR / "split_meta.json"
    if meta.exists():
        try:
            json.loads(meta.read_text())
            return meta
        except Exception:
            pass  # rewrite

    try:
        import pandas as pd  # local import for speed on non-pandas steps

        df = pd.read_csv(full_csv, parse_dates=["Date"], low_memory=False)
        if len(df) < 10:
            LOGGER.warning("Too few rows to infer split meta.")
            return None
        split_idx = max(1, int(len(df) * ratio))
        split_date = pd.to_datetime(df.loc[split_idx, "Date"])
        meta.write_text(json.dumps({"split_index": int(split_idx), "split_date": str(split_date)}))
        LOGGER.info("[train] wrote OOS split meta → %s (start=%s)", meta, split_date.date())
        return meta
    except Exception as e:
        LOGGER.warning("Could not infer split meta: %s", e)
        return None


def _limit_df_oos(csv_path: Path, out_path: Path | None = None) -> tuple[bool, int]:
    """
    Filter logs/labeled_predictions.csv to OOS based on models/split_meta.json.
    If out_path provided, write filtered CSV; return (ok, rows_kept).
    """
    meta = MODELS_DIR / "split_meta.json"
    if not meta.exists():
        LOGGER.info("[eval] split_meta.json not found; skipping OOS filter.")
        return False, 0

    try:
        meta_j = json.loads(meta.read_text())
        split_date = dt.datetime.fromisoformat(meta_j["split_date"]).date()

        import pandas as pd

        df = pd.read_csv(csv_path, parse_dates=["Date"], low_memory=False)
        before = len(df)
        df = df[df["Date"].dt.date >= split_date].copy()
        kept = len(df)
        if out_path:
            df.to_csv(out_path, index=False)
        LOGGER.info("[eval] OOS filter: kept %d/%d rows from %s+", kept, before, split_date)
        return True, kept
    except Exception as e:
        LOGGER.warning("OOS filter failed: %s", e)
        return False, 0


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------
def step_refresh_data() -> StepResult:
    start = time.time()
    name = "Refresh data (optional)"
    updated_prices = False
    updated_signals = False

    try:
        # utils.* price updaters (best-effort)
        utils_mod = _import_module("utils")
        if utils_mod:
            fn = _find_callable(
                utils_mod, ["update_spy_data", "refresh_prices", "update_yfinance_data"]
            )
            if fn:
                LOGGER.info("utils.%s()", fn.__name__)
                try:
                    fn()
                    updated_prices = True
                except Exception as e:
                    LOGGER.warning("utils %s failed: %s", fn.__name__, e)

        # external_signals.* (best-effort)
        ext_mod = _import_module("external_signals")
        if ext_mod:
            fn = _find_callable(ext_mod, ["refresh_all", "update_all", "main"])
            if fn:
                LOGGER.info("external_signals.%s()", fn.__name__)
                try:
                    fn()
                    updated_signals = True
                except Exception as e:
                    LOGGER.warning("external_signals %s failed: %s", fn.__name__, e)

        ok = updated_prices or updated_signals
        return StepResult(
            name,
            ok=ok,
            seconds=time.time() - start,
            extra={"prices": updated_prices, "signals": updated_signals},
        )
    except Exception:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=traceback.format_exc())


def step_select_top_signals() -> StepResult:
    start = time.time()
    name = "Select top signals (optional)"
    try:
        mod = _import_module("select_top_signals")
        if mod:
            fn = _find_callable(mod, ["main", "run", "select"])
            if fn:
                LOGGER.info("select_top_signals.%s()", fn.__name__)
                try:
                    res = fn()
                    return StepResult(
                        name, ok=True, seconds=time.time() - start, extra={"result": str(res)[:300]}
                    )
                except Exception as e:
                    LOGGER.warning("select_top_signals failed: %s", e)
        LOGGER.info("select_top_signals not present; skipping.")
        return StepResult(name, ok=True, seconds=time.time() - start, extra={"skipped": True})
    except Exception:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=traceback.format_exc())


def step_train(oos_ratio: float) -> StepResult:
    start = time.time()
    name = "Train model(s)"
    tf = ROOT / "train_from_labels.py"
    try:
        mod = _import_module("train_from_labels")
        if mod:
            fn = _find_callable(mod, ["train_model", "main", "run"])
            if fn:
                LOGGER.info("train_from_labels.%s()", fn.__name__)
                try:
                    # Support both zero-arg and kw variants
                    result = fn() if fn.__code__.co_argcount == 0 else fn(**{})
                except SystemExit:
                    # Some scripts call argparse+exit; treat as success.
                    result = "SystemExit(0)"
                except Exception as e:
                    LOGGER.warning(
                        "train_from_labels import-run failed: %s; falling back to script", e
                    )
                    ok, out, err = _call_script(tf)
                    if not ok:
                        raise RuntimeError("train_from_labels script failed")
                # Write OOS meta if missing
                _write_split_meta_from_training(SPY_DAILY_CSV, ratio=oos_ratio)
                return StepResult(
                    name, ok=True, seconds=time.time() - start, extra={"result": str(result)[:300]}
                )
        # Fallback to script
        ok, out, err = _call_script(tf)
        if ok:
            _write_split_meta_from_training(SPY_DAILY_CSV, ratio=oos_ratio)
        return StepResult(
            name, ok=ok, seconds=time.time() - start, extra={"stdout": (out or "")[:500]}
        )
    except Exception:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=traceback.format_exc())


def step_backfill_predictions() -> StepResult:
    start = time.time()
    name = "Backfill/live predict → labeled_predictions.csv"
    pf = ROOT / "predict.py"
    try:
        _ensure_thresholds_json()
        mod = _import_module("predict")
        if mod:
            fn = _find_callable(mod, ["main", "run", "run_predictions"])
            if fn:
                LOGGER.info("predict.%s()", fn.__name__)
                try:
                    result = fn() if fn.__code__.co_argcount == 0 else fn(**{})
                except SystemExit:
                    result = "SystemExit(0)"
                except Exception as e:
                    LOGGER.warning("predict import-run failed: %s; falling back to script", e)
                    ok, out, err = _call_script(pf)
                    if not ok:
                        raise RuntimeError("predict script failed")
                # Confirm file exists
                out_csv = LOGS_DIR / "labeled_predictions.csv"
                exists = out_csv.exists()
                return StepResult(
                    name,
                    ok=exists,
                    seconds=time.time() - start,
                    extra={"labeled_predictions": str(out_csv) if exists else "missing"},
                )
        ok, out, err = _call_script(pf)
        out_csv = LOGS_DIR / "labeled_predictions.csv"
        return StepResult(
            name,
            ok=out_csv.exists(),
            seconds=time.time() - start,
            extra={"labeled_predictions": str(out_csv) if out_csv.exists() else "missing"},
        )
    except Exception:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=traceback.format_exc())


def step_backtest(window_days: int | None, oos_only: bool) -> StepResult:
    start = time.time()
    name = "Backtest"
    bf = ROOT / "backtest_module.py"
    try:
        # Optionally filter predictions to OOS before running the backtest script (non-invasive)
        labeled = LOGS_DIR / "labeled_predictions.csv"
        filtered = LOGS_DIR / "labeled_predictions_oos.csv"
        if oos_only and labeled.exists():
            ok_oos, kept = _limit_df_oos(labeled, out_path=filtered)
            if ok_oos and kept > 0:
                os.environ["NEUROVEST_PREDICTIONS_CSV"] = str(filtered)

        mod = _import_module("backtest_module")
        args_for_script: list[str] = []
        if window_days:
            os.environ["NEUROVEST_BACKTEST_WINDOW_DAYS"] = str(window_days)

        if mod:
            fn = _find_callable(mod, ["run_backtest", "main", "run"])
            if fn:
                LOGGER.info("backtest_module.%s()", fn.__name__)
                try:
                    result = fn() if fn.__code__.co_argcount == 0 else fn(**{})
                except SystemExit:
                    result = "SystemExit(0)"
                except Exception as e:
                    LOGGER.warning(
                        "backtest_module import-run failed: %s; falling back to script", e
                    )
                    ok, out, err = _call_script(bf, args_for_script)
                    if not ok:
                        raise RuntimeError("backtest_module script failed")
                return StepResult(
                    name, ok=True, seconds=time.time() - start, extra={"window_days": window_days}
                )
        ok, out, err = _call_script(bf, args_for_script)
        return StepResult(
            name, ok=ok, seconds=time.time() - start, extra={"window_days": window_days}
        )
    except Exception:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=traceback.format_exc())


def step_tearsheet() -> StepResult:
    start = time.time()
    name = "Generate HTML tearsheet"
    tf = ROOT / "tearsheet.py"
    try:
        mod = _import_module("tearsheet")
        if mod:
            fn = _find_callable(mod, ["main", "run", "generate"])
            if fn:
                LOGGER.info("tearsheet.%s()", fn.__name__)
                try:
                    result = fn() if fn.__code__.co_argcount == 0 else fn(**{})
                except SystemExit:
                    result = "SystemExit(0)"
                except Exception as e:
                    LOGGER.warning("tearsheet import-run failed: %s; falling back to script", e)
                    ok, out, err = _call_script(tf)
                    if not ok:
                        raise RuntimeError("tearsheet script failed")

        # Find the latest generated HTML in outputs/
        latest = None
        try:
            cands = sorted(OUTPUTS_DIR.glob("tearsheet_*.html"))
            latest = str(cands[-1]) if cands else None
        except Exception:
            pass

        return StepResult(
            name,
            ok=bool(latest),
            seconds=time.time() - start,
            extra={"file": latest or "not found"},
        )
    except Exception:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=traceback.format_exc())


# --------------------------------------------------------------------------------------
# CLI & main
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full NeuroVest pipeline.")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--all", action="store_true", help="Run all steps (default).")
    group.add_argument("--predict-only", action="store_true", help="Only run prediction stage.")

    p.add_argument("--skip-refresh", action="store_true", help="Skip refreshing data sources.")
    p.add_argument("--skip-select", action="store_true", help="Skip signal selection.")
    p.add_argument("--skip-train", action="store_true", help="Skip model training.")
    p.add_argument("--skip-backtest", action="store_true", help="Skip backtest.")
    p.add_argument("--skip-tearsheet", action="store_true", help="Skip tearsheet generation.")

    p.add_argument(
        "--backtest-window", type=int, default=None, help="Limit backtest to last N days."
    )
    p.add_argument("--oos-only", action="store_true", help="Evaluate backtest on OOS only.")
    p.add_argument(
        "--oos-ratio",
        type=float,
        default=0.75,
        help="If split meta missing, use this train ratio to infer OOS.",
    )

    args = p.parse_args(argv)
    if not (args.all or args.predict_only):
        args.all = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    LOGGER.info("Run config: %s", json.dumps(vars(args), indent=2, default=str))

    results: list[StepResult] = []

    # 1) Refresh (optional)
    if args.all and not args.skip_refresh and not args.predict_only:
        results.append(step_refresh_data())

    # 2) Select signals (optional)
    if args.all and not args.skip_select and not args.predict_only:
        results.append(step_select_top_signals())

    # 3) Train
    if args.all and not args.skip_train and not args.predict_only:
        results.append(step_train(oos_ratio=args.oos_ratio))

    # 4) Predict/backfill (always run unless predict-only toggled)
    results.append(step_backfill_predictions())

    # 5) Backtest
    if (args.all and not args.skip_backtest) and not args.predict_only:
        results.append(step_backtest(window_days=args.backtest_window, oos_only=args.oos_only))

    # 6) Tearsheet
    if args.all and not args.skip_tearsheet and not args.predict_only:
        results.append(step_tearsheet())

    # Summary
    LOGGER.info("\n==== PIPELINE SUMMARY ====")
    for r in results:
        status = "✅" if r.ok else "❌"
        LOGGER.info("%s %s (%.1fs)", status, r.name, r.seconds)
        if r.error:
            LOGGER.info("   Error: %s", r.error.strip().splitlines()[-1] if r.error else r.error)
        if r.extra:
            trimmed = {
                k: (str(v)[:140] + ("…" if len(str(v)) > 140 else "")) for k, v in r.extra.items()
            }
            LOGGER.info("   Extra: %s", trimmed)

    all_ok = all(r.ok for r in results)
    LOGGER.info("%s Pipeline complete.", "✅" if all_ok else "⚠️")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
