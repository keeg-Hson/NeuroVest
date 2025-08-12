#!/usr/bin/env python3
"""
Unified pipeline runner for the Market Trading Bot.

This script orchestrates the entire daily workflow end-to-end without relying on
shelling out to `python` for sub-scripts. It imports modules directly when
possible, falls back to subprocess execution if needed, and produces clear logs
+ artifacts for each step.

Steps (toggle-able via CLI flags):
  1) Refresh data sources (SPY + external signals)
  2) Select top signals (correlation-based ranking)
  3) Train model(s) with time-series CV, scaling, early stopping
  4) Analyze signals (correlation heatmap, trend plots)
  5) Run live predictions + log results
  6) Backtest and export performance visualizations

Usage examples:
  python run_all.py --all
  python run_all.py --skip-refresh --models xgb rf --backtest-window 365
  python run_all.py --predict-only

Notes:
- The script tries to call functions in your local modules if they exist. For
  example, it searches these by default:
    - select_top_signals: main(), run(), select()
    - train: train_model(), main()
    - analyze_signals: generate_plots(), main()
    - predict: run_predictions(), main()
    - backtest: run_backtest(), main()
    - external_signals: refresh_all(), update_all(), main()
    - utils: update_spy_data(), refresh_prices()
- If a function name differs in your repo, either:
    a) pass --use-subprocess to execute the module as a script, or
    b) add a thin wrapper in the respective module exporting the expected name.

Outputs:
- Logs saved under logs/run_all_YYYYmmdd_HHMMSS.log
- Plots/figures saved by their respective modules (e.g., /graphs or /logs)

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import inspect


# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _make_logger() -> logging.Logger:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_all_{ts}.log"

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
# Utilities
# --------------------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    extra: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        LOGGER.warning("Could not import %s: %s", module_name, e)
        return None


def _find_callable(mod, candidates: Iterable[str]) -> Optional[Callable]:
    if mod is None:
        return None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _call_subprocess(py_module: str, args: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Fallback: execute a python module as a script using sys.executable -m."""
    args = args or []
    cmd = [sys.executable, "-m", py_module] + args
    try:
        LOGGER.info("Subprocess: %s", " ".join(cmd))
        out = subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
        if out.stdout:
            LOGGER.info(out.stdout.strip())
        if out.stderr:
            LOGGER.warning(out.stderr.strip())
        return True, out.stdout
    except subprocess.CalledProcessError as e:
        LOGGER.error("Subprocess failed: %s", e)
        if e.stdout:
            LOGGER.error(e.stdout)
        if e.stderr:
            LOGGER.error(e.stderr)
        return False, str(e)


# --------------------------------------------------------------------------------------
# Pipeline Steps
# --------------------------------------------------------------------------------------

def step_refresh_data(use_subprocess: bool = False) -> StepResult:
    start = time.time()
    name = "Refresh data (prices + external signals)"
    try:
        # Try utils price updater if present
        utils_mod = _import_module("utils")
        updated_prices = False
        if utils_mod:
            update_fn = _find_callable(utils_mod, ["update_spy_data", "refresh_prices", "update_yfinance_data"]) 
            if update_fn:
                LOGGER.info("Updating price data via utils.%s()", update_fn.__name__)
                update_fn()
                updated_prices = True

        # External signals
        ext_mod = _import_module("external_signals")
        updated_signals = False
        if ext_mod:
            ext_fn = _find_callable(ext_mod, ["refresh_all", "update_all", "main"]) 
            if ext_fn and not use_subprocess:
                LOGGER.info("Refreshing external signals via external_signals.%s()", ext_fn.__name__)
                ext_fn()
                updated_signals = True
            elif use_subprocess:
                ok, _ = _call_subprocess("external_signals")
                updated_signals = ok

        ok = updated_prices or updated_signals
        return StepResult(name, ok=ok, seconds=time.time() - start, extra={"prices": updated_prices, "signals": updated_signals})
    except Exception as e:
        LOGGER.exception("%s failed", name)
        
    if not (updated_prices or updated_signals):
        # Try legacy script as a fallback
        LOGGER.info("No refresh functions found; falling back to update_spy_data module")
        ok, _ = _call_subprocess("update_spy_data")
        updated_prices = updated_prices or ok

        return StepResult(name, ok=False, seconds=time.time() - start, error=str(e))


def step_select_top_signals(use_subprocess: bool = False) -> StepResult:
    start = time.time()
    name = "Select top signals"
    try:
        mod = _import_module("select_top_signals")
        if mod and not use_subprocess:
            fn = _find_callable(mod, ["main", "run", "select"])
            if fn:
                LOGGER.info("Selecting top signals via select_top_signals.%s()", fn.__name__)
                result = fn()
                return StepResult(name, ok=True, seconds=time.time() - start, extra={"result": str(result)[:300]})
        # Fallback
        ok, out = _call_subprocess("select_top_signals")
        return StepResult(name, ok=ok, seconds=time.time() - start, extra={"stdout": out})
    except Exception as e:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=str(e))


def step_train(models: List[str], use_subprocess: bool = False, fast: bool = False) -> StepResult:
    start = time.time()
    name = f"Train models ({', '.join(models)})"
    try:
        mod = _import_module("train")
        payload = {"models": models, "fast": fast}
        if mod and not use_subprocess:
            fn = _find_callable(mod, ["train_model", "main", "run"])
            if fn:
                LOGGER.info("Training via train.%s(models=%s, fast=%s)", fn.__name__, models, fast)
                result = fn(models=models, fast=fast) if fn.__code__.co_argcount else fn()
                return StepResult(name, ok=True, seconds=time.time() - start, extra={"result": str(result)[:300]})
        # Fallback
        args = ["--models", *models]
        if fast:
            args.append("--fast")
        ok, out = _call_subprocess("train", args)
        return StepResult(name, ok=ok, seconds=time.time() - start, extra={"stdout": out, **payload})
    except Exception as e:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=str(e))


def step_analyze_signals(use_subprocess: bool = False) -> StepResult:
    start = time.time()
    name = "Analyze signals (plots)"
    try:
        mod = _import_module("analyze_signals")
        if mod and not use_subprocess:
            fn = _find_callable(mod, ["generate_plots", "main", "run"])
            if fn:
                LOGGER.info("Generating analysis via analyze_signals.%s()", fn.__name__)
                result = fn()
                return StepResult(name, ok=True, seconds=time.time() - start, extra={"result": str(result)[:300]})
        ok, out = _call_subprocess("analyze_signals")
        return StepResult(name, ok=ok, seconds=time.time() - start, extra={"stdout": out})
    except Exception as e:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=str(e))


def step_predict(use_subprocess: bool = False) -> StepResult:
    start = time.time()
    name = "Run live predictions"
    try:
        mod = _import_module("predict")
        if mod and not use_subprocess:
            fn = _find_callable(mod, ["run_predictions", "main", "run"])
            if fn:
                LOGGER.info("Running predictions via predict.%s()", fn.__name__)
                result = fn()
                return StepResult(name, ok=True, seconds=time.time() - start, extra={"result": str(result)[:300]})
        ok, out = _call_subprocess("predict")
        return StepResult(name, ok=ok, seconds=time.time() - start, extra={"stdout": out})
    except Exception as e:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=str(e))


def step_backtest(window_days: Optional[int], use_subprocess: bool = False) -> StepResult:
    start = time.time()
    name = "Backtest"
    try:
        mod = _import_module("backtest")
        if mod and not use_subprocess:
            fn = _find_callable(mod, ["run_backtest", "main", "run"])
            if fn:
                LOGGER.info("Backtesting via backtest.%s(window_days=%s)", fn.__name__, window_days)
                sig = inspect.signature(fn)
                if "window_days" in sig.parameters and window_days is not None:
                    result = fn(window_days=window_days)
                else:
                    result = fn()  # no kwargs if not supported
                return StepResult(name, ok=True, seconds=time.time() - start, extra={"result": str(result)[:300]})
        # Fallback
        args = ["--window-days", str(window_days)] if window_days else []
        ok, out = _call_subprocess("backtest", args)
        return StepResult(name, ok=ok, seconds=time.time() - start, extra={"stdout": out, "window_days": window_days})
    except Exception as e:
        LOGGER.exception("%s failed", name)
        return StepResult(name, ok=False, seconds=time.time() - start, error=str(e))


# --------------------------------------------------------------------------------------
# CLI + Main
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Market Trading Bot pipeline.")

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--all", action="store_true", help="Run all steps (default).")
    group.add_argument("--predict-only", action="store_true", help="Only run live predictions.")

    p.add_argument("--skip-refresh", action="store_true", help="Skip refreshing data sources.")
    p.add_argument("--skip-train", action="store_true", help="Skip training.")
    p.add_argument("--skip-analyze", action="store_true", help="Skip signal analysis plots.")
    p.add_argument("--skip-backtest", action="store_true", help="Skip backtest.")

    p.add_argument("--models", nargs="+", default=["xgb"], help="Models to train: xgb lgbm rf (space-separated).")
    p.add_argument("--fast", action="store_true", help="Faster runs (e.g., fewer CV folds, smaller grids).")
    p.add_argument("--use-subprocess", action="store_true", help="Invoke submodules via subprocess instead of import.")
    p.add_argument("--backtest-window", type=int, default=None, help="Limit backtest to N most recent days.")

    args = p.parse_args(argv)
    if not (args.all or args.predict_only):
        args.all = True  # default to all
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    LOGGER.info("Run config: %s", json.dumps(vars(args), indent=2))

    results: List[StepResult] = []

    # 1) Refresh data
    if args.all and not args.skip_refresh and not args.predict_only:
        results.append(step_refresh_data(use_subprocess=args.use_subprocess))

    # 2) Select top signals
    if args.all and not args.predict_only:
        results.append(step_select_top_signals(use_subprocess=args.use_subprocess))

    # 3) Train (scaling + TS CV + early stopping assumed inside train.py)
    if args.all and not args.skip_train and not args.predict_only:
        results.append(step_train(models=args.models, use_subprocess=args.use_subprocess, fast=args.fast))

    # 4) Analyze signals
    if args.all and not args.skip_analyze and not args.predict_only:
        results.append(step_analyze_signals(use_subprocess=args.use_subprocess))

    # 5) Predict (always safe to run)
    results.append(step_predict(use_subprocess=args.use_subprocess))

    # 6) Backtest
    if (args.all and not args.skip_backtest) and not args.predict_only:
        results.append(step_backtest(window_days=args.backtest_window, use_subprocess=args.use_subprocess))

    # Summary
    LOGGER.info("\n==== PIPELINE SUMMARY ====")
    for r in results:
        status = "✅" if r.ok else "❌"
        LOGGER.info("%s %s (%.1fs)", status, r.name, r.seconds)
        if r.error:
            LOGGER.info("   Error: %s", r.error)
        if r.extra:
            LOGGER.info("   Extra: %s", {k: (str(v)[:120] + ("…" if len(str(v)) > 120 else "")) for k, v in r.extra.items()})

    all_ok = all(r.ok for r in results)
    LOGGER.info("%s All pipeline steps complete.", "✅" if all_ok else "⚠️")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
