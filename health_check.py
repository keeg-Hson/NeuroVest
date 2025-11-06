#!/usr/bin/env python3
"""
health_check.py — End-to-end system health audit for NeuroVest.

What it checks (with file-wise citations wherever issues are found):
  1) Config & filesystem: required files/dirs exist and load.
  2) Data sanity: SPY csv readability, Date parsing, required columns, dtypes.
  3) Model integrity: model file exists, joblib loads, schema (features) present.
  4) Predict pipeline: import predict.py, run live_predict() safely (no GUI),
     verify labeled_predictions.csv shape (Date/Pred/Prediction/Proba/Label).
  5) Backtest readiness: import backtest_module, dry-run merge/eval only (no UI).
  6) Log forensics: parse logs/* for recent Tracebacks, cite file:line locations.
  7) Code hygiene: scan .py files for tabs, mixed indentation, BOM, CRLF, and
     warn about patterns that previously caused errors (e.g., 'True_Label' only).
  8) Deprecations/FutureWarnings likely: looks for pct_change(...), low_memory, etc.

Usage:
  python health_check.py                   # pretty text summary
  python health_check.py --json report.json  # JSON report
  python health_check.py --no-run            # skip import/execution checks
  python health_check.py --quick             # fewer, faster checks

Exit code: 0 if all critical checks passed; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make stdout unbuffered for CI
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
CONFIG_IMPORT = "config"


# Soft deps inside checks
def _try_import_config() -> tuple[Any | None, list[str]]:
    issues = []
    try:
        cfg = __import__(CONFIG_IMPORT, fromlist=["*"])
        return cfg, issues
    except Exception as e:
        issues.append(f"Cannot import config.py: {e}")
        return None, issues


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]
    citations: list[str]  # "path[:line]" or just "path" for file-wise reference


@dataclass
class Report:
    ok: bool
    results: list[CheckResult]

    def to_dict(self):
        return {
            "ok": self.ok,
            "results": [asdict(r) for r in self.results],
        }


def _status(ok: bool) -> str:
    return "✅" if ok else "❌"


# -----------------------------------------------------------------------------
# 1) Config & filesystem
# -----------------------------------------------------------------------------
def check_config_files() -> CheckResult:
    details, cites = [], []
    ok = True

    cfg, cfg_issues = _try_import_config()
    if cfg is None:
        ok = False
        details += cfg_issues
        cites.append("config.py")
        return CheckResult("Config import", ok, details, citations=cites)

    # Required attributes
    required = ["SPY_DAILY_CSV", "LOGS_DIR", "MODELS_DIR"]
    for attr in required:
        if not hasattr(cfg, attr):
            ok = False
            details.append(f"Missing config.{attr}")
            cites.append("config.py")
        else:
            p = Path(getattr(cfg, attr))
            if attr.endswith("_DIR"):
                if not p.exists():
                    details.append(f"{attr} directory missing: {p}")
                    cites.append(str(p))
                elif not p.is_dir():
                    ok = False
                    details.append(f"{attr} is not a directory: {p}")
                    cites.append(str(p))
            else:
                if not p.exists():
                    ok = False
                    details.append(f"{attr} file not found: {p}")
                    cites.append(str(p))

    return CheckResult("Config & filesystem", ok, details, citations=cites or ["config.py"])


# -----------------------------------------------------------------------------
# 2) Data sanity (SPY)
# -----------------------------------------------------------------------------
def check_spy_csv(quick: bool) -> CheckResult:
    details, cites = [], []
    ok = True

    cfg, _ = _try_import_config()
    if cfg is None:
        return CheckResult("SPY CSV", False, ["config import failed"], ["config.py"])

    spy_path = Path(cfg.SPY_DAILY_CSV)
    cites.append(str(spy_path))
    if not spy_path.exists():
        return CheckResult("SPY CSV", False, [f"Missing: {spy_path}"], cites)

    import pandas as pd

    try:
        # low_memory=False to mirror training behavior and avoid split dtypes
        df = pd.read_csv(spy_path, low_memory=False)
        if "Date" not in df.columns:
            ok = False
            details.append("Missing 'Date' column")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            bad = df["Date"].isna().sum()
            if bad > 0:
                ok = False
                details.append(f"Unparseable Date rows: {bad}")
        # Check price columns
        close_like = [c for c in df.columns if str(c).lower() in ("close", "adjclose", "adj close")]
        if not close_like:
            close_like = [c for c in df.columns if "close" in str(c).lower()]
        if not close_like:
            ok = False
            details.append("No Close/AdjClose column found")

        if not quick:
            # Monotonic dates & duplicates
            sorted_ok = df["Date"].is_monotonic_increasing
            if not sorted_ok:
                details.append("Date column is not sorted ascending")
            dups = int(df["Date"].duplicated().sum())
            if dups:
                details.append(f"Duplicate Date rows: {dups}")
            ok = ok and sorted_ok and dups == 0

    except Exception as e:
        ok = False
        details.append(f"Read failed: {e}")

    return CheckResult("Data sanity (SPY)", ok, details, citations=cites)


# -----------------------------------------------------------------------------
# 3) Model integrity
# -----------------------------------------------------------------------------
def check_model() -> CheckResult:
    details, cites = [], []
    ok = True

    cfg, _ = _try_import_config()
    if cfg is None:
        return CheckResult("Model", False, ["config import failed"], ["config.py"])

    model_path = Path(cfg.MODELS_DIR) / "market_crash_model.pkl"
    cites.append(str(model_path))
    if not model_path.exists():
        return CheckResult("Model", False, [f"Missing model: {model_path}"], cites)

    try:
        import joblib

        obj = joblib.load(model_path)
        if isinstance(obj, dict):
            if "model" not in obj:
                ok = False
                details.append("Model dict missing key 'model'")
            feats = obj.get("features", [])
            if not feats:
                details.append(
                    "Model dict has empty or missing 'features' (will limit schema alignment)"
                )
        else:
            details.append("Legacy bare estimator loaded (no saved feature list)")
    except Exception as e:
        ok = False
        details.append(f"Joblib load failed: {e}")

    # Split meta (OOS boundary)
    split_meta = MODELS / "split_meta.json"
    cites.append(str(split_meta))
    if split_meta.exists():
        try:
            meta = json.loads(split_meta.read_text())
            if "split_date" not in meta:
                details.append("split_meta.json present but missing 'split_date'")
        except Exception as e:
            details.append(f"split_meta.json unreadable: {e}")

    return CheckResult("Model integrity", ok, details, citations=cites)


# -----------------------------------------------------------------------------
# 4) Predict pipeline
# -----------------------------------------------------------------------------
def check_predict(no_run: bool) -> CheckResult:
    details, cites = [], []
    ok = True

    path = ROOT / "predict.py"
    cites.append(str(path))
    if not path.exists():
        return CheckResult("Predict pipeline", False, ["predict.py not found"], cites)

    # Import & live predict
    if not no_run:
        try:
            # Ensure no GUI backends get spawned anywhere
            os.environ.setdefault("MPLBACKEND", "Agg")
            import importlib

            predict = importlib.import_module("predict")
            if not hasattr(predict, "live_predict"):
                ok = False
                details.append("predict.live_predict() not found")
            else:
                decision, p1, when = predict.live_predict()
                details.append(f"live_predict ok → {when.date()} p1={p1:.4f} decision={decision}")
        except Exception as e:
            ok = False
            tb = traceback.format_exc()
            details.append(f"live_predict failed: {e}")
            # Cite file/line locations from traceback
            for m in re.finditer(r'File "([^"]+)", line (\d+)', tb):
                cites.append(f"{m.group(1)}:{m.group(2)}")

    # labeled_predictions sanity
    cfg, _ = _try_import_config()
    if cfg:
        labels_path = Path(cfg.LOGS_DIR) / "labeled_predictions.csv"
        cites.append(str(labels_path))
        if labels_path.exists():
            import pandas as pd

            try:
                df = pd.read_csv(labels_path, parse_dates=["Date"])
                required = ["Date", "Proba"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    ok = False
                    details.append(f"labeled_predictions missing columns: {missing}")
                if "Prediction" not in df.columns and "Pred" not in df.columns:
                    ok = False
                    details.append("No Prediction/Pred column in labeled_predictions")
                # basic date quality
                if df["Date"].isna().any():
                    details.append("labeled_predictions contains NaT dates")
                if df["Date"].duplicated().any():
                    details.append("labeled_predictions contains duplicate Date rows")
            except Exception as e:
                ok = False
                details.append(f"Cannot read labeled_predictions.csv: {e}")
        else:
            details.append("labeled_predictions.csv not found (ok if first run)")

    return CheckResult("Predict pipeline", ok, details, citations=cites)


# -----------------------------------------------------------------------------
# 5) Backtest readiness (no plotting)
# -----------------------------------------------------------------------------
def check_backtest(no_run: bool) -> CheckResult:
    details, cites = [], []
    ok = True

    path = ROOT / "backtest_module.py"
    cites.append(str(path))
    if not path.exists():
        return CheckResult("Backtest readiness", False, ["backtest_module.py not found"], cites)

    if no_run:
        return CheckResult("Backtest readiness", ok, details, citations=cites)

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import pandas as pd

        # Recreate the non-GUI parts of run_backtest()
        cfg, _ = _try_import_config()
        spy = pd.read_csv(cfg.SPY_DAILY_CSV, parse_dates=["Date"]).sort_values("Date")
        preds_path = Path(cfg.LOGS_DIR) / "labeled_predictions.csv"
        if not preds_path.exists():
            details.append(
                "No labeled_predictions.csv — backtest will have nothing to evaluate yet."
            )
            return CheckResult("Backtest readiness", ok, details, citations=cites)

        preds = pd.read_csv(preds_path, parse_dates=["Date"]).sort_values("Date")
        merged = pd.merge(spy, preds, on="Date", how="inner")
        if merged.empty:
            ok = False
            details.append("Merge of SPY with predictions is empty (date mismatch?)")
        else:
            details.append(f"Merged rows: {len(merged)}")
            # Label column fallback
            label_col = (
                "True_Label"
                if "True_Label" in merged.columns
                else ("Label" if "Label" in merged.columns else None)
            )
            if label_col is None:
                details.append("No ground-truth labels present; metrics will be limited.")
            # Find a usable price column
            price_candidates = [
                "Close",
                "AdjClose",
                "Close_SPY",
                "AdjClose_SPY",
                "Close_y",
                "Close_x",
            ]
            price_col = next((c for c in price_candidates if c in merged.columns), None)
            if not price_col:
                close_like = [c for c in merged.columns if "close" in str(c).lower()]
                if close_like:
                    price_col = close_like[0]
            if not price_col:
                ok = False
                details.append("No Close/AdjClose column in merged backtest frame")
            else:
                details.append(f"Using price column: {price_col}")
    except Exception as e:
        ok = False
        tb = traceback.format_exc()
        details.append(f"Backtest dry-run failed: {e}")
        for m in re.finditer(r'File "([^"]+)", line (\d+)', tb):
            cites.append(f"{m.group(1)}:{m.group(2)}")

    return CheckResult("Backtest readiness", ok, details, citations=cites)


# -----------------------------------------------------------------------------
# 6) Log forensics — extract recent tracebacks with file:line
# -----------------------------------------------------------------------------
def check_logs_for_errors() -> CheckResult:
    details, cites = [], []
    ok = True

    if not LOGS.exists():
        return CheckResult(
            "Logs", True, ["logs/ directory not found (ok if fresh repo)"], [str(LOGS)]
        )

    # Latest N log-like files
    candidates = sorted(LOGS.glob("**/*"), key=lambda p: p.stat().st_mtime, reverse=True)[:25]
    tb_hits = 0
    for p in candidates:
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        if "Traceback (most recent call last)" in text or "Error:" in text:
            # Pull file:line frames
            for m in re.finditer(r'File "([^"]+)", line (\d+)', text):
                cites.append(f"{m.group(1)}:{m.group(2)}")
                tb_hits += 1
            # Include a short summary line
            lines = [
                ln
                for ln in text.splitlines()
                if "Error" in ln or "IndentationError" in ln or "KeyError" in ln
            ]
            for ln in lines[:5]:
                details.append(f"{p.name}: {ln.strip()}")

    if tb_hits:
        ok = False
        details.append(f"Traceback frames found: {tb_hits}")

    return CheckResult("Logs (errors/tracebacks)", ok, details, citations=cites or [str(LOGS)])


# -----------------------------------------------------------------------------
# 7) Code hygiene — tabs, mixed indentation, CRLF, suspicious patterns
# -----------------------------------------------------------------------------
SUS_PATTERNS = [
    (
        r"\bTrue_Label\b",
        "Backtest may fail if your dataframe doesn’t have True_Label; ensure fallback to Label exists.",
    ),
    (r"pct_change\(", "Prefer pct_change(..., fill_method=None) to avoid FutureWarning."),
    (
        r"read_csv\([^)]*low_memory\s*=\s*True",
        "read_csv(..., low_memory=False) recommended to avoid mixed dtypes.",
    ),
]


def check_code_hygiene(quick: bool) -> CheckResult:
    details, cites = [], []
    ok = True
    files = (
        list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
        if (ROOT / "scripts").exists()
        else list(ROOT.glob("*.py"))
    )
    for p in files:
        try:
            data = p.read_bytes()
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            continue

        file_issues = []
        if b"\xef\xbb\xbf" in data[:3]:
            file_issues.append("UTF-8 BOM present")
        if "\r\n" in text:
            file_issues.append("CRLF line endings")
        if "\t" in text:
            file_issues.append("Tab characters present (may cause mixed indentation)")
        # very rough mixed indentation sniff
        if re.search(r"^\s+\S", text, re.M):
            indents = re.findall(r"^([ \t]+)\S", text, re.M)
            if any("\t" in s for s in indents) and any("  " in s for s in indents):
                file_issues.append("Mixed tabs/spaces detected")

        # Suspicious patterns
        if not quick:
            for pat, msg in SUS_PATTERNS:
                if re.search(pat, text):
                    file_issues.append(msg)

        if file_issues:
            cites.append(str(p))
            ok = False
            details.append(f"{p.name}: " + "; ".join(sorted(set(file_issues))))

    return CheckResult("Code hygiene", ok, details, citations=cites or [str(ROOT)])


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def run(quick: bool, no_run: bool) -> Report:
    results: list[CheckResult] = []
    results.append(check_config_files())
    results.append(check_spy_csv(quick=quick))
    results.append(check_model())
    results.append(check_predict(no_run=no_run))
    results.append(check_backtest(no_run=no_run))
    results.append(check_logs_for_errors())
    results.append(check_code_hygiene(quick=quick))

    ok = all(r.ok for r in results if r.name not in {"Logs (errors/tracebacks)", "Code hygiene"})
    # Logs & hygiene are informative; don’t fail the whole run solely on those.

    return Report(ok=ok, results=results)


def _print_pretty(report: Report) -> None:
    print("\n==== NeuroVest System Health ====\n")
    for r in report.results:
        print(f"{_status(r.ok)} {r.name}")
        for d in r.details[:8]:
            print(f"   - {d}")
        if len(r.details) > 8:
            print(f"   … ({len(r.details)-8} more)")
        if r.citations:
            # unique, compact
            cites = list(dict.fromkeys(r.citations))[:6]
            print("   Files:", "; ".join(cites) + (" …" if len(r.citations) > 6 else ""))
        print()
    print(f"Overall: {_status(report.ok)}")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroVest system health checker")
    p.add_argument("--json", type=str, default=None, help="Write full JSON report to this path")
    p.add_argument("--quick", action="store_true", help="Skip slower/smarter scans")
    p.add_argument("--no-run", action="store_true", help="Do not import/execute predict/backtest")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rep = run(quick=args.quick, no_run=args.no_run)

    if args.json:
        out = ROOT / args.json
        out.write_text(json.dumps(rep.to_dict(), indent=2))
        print(f"Wrote JSON report → {out}")

    _print_pretty(rep)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
