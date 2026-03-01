#!/usr/bin/env python3
"""
CONSOLIDATED STATS DASHBOARD
============================
Single source of truth for all NeuroVest metrics.
Run: python3 consolidated_stats.py

This script consolidates metrics from:
- evaluate.py (model accuracy, AUC, classification metrics)
- generate_backtest_metrics.py (Sharpe, Sortino, drawdown)
- walk_forward_validation.py (out-of-sample performance)
- system_health.py (data/model status)
"""

import warnings
warnings.filterwarnings('ignore')

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def header(title: str, char: str = "═") -> None:
    width = 80
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def subheader(title: str) -> None:
    print(f"\n┌{'─' * 78}┐")
    print(f"│  {title:<74}  │")
    print(f"└{'─' * 78}┘")


def metric_row(name: str, value: Any, status: str = "") -> None:
    if isinstance(value, float):
        if abs(value) < 0.01:
            val_str = f"{value:.4f}"
        elif value > 1:
            val_str = f"{value:.2f}"
        else:
            val_str = f"{value:.2%}"
    else:
        val_str = str(value)

    status_icon = {"ok": "✓", "warn": "⚠", "error": "✗", "": ""}.get(status, "")
    print(f"   {name:<30} {val_str:>15}  {status_icon}")


def status_badge(ok: bool, label_ok: str = "PASS", label_fail: str = "FAIL") -> str:
    return f"[{'✓ ' + label_ok if ok else '✗ ' + label_fail}]"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_model_metrics() -> dict:
    """Collect metrics from model evaluation."""
    metrics = {}

    # Try reading labeled_predictions.csv
    pred_path = Path("logs/labeled_predictions.csv")
    if pred_path.exists():
        try:
            df = pd.read_csv(pred_path)
            metrics["total_samples"] = len(df)

            # Accuracy from PredLong vs Actual_Event
            if "PredLong" in df.columns and "Actual_Event" in df.columns:
                correct = (df["PredLong"] == df["Actual_Event"]).sum()
                metrics["accuracy"] = correct / len(df)

                # Class distribution
                metrics["pred_positive"] = int(df["PredLong"].sum())
                metrics["actual_positive"] = int(df["Actual_Event"].sum())
                metrics["pred_ratio"] = metrics["pred_positive"] / len(df)
                metrics["actual_ratio"] = metrics["actual_positive"] / len(df)

                # Precision/Recall for positive class
                tp = ((df["PredLong"] == 1) & (df["Actual_Event"] == 1)).sum()
                fp = ((df["PredLong"] == 1) & (df["Actual_Event"] == 0)).sum()
                fn = ((df["PredLong"] == 0) & (df["Actual_Event"] == 1)).sum()

                metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
                    metrics["precision"] + metrics["recall"]
                ) if (metrics["precision"] + metrics["recall"]) > 0 else 0

            # AUC if Proba column exists
            if "Proba" in df.columns and "Actual_Event" in df.columns:
                try:
                    from sklearn.metrics import roc_auc_score
                    metrics["auc"] = roc_auc_score(df["Actual_Event"], df["Proba"])
                except Exception:
                    pass

        except Exception as e:
            metrics["error"] = str(e)
    else:
        metrics["error"] = "logs/labeled_predictions.csv not found"

    return metrics


def collect_backtest_metrics() -> dict:
    """Collect metrics from backtest runs."""
    metrics = {}

    # Try logs/latest.json first
    latest_path = Path("logs/latest.json")
    if latest_path.exists():
        try:
            with open(latest_path) as f:
                data = json.load(f)
            # Handle both formats:
            # 1) backtest.py format: {"metrics": {...}, "config": {...}}
            # 2) generate_backtest_metrics.py format: {"total_return": ..., ...}
            if "metrics" in data:
                metrics.update(data.get("metrics", {}))
            else:
                # Direct format from generate_backtest_metrics.py
                metrics.update(data)
            metrics["source"] = "logs/latest.json"
        except Exception:
            pass

    # Try run_history.jsonl for historical context
    history_path = Path("logs/run_history.jsonl")
    if history_path.exists():
        try:
            runs = []
            with open(history_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            runs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            metrics["historical_runs"] = len(runs)
        except Exception:
            pass

    return metrics


def collect_walk_forward_metrics() -> dict:
    """Collect walk-forward validation results."""
    metrics = {}

    wf_path = Path("walk_forward_results.csv")
    if wf_path.exists():
        try:
            df = pd.read_csv(wf_path)
            metrics["n_folds"] = len(df)

            for col in ["lgb_accuracy", "xgb_accuracy", "ensemble_accuracy"]:
                if col in df.columns:
                    model_name = col.replace("_accuracy", "")
                    metrics[f"{model_name}_mean"] = df[col].mean()
                    metrics[f"{model_name}_std"] = df[col].std()

        except Exception as e:
            metrics["error"] = str(e)
    else:
        metrics["not_found"] = True

    return metrics


def collect_data_status() -> dict:
    """Check data files and integrity."""
    status = {"assets": {}, "issues": []}

    # Check primary data files
    data_files = {
        "SPY.csv": "data/SPY.csv",
        "DXY.csv": "data/DXY.csv",
        "HYG.csv": "data/HYG.csv",
        "LQD.csv": "data/LQD.csv",
        "TNX.csv": "data/TNX.csv",
        "cross_asset": "data/cross_asset_features.csv",
    }

    for name, path in data_files.items():
        p = Path(path)
        if p.exists():
            try:
                df = pd.read_csv(p)
                status["assets"][name] = {
                    "rows": len(df),
                    "ok": True,
                    "date_range": f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}" if "Date" in df.columns else "N/A"
                }
            except Exception as e:
                status["assets"][name] = {"ok": False, "error": str(e)}
                status["issues"].append(f"{name}: {e}")
        else:
            status["assets"][name] = {"ok": False, "missing": True}

    # Check model files
    models_dir = Path("models")
    required_models = ["xgboost_multi_asset.pkl", "lightgbm_multi_asset.pkl", "catboost_multi_asset.pkl"]
    status["models"] = {}

    for model in required_models:
        path = models_dir / model
        if path.exists():
            status["models"][model] = {"ok": True, "size_mb": path.stat().st_size / (1024*1024)}
        else:
            status["models"][model] = {"ok": False, "missing": True}
            status["issues"].append(f"Model missing: {model}")

    return status


def collect_config_status() -> dict:
    """Check configuration consistency."""
    config = {}

    # Best thresholds
    thresh_path = Path("configs/best_thresholds.json")
    if thresh_path.exists():
        try:
            with open(thresh_path) as f:
                config["thresholds"] = json.load(f)
        except Exception:
            pass

    # Asset config
    asset_path = Path("config/assets.yaml")
    if asset_path.exists():
        try:
            import yaml
            with open(asset_path) as f:
                asset_cfg = yaml.safe_load(f)
            # Count enabled assets
            total_assets = 0
            for group in ["equity_major_indices", "equity_international", "equity_sectors",
                         "equity_style", "equity_thematic", "bonds", "commodities", "crypto"]:
                if group in asset_cfg:
                    for ticker, info in asset_cfg[group].items():
                        if isinstance(info, dict) and info.get("enabled", True):
                            total_assets += 1
            config["total_assets_configured"] = total_assets
        except Exception:
            pass

    return config


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  NEUROVEST CONSOLIDATED STATS DASHBOARD".center(78) + "█")
    print("█" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Collect all metrics
    model_metrics = collect_model_metrics()
    backtest_metrics = collect_backtest_metrics()
    wf_metrics = collect_walk_forward_metrics()
    data_status = collect_data_status()
    config_status = collect_config_status()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: MODEL PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    header("MODEL PERFORMANCE", "═")

    subheader("Classification Metrics")
    if "accuracy" in model_metrics:
        metric_row("Accuracy", model_metrics["accuracy"], "ok" if model_metrics["accuracy"] > 0.6 else "warn")
        metric_row("AUC-ROC", model_metrics.get("auc", "N/A"), "ok" if model_metrics.get("auc", 0) > 0.7 else "warn")
        metric_row("Precision (Class 1)", model_metrics.get("precision", 0), "ok" if model_metrics.get("precision", 0) > 0.5 else "warn")
        metric_row("Recall (Class 1)", model_metrics.get("recall", 0), "error" if model_metrics.get("recall", 0) < 0.3 else "ok")
        metric_row("F1 Score", model_metrics.get("f1", 0), "warn" if model_metrics.get("f1", 0) < 0.5 else "ok")
    else:
        print(f"   ⚠ {model_metrics.get('error', 'No model metrics available')}")

    subheader("Prediction Distribution")
    if "pred_positive" in model_metrics:
        metric_row("Total Samples", model_metrics["total_samples"])
        metric_row("Predicted Long", f"{model_metrics['pred_positive']} ({model_metrics['pred_ratio']:.1%})")
        metric_row("Actual Positive", f"{model_metrics['actual_positive']} ({model_metrics['actual_ratio']:.1%})")

        # Flag imbalance
        imbalance = abs(model_metrics['pred_ratio'] - model_metrics['actual_ratio'])
        if imbalance > 0.15:
            print(f"\n   ⚠ WARNING: Prediction imbalance detected!")
            print(f"     Model predicts {model_metrics['pred_ratio']:.1%} long vs {model_metrics['actual_ratio']:.1%} actual")
            print(f"     Run: python3 evaluate.py to see updated metrics with new threshold")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: BACKTEST PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    header("BACKTEST PERFORMANCE", "═")

    subheader("Returns & Risk")
    if backtest_metrics and backtest_metrics.get("source"):
        # Handle both key formats (backtest.py vs generate_backtest_metrics.py)
        total_ret = backtest_metrics.get("total_return", "N/A")
        annual_ret = backtest_metrics.get("annual_return", backtest_metrics.get("annualized_return", "N/A"))
        sharpe = backtest_metrics.get("sharpe_ratio", backtest_metrics.get("sharpe", "N/A"))
        sortino = backtest_metrics.get("sortino_ratio", backtest_metrics.get("sortino", "N/A"))
        calmar = backtest_metrics.get("calmar_ratio", backtest_metrics.get("calmar", "N/A"))
        max_dd = backtest_metrics.get("max_drawdown", "N/A")

        # Convert percentage values (total_return=191.0 means 191%)
        if isinstance(total_ret, (int, float)) and total_ret > 1:
            total_ret = total_ret / 100  # Convert 191.0 to 1.91
        if isinstance(annual_ret, (int, float)) and annual_ret > 1:
            annual_ret = annual_ret / 100
        if isinstance(max_dd, (int, float)) and max_dd < -1:
            max_dd = max_dd / 100

        metric_row("Total Return", total_ret)
        metric_row("Annualized Return", annual_ret)
        sharpe_val = sharpe if isinstance(sharpe, str) else sharpe
        metric_row("Sharpe Ratio", sharpe_val, "ok" if isinstance(sharpe_val, (int, float)) and sharpe_val > 1.5 else "warn")
        metric_row("Sortino Ratio", sortino)
        dd_val = max_dd if isinstance(max_dd, str) else max_dd
        metric_row("Max Drawdown", dd_val, "ok" if isinstance(dd_val, (int, float)) and abs(dd_val) < 0.15 else "warn")
        metric_row("Calmar Ratio", calmar)

        subheader("Trade Statistics")
        trades = backtest_metrics.get("total_trades", backtest_metrics.get("trades", "N/A"))
        win_rate = backtest_metrics.get("win_rate", "N/A")
        if isinstance(win_rate, (int, float)) and win_rate > 1:
            win_rate = win_rate / 100  # Convert 54.0 to 0.54
        pf = backtest_metrics.get("profit_factor", "N/A")

        metric_row("Total Trades", trades)
        metric_row("Win Rate", win_rate)
        pf_val = pf if isinstance(pf, str) else pf
        metric_row("Profit Factor", pf_val, "ok" if isinstance(pf_val, (int, float)) and pf_val > 1.5 else "warn")
    else:
        print("   ⚠ No backtest metrics found (run backtest.py or generate_backtest_metrics.py)")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: WALK-FORWARD VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    header("WALK-FORWARD VALIDATION (Out-of-Sample)", "═")

    if not wf_metrics.get("not_found"):
        subheader("Cross-Validated Performance")
        metric_row("Number of Folds", wf_metrics.get("n_folds", "N/A"))

        for model in ["lgb", "xgb", "ensemble"]:
            mean_key = f"{model}_mean"
            std_key = f"{model}_std"
            if mean_key in wf_metrics:
                name = {"lgb": "LightGBM", "xgb": "XGBoost", "ensemble": "Ensemble"}[model]
                value = f"{wf_metrics[mean_key]:.1%} ± {wf_metrics[std_key]:.1%}"
                metric_row(name, value, "ok" if wf_metrics[mean_key] > 0.6 else "warn")
    else:
        print("   ⚠ No walk-forward results (run walk_forward_validation.py first)")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: DATA & MODEL STATUS
    # ═══════════════════════════════════════════════════════════════════════════
    header("SYSTEM STATUS", "═")

    subheader("Data Files")
    for name, info in data_status["assets"].items():
        if info.get("ok"):
            print(f"   ✓ {name:<25} {info['rows']:>6,} rows   {info.get('date_range', '')}")
        elif info.get("missing"):
            print(f"   ✗ {name:<25} MISSING")
        else:
            print(f"   ⚠ {name:<25} ERROR: {info.get('error', 'unknown')}")

    subheader("Model Files")
    for name, info in data_status["models"].items():
        if info.get("ok"):
            print(f"   ✓ {name:<35} {info['size_mb']:.2f} MB")
        else:
            print(f"   ✗ {name:<35} MISSING")

    subheader("Configuration")
    if "thresholds" in config_status:
        th = config_status["thresholds"]
        print(f"   Prediction Threshold:   {th.get('threshold', 'N/A')}")
        print(f"   Confidence Threshold:   {th.get('confidence_thresh', 'N/A')}")
        print(f"   Invert Probability:     {th.get('invert_proba', 'N/A')}")

    if "total_assets_configured" in config_status:
        print(f"   Assets Configured:      {config_status['total_assets_configured']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: ISSUES & RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    header("ISSUES & RECOMMENDATIONS", "═")

    issues = []
    recommendations = []

    # Check for issues
    current_thresh = config_status.get("thresholds", {}).get("threshold", 0.45)
    if model_metrics.get("recall", 1) < 0.3:
        pred_pct = model_metrics.get("pred_ratio", 0) * 100
        actual_pct = model_metrics.get("actual_ratio", 0) * 100
        issues.append(f"LOW RECALL: Model predicting 'long' {pred_pct:.0f}% of time vs {actual_pct:.0f}% actual")
        if current_thresh >= 0.40:
            recommendations.append(f"Lower prediction threshold from {current_thresh} to ~0.35-0.38")
        else:
            recommendations.append("Re-run evaluate.py to see updated metrics with new threshold")

    if not any(data_status["models"].get(m, {}).get("ok") for m in data_status["models"]):
        issues.append("MODELS MISSING: No trained models found in models/")
        recommendations.append("Run: python3 train_multi_asset.py")

    if data_status.get("issues"):
        for issue in data_status["issues"]:
            issues.append(f"DATA: {issue}")

    # Accuracy discrepancy check
    if model_metrics.get("accuracy") and backtest_metrics.get("model_accuracy"):
        diff = abs(model_metrics["accuracy"] - backtest_metrics["model_accuracy"])
        if diff > 0.05:
            issues.append(f"ACCURACY MISMATCH: evaluate.py ({model_metrics['accuracy']:.1%}) vs backtest ({backtest_metrics['model_accuracy']:.1%})")
            recommendations.append("Ensure same threshold/data used in both evaluations")

    if issues:
        print("\n   ⚠ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
    else:
        print("\n   ✓ No critical issues detected")

    if recommendations:
        print("\n   📋 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  Single Source of Truth: python3 consolidated_stats.py")
    print("  Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═" * 80 + "\n")

    return {
        "model": model_metrics,
        "backtest": backtest_metrics,
        "walk_forward": wf_metrics,
        "data": data_status,
        "config": config_status,
        "issues": issues,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    result = main()
