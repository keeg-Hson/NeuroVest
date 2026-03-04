# core/regime_adaptive_thresholds.py
"""
Regime-Adaptive Threshold Management

Purpose:
    Computes and applies different prediction thresholds based on market regime.
    Model performance varies significantly across regimes - using adaptive thresholds
    improves risk management by:

    - Higher thresholds in high volatility (fewer, higher-confidence trades)
    - Lower thresholds in low volatility (capture more opportunities)
    - Tighter thresholds in bear markets (risk-off positioning)

Usage:
    from core.regime_adaptive_thresholds import (
        RegimeAdaptiveThresholds,
        compute_regime_thresholds,
        get_threshold_for_regime,
    )

    # During training
    rat = RegimeAdaptiveThresholds(df_with_features)
    rat.compute_optimal_thresholds(y_true, y_proba)
    rat.save('models/regime_thresholds.json')

    # During prediction
    current_regime = rat.detect_current_regime(latest_df)
    threshold = rat.get_threshold(current_regime)

Default Thresholds by Regime (can be overridden by optimization):
    - volatility:
        - low: 0.40 (capture more opportunities in calm markets)
        - medium: 0.45 (standard threshold)
        - high: 0.55 (require higher confidence when volatile)

    - trend:
        - bull: 0.42 (slightly more aggressive in uptrend)
        - sideways: 0.45 (standard)
        - bear: 0.50 (more conservative in downtrend)

    - risk_appetite:
        - risk_on: 0.42
        - risk_off: 0.52
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score


@dataclass
class RegimeThresholdConfig:
    """Configuration for regime-adaptive thresholds."""

    # Default thresholds per regime (before optimization)
    # These are conservative starting points
    default_thresholds: dict = field(default_factory=lambda: {
        "volatility": {
            "low": 0.40,
            "medium": 0.45,
            "high": 0.55,
        },
        "trend": {
            "bull": 0.42,
            "sideways": 0.45,
            "bear": 0.50,
        },
        "risk_appetite": {
            "risk_on": 0.42,
            "risk_off": 0.52,
        },
    })

    # Threshold bounds (clamp all thresholds to this range)
    min_threshold: float = 0.25
    max_threshold: float = 0.70

    # Optimization settings
    min_samples_per_regime: int = 50  # Min samples to compute regime-specific threshold
    min_precision: float = 0.30  # Minimum acceptable precision
    target_recall: float = 0.20  # Target recall at precision threshold

    # Primary regime for threshold selection (volatility is most predictive)
    primary_regime: str = "volatility"


class RegimeAdaptiveThresholds:
    """
    Computes and applies regime-adaptive prediction thresholds.

    The key insight: model performance varies across market regimes.
    Using a single threshold leaves money on the table in calm markets
    and takes excessive risk in volatile markets.
    """

    THRESHOLD_FILE = "models/regime_thresholds.json"

    def __init__(
        self,
        df: pd.DataFrame = None,
        config: RegimeThresholdConfig = None,
    ):
        """
        Initialize with price/feature DataFrame.

        Args:
            df: DataFrame with OHLCV + features (DatetimeIndex or Date column)
            config: Threshold configuration
        """
        self.df = df
        self.config = config or RegimeThresholdConfig()
        self.regimes = None
        self.thresholds = self.config.default_thresholds.copy()
        self.optimization_meta = {}

    def _ensure_regimes(self):
        """Detect regimes if not already done."""
        if self.regimes is not None:
            return

        if self.df is None:
            raise ValueError("No DataFrame provided for regime detection")

        from core.regime_backtest import RegimeDetector
        detector = RegimeDetector()
        self.regimes = detector.detect_all_regimes(self.df)

    def detect_current_regime(self, df: pd.DataFrame = None) -> dict:
        """
        Detect current market regime from latest data.

        Args:
            df: DataFrame with latest price/feature data

        Returns:
            dict with regime classifications
        """
        if df is not None:
            self.df = df
            self.regimes = None  # Force re-detection

        self._ensure_regimes()

        latest = self.regimes.iloc[-1]
        return {
            "date": str(self.regimes.index[-1]),
            "volatility": latest["volatility"],
            "trend": latest["trend"],
            "risk_appetite": latest["risk_appetite"],
            "market_phase": latest["market_phase"],
        }

    def get_threshold(
        self,
        regime: dict = None,
        regime_type: str = None,
    ) -> float:
        """
        Get the appropriate threshold for a given regime.

        Args:
            regime: dict with regime classifications (from detect_current_regime)
            regime_type: Override to use specific regime type (default: primary_regime)

        Returns:
            Probability threshold to use for predictions
        """
        if regime is None:
            regime = self.detect_current_regime()

        regime_type = regime_type or self.config.primary_regime

        if regime_type not in self.thresholds:
            # Fall back to base threshold
            return self.config.default_thresholds["volatility"]["medium"]

        regime_value = regime.get(regime_type, "medium")

        if regime_value not in self.thresholds[regime_type]:
            # Unknown regime value - use medium/default
            return self.config.default_thresholds.get(regime_type, {}).get(
                "medium", 0.45
            )

        return self.thresholds[regime_type][regime_value]

    def compute_optimal_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        dates: pd.DatetimeIndex = None,
    ) -> dict:
        """
        Compute optimal thresholds per regime from OOF predictions.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            dates: DatetimeIndex aligned with y_true/y_proba

        Returns:
            dict with optimized thresholds per regime
        """
        self._ensure_regimes()

        if dates is None and self.df is not None:
            dates = self.regimes.index[-len(y_true):]

        if dates is None or len(dates) != len(y_true):
            print("[warn] Cannot align dates for regime-specific optimization")
            print("       Using default thresholds")
            return self.thresholds

        # Align regimes with predictions
        regime_df = self.regimes.loc[dates].copy()

        optimized = {}
        meta = {}

        for regime_type in ["volatility", "trend", "risk_appetite"]:
            optimized[regime_type] = {}
            meta[regime_type] = {}

            for regime_value in regime_df[regime_type].unique():
                if pd.isna(regime_value):
                    continue

                mask = regime_df[regime_type] == regime_value
                n_samples = mask.sum()

                if n_samples < self.config.min_samples_per_regime:
                    # Not enough samples - use default
                    default = self.config.default_thresholds.get(regime_type, {}).get(
                        regime_value, 0.45
                    )
                    optimized[regime_type][regime_value] = default
                    meta[regime_type][regime_value] = {
                        "n_samples": n_samples,
                        "source": "default (insufficient samples)",
                    }
                    continue

                # Get regime-specific predictions
                y_true_regime = y_true[mask]
                y_proba_regime = y_proba[mask]

                # Optimize threshold for this regime
                best_thresh = self._optimize_single_threshold(
                    y_true_regime,
                    y_proba_regime,
                    regime_type,
                    regime_value,
                )

                optimized[regime_type][regime_value] = best_thresh

                # Compute metrics at this threshold
                y_pred = (y_proba_regime >= best_thresh).astype(int)
                precision = (y_pred & y_true_regime).sum() / max(1, y_pred.sum())
                recall = (y_pred & y_true_regime).sum() / max(1, y_true_regime.sum())

                meta[regime_type][regime_value] = {
                    "n_samples": int(n_samples),
                    "n_positive": int(y_true_regime.sum()),
                    "threshold": round(best_thresh, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "source": "optimized",
                }

        self.thresholds = optimized
        self.optimization_meta = meta

        return optimized

    def _optimize_single_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        regime_type: str,
        regime_value: str,
    ) -> float:
        """
        Optimize threshold for a single regime using precision-recall tradeoff.

        Strategy: Find threshold that achieves minimum precision while
        maximizing recall (or F1).
        """
        # Get default as fallback
        default = self.config.default_thresholds.get(regime_type, {}).get(
            regime_value, 0.45
        )

        try:
            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

            # Find thresholds that meet minimum precision
            valid_mask = precision[:-1] >= self.config.min_precision

            if not valid_mask.any():
                # Can't meet precision target - use conservative threshold
                return min(default + 0.05, self.config.max_threshold)

            valid_thresholds = thresholds[valid_mask]
            valid_recall = recall[:-1][valid_mask]
            valid_precision = precision[:-1][valid_mask]

            # Compute F1 for valid thresholds
            f1_scores = 2 * (valid_precision * valid_recall) / (
                valid_precision + valid_recall + 1e-9
            )

            # Select threshold with best F1
            best_idx = f1_scores.argmax()
            best_thresh = valid_thresholds[best_idx]

            # Clamp to bounds
            best_thresh = max(self.config.min_threshold,
                            min(self.config.max_threshold, best_thresh))

            return round(best_thresh, 4)

        except Exception as e:
            print(f"[warn] Threshold optimization failed for {regime_type}={regime_value}: {e}")
            return default

    def save(self, path: str = None) -> None:
        """Save thresholds to JSON file."""
        path = path or self.THRESHOLD_FILE
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "thresholds": self.thresholds,
            "meta": self.optimization_meta,
            "config": {
                "primary_regime": self.config.primary_regime,
                "min_threshold": self.config.min_threshold,
                "max_threshold": self.config.max_threshold,
                "min_precision": self.config.min_precision,
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Saved regime thresholds to {path}")

    def load(self, path: str = None) -> dict:
        """Load thresholds from JSON file."""
        path = path or self.THRESHOLD_FILE

        try:
            with open(path) as f:
                data = json.load(f)

            self.thresholds = data.get("thresholds", self.config.default_thresholds)
            self.optimization_meta = data.get("meta", {})

            if "config" in data:
                self.config.primary_regime = data["config"].get(
                    "primary_regime", self.config.primary_regime
                )

            print(f"Loaded regime thresholds from {path}")
            return self.thresholds

        except FileNotFoundError:
            print(f"[info] No regime thresholds found at {path}, using defaults")
            return self.thresholds
        except Exception as e:
            print(f"[warn] Failed to load regime thresholds: {e}")
            return self.thresholds

    def get_summary(self) -> str:
        """Get human-readable summary of thresholds."""
        lines = ["Regime-Adaptive Thresholds:", "=" * 40]

        for regime_type in ["volatility", "trend", "risk_appetite"]:
            if regime_type not in self.thresholds:
                continue

            primary = "(PRIMARY)" if regime_type == self.config.primary_regime else ""
            lines.append(f"\n{regime_type.upper()} {primary}")

            for regime_value, thresh in self.thresholds[regime_type].items():
                meta = self.optimization_meta.get(regime_type, {}).get(regime_value, {})
                source = meta.get("source", "default")
                prec = meta.get("precision", "N/A")
                recall = meta.get("recall", "N/A")

                if isinstance(prec, float):
                    prec = f"{prec:.1%}"
                if isinstance(recall, float):
                    recall = f"{recall:.1%}"

                lines.append(
                    f"  {regime_value:12s}: {thresh:.2f}  "
                    f"(P={prec}, R={recall}, {source})"
                )

        return "\n".join(lines)


# =============================================================================
# Convenience functions
# =============================================================================

def get_threshold_for_regime(
    df: pd.DataFrame,
    threshold_path: str = "models/regime_thresholds.json",
) -> tuple[float, dict]:
    """
    Get the appropriate threshold for current market regime.

    Args:
        df: DataFrame with latest price/feature data
        threshold_path: Path to regime thresholds JSON

    Returns:
        (threshold, regime_dict)
    """
    rat = RegimeAdaptiveThresholds(df)
    rat.load(threshold_path)

    regime = rat.detect_current_regime()
    threshold = rat.get_threshold(regime)

    return threshold, regime


def compute_regime_thresholds(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str = "models/regime_thresholds.json",
) -> dict:
    """
    Compute and save optimal thresholds per regime.

    Args:
        df: DataFrame with price/features (must have DatetimeIndex)
        y_true: True binary labels
        y_proba: Predicted probabilities
        output_path: Where to save thresholds

    Returns:
        dict with optimized thresholds
    """
    rat = RegimeAdaptiveThresholds(df)
    thresholds = rat.compute_optimal_thresholds(y_true, y_proba)
    rat.save(output_path)

    print(rat.get_summary())

    return thresholds


if __name__ == "__main__":
    # Demo: compute regime thresholds from labeled predictions
    import argparse

    parser = argparse.ArgumentParser(description="Compute regime-adaptive thresholds")
    parser.add_argument("--predictions", default="logs/labeled_predictions.csv",
                       help="Path to predictions CSV with y_true and probability")
    parser.add_argument("--output", default="models/regime_thresholds.json",
                       help="Output path for thresholds JSON")
    args = parser.parse_args()

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        print("Run predict.py with backfill first")
        exit(1)

    preds = pd.read_csv(pred_path)

    # Load price data for regime detection
    from utils import load_SPY_data
    df = load_SPY_data()

    # Align dates
    preds["Date"] = pd.to_datetime(preds["Date"])
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    # Get labels and probabilities
    # Handle different column naming conventions
    if "y_true" in preds.columns:
        y_true = preds["y_true"].values
    elif "Actual" in preds.columns:
        y_true = (preds["Actual"] == 2).astype(int).values  # SPIKE = positive
    else:
        print("Cannot find true labels in predictions file")
        exit(1)

    if "Probability" in preds.columns:
        y_proba = preds["Probability"].values
    elif "Confidence" in preds.columns:
        y_proba = preds["Confidence"].values
    else:
        print("Cannot find probabilities in predictions file")
        exit(1)

    # Compute thresholds
    rat = RegimeAdaptiveThresholds(df)

    # Set dates from predictions
    dates = pd.DatetimeIndex(preds["Date"])
    rat.compute_optimal_thresholds(y_true, y_proba, dates)

    rat.save(args.output)
    print("\n" + rat.get_summary())
