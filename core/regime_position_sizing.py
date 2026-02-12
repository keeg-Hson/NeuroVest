# core/regime_position_sizing.py
"""
Regime-Adaptive Position Sizing

Purpose:
    Adjusts position sizes based on market regime to improve drawdown control.
    In high volatility or bear markets, reduce position sizes to limit risk.
    In calm bull markets, allow larger positions to capture opportunities.

Strategy:
    - High volatility → Reduce to 50-70% of base size (smaller positions when uncertain)
    - Low volatility → Allow 100-120% of base size (capture more opportunity)
    - Bear market → Reduce to 60-80% of base (protect capital)
    - Bull market → Allow 100-110% of base (ride trends)
    - Risk-off → Reduce to 50-70% of base (preserve capital)
    - Risk-on → Allow 100-120% of base (capitalize on momentum)

Usage:
    from core.regime_position_sizing import (
        RegimePositionSizer,
        get_position_multiplier,
    )

    sizer = RegimePositionSizer(df)
    multiplier = sizer.get_multiplier()  # Returns 0.5 to 1.2

    actual_size = base_position_size * multiplier
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class PositionSizingConfig:
    """Configuration for regime-adaptive position sizing."""

    # Base multipliers by regime (multiply base position by these)
    volatility_multipliers: dict = field(default_factory=lambda: {
        "low": 1.10,      # More aggressive in calm markets
        "medium": 1.00,   # Standard sizing
        "high": 0.60,     # Reduce size in high vol (drawdown control)
    })

    trend_multipliers: dict = field(default_factory=lambda: {
        "bull": 1.05,     # Slightly more aggressive in uptrend
        "sideways": 1.00, # Standard
        "bear": 0.70,     # Reduce in downtrend (protect capital)
    })

    risk_appetite_multipliers: dict = field(default_factory=lambda: {
        "risk_on": 1.10,  # More aggressive when market risk-on
        "risk_off": 0.60, # Conservative when risk-off
    })

    # How to combine multiple regime signals
    # Options: "minimum", "average", "product"
    combination_method: str = "minimum"

    # Hard bounds
    min_multiplier: float = 0.40   # Never go below 40% of base
    max_multiplier: float = 1.25   # Never exceed 125% of base

    # Primary regime for sizing (volatility most predictive of drawdowns)
    primary_regime: str = "volatility"

    # Weight for primary vs secondary regimes
    primary_weight: float = 0.6
    secondary_weight: float = 0.2


class RegimePositionSizer:
    """
    Computes position size multipliers based on market regime.

    Lower multipliers in risky regimes (high vol, bear, risk-off)
    help control drawdowns without completely exiting the market.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        config: PositionSizingConfig = None,
    ):
        """
        Initialize with price/feature DataFrame.

        Args:
            df: DataFrame with OHLCV + features (for regime detection)
            config: Position sizing configuration
        """
        self.df = df
        self.config = config or PositionSizingConfig()
        self.current_regime = None
        self._detector = None

    def _get_detector(self):
        """Lazy load regime detector."""
        if self._detector is None:
            from core.regime_backtest import RegimeDetector
            self._detector = RegimeDetector()
        return self._detector

    def detect_regime(self, df: pd.DataFrame = None) -> dict:
        """
        Detect current market regime.

        Args:
            df: DataFrame with price data (uses stored df if None)

        Returns:
            dict with regime classifications
        """
        if df is not None:
            self.df = df

        if self.df is None:
            raise ValueError("No DataFrame provided for regime detection")

        detector = self._get_detector()
        regimes = detector.detect_all_regimes(self.df)

        latest = regimes.iloc[-1]
        self.current_regime = {
            "date": str(regimes.index[-1]),
            "volatility": latest["volatility"],
            "trend": latest["trend"],
            "risk_appetite": latest["risk_appetite"],
            "market_phase": latest["market_phase"],
        }

        return self.current_regime

    def get_multiplier(
        self,
        regime: dict = None,
        method: str = None,
    ) -> float:
        """
        Get position size multiplier for current or specified regime.

        Args:
            regime: Regime dict (auto-detected if None)
            method: Combination method override

        Returns:
            Multiplier between min_multiplier and max_multiplier
        """
        if regime is None:
            regime = self.detect_regime()

        method = method or self.config.combination_method

        # Get multipliers for each regime type
        vol_mult = self.config.volatility_multipliers.get(
            regime.get("volatility", "medium"), 1.0
        )
        trend_mult = self.config.trend_multipliers.get(
            regime.get("trend", "sideways"), 1.0
        )
        risk_mult = self.config.risk_appetite_multipliers.get(
            regime.get("risk_appetite", "risk_on"), 1.0
        )

        # Combine multipliers based on method
        if method == "minimum":
            # Most conservative: use smallest multiplier
            combined = min(vol_mult, trend_mult, risk_mult)
        elif method == "average":
            # Balanced: average of all
            combined = (vol_mult + trend_mult + risk_mult) / 3
        elif method == "product":
            # Aggressive compounding
            combined = vol_mult * trend_mult * risk_mult
        elif method == "weighted":
            # Weighted average with primary regime emphasis
            combined = (
                self.config.primary_weight * vol_mult +
                self.config.secondary_weight * trend_mult +
                self.config.secondary_weight * risk_mult
            )
        else:
            combined = vol_mult  # Default to volatility only

        # Clamp to bounds
        multiplier = max(
            self.config.min_multiplier,
            min(self.config.max_multiplier, combined)
        )

        return round(multiplier, 3)

    def get_sizing_recommendation(self, base_size: float = 1.0) -> dict:
        """
        Get detailed sizing recommendation.

        Args:
            base_size: Base position size (e.g., 0.05 for 5% of portfolio)

        Returns:
            dict with recommendation details
        """
        regime = self.detect_regime()
        multiplier = self.get_multiplier(regime)
        adjusted_size = base_size * multiplier

        # Get individual multipliers for reporting
        vol_mult = self.config.volatility_multipliers.get(
            regime.get("volatility", "medium"), 1.0
        )
        trend_mult = self.config.trend_multipliers.get(
            regime.get("trend", "sideways"), 1.0
        )
        risk_mult = self.config.risk_appetite_multipliers.get(
            regime.get("risk_appetite", "risk_on"), 1.0
        )

        return {
            "regime": regime,
            "multiplier": multiplier,
            "base_size": base_size,
            "adjusted_size": round(adjusted_size, 4),
            "components": {
                "volatility": {"regime": regime.get("volatility"), "mult": vol_mult},
                "trend": {"regime": regime.get("trend"), "mult": trend_mult},
                "risk_appetite": {"regime": regime.get("risk_appetite"), "mult": risk_mult},
            },
            "recommendation": self._get_recommendation_text(regime, multiplier),
        }

    def _get_recommendation_text(self, regime: dict, multiplier: float) -> str:
        """Generate human-readable recommendation."""
        if multiplier <= 0.5:
            risk_level = "VERY CONSERVATIVE"
            action = "Reduce position sizes significantly. High risk environment."
        elif multiplier <= 0.7:
            risk_level = "CONSERVATIVE"
            action = "Reduce position sizes. Elevated uncertainty."
        elif multiplier <= 0.9:
            risk_level = "CAUTIOUS"
            action = "Slightly reduce sizes. Mixed signals."
        elif multiplier <= 1.05:
            risk_level = "NORMAL"
            action = "Standard position sizes. Balanced conditions."
        else:
            risk_level = "OPPORTUNISTIC"
            action = "Can increase sizes modestly. Favorable conditions."

        vol = regime.get("volatility", "unknown")
        trend = regime.get("trend", "unknown")

        return f"{risk_level}: {action} (Vol={vol}, Trend={trend}, Mult={multiplier:.2f})"


# =============================================================================
# Convenience Functions
# =============================================================================

def get_position_multiplier(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Get position size multiplier for current market regime.

    Args:
        df: DataFrame with price/feature data

    Returns:
        (multiplier, regime_dict)
    """
    sizer = RegimePositionSizer(df)
    regime = sizer.detect_regime()
    multiplier = sizer.get_multiplier(regime)
    return multiplier, regime


def apply_regime_sizing(
    base_size: float,
    df: pd.DataFrame,
    verbose: bool = True,
) -> float:
    """
    Apply regime-adaptive sizing to a base position size.

    Args:
        base_size: Base position size (e.g., 0.05 for 5%)
        df: DataFrame for regime detection
        verbose: Print recommendation

    Returns:
        Adjusted position size
    """
    sizer = RegimePositionSizer(df)
    rec = sizer.get_sizing_recommendation(base_size)

    if verbose:
        print(f"\n[Position Sizing] {rec['recommendation']}")
        print(f"  Base: {base_size:.2%} → Adjusted: {rec['adjusted_size']:.2%}")

    return rec["adjusted_size"]


if __name__ == "__main__":
    # Demo usage
    print("Regime-Adaptive Position Sizing Demo")
    print("=" * 50)

    # Load data
    try:
        from utils import load_SPY_data
        df = load_SPY_data()

        sizer = RegimePositionSizer(df)
        rec = sizer.get_sizing_recommendation(base_size=0.05)

        print(f"\nCurrent Regime:")
        for k, v in rec["regime"].items():
            print(f"  {k}: {v}")

        print(f"\nPosition Sizing:")
        print(f"  Multiplier: {rec['multiplier']:.2f}")
        print(f"  Base Size: {rec['base_size']:.2%}")
        print(f"  Adjusted Size: {rec['adjusted_size']:.2%}")

        print(f"\nComponents:")
        for comp, data in rec["components"].items():
            print(f"  {comp}: {data['regime']} → {data['mult']:.2f}x")

        print(f"\n{rec['recommendation']}")

    except Exception as e:
        print(f"Demo requires data: {e}")
        print("\nManual test with sample regimes:")

        config = PositionSizingConfig()
        sizer = RegimePositionSizer(config=config)

        test_regimes = [
            {"volatility": "low", "trend": "bull", "risk_appetite": "risk_on"},
            {"volatility": "medium", "trend": "sideways", "risk_appetite": "risk_on"},
            {"volatility": "high", "trend": "bear", "risk_appetite": "risk_off"},
        ]

        for regime in test_regimes:
            sizer.current_regime = regime
            mult = sizer.get_multiplier(regime)
            print(f"\n{regime}")
            print(f"  → Multiplier: {mult:.2f}")
