# core/regime_backtest.py
"""
Regime-Specific Backtesting Framework

Purpose:
    Analyze model performance across different market regimes to:
    1. Identify regime-dependent strengths and weaknesses
    2. Enable regime-adaptive trading strategies
    3. Improve model calibration for different market conditions

Market Regimes Detected:
    - Trend: Bull / Bear / Sideways
    - Volatility: Low / Medium / High
    - Risk Appetite: Risk-On / Risk-Off
    - Market Phase: Expansion / Contraction / Recovery / Peak

Usage:
    from core.regime_backtest import RegimeBacktester, RegimeType

    rb = RegimeBacktester(df)
    rb.detect_regimes()
    results = rb.backtest_by_regime('volatility')
    rb.generate_report()
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


class RegimeType(Enum):
    """Types of market regimes to detect and analyze."""
    TREND = "trend"
    VOLATILITY = "volatility"
    RISK_APPETITE = "risk_appetite"
    MARKET_PHASE = "market_phase"


@dataclass
class RegimeConfig:
    """Configuration for regime detection thresholds."""
    # Trend detection
    trend_ma_period: int = 200
    trend_slope_period: int = 20
    trend_strength_threshold: float = 0.02  # 2% above/below MA200

    # Volatility detection
    vol_lookback: int = 252
    vol_low_percentile: float = 0.33
    vol_high_percentile: float = 0.67

    # Risk appetite detection
    risk_on_threshold: float = 0.5

    # Market phase detection
    expansion_threshold: float = 0.03  # 3% positive momentum
    contraction_threshold: float = -0.03


class RegimeDetector:
    """Detects market regimes from price and feature data."""

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()

    def detect_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend regime: Bull / Bear / Sideways

        Logic:
            - Bull: Price > MA200 and MA200 slope positive
            - Bear: Price < MA200 and MA200 slope negative
            - Sideways: Otherwise
        """
        close = df["Close"]
        ma200 = close.rolling(self.config.trend_ma_period, min_periods=100).mean()
        ma200_slope = ma200.pct_change(self.config.trend_slope_period)

        price_vs_ma = close / ma200

        regime = pd.Series(index=df.index, data="sideways", dtype=str)

        bull_mask = (price_vs_ma > (1 + self.config.trend_strength_threshold)) & (ma200_slope > 0)
        bear_mask = (price_vs_ma < (1 - self.config.trend_strength_threshold)) & (ma200_slope < 0)

        regime[bull_mask] = "bull"
        regime[bear_mask] = "bear"

        return regime

    def detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime: Low / Medium / High

        Logic:
            - Uses rolling percentile of realized volatility
            - Low: < 33rd percentile
            - Medium: 33rd - 67th percentile
            - High: > 67th percentile
        """
        if "Volatility" in df.columns:
            vol = df["Volatility"]
        else:
            vol = df["Close"].pct_change().rolling(20).std()

        vol_percentile = vol.rolling(
            self.config.vol_lookback, min_periods=60
        ).rank(pct=True)

        regime = pd.Series(index=df.index, data="medium", dtype=str)
        regime[vol_percentile < self.config.vol_low_percentile] = "low"
        regime[vol_percentile > self.config.vol_high_percentile] = "high"

        return regime

    def detect_risk_appetite_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect risk appetite regime: Risk-On / Risk-Off

        Logic:
            Combines multiple signals:
            - Price vs MA200
            - Volatility level
            - Momentum
            - Credit stress (if available)
        """
        score = pd.Series(index=df.index, data=0.0)
        n_signals = 0

        # Signal 1: Price above MA200
        if "Close" in df.columns:
            ma200 = df["Close"].rolling(200, min_periods=100).mean()
            score += (df["Close"] > ma200).astype(float)
            n_signals += 1

        # Signal 2: Low volatility
        if "Volatility" in df.columns:
            vol_pct = df["Volatility"].rolling(252, min_periods=60).rank(pct=True)
            score += (vol_pct < 0.5).astype(float)
            n_signals += 1

        # Signal 3: Positive momentum
        if "Return_Lag5" in df.columns:
            score += (df["Return_Lag5"] > 0).astype(float)
            n_signals += 1
        elif "Close" in df.columns:
            mom = df["Close"].pct_change(5)
            score += (mom > 0).astype(float)
            n_signals += 1

        # Signal 4: Low credit stress
        if "Credit_Stress" in df.columns:
            score += (df["Credit_Stress"] == 0).astype(float)
            n_signals += 1

        # Normalize
        if n_signals > 0:
            score = score / n_signals

        regime = pd.Series(index=df.index, data="risk_off", dtype=str)
        regime[score > self.config.risk_on_threshold] = "risk_on"

        return regime

    def detect_market_phase(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market phase: Expansion / Peak / Contraction / Recovery

        Logic:
            - Expansion: Positive momentum, rising prices
            - Peak: High prices, momentum slowing
            - Contraction: Negative momentum, falling prices
            - Recovery: Low prices, momentum improving
        """
        close = df["Close"]
        mom_3m = close.pct_change(63)  # ~3 months
        mom_1m = close.pct_change(21)  # ~1 month

        # Price relative to 52-week range
        rolling_high = close.rolling(252, min_periods=126).max()
        rolling_low = close.rolling(252, min_periods=126).min()
        position = (close - rolling_low) / (rolling_high - rolling_low + 1e-9)

        regime = pd.Series(index=df.index, data="expansion", dtype=str)

        # Contraction: Negative 3-month momentum
        contraction = mom_3m < self.config.contraction_threshold
        regime[contraction] = "contraction"

        # Recovery: Low position but improving momentum
        recovery = (position < 0.3) & (mom_1m > 0) & ~contraction
        regime[recovery] = "recovery"

        # Peak: High position but slowing momentum
        peak = (position > 0.85) & (mom_1m < mom_3m / 3)
        regime[peak] = "peak"

        return regime

    def detect_all_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all regime types and return as DataFrame."""
        regimes = pd.DataFrame(index=df.index)
        regimes["trend"] = self.detect_trend_regime(df)
        regimes["volatility"] = self.detect_volatility_regime(df)
        regimes["risk_appetite"] = self.detect_risk_appetite_regime(df)
        regimes["market_phase"] = self.detect_market_phase(df)
        return regimes


class RegimeBacktester:
    """
    Backtests model performance across different market regimes.

    Usage:
        rb = RegimeBacktester(price_df, predictions_df)
        rb.detect_regimes()
        results = rb.backtest_by_regime(RegimeType.VOLATILITY)
        rb.generate_report('logs/regime_analysis.json')
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        predictions_df: pd.DataFrame = None,
        config: RegimeConfig = None,
    ):
        """
        Initialize regime backtester.

        Args:
            price_df: DataFrame with OHLCV data, DatetimeIndex
            predictions_df: DataFrame with predictions (Date, Prediction, etc.)
            config: RegimeConfig for detection thresholds
        """
        self.price_df = price_df.copy()
        self.predictions_df = predictions_df
        self.detector = RegimeDetector(config)
        self.regimes = None
        self.results = {}

    def detect_regimes(self) -> pd.DataFrame:
        """Detect all regimes and store for analysis."""
        self.regimes = self.detector.detect_all_regimes(self.price_df)
        return self.regimes

    def _get_regime_periods(self, regime_type: str) -> dict:
        """Get date ranges for each regime value."""
        if self.regimes is None:
            self.detect_regimes()

        regime_col = self.regimes[regime_type]
        periods = {}

        for regime_value in regime_col.dropna().unique():
            mask = regime_col == regime_value
            dates = self.regimes.index[mask]
            if len(dates) > 0:
                periods[regime_value] = dates

        return periods

    def backtest_by_regime(
        self,
        regime_type: str,
        backtest_fn: Callable = None,
    ) -> dict:
        """
        Run backtests for each regime value separately.

        Args:
            regime_type: One of 'trend', 'volatility', 'risk_appetite', 'market_phase'
            backtest_fn: Optional custom backtest function

        Returns:
            dict with metrics per regime value
        """
        from backtest import run_backtest

        periods = self._get_regime_periods(regime_type)
        results = {}

        for regime_value, dates in periods.items():
            print(f"\n{'='*60}")
            print(f"Backtesting {regime_type.upper()} = {regime_value.upper()}")
            print(f"Period: {dates.min()} to {dates.max()} ({len(dates)} days)")
            print(f"{'='*60}")

            # Filter predictions to this regime period
            if self.predictions_df is not None:
                pred_dates = pd.to_datetime(self.predictions_df["Date"])
                mask = pred_dates.isin(dates)
                regime_preds = self.predictions_df[mask].copy()

                if len(regime_preds) < 10:
                    print(f"  Skipping: only {len(regime_preds)} predictions in this regime")
                    results[regime_value] = {
                        "n_days": len(dates),
                        "n_predictions": len(regime_preds),
                        "skipped": True,
                        "reason": "insufficient_data",
                    }
                    continue

                # Save filtered predictions temporarily
                temp_path = Path("logs") / "regime_predictions_temp.csv"
                regime_preds.to_csv(temp_path, index=False)

            try:
                if backtest_fn:
                    trades, metrics, _ = backtest_fn()
                else:
                    # Use default backtest with regime filter disabled
                    # to see raw performance in this regime
                    trades, metrics, _ = run_backtest(
                        window_days=None,
                        use_regime_filter=False,
                        use_weekly_trend=False,
                    )

                results[regime_value] = {
                    "n_days": len(dates),
                    "n_trades": metrics.get("trades", 0),
                    "total_return": metrics.get("total_return", 0),
                    "sharpe": metrics.get("sharpe", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "avg_return": metrics.get("avg_return", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                }

                print(f"\n  Results for {regime_value}:")
                print(f"    Trades: {metrics.get('trades', 0)}")
                print(f"    Return: {metrics.get('total_return', 0):.2%}")
                print(f"    Sharpe: {metrics.get('sharpe', 0):.2f}")
                print(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")

            except Exception as e:
                print(f"  Error: {e}")
                results[regime_value] = {
                    "n_days": len(dates),
                    "error": str(e),
                }

        self.results[regime_type] = results
        return results

    def analyze_regime_transitions(self) -> pd.DataFrame:
        """
        Analyze model performance around regime transitions.

        Returns:
            DataFrame showing performance before/after regime changes
        """
        if self.regimes is None:
            self.detect_regimes()

        transitions = []
        for regime_type in ["trend", "volatility", "risk_appetite"]:
            regime_col = self.regimes[regime_type]
            changes = regime_col != regime_col.shift(1)
            change_dates = self.regimes.index[changes]

            for date in change_dates:
                if pd.isna(regime_col.shift(1).get(date)):
                    continue
                transitions.append({
                    "date": date,
                    "regime_type": regime_type,
                    "from_regime": regime_col.shift(1).get(date),
                    "to_regime": regime_col.get(date),
                })

        return pd.DataFrame(transitions)

    def get_current_regime(self) -> dict:
        """Get the most recent regime classification."""
        if self.regimes is None:
            self.detect_regimes()

        latest = self.regimes.iloc[-1]
        return {
            "date": str(self.regimes.index[-1]),
            "trend": latest["trend"],
            "volatility": latest["volatility"],
            "risk_appetite": latest["risk_appetite"],
            "market_phase": latest["market_phase"],
        }

    def generate_report(self, output_path: str = None) -> dict:
        """
        Generate comprehensive regime analysis report.

        Args:
            output_path: Optional path to save JSON report

        Returns:
            dict with full regime analysis
        """
        if self.regimes is None:
            self.detect_regimes()

        report = {
            "current_regime": self.get_current_regime(),
            "regime_distribution": {},
            "backtest_results": self.results,
            "recommendations": [],
        }

        # Regime distribution
        for regime_type in ["trend", "volatility", "risk_appetite", "market_phase"]:
            counts = self.regimes[regime_type].value_counts()
            report["regime_distribution"][regime_type] = counts.to_dict()

        # Generate recommendations based on results
        recommendations = []

        if "volatility" in self.results:
            vol_results = self.results["volatility"]
            if "high" in vol_results and "low" in vol_results:
                high_sharpe = vol_results["high"].get("sharpe", 0)
                low_sharpe = vol_results["low"].get("sharpe", 0)

                if high_sharpe < low_sharpe - 0.5:
                    recommendations.append({
                        "type": "volatility_dependent",
                        "finding": "Model performs significantly worse in high volatility",
                        "action": "Consider reducing position sizes or using tighter stops in high vol regimes",
                        "metrics": {"high_vol_sharpe": high_sharpe, "low_vol_sharpe": low_sharpe},
                    })

        if "trend" in self.results:
            trend_results = self.results["trend"]
            if "bear" in trend_results:
                bear_return = trend_results["bear"].get("total_return", 0)
                if bear_return < -0.1:
                    recommendations.append({
                        "type": "trend_dependent",
                        "finding": "Model loses money in bear markets",
                        "action": "Consider adding bear market filters or switching to short-only in bear regimes",
                        "metrics": {"bear_return": bear_return},
                    })

        report["recommendations"] = recommendations

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Saved regime analysis report to {output_path}")

        return report


def run_regime_analysis(
    price_data_path: str = "data/SPY.csv",
    predictions_path: str = "logs/daily_predictions.csv",
    output_dir: str = "logs/regime_analysis",
):
    """
    Run full regime analysis pipeline.

    Args:
        price_data_path: Path to price data CSV
        predictions_path: Path to predictions CSV
        output_dir: Directory for output files
    """
    from utils import load_SPY_data

    print("Loading data...")
    price_df = load_SPY_data()

    predictions_df = None
    if Path(predictions_path).exists():
        predictions_df = pd.read_csv(predictions_path)
        print(f"Loaded {len(predictions_df)} predictions")

    rb = RegimeBacktester(price_df, predictions_df)

    print("\nDetecting regimes...")
    regimes = rb.detect_regimes()

    # Save regime classifications
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    regimes.to_csv(output_path / "regime_classifications.csv")

    print("\nCurrent regime:")
    current = rb.get_current_regime()
    for k, v in current.items():
        print(f"  {k}: {v}")

    # Backtest each regime type
    for regime_type in ["trend", "volatility", "risk_appetite"]:
        print(f"\n{'='*80}")
        print(f"ANALYZING {regime_type.upper()} REGIMES")
        print(f"{'='*80}")
        rb.backtest_by_regime(regime_type)

    # Generate report
    report = rb.generate_report(str(output_path / "regime_report.json"))

    # Print recommendations
    if report["recommendations"]:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for rec in report["recommendations"]:
            print(f"\n[{rec['type']}]")
            print(f"  Finding: {rec['finding']}")
            print(f"  Action: {rec['action']}")

    return rb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regime-specific backtest analysis")
    parser.add_argument(
        "--regime-type",
        choices=["trend", "volatility", "risk_appetite", "market_phase", "all"],
        default="all",
        help="Which regime type to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/regime_analysis",
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_regime_analysis(output_dir=args.output_dir)
