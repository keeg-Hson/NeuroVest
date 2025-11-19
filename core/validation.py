"""
Proper Validation Framework for Trading Strategies

This module implements rigorous validation methodologies to prevent overfitting
and ensure realistic performance estimates.

Methodologies:
1. Train/Test/Validation Split
2. Walk-Forward Analysis
3. Monte Carlo Simulation
4. Regime Analysis
5. Statistical Significance Testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class ValidationResult:
    """Results from a validation run"""
    method: str
    train_period: str
    test_period: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    std_trade_return: float
    profitable: bool
    notes: str


@dataclass
class TradeResult:
    """Individual trade result"""
    entry_date: datetime
    exit_date: datetime
    ticker: str
    return_pct: float
    profit_loss: float
    position_size: float


class TrainTestSplit:
    """
    Implements proper train/test/validation split for time series data

    Usage:
        splitter = TrainTestSplit(data, train_pct=0.6, val_pct=0.2, test_pct=0.2)
        train_data = splitter.get_train()
        val_data = splitter.get_validation()
        test_data = splitter.get_test()
    """

    def __init__(self, data: pd.DataFrame, train_pct: float = 0.6,
                 val_pct: float = 0.2, test_pct: float = 0.2):
        """
        Initialize train/test/validation split

        Args:
            data: Time series data (must have datetime index)
            train_pct: Percentage for training (default 60%)
            val_pct: Percentage for validation (default 20%)
            test_pct: Percentage for testing (default 20%)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")

        if abs(train_pct + val_pct + test_pct - 1.0) > 0.001:
            raise ValueError("Percentages must sum to 1.0")

        self.data = data.sort_index()
        total_days = len(data)

        # Calculate split points
        train_end = int(total_days * train_pct)
        val_end = int(total_days * (train_pct + val_pct))

        # Split data
        self.train_data = data.iloc[:train_end]
        self.validation_data = data.iloc[train_end:val_end]
        self.test_data = data.iloc[val_end:]

        print(f"Data split:")
        print(f"  Train: {self.train_data.index[0]} to {self.train_data.index[-1]} ({len(self.train_data)} days)")
        print(f"  Validation: {self.validation_data.index[0]} to {self.validation_data.index[-1]} ({len(self.validation_data)} days)")
        print(f"  Test: {self.test_data.index[0]} to {self.test_data.index[-1]} ({len(self.test_data)} days)")

    def get_train(self) -> pd.DataFrame:
        """Get training data"""
        return self.train_data

    def get_validation(self) -> pd.DataFrame:
        """Get validation data"""
        return self.validation_data

    def get_test(self) -> pd.DataFrame:
        """Get test data (only use once!)"""
        warnings.warn("Using test data! Only evaluate on test data ONCE at the end.", UserWarning)
        return self.test_data


class WalkForwardAnalysis:
    """
    Implements walk-forward analysis for more robust validation

    Walk-forward analysis trains on a window of data, tests on the next period,
    then rolls the window forward. This simulates how a strategy would perform
    if retrained regularly.

    Usage:
        wfa = WalkForwardAnalysis(data, train_months=12, test_months=3)
        results = wfa.run(backtest_function)
    """

    def __init__(self, data: pd.DataFrame, train_months: int = 12,
                 test_months: int = 3, step_months: int = 3):
        """
        Initialize walk-forward analysis

        Args:
            data: Time series data (must have datetime index)
            train_months: Number of months to train on
            test_months: Number of months to test on
            step_months: Number of months to step forward (default = test_months)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")

        self.data = data.sort_index()
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

        # Calculate windows
        self.windows = self._calculate_windows()
        print(f"Walk-forward analysis: {len(self.windows)} windows")

    def _calculate_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Calculate train/test windows"""
        windows = []
        start_date = self.data.index[0]
        end_date = self.data.index[-1]

        current_date = start_date
        while current_date + pd.DateOffset(months=self.train_months + self.test_months) <= end_date:
            train_start = current_date
            train_end = current_date + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Ensure dates exist in data
            train_end = self.data.index[self.data.index <= train_end][-1]
            test_start = self.data.index[self.data.index >= test_start][0]
            test_end = self.data.index[self.data.index <= test_end][-1]

            windows.append((train_start, train_end, test_start, test_end))

            # Step forward
            current_date += pd.DateOffset(months=self.step_months)

        return windows

    def run(self, strategy_func: Callable, **kwargs) -> List[ValidationResult]:
        """
        Run walk-forward analysis

        Args:
            strategy_func: Function that takes (train_data, test_data, **kwargs)
                         and returns ValidationResult
            **kwargs: Additional arguments to pass to strategy_func

        Returns:
            List of ValidationResult objects, one per window
        """
        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(self.windows):
            print(f"\nWindow {i+1}/{len(self.windows)}")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {test_end.date()}")

            train_data = self.data[train_start:train_end]
            test_data = self.data[test_start:test_end]

            result = strategy_func(train_data, test_data, **kwargs)
            result.method = f"Walk-Forward Window {i+1}"
            result.train_period = f"{train_start.date()} to {train_end.date()}"
            result.test_period = f"{test_start.date()} to {test_end.date()}"

            results.append(result)

            print(f"  Return: {result.total_return:.2%}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Drawdown: {result.max_drawdown:.2%}")

        # Print summary
        print(f"\n{'='*60}")
        print("Walk-Forward Analysis Summary")
        print(f"{'='*60}")
        avg_return = np.mean([r.total_return for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_drawdown = np.mean([r.max_drawdown for r in results])
        win_pct = sum([1 for r in results if r.profitable]) / len(results)

        print(f"Average Return: {avg_return:.2%}")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Average Drawdown: {avg_drawdown:.2%}")
        print(f"Profitable Windows: {win_pct:.1%} ({sum([1 for r in results if r.profitable])}/{len(results)})")
        print(f"{'='*60}")

        return results


class MonteCarloSimulation:
    """
    Monte Carlo simulation for assessing luck vs. skill

    Randomly shuffles trade order and re-runs backtest many times to see
    if results are consistent or depend on specific trade sequence.

    Usage:
        mc = MonteCarloSimulation(trades)
        results = mc.run(iterations=1000)
        mc.analyze_results(results)
    """

    def __init__(self, trades: List[TradeResult]):
        """
        Initialize Monte Carlo simulation

        Args:
            trades: List of individual trade results
        """
        self.trades = trades
        self.trade_returns = [t.return_pct for t in trades]

    def run(self, iterations: int = 1000, initial_capital: float = 100000) -> Dict:
        """
        Run Monte Carlo simulation

        Args:
            iterations: Number of simulation runs
            initial_capital: Starting capital

        Returns:
            Dictionary with simulation results
        """
        print(f"Running Monte Carlo simulation ({iterations} iterations)...")

        final_values = []
        max_drawdowns = []
        sharpe_ratios = []

        for i in range(iterations):
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1}/{iterations}")

            # Shuffle trade order
            shuffled_returns = np.random.choice(self.trade_returns,
                                               size=len(self.trade_returns),
                                               replace=False)

            # Calculate equity curve
            capital = initial_capital
            equity_curve = [capital]

            for ret in shuffled_returns:
                capital *= (1 + ret)
                equity_curve.append(capital)

            # Calculate metrics
            final_value = equity_curve[-1]
            final_values.append(final_value)

            # Calculate max drawdown
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns.append(max_dd)

            # Calculate Sharpe ratio
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        results = {
            'final_values': final_values,
            'max_drawdowns': max_drawdowns,
            'sharpe_ratios': sharpe_ratios,
            'initial_capital': initial_capital,
            'iterations': iterations
        }

        return results

    def analyze_results(self, results: Dict, confidence_level: float = 0.95):
        """
        Analyze Monte Carlo results

        Args:
            results: Results from run()
            confidence_level: Confidence level for intervals (default 95%)
        """
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        print(f"\n{'='*60}")
        print(f"Monte Carlo Analysis ({results['iterations']} iterations)")
        print(f"{'='*60}")

        # Final value analysis
        final_values = results['final_values']
        initial = results['initial_capital']

        print(f"\nFinal Portfolio Value:")
        print(f"  Mean: ${np.mean(final_values):,.0f} ({(np.mean(final_values)/initial - 1)*100:.1f}%)")
        print(f"  Median: ${np.median(final_values):,.0f} ({(np.median(final_values)/initial - 1)*100:.1f}%)")
        print(f"  Std Dev: ${np.std(final_values):,.0f}")
        print(f"  {confidence_level*100:.0f}% CI: ${np.percentile(final_values, lower_percentile):,.0f} to ${np.percentile(final_values, upper_percentile):,.0f}")

        # Probability of profit
        prob_profit = sum([1 for v in final_values if v > initial]) / len(final_values)
        print(f"  Probability of Profit: {prob_profit:.1%}")

        # Max drawdown analysis
        max_drawdowns = results['max_drawdowns']
        print(f"\nMaximum Drawdown:")
        print(f"  Mean: {np.mean(max_drawdowns)*100:.1f}%")
        print(f"  Median: {np.median(max_drawdowns)*100:.1f}%")
        print(f"  {confidence_level*100:.0f}% CI: {np.percentile(max_drawdowns, lower_percentile)*100:.1f}% to {np.percentile(max_drawdowns, upper_percentile)*100:.1f}%")
        print(f"  Worst Case: {np.max(max_drawdowns)*100:.1f}%")

        # Sharpe ratio analysis
        sharpe_ratios = results['sharpe_ratios']
        print(f"\nSharpe Ratio:")
        print(f"  Mean: {np.mean(sharpe_ratios):.2f}")
        print(f"  Median: {np.median(sharpe_ratios):.2f}")
        print(f"  {confidence_level*100:.0f}% CI: {np.percentile(sharpe_ratios, lower_percentile):.2f} to {np.percentile(sharpe_ratios, upper_percentile):.2f}")

        print(f"{'='*60}")

        # Statistical significance
        if prob_profit < 0.60:
            print(f"\n⚠️  WARNING: Probability of profit is only {prob_profit:.1%}")
            print("   Strategy may not have a genuine edge (could be luck)")

        if np.percentile(max_drawdowns, upper_percentile) > 0.30:
            print(f"\n⚠️  WARNING: 95th percentile drawdown is {np.percentile(max_drawdowns, upper_percentile)*100:.1f}%")
            print("   Unacceptably high drawdown risk")

        return results


class StatisticalSignificance:
    """
    Test statistical significance of trading results

    Determines whether results are likely due to skill or random chance
    """

    @staticmethod
    def z_test(trades: List[TradeResult], expected_win_rate: float = 0.50) -> Dict:
        """
        Z-test for win rate statistical significance

        Args:
            trades: List of trade results
            expected_win_rate: Expected win rate under null hypothesis (default 50%)

        Returns:
            Dictionary with test results
        """
        n = len(trades)
        wins = sum([1 for t in trades if t.return_pct > 0])
        win_rate = wins / n

        # Z-test
        p = expected_win_rate
        z = (win_rate - p) / np.sqrt(p * (1 - p) / n)

        # Two-tailed p-value
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(win_rate)) - np.arcsin(np.sqrt(p)))

        print(f"\n{'='*60}")
        print("Statistical Significance Test (Win Rate)")
        print(f"{'='*60}")
        print(f"Sample Size: {n} trades")
        print(f"Observed Win Rate: {win_rate:.1%}")
        print(f"Expected Win Rate (null): {expected_win_rate:.1%}")
        print(f"Z-Score: {z:.2f}")
        print(f"P-Value: {p_value:.4f}")
        print(f"Effect Size (Cohen's h): {h:.3f}")

        if p_value < 0.05:
            print(f"\n✓ SIGNIFICANT at p < 0.05")
            if win_rate > expected_win_rate:
                print(f"  Win rate is significantly HIGHER than chance")
            else:
                print(f"  Win rate is significantly LOWER than chance")
        else:
            print(f"\n✗ NOT SIGNIFICANT (p = {p_value:.4f})")
            print(f"  Cannot reject null hypothesis (results may be due to chance)")

        if n < 100:
            print(f"\n⚠️  WARNING: Sample size ({n}) is small. Need 100+ trades for reliable results.")

        print(f"{'='*60}")

        return {
            'n': n,
            'wins': wins,
            'win_rate': win_rate,
            'expected_win_rate': expected_win_rate,
            'z_score': z,
            'p_value': p_value,
            'effect_size': h,
            'significant': p_value < 0.05
        }

    @staticmethod
    def sharpe_ratio_significance(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
        """
        Test if Sharpe ratio is significantly different from zero

        Args:
            returns: Array of period returns
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Dictionary with test results
        """
        n = len(returns)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            print("ERROR: Zero standard deviation, cannot calculate Sharpe ratio")
            return {}

        sharpe = mean_excess / std_excess * np.sqrt(252)  # Annualized

        # T-test for Sharpe ratio
        t_stat = np.sqrt(n) * sharpe / np.sqrt(1 + 0.5 * sharpe**2)

        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

        print(f"\n{'='*60}")
        print("Statistical Significance Test (Sharpe Ratio)")
        print(f"{'='*60}")
        print(f"Sample Size: {n} periods")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"T-Statistic: {t_stat:.2f}")
        print(f"P-Value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"\n✓ SIGNIFICANT at p < 0.05")
            print(f"  Sharpe ratio is significantly different from zero")
        else:
            print(f"\n✗ NOT SIGNIFICANT (p = {p_value:.4f})")
            print(f"  Cannot reject null hypothesis (returns may be due to chance)")

        print(f"{'='*60}")

        return {
            'n': n,
            'sharpe_ratio': sharpe,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


# Example usage and testing
if __name__ == "__main__":
    print("Validation Framework Loaded")
    print("\nExample usage:")
    print("""
    # 1. Train/Test Split
    splitter = TrainTestSplit(data, train_pct=0.6, val_pct=0.2, test_pct=0.2)
    train = splitter.get_train()
    val = splitter.get_validation()
    test = splitter.get_test()  # Only use ONCE!

    # 2. Walk-Forward Analysis
    wfa = WalkForwardAnalysis(data, train_months=12, test_months=3)
    results = wfa.run(my_backtest_function)

    # 3. Monte Carlo
    mc = MonteCarloSimulation(trades)
    mc_results = mc.run(iterations=1000)
    mc.analyze_results(mc_results)

    # 4. Statistical Significance
    sig = StatisticalSignificance()
    sig.z_test(trades, expected_win_rate=0.50)
    sig.sharpe_ratio_significance(returns)
    """)
