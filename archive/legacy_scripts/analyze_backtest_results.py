#!/usr/bin/env python3
"""
Detailed Backtest Analysis
Provides real trading metrics beyond just accuracy
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_latest_backtest():
    """Load most recent backtest results"""
    latest_path = Path("logs/latest.json")
    if not latest_path.exists():
        raise FileNotFoundError("No backtest results found. Run: python3 backtest.py")

    with open(latest_path) as f:
        return json.load(f)

def analyze_metrics(results):
    """Analyze and explain backtest metrics"""
    metrics = results.get("metrics", {})

    print("=" * 80)
    print("BACKTEST RESULTS ANALYSIS")
    print("=" * 80)

    # Basic Stats
    print("\n📊 BASIC STATISTICS")
    print("-" * 80)
    trades = metrics.get("trades", 0)
    total_return = metrics.get("total_return", 0) * 100
    sharpe = metrics.get("sharpe", 0)
    max_dd = metrics.get("max_drawdown", 0) * 100
    win_rate = metrics.get("win_rate", 0) * 100

    print(f"Total Trades:      {trades:,}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Total Return:      {total_return:.2f}%")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")

    # Annualized metrics
    ann_return = metrics.get("annualized_return")
    if ann_return:
        print(f"Annualized Return: {ann_return * 100:.2f}%")

    # Risk Metrics
    print("\n📉 RISK ANALYSIS")
    print("-" * 80)
    avg_return = metrics.get("avg_return", 0) * 100
    median_return = metrics.get("median_return", 0) * 100
    profit_factor = metrics.get("profit_factor", 0)

    print(f"Average Trade:     {avg_return:.3f}%")
    print(f"Median Trade:      {median_return:.3f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")

    # Interpretations
    print("\n💡 INTERPRETATION")
    print("-" * 80)

    # Sharpe Ratio
    if sharpe < 0.5:
        sharpe_grade = "❌ Poor (< 0.5) - Returns barely justify risk"
    elif sharpe < 1.0:
        sharpe_grade = "⚠️  Fair (0.5-1.0) - Moderate risk-adjusted returns"
    elif sharpe < 2.0:
        sharpe_grade = "✅ Good (1.0-2.0) - Strong risk-adjusted returns"
    else:
        sharpe_grade = "🌟 Excellent (> 2.0) - Outstanding risk-adjusted returns"
    print(f"Sharpe Ratio: {sharpe_grade}")

    # Win Rate
    if win_rate < 45:
        wr_grade = "❌ Low - Losing more than half of trades"
    elif win_rate < 55:
        wr_grade = "⚠️  Neutral - Close to 50/50"
    elif win_rate < 65:
        wr_grade = "✅ Good - Winning majority of trades"
    else:
        wr_grade = "🌟 Excellent - High win rate"
    print(f"Win Rate:     {wr_grade}")

    # Max Drawdown
    if abs(max_dd) > 30:
        dd_grade = "❌ High risk - Large potential losses"
    elif abs(max_dd) > 20:
        dd_grade = "⚠️  Moderate risk - Notable drawdowns"
    elif abs(max_dd) > 10:
        dd_grade = "✅ Acceptable - Manageable drawdowns"
    else:
        dd_grade = "🌟 Low risk - Small drawdowns"
    print(f"Max Drawdown: {dd_grade}")

    # Profit Factor
    if profit_factor < 1.0:
        pf_grade = "❌ Losing strategy - Losses exceed profits"
    elif profit_factor < 1.5:
        pf_grade = "⚠️  Barely profitable - Thin margins"
    elif profit_factor < 2.0:
        pf_grade = "✅ Profitable - Decent profit margins"
    else:
        pf_grade = "🌟 Very profitable - Strong profit margins"
    print(f"Profit Factor: {pf_grade}")

    # Trading Activity
    print("\n📈 TRADING ACTIVITY")
    print("-" * 80)

    # Estimate trading frequency
    # Assuming ~252 trading days per year and 25 years of data
    days = 252 * 25
    trades_per_year = trades / 25 if trades > 0 else 0
    trades_per_month = trades_per_year / 12

    print(f"Trades per Year:   ~{trades_per_year:.0f}")
    print(f"Trades per Month:  ~{trades_per_month:.0f}")
    print(f"Hold Period:       ~2.5 days (from backtest config)")

    if trades_per_month < 1:
        activity = "❌ Very Low - Not enough trading opportunities"
    elif trades_per_month < 4:
        activity = "⚠️  Low - Only a few trades per month"
    elif trades_per_month < 12:
        activity = "✅ Moderate - Regular trading opportunities"
    else:
        activity = "🌟 High - Frequent trading"
    print(f"Activity Level:    {activity}")

    # Final Grade
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    score = 0
    if sharpe > 1.0: score += 1
    if win_rate > 55: score += 1
    if abs(max_dd) < 20: score += 1
    if profit_factor > 1.5: score += 1
    if total_return > 20: score += 1

    if score >= 4:
        grade = "🌟 EXCELLENT - Strong strategy with good metrics"
    elif score >= 3:
        grade = "✅ GOOD - Decent strategy, some room for improvement"
    elif score >= 2:
        grade = "⚠️  FAIR - Marginal strategy, needs optimization"
    else:
        grade = "❌ POOR - Strategy needs significant work"

    print(f"\nScore: {score}/5")
    print(f"Grade: {grade}")

    # Specific recommendations
    print("\n📝 RECOMMENDATIONS")
    print("-" * 80)

    if sharpe < 1.0:
        print("• Low Sharpe: Consider adding stop-losses or position sizing")
    if win_rate < 50:
        print("• Low Win Rate: Model may be too aggressive, adjust threshold")
    if abs(max_dd) > 25:
        print("• High Drawdown: Implement stricter risk management")
    if profit_factor < 1.5:
        print("• Low Profit Factor: Optimize entry/exit timing or fees")
    if trades_per_month < 2:
        print("• Low Activity: Model is too conservative, lower threshold")
    if trades_per_month > 20:
        print("• High Activity: May be overtrading, increase threshold")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        results = load_latest_backtest()
        analyze_metrics(results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error analyzing backtest: {e}")
        import traceback
        traceback.print_exc()
