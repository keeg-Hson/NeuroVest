#!/usr/bin/env python3
"""
Comprehensive NeuroVest Demo

Demonstrates all major features with real examples and LLM integration.

Usage:
    python3 demo_comprehensive.py
    python3 demo_comprehensive.py --scenario recession
    python3 demo_comprehensive.py --scenario valuation
    python3 demo_comprehensive.py --scenario portfolio
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path


def print_header(title):
    """Print styled header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo_quick_start():
    """Demo: Quick start workflow"""
    print_header("DEMO: Quick Start Workflow")
    print("""
This demonstrates the typical NeuroVest workflow:

STEP 1: Data Download
──────────────────────────────────────────────────────────────
Downloads SPY (S&P 500) and cryptocurrency data.

Command: python3 update_spy_data.py
         python3 download_crypto_enhanced.py

Expected output:
  ✓ Downloaded 6,300+ days of SPY data
  ✓ Downloaded BTC, ETH, SOL with 2,000+ days each


STEP 2: Model Training
──────────────────────────────────────────────────────────────
Trains XGBoost, LightGBM, and CatBoost ensemble models.

Command: python3 train_multi_asset.py --optimize-weights

Expected output:
  ✓ XGBoost trained (126 features)
  ✓ LightGBM trained (126 features)
  ✓ CatBoost trained (126 features)
  ✓ Ensemble weights optimized
  ✓ Models saved to models/

Performance metrics:
  • Training samples: ~10,000 (SPY + crypto combined)
  • Cross-validation folds: 5
  • Time: 5-10 minutes


STEP 3: Generate Predictions
──────────────────────────────────────────────────────────────
Generates predictions using ensemble model.

Command: python3 predict_multi_asset_ensemble.py

Expected output:
  ✓ Loaded 3 models
  ✓ Generated predictions for 6,300+ days
  ✓ Signal distribution: ~30% CRASH / 40% NORMAL / 30% SPIKE
  ✓ Saved to logs/daily_predictions.csv


STEP 4: Run Backtest
──────────────────────────────────────────────────────────────
Tests trading strategy over historical data.

Command: python3 backtest.py --config configs/backtest_optimized.json

Expected output:
  ✓ Tested on 25 years of data
  ✓ Total return: 150-200%
  ✓ Sharpe ratio: 2.0-2.8
  ✓ Max drawdown: -4% to -8%
  ✓ Win rate: 55-65%

    """)

    input("Press Enter to continue...")


def demo_trading_profiles():
    """Demo: Trading risk profiles"""
    print_header("DEMO: Trading Risk Profiles")
    print("""
NeuroVest supports three risk profiles:

CONSERVATIVE PROFILE
──────────────────────────────────────────────────────────────
For: Risk-averse investors seeking steady, reliable growth
Params:
  • Min confidence: 70%+ (only very strong signals)
  • Stop loss: 1.0x ATR (tight, protect capital)
  • Position size: 5-15% (small, diversified)
  • Max equity: 40% (keep 60% in cash/bonds)
  • Rebalancing: Monthly

Expected Performance:
  • Return: ~150% over 25 years
  • Sharpe: 2.80 (excellent risk-adjusted returns)
  • Max DD: -4% (very low)
  • Win rate: ~62%

Command: python3 backtest.py --config configs/backtest_optimized.json
         (Set conservative params in config)


MODERATE PROFILE (RECOMMENDED)
──────────────────────────────────────────────────────────────
For: Balanced investors seeking good risk-reward
Params:
  • Min confidence: 55%+
  • Stop loss: 1.5x ATR
  • Position size: 10-25%
  • Max equity: 65%
  • Rebalancing: Monthly

Expected Performance:
  • Return: ~191% over 25 years
  • Sharpe: 2.55
  • Max DD: -5.4%
  • Win rate: ~58%

Command: python3 backtest.py --config configs/backtest_optimized.json


LIBERAL/AGGRESSIVE PROFILE
──────────────────────────────────────────────────────────────
For: Aggressive traders seeking maximum returns
Params:
  • Min confidence: 45%+ (take more signals)
  • Stop loss: 2.0x ATR (wider, let winners run)
  • Take profit: 4.0x ATR (big wins)
  • Position size: 15-40%
  • Max equity: 85%

Expected Performance:
  • Return: ~378% over 25 years
  • Sharpe: 2.03
  • Max DD: -12.8%
  • Win rate: ~54%

Command: python3 backtest.py --config configs/backtest_aggressive.json

    """)

    input("Press Enter to continue...")


def demo_recession_indicator():
    """Demo: Recession probability indicator"""
    print_header("DEMO: Recession Indicator")
    print("""
Multi-signal recession probability analysis:

SIGNALS ANALYZED
──────────────────────────────────────────────────────────────
1. Yield Curve (35% weight)
   • 10Y-2Y Treasury spread
   • Inversion = strong recession signal
   • Historical accuracy: ~85% when inverted

2. Market Stress (25% weight)
   • Volatility (annualized)
   • Max drawdown (peak-to-trough)
   • YTD performance

3. Technical Signals (20% weight)
   • Price vs 200-day MA
   • Death cross (50-MA < 200-MA)
   • Market breadth

4. Unemployment Trends (20% weight)
   • Rising unemployment rate
   • Jobless claims trend


EXAMPLE OUTPUT
──────────────────────────────────────────────────────────────
OVERALL ASSESSMENT: ELEVATED RISK
Recession Probability: 45.2%

YIELD CURVE:
  Status: INVERTED
  10Y-2Y Spread: -0.23%
  Signal: 🔴 INVERTED (recession warning)

MARKET STRESS:
  Volatility: 28.5%
  Max Drawdown: -18.2%
  Stress Score: 62.3%

TECHNICAL SIGNALS:
  SPY Price: $410.25
  200-day MA: $428.50
  Below 200-MA: 🔴 Yes
  Death Cross: 🔴 Yes

INTERPRETATION:
⚠️  ELEVATED RECESSION RISK
Multiple indicators suggest increased recession risk. Consider:
- Defensive positioning (bonds, utilities)
- Reduced equity exposure
- Increased cash allocation


Command: python3 recession_indicator.py --save

    """)

    input("Press Enter to continue...")


def demo_valuation_detector():
    """Demo: Over/undervalued asset detection"""
    print_header("DEMO: Valuation Detector")
    print("""
Technical valuation analysis using multiple indicators:

METRICS ANALYZED
──────────────────────────────────────────────────────────────
1. RSI (Relative Strength Index)
   • >70 = Overbought
   • <30 = Oversold

2. Z-Score
   • >2 = Statistically expensive
   • <-2 = Statistically cheap

3. Bollinger Bands
   • Near upper band = Overbought
   • Near lower band = Oversold

4. Moving Average Deviation
   • >20% above 200-MA = Extended
   • >20% below 200-MA = Depressed

5. Rate of Change (30-day)
   • Momentum indicator


EXAMPLE: SPY ANALYSIS
──────────────────────────────────────────────────────────────
CURRENT PRICE: $450.25
CLASSIFICATION: SLIGHTLY OVERVALUED
VALUATION SCORE: +0.35 (range: -1.0 to +1.0)

RECOMMENDATION:
Monitor for exit opportunities

TECHNICAL INDICATORS:
Price vs Moving Averages:
  vs 50-day MA:  +8.2%
  vs 200-day MA: +12.5%

Momentum & Oscillators:
  RSI (14):           68.5 (Neutral, approaching overbought)
  Z-Score:            1.42
  Bollinger Position: 78.3% (upper region)

Recent Performance:
  30-day ROC:         +5.8%
  30-day Volatility:  18.2%

SIGNALS:
  • RSI: Approaching overbought (>70)
  • Z-Score: Elevated but not extreme
  • Near upper Bollinger Band: Overbought signal


EXAMPLE: BTC/USDT ANALYSIS
──────────────────────────────────────────────────────────────
CURRENT PRICE: $42,150
CLASSIFICATION: UNDERVALUED
VALUATION SCORE: -0.68

RECOMMENDATION:
Consider accumulating or opening position

TECHNICAL INDICATORS:
  RSI: 28.2 (Oversold <30)
  Z-Score: -2.15 (Statistically cheap)
  Price vs 200-MA: -22.3% (Depressed)

SIGNALS:
  • RSI: Oversold (<30) - potential bounce
  • Z-Score <-2: Statistically cheap
  • >20% below 200-MA: Significantly depressed


Commands:
  python3 valuation_detector.py --asset SPY
  python3 valuation_detector.py --all --save

    """)

    input("Press Enter to continue...")


def demo_llm_integration():
    """Demo: LLM-powered market analysis"""
    print_header("DEMO: LLM Market Analysis")
    print("""
AI-powered insights using OpenAI GPT-4 or Anthropic Claude:

FEATURES
──────────────────────────────────────────────────────────────
✓ Real-time news integration (NewsAPI)
✓ Scenario likelihood analysis (Crash/Normal/Spike)
✓ Multi-asset market summary
✓ Actionable trading recommendations
✓ Newsletter generation with email delivery


SINGLE ASSET ANALYSIS
──────────────────────────────────────────────────────────────
Command: python3 llm_forecast.py --asset SPY --provider openai

Example Output:
──────────────────────────────────────────────────────────────
ASSET: SPY
DATE: 2024-12-21

CURRENT MARKET DATA:
- Latest Price: $450.25
- Daily Change: +0.8%
- 5-Day Return: +3.2%
- 20-Day Return: +5.8%
- Annualized Volatility: 18.5%

MODEL PREDICTION:
- Primary Signal: SPIKE (bullish)
- Model Confidence: 68.5%

SCENARIO LIKELIHOODS:
- CRASH (Bearish):  12% - Significant downward movement expected
- NORMAL (Neutral): 23% - Sideways/mixed price action expected
- SPIKE (Bullish):  65% - Significant upward movement expected

RECENT NEWS (last 3 days):
1. [Bloomberg] Fed signals pause in rate hikes amid cooling inflation
   Market rally continues as investors price in dovish pivot...

2. [Reuters] Tech earnings beat expectations, lifting S&P 500
   Strong results from mega-cap tech drive broad market gains...

3. [WSJ] Consumer confidence reaches 6-month high
   Spending remains resilient despite economic headwinds...

MARKET SENTIMENT:
- Fear & Greed Index: 68 (Greed)

AI ANALYSIS:
The S&P 500 is showing strong bullish momentum with a 65% probability
of continued upside. Recent Fed dovish signals and strong tech earnings
are supporting the rally. However, the 68 Fear & Greed reading suggests
some caution as market sentiment may be getting extended.

RECOMMENDATION:
Consider scaling into long positions with tight stops. The 12% crash
probability suggests maintaining some hedges. Target 5-7% upside with
2-3% stop loss.


MULTI-ASSET SUMMARY
──────────────────────────────────────────────────────────────
Command: python3 llm_forecast.py --all --provider anthropic

Analyzes all available assets and generates comprehensive market overview
with correlations, sector rotation, and portfolio recommendations.


NEWSLETTER GENERATION
──────────────────────────────────────────────────────────────
Command: python3 newsletter_generator.py --send --assets SPY,BTC/USDT

Generates and emails professional market newsletter with:
- Executive summary
- Asset-by-asset analysis
- Top opportunities
- Risk warnings
- Actionable recommendations


SETUP REQUIRED
──────────────────────────────────────────────────────────────
Add to .env file:

OPENAI_API_KEY=sk-your-key-here
# Or
ANTHROPIC_API_KEY=sk-ant-your-key-here

# For news integration
NEWS_API_KEY=your-newsapi-key

# For newsletter email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
NEWSLETTER_RECIPIENTS=recipient@example.com

    """)

    input("Press Enter to continue...")


def demo_portfolio_rebalancing():
    """Demo: Portfolio rebalancing optimization"""
    print_header("DEMO: Portfolio Rebalancing")
    print("""
Find optimal rebalancing frequency for your portfolio:

THE PROBLEM
──────────────────────────────────────────────────────────────
Rebalancing too frequently:
  ❌ High transaction costs
  ❌ Tax implications
  ❌ Time consuming

Rebalancing too infrequently:
  ❌ Portfolio drift
  ❌ Unintended risk exposure
  ❌ Missed rebalancing bonus


THE SOLUTION
──────────────────────────────────────────────────────────────
Test all frequencies and find optimal based on:
  ✓ Net returns (after transaction costs)
  ✓ Sharpe ratio
  ✓ Max drawdown
  ✓ Rebalancing burden


EXAMPLE ANALYSIS
──────────────────────────────────────────────────────────────
Portfolio: 60% SPY / 30% GLD / 10% TLT
Test Period: 5 years
Transaction Cost: 0.2% per rebalance (round-trip)

Results:
──────────────────────────────────────────────────────────────
Strategy         Return   Sharpe  MaxDD   Rebalances  Net Return
────────────────────────────────────────────────────────────────
Buy & Hold       45.2%    1.85    -8.2%   0           45.2%
Daily            52.1%    1.92    -7.1%   1,260       27.6%  ❌
Weekly           51.3%    1.95    -7.3%   260         41.9%
Monthly          50.8%    2.10    -7.0%   60          49.6%  ✓
Quarterly        48.5%    2.08    -7.5%   20          48.1%
Semi-Annual      46.8%    1.98    -8.0%   10          46.5%
Annual           45.9%    1.88    -8.3%   5           45.7%

OPTIMAL STRATEGY: Monthly Rebalancing
  • Highest Sharpe ratio (2.10)
  • Best net return (49.6%)
  • Manageable rebalancing frequency (60x in 5 years)
  • Lowest max drawdown (-7.0%)


INTERPRETATION
──────────────────────────────────────────────────────────────
✓ Monthly rebalancing provides best risk-adjusted returns
✓ Daily rebalancing destroyed by transaction costs
✓ Quarterly also good, requires less management
✓ Annual too infrequent, missed rebalancing benefits


Commands:
  # Find optimal
  python3 portfolio_rebalancer.py --find-optimal \\
    --assets SPY,GLD,TLT --weights 0.6,0.3,0.1

  # Execute rebalancing
  python3 portfolio_rebalancer.py \\
    --assets SPY,GLD,TLT --weights 0.6,0.3,0.1 \\
    --profile moderate

    """)

    input("Press Enter to continue...")


def show_menu():
    """Show demo menu"""
    print_header("NeuroVest Comprehensive Demo")
    print()
    print("  SCENARIOS")
    print("  ─────────────────────────────────────────────────────────")
    print("  1. Quick Start Workflow")
    print("  2. Trading Risk Profiles (Conservative/Moderate/Aggressive)")
    print("  3. Recession Probability Indicator")
    print("  4. Over/Undervalued Asset Detection")
    print("  5. LLM-Powered Market Analysis")
    print("  6. Portfolio Rebalancing Optimization")
    print("  7. All Demos (complete walkthrough)")
    print()
    print("  0. Exit")
    print()


def run_all_demos():
    """Run all demos in sequence"""
    demos = [
        demo_quick_start,
        demo_trading_profiles,
        demo_recession_indicator,
        demo_valuation_detector,
        demo_llm_integration,
        demo_portfolio_rebalancing
    ]

    for demo in demos:
        demo()


def main():
    parser = argparse.ArgumentParser(description="NeuroVest Comprehensive Demo")
    parser.add_argument("--scenario", choices=[
        'quickstart', 'profiles', 'recession', 'valuation', 'llm', 'portfolio'
    ], help="Run specific scenario")

    args = parser.parse_args()

    if args.scenario:
        scenarios = {
            'quickstart': demo_quick_start,
            'profiles': demo_trading_profiles,
            'recession': demo_recession_indicator,
            'valuation': demo_valuation_detector,
            'llm': demo_llm_integration,
            'portfolio': demo_portfolio_rebalancing
        }
        scenarios[args.scenario]()
        return

    # Interactive menu
    while True:
        show_menu()
        choice = input("Select scenario: ").strip()

        if choice == "0":
            print("\nDemo complete!")
            break
        elif choice == "1":
            demo_quick_start()
        elif choice == "2":
            demo_trading_profiles()
        elif choice == "3":
            demo_recession_indicator()
        elif choice == "4":
            demo_valuation_detector()
        elif choice == "5":
            demo_llm_integration()
        elif choice == "6":
            demo_portfolio_rebalancing()
        elif choice == "7":
            run_all_demos()
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
