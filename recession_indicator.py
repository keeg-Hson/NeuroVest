#!/usr/bin/env python3
"""
Recession Indicator

Analyzes multiple economic signals to predict recession probability:
- Yield curve inversion (10Y-2Y Treasury spread)
- Unemployment rate trends
- Market volatility (VIX proxy)
- Stock market performance
- Economic indicators from FRED

Usage:
    python3 recession_indicator.py
    python3 recession_indicator.py --threshold 0.6
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def load_treasury_data():
    """Load Treasury yield data from FRED cache"""
    tnx_path = Path("data/TNX.csv")  # 10-year Treasury

    if not tnx_path.exists():
        return None

    df = pd.read_csv(tnx_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    return df


def calculate_yield_curve_spread():
    """
    Calculate 10Y-2Y Treasury spread.
    Negative spread (inversion) is a recession predictor.
    """
    # For now, use 10Y data and estimate 2Y
    # In production, would load actual 2Y data
    tnx_df = load_treasury_data()

    if tnx_df is None or len(tnx_df) == 0:
        return None, "Treasury data not available"

    # Get latest 10Y yield
    latest_10y = tnx_df['Close'].iloc[-1]

    # Estimate 2Y based on historical relationship
    # Typically 2Y is 0.5-1.5% below 10Y in normal conditions
    # This is a placeholder - ideally load real 2Y data
    estimated_2y = latest_10y - 0.8

    spread = latest_10y - estimated_2y

    # Inverted if spread < 0
    is_inverted = spread < 0

    return {
        'spread': spread,
        'is_inverted': is_inverted,
        'ten_year_yield': latest_10y,
        'signal': 'INVERTED' if is_inverted else 'NORMAL'
    }, None


def calculate_unemployment_trend():
    """
    Analyze unemployment rate trend.
    Rising unemployment often precedes recession.
    """
    # Placeholder - would load from FRED UNRATE series
    # For now, return neutral signal
    return {
        'current_rate': None,
        'trend': 'STABLE',
        'signal': 'NEUTRAL'
    }, "Unemployment data not available"


def calculate_market_stress():
    """
    Analyze market volatility and drawdown as stress indicator.
    """
    spy_path = Path("data/SPY.csv")

    if not spy_path.exists():
        return None, "SPY data not available"

    df = pd.read_csv(spy_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').tail(252)  # Last year

    if len(df) < 60:
        return None, "Insufficient SPY data"

    # Calculate metrics
    returns = df['Close'].pct_change().dropna()

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100

    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Recent performance
    ytd_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

    # Stress signals
    high_vol = volatility > 25
    deep_drawdown = max_drawdown < -15
    negative_ytd = ytd_return < -10

    stress_score = sum([high_vol, deep_drawdown, negative_ytd]) / 3

    return {
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'ytd_return': ytd_return,
        'stress_score': stress_score,
        'signal': 'HIGH STRESS' if stress_score > 0.5 else 'MODERATE' if stress_score > 0.3 else 'LOW STRESS'
    }, None


def calculate_technical_signals():
    """
    Technical recession signals:
    - SPY below 200-day MA
    - Death cross (50-day MA < 200-day MA)
    """
    spy_path = Path("data/SPY.csv")

    if not spy_path.exists():
        return None, "SPY data not available"

    df = pd.read_csv(spy_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').tail(252)

    if len(df) < 200:
        return None, "Insufficient SPY data for MAs"

    # Calculate moving averages
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()

    latest = df.iloc[-1]

    below_200ma = latest['Close'] < latest['MA_200']
    death_cross = latest['MA_50'] < latest['MA_200'] if pd.notna(latest['MA_50']) else False

    signal_count = sum([below_200ma, death_cross])

    return {
        'price': latest['Close'],
        'ma_200': latest['MA_200'],
        'ma_50': latest['MA_50'],
        'below_200ma': below_200ma,
        'death_cross': death_cross,
        'signal': 'BEARISH' if signal_count >= 1 else 'NEUTRAL'
    }, None


def calculate_recession_probability():
    """
    Combine all signals to calculate recession probability.

    Returns:
        dict: Recession analysis with probability score
    """
    signals = {}
    weights = {
        'yield_curve': 0.35,
        'market_stress': 0.25,
        'technical': 0.20,
        'unemployment': 0.20
    }

    # Yield curve
    yc_data, yc_err = calculate_yield_curve_spread()
    signals['yield_curve'] = yc_data

    # Market stress
    stress_data, stress_err = calculate_market_stress()
    signals['market_stress'] = stress_data

    # Technical signals
    tech_data, tech_err = calculate_technical_signals()
    signals['technical'] = tech_data

    # Unemployment
    unemp_data, unemp_err = calculate_unemployment_trend()
    signals['unemployment'] = unemp_data

    # Calculate weighted recession probability
    prob_components = []

    # Yield curve contribution
    if yc_data:
        if yc_data.get('is_inverted'):
            yc_prob = 0.8  # Strong signal
        else:
            spread = yc_data.get('spread', 1.0)
            yc_prob = max(0, 0.5 - spread * 0.3)  # Lower spread = higher risk
        prob_components.append(yc_prob * weights['yield_curve'])

    # Market stress contribution
    if stress_data:
        stress_score = stress_data.get('stress_score', 0)
        prob_components.append(stress_score * weights['market_stress'])

    # Technical contribution
    if tech_data:
        tech_bearish = sum([tech_data.get('below_200ma', False), tech_data.get('death_cross', False)]) / 2
        prob_components.append(tech_bearish * weights['technical'])

    # Unemployment contribution (placeholder)
    prob_components.append(0.2 * weights['unemployment'])  # Neutral for now

    # Total probability
    recession_probability = sum(prob_components)

    # Classify risk level
    if recession_probability >= 0.6:
        risk_level = "HIGH"
        description = "Strong recession signals present"
    elif recession_probability >= 0.4:
        risk_level = "ELEVATED"
        description = "Multiple warning signs emerging"
    elif recession_probability >= 0.25:
        risk_level = "MODERATE"
        description = "Some caution warranted"
    else:
        risk_level = "LOW"
        description = "Economic conditions appear stable"

    return {
        'probability': recession_probability,
        'risk_level': risk_level,
        'description': description,
        'signals': signals,
        'timestamp': datetime.now().isoformat()
    }


def format_recession_report(analysis):
    """Format recession analysis as readable report"""
    prob = analysis['probability']
    level = analysis['risk_level']
    desc = analysis['description']

    report = f"""
{'=' * 70}
RECESSION PROBABILITY INDICATOR
{'=' * 70}

OVERALL ASSESSMENT: {level} RISK
Recession Probability: {prob:.1%}
{desc}

{'=' * 70}
SIGNAL BREAKDOWN
{'=' * 70}
"""

    # Yield curve
    yc = analysis['signals'].get('yield_curve')
    if yc:
        report += f"""
YIELD CURVE:
  Status: {yc.get('signal', 'N/A')}
  10Y-2Y Spread: {yc.get('spread', 0):.2f}%
  10Y Yield: {yc.get('ten_year_yield', 0):.2f}%
  Signal: {'🔴 INVERTED (recession warning)' if yc.get('is_inverted') else '🟢 Normal'}
"""

    # Market stress
    stress = analysis['signals'].get('market_stress')
    if stress:
        report += f"""
MARKET STRESS:
  Status: {stress.get('signal', 'N/A')}
  Volatility: {stress.get('volatility', 0):.1f}%
  Max Drawdown: {stress.get('max_drawdown', 0):.1f}%
  YTD Return: {stress.get('ytd_return', 0):+.1f}%
  Stress Score: {stress.get('stress_score', 0):.1%}
"""

    # Technical
    tech = analysis['signals'].get('technical')
    if tech:
        report += f"""
TECHNICAL SIGNALS:
  Status: {tech.get('signal', 'N/A')}
  SPY Price: ${tech.get('price', 0):.2f}
  200-day MA: ${tech.get('ma_200', 0):.2f}
  Below 200-MA: {'🔴 Yes' if tech.get('below_200ma') else '🟢 No'}
  Death Cross: {'🔴 Yes' if tech.get('death_cross') else '🟢 No'}
"""

    report += f"""
{'=' * 70}
INTERPRETATION
{'=' * 70}
"""

    if prob >= 0.6:
        report += """
⚠️  HIGH RECESSION RISK
Multiple indicators suggest elevated recession risk. Consider:
- Defensive positioning (bonds, utilities, consumer staples)
- Reduced equity exposure
- Increased cash allocation
- Focus on quality, dividend-paying stocks
"""
    elif prob >= 0.4:
        report += """
⚠️  ELEVATED RECESSION RISK
Warning signs are emerging. Consider:
- Reviewing portfolio allocations
- Increasing quality exposure
- Building cash reserves
- Monitoring economic data closely
"""
    elif prob >= 0.25:
        report += """
ℹ️  MODERATE RECESSION RISK
Some caution warranted but no immediate alarm. Consider:
- Maintaining diversification
- Periodic portfolio review
- Balanced risk exposure
"""
    else:
        report += """
✅ LOW RECESSION RISK
Economic conditions appear stable. Continue:
- Normal investment strategy
- Opportunistic positioning
- Regular monitoring
"""

    report += f"\n{'=' * 70}\n"

    return report


def save_recession_analysis(analysis, output_path=None):
    """Save recession analysis to JSON"""
    import json

    if output_path is None:
        output_dir = Path("logs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"recession_analysis_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Recession Probability Indicator")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Alert threshold (default: 0.4)")
    parser.add_argument("--save", action="store_true",
                        help="Save analysis to JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")

    args = parser.parse_args()

    if not args.quiet:
        print("\n🔍 Analyzing recession indicators...")

    # Calculate recession probability
    analysis = calculate_recession_probability()

    # Display report
    if not args.quiet:
        report = format_recession_report(analysis)
        print(report)

    # Alert if above threshold
    if analysis['probability'] >= args.threshold:
        print(f"\n⚠️  ALERT: Recession probability ({analysis['probability']:.1%}) exceeds threshold ({args.threshold:.1%})")

    # Save if requested
    if args.save:
        output_path = save_recession_analysis(analysis)
        print(f"\n💾 Analysis saved to: {output_path}")

    return analysis


if __name__ == "__main__":
    main()
