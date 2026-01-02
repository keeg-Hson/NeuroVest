#!/usr/bin/env python3
"""
Over/Undervalued Asset Detector

Analyzes assets using multiple valuation metrics:
- Price vs moving averages (50/200-day)
- RSI (Relative Strength Index)
- Z-score vs historical mean
- Bollinger Bands position
- Rate of change

Usage:
    python3 valuation_detector.py
    python3 valuation_detector.py --asset BTC/USDT
    python3 valuation_detector.py --all
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    deltas = prices.diff()

    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_zscore(prices, period=252):
    """Calculate z-score (standard deviations from mean)"""
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()

    zscore = (prices - rolling_mean) / rolling_std

    return zscore


def calculate_bollinger_position(prices, period=20, num_std=2):
    """
    Calculate position within Bollinger Bands.
    Returns 0-1 where 0=lower band, 0.5=middle, 1=upper band
    """
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    # Position within bands (0 to 1)
    position = (prices - lower_band) / (upper_band - lower_band)

    return position.clip(0, 1)


def analyze_valuation(asset="SPY"):
    """
    Comprehensive valuation analysis for an asset.

    Returns:
        dict: Valuation metrics and signals
    """
    # Load data
    if asset == "SPY":
        data_path = Path("data/SPY.csv")
    else:
        asset_file = asset.replace("/", "_") + "_1d.csv"
        data_path = Path(f"data_cache/{asset_file}")

    if not data_path.exists():
        return None, f"Data not found: {data_path}"

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if len(df) < 200:
        return None, f"Insufficient data (need 200+ days, have {len(df)})"

    # Calculate indicators
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['ZScore'] = calculate_zscore(df['Close'], period=252)
    df['BB_Position'] = calculate_bollinger_position(df['Close'])

    # Get latest values
    latest = df.iloc[-1]
    price = latest['Close']

    # Calculate metrics
    ma50_pct = ((price / latest['MA_50']) - 1) * 100 if pd.notna(latest['MA_50']) else 0
    ma200_pct = ((price / latest['MA_200']) - 1) * 100 if pd.notna(latest['MA_200']) else 0
    rsi = latest['RSI']
    zscore = latest['ZScore']
    bb_position = latest['BB_Position']

    # Recent volatility
    returns = df['Close'].pct_change().dropna()
    volatility = returns.tail(30).std() * np.sqrt(252) * 100

    # Rate of change (1 month)
    if len(df) >= 30:
        roc_30d = ((price / df['Close'].iloc[-30]) - 1) * 100
    else:
        roc_30d = 0

    # Valuation signals
    signals = []
    valuation_score = 0  # -1 (undervalued) to +1 (overvalued)

    # RSI signals
    if pd.notna(rsi):
        if rsi > 70:
            signals.append("RSI: Overbought (>70)")
            valuation_score += 0.3
        elif rsi < 30:
            signals.append("RSI: Oversold (<30)")
            valuation_score -= 0.3
        else:
            signals.append(f"RSI: Neutral ({rsi:.1f})")

    # Moving average signals
    if ma200_pct > 20:
        signals.append("Price >20% above 200-MA: Extended")
        valuation_score += 0.25
    elif ma200_pct < -20:
        signals.append("Price >20% below 200-MA: Depressed")
        valuation_score -= 0.25

    # Z-score signals
    if pd.notna(zscore):
        if zscore > 2:
            signals.append("Z-Score >2: Statistically expensive")
            valuation_score += 0.25
        elif zscore < -2:
            signals.append("Z-Score <-2: Statistically cheap")
            valuation_score -= 0.25

    # Bollinger Band signals
    if pd.notna(bb_position):
        if bb_position > 0.9:
            signals.append("Near upper Bollinger Band: Overbought")
            valuation_score += 0.2
        elif bb_position < 0.1:
            signals.append("Near lower Bollinger Band: Oversold")
            valuation_score -= 0.2

    # Overall valuation classification
    if valuation_score > 0.5:
        classification = "OVERVALUED"
        recommendation = "Consider taking profits or reducing position"
    elif valuation_score < -0.5:
        classification = "UNDERVALUED"
        recommendation = "Consider accumulating or opening position"
    elif valuation_score > 0.2:
        classification = "SLIGHTLY OVERVALUED"
        recommendation = "Monitor for exit opportunities"
    elif valuation_score < -0.2:
        classification = "SLIGHTLY UNDERVALUED"
        recommendation = "Consider gradual accumulation"
    else:
        classification = "FAIRLY VALUED"
        recommendation = "No strong valuation signal"

    analysis = {
        'asset': asset,
        'price': price,
        'valuation_score': valuation_score,
        'classification': classification,
        'recommendation': recommendation,
        'metrics': {
            'rsi': rsi,
            'zscore': zscore,
            'bb_position': bb_position,
            'ma50_deviation': ma50_pct,
            'ma200_deviation': ma200_pct,
            'roc_30d': roc_30d,
            'volatility': volatility
        },
        'signals': signals,
        'timestamp': datetime.now().isoformat()
    }

    return analysis, None


def analyze_all_assets():
    """Analyze all available assets"""
    from main import get_available_assets

    assets = get_available_assets()

    results = []

    print(f"\n{'=' * 80}")
    print(f"VALUATION ANALYSIS - {len(assets)} ASSETS")
    print(f"{'=' * 80}\n")

    for asset in assets:
        analysis, error = analyze_valuation(asset)

        if error:
            print(f"❌ {asset:15s}: {error}")
            continue

        results.append(analysis)

        # Print summary
        score = analysis['valuation_score']
        classification = analysis['classification']
        price = analysis['price']

        # Color coding
        if score > 0.5:
            emoji = "🔴"
        elif score < -0.5:
            emoji = "🟢"
        elif abs(score) > 0.2:
            emoji = "🟡"
        else:
            emoji = "⚪"

        print(f"{emoji} {asset:15s} ${price:>10.2f}  |  {classification:20s}  |  Score: {score:+.2f}")

    return results


def format_valuation_report(analysis):
    """Format single asset valuation analysis"""
    asset = analysis['asset']
    price = analysis['price']
    score = analysis['valuation_score']
    classification = analysis['classification']
    recommendation = analysis['recommendation']
    metrics = analysis['metrics']
    signals = analysis['signals']

    report = f"""
{'=' * 70}
VALUATION ANALYSIS: {asset}
{'=' * 70}

CURRENT PRICE: ${price:.2f}
CLASSIFICATION: {classification}
VALUATION SCORE: {score:+.2f} (-1 = undervalued, +1 = overvalued)

RECOMMENDATION:
{recommendation}

{'=' * 70}
TECHNICAL INDICATORS
{'=' * 70}

Price vs Moving Averages:
  vs 50-day MA:  {metrics['ma50_deviation']:+.2f}%
  vs 200-day MA: {metrics['ma200_deviation']:+.2f}%

Momentum & Oscillators:
  RSI (14):           {metrics['rsi']:.1f} {'(Overbought)' if metrics['rsi'] > 70 else '(Oversold)' if metrics['rsi'] < 30 else '(Neutral)'}
  Z-Score:            {metrics['zscore']:.2f}
  Bollinger Position: {metrics['bb_position']:.1%}

Recent Performance:
  30-day ROC:         {metrics['roc_30d']:+.2f}%
  30-day Volatility:  {metrics['volatility']:.1f}%

{'=' * 70}
SIGNALS
{'=' * 70}
"""

    for signal in signals:
        report += f"  • {signal}\n"

    report += f"\n{'=' * 70}\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Over/Undervalued Asset Detector")
    parser.add_argument("--asset", help="Analyze specific asset (e.g., SPY, BTC/USDT)")
    parser.add_argument("--all", action="store_true", help="Analyze all assets")
    parser.add_argument("--save", action="store_true", help="Save analysis to JSON")

    args = parser.parse_args()

    if args.all:
        results = analyze_all_assets()

        # Summary statistics
        if results:
            print(f"\n{'=' * 80}")
            print("SUMMARY")
            print(f"{'=' * 80}\n")

            overvalued = [r for r in results if r['valuation_score'] > 0.5]
            undervalued = [r for r in results if r['valuation_score'] < -0.5]
            neutral = [r for r in results if abs(r['valuation_score']) <= 0.5]

            print(f"Overvalued:   {len(overvalued):2d} assets")
            print(f"Undervalued:  {len(undervalued):2d} assets")
            print(f"Fairly valued: {len(neutral):2d} assets")

            if undervalued:
                print(f"\nMost Undervalued:")
                sorted_under = sorted(undervalued, key=lambda x: x['valuation_score'])[:5]
                for r in sorted_under:
                    print(f"  {r['asset']:15s}: {r['valuation_score']:+.2f}  (${r['price']:.2f})")

            if overvalued:
                print(f"\nMost Overvalued:")
                sorted_over = sorted(overvalued, key=lambda x: x['valuation_score'], reverse=True)[:5]
                for r in sorted_over:
                    print(f"  {r['asset']:15s}: {r['valuation_score']:+.2f}  (${r['price']:.2f})")

            print(f"\n{'=' * 80}\n")

            # Save if requested
            if args.save:
                import json
                output_dir = Path("logs")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = output_dir / f"valuation_analysis_{timestamp}.json"

                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                print(f"💾 Analysis saved to: {output_path}")

    elif args.asset:
        print(f"\n🔍 Analyzing {args.asset}...")

        analysis, error = analyze_valuation(args.asset)

        if error:
            print(f"\n❌ Error: {error}")
            return

        # Display report
        report = format_valuation_report(analysis)
        print(report)

        # Save if requested
        if args.save:
            import json
            output_dir = Path("logs")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            asset_file = args.asset.replace("/", "_")
            output_path = output_dir / f"valuation_{asset_file}_{timestamp}.json"

            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

            print(f"💾 Analysis saved to: {output_path}")

    else:
        # Default: analyze SPY
        print("\n🔍 Analyzing SPY...")

        analysis, error = analyze_valuation("SPY")

        if error:
            print(f"\n❌ Error: {error}")
            return

        report = format_valuation_report(analysis)
        print(report)


if __name__ == "__main__":
    main()
