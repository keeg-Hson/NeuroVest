#!/usr/bin/env python3
"""
LLM-Enhanced Market Analysis

Combines ML model predictions with LLM-generated market commentary.
Supports OpenAI and Anthropic APIs.

Usage:
    python3 llm_forecast.py --asset SPY
    python3 llm_forecast.py --asset BTC/USDT --provider anthropic
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_latest_predictions(asset="SPY"):
    """Load latest predictions for an asset"""
    pred_path = Path("logs/daily_predictions.csv")
    if not pred_path.exists():
        return None

    df = pd.read_csv(pred_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Get latest prediction
    latest = df.iloc[-1].to_dict()
    return latest

def load_asset_data(asset="SPY"):
    """Load recent price data for context"""
    if asset == "SPY":
        data_path = Path("data/SPY.csv")
    else:
        asset_file = asset.replace("/", "_") + "_1d.csv"
        data_path = Path(f"data_cache/{asset_file}")

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Get last 20 days
    recent = df.tail(20)
    return recent

def build_context(asset, prediction, price_data):
    """Build context for LLM analysis"""
    if price_data is None or len(price_data) == 0:
        return None

    # Calculate key metrics
    latest_price = price_data['Close'].iloc[-1]
    prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else latest_price
    daily_change = (latest_price / prev_price - 1) * 100

    # 5-day and 20-day returns
    if len(price_data) >= 5:
        five_day_return = (latest_price / price_data['Close'].iloc[-5] - 1) * 100
    else:
        five_day_return = 0

    twenty_day_return = (latest_price / price_data['Close'].iloc[0] - 1) * 100

    # Volatility
    returns = price_data['Close'].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5) * 100  # Annualized

    # Prediction info
    pred_label = prediction.get('Prediction', 1)
    if pred_label == 0:
        signal = "CRASH (bearish)"
    elif pred_label == 2:
        signal = "SPIKE (bullish)"
    else:
        signal = "NORMAL (neutral)"

    crash_conf = prediction.get('Crash_Conf', 0)
    spike_conf = prediction.get('Spike_Conf', 0)

    context = f"""
ASSET: {asset}
DATE: {datetime.now().strftime('%Y-%m-%d')}

CURRENT MARKET DATA:
- Latest Price: ${latest_price:.2f}
- Daily Change: {daily_change:+.2f}%
- 5-Day Return: {five_day_return:+.2f}%
- 20-Day Return: {twenty_day_return:+.2f}%
- Annualized Volatility: {volatility:.1f}%

MODEL PREDICTION:
- Signal: {signal}
- Crash Probability: {crash_conf:.1%}
- Spike Probability: {spike_conf:.1%}
- Confidence: {max(crash_conf, spike_conf):.1%}

RECENT PRICE HISTORY:
{price_data[['Date', 'Close']].tail(5).to_string(index=False)}
"""
    return context

def get_llm_analysis(context, provider="openai"):
    """Get LLM-generated market analysis"""

    system_prompt = """You are a quantitative market analyst. Analyze the provided market data and ML model predictions.
Provide a concise analysis covering:
1. Current market conditions (2-3 sentences)
2. Model signal interpretation (2-3 sentences)
3. Key risk factors to watch (2-3 bullet points)
4. Actionable insight (1-2 sentences)

Be direct and factual. Avoid excessive hedging language."""

    user_prompt = f"""Analyze this market data and provide trading insights:

{context}

Provide your analysis:"""

    if provider == "openai":
        return _call_openai(system_prompt, user_prompt)
    elif provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    else:
        return f"[LLM provider '{provider}' not supported. Set OPENAI_API_KEY or ANTHROPIC_API_KEY]"

def _call_openai(system_prompt, user_prompt):
    """Call OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[OpenAI API key not set. Export OPENAI_API_KEY to enable LLM analysis]"

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenAI error: {e}]"

def _call_anthropic(system_prompt, user_prompt):
    """Call Anthropic API"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[Anthropic API key not set. Export ANTHROPIC_API_KEY to enable LLM analysis]"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.content[0].text
    except Exception as e:
        return f"[Anthropic error: {e}]"

def generate_forecast(asset="SPY", provider="openai"):
    """Generate complete forecast with LLM analysis"""
    print(f"\n{'='*60}")
    print(f"  LLM-ENHANCED MARKET ANALYSIS: {asset}")
    print(f"{'='*60}\n")

    # Load data
    prediction = load_latest_predictions(asset)
    if prediction is None:
        print("❌ No predictions found. Run predict_multi_asset_ensemble.py first.")
        return None

    price_data = load_asset_data(asset)
    if price_data is None:
        print(f"❌ No price data found for {asset}.")
        return None

    # Build context
    context = build_context(asset, prediction, price_data)
    if context is None:
        print("❌ Could not build context.")
        return None

    print("📊 Market Context:")
    print(context)

    # Get LLM analysis
    print(f"\n🤖 LLM Analysis ({provider}):")
    print("-" * 60)
    analysis = get_llm_analysis(context, provider)
    print(analysis)
    print("-" * 60)

    # Save to file
    output = {
        "asset": asset,
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "analysis": analysis,
        "prediction": prediction
    }

    output_path = Path("logs") / f"llm_forecast_{asset.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n📝 Saved to: {output_path}")

    return output

def main():
    parser = argparse.ArgumentParser(description="LLM-Enhanced Market Analysis")
    parser.add_argument("--asset", default="SPY", help="Asset to analyze (e.g., SPY, BTC/USDT)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider to use")
    args = parser.parse_args()

    generate_forecast(args.asset, args.provider)

if __name__ == "__main__":
    main()
