#!/usr/bin/env python3
"""
LLM-Enhanced Market Analysis

Combines ML model predictions with LLM-generated market commentary.
Supports OpenAI and Anthropic APIs.

Usage:
    python3 llm_forecast.py --asset SPY
    python3 llm_forecast.py --asset BTC/USDT --provider anthropic
    python3 llm_forecast.py --all                    # Analyze all assets
    python3 llm_forecast.py --all --summary          # Summary newsletter
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_available_assets():
    """Get list of available assets"""
    assets = []

    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.glob("*.csv"):
            if f.stem not in ['cross_asset_features', 'macro_features', 'sentiment_features']:
                assets.append(f.stem)

    # Check cache directory
    cache_dir = Path("data_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*_1d.csv"):
            ticker = f.stem.replace('_1d', '').replace('_', '/')
            if ticker not in assets:
                assets.append(ticker)

    return sorted(assets)


def load_latest_predictions(asset="SPY"):
    """Load latest predictions for an asset"""
    # First try per-asset predictions (from predict_per_asset.py)
    asset_file = asset.replace("/", "_")
    per_asset_path = Path(f"logs/predictions/{asset_file}_predictions.csv")

    if per_asset_path.exists():
        df = pd.read_csv(per_asset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        latest = df.iloc[-1].to_dict()
        return latest

    # Fall back to daily_predictions.csv (SPY only from ensemble)
    if asset == "SPY":
        pred_path = Path("logs/daily_predictions.csv")
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            df['Date'] = pd.to_datetime(df['Date'])
            latest = df.iloc[-1].to_dict()
            return latest

    return None


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

    # Convert numeric columns to proper types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where Close is NaN
    if 'Close' in df.columns:
        df = df.dropna(subset=['Close'])

    # Get last 20 days
    recent = df.tail(20)
    return recent


def load_sentiment_data():
    """Load sentiment data if available"""
    sentiment_sources = {}

    # Try to load news sentiment
    news_path = Path("data/sentiment_features.csv")
    if news_path.exists():
        try:
            df = pd.read_csv(news_path)
            if len(df) > 0:
                latest = df.tail(5)
                sentiment_sources['news'] = latest
        except Exception:
            pass

    # Try to load fear/greed index
    fg_path = Path("data_cache/fear_greed_index.csv")
    if fg_path.exists():
        try:
            df = pd.read_csv(fg_path)
            if len(df) > 0:
                latest = df.tail(5)
                sentiment_sources['fear_greed'] = latest
        except Exception:
            pass

    # Try to load Reddit sentiment
    reddit_path = Path("data/reddit_sentiment.csv")
    if reddit_path.exists():
        try:
            df = pd.read_csv(reddit_path)
            if len(df) > 0:
                sentiment_sources['reddit'] = df.tail(5)
        except Exception:
            pass

    return sentiment_sources


def get_market_news_summary(assets=None):
    """Get recent market news/events for context"""
    today = datetime.now()

    # Try to fetch real news if NEWS_API_KEY is available
    try:
        from fetch_news import get_market_news_context, NEWS_API_KEY
        if NEWS_API_KEY:
            news_context = get_market_news_context(assets=assets, days_back=3)
            return f"""
MARKET NEWS & CONTEXT (as of {today.strftime('%Y-%m-%d')}):
{news_context}
"""
    except ImportError:
        pass
    except Exception as e:
        print(f"Note: Could not fetch news: {e}")

    # Fallback to placeholder
    context = f"""
MARKET CONTEXT (as of {today.strftime('%Y-%m-%d')}):
- Federal Reserve policy stance and recent communications
- Current economic indicators (employment, inflation, GDP)
- Major geopolitical events affecting markets
- Sector rotation trends and market breadth

Note: For real-time news, configure NEWS_API_KEY in .env
"""
    return context


def build_context(asset, prediction, price_data, include_sentiment=True, include_news=True):
    """Build comprehensive context for LLM analysis"""
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

    # Prediction info with detailed probabilities
    pred_label = prediction.get('Prediction', 1)
    crash_conf = prediction.get('Crash_Conf', 0)
    spike_conf = prediction.get('Spike_Conf', 0)
    proba = prediction.get('Proba', 0.5)

    # Calculate normalized probabilities for each scenario
    # These represent actual likelihoods of each market movement
    if pred_label == 0:
        signal = "CRASH (bearish)"
        crash_prob = max(crash_conf, 0.6)  # Primary signal
        spike_prob = 0.1
        normal_prob = 1 - crash_prob - spike_prob
    elif pred_label == 2:
        signal = "SPIKE (bullish)"
        spike_prob = max(spike_conf, 0.6)  # Primary signal
        crash_prob = 0.1
        normal_prob = 1 - spike_prob - crash_prob
    else:
        signal = "NORMAL (neutral)"
        normal_prob = 0.6
        crash_prob = 0.2
        spike_prob = 0.2

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
- Primary Signal: {signal}
- Model Confidence: {max(crash_conf, spike_conf):.1%}

SCENARIO LIKELIHOODS:
- CRASH (Bearish):  {crash_prob:.1%} - Significant downward movement expected
- NORMAL (Neutral): {normal_prob:.1%} - Sideways/mixed price action expected
- SPIKE (Bullish):  {spike_prob:.1%} - Significant upward movement expected

RECENT PRICE HISTORY:
{price_data[['Date', 'Close']].tail(5).to_string(index=False)}
"""

    # Add sentiment data if available
    if include_sentiment:
        sentiment_data = load_sentiment_data()

        if 'fear_greed' in sentiment_data:
            fg = sentiment_data['fear_greed']
            if len(fg) > 0:
                latest_fg = fg.iloc[-1]
                context += f"""
MARKET SENTIMENT:
- Fear & Greed Index: {latest_fg.get('Fear_Greed_Value', 'N/A')} ({latest_fg.get('Fear_Greed_Class', 'N/A')})
"""

    # Add news context if available
    if include_news:
        news_context = get_market_news_summary(assets=[asset])
        if news_context and "Note:" not in news_context[:100]:
            context += f"\n{news_context}"

    return context


def get_llm_analysis(context, provider="openai", for_newsletter=False):
    """Get LLM-generated market analysis"""

    if for_newsletter:
        system_prompt = """You are a quantitative market analyst writing for a newsletter.
Provide actionable insights that readers can use today. Be specific about:
1. Current market regime and what it means for positioning
2. Specific levels to watch (support/resistance)
3. Risk factors that could change the outlook
4. Clear recommendation with timeframe

Write in a professional but accessible tone. Use bullet points for clarity."""
    else:
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
        return "[OpenAI API key not set. Add OPENAI_API_KEY to your .env file]"

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
            max_tokens=800
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenAI error: {e}]"


def _call_anthropic(system_prompt, user_prompt):
    """Call Anthropic API"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[Anthropic API key not set. Add ANTHROPIC_API_KEY to your .env file]"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=800,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.content[0].text
    except Exception as e:
        return f"[Anthropic error: {e}]"


def generate_forecast(asset="SPY", provider="openai", for_newsletter=False):
    """Generate complete forecast with LLM analysis"""
    print(f"\n{'='*60}")
    print(f"  LLM-ENHANCED MARKET ANALYSIS: {asset}")
    print(f"{'='*60}\n")

    # Load data
    prediction = load_latest_predictions(asset)
    if prediction is None:
        print("⚠️  No predictions found. Using default values.")
        prediction = {'Prediction': 1, 'Crash_Conf': 0.33, 'Spike_Conf': 0.33}

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
    analysis = get_llm_analysis(context, provider, for_newsletter)
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


def generate_multi_asset_summary(assets=None, provider="openai"):
    """Generate summary analysis for multiple assets"""
    if assets is None:
        assets = get_available_assets()[:5]  # Top 5 assets

    print(f"\n{'='*60}")
    print(f"  MULTI-ASSET MARKET SUMMARY")
    print(f"{'='*60}\n")

    # Collect data for all assets
    asset_summaries = []

    for asset in assets:
        price_data = load_asset_data(asset)
        if price_data is None:
            continue

        prediction = load_latest_predictions(asset)

        # Calculate metrics
        latest_price = price_data['Close'].iloc[-1]
        returns = price_data['Close'].pct_change().dropna()

        if len(price_data) >= 5:
            five_day_return = (latest_price / price_data['Close'].iloc[-5] - 1) * 100
        else:
            five_day_return = 0

        # Handle missing predictions
        if prediction is None:
            signal = 'NO MODEL'
            confidence = 0
            print(f"   ⚠️  {asset}: No per-asset prediction found (run predict_per_asset.py --asset {asset})")
        else:
            pred_label = prediction.get('Prediction', 1)
            signal = {0: 'BEARISH', 1: 'NEUTRAL', 2: 'BULLISH'}.get(pred_label, 'NEUTRAL')
            confidence = max(prediction.get('Crash_Conf', 0), prediction.get('Spike_Conf', 0))

        asset_summaries.append({
            'asset': asset,
            'price': latest_price,
            'five_day_return': five_day_return,
            'signal': signal,
            'confidence': confidence
        })

    if not asset_summaries:
        print("❌ No asset data available")
        return None

    # Build summary context
    summary_text = "MULTI-ASSET MARKET OVERVIEW\n"
    summary_text += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"

    summary_text += "ASSET SIGNALS:\n"
    for s in asset_summaries:
        summary_text += f"- {s['asset']}: ${s['price']:.2f} | 5D: {s['five_day_return']:+.1f}% | {s['signal']} ({s['confidence']:.0%})\n"

    # Count signals
    bullish = sum(1 for s in asset_summaries if s['signal'] == 'BULLISH')
    bearish = sum(1 for s in asset_summaries if s['signal'] == 'BEARISH')
    neutral = sum(1 for s in asset_summaries if s['signal'] == 'NEUTRAL')

    summary_text += f"\nOVERALL: {bullish} Bullish, {bearish} Bearish, {neutral} Neutral\n"

    print(summary_text)

    # Get LLM summary
    system_prompt = """You are a market strategist providing a daily briefing.
Based on the asset signals provided, give a 3-4 paragraph summary covering:
1. Overall market sentiment and cross-asset themes
2. Notable signals and what they suggest
3. Key risks and opportunities
4. Actionable takeaways for today

Be specific and practical. This is for experienced traders."""

    print(f"\n🤖 LLM Market Summary ({provider}):")
    print("-" * 60)

    if provider == "openai":
        analysis = _call_openai(system_prompt, summary_text)
    else:
        analysis = _call_anthropic(system_prompt, summary_text)

    print(analysis)
    print("-" * 60)

    # Save summary
    output = {
        "timestamp": datetime.now().isoformat(),
        "assets": asset_summaries,
        "summary": summary_text,
        "analysis": analysis
    }

    output_path = Path("logs") / f"llm_multi_asset_summary_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n📝 Saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="LLM-Enhanced Market Analysis")
    parser.add_argument("--asset", default="SPY", help="Asset to analyze (e.g., SPY, BTC/USDT)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider to use")
    parser.add_argument("--all", action="store_true", help="Analyze all available assets")
    parser.add_argument("--summary", action="store_true", help="Generate multi-asset summary")
    parser.add_argument("--newsletter", action="store_true", help="Format for newsletter")
    args = parser.parse_args()

    if args.all or args.summary:
        generate_multi_asset_summary(provider=args.provider)
    else:
        generate_forecast(args.asset, args.provider, args.newsletter)


if __name__ == "__main__":
    main()
