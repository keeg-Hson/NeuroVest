#!/usr/bin/env python3
"""
Real-time News Fetcher for Market Analysis

Fetches financial news from NewsAPI for LLM context and sentiment analysis.

Usage:
    python3 fetch_news.py                     # Fetch general market news
    python3 fetch_news.py --query "Bitcoin"   # Fetch specific topic news
    python3 fetch_news.py --save              # Save to data directory
"""

import os
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Default financial news sources
FINANCIAL_SOURCES = [
    "bloomberg",
    "business-insider",
    "financial-times",
    "the-wall-street-journal",
    "reuters",
    "cnbc",
    "fortune",
    "the-economist"
]

# Market-related keywords
MARKET_KEYWORDS = [
    "stock market", "S&P 500", "nasdaq", "dow jones",
    "federal reserve", "interest rates", "inflation",
    "earnings", "IPO", "merger", "acquisition",
    "cryptocurrency", "bitcoin", "ethereum",
    "oil prices", "gold", "commodities"
]


def fetch_news(query=None, days_back=3, max_articles=20, language="en"):
    """
    Fetch news articles from NewsAPI.

    Args:
        query: Search query (default: market news)
        days_back: How many days back to search
        max_articles: Maximum number of articles to return
        language: Language filter

    Returns:
        List of article dictionaries
    """
    if not NEWS_API_KEY:
        return {
            "error": "NEWS_API_KEY not set in .env file",
            "articles": []
        }

    # Default query for market news
    if not query:
        query = "stock market OR federal reserve OR S&P 500 OR nasdaq"

    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)

    params = {
        "q": query,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "language": language,
        "sortBy": "relevancy",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "ok":
            return {
                "error": data.get("message", "Unknown error"),
                "articles": []
            }

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "content": article.get("content", "")[:500] if article.get("content") else ""
            })

        return {
            "status": "ok",
            "total_results": data.get("totalResults", 0),
            "articles": articles,
            "query": query,
            "date_range": f"{from_date.date()} to {to_date.date()}"
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Request failed: {str(e)}",
            "articles": []
        }


def fetch_asset_news(asset, days_back=3, max_articles=10):
    """
    Fetch news for a specific asset.

    Args:
        asset: Asset ticker (e.g., "SPY", "BTC/USDT")
        days_back: Days back to search
        max_articles: Maximum articles

    Returns:
        News data dictionary
    """
    # Build query based on asset
    if asset == "SPY":
        query = "S&P 500 OR SPY ETF OR stock market index"
    elif "BTC" in asset:
        query = "Bitcoin OR BTC OR cryptocurrency"
    elif "ETH" in asset:
        query = "Ethereum OR ETH OR cryptocurrency"
    elif "SOL" in asset:
        query = "Solana OR SOL cryptocurrency"
    elif asset == "GLD":
        query = "gold prices OR gold ETF OR precious metals"
    elif asset == "TLT":
        query = "treasury bonds OR interest rates OR federal reserve"
    elif asset == "QQQ":
        query = "nasdaq OR tech stocks OR QQQ ETF"
    else:
        query = f"{asset} stock OR {asset} ETF"

    return fetch_news(query, days_back, max_articles)


def format_news_for_llm(news_data, max_articles=5):
    """
    Format news articles for LLM context.

    Args:
        news_data: News data from fetch_news()
        max_articles: Maximum articles to include

    Returns:
        Formatted string for LLM prompt
    """
    if "error" in news_data:
        return f"[News fetch error: {news_data['error']}]"

    articles = news_data.get("articles", [])[:max_articles]

    if not articles:
        return "[No recent news articles found]"

    formatted = f"RECENT NEWS ({news_data.get('date_range', 'last 3 days')}):\n"

    for i, article in enumerate(articles, 1):
        title = article.get("title", "No title")
        source = article.get("source", "Unknown")
        description = article.get("description", "")

        # Truncate description if too long
        if description and len(description) > 200:
            description = description[:200] + "..."

        formatted += f"\n{i}. [{source}] {title}\n"
        if description:
            formatted += f"   {description}\n"

    return formatted


def get_market_news_context(assets=None, days_back=3):
    """
    Get comprehensive market news context for LLM analysis.

    Args:
        assets: List of assets to get news for
        days_back: Days back to search

    Returns:
        Formatted news context string
    """
    context_parts = []

    # General market news
    general_news = fetch_news(
        "stock market OR federal reserve OR economic outlook",
        days_back=days_back,
        max_articles=5
    )
    context_parts.append("GENERAL MARKET NEWS:")
    context_parts.append(format_news_for_llm(general_news, max_articles=3))

    # Asset-specific news
    if assets:
        for asset in assets[:3]:  # Limit to 3 assets
            asset_news = fetch_asset_news(asset, days_back=days_back, max_articles=5)
            if asset_news.get("articles"):
                context_parts.append(f"\n{asset} SPECIFIC NEWS:")
                context_parts.append(format_news_for_llm(asset_news, max_articles=2))

    # Crypto market news (if crypto assets present)
    if assets and any("USDT" in str(a) or "BTC" in str(a) for a in assets):
        crypto_news = fetch_news(
            "cryptocurrency market OR bitcoin OR ethereum",
            days_back=days_back,
            max_articles=5
        )
        context_parts.append("\nCRYPTO MARKET NEWS:")
        context_parts.append(format_news_for_llm(crypto_news, max_articles=2))

    return "\n".join(context_parts)


def save_news_data(news_data, filename=None):
    """Save news data to JSON file."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    if not filename:
        filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(news_data, f, indent=2, default=str)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch financial news from NewsAPI")
    parser.add_argument("--query", help="Custom search query")
    parser.add_argument("--asset", help="Fetch news for specific asset")
    parser.add_argument("--days", type=int, default=3, help="Days back to search")
    parser.add_argument("--max", type=int, default=20, help="Maximum articles")
    parser.add_argument("--save", action="store_true", help="Save to data directory")
    parser.add_argument("--llm-format", action="store_true", help="Output in LLM-ready format")

    args = parser.parse_args()

    print("=" * 60)
    print("  NEWS FETCHER - Real-time Market News")
    print("=" * 60)

    if not NEWS_API_KEY:
        print("\n[!] NEWS_API_KEY not set in .env file")
        print("    Add: NEWS_API_KEY=your_key_here")
        return

    # Fetch news
    if args.asset:
        print(f"\nFetching news for: {args.asset}")
        news_data = fetch_asset_news(args.asset, args.days, args.max)
    elif args.query:
        print(f"\nSearch query: {args.query}")
        news_data = fetch_news(args.query, args.days, args.max)
    else:
        print("\nFetching general market news...")
        news_data = fetch_news(days_back=args.days, max_articles=args.max)

    # Check for errors
    if "error" in news_data:
        print(f"\n[!] Error: {news_data['error']}")
        return

    # Display results
    articles = news_data.get("articles", [])
    print(f"\nFound {news_data.get('total_results', 0)} total results")
    print(f"Showing {len(articles)} articles")
    print(f"Date range: {news_data.get('date_range', 'N/A')}")

    if args.llm_format:
        print("\n" + "-" * 60)
        print(format_news_for_llm(news_data, max_articles=10))
    else:
        print("\n" + "-" * 60)
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Published: {article['published_at']}")
            if article.get('description'):
                desc = article['description'][:150] + "..." if len(article['description']) > 150 else article['description']
                print(f"   {desc}")

    # Save if requested
    if args.save:
        output_path = save_news_data(news_data)
        print(f"\n[+] Saved to: {output_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
