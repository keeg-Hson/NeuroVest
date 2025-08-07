import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from textblob import TextBlob
from fredapi import Fred
import praw
from sklearn.preprocessing import MinMaxScaler



# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_ACTUAL_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# --- News Sentiment via NewsAPI + TextBlob ---
API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_news_sentiment(topic="stock market", days=7, page_size=50):
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "q": topic,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": API_KEY,
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
    except Exception as e:
        print(f"❌ Failed to fetch news: {e}")
        return pd.DataFrame()

    records = []
    for article in articles:
        date = article["publishedAt"][:10]
        text = (article.get("title") or "") + " " + (article.get("description") or "")
        sentiment = TextBlob(text).sentiment.polarity
        records.append({"date": date, "sentiment": sentiment})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    daily_sentiment = df.groupby("date").mean().reset_index()
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    daily_sentiment.rename(columns={"sentiment": "News_Sentiment"}, inplace=True)
    return daily_sentiment

# --- Reddit Sentiment via Pushshift + TextBlob ---

# Reddit setup
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_reddit_sentiment(subreddit="stocks", days=7, limit=100):
    """
    Fetch posts from Reddit using PRAW and compute sentiment via TextBlob.
    Returns daily average sentiment DataFrame.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    sentiments = []

    for submission in reddit.subreddit(subreddit).top(time_filter="week", limit=limit):
        if submission.created_utc < start_date.timestamp():
            continue

        date = datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d")
        text = submission.title + " " + (submission.selftext or "")
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append({"date": date, "sentiment": sentiment})

    if not sentiments:
        print("❌ No Reddit data found.")
        return pd.DataFrame()

    df = pd.DataFrame(sentiments)
    daily_sentiment = df.groupby("date").mean().reset_index()
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    daily_sentiment.rename(columns={"sentiment": "Reddit_Sentiment"}, inplace=True)

    return daily_sentiment

def add_reddit_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    reddit_df = fetch_reddit_sentiment()
    if reddit_df.empty:
        df["Reddit_Sentiment"] = np.nan
        return df

    return df.merge(reddit_df, left_on="Date", right_on="date", how="left").drop(columns=["date"])


# --- Macro Signals via FRED ---
def fetch_fred_macro_signals():
    series_ids = {
        "CPI": "CPIAUCSL",
        "Unemployment": "UNRATE",
        "InterestRate": "DFF",
        "YieldCurve": "T10Y2Y",
        "ConsumerSentiment": "UMCSENT",
        "IndustrialProduction": "INDPRO",
        "VIX": "VIXCLS"
    }

    frames = []
    for name, series_id in series_ids.items():
        try:
            data = fred.get_series(series_id)
            df = data.reset_index()
            df.columns = ["Date", name]
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Could not fetch {name}: {e}")

    macro_df = frames[0]
    for df in frames[1:]:
        macro_df = macro_df.merge(df, on="Date", how="outer")

    macro_df["Date"] = pd.to_datetime(macro_df["Date"])
    return macro_df

# --- Unified Signal Aggregator ---
def add_external_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fetch and merge FRED macro data
    macro_df = fetch_fred_macro_signals()
    df = df.merge(macro_df, on="Date", how="left")

    # Fetch and merge News sentiment
    news_df = fetch_news_sentiment()
    df = df.merge(news_df, left_on="Date", right_on="date", how="left").drop(columns=["date"], errors="ignore")

    # Fetch and merge Reddit sentiment
    reddit_df = fetch_reddit_sentiment()
    if not reddit_df.empty and "date" in reddit_df.columns:
        df = df.merge(reddit_df, left_on="Date", right_on="date", how="left").drop(columns=["date"], errors="ignore")
    else:
        print("⚠️ Skipping Reddit sentiment — no data available.")

    

    return df


def normalize_signals(df: pd.DataFrame, signal_columns: list) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()

    for col in signal_columns:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
        else:
            print(f"⚠️ Signal column missing: {col}")

    return df

def fill_missing_signals(df: pd.DataFrame, signal_columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in signal_columns:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
        else:
            print(f"⚠️ Cannot fill — missing column: {col}")
    return df



if __name__ == "__main__":
    import pandas as pd
    from utils import load_SPY_data

    print("📈 Loading SPY data...")
    df = load_SPY_data()

    print("📡 Adding external signals...")
    df_with_signals = add_external_signals(df)

    signal_cols = [
        "CPI", "Unemployment", "InterestRate", "YieldCurve",
        "ConsumerSentiment", "IndustrialProduction", "VIX",
        "News_Sentiment", "Reddit_Sentiment" 
    ]

    df_with_signals = fill_missing_signals(df_with_signals, signal_cols)
    df_with_signals = normalize_signals(df_with_signals, signal_cols)


    print("✅ Signals added. Here’s a preview:")
    print(df_with_signals.tail())

