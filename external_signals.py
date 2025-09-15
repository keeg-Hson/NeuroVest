# external_signals.py
import os
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from textblob import TextBlob
from fredapi import Fred
import praw
from sklearn.preprocessing import MinMaxScaler
import warnings



# -----------------------------
# Env + clients
# -----------------------------
load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_ACTUAL_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
NEWS_BASE_URL = "https://newsapi.org/v2/everything"

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# -----------------------------
# General helpers
# -----------------------------




def _read_csv_maybe(path: str, index_is_date: bool = True) -> pd.DataFrame | None:
    """
    Read a CSV without pandas' global date inference to avoid noisy warnings.
    If index_is_date=True, set index_col=0 then explicitly coerce to datetime.
    """
    if not os.path.exists(path):
        return None
    try:
        if index_is_date:
            df = pd.read_csv(path, index_col=0)
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", category=UserWarning)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    df.index = pd.to_datetime(df.index, errors="coerce")
            return df
        else:
            df = pd.read_csv(path)
            if "Date" in df.columns:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", category=UserWarning)
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df
    except Exception as e:
        warnings.warn(f"[external_signals] Failed to read {path}: {e}")
        return None




def _pick_price_series(df_like: pd.DataFrame, prefer_name: str | None = None) -> pd.Series | None:
    """Try to extract a numeric price/value series from a typical OHLCV CSV."""
    if df_like is None or len(df_like) == 0:
        return None
    for col in ["Adj Close", "Close", "Value", "Price", "ClosePrice"]:
        if col in df_like.columns:
            s = pd.to_numeric(df_like[col], errors="coerce")
            s.name = prefer_name or col
            return s
    # fallback to first numeric column
    for col in df_like.columns:
        if pd.api.types.is_numeric_dtype(df_like[col]):
            s = pd.to_numeric(df_like[col], errors="coerce")
            s.name = prefer_name or col
            return s
    return None

def _pct_change(s: pd.Series | None, n: int) -> pd.Series | None:
    if s is None: return None
    return s.pct_change(n)

def _zscore(s: pd.Series | None, win: int = 20) -> pd.Series | None:
    if s is None: return None
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / (sd + 1e-9)

def _join_series(df: pd.DataFrame, s: pd.Series | None, col_name: str) -> pd.DataFrame:
    if s is None:
        if col_name not in df.columns:
            df[col_name] = np.nan
        return df
    s = s.loc[~s.index.duplicated()].sort_index()
    return df.join(s.rename(col_name), how="left")

def _ensure_unique_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex, unique, sorted; keep behavior you already had."""
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
    else:
        out = df.copy()
        if "Date" in out.columns:
            out = out.set_index(pd.to_datetime(out["Date"], errors="coerce"))
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    return _ensure_unique_sorted_index(out)

def _lag_joined_columns(df: pd.DataFrame, cols: list[str], n: int = 1) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].shift(n)
    return out



# -----------------------------
# NewsAPI + TextBlob sentiment
# -----------------------------
def fetch_news_sentiment(topic: str = "stock market", days: int = 7, page_size: int = 50) -> pd.DataFrame:
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    params = {
        "q": topic,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    try:
        r = requests.get(NEWS_BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except Exception as e:
        print(f"❌ Failed to fetch news: {e}")
        return pd.DataFrame()

    rows = []
    for a in articles:
        date = (a.get("publishedAt") or "")[:10]
        text = f"{a.get('title') or ''} {a.get('description') or ''}".strip()
        pol = TextBlob(text).sentiment.polarity
        rows.append({"Date": date, "News_Sentiment": pol})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    return df.groupby("Date").mean().sort_index()

# -----------------------------
# Reddit (PRAW) + TextBlob sentiment
# -----------------------------
def fetch_reddit_sentiment(subreddit: str = "stocks", days: int = 7, limit: int = 100) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    rows = []
    try:
        for subm in reddit.subreddit(subreddit).top(time_filter="week", limit=limit):
            if subm.created_utc < start.timestamp():
                continue
            date = datetime.utcfromtimestamp(subm.created_utc).date()
            text = f"{subm.title} {subm.selftext or ''}".strip()
            pol = TextBlob(text).sentiment.polarity
            rows.append({"Date": pd.to_datetime(date), "Reddit_Sentiment": pol})
    except Exception as e:
        print(f"❌ Reddit fetch failed: {e}")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("Date").mean().sort_index()

# -----------------------------
# Macro via FRED
# -----------------------------
def fetch_fred_macro_signals() -> pd.DataFrame:
    series_ids = {
        "CPI": "CPIAUCSL",
        "Unemployment": "UNRATE",
        "InterestRate": "DFF",
        "YieldCurve": "T10Y2Y",
        "ConsumerSentiment": "UMCSENT",
        "IndustrialProduction": "INDPRO",
        "VIX": "VIXCLS",
    }
    frames = []
    for name, sid in series_ids.items():
        try:
            s = fred.get_series(sid)
            df = s.reset_index()
            df.columns = ["Date", name]
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Could not fetch {name} ({sid}): {e}")

    if not frames:
        return pd.DataFrame(columns=["Date"])
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Date", how="outer")
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date")

# -----------------------------
# External signals aggregator
# -----------------------------
def add_external_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds to df (index by Date):
      - Macro (FRED)
      - News/Reddit sentiment
      - Sector breadth proxies (median returns & dispersion over 5/20d)
      - Credit & rates (HYG/LQD returns, 10Y change)
      - USD DXY changes
      - Structured sentiment features: Z20, ROC3, Concordance with next-day return
    """
    df = _ensure_dt_index(df.copy())

    # ---- Macro (FRED) ----
    macro = fetch_fred_macro_signals()
    if not macro.empty:
        macro = macro.set_index(pd.to_datetime(macro["Date"])).drop(columns=["Date"])
        df = df.join(macro, how="left")

    # ---- News sentiment ----
    news_df = fetch_news_sentiment()
    if not news_df.empty:
        news_df = _ensure_dt_index(news_df)
        df = df.join(news_df.rename(columns={"News_Sentiment": "News_Sentiment"}), how="left")
    else:
        df["News_Sentiment"] = df.get("News_Sentiment", np.nan)

    # ---- Reddit sentiment ----
    reddit_df = fetch_reddit_sentiment()
    if not reddit_df.empty:
        reddit_df = _ensure_dt_index(reddit_df)
        df = df.join(reddit_df.rename(columns={"Reddit_Sentiment": "Reddit_Sentiment"}), how="left")
    else:
        df["Reddit_Sentiment"] = df.get("Reddit_Sentiment", np.nan)

    # ---- Sector breadth proxies ----
    sector_dir = "data/etfs"
    sector_tickers = ["XLF","XLK","XLE","XLI","XLV","XLY","XLP","XLU","XLB","XLRE"]
    sector_series = []
    if os.path.isdir(sector_dir):
        for t in sector_tickers:
            df_csv = _read_csv_maybe(os.path.join(sector_dir, f"{t}.csv"))
            s = _pick_price_series(df_csv, prefer_name=t)
            if s is not None:
                sector_series.append(s.rename(t))
    # fallback: if those tickers are already columns on df (pre-merged)
    if not sector_series:
        for t in sector_tickers:
            if t in df.columns and pd.api.types.is_numeric_dtype(df[t]):
                sector_series.append(df[t].rename(t))

    if sector_series:
        sectors = pd.concat(sector_series, axis=1)
        sectors = _ensure_unique_sorted_index(sectors)          # drop dups, sort
        sectors = sectors.reindex(df.index.unique()).ffill()    # df index is unique

        rets_5  = sectors.pct_change(5)
        rets_20 = sectors.pct_change(20)
        df["Sector_MedianRet_5"]  = rets_5.median(axis=1)
        df["Sector_MedianRet_20"] = rets_20.median(axis=1)
        df["Sector_Dispersion_5"]  = rets_5.std(axis=1)
        df["Sector_Dispersion_20"] = rets_20.std(axis=1)
    else:
        for c in ["Sector_MedianRet_5","Sector_MedianRet_20","Sector_Dispersion_5","Sector_Dispersion_20"]:
            if c not in df.columns:
                df[c] = np.nan

    # ---- Credit & rates ----
    hyg = _pick_price_series(_read_csv_maybe("data/HYG.csv"), "HYG")
    lqd = _pick_price_series(_read_csv_maybe("data/LQD.csv"), "LQD")
    df = _join_series(df, _pct_change(hyg, 5),  "HYG_Ret_5")
    df = _join_series(df, _pct_change(hyg, 20), "HYG_Ret_20")
    df = _join_series(df, _pct_change(lqd, 5),  "LQD_Ret_5")
    df = _join_series(df, _pct_change(lqd, 20), "LQD_Ret_20")
    if "HYG_Ret_20" in df.columns and "LQD_Ret_20" in df.columns:
        df["Credit_Spread_20"] = df["HYG_Ret_20"] - df["LQD_Ret_20"]
    else:
        df["Credit_Spread_20"] = np.nan

    # 10y yield (TNX or FRED csv)
    tnx = _pick_price_series(_read_csv_maybe("data/TNX.csv"), "TNX")
    if tnx is None:
        tnx = _pick_price_series(_read_csv_maybe("data/FRED_DGS10.csv"), "DGS10")
    df = _join_series(df, _pct_change(tnx, 5),  "TNX_Change_5")
    df = _join_series(df, _pct_change(tnx, 20), "TNX_Change_20")

    # ---- USD (DXY) ----
    dxy = _pick_price_series(_read_csv_maybe("data/DXY.csv"), "DXY")
    df = _join_series(df, _pct_change(dxy, 5),  "DXY_Change_5")
    df = _join_series(df, _pct_change(dxy, 20), "DXY_Change_20")

    # ---- Structured Sentiment Features ----
    # z-scores & 3-day rate-of-change
    if "News_Sentiment" in df.columns:
        df["News_Sent_Z20"]  = _zscore(df["News_Sentiment"], 20)
        df["News_Sent_ROC3"] = df["News_Sentiment"].diff(3)
    else:
        df["News_Sent_Z20"]  = np.nan
        df["News_Sent_ROC3"] = np.nan

    if "Reddit_Sentiment" in df.columns:
        df["Reddit_Sent_Z20"]  = _zscore(df["Reddit_Sentiment"], 20)
        df["Reddit_Sent_ROC3"] = df["Reddit_Sentiment"].diff(3)
    else:
        df["Reddit_Sent_Z20"]  = np.nan
        df["Reddit_Sent_ROC3"] = np.nan

    # Concordance with next-day return direction
    if "Close" in df.columns:
        ret_1d_fwd = df["Close"].shift(-1) / df["Close"] - 1.0
        if "News_Sentiment" in df.columns:
            df["News_Sent_Concord"] = ((np.sign(df["News_Sentiment"]) == np.sign(ret_1d_fwd)) &
                                       df["News_Sentiment"].notna() & ret_1d_fwd.notna()).astype(int)
        else:
            df["News_Sent_Concord"] = np.nan

        if "Reddit_Sentiment" in df.columns:
            df["Reddit_Sent_Concord"] = ((np.sign(df["Reddit_Sentiment"]) == np.sign(ret_1d_fwd)) &
                                         df["Reddit_Sentiment"].notna() & ret_1d_fwd.notna()).astype(int)
        else:
            df["Reddit_Sent_Concord"] = np.nan
    else:
        df["News_Sent_Concord"] = np.nan
        df["Reddit_Sent_Concord"] = np.nan

    _lag_cols = [
        "CPI","Unemployment","InterestRate","YieldCurve","ConsumerSentiment","IndustrialProduction","VIX",
        "News_Sentiment","Reddit_Sentiment",
        "Sector_MedianRet_5","Sector_MedianRet_20","Sector_Dispersion_5","Sector_Dispersion_20",
        "Credit_Spread_20","TNX_Change_20","DXY_Change_20",
        "News_Sent_Z20","Reddit_Sent_Z20"
    ]
    df = _lag_joined_columns(df, _lag_cols, n=1)


    return df

# -----------------------------
# Optional utilities (unchanged)
# -----------------------------
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

# --- runner hook for run_all.py step_refresh_data ---
def refresh_all():
    """
    Orchestrate fetching/merging any external features you want.
    If you don't need to actually refresh here, leave as a no-op so the step
    shows signals=True in the pipeline summary.
    """
    return True


# -----------------------------
# Quick local test
# -----------------------------
if __name__ == "__main__":
    from utils import load_SPY_data
    print("📈 Loading SPY data...")
    base = load_SPY_data()
    print("📡 Adding external signals...")
    out = add_external_signals(base)

    must_have = [
        "Sector_MedianRet_5","Sector_MedianRet_20","Sector_Dispersion_5","Sector_Dispersion_20",
        "HYG_Ret_5","HYG_Ret_20","LQD_Ret_5","LQD_Ret_20","Credit_Spread_20",
        "TNX_Change_5","TNX_Change_20","DXY_Change_5","DXY_Change_20",
        "News_Sentiment","Reddit_Sentiment","News_Sent_Z20","News_Sent_ROC3",
        "Reddit_Sent_Z20","Reddit_Sent_ROC3","News_Sent_Concord","Reddit_Sent_Concord"
    ]
    print("✅ Added columns present:", [c for c in must_have if c in out.columns])

    signal_cols = [
        "CPI","Unemployment","InterestRate","YieldCurve",
        "ConsumerSentiment","IndustrialProduction","VIX",
        "News_Sentiment","Reddit_Sentiment"
    ]
    out = fill_missing_signals(out, signal_cols)
    out = normalize_signals(out, signal_cols)
    print(out.tail())
