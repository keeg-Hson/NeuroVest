# external_signals.py
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(".env", override=True)


def _flag(name, default="0"):
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


import contextlib
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Try to use TextBlob, but don't require it
try:
    from textblob import TextBlob

    _HAS_TEXTBLOB = True
except Exception:
    _HAS_TEXTBLOB = False


# -----------------------------
# Env
# -----------------------------

OFFLINE = os.getenv("OFFLINE_MODE", "0").lower() in {"1", "true", "yes", "on"}

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_BASE_URL = "https://newsapi.org/v2/everything"

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT") or "market-bot/0.1 by <you>"

FRED_API_KEY = os.getenv("FRED_API_KEY")  # optional for pandas_datareader; helps with rate limits

# Cache dir for FRED
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sentiment caches
CACHE_NEWS_SENT = CACHE_DIR / "news_sent.csv"
CACHE_REDDIT_SENT = CACHE_DIR / "reddit_sent.csv"


def _load_daily_sent(cache_path: Path, col_name: str) -> pd.DataFrame:
    """Load cached daily sentiment as Date-indexed DF with one column."""
    if not cache_path.exists():
        return pd.DataFrame(columns=[col_name])
    try:
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        df = df[df["Date"].notna()]
        df = df.set_index("Date").sort_index()
        # keep only target column if present; otherwise rename the first numeric
        if col_name in df.columns:
            return df[[col_name]]
        # fallback
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            return df[[num_cols[0]]].rename(columns={num_cols[0]: col_name})
        return pd.DataFrame(columns=[col_name])
    except Exception:
        return pd.DataFrame(columns=[col_name])


def _save_daily_sent(cache_path: Path, df: pd.DataFrame):
    """Save Date-indexed DF with one column back to CSV."""
    if df.empty:
        return
    out = df.copy()
    out = out.reset_index()
    out = out.rename(columns={"index": "Date"})
    out.to_csv(cache_path, index=False)


def _merge_sent_cache(cache_path: Path, new_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Concatenate cached + new, group by date, mean, and persist."""
    old = _load_daily_sent(cache_path, col_name)
    frames = [old] + ([new_df] if new_df is not None and not new_df.empty else [])
    merged = pd.concat(frames, axis=0)
    if merged.empty:
        return merged
    # ensure proper shape
    merged = merged.copy()
    merged.index = pd.to_datetime(merged.index, errors="coerce")
    merged = merged[merged.index.notna()]
    # average by day (if multiple rows per day)
    merged = merged.groupby(merged.index.normalize())[col_name].mean().to_frame(col_name)
    merged = merged.sort_index()
    _save_daily_sent(cache_path, merged)
    return merged


# -----------------------------
# General helpers
# -----------------------------


def _find_csv_anywhere(filename: str, roots: list[str] = None) -> str | None:
    if roots is None:
        roots = [".", "data", "data/etfs"]
    target = filename.lower()
    ticker = target.replace(".csv", "")
    for root in roots:
        try:
            for dirpath, _, files in os.walk(root):
                for f in files:
                    fl = f.lower()
                    if fl == target:  # exact
                        return os.path.join(dirpath, f)
                    if fl.endswith(".csv"):
                        base = fl[:-4]
                        if base == ticker or base.startswith(ticker) or ticker in base:  # fuzzy
                            return os.path.join(dirpath, f)
        except Exception:
            continue
    return None


def _ensure_unique_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex, unique, sorted; keep your behavior."""
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
    else:
        out = df.copy()
        if "Date" in out.columns:
            out.index = pd.to_datetime(out["Date"], errors="coerce")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    return _ensure_unique_sorted_index(out)


def _read_csv_maybe(path: str, index_is_date: bool = True) -> pd.DataFrame | None:
    """
    Robust CSV reader for Yahoo-style files (and similar two-row headers where
    one level is the ticker and the other is the field: Close/Open/High/Low/Volume/Date).
    Returns a DataFrame indexed by Date (if present) and sorted ascending.
    """
    if not os.path.exists(path):
        return None

    FIELD_NAMES = {
        "adj close",
        "adjusted close",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "date",
        "closeprice",
        "last",
        "last price",
    }

    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        # --- PROMOTE DATE-LIKE COLUMN IF NEEDED ---
        if "Date" not in df.columns:
            # check a few left-most columns for parseable dates
            for c in list(df.columns)[:3]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        # pandas >=2.0 supports format="mixed"; if older, this still works and warnings are silenced
                        sample = pd.to_datetime(df[c], errors="coerce")
                    if sample.notna().mean() >= 0.90:
                        df = df.rename(columns={c: "Date"})
                        break
                except Exception:
                    pass

        # If a Date column exists, prefer it as index
        if "Date" in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        if index_is_date:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        return df.sort_index()

    try:
        # First try normal single-header read
        df0 = pd.read_csv(path, low_memory=False)

        # Heuristic: if first row looks like header (contains 'Date' or field names),
        # promote it to header.
        if df0.shape[0] > 0:
            first_row_vals = [str(v) for v in df0.iloc[0].tolist()]
            hits = sum(v.strip().lower() in FIELD_NAMES for v in first_row_vals)
            if hits >= 2 or any(v.strip().lower() == "date" for v in first_row_vals):
                df0.columns = first_row_vals
                df0 = df0.drop(df0.index[0]).reset_index(drop=True)

        # If we can already finalize this, do it.
        simple = _finalize(df0.copy())
        if simple.shape[1] > 0:
            return simple

        # Try two-row header read
        df1 = pd.read_csv(path, header=[0, 1], low_memory=False)

        # Some exports include a bogus first data row like ['Date', ...] etc.
        try:
            first_cell = str(df1.iloc[0, 0])
            if first_cell.lower() in {"date", "0"} or "date" in first_cell.lower():
                df1 = df1.iloc[1:].reset_index(drop=True)
        except Exception:
            pass

        # If we genuinely have a MultiIndex header, decide which level is "field" vs "ticker".
        if isinstance(df1.columns, pd.MultiIndex):
            level0 = [str(c[0]).strip().lower() for c in df1.columns]
            level1 = [str(c[1]).strip().lower() for c in df1.columns]
            hits0 = sum(x in FIELD_NAMES for x in level0)
            hits1 = sum(x in FIELD_NAMES for x in level1)

            # Choose the level with more field-name hits as the column names
            if hits0 >= hits1:
                df1.columns = [c[0] for c in df1.columns]
            else:
                df1.columns = [c[1] for c in df1.columns]
        else:
            # Not a MultiIndex; fall back to df0 behavior
            return _finalize(df0)

        # If a 'Ticker' column survived, drop it
        if "Ticker" in df1.columns:
            df1 = df1.drop(columns=["Ticker"], errors="ignore")

        # If we still have duplicates (e.g., repeated field names per ticker), keep first occurrence
        keep_order = []
        seen = set()
        for c in df1.columns:
            if c not in seen:
                keep_order.append(c)
                seen.add(c)
        df1 = df1.loc[:, keep_order]

        return _finalize(df1)

    except Exception as e:
        warnings.warn(f"[external_signals] Failed to read {path}: {e}", stacklevel=2)
        return None


def _pick_price_series(
    df_like: pd.DataFrame | None, prefer_name: str | None = None
) -> pd.Series | None:
    if df_like is None or len(df_like) == 0:
        return None
    candidates = [
        "Adj Close",
        "AdjClose",
        "Adjusted Close",
        "Close",
        "close",
        "CLOSE",
        "Value",
        "Price",
        "ClosePrice",
        "Last",
        "Last Price",
        "Close*",
        "Adj Close*",
    ]
    for col in candidates:
        if col in df_like.columns:
            s = pd.to_numeric(df_like[col], errors="coerce")
            s.name = prefer_name or col
            return s
    for col in df_like.columns:
        if pd.api.types.is_numeric_dtype(df_like[col]):
            s = pd.to_numeric(df_like[col], errors="coerce")
            s.name = prefer_name or col
            return s
    return None


def _clean_series_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _pct_change(s: pd.Series | None, n: int) -> pd.Series | None:
    if s is None:
        return None
    # Avoid FutureWarning: explicitly disable pad-fill during pct_change
    return s.pct_change(n, fill_method=None)


def _zscore(s: pd.Series | None, win: int = 20) -> pd.Series | None:
    if s is None:
        return None
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / (sd + 1e-9)


def _join_series(df: pd.DataFrame, s: pd.Series | None, col_name: str) -> pd.DataFrame:
    out = df.copy()
    if s is None:
        if col_name not in out.columns:
            out[col_name] = np.nan
        return out
    s = s.loc[~s.index.duplicated()].sort_index()
    return out.join(s.rename(col_name), how="left")


def _lag_joined_columns(df: pd.DataFrame, cols: list[str], n: int = 1) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].shift(n)
    return out


def _polarity(text: str) -> float:
    if not text:
        return 0.0
    if _HAS_TEXTBLOB:
        try:
            return float(TextBlob(text).sentiment.polarity)
        except Exception:
            return 0.0
    # Fallback: naive neutral
    return 0.0


# -----------------------------
# FRED via pandas_datareader (+ cache)
# -----------------------------
def _fetch_from_fred(series_id: str, start: datetime | None = None) -> pd.DataFrame:
    """
    Pull a single FRED series using pandas_datareader. Returns a DataFrame with:
      index = DatetimeIndex
      column = 'value'
    """
    if OFFLINE:
        return pd.DataFrame()
    from pandas_datareader import data as pdr

    kwargs = {"start": start} if start else {}
    # If you want to pass an API key, pandas_datareader supports it via environment too.
    df = pdr.DataReader(series_id, "fred", **kwargs)  # raises on network error
    if isinstance(df, pd.Series):
        df = df.to_frame("value")
    else:
        # Sometimes column name is series_id; normalize to 'value'
        c0 = df.columns[0]
        df = df.rename(columns={c0: "value"})
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df


def fetch_fred_cached(series_id: str, start: datetime | None = None) -> pd.DataFrame:
    """
    Cached FRED fetch. On OFFLINE or failure returns empty frame.
    """
    if OFFLINE:
        return pd.DataFrame()
    cache = CACHE_DIR / f"fred_{series_id}.csv"
    if cache.exists():
        try:
            out = pd.read_csv(cache, parse_dates=["DATE"])
            out = out.rename(columns={"DATE": "Date"})
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out = out.set_index("Date")
            return out
        except Exception:
            pass
    df = _fetch_from_fred(series_id, start=start)
    df = df.rename_axis("Date")
    with contextlib.suppress(Exception):
        df.to_csv(cache, index_label="DATE")
    return df


def fetch_fred_macro_signals() -> pd.DataFrame:
    """
    Returns wide DataFrame with macro columns; empty if offline or all fail.
    """
    if OFFLINE:
        return pd.DataFrame(columns=["Date"])

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
            df = fetch_fred_cached(sid)  # -> index=Date, col='value'
            if df.empty:
                raise RuntimeError("empty")
            df = df.rename(columns={"value": name})
            df = df.reset_index()
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Could not fetch {name} ({sid}): {e}")
    if not frames:
        return pd.DataFrame(columns=["Date"])
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Date", how="outer")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out.sort_values("Date")


# -----------------------------
# NewsAPI + sentiment
# -----------------------------
def fetch_news_sentiment(
    topic: str = "stock market", days: int = 7, page_size: int = 50
) -> pd.DataFrame:
    if OFFLINE or not NEWS_API_KEY:
        # return whatever we've cached so far, if anything
        return _load_daily_sent(CACHE_NEWS_SENT, "News_Sentiment")

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

    articles = []
    try:
        r = requests.get(NEWS_BASE_URL, params=params, timeout=15)
        if r.status_code == 426:
            print("❌ NewsAPI 426: plan limit — falling back to top-headlines.")
            r2 = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "category": "business",
                    "language": "en",
                    "pageSize": page_size,
                    "apiKey": NEWS_API_KEY,
                },
                timeout=15,
            )
            r2.raise_for_status()
            articles = r2.json().get("articles", [])
        elif r.status_code == 401:
            print("❌ NewsAPI 401: invalid key — skipping.")
            # return cache only
            return _load_daily_sent(CACHE_NEWS_SENT, "News_Sentiment")
        else:
            r.raise_for_status()
            articles = r.json().get("articles", [])
    except Exception as e:
        print(f"❌ Failed to fetch news: {e}")
        # return cache only
        return _load_daily_sent(CACHE_NEWS_SENT, "News_Sentiment")

    rows = []
    for a in articles or []:
        date = (a.get("publishedAt") or "")[:10]
        text = f"{a.get('title') or ''} {a.get('description') or ''}".strip()
        pol = _polarity(text)
        if date:
            rows.append({"Date": date, "News_Sentiment": pol})

    if rows:
        df_new = pd.DataFrame(rows)
        df_new["Date"] = pd.to_datetime(df_new["Date"])
        df_new = df_new.set_index("Date").sort_index()
    else:
        df_new = pd.DataFrame(columns=["News_Sentiment"])

    # Merge with cache and persist
    merged = _merge_sent_cache(CACHE_NEWS_SENT, df_new, "News_Sentiment")
    return merged


# -----------------------------
# Reddit (PRAW) + sentiment
# -----------------------------
def fetch_reddit_sentiment(
    subreddit: str = "stocks", days: int = 7, limit: int = 100
) -> pd.DataFrame:
    # If offline or creds missing, return cache only
    rid = os.getenv("REDDIT_CLIENT_ID")
    sec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT")
    if OFFLINE or not (rid and sec and ua):
        if OFFLINE:
            print("⚠️ Reddit offline — using cached sentiment only.")
        else:
            print("⚠️ Reddit not configured — using cached sentiment only.")
        return _load_daily_sent(CACHE_REDDIT_SENT, "Reddit_Sentiment")

    try:
        import praw

        usr = os.getenv("REDDIT_USERNAME")
        pwd = os.getenv("REDDIT_PASSWORD")
        kwargs = dict(client_id=rid, client_secret=sec, user_agent=ua, check_for_async=False)
        if usr and pwd:
            kwargs["username"] = usr
            kwargs["password"] = pwd
        reddit = praw.Reddit(**kwargs)
        reddit.read_only = True
        with contextlib.suppress(Exception):
            _ = reddit.auth.scopes()
    except Exception as e:
        print(f"❌ Reddit client init failed: {e}")
        return _load_daily_sent(CACHE_REDDIT_SENT, "Reddit_Sentiment")

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    rows = []
    try:
        # top over week is fine for a daily signal; we accumulate to cache
        for subm in reddit.subreddit(subreddit).top(time_filter="week", limit=limit):
            if subm.created_utc < start.timestamp():
                continue
            date = datetime.utcfromtimestamp(subm.created_utc).date()
            text = f"{subm.title} {getattr(subm, 'selftext', '') or ''}".strip()
            pol = _polarity(text)
            rows.append({"Date": pd.to_datetime(date), "Reddit_Sentiment": pol})
    except Exception as e:
        print(f"❌ Reddit fetch failed: {e}")
        return _load_daily_sent(CACHE_REDDIT_SENT, "Reddit_Sentiment")

    if rows:
        df_new = pd.DataFrame(rows).set_index("Date").sort_index()
    else:
        df_new = pd.DataFrame(columns=["Reddit_Sentiment"])

    # Merge with cache and persist
    merged = _merge_sent_cache(CACHE_REDDIT_SENT, df_new, "Reddit_Sentiment")
    return merged


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
    out = _ensure_dt_index(df.copy())

    # ---- Macro (FRED) ----
    macro = fetch_fred_macro_signals()
    if not macro.empty:
        macro = macro.set_index(pd.to_datetime(macro["Date"])).drop(columns=["Date"])
        out = out.join(macro, how="left")

    # ---- News sentiment ----
    news_df = fetch_news_sentiment(days=365)
    if not news_df.empty:
        news_df = _ensure_dt_index(news_df)
        out = out.join(news_df, how="left")
    else:
        if "News_Sentiment" not in out.columns:
            out["News_Sentiment"] = 0.0

    # ---- Reddit sentiment ----
    reddit_df = fetch_reddit_sentiment(days=365, limit=300)
    if not reddit_df.empty:
        reddit_df = _ensure_dt_index(reddit_df)
        out = out.join(reddit_df, how="left")
    else:
        if "Reddit_Sentiment" not in out.columns:
            out["Reddit_Sentiment"] = 0.0

    # ---- Sector breadth proxies ----
    candidate_dirs = ["data/etfs", "data", "."]
    sector_tickers = ["XLF", "XLK", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLB", "XLRE"]
    sector_series = []
    missing = []

    for t in sector_tickers:
        csv_path = None
        # try direct paths first
        for d in candidate_dirs:
            p = os.path.join(d, f"{t}.csv")
            if os.path.exists(p):
                csv_path = p
                break
        # otherwise fuzzy-search anywhere
        if csv_path is None:
            csv_path = _find_csv_anywhere(f"{t}.csv")

        if not csv_path:
            missing.append(t)
            continue

        df_csv = _read_csv_maybe(csv_path)
        if df_csv is None or df_csv.empty:
            print(f"[extsig] {t}: file found but empty/unreadable -> {csv_path}")
            continue

        s = _pick_price_series(df_csv, prefer_name=t)
        if s is None:
            print(
                f"[extsig] {t}: no numeric close-like column -> {csv_path}; cols={list(df_csv.columns)[:8]}"
            )
            continue

        s = _clean_series_index(s).rename(t)
        sector_series.append(s)

    # one concise summary line
    print(
        "[extsig] sector tickers loaded:",
        len(sector_series),
        [getattr(s, "name", "unknown") for s in sector_series][:5],
    )
    if missing:
        print("[extsig] sector tickers missing (no CSV found):", missing)

    # fallback: if tickers already in 'out'
    if not sector_series:
        for t in sector_tickers:
            if t in out.columns and pd.api.types.is_numeric_dtype(out[t]):
                sector_series.append(out[t].rename(t))

    if sector_series:
        sectors = pd.concat(sector_series, axis=1)
        sectors = _ensure_unique_sorted_index(sectors)
        sectors = sectors.reindex(out.index.unique()).ffill()

        # ensure numeric before pct_change
        sectors = sectors.astype("float64")

        # avoid deprecated default pad-fill; compute clean percentage changes
        rets_5 = sectors.pct_change(5, fill_method=None)
        rets_20 = sectors.pct_change(20, fill_method=None)

        out["Sector_MedianRet_5"] = rets_5.median(axis=1)
        out["Sector_MedianRet_20"] = rets_20.median(axis=1)
        out["Sector_Dispersion_5"] = rets_5.std(axis=1)
        out["Sector_Dispersion_20"] = rets_20.std(axis=1)

        print(
            "coverage:",
            float(out["Sector_MedianRet_20"].notna().mean()),
            float(out["Sector_Dispersion_20"].notna().mean()),
        )

    else:
        for c in [
            "Sector_MedianRet_5",
            "Sector_MedianRet_20",
            "Sector_Dispersion_5",
            "Sector_Dispersion_20",
        ]:
            if c not in out.columns:
                out[c] = np.nan

    # ---- Credit & rates ----
    hyg = _pick_price_series(_read_csv_maybe("data/HYG.csv"), "HYG")
    hyg = _clean_series_index(hyg) if hyg is not None else None

    lqd = _pick_price_series(_read_csv_maybe("data/LQD.csv"), "LQD")
    lqd = _clean_series_index(lqd) if lqd is not None else None

    out = _join_series(out, _pct_change(hyg, 5), "HYG_Ret_5")
    out = _join_series(out, _pct_change(hyg, 20), "HYG_Ret_20")
    out = _join_series(out, _pct_change(lqd, 5), "LQD_Ret_5")
    out = _join_series(out, _pct_change(lqd, 20), "LQD_Ret_20")

    if "HYG_Ret_20" in out.columns and "LQD_Ret_20" in out.columns:
        out["Credit_Spread_20"] = out["HYG_Ret_20"] - out["LQD_Ret_20"]
    else:
        out["Credit_Spread_20"] = np.nan

    # 10y yield (TNX or FRED csv)
    tnx = _pick_price_series(_read_csv_maybe("data/TNX.csv"), "TNX")
    tnx = _clean_series_index(tnx) if tnx is not None else None
    if tnx is None:
        tnx = _pick_price_series(_read_csv_maybe("data/FRED_DGS10.csv"), "DGS10")
        tnx = _clean_series_index(tnx) if tnx is not None else None

    out = _join_series(out, _pct_change(tnx, 5), "TNX_Change_5")
    out = _join_series(out, _pct_change(tnx, 20), "TNX_Change_20")

    # ---- USD (DXY) ----
    dxy = _pick_price_series(_read_csv_maybe("data/DXY.csv"), "DXY")
    dxy = _clean_series_index(dxy) if dxy is not None else None

    # Always also load UUP so we can backfill gaps
    uup = _pick_price_series(_read_csv_maybe("data/UUP.csv"), "UUP")
    uup = _clean_series_index(uup) if uup is not None else None

    dxy5 = _pct_change(dxy, 5) if dxy is not None else None
    dxy20 = _pct_change(dxy, 20) if dxy is not None else None
    uup5 = _pct_change(uup, 5) if uup is not None else None
    uup20 = _pct_change(uup, 20) if uup is not None else None

    # Join DXY first, then fill any gaps with UUP
    out = _join_series(out, dxy5, "DXY_Change_5")
    out = _join_series(out, dxy20, "DXY_Change_20")
    if uup5 is not None:
        out["DXY_Change_5"] = out["DXY_Change_5"].fillna(uup5)
    if uup20 is not None:
        out["DXY_Change_20"] = out["DXY_Change_20"].fillna(uup20)

    # ---- Structured Sentiment Features ----

    if "News_Sentiment" in out.columns:
        out["News_Sentiment"] = out["News_Sentiment"].fillna(0.0)
    if "Reddit_Sentiment" in out.columns:
        out["Reddit_Sentiment"] = out["Reddit_Sentiment"].fillna(0.0)

    if "News_Sentiment" in out.columns:
        out["News_Sent_Z20"] = _zscore(out["News_Sentiment"], 20)
        out["News_Sent_ROC3"] = out["News_Sentiment"].diff(3)
    else:
        out["News_Sent_Z20"] = np.nan
        out["News_Sent_ROC3"] = np.nan

    if "Reddit_Sentiment" in out.columns:
        out["Reddit_Sent_Z20"] = _zscore(out["Reddit_Sentiment"], 20)
        out["Reddit_Sent_ROC3"] = out["Reddit_Sentiment"].diff(3)
    else:
        out["Reddit_Sent_Z20"] = np.nan
        out["Reddit_Sent_ROC3"] = np.nan

    # Concordance with next-day return direction
    if "Close" in out.columns:
        ret_1d_fwd = out["Close"].shift(-1) / out["Close"] - 1.0
        if "News_Sentiment" in out.columns:
            out["News_Sent_Concord"] = (
                (np.sign(out["News_Sentiment"]) == np.sign(ret_1d_fwd))
                & out["News_Sentiment"].notna()
                & ret_1d_fwd.notna()
            ).astype(int)
        else:
            out["News_Sent_Concord"] = np.nan

        if "Reddit_Sentiment" in out.columns:
            out["Reddit_Sent_Concord"] = (
                (np.sign(out["Reddit_Sentiment"]) == np.sign(ret_1d_fwd))
                & out["Reddit_Sentiment"].notna()
                & ret_1d_fwd.notna()
            ).astype(int)
        else:
            out["Reddit_Sent_Concord"] = np.nan
    else:
        out["News_Sent_Concord"] = np.nan
        out["Reddit_Sent_Concord"] = np.nan

    # Lag externals one day so they can't “see the future”
    _lag_cols = [
        "CPI",
        "Unemployment",
        "InterestRate",
        "YieldCurve",
        "ConsumerSentiment",
        "IndustrialProduction",
        "VIX",
        "News_Sentiment",
        "Reddit_Sentiment",
        "Sector_MedianRet_5",
        "Sector_MedianRet_20",
        "Sector_Dispersion_5",
        "Sector_Dispersion_20",
        "Credit_Spread_20",
        "TNX_Change_20",
        "DXY_Change_20",
        "News_Sent_Z20",
        "Reddit_Sent_Z20",
    ]
    out = _lag_joined_columns(out, _lag_cols, n=1)

    print(
        "sample sector rows:",
        out[["Sector_MedianRet_20", "Sector_Dispersion_20"]].dropna().tail(3).to_dict("records"),
    )

    return out


# -----------------------------
# Optional utilities
# -----------------------------
from sklearn.preprocessing import MinMaxScaler


def normalize_signals(df: pd.DataFrame, signal_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()
    for col in signal_columns:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
        else:
            print(f"⚠️ Signal column missing: {col}")
    return df


def fill_missing_signals(df: pd.DataFrame, signal_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in signal_columns:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
        else:
            print(f"⚠️ Cannot fill — missing column: {col}")
    return df


# -----------------------------
# Runner hook for run_all.py
# -----------------------------
def refresh_all():
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
        "Sector_MedianRet_5",
        "Sector_MedianRet_20",
        "Sector_Dispersion_5",
        "Sector_Dispersion_20",
        "HYG_Ret_5",
        "HYG_Ret_20",
        "LQD_Ret_5",
        "LQD_Ret_20",
        "Credit_Spread_20",
        "TNX_Change_5",
        "TNX_Change_20",
        "DXY_Change_5",
        "DXY_Change_20",
        "News_Sentiment",
        "Reddit_Sentiment",
        "News_Sent_Z20",
        "News_Sent_ROC3",
        "Reddit_Sent_Z20",
        "Reddit_Sent_ROC3",
        "News_Sent_Concord",
        "Reddit_Sent_Concord",
    ]
    print("✅ Added columns present:", [c for c in must_have if c in out.columns])

    signal_cols = [
        "CPI",
        "Unemployment",
        "InterestRate",
        "YieldCurve",
        "ConsumerSentiment",
        "IndustrialProduction",
        "VIX",
        "News_Sentiment",
        "Reddit_Sentiment",
    ]
    out = fill_missing_signals(out, signal_cols)
    out = normalize_signals(out, signal_cols)
    print(out.tail())
