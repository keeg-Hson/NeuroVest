# data_utils.py
from config import SPY_DAILY_CSV
import pandas as pd
import yfinance as yf
from datetime import datetime
import os


from utils import safe_read_csv
from config import SPY_DAILY_CSV
import pandas as pd
import os

def load_spy_daily_data(path: str | None = None) -> pd.DataFrame:
    """
    Loads SPY daily data, tolerates different CSV schemas (AlphaVantage/yfinance),
    returns a DataFrame with a DatetimeIndex and a 'Date' column present.
    """
    path = path or SPY_DAILY_CSV if "SPY_DAILY_CSV" in globals() else "data/SPY.csv"

    # Robust read (creates Date column; sets as index)
    df = safe_read_csv(path, prefer_index=True)

    # If yfinance layout, columns are standard already (Open/High/Low/Close/Adj Close/Volume)
    # If AlphaVantage layout, rename:
    rename_map = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume",
        # some variants:
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Keep canonical OHLCV if present
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if keep:
        df = df[keep]

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop trailing incomplete last row(s)
    crit = [c for c in ["High","Low","Volume"] if c in df.columns]
    while len(df) and crit and df.iloc[-1][crit].isna().any():
        df = df.iloc[:-1]

    # Make sure Date column is present (safe_read_csv already did, but keep it explicit)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "Date"
    df["Date"] = df.index

    print("🔍 SPY data columns loaded:", df.columns.tolist())
    return df




def log_rolling_accuracy(current_date, acc_7d, acc_30d, filepath="logs/model_performance.csv"):
    df = pd.DataFrame([{
        "Date": current_date.strftime("%Y-%m-%d"),
        "Accuracy_7d": acc_7d,
        "Accuracy_30d": acc_30d
    }])
    df.to_csv(filepath, mode='a', index=False, header=not os.path.exists(filepath))