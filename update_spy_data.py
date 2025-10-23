# update_spy_data.py
from datetime import date, timedelta
from pandas.tseries.offsets import BDay
import pandas as pd
import yfinance as yf
import os

# Use the single source of truth path from utils.py
from utils import CSV_PATH  # points to "data/SPY.csv"

def _ensure_dirs():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def _exclusive_end_today() -> str:
    # yfinance end is exclusive; add +1 day to include today's bar when available
    return (date.today() + timedelta(days=1)).isoformat()

def _bootstrap():
    base = "2010-01-01"
    end  = _exclusive_end_today()
    df = yf.download("SPY", start=base, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        print("❌ Could not download initial SPY history.")
        return False
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Created {CSV_PATH} with {len(df)} rows ({base} → {end}).")
    return True

def _append_new_rows():
    # Parse dates so .date() is valid later
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    if df.empty:
        return _bootstrap()

    # Robust last_date extraction
    last_date = pd.to_datetime(df["Date"]).max().date()

    # Start next business day after last_date
    start = (pd.Timestamp(last_date) + BDay(1)).date()
    end = date.today() + timedelta(days=1)  # exclusive end

    if start >= end:
        print("ℹ️ SPY.csv is already up to date — no new days to fetch.")
        return True

    newdf = yf.download(
        "SPY",
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    if newdf is None or newdf.empty:
        print("ℹ️ No new SPY data returned by yfinance.")
        return True

    newdf = newdf.reset_index()
    newdf["Date"] = pd.to_datetime(newdf["Date"])
    newdf = newdf[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    merged = pd.concat([df, newdf], ignore_index=True)
    merged.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    merged.sort_values("Date", inplace=True)
    merged.to_csv(CSV_PATH, index=False)

    print(f"✅ Appended {len(newdf)} new rows to {CSV_PATH} ({start} → {end}).")
    return True

def main():
    _ensure_dirs()
    if not os.path.exists(CSV_PATH):
        _bootstrap()
    else:
        _append_new_rows()

if __name__ == "__main__":
    main()
