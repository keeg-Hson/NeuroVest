# update_spy_data.py
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

spy_path = "data/spy.csv"
os.makedirs("data", exist_ok=True)

if os.path.exists(spy_path):
    spy_existing = pd.read_csv(spy_path, parse_dates=["Date"])
    spy_existing.sort_values("Date", inplace=True)
    last_date = spy_existing["Date"].max() + pd.Timedelta(days=1)
    print(f"🔄 Appending new data from {last_date.date()}")
else:
    spy_existing = pd.DataFrame()
    last_date = datetime(2000, 1, 1)
    print("🆕 No existing SPY file — downloading full history from 2000.")

end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

spy_new = yf.download("SPY", start=last_date.strftime("%Y-%m-%d"), end=end_date, interval="1d")

if not spy_new.empty:
    #if MultiIndex columns (e.g. ('Close', 'SPY')), flatten them
    if isinstance(spy_new.columns, pd.MultiIndex):
        spy_new.columns = [col[0] for col in spy_new.columns]  # Drop second level

    spy_new = spy_new.reset_index()
    print(f"🔍 New SPY columns: {list(spy_new.columns)}")

    #standardize column names
    expected_cols = ["Date", "Open", "Close"]
    spy_new = spy_new[[col for col in expected_cols if col in spy_new.columns]]

    if "Date" not in spy_new.columns:
        raise ValueError("❌ 'Date' column missing in fetched data. Columns present: " + str(spy_new.columns))

    spy_combined = pd.concat([spy_existing, spy_new], ignore_index=True)
    spy_combined.drop_duplicates(subset="Date", keep="last", inplace=True)
    spy_combined.sort_values("Date", inplace=True)

    spy_combined.to_csv(spy_path, index=False)
    print(f"✅ SPY data updated through {spy_combined['Date'].max().strftime('%Y-%m-%d')}")
else:
    print("⚠️ No new SPY data downloaded.")
