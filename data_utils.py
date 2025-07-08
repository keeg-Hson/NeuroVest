# data_utils.py
from config import SPY_DAILY_CSV
import pandas as pd
import yfinance as yf

def load_spy_daily_data() -> pd.DataFrame:
    # load & parse your CSV
    df = pd.read_csv(
        SPY_DAILY_CSV,
        parse_dates=["Date"],    # or whatever your date column is
        index_col="Date"
    )

    # rename AlphaVantage-style columns to something cleaner:
    df = df.rename(columns={
        "1. open":   "Open",
        "2. high":   "High",
        "3. low":    "Low",
        "4. close":  "Close",
        "5. volume": "Volume"
    })
    return df