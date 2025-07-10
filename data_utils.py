# data_utils.py
from config import SPY_DAILY_CSV
import pandas as pd
import yfinance as yf
from datetime import datetime


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



def log_rolling_accuracy(current_date, acc_7d, acc_30d, filepath="logs/model_performance.csv"):
    df = pd.DataFrame([{
        "Date": current_date.strftime("%Y-%m-%d"),
        "Accuracy_7d": acc_7d,
        "Accuracy_30d": acc_30d
    }])
    df.to_csv(filepath, mode='a', index=False, header=not pd.io.common.file_exists(filepath))