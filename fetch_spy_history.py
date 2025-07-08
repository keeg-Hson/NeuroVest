import pandas as pd
from config import DATA_DIR
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key="YOUR_KEY", output_format='pandas')

try:
    # free daily (no premium)
    df, _ = ts.get_daily(symbol="SPY", outputsize="full") #this is a premium feature
    df.index.name = "Date"
except ValueError as e:
    print("⚠️ AlphaVantage daily_adjusted failed, falling back to yfinance:", e)
    import yfinance as yf
    df = yf.download("SPY", start="2000-01-01")
    df.index.name = "Date"

DATA_DIR.mkdir(exist_ok=True)
df.to_csv(DATA_DIR / "spy_daily.csv")
print(f"[✅] Saved SPY history to {DATA_DIR / 'spy_daily.csv'}")
