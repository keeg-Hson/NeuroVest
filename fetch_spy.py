# fetch_spy.py
import yfinance as yf

df = yf.download("SPY", start="2000-01-01")
df.index.name = "Date"  # gives index a column name
df.to_csv("data/spy_daily.csv")
