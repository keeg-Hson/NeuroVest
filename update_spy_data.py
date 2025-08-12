from datetime import date, timedelta
import pandas as pd
import yfinance as yf

def _today() -> date:
    # Keep it simple,  could use NYSE trading-day logic later on
    return date.today()

# update_spy_data.py
import pandas as pd, yfinance as yf
from datetime import date

# update_spy_data.py
import pandas as pd, yfinance as yf
from datetime import date

def main():
    try:
        df = pd.read_csv("SPY.csv", parse_dates=["Date"])
        have_file = True
    except FileNotFoundError:
        have_file = False

    today = date.today().isoformat()

    if not have_file:
        # First run bootstrap
        base = "2010-01-01"
        spy = yf.download("SPY", start=base, end=today, interval="1d", auto_adjust=False)
        if spy is None or spy.empty:
            print("❌ Could not download initial SPY history.")
            return
        spy = spy.reset_index()
        spy["Date"] = pd.to_datetime(spy["Date"])
        spy = spy[["Date","Open","High","Low","Close","Adj Close","Volume"]]
        spy.to_csv("SPY.csv", index=False)
        print(f"✅ Created SPY.csv with {len(spy)} rows from {base} to {today}.")
        return

    


if __name__ == "__main__":
    main()