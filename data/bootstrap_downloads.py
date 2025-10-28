# data/bootstrap_downloads.py
import os
from datetime import datetime
import pandas as pd

import yfinance as yf
d = yf.download("SPY", auto_adjust=False, progress=False)
d.to_csv("data/SPY.csv")
print("✅ Wrote data/SPY.csv with", len(d), "rows")


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save_csv(df, path):
    if df is None or df.empty:
        print(f"⚠️ Empty dataframe for {path} — skipping.")
        return
    df.to_csv(path)
    print(f"✅ Saved {path}")

def main():
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Please: pip install yfinance")

    _ensure_dir("data")
    _ensure_dir("data/etfs")

    # ---- Sector ETFs (S&P sectors)
    sectors = ["XLF","XLK","XLE","XLI","XLV","XLY","XLP","XLU","XLB","XLRE"]
    for t in sectors:
        df = yf.download(t, auto_adjust=False, progress=False)
        _save_csv(df, f"data/etfs/{t}.csv")

    # ---- Credit (HYG/LQD)
    for t in ["HYG","LQD"]:
        df = yf.download(t, auto_adjust=False, progress=False)
        _save_csv(df, f"data/{t}.csv")

    # ---- USD index (DXY) → use ^DXY on Yahoo
    dxy = yf.download("DX-Y.NYB", auto_adjust=False, progress=False)  # ICE Dollar Index on NYBOT
    if dxy is None or dxy.empty:
        # Fallback: Yahoo synthetic ^DXY (sometimes blocked)
        dxy = yf.download("DX-Y.NYB", auto_adjust=False, progress=False)
    # Normalize name to DXY.csv regardless of symbol used
    _save_csv(dxy, "data/DXY.csv")

    # ---- 10Y yield proxy (TNX)
    tnx = yf.download("^TNX", auto_adjust=False, progress=False)
    _save_csv(tnx, "data/TNX.csv")

    print("🎉 Bootstrap complete.")

if __name__ == "__main__":
    main()
