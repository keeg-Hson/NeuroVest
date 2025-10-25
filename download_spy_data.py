#download_spy_data.py

from __future__ import annotations
import os
import pandas as pd
from utils import load_SPY_data  # canonical SPY loader (data/SPY.csv)

PREDICTIONS_PATH = "logs/daily_predictions_cleaned.csv"
OUTPUT_PATH      = "logs/daily_predictions_enriched.csv"

def main():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"{PREDICTIONS_PATH} not found.")

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    if "Date" not in pred_df.columns:
        raise KeyError(f"'Date' column missing in {PREDICTIONS_PATH}")
    pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors="coerce")

    # Canonical SPY (Date index) -> reset to column for merge
    spy = load_SPY_data().reset_index()

    # Keep only the OHLCV columns we actually need for enrichment
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in spy.columns]
    spy  = spy[keep]

    merged = pred_df.merge(spy, on="Date", how="left")
    merged = merged[merged["Date"].notna()]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Final enriched predictions written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()