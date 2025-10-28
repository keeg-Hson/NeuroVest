from __future__ import annotations

import os

import pandas as pd

from utils import load_SPY_data  # canonical SPY loader (data/SPY.csv)

PREDICTIONS_PATH = "logs/daily_predictions_cleaned.csv"
OUTPUT_PATH = "logs/daily_predictions_enriched.csv"


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"{PREDICTIONS_PATH} not found.")

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    if "Date" not in pred_df.columns:
        raise KeyError(f"'Date' column missing in {PREDICTIONS_PATH}")

    # Robust date parsing with fast-path ISO and a flexible fallback
    date_series = pred_df["Date"].astype(str).str.strip()
    try:
        pred_df["Date"] = pd.to_datetime(date_series, format="%Y-%m-%d", errors="coerce")
        if pred_df["Date"].isna().all():
            raise ValueError("all-na after strict ISO parse; falling back")
    except Exception:
        pred_df["Date"] = pd.to_datetime(date_series, errors="coerce", utc=False)

    # Canonical SPY → reset index to merge on "Date"
    spy = load_SPY_data().reset_index()
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in spy.columns]
    spy = spy[keep]

    merged = pred_df.merge(spy, on="Date", how="left")
    merged = merged[merged["Date"].notna()]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Final enriched predictions written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
