import pandas as pd
import os

# File paths
PREDICTIONS_PATH = "logs/daily_predictions_cleaned.csv"
SPY_PATH = "data/SPY.csv"
OUTPUT_PATH = "logs/daily_predictions_enriched.csv"

# Load predictions
pred_df = pd.read_csv(PREDICTIONS_PATH)
pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors="coerce")

# Load SPY data
spy_df = pd.read_csv(SPY_PATH, index_col=0, parse_dates=True)
spy_df.reset_index(inplace=True)
spy_df.rename(columns={spy_df.columns[0]: "Date"}, inplace=True)

# Merge on date
merged_df = pd.merge(pred_df, spy_df, on="Date", how="left")

# Fill in Open, Close, Low, High — override or combine as needed
for col in ["Open", "Close", "Low", "High"]:
    merged_df[col] = merged_df.get(col, pd.Series([None]*len(merged_df)))  # ensure column exists
    if f"{col}_y" in merged_df.columns:
        merged_df[col] = merged_df[f"{col}_y"]

# Drop redundant or confusing columns
merged_df = merged_df.drop(columns=[c for c in merged_df.columns if "_x" in c or "_y" in c or c in ["Price", "Volume", "Close_Price"]])
merged_df = merged_df[merged_df["Date"].notna()]


# Save clean enriched output
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"[✅] Final enriched predictions written to {OUTPUT_PATH}")
