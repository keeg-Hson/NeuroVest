import pandas as pd

CORR_FILE = "logs/signal_correlations.csv"
OUTPUT_FILE = "logs/top_signals.txt"
TOP_N = 10  # You can change this

# Load correlation matrix
df_corr = pd.read_csv(CORR_FILE, index_col=0)

# Get correlation with 'Close'
correlation_with_close = df_corr["Close"].drop("Close", errors='ignore').abs()

# Get top N signals
top_signals = correlation_with_close.sort_values(ascending=False).head(TOP_N)
top_signals.to_csv(OUTPUT_FILE, header=False)

print(f"✅ Top {TOP_N} signals saved to {OUTPUT_FILE}:\n")
print(top_signals)
