# select_top_signals.py
import os

import pandas as pd

# This must match what analyze_signals.py writes
CORR_FILE = "logs/signal_corr_to_fwd_return.csv"
OUTPUT_FILE = "logs/top_signals.txt"
TOP_N = 10  # adjust as needed


def main(top_n: int = TOP_N, corr_file: str = CORR_FILE, out_file: str = OUTPUT_FILE) -> str:
    """
    Load the forward-return correlation series, pick top-N absolute correlations,
    and save them to logs/top_signals.txt. Returns a short pretty string for logs.

    analyze_signals.py writes a single-column CSV with header 'corr':
      - index = feature names
      - value = correlation to forward net return (fwd_ret_net)
    """
    if not os.path.exists(corr_file):
        raise FileNotFoundError(
            f"{corr_file} not found. Make sure analyze_signals.py has run and wrote that file."
        )

    corr = pd.read_csv(corr_file, index_col=0).squeeze("columns")

    if corr.empty:
        raise ValueError(f"{corr_file} is empty — cannot pick top signals.")

    top = corr.abs().sort_values(ascending=False).head(top_n)

    # Write to txt (same format training expects to read)
    # one line per signal: "<name>,<value>"
    with open(out_file, "w") as f:
        f.write(f"Top {top_n} signals by |corr to forward net return|\n")
        for name, val in top.items():
            f.write(f"{name},{val}\n")

    pretty = f"✅ Top {top_n} signals saved to {out_file}:\n\n{top.to_string()}"
    print(pretty)
    return pretty


if __name__ == "__main__":
    main()
