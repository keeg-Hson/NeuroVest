# data_utils.py
from __future__ import annotations

import os

import pandas as pd

from utils import load_SPY_data  # canonical loader (reads data/SPY.csv)


def load_spy_daily_data(path: str | None = None) -> pd.DataFrame:
    """
    Backward-compatible helper that always delegates to the canonical loader.
    Returns a DataFrame with a DatetimeIndex and a mirrored 'Date' column
    for legacy call-sites that expect it.
    """
    df = load_SPY_data()
    out = df.copy()
    out["Date"] = out.index
    return out


def log_rolling_accuracy(
    current_date, acc_7d, acc_30d, filepath: str = "logs/model_performance.csv"
) -> None:
    """
    Append rolling accuracy metrics to a CSV. Creates the file if missing.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # normalize date -> 'YYYY-MM-DD'
    try:
        date_str = current_date.strftime("%Y-%m-%d")
    except Exception:
        # accept datetime/date/str
        date_str = str(getattr(current_date, "date", lambda: current_date)())

    row = pd.DataFrame(
        [
            {
                "Date": date_str,
                "Accuracy_7d": float(acc_7d),
                "Accuracy_30d": float(acc_30d),
            }
        ]
    )

    header_needed = (not os.path.exists(filepath)) or os.path.getsize(filepath) == 0
    row.to_csv(filepath, mode="a", index=False, header=header_needed)
