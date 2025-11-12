#!/usr/bin/env python3
"""
signal_logger.py

Derives human-readable trade signals from daily prediction logs.

Assumptions
-----------
- Logs come from a model using the unified 3-class convention:
      0 = SPIKE
      1 = NORMAL
      2 = CRASH
- Spike_Conf approximates the model's confidence that the day is a SPIKE regime.
- Crash_Conf approximates the model's confidence that the day is a CRASH regime.

Signal mapping
--------------
- If Spike_Conf >= SPIKE_THRESHOLD  → "SELL"
- Else if Crash_Conf >= CRASH_THRESHOLD → "BUY"
- Else → "HOLD"
"""

import os

import pandas as pd

from utils import safe_read_csv

# === CONFIGURABLE THRESHOLDS ===
SPIKE_THRESHOLD = 0.7  # Confidence required to trigger SELL
CRASH_THRESHOLD = 0.7  # Confidence required to trigger BUY


def load_predictions(path: str = "logs/daily_predictions.csv") -> pd.DataFrame:
    # This loader keeps all columns as-is so downstream logic can work with either Close or Close_Price.
    df = safe_read_csv(path, prefer_index=False)
    return df


def determine_signal(row: pd.Series) -> tuple[str, float]:
    """
    Decide the signal for a single row given spike/crash confidences.

    Spike_Conf drives SELL (taking profits in a spike regime),
    Crash_Conf drives BUY (buying into a crash regime).
    """
    if row["Spike_Conf"] >= SPIKE_THRESHOLD:
        return "SELL", float(row["Spike_Conf"])
    elif row["Crash_Conf"] >= CRASH_THRESHOLD:
        return "BUY", float(row["Crash_Conf"])
    else:
        return "HOLD", float(max(row["Spike_Conf"], row["Crash_Conf"]))


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    signals = []
    for _, row in df.iterrows():
        signal, conf = determine_signal(row)
        signals.append(
            {
                "Date": row["Date"],
                "Signal": signal,
                "Confidence": round(conf, 3),
                # Uses Close_Price if present; otherwise falls back to Close.
                "Price": row.get("Close_Price", row.get("Close", None)),
            }
        )
    return pd.DataFrame(signals)


def save_signals(signal_df: pd.DataFrame, output_path: str = "logs/signals.csv") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    signal_df.to_csv(output_path, index=False)
    print(f"✅ Signals saved to {output_path}")


def main() -> None:
    df = load_predictions()
    signals_df = generate_signals(df)
    save_signals(signals_df)


if __name__ == "__main__":
    main()
