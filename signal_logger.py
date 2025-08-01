# signal_logger.py
import pandas as pd
import os

# === CONFIGURABLE THRESHOLDS ===
SPIKE_THRESHOLD = 0.7  # Confidence required to trigger SELL
CRASH_THRESHOLD = 0.7  # Confidence required to trigger BUY

def load_predictions(path="logs/daily_predictions.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

def determine_signal(row):
    if row["Spike_Conf"] >= SPIKE_THRESHOLD:
        return "SELL", row["Spike_Conf"]
    elif row["Crash_Conf"] >= CRASH_THRESHOLD:
        return "BUY", row["Crash_Conf"]
    else:
        return "HOLD", max(row["Spike_Conf"], row["Crash_Conf"])

def generate_signals(df):
    signals = []
    for _, row in df.iterrows():
        signal, conf = determine_signal(row)
        signals.append({
            "Date": row["Date"],
            "Signal": signal,
            "Confidence": round(conf, 3),
            "Price": row.get("Close_Price", row.get("Close", None))  # support either
        })
    return pd.DataFrame(signals)

def save_signals(signal_df, output_path="logs/signals.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    signal_df.to_csv(output_path, index=False)
    print(f"✅ Signals saved to {output_path}")

def main():
    df = load_predictions()
    signals_df = generate_signals(df)
    save_signals(signals_df)

if __name__ == "__main__":
    main()
