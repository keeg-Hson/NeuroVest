import pandas as pd


def generate_labels(
    prediction_file="logs/daily_predictions.csv", output_file="logs/labeled_predictions.csv"
):
    df = pd.read_csv(prediction_file, parse_dates=["Timestamp"])

    # Example rule-based labeling: label spikes and crashes based on future returns
    df["Label"] = 0  # 0 = neutral
    df.loc[df["Spike_Conf"] > 0.7, "Label"] = 2  # 2 = spike
    df.loc[df["Crash_Conf"] > 0.7, "Label"] = 1  # 1 = crash

    df.to_csv(output_file, index=False)
    print(f"[✔] Labeled data saved to {output_file}")


if __name__ == "__main__":
    generate_labels()
