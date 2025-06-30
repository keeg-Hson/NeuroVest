#viz.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_model_performance(csv_path="logs/model_performance.csv"):
    if not os.path.exists(csv_path):
        print(f"[⚠️] No model performance file found at {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    if df.empty or "Accuracy" not in df.columns:
        print("[⚠️] Performance file is empty or malformed.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Accuracy"], label="Accuracy", marker="o")
    plt.plot(df["Date"], df["Precision"], label="Precision", marker="x")
    plt.plot(df["Date"], df["Recall"], label="Recall", marker="^")
    plt.plot(df["Date"], df["F1"], label="F1 Score", marker="s")

    plt.title("Model Performance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = "graphs/model_performance_trend.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[✅] Performance trend plot saved to {output_path}")
