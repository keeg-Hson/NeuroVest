#backtest_module.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_data():
    spy = pd.read_csv("data/SPY.csv", names=["Date", "Close", "High", "Low", "Open", "Volume"], header=0, parse_dates=["Date"])
    preds = pd.read_csv("logs/labeled_predictions.csv", parse_dates=["Date"])

    #clean up the label and prediction fields for clarity
    label_map = {0: "none", 1: "crash", 2: "spike"}
    preds["True_Label"] = preds["Label"].map(label_map)
    preds["Prediction"] = preds["Prediction"].map(label_map)

    #rename Close_Price for consistency
    preds = preds.rename(columns={"Close_Price": "Close"})

    return spy, preds

def plot_predictions(merged_df, output_path="graphs/prediction_overlay.png"):
    plt.figure(figsize=(14, 6))

    # Plot actual SPY close prices
    plt.plot(merged_df["Date"], merged_df["Close_x"], label="SPY Close", color="black", alpha=0.6)

    # Overlay predictions
    spikes = merged_df[merged_df["Prediction"] == "spike"]
    crashes = merged_df[merged_df["Prediction"] == "crash"]

    plt.scatter(spikes["Date"], spikes["Close_x"], color="green", label="Predicted Spike", marker="^", s=60)
    plt.scatter(crashes["Date"], crashes["Close_x"], color="red", label="Predicted Crash", marker="v", s=60)

    plt.legend()
    plt.title("SPY Price with Predicted Spikes/Crashes")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)  # save as string path
    plt.close()



def evaluate_predictions(pred_df, output_path="logs/model_performance.csv"):
    true_labels = pred_df["True_Label"]
    predicted = pred_df["Prediction"]

    # Filter out "none" class if desired
    filtered = pred_df[pred_df["Prediction"] != "none"]
    if not filtered.empty:
        report = classification_report(filtered["True_Label"], filtered["Prediction"], output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(output_path)
        print("✅ Evaluation metrics saved to:", output_path)
    else:
        print("⚠️ No non-'none' predictions to evaluate.")

def run_backtest():
    spy_df, pred_df = load_data()

    #ensure datetime format
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors='coerce')

    #normalize to remove time portion
    pred_df["Date"] = pred_df["Date"].dt.normalize()

    #drop rows where parsing failed
    pred_df = pred_df.dropna(subset=["Date"])

    #merge for plotting
    merged = pd.merge(spy_df, pred_df, on="Date", how="inner")

    print("Merged Columns:", merged.columns.tolist())


    plot_predictions(merged)
    evaluate_predictions(merged)



if __name__ == "__main__":
    run_backtest()
