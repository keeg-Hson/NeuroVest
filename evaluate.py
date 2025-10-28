# evaluate.py
# Model Evaluation + Outcome Labeling
import pandas as pd
from sklearn.metrics import classification_report

from utils import load_SPY_data

LABELED_LOG = "logs/labeled_predictions.csv"
METRIC_OUTPUT_CSV = "logs/model_performance.csv"


def label_prediction_outcomes():
    print("🔖 Labeling prediction outcomes...")

    try:
        pred = pd.read_csv("logs/daily_predictions.csv", parse_dates=["Timestamp"])
        spy = load_SPY_data()  # index = Date

        spy["Date"] = pd.to_datetime(spy["Date"])

        # align formats
        pred["Date"] = pred["Timestamp"].dt.date
        spy["Date"] = spy["Date"].dt.date

        merged = pd.merge(pred, spy, on="Date", how="left")

        if "4. close" not in merged.columns:
            raise KeyError("'4. close' column missing from SPY data.")

        # create 'Actual_Event': 1 if tomorrow’s return > 0.5%, else 0
        merged["next_close"] = merged["4. close"].shift(-1)
        merged["return_tomorrow"] = (merged["next_close"] - merged["4. close"]) / merged["4. close"]
        merged["Actual_Event"] = (merged["return_tomorrow"] > 0.005).astype(int)

        # add Outcome label
        def classify_outcome(row):
            if row["Prediction"] == 1 and row["Actual_Event"] == 1:
                return "TP"
            elif row["Prediction"] == 1 and row["Actual_Event"] == 0:
                return "FP"
            elif row["Prediction"] == 0 and row["Actual_Event"] == 0:
                return "TN"
            elif row["Prediction"] == 0 and row["Actual_Event"] == 1:
                return "FN"
            else:
                return "?"

        merged["Outcome"] = merged.apply(classify_outcome, axis=1)

        merged.to_csv(LABELED_LOG, index=False)
        print(f"✅ Labeled outcomes saved to {LABELED_LOG}")
    except Exception as e:
        print(f"[❌] Failed to label outcomes: {e}")


def evaluate_predictions():
    print("📊 Evaluating classification metrics...")
    try:
        df = pd.read_csv(LABELED_LOG, parse_dates=["Timestamp"])
        df = df.dropna(subset=["Actual_Event"])
        df["Actual_Event"] = df["Actual_Event"].astype(int)

        if "Outcome" not in df.columns:
            raise ValueError("Missing Outcome column — did labeling fail?")

        total = len(df)
        correct = (df["Prediction"] == df["Actual_Event"]).sum()
        accuracy = correct / total

        print("\n📊 Evaluation Report:")
        print(f"Total evaluated: {total}")
        print(f"Accuracy: {accuracy:.2%}")

        # get full classification metrics
        clf_report = classification_report(
            df["Actual_Event"],
            df["Prediction"],
            output_dict=True,
            zero_division=0,  # prevent errors when division by zero
        )

        print("\n🔍 Classification Report:")
        print(pd.DataFrame(clf_report).transpose())

        # save report to CSV
        pd.DataFrame(clf_report).transpose().to_csv(METRIC_OUTPUT_CSV)
        print(f"\n💾 Saved detailed report to {METRIC_OUTPUT_CSV}")

    except FileNotFoundError:
        print("[❌] Labeled predictions file not found. Run labeling first.")
    except Exception as e:
        print(f"[⚠️] Evaluation failed: {e}")


if __name__ == "__main__":
    label_prediction_outcomes()
    evaluate_predictions()
