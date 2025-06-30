# evaluate.py
#model evaluation  

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

LABELED_LOG = "logs/labeled_predictions.csv"

def evaluate_predictions():
    print("🔍 Evaluating predictions...")
    try:
        df = pd.read_csv(LABELED_LOG, parse_dates=["Timestamp"])

        # Drop rows where Actual_Event is missing (NaN)
        df = df.dropna(subset=["Actual_Event"])
        df["Actual_Event"] = df["Actual_Event"].astype(int)

        total = len(df)
        if total == 0:
            print("[⚠️] No labeled actual events found — cannot evaluate.")
            return

        # Basic comparison
        correct = (df["Prediction"] == df["Actual_Event"]).sum()
        accuracy = correct / total

        print(f"\n📊 Evaluation Report:")
        print(f"Total predictions with real outcomes: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Basic accuracy: {accuracy*100:.2f}%\n")

        print("🔍 Classification Report:")
        print(classification_report(df["Actual_Event"], df["Prediction"], digits=2))

        print("\n📊 Class distribution:")
        print(df['Actual_Event'].value_counts())


    except FileNotFoundError:
        print("[❌] Labeled predictions file not found. Run labeling first.")
    except Exception as e:
        print(f"[⚠️] Evaluation failed: {e}")

if __name__ == "__main__":
    evaluate_predictions()
