import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model(labeled_file="logs/labeled_predictions.csv", model_output="models/rf_trained.pkl"):
    df = pd.read_csv(labeled_file)

    # Example features and label
    X = df[["Crash_Conf", "Spike_Conf"]]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_output)
    print(f"[✔] Trained model saved to {model_output}")
    print(f"[ℹ] Training accuracy: {model.score(X_train, y_train):.2%}")

if __name__ == "__main__":
    train_model()