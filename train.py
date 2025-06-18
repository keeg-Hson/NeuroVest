# train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
os.makedirs('models', exist_ok=True) #ensures relevant directories exist


def train_model(df, features=["RSI", "MA_20", "Volatility", "Return"], target="Event"):
    X=df[features]
    y=df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('\n📊 Model Performance:')
    print(classification_report(y_test, y_pred))
    print(f"Features used for training: {features}")


    joblib.dump(model, 'models/market_crash_model.pkl')
    print("✅ Model trained and saved as 'models/market_crash_model.pkl'")
    return model