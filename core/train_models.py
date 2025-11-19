"""
Train ML models for stock prediction

Uses joblib for better model persistence and compatibility
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils import add_features
except ImportError:
    # Fallback: define simplified feature engineering
    def add_features(df):
        """Simplified feature engineering"""
        d = df.copy()

        # Returns
        d['Daily_Return'] = d['Close'].pct_change()
        d['Return_5d'] = d['Close'].pct_change(5)
        d['Return_20d'] = d['Close'].pct_change(20)

        # Moving averages
        for period in [10, 20, 50, 200]:
            d[f'MA_{period}'] = d['Close'].rolling(period).mean()
            d[f'Close_to_MA_{period}'] = d['Close'] / d[f'MA_{period}'] - 1

        # Volatility
        d['Volatility_20'] = d['Daily_Return'].rolling(20).std()
        d['ATR'] = ((d['High'] - d['Low']).rolling(14).mean())

        # Volume
        d['Volume_MA_20'] = d['Volume'].rolling(20).mean()
        d['Volume_Ratio'] = d['Volume'] / d['Volume_MA_20']

        # RSI
        delta = d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        d['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = d['Close'].ewm(span=12).mean()
        ema26 = d['Close'].ewm(span=26).mean()
        d['MACD'] = ema12 - ema26
        d['MACD_Signal'] = d['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        d['BB_Middle'] = d['Close'].rolling(20).mean()
        d['BB_Std'] = d['Close'].rolling(20).std()
        d['BB_Upper'] = d['BB_Middle'] + 2 * d['BB_Std']
        d['BB_Lower'] = d['BB_Middle'] - 2 * d['BB_Std']
        d['BB_Position'] = (d['Close'] - d['BB_Lower']) / (d['BB_Upper'] - d['BB_Lower'])

        return d


class ModelTrainer:
    """
    Train ensemble of ML models for stock prediction
    """

    def __init__(self, output_dir='../models'):
        """
        Initialize model trainer

        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.scaler = None
        self.feature_columns = None

    def prepare_data(self, df, target_days=10, min_return_threshold=0.02):
        """
        Prepare data for training

        Args:
            df: DataFrame with OHLCV data
            target_days: Number of days to look ahead for target
            min_return_threshold: Minimum return to classify as positive (2%)

        Returns:
            X, y: Features and labels
        """
        print(f"\nPreparing training data...")
        print(f"  Input data: {len(df)} rows")

        # Add technical features
        result = add_features(df)
        if isinstance(result, tuple):
            df, _ = result  # Unpack tuple (dataframe, feature_columns)
        else:
            df = result

        # Create target: future return over target_days
        df['Future_Return'] = df['Close'].shift(-target_days) / df['Close'] - 1

        # Binary classification: 1 if return > threshold, 0 otherwise
        df['Target'] = (df['Future_Return'] > min_return_threshold).astype(int)

        # Drop rows with NaN target
        df = df.dropna(subset=['Target', 'Future_Return'])

        print(f"  After feature engineering: {len(df)} rows")

        # Select feature columns (exclude target and metadata)
        exclude_cols = ['Target', 'Future_Return', 'Date']
        self.feature_columns = [col for col in df.columns
                                if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

        print(f"  Features: {len(self.feature_columns)}")

        # Fill NaN values with median
        for col in self.feature_columns:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Replace infinite values
        df[self.feature_columns] = df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        df[self.feature_columns] = df[self.feature_columns].fillna(0)

        # Extract features and target
        X = df[self.feature_columns].values
        y = df['Target'].values

        # Check class balance
        pos_pct = (y == 1).sum() / len(y) * 100
        print(f"  Class balance: {pos_pct:.1f}% positive, {100-pos_pct:.1f}% negative")

        return X, y, df

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model

        Returns:
            Trained model
        """
        print("\nTraining XGBoost...")

        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Val accuracy: {val_acc:.3f}")

        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """
        Train LightGBM model

        Returns:
            Trained model
        """
        print("\nTraining LightGBM...")

        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # Evaluate
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Val accuracy: {val_acc:.3f}")

        return model

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Train Random Forest model

        Returns:
            Trained model
        """
        print("\nTraining Random Forest...")

        params = {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Val accuracy: {val_acc:.3f}")

        return model

    def train_all(self, X, y, test_size=0.2):
        """
        Train all models in ensemble

        Args:
            X: Feature matrix
            y: Target labels
            test_size: Validation set size

        Returns:
            Dict of trained models
        """
        print("=" * 70)
        print("TRAINING ENSEMBLE MODELS")
        print("=" * 70)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")

        # Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train models
        self.models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        self.models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.models['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)

        print("\n" + "=" * 70)
        print("✓ Training complete")
        print("=" * 70)

        return self.models

    def save_models(self, prefix='stock'):
        """
        Save all models using joblib

        Args:
            prefix: Prefix for model filenames
        """
        print(f"\nSaving models to {self.output_dir}...")

        # Save each model
        for name, model in self.models.items():
            filepath = self.output_dir / f"{prefix}_{name}.joblib"
            joblib.dump(model, filepath)
            print(f"  ✓ Saved {name} to {filepath}")

        # Save scaler
        scaler_path = self.output_dir / f"{prefix}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Saved scaler to {scaler_path}")

        # Save feature columns
        features_path = self.output_dir / f"{prefix}_features.joblib"
        joblib.dump(self.feature_columns, features_path)
        print(f"  ✓ Saved feature columns to {features_path}")

        print("\n✓ All models saved successfully")

    def load_models(self, prefix='stock'):
        """
        Load all models using joblib

        Args:
            prefix: Prefix for model filenames

        Returns:
            Dict of loaded models
        """
        print(f"\nLoading models from {self.output_dir}...")

        model_names = ['xgboost', 'lightgbm', 'random_forest']

        for name in model_names:
            filepath = self.output_dir / f"{prefix}_{name}.joblib"
            if filepath.exists():
                self.models[name] = joblib.load(filepath)
                print(f"  ✓ Loaded {name}")
            else:
                print(f"  ✗ Model not found: {filepath}")

        # Load scaler
        scaler_path = self.output_dir / f"{prefix}_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Loaded scaler")

        # Load feature columns
        features_path = self.output_dir / f"{prefix}_features.joblib"
        if features_path.exists():
            self.feature_columns = joblib.load(features_path)
            print(f"  ✓ Loaded feature columns ({len(self.feature_columns)} features)")

        return self.models

    def predict_ensemble(self, X, voting_threshold=0.5):
        """
        Make ensemble prediction

        Args:
            X: Feature matrix
            voting_threshold: Fraction of models that must agree (0.5 = majority)

        Returns:
            predictions: Binary predictions
            probabilities: Average probability across models
        """
        if not self.models:
            raise ValueError("No models loaded. Train or load models first.")

        predictions = []
        probabilities = []

        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            pred = (pred_proba > 0.5).astype(int)

            predictions.append(pred)
            probabilities.append(pred_proba)

        # Stack predictions
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        probabilities = np.array(probabilities)

        # Ensemble voting
        n_models = len(self.models)
        votes_required = int(np.ceil(n_models * voting_threshold))

        ensemble_pred = (predictions.sum(axis=0) >= votes_required).astype(int)
        ensemble_proba = probabilities.mean(axis=0)

        return ensemble_pred, ensemble_proba


def main():
    """
    Main training function
    """
    from data_loader import DataLoader

    print("=" * 70)
    print("STOCK MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading SPY data...")
    loader = DataLoader(data_dir='..')
    spy = loader.load_csv('SPY')
    print(f"✓ Loaded {len(spy)} days of SPY data")

    # Initialize trainer
    trainer = ModelTrainer()

    # Prepare data
    X, y, df = trainer.prepare_data(spy, target_days=10, min_return_threshold=0.02)

    # Train models
    models = trainer.train_all(X, y, test_size=0.2)

    # Save models
    trainer.save_models(prefix='stock')

    # Test ensemble prediction
    print("\n" + "=" * 70)
    print("TESTING ENSEMBLE PREDICTION")
    print("=" * 70)

    # Get predictions on validation set (last 20%)
    val_start = int(len(X) * 0.8)
    X_val = X[val_start:]
    y_val = y[val_start:]

    ensemble_pred, ensemble_proba = trainer.predict_ensemble(X_val, voting_threshold=0.5)

    accuracy = (ensemble_pred == y_val).mean()
    print(f"\nEnsemble validation accuracy: {accuracy:.3f}")

    # Analyze high-confidence predictions
    high_conf_mask = ensemble_proba > 0.6
    if high_conf_mask.sum() > 0:
        high_conf_acc = (ensemble_pred[high_conf_mask] == y_val[high_conf_mask]).mean()
        print(f"High-confidence predictions (>60%): {high_conf_mask.sum()} trades, "
              f"{high_conf_acc:.3f} accuracy")

    print("\n" + "=" * 70)
    print("✓ Training complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
