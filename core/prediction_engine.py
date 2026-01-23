"""
Unified Prediction Engine for NeuroVest

Consolidates prediction logic from multiple predict_*.py files:
- predict.py
- predict_all_assets.py
- predict_per_asset.py
- predict_multi_asset_ensemble.py

Provides:
- ThresholdManager: Centralized threshold loading and management
- ModelEnsembleLoader: Unified model loading
- PredictionConverter: Binary to 3-class conversion strategies
- BasePredictionEngine: Common prediction workflow

Usage:
    from core.prediction_engine import PredictionEngine

    engine = PredictionEngine()
    engine.load_models()
    predictions = engine.predict('SPY')
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, LOGS_DIR, PREDICT_CFG, PREDICTION_THRESHOLD
from utils import add_features, finalize_features, get_feature_list


# =============================================================================
# Threshold Management
# =============================================================================

class ThresholdManager:
    """
    Centralized threshold management with precedence system.

    Threshold precedence (highest to lowest):
    1. Environment variable: THRESH_PATH
    2. Explicit path argument
    3. configs/best_thresholds.json
    4. models/thresholds_fwd.json
    5. models/thresholds.json
    6. config.py PREDICT_CFG defaults
    """

    DEFAULT_PATHS = [
        'configs/best_thresholds.json',
        'models/thresholds_fwd.json',
        'models/thresholds.json',
    ]

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or '.')
        self.thresholds = None
        self._loaded_from = None

    def load(self, path: str = None) -> Dict:
        """
        Load thresholds with precedence system.

        Args:
            path: Optional explicit path

        Returns:
            Dict with 'p_min', 'ev_min', etc.
        """
        # Check environment variable first
        env_path = os.getenv('THRESH_PATH')
        if env_path and os.path.exists(env_path):
            self.thresholds = self._load_json(env_path)
            self._loaded_from = f"env:THRESH_PATH={env_path}"
            return self.thresholds

        # Check explicit path
        if path and os.path.exists(path):
            self.thresholds = self._load_json(path)
            self._loaded_from = f"explicit:{path}"
            return self.thresholds

        # Try default paths
        for default_path in self.DEFAULT_PATHS:
            full_path = self.base_dir / default_path
            if full_path.exists():
                self.thresholds = self._load_json(str(full_path))
                self._loaded_from = str(full_path)
                return self.thresholds

        # Fall back to config defaults
        self.thresholds = {
            'p_min': PREDICT_CFG.get('p_min', PREDICTION_THRESHOLD),
            'ev_min': PREDICT_CFG.get('ev_min', 0.0005),
            'avg_gain': PREDICT_CFG.get('avg_gain', 0.004),
            'avg_loss': PREDICT_CFG.get('avg_loss', 0.003),
        }
        self._loaded_from = "config.py defaults"
        return self.thresholds

    def _load_json(self, path: str) -> Dict:
        """Load and parse threshold JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        # Normalize keys
        result = {
            'p_min': data.get('p_min', data.get('threshold', PREDICTION_THRESHOLD)),
            'ev_min': data.get('ev_min', 0.0005),
            'avg_gain': data.get('avg_gain', 0.004),
            'avg_loss': data.get('avg_loss', 0.003),
        }

        # Clamp threshold to valid range
        result['p_min'] = max(0.1, min(0.9, result['p_min']))

        return result

    def get_threshold(self, key: str = 'p_min', default: float = None) -> float:
        """Get a specific threshold value"""
        if self.thresholds is None:
            self.load()
        return self.thresholds.get(key, default)

    @property
    def loaded_from(self) -> str:
        """Get the source of loaded thresholds"""
        return self._loaded_from or "not loaded"


# =============================================================================
# Model Loading
# =============================================================================

class ModelEnsembleLoader:
    """
    Unified model loading for ensemble predictions.

    Supports loading multiple model types (xgboost, lightgbm, catboost)
    with consistent error handling and feature alignment.
    """

    MODEL_TYPES = ['xgboost', 'lightgbm', 'catboost']

    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.models = {}
        self.scalers = {}
        self.feature_names = {}

    def load_ensemble(
        self,
        prefix: str = 'multi_asset',
        model_types: List[str] = None,
    ) -> Dict:
        """
        Load ensemble of models.

        Args:
            prefix: Model file prefix (e.g., 'multi_asset', 'per_asset_SPY')
            model_types: List of model types to load

        Returns:
            Dict mapping model_type to model object
        """
        model_types = model_types or self.MODEL_TYPES
        loaded = {}

        for model_type in model_types:
            model_path = self.models_dir / f"{prefix}_{model_type}.pkl"
            if not model_path.exists():
                model_path = self.models_dir / f"{prefix}_{model_type}.joblib"

            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    if isinstance(model_data, dict):
                        loaded[model_type] = model_data.get('model', model_data)
                        if 'scaler' in model_data:
                            self.scalers[model_type] = model_data['scaler']
                        if 'feature_names' in model_data:
                            self.feature_names[model_type] = model_data['feature_names']
                    else:
                        loaded[model_type] = model_data
                    print(f"[ModelLoader] Loaded {model_type} from {model_path}")
                except Exception as e:
                    print(f"[ModelLoader] Error loading {model_type}: {e}")
            else:
                print(f"[ModelLoader] Not found: {model_path}")

        self.models = loaded
        return loaded

    def get_feature_names(self, model_type: str = None) -> List[str]:
        """Get feature names for a model"""
        if model_type:
            return self.feature_names.get(model_type, [])

        # Return first available
        for names in self.feature_names.values():
            if names:
                return names

        # Fall back to default feature list
        return get_feature_list()


# =============================================================================
# Prediction Conversion Strategies
# =============================================================================

class PredictionConverter(ABC):
    """Abstract base for prediction conversion strategies"""

    @abstractmethod
    def convert(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert probabilities to class labels"""
        pass


class FixedThresholdConverter(PredictionConverter):
    """
    Convert binary probabilities to 3-class using fixed thresholds.

    Used by: predict.py
    """

    def __init__(self, crash_threshold: float = 0.3, spike_threshold: float = 0.7):
        self.crash_threshold = crash_threshold
        self.spike_threshold = spike_threshold

    def convert(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to 3-class labels.

        Args:
            probabilities: Array of probabilities (0 to 1)

        Returns:
            Array of labels: 0=CRASH, 1=NORMAL, 2=SPIKE
        """
        labels = np.ones(len(probabilities), dtype=int)  # Default: NORMAL
        labels[probabilities < self.crash_threshold] = 0  # CRASH
        labels[probabilities > self.spike_threshold] = 2  # SPIKE
        return labels


class PercentileConverter(PredictionConverter):
    """
    Convert probabilities to 3-class using percentile thresholds.

    Used by: predict_all_assets.py, predict_multi_asset_ensemble.py
    """

    def __init__(self, crash_percentile: float = 30, spike_percentile: float = 70):
        self.crash_percentile = crash_percentile
        self.spike_percentile = spike_percentile

    def convert(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Convert probabilities using dynamic percentile thresholds.

        Args:
            probabilities: Array of probabilities

        Returns:
            Array of labels: 0=CRASH, 1=NORMAL, 2=SPIKE
        """
        crash_thresh = np.percentile(probabilities, self.crash_percentile)
        spike_thresh = np.percentile(probabilities, self.spike_percentile)

        labels = np.ones(len(probabilities), dtype=int)
        labels[probabilities <= crash_thresh] = 0
        labels[probabilities >= spike_thresh] = 2
        return labels


# =============================================================================
# Main Prediction Engine
# =============================================================================

@dataclass
class PredictionResult:
    """Container for prediction results"""
    ticker: str
    date: pd.Timestamp
    label: int  # 0=CRASH, 1=NORMAL, 2=SPIKE
    label_name: str  # Human-readable label
    ensemble_prob: float
    confidence: float
    model_agreement: bool
    individual_probs: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'date': self.date,
            'label': self.label,
            'label_name': self.label_name,
            'ensemble_prob': self.ensemble_prob,
            'confidence': self.confidence,
            'model_agreement': self.model_agreement,
            **{f'{k}_prob': v for k, v in self.individual_probs.items()},
        }


class PredictionEngine:
    """
    Unified prediction engine for all assets and model types.

    Consolidates logic from:
    - predict.py
    - predict_all_assets.py
    - predict_per_asset.py
    - predict_multi_asset_ensemble.py

    Usage:
        engine = PredictionEngine()
        engine.load_models('multi_asset')

        # Single prediction
        result = engine.predict_latest('SPY')

        # Batch prediction
        results = engine.predict_history('SPY', days=30)
    """

    LABEL_NAMES = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}

    def __init__(
        self,
        models_dir: str = None,
        converter: PredictionConverter = None,
    ):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.model_loader = ModelEnsembleLoader(self.models_dir)
        self.threshold_manager = ThresholdManager()
        self.converter = converter or PercentileConverter()

        self.models = {}
        self.scaler = None
        self.feature_names = None

    def load_models(self, prefix: str = 'multi_asset') -> bool:
        """
        Load ensemble models.

        Args:
            prefix: Model prefix (e.g., 'multi_asset', 'per_asset_SPY')

        Returns:
            True if at least one model loaded
        """
        self.models = self.model_loader.load_ensemble(prefix)
        self.feature_names = self.model_loader.get_feature_names()

        # Load scaler if exists
        scaler_path = self.models_dir / f"{prefix}_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = self.models_dir / f"{prefix}_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        return len(self.models) > 0

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None,
    ) -> np.ndarray:
        """
        Prepare features for prediction.

        Args:
            df: DataFrame with OHLCV data
            feature_names: Expected feature names

        Returns:
            Feature matrix ready for prediction
        """
        feature_names = feature_names or self.feature_names or get_feature_list()

        # Add features
        df_feat, _ = add_features(df)

        # Prepare and align features
        X_df = finalize_features(df_feat, feature_names)
        X = X_df.values

        # Scale if scaler available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X, df_feat.index

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get ensemble probabilities.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (ensemble_probs, individual_model_probs)
        """
        if not self.models:
            raise RuntimeError("No models loaded. Call load_models() first.")

        individual_probs = {}
        all_probs = []

        for model_type, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    # Handle both binary and multi-class
                    if probs.ndim == 2 and probs.shape[1] >= 2:
                        probs = probs[:, 1]  # Get positive class probability
                    individual_probs[model_type] = probs
                    all_probs.append(probs)
            except Exception as e:
                print(f"[PredictionEngine] Error with {model_type}: {e}")

        if not all_probs:
            raise RuntimeError("No models produced predictions")

        # Ensemble: average probabilities
        ensemble_probs = np.mean(all_probs, axis=0)

        return ensemble_probs, individual_probs

    def predict(
        self,
        df: pd.DataFrame,
        ticker: str = 'SPY',
    ) -> pd.DataFrame:
        """
        Generate predictions for a DataFrame.

        Args:
            df: DataFrame with OHLCV data
            ticker: Asset ticker for labeling

        Returns:
            DataFrame with predictions
        """
        X, dates = self.prepare_features(df)
        ensemble_probs, individual_probs = self.predict_proba(X)

        # Convert to 3-class labels
        labels = self.converter.convert(ensemble_probs)

        # Calculate confidence and agreement
        confidence = np.abs(ensemble_probs - 0.5) * 2  # Scale to 0-1

        # Model agreement: check if all models agree on direction
        if len(individual_probs) > 1:
            all_probs = np.array(list(individual_probs.values()))
            all_labels = (all_probs > 0.5).astype(int)
            agreement = (all_labels == all_labels[0]).all(axis=0)
        else:
            agreement = np.ones(len(ensemble_probs), dtype=bool)

        # Build results DataFrame
        results = pd.DataFrame({
            'Date': dates,
            'ticker': ticker,
            'ensemble_prob': ensemble_probs,
            'prediction_label': [self.LABEL_NAMES[l] for l in labels],
            'prediction': labels,
            'confidence_score': confidence,
            'model_agreement': agreement,
        })

        # Add individual model probabilities
        for model_type, probs in individual_probs.items():
            results[f'{model_type}_prob'] = probs

        return results

    def predict_latest(
        self,
        df: pd.DataFrame,
        ticker: str = 'SPY',
    ) -> PredictionResult:
        """
        Get prediction for latest data point.

        Args:
            df: DataFrame with OHLCV data
            ticker: Asset ticker

        Returns:
            PredictionResult for latest date
        """
        results = self.predict(df.tail(1), ticker)
        row = results.iloc[0]

        return PredictionResult(
            ticker=ticker,
            date=row['Date'],
            label=row['prediction'],
            label_name=row['prediction_label'],
            ensemble_prob=row['ensemble_prob'],
            confidence=row['confidence_score'],
            model_agreement=row['model_agreement'],
            individual_probs={
                k.replace('_prob', ''): row[k]
                for k in results.columns if k.endswith('_prob') and k != 'ensemble_prob'
            },
        )

    def save_predictions(
        self,
        results: pd.DataFrame,
        path: str = None,
        append: bool = True,
    ):
        """
        Save predictions to CSV.

        Args:
            results: Predictions DataFrame
            path: Output path (defaults to logs/predictions.csv)
            append: Whether to append to existing file
        """
        path = path or str(LOGS_DIR / 'predictions.csv')

        if append and os.path.exists(path):
            existing = pd.read_csv(path, parse_dates=['Date'])
            # De-duplicate by ticker and date
            combined = pd.concat([existing, results])
            combined = combined.drop_duplicates(
                subset=['ticker', 'Date'],
                keep='last'
            )
            combined.to_csv(path, index=False)
        else:
            results.to_csv(path, index=False)

        print(f"[PredictionEngine] Saved {len(results)} predictions to {path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def run_single_prediction(
    ticker: str = 'SPY',
    model_prefix: str = 'multi_asset',
) -> PredictionResult:
    """
    Convenience function for single-asset prediction.

    Args:
        ticker: Asset ticker
        model_prefix: Model file prefix

    Returns:
        PredictionResult for latest date
    """
    from utils import load_asset_data

    engine = PredictionEngine()
    engine.load_models(model_prefix)

    df = load_asset_data(ticker)
    return engine.predict_latest(df, ticker)


def run_batch_predictions(
    tickers: List[str],
    model_prefix: str = 'multi_asset',
    save_path: str = None,
) -> pd.DataFrame:
    """
    Run predictions for multiple assets.

    Args:
        tickers: List of asset tickers
        model_prefix: Model file prefix
        save_path: Optional path to save results

    Returns:
        DataFrame with all predictions
    """
    from utils import load_asset_data

    engine = PredictionEngine()
    engine.load_models(model_prefix)

    all_results = []
    for ticker in tickers:
        try:
            df = load_asset_data(ticker)
            results = engine.predict(df.tail(30), ticker)
            all_results.append(results)
            print(f"[Batch] {ticker}: {len(results)} predictions")
        except Exception as e:
            print(f"[Batch] {ticker}: Error - {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    if save_path:
        engine.save_predictions(combined, save_path, append=False)

    return combined
