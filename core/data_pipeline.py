"""
Unified Data Pipeline for NeuroVest

Consolidates all data loading, feature engineering, and preprocessing logic
that was previously duplicated across 21+ training files.

Usage:
    from core.data_pipeline import DataPipeline

    # Simple usage
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, feature_cols = pipeline.prepare_training_data('SPY')

    # Advanced usage with config
    pipeline = DataPipeline(config={
        'horizon': 5,
        'pos_threshold': 0.005,
        'test_size': 0.2,
        'use_sequences': False,
    })
    data = pipeline.prepare_training_data(['SPY', 'QQQ'], multi_asset=True)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAIN_CFG, DATA_DIR, CACHE_DIR
from utils import (
    add_features,
    finalize_features,
    get_feature_list,
    load_asset_data,
    load_SPY_data,
    add_forward_returns_and_labels,
    compute_sample_weights,
    ensure_no_future_leakage,
)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    # Labeling
    horizon: int = 1
    price_col: str = "Close"
    pos_threshold: float = 0.005
    fee_bps: float = 1.5
    slippage_bps: float = 2.0
    long_only: bool = True
    volatility_adjusted: bool = True

    # Data split
    test_size: float = 0.2
    time_based_split: bool = True  # Use time-based split (recommended for time series)

    # Feature handling
    use_cross_asset: bool = True
    use_macro: bool = True
    scale_features: bool = True

    # Sequence data (for LSTM/Transformer)
    use_sequences: bool = False
    sequence_length: int = 20

    # Sample weighting
    use_sample_weights: bool = True
    min_weight: float = 0.5
    max_weight: float = 5.0
    weight_power: float = 1.75

    # Multi-asset
    asset_type_features: bool = True  # Add asset_type_stock, asset_type_crypto flags

    # 3-class labeling (CRASH/NORMAL/SPIKE)
    use_3_class: bool = False
    crash_threshold: float = -0.02
    spike_threshold: float = 0.02


class DataPipeline:
    """
    Unified data pipeline for training and prediction.

    Consolidates data loading, feature engineering, labeling, and preprocessing
    that was previously duplicated across many training scripts.
    """

    def __init__(self, config: Optional[Union[PipelineConfig, Dict]] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: PipelineConfig instance or dict of config options
        """
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config

        self.scaler = None
        self.feature_cols = None
        self._cross_asset_df = None
        self._macro_df = None

    def load_data(
        self,
        assets: Union[str, List[str]] = 'SPY',
        data_dir: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for one or more assets.

        Args:
            assets: Single ticker or list of tickers
            data_dir: Data directory (defaults to data_cache/)

        Returns:
            Dict mapping ticker to DataFrame
        """
        if isinstance(assets, str):
            assets = [assets]

        data_dir = data_dir or str(CACHE_DIR)
        result = {}

        for ticker in assets:
            try:
                df = load_asset_data(ticker, data_dir=data_dir)
                result[ticker] = df
                print(f"[DataPipeline] Loaded {ticker}: {len(df)} rows, "
                      f"{df.index.min().date()} to {df.index.max().date()}")
            except FileNotFoundError as e:
                print(f"[DataPipeline] Warning: {e}")

        return result

    def add_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Add all technical and derived features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (DataFrame with features, list of feature columns)
        """
        return add_features(df)

    def add_labels(
        self,
        df: pd.DataFrame,
        horizon: int = None,
        pos_threshold: float = None,
    ) -> pd.DataFrame:
        """
        Add target labels for training.

        Args:
            df: DataFrame with features
            horizon: Forward horizon in days (default from config)
            pos_threshold: Minimum return threshold (default from config)

        Returns:
            DataFrame with 'y' column added
        """
        horizon = horizon or self.config.horizon
        pos_threshold = pos_threshold or self.config.pos_threshold

        if self.config.use_3_class:
            # 3-class labeling: CRASH (0), NORMAL (1), SPIKE (2)
            df = self._add_3_class_labels(df, horizon)
        else:
            # Binary labeling
            df = add_forward_returns_and_labels(
                df,
                price_col=self.config.price_col,
                horizon=horizon,
                fee_bps=self.config.fee_bps,
                slippage_bps=self.config.slippage_bps,
                long_only=self.config.long_only,
                pos_threshold=pos_threshold,
                volatility_adjusted=self.config.volatility_adjusted,
            )

        return df

    def _add_3_class_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add 3-class labels: CRASH (0), NORMAL (1), SPIKE (2)"""
        d = df.copy()
        price_col = self.config.price_col

        d['fwd_price'] = d[price_col].shift(-horizon)
        d['fwd_ret_raw'] = (d['fwd_price'] / d[price_col]) - 1.0

        # Apply transaction costs
        cost = (self.config.fee_bps + self.config.slippage_bps) * 1e-4
        d['fwd_ret_net'] = d['fwd_ret_raw'] - cost

        # 3-class labels
        d['y'] = np.select(
            [
                d['fwd_ret_net'] <= self.config.crash_threshold,
                d['fwd_ret_net'] >= self.config.spike_threshold,
            ],
            [0, 2],  # 0=CRASH, 2=SPIKE
            default=1,  # 1=NORMAL
        )

        d['horizon_forward'] = horizon
        return d

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Finalize features: fill NaN, handle infinities, etc.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names

        Returns:
            DataFrame with only feature columns, cleaned
        """
        return finalize_features(df, feature_cols)

    def split_data(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets.

        Args:
            df: Original DataFrame (for time-based split using index)
            X: Feature matrix
            y: Labels
            test_size: Fraction for test set (default from config)

        Returns:
            X_train, X_test, y_train, y_test, train_idx, test_idx
        """
        test_size = test_size or self.config.test_size
        n = len(X)

        if self.config.time_based_split:
            # Time-based split (no future leakage)
            split_idx = int(n * (1 - test_size))
            train_idx = np.arange(split_idx)
            test_idx = np.arange(split_idx, n)
        else:
            # Random split (for non-time-series or cross-validation)
            from sklearn.model_selection import train_test_split
            indices = np.arange(n)
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=42
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        return X_train, X_test, y_train, y_test, train_idx, test_idx

    def scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray = None,
        fit: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            Scaled X_train, scaled X_test (or None)
        """
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute sample weights for training.

        Args:
            df: DataFrame with 'fwd_ret_net' column

        Returns:
            Array of sample weights
        """
        if not self.config.use_sample_weights:
            return np.ones(len(df))

        return compute_sample_weights(
            df,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
            power=self.config.weight_power,
            long_only=self.config.long_only,
        )

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Args:
            y: Label array

        Returns:
            Dict mapping class to weight
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels
            sequence_length: Sequence length (default from config)

        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,)
        """
        sequence_length = sequence_length or self.config.sequence_length

        n_samples = len(X)
        n_sequences = n_samples - sequence_length

        if n_sequences <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for sequence length {sequence_length}")

        X_seq = np.array([X[i:i+sequence_length] for i in range(n_sequences)])
        y_seq = y[sequence_length:]

        return X_seq, y_seq

    def prepare_training_data(
        self,
        assets: Union[str, List[str]] = 'SPY',
        multi_asset: bool = False,
    ) -> Dict:
        """
        Complete pipeline to prepare training data.

        This is the main entry point that replaces the duplicated
        data preparation code in all training scripts.

        Args:
            assets: Ticker(s) to load
            multi_asset: If True, combine multiple assets into single dataset

        Returns:
            Dict containing:
                - X_train, X_test: Feature matrices
                - y_train, y_test: Labels
                - feature_cols: Feature column names
                - scaler: Fitted StandardScaler
                - sample_weights: Training sample weights
                - class_weights: Class weights for imbalanced data
                - df_train, df_test: Original DataFrames (for analysis)
        """
        # Load data
        data = self.load_data(assets)

        if not data:
            raise ValueError(f"No data loaded for assets: {assets}")

        if multi_asset and len(data) > 1:
            # Combine multiple assets
            df, feature_cols = self._prepare_multi_asset(data)
        else:
            # Single asset
            ticker = list(data.keys())[0]
            df = data[ticker]
            df, feature_cols = self.add_features(df)

        self.feature_cols = feature_cols

        # Add labels
        df = self.add_labels(df)

        # Drop rows with NaN labels
        df = df.dropna(subset=['y'])

        # Verify no leakage
        ensure_no_future_leakage(df, feature_cols, ['y'])

        # Prepare feature matrix
        X_df = self.prepare_features(df, feature_cols)
        X = X_df.values
        y = df['y'].values.astype(int)

        # Split data
        X_train, X_test, y_train, y_test, train_idx, test_idx = self.split_data(df, X, y)

        # Scale features
        if self.config.scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)

        # Create sequences if needed
        if self.config.use_sequences:
            X_train, y_train = self.create_sequences(X_train, y_train)
            X_test, y_test = self.create_sequences(X_test, y_test)

        # Compute weights
        df_train = df.iloc[train_idx]
        sample_weights = self.compute_sample_weights(df_train)

        # Adjust weights if sequences were created
        if self.config.use_sequences:
            sample_weights = sample_weights[self.config.sequence_length:]

        class_weights = self.compute_class_weights(y_train)

        # Report stats
        print(f"\n[DataPipeline] Training data prepared:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'scaler': self.scaler,
            'sample_weights': sample_weights,
            'class_weights': class_weights,
            'df_train': df_train,
            'df_test': df.iloc[test_idx],
        }

    def _prepare_multi_asset(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare multi-asset training data.

        Combines multiple assets into a single DataFrame with
        asset type indicators.
        """
        dfs = []
        feature_cols = None

        for ticker, df in data.items():
            df_feat, cols = self.add_features(df)

            if feature_cols is None:
                feature_cols = cols
            else:
                # Use intersection of features
                feature_cols = list(set(feature_cols) & set(cols))

            # Add asset identifier
            df_feat['ticker'] = ticker

            # Add asset type flags
            if self.config.asset_type_features:
                is_crypto = '/' in ticker or 'USDT' in ticker
                df_feat['asset_type_stock'] = 0 if is_crypto else 1
                df_feat['asset_type_crypto'] = 1 if is_crypto else 0

            dfs.append(df_feat)

        # Combine
        combined = pd.concat(dfs, axis=0)
        combined = combined.sort_index()

        # Add asset type features to feature list
        if self.config.asset_type_features:
            feature_cols = feature_cols + ['asset_type_stock', 'asset_type_crypto']

        return combined, feature_cols

    def prepare_prediction_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> np.ndarray:
        """
        Prepare data for prediction (no labels).

        Args:
            df: DataFrame with OHLCV data
            feature_cols: Feature columns (use saved if None)

        Returns:
            Feature matrix ready for prediction
        """
        feature_cols = feature_cols or self.feature_cols

        if feature_cols is None:
            raise ValueError("feature_cols must be provided or set via prepare_training_data")

        # Add features
        df, _ = self.add_features(df)

        # Prepare features
        X_df = self.prepare_features(df, feature_cols)
        X = X_df.values

        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Create sequences if configured
        if self.config.use_sequences:
            X, _ = self.create_sequences(X, np.zeros(len(X)))

        return X

    def save(self, path: str):
        """Save pipeline state (scaler, feature_cols, config)"""
        import joblib

        state = {
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'config': self.config,
        }
        joblib.dump(state, path)
        print(f"[DataPipeline] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'DataPipeline':
        """Load pipeline from saved state"""
        import joblib

        state = joblib.load(path)
        pipeline = cls(config=state['config'])
        pipeline.scaler = state['scaler']
        pipeline.feature_cols = state['feature_cols']
        print(f"[DataPipeline] Loaded from {path}")
        return pipeline


# Convenience functions for backward compatibility
def prepare_spy_training_data(**kwargs) -> Dict:
    """Convenience function to prepare SPY training data"""
    pipeline = DataPipeline(config=kwargs)
    return pipeline.prepare_training_data('SPY')


def prepare_multi_asset_training_data(assets: List[str], **kwargs) -> Dict:
    """Convenience function to prepare multi-asset training data"""
    pipeline = DataPipeline(config=kwargs)
    return pipeline.prepare_training_data(assets, multi_asset=True)
