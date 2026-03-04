"""
Base Model Classes for NeuroVest

Consolidates model architectures from multiple training scripts:
- Tree-based: XGBoost, LightGBM, CatBoost (from train_lightgbm_catboost.py, train_improved.py)
- LSTM variants: Standard, Attention, CNN-LSTM (from train_lstm_*.py)
- Transformer: Encoder-based (from train_transformer_model.py)
- Meta-learners: Stacking ensemble (from train_ensemble_stacking.py)

All models implement a common interface:
    model.fit(X, y, **kwargs)
    model.predict(X) -> predictions
    model.predict_proba(X) -> probabilities
    model.save(path)
    model.load(path)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Base configuration for all models"""
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Provides common interface for training, prediction, and serialization.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.classes_ = None
        self.training_history = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        pass

    def save(self, path: str):
        """Save model to disk"""
        state = {
            'model': self.model,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'classes_': self.classes_,
            'training_history': self.training_history,
        }
        joblib.dump(state, path)
        print(f"[{self.__class__.__name__}] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk"""
        state = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = state['model']
        instance.config = state['config']
        instance.is_fitted = state['is_fitted']
        instance.feature_names = state['feature_names']
        instance.classes_ = state['classes_']
        instance.training_history = state.get('training_history', {})
        print(f"[{cls.__name__}] Loaded from {path}")
        return instance

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return None


# =============================================================================
# Tree-Based Models (XGBoost, LightGBM, CatBoost)
# =============================================================================

@dataclass
class TreeModelConfig(ModelConfig):
    """Configuration for tree-based models"""
    model_type: str = 'xgboost'  # xgboost, lightgbm, catboost
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50

    # Custom scoring
    scoring: str = 'logloss'  # logloss, profit, sharpe

    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0


class TreeEnsembleModel(BaseModel):
    """
    Tree-based ensemble model supporting XGBoost, LightGBM, and CatBoost.

    Consolidates:
    - train_lightgbm_catboost.py
    - train_improved.py
    - train_profit_optimized.py
    - core/train_models.py

    Usage:
        model = TreeEnsembleModel(model_type='xgboost')
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        probs = model.predict_proba(X_test)
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        config: Optional[TreeModelConfig] = None,
        **kwargs
    ):
        if config is None:
            config = TreeModelConfig(model_type=model_type, **kwargs)
        super().__init__(config)
        self.model_type = model_type

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_weight: np.ndarray = None,
        feature_names: List[str] = None,
        **kwargs
    ) -> 'TreeEnsembleModel':
        """
        Train the tree model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            sample_weight: Sample weights
            feature_names: Feature names for importance

        Returns:
            self
        """
        self.feature_names = feature_names
        self.classes_ = np.unique(y)

        if self.model_type == 'xgboost':
            self._fit_xgboost(X, y, X_val, y_val, sample_weight)
        elif self.model_type == 'lightgbm':
            self._fit_lightgbm(X, y, X_val, y_val, sample_weight)
        elif self.model_type == 'catboost':
            self._fit_catboost(X, y, X_val, y_val, sample_weight)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.is_fitted = True
        return self

    def _fit_xgboost(self, X, y, X_val, y_val, sample_weight):
        """Train XGBoost model"""
        import xgboost as xgb

        params = {
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
        }

        self.model = xgb.XGBClassifier(**params)

        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = self.config.verbose > 0

        self.model.fit(X, y, **fit_params)

    def _fit_lightgbm(self, X, y, X_val, y_val, sample_weight):
        """Train LightGBM model"""
        import lightgbm as lgb

        params = {
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
        }

        self.model = lgb.LGBMClassifier(**params)

        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['callbacks'] = [
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            ]

        self.model.fit(X, y, **fit_params)

    def _fit_catboost(self, X, y, X_val, y_val, sample_weight):
        """Train CatBoost model"""
        from catboost import CatBoostClassifier

        params = {
            'depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'iterations': self.config.n_estimators,
            'subsample': self.config.subsample,
            'rsm': self.config.colsample_bytree,
            'l2_leaf_reg': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'thread_count': self.config.n_jobs if self.config.n_jobs > 0 else -1,
            'loss_function': 'Logloss',
            'verbose': self.config.verbose,
        }

        self.model = CatBoostClassifier(**params)

        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = self.config.early_stopping_rounds

        self.model.fit(X, y, **fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance"""
        if not self.is_fitted or self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            return dict(enumerate(importances))

        return None


# =============================================================================
# LSTM Models
# =============================================================================

@dataclass
class LSTMConfig(ModelConfig):
    """Configuration for LSTM models"""
    layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    recurrent_dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5

    # Architecture variants
    use_attention: bool = False
    use_cnn: bool = False
    cnn_filters: int = 64
    cnn_kernel_size: int = 3

    # Loss function
    loss_type: str = 'binary_crossentropy'  # binary_crossentropy, focal


class LSTMModel(BaseModel):
    """
    LSTM model with configurable architecture.

    Consolidates:
    - train_lstm_model.py
    - train_lstm_focal_loss.py
    - train_lstm_v2_focal.py
    - train_attention_lstm.py
    - train_cnn_lstm.py

    Usage:
        # Standard LSTM
        model = LSTMModel(layers=[128, 64])
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # LSTM with attention
        model = LSTMModel(use_attention=True)

        # CNN-LSTM
        model = LSTMModel(use_cnn=True)

        # Focal loss for class imbalance
        model = LSTMModel(loss_type='focal')
    """

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        **kwargs
    ):
        if config is None:
            config = LSTMConfig(**kwargs)
        super().__init__(config)
        self.history = None

    def _build_model(self, input_shape: Tuple[int, int], n_classes: int):
        """Build Keras model"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers, Model
        except ImportError:
            from keras import layers, Model
            import keras

        inputs = layers.Input(shape=input_shape)
        x = inputs

        # Optional CNN layer
        if self.config.use_cnn:
            x = layers.Conv1D(
                filters=self.config.cnn_filters,
                kernel_size=self.config.cnn_kernel_size,
                padding='same',
                activation='relu'
            )(x)
            x = layers.MaxPooling1D(pool_size=2)(x)

        # LSTM layers
        for i, units in enumerate(self.config.layers):
            return_sequences = (i < len(self.config.layers) - 1) or self.config.use_attention
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout,
                recurrent_dropout=self.config.recurrent_dropout,
            )(x)

        # Optional attention layer
        if self.config.use_attention:
            # Simple attention mechanism
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(self.config.layers[-1])(attention)
            attention = layers.Permute([2, 1])(attention)

            x = layers.Multiply()([x, attention])
            x = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(x)

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout)(x)

        # Output
        if n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(n_classes, activation='softmax')(x)

        model = Model(inputs, outputs)

        # Compile
        loss = self._get_loss_function(n_classes)
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model

    def _get_loss_function(self, n_classes: int):
        """Get loss function based on config"""
        try:
            from tensorflow import keras
            import tensorflow.keras.backend as K
        except ImportError:
            import keras
            import keras.backend as K

        if self.config.loss_type == 'focal':
            # Focal loss for class imbalance
            def focal_loss(gamma=2.0, alpha=0.25):
                def loss_fn(y_true, y_pred):
                    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                    focal_weight = (alpha * y_true + (1 - alpha) * (1 - y_true)) * K.pow(1 - pt, gamma)
                    return -K.mean(focal_weight * K.log(pt))
                return loss_fn
            return focal_loss()
        elif n_classes == 2:
            return 'binary_crossentropy'
        else:
            return 'categorical_crossentropy'

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        class_weight: Dict = None,
        **kwargs
    ) -> 'LSTMModel':
        """
        Train the LSTM model.

        Args:
            X: Training sequences (n_samples, sequence_length, n_features)
            y: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            class_weight: Class weights for imbalanced data

        Returns:
            self
        """
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        input_shape = (X.shape[1], X.shape[2])

        # Build model
        self.model = self._build_model(input_shape, n_classes)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=self.config.verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                verbose=self.config.verbose
            ),
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train
        self.history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.config.verbose,
        )

        self.training_history = {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'accuracy': self.history.history.get('accuracy', []),
            'val_accuracy': self.history.history.get('val_accuracy', []),
        }

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        probs = self.model.predict(X, verbose=0)
        if len(self.classes_) == 2:
            return (probs.flatten() > 0.5).astype(int)
        else:
            return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        probs = self.model.predict(X, verbose=0)
        if len(self.classes_) == 2:
            probs = probs.flatten()
            return np.column_stack([1 - probs, probs])
        return probs

    def save(self, path: str):
        """Save LSTM model"""
        # Save Keras model separately
        model_path = path.replace('.joblib', '_keras.h5')
        self.model.save(model_path)

        # Save other state
        state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_,
            'training_history': self.training_history,
            'model_path': model_path,
        }
        joblib.dump(state, path)
        print(f"[LSTMModel] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """Load LSTM model"""
        try:
            from tensorflow import keras
        except ImportError:
            import keras

        state = joblib.load(path)
        instance = cls(config=state['config'])
        instance.is_fitted = state['is_fitted']
        instance.classes_ = state['classes_']
        instance.training_history = state.get('training_history', {})
        instance.model = keras.models.load_model(state['model_path'], compile=False)
        print(f"[LSTMModel] Loaded from {path}")
        return instance


# =============================================================================
# Transformer Model
# =============================================================================

@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Transformer model"""
    num_heads: int = 4
    num_blocks: int = 3
    ff_dim: int = 128
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15


class TransformerModel(BaseModel):
    """
    Transformer encoder model for time series.

    Consolidates: train_transformer_model.py

    Usage:
        model = TransformerModel(num_heads=4, num_blocks=3)
        model.fit(X_train, y_train)
    """

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        **kwargs
    ):
        if config is None:
            config = TransformerConfig(**kwargs)
        super().__init__(config)
        self.history = None

    def _build_model(self, input_shape: Tuple[int, int], n_classes: int):
        """Build Transformer model"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers, Model
        except ImportError:
            from keras import layers, Model
            import keras

        inputs = layers.Input(shape=input_shape)
        x = inputs

        # Positional encoding (simple learned embedding)
        seq_len, d_model = input_shape
        positions = layers.Embedding(seq_len, d_model)(
            keras.backend.arange(seq_len)
        )
        x = x + positions

        # Transformer blocks
        for _ in range(self.config.num_blocks):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=d_model // self.config.num_heads,
            )(x, x)
            attn_output = layers.Dropout(self.config.dropout)(attn_output)
            x = layers.LayerNormalization()(x + attn_output)

            # Feed-forward network
            ff_output = layers.Dense(self.config.ff_dim, activation='relu')(x)
            ff_output = layers.Dense(d_model)(ff_output)
            ff_output = layers.Dropout(self.config.dropout)(ff_output)
            x = layers.LayerNormalization()(x + ff_output)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout)(x)

        if n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(n_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'

        model = Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        class_weight: Dict = None,
        **kwargs
    ) -> 'TransformerModel':
        """Train the Transformer model"""
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        input_shape = (X.shape[1], X.shape[2])

        self.model = self._build_model(input_shape, n_classes)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
            ),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        self.history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.config.verbose,
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        probs = self.model.predict(X, verbose=0)
        if len(self.classes_) == 2:
            return (probs.flatten() > 0.5).astype(int)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        probs = self.model.predict(X, verbose=0)
        if len(self.classes_) == 2:
            probs = probs.flatten()
            return np.column_stack([1 - probs, probs])
        return probs

    def save(self, path: str):
        """Save Transformer model"""
        model_path = path.replace('.joblib', '_keras.h5')
        self.model.save(model_path)

        state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_,
            'model_path': model_path,
        }
        joblib.dump(state, path)
        print(f"[TransformerModel] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TransformerModel':
        """Load Transformer model"""
        try:
            from tensorflow import keras
        except ImportError:
            import keras

        state = joblib.load(path)
        instance = cls(config=state['config'])
        instance.is_fitted = state['is_fitted']
        instance.classes_ = state['classes_']
        instance.model = keras.models.load_model(state['model_path'], compile=False)
        return instance


# =============================================================================
# Meta-Learner Model (Ensemble Stacking)
# =============================================================================

@dataclass
class MetaLearnerConfig(ModelConfig):
    """Configuration for meta-learner"""
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32


class MetaLearnerModel(BaseModel):
    """
    Neural meta-learner for combining base model predictions.

    Consolidates:
    - train_ensemble_stacking.py
    - train_neural_meta_learner.py
    - train_simple_meta_learner.py

    Usage:
        # Train base models and get their predictions
        base_predictions = np.column_stack([
            model1.predict_proba(X)[:, 1],
            model2.predict_proba(X)[:, 1],
            model3.predict_proba(X)[:, 1],
        ])

        # Train meta-learner
        meta = MetaLearnerModel()
        meta.fit(base_predictions, y)

        # Predict
        final_probs = meta.predict_proba(base_predictions_test)
    """

    def __init__(
        self,
        config: Optional[MetaLearnerConfig] = None,
        **kwargs
    ):
        if config is None:
            config = MetaLearnerConfig(**kwargs)
        super().__init__(config)

    def _build_model(self, input_dim: int, n_classes: int):
        """Build meta-learner neural network"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers, Model
        except ImportError:
            from keras import layers, Model
            import keras

        inputs = layers.Input(shape=(input_dim,))
        x = inputs

        for units in self.config.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.config.dropout)(x)

        if n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(n_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'

        model = Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> 'MetaLearnerModel':
        """
        Train the meta-learner.

        Args:
            X: Base model predictions (n_samples, n_base_models)
            y: True labels
            X_val: Validation predictions
            y_val: Validation labels

        Returns:
            self
        """
        try:
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            from keras.callbacks import EarlyStopping

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        input_dim = X.shape[1]

        self.model = self._build_model(input_dim, n_classes)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
            ),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        probs = self.model.predict(X, verbose=0)
        if len(self.classes_) == 2:
            return (probs.flatten() > 0.5).astype(int)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        probs = self.model.predict(X, verbose=0)
        if len(self.classes_) == 2:
            probs = probs.flatten()
            return np.column_stack([1 - probs, probs])
        return probs

    def save(self, path: str):
        """Save meta-learner"""
        model_path = path.replace('.joblib', '_keras.h5')
        self.model.save(model_path)

        state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_,
            'model_path': model_path,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'MetaLearnerModel':
        """Load meta-learner"""
        try:
            from tensorflow import keras
        except ImportError:
            import keras

        state = joblib.load(path)
        instance = cls(config=state['config'])
        instance.is_fitted = state['is_fitted']
        instance.classes_ = state['classes_']
        instance.model = keras.models.load_model(state['model_path'], compile=False)
        return instance


# =============================================================================
# Factory function for easy model creation
# =============================================================================

def create_model(
    model_type: str,
    **kwargs
) -> BaseModel:
    """
    Factory function to create models.

    Args:
        model_type: One of 'xgboost', 'lightgbm', 'catboost', 'lstm',
                    'attention_lstm', 'cnn_lstm', 'transformer', 'meta_learner'
        **kwargs: Model-specific configuration

    Returns:
        Model instance

    Usage:
        model = create_model('xgboost', max_depth=8, learning_rate=0.1)
        model = create_model('lstm', layers=[128, 64], dropout=0.4)
        model = create_model('attention_lstm')
    """
    model_type = model_type.lower()

    if model_type in ['xgboost', 'lightgbm', 'catboost']:
        return TreeEnsembleModel(model_type=model_type, **kwargs)

    elif model_type == 'lstm':
        return LSTMModel(**kwargs)

    elif model_type == 'attention_lstm':
        return LSTMModel(use_attention=True, **kwargs)

    elif model_type == 'cnn_lstm':
        return LSTMModel(use_cnn=True, **kwargs)

    elif model_type == 'focal_lstm':
        return LSTMModel(loss_type='focal', **kwargs)

    elif model_type == 'transformer':
        return TransformerModel(**kwargs)

    elif model_type == 'meta_learner':
        return MetaLearnerModel(**kwargs)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Supported: xgboost, lightgbm, catboost, lstm, "
                        f"attention_lstm, cnn_lstm, focal_lstm, transformer, meta_learner")
