"""
NeuroVest Model Classes

Consolidated model architectures for training and prediction.

Available Models:
- TreeEnsembleModel: XGBoost, LightGBM, CatBoost ensemble
- LSTMModel: LSTM with configurable architecture
- TransformerModel: Transformer encoder for time series
- MetaLearnerModel: Neural meta-learner for ensemble combining

Usage:
    from core.models import TreeEnsembleModel, LSTMModel

    # Tree-based ensemble
    model = TreeEnsembleModel(model_type='xgboost')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # LSTM
    model = LSTMModel(layers=[128, 64], dropout=0.3)
    model.fit(X_train, y_train, epochs=100)
"""

from .base_models import (
    BaseModel,
    TreeEnsembleModel,
    LSTMModel,
    TransformerModel,
    MetaLearnerModel,
)

__all__ = [
    'BaseModel',
    'TreeEnsembleModel',
    'LSTMModel',
    'TransformerModel',
    'MetaLearnerModel',
]
