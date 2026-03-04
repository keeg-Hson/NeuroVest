"""
NeuroVest Core Module

This package contains the core infrastructure for NeuroVest:
- Data management (data_manager_postgres.py - canonical)
- Model architectures (models/base_models.py)
- Training pipeline (train_models.py)
- Prediction engine (prediction_engine.py)
- Feature selection (feature_selection.py)
- Model improvements (model_improvements.py)

Usage:
    from core.models import TreeEnsembleModel, create_model
    from core.feature_selection import FeatureSelector
    from core.model_improvements import EnhancedEnsemble, WalkForwardValidator
"""

# Expose key classes at package level for convenience
from core.feature_selection import (
    FeatureSelector,
    FeatureSelectionConfig,
    select_features_for_training,
)

from core.model_improvements import (
    EnhancedEnsemble,
    EnsembleConfig,
    CalibratedModel,
    WalkForwardValidator,
    WalkForwardConfig,
    prune_features_rfe,
    RFEConfig,
    train_with_improvements,
)

__all__ = [
    # Feature selection
    'FeatureSelector',
    'FeatureSelectionConfig',
    'select_features_for_training',
    # Model improvements
    'EnhancedEnsemble',
    'EnsembleConfig',
    'CalibratedModel',
    'WalkForwardValidator',
    'WalkForwardConfig',
    'prune_features_rfe',
    'RFEConfig',
    'train_with_improvements',
]
