"""
Feature Selection Module for NeuroVest

Provides SHAP-based feature importance analysis and automated feature selection
to reduce overfitting from redundant features.

Key capabilities:
- Remove highly correlated features (>0.95 correlation)
- SHAP-based importance ranking
- Configurable feature count targets
- Caching of selected features for consistent train/predict usage
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    # TIGHTENED (Feb 2026): Lowered from 0.85 to 0.75 to reduce overfitting
    # More aggressive correlation pruning removes redundant features that
    # contribute to overfitting without adding predictive value
    correlation_threshold: float = 0.75
    min_features: int = 40  # Minimum features to keep
    max_features: int = 70  # Maximum features to keep
    shap_sample_size: int = 1000  # Samples to use for SHAP calculation
    use_shap: bool = True  # Whether to use SHAP (slower but more accurate)
    use_rfe: bool = True  # Whether to use RFE for additional pruning
    random_state: int = 42


class FeatureSelector:
    """
    Feature selection using correlation filtering and SHAP importance.

    Usage:
        selector = FeatureSelector()

        # Fit on training data
        X_selected = selector.fit_transform(X_train, y_train)

        # Transform test data
        X_test_selected = selector.transform(X_test)

        # Get selected feature names
        selected_features = selector.selected_features_
    """

    def __init__(self, config: FeatureSelectionConfig = None):
        self.config = config or FeatureSelectionConfig()
        self.selected_features_: List[str] = []
        self.dropped_correlated_: List[str] = []
        self.feature_importance_: Optional[pd.Series] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> 'FeatureSelector':
        """
        Fit the feature selector.

        Args:
            X: Feature matrix (DataFrame or array)
            y: Target labels
            feature_names: Optional feature names if X is array

        Returns:
            self
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)

        print(f"\n[FeatureSelector] Starting with {X.shape[1]} features")

        # Step 1: Remove highly correlated features
        X_decorrelated, dropped = self._remove_correlated_features(X)
        self.dropped_correlated_ = dropped
        print(f"[FeatureSelector] Removed {len(dropped)} highly correlated features")

        # Step 2: Calculate feature importance
        if self.config.use_shap:
            importance = self._calculate_shap_importance(X_decorrelated, y)
        else:
            importance = self._calculate_tree_importance(X_decorrelated, y)

        self.feature_importance_ = importance.sort_values(ascending=False)

        # Step 3: Select top features
        n_features = min(
            self.config.max_features,
            max(self.config.min_features, len(X_decorrelated.columns))
        )

        self.selected_features_ = self.feature_importance_.head(n_features).index.tolist()

        print(f"[FeatureSelector] Selected {len(self.selected_features_)} features")
        print(f"[FeatureSelector] Top 10 features by importance:")
        for i, (feat, imp) in enumerate(self.feature_importance_.head(10).items()):
            print(f"    {i+1:2d}. {feat}: {imp:.4f}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to selected subset.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with only selected features
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector not fitted. Call fit() first.")

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Select only the fitted features (handle missing columns gracefully)
        available = [f for f in self.selected_features_ if f in X.columns]
        missing = set(self.selected_features_) - set(available)

        if missing:
            print(f"[FeatureSelector] Warning: {len(missing)} features missing, filling with 0")
            for feat in missing:
                X[feat] = 0

        return X[self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _remove_correlated_features(
        self,
        X: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with correlation above threshold"""
        corr_matrix = X.corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation above threshold
        to_drop = []
        for column in upper.columns:
            correlated_features = upper.index[upper[column] > self.config.correlation_threshold].tolist()
            if correlated_features:
                # Keep the first one, drop the rest
                to_drop.extend(correlated_features)

        to_drop = list(set(to_drop))  # Remove duplicates

        return X.drop(columns=to_drop, errors='ignore'), to_drop

    def _calculate_shap_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.Series:
        """Calculate SHAP-based feature importance"""
        try:
            import shap
            from lightgbm import LGBMClassifier
        except ImportError:
            print("[FeatureSelector] SHAP or LightGBM not available, falling back to tree importance")
            return self._calculate_tree_importance(X, y)

        print("[FeatureSelector] Calculating SHAP importance...")

        # Train a quick LightGBM model
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state,
            verbose=-1,
            force_col_wise=True,
        )
        model.fit(X, y)

        # Calculate SHAP values on a sample
        sample_size = min(self.config.shap_sample_size, len(X))
        X_sample = X.sample(n=sample_size, random_state=self.config.random_state)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification (shap_values may be list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)

        return pd.Series(mean_shap, index=X.columns)

    def _calculate_tree_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.Series:
        """Calculate tree-based feature importance (faster alternative to SHAP)"""
        from sklearn.ensemble import ExtraTreesClassifier

        print("[FeatureSelector] Calculating tree-based importance...")

        model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)

        return pd.Series(model.feature_importances_, index=X.columns)

    def save(self, path: str):
        """Save selector state"""
        state = {
            'config': self.config,
            'selected_features_': self.selected_features_,
            'dropped_correlated_': self.dropped_correlated_,
            'feature_importance_': self.feature_importance_,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(state, path)
        print(f"[FeatureSelector] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureSelector':
        """Load selector state"""
        state = joblib.load(path)
        instance = cls(config=state['config'])
        instance.selected_features_ = state['selected_features_']
        instance.dropped_correlated_ = state['dropped_correlated_']
        instance.feature_importance_ = state['feature_importance_']
        instance.is_fitted = state['is_fitted']
        return instance

    def get_feature_report(self) -> str:
        """Generate a report of feature selection results"""
        if not self.is_fitted:
            return "FeatureSelector not fitted yet."

        lines = [
            "=" * 60,
            "FEATURE SELECTION REPORT",
            "=" * 60,
            f"Original features: {len(self.feature_importance_) + len(self.dropped_correlated_)}",
            f"Dropped (correlation > {self.config.correlation_threshold}): {len(self.dropped_correlated_)}",
            f"Selected features: {len(self.selected_features_)}",
            "",
            "Top 20 Features by Importance:",
            "-" * 40,
        ]

        for i, (feat, imp) in enumerate(self.feature_importance_.head(20).items()):
            marker = "✓" if feat in self.selected_features_ else " "
            lines.append(f"  {marker} {i+1:2d}. {feat:40s} {imp:.4f}")

        if self.dropped_correlated_:
            lines.extend([
                "",
                "Dropped Correlated Features:",
                "-" * 40,
            ])
            for feat in self.dropped_correlated_[:20]:
                lines.append(f"     - {feat}")
            if len(self.dropped_correlated_) > 20:
                lines.append(f"     ... and {len(self.dropped_correlated_) - 20} more")

        return "\n".join(lines)


def select_features_for_training(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    config: FeatureSelectionConfig = None,
    save_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureSelector]:
    """
    Convenience function for feature selection in training.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features to transform
        config: Feature selection configuration
        save_path: Optional path to save the selector

    Returns:
        Tuple of (X_train_selected, X_test_selected, selector)
    """
    selector = FeatureSelector(config)
    X_train_selected = selector.fit_transform(X_train, y_train)

    X_test_selected = None
    if X_test is not None:
        X_test_selected = selector.transform(X_test)

    if save_path:
        selector.save(save_path)

    print(selector.get_feature_report())

    return X_train_selected, X_test_selected, selector
