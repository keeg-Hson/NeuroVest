"""
Model Improvements Module for NeuroVest

Provides enhanced model training capabilities:
- Feature pruning with recursive feature elimination
- Ensemble improvements with calibration
- Walk-forward cross-validation
- Model averaging with uncertainty estimates

Usage:
    from core.model_improvements import (
        EnhancedEnsemble,
        WalkForwardValidator,
        CalibratedModel,
        prune_features_rfe,
    )
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Feature Pruning with Recursive Feature Elimination
# =============================================================================

@dataclass
class RFEConfig:
    """Configuration for Recursive Feature Elimination"""
    n_features_to_select: int = 50
    step: Union[int, float] = 0.1  # Features to remove at each step
    cv: int = 3  # Cross-validation folds
    n_jobs: int = -1
    random_state: int = 42


def prune_features_rfe(
    X: pd.DataFrame,
    y: np.ndarray,
    config: RFEConfig = None,
    estimator=None,
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Perform Recursive Feature Elimination to prune features.

    Args:
        X: Feature matrix
        y: Target labels
        config: RFE configuration
        estimator: Base estimator (defaults to LightGBM)

    Returns:
        Tuple of (X_selected, selected_feature_names, feature_ranking)
    """
    from sklearn.feature_selection import RFECV

    config = config or RFEConfig()

    if estimator is None:
        try:
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.random_state,
                verbose=-1,
                force_col_wise=True,
                n_jobs=config.n_jobs,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )

    print(f"[RFE] Starting with {X.shape[1]} features...")
    print(f"[RFE] Target: {config.n_features_to_select} features")

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

    feature_names = X.columns.tolist()

    # Run RFECV
    selector = RFECV(
        estimator=estimator,
        min_features_to_select=config.n_features_to_select,
        step=config.step,
        cv=config.cv,
        n_jobs=config.n_jobs,
        scoring='f1',
    )

    selector.fit(X, y)

    selected_mask = selector.support_
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    ranking = selector.ranking_

    print(f"[RFE] Selected {len(selected_features)} features")
    print(f"[RFE] Optimal CV score: {selector.cv_results_['mean_test_score'].max():.4f}")

    X_selected = X[selected_features]

    return X_selected, selected_features, ranking


# =============================================================================
# Calibrated Model Wrapper
# =============================================================================

class CalibratedModel:
    """
    Wrapper that adds probability calibration to any classifier.

    Calibration improves probability estimates using isotonic regression
    or Platt scaling, making predictions more reliable for threshold-based
    trading decisions.

    Usage:
        model = CalibratedModel(base_estimator=XGBClassifier())
        model.fit(X_train, y_train, X_val, y_val)
        calibrated_probs = model.predict_proba(X_test)
    """

    def __init__(
        self,
        base_estimator=None,
        method: str = 'isotonic',  # 'isotonic' or 'sigmoid'
        cv: int = 5,
    ):
        """
        Initialize calibrated model.

        Args:
            base_estimator: Underlying classifier (defaults to XGBoost)
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Cross-validation folds for calibration
        """
        from sklearn.calibration import CalibratedClassifierCV

        self.method = method
        self.cv = cv

        if base_estimator is None:
            from xgboost import XGBClassifier
            base_estimator = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
            )

        self.base_estimator = base_estimator
        self.calibrated_clf = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_cal: np.ndarray = None,
        y_cal: np.ndarray = None,
        **kwargs
    ) -> 'CalibratedModel':
        """
        Fit the calibrated model.

        If X_cal/y_cal are provided, uses prefit calibration.
        Otherwise uses cross-validation calibration.

        Args:
            X: Training features
            y: Training labels
            X_cal: Calibration features (optional)
            y_cal: Calibration labels (optional)

        Returns:
            self
        """
        from sklearn.calibration import CalibratedClassifierCV

        if X_cal is not None and y_cal is not None:
            # Prefit calibration: train base model, then calibrate on held-out data
            print("[CalibratedModel] Using prefit calibration...")
            self.base_estimator.fit(X, y, **kwargs)
            self.calibrated_clf = CalibratedClassifierCV(
                estimator=self.base_estimator,
                method=self.method,
                cv='prefit',
            )
            self.calibrated_clf.fit(X_cal, y_cal)
        else:
            # CV calibration
            print(f"[CalibratedModel] Using {self.cv}-fold CV calibration...")
            self.calibrated_clf = CalibratedClassifierCV(
                estimator=self.base_estimator,
                method=self.method,
                cv=self.cv,
            )
            self.calibrated_clf.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.calibrated_clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated probability estimates"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.calibrated_clf.predict_proba(X)

    def save(self, path: str):
        """Save model"""
        state = {
            'calibrated_clf': self.calibrated_clf,
            'base_estimator': self.base_estimator,
            'method': self.method,
            'cv': self.cv,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'CalibratedModel':
        """Load model"""
        state = joblib.load(path)
        instance = cls(
            base_estimator=state['base_estimator'],
            method=state['method'],
            cv=state['cv'],
        )
        instance.calibrated_clf = state['calibrated_clf']
        instance.is_fitted = state['is_fitted']
        return instance


# =============================================================================
# Enhanced Ensemble with Uncertainty
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for enhanced ensemble"""
    n_models: int = 5
    model_types: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm', 'catboost'])
    use_calibration: bool = True
    aggregation: str = 'mean'  # 'mean', 'median', 'weighted'
    random_state: int = 42


class EnhancedEnsemble:
    """
    Enhanced ensemble with multiple model types and uncertainty estimation.

    Combines predictions from XGBoost, LightGBM, and CatBoost with:
    - Optional probability calibration
    - Uncertainty estimates via prediction variance
    - Weighted averaging based on validation performance

    Usage:
        ensemble = EnhancedEnsemble()
        ensemble.fit(X_train, y_train, X_val, y_val)

        # Get predictions with uncertainty
        probs, uncertainty = ensemble.predict_proba_with_uncertainty(X_test)

        # High-confidence predictions only
        confident_mask = uncertainty < 0.1
    """

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models: List = []
        self.weights: np.ndarray = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> 'EnhancedEnsemble':
        """
        Fit the ensemble.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for weight calibration)
            y_val: Validation labels

        Returns:
            self
        """
        from sklearn.metrics import log_loss

        print(f"[EnhancedEnsemble] Training {len(self.config.model_types)} model types...")

        self.models = []

        for model_type in self.config.model_types:
            print(f"[EnhancedEnsemble] Training {model_type}...")

            if model_type == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            elif model_type == 'lightgbm':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            elif model_type == 'catboost':
                try:
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier(
                        iterations=200,
                        depth=6,
                        learning_rate=0.05,
                        random_state=self.config.random_state,
                        verbose=False,
                    )
                except ImportError:
                    print(f"[EnhancedEnsemble] CatBoost not available, skipping...")
                    continue
            else:
                print(f"[EnhancedEnsemble] Unknown model type: {model_type}")
                continue

            # Optionally wrap with calibration
            if self.config.use_calibration and X_val is not None:
                model = CalibratedModel(base_estimator=model, method='isotonic')
                model.fit(X, y, X_val, y_val)
            else:
                model.fit(X, y)

            self.models.append((model_type, model))

        # Calculate weights based on validation performance
        if X_val is not None and y_val is not None:
            print("[EnhancedEnsemble] Calculating model weights...")
            weights = []
            for model_type, model in self.models:
                probs = model.predict_proba(X_val)
                if probs.ndim > 1:
                    probs = probs[:, 1]
                ll = log_loss(y_val, probs)
                # Inverse log loss as weight (lower loss = higher weight)
                weights.append(1.0 / (ll + 0.01))

            self.weights = np.array(weights)
            self.weights /= self.weights.sum()

            for (model_type, _), w in zip(self.models, self.weights):
                print(f"    {model_type}: weight={w:.3f}")
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        self.is_fitted = True
        print(f"[EnhancedEnsemble] Fitted {len(self.models)} models")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        all_probs = []
        for model_type, model in self.models:
            probs = model.predict_proba(X)
            if probs.ndim > 1:
                probs = probs[:, 1]
            all_probs.append(probs)

        all_probs = np.array(all_probs)  # Shape: (n_models, n_samples)

        if self.config.aggregation == 'mean':
            ensemble_probs = np.average(all_probs, axis=0, weights=self.weights)
        elif self.config.aggregation == 'median':
            ensemble_probs = np.median(all_probs, axis=0)
        elif self.config.aggregation == 'weighted':
            ensemble_probs = np.average(all_probs, axis=0, weights=self.weights)
        else:
            ensemble_probs = np.mean(all_probs, axis=0)

        return np.column_stack([1 - ensemble_probs, ensemble_probs])

    def predict_proba_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty estimates.

        Args:
            X: Features

        Returns:
            Tuple of (probabilities, uncertainty)
            - probabilities: shape (n_samples,) positive class probability
            - uncertainty: shape (n_samples,) standard deviation across models
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        all_probs = []
        for model_type, model in self.models:
            probs = model.predict_proba(X)
            if probs.ndim > 1:
                probs = probs[:, 1]
            all_probs.append(probs)

        all_probs = np.array(all_probs)

        # Mean prediction
        mean_probs = np.average(all_probs, axis=0, weights=self.weights)

        # Uncertainty as weighted standard deviation
        variance = np.average((all_probs - mean_probs) ** 2, axis=0, weights=self.weights)
        uncertainty = np.sqrt(variance)

        return mean_probs, uncertainty

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    def save(self, path: str):
        """Save ensemble"""
        state = {
            'config': self.config,
            'models': self.models,
            'weights': self.weights,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'EnhancedEnsemble':
        """Load ensemble"""
        state = joblib.load(path)
        instance = cls(config=state['config'])
        instance.models = state['models']
        instance.weights = state['weights']
        instance.is_fitted = state['is_fitted']
        return instance


# =============================================================================
# Walk-Forward Cross-Validation
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    n_splits: int = 5
    train_size: float = 0.7  # Proportion of data for initial training
    test_size: float = 0.1  # Proportion of each test window
    gap: int = 0  # Gap between train and test to avoid lookahead bias
    expanding: bool = True  # If True, training window expands; if False, slides


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.

    Provides proper temporal splits that respect the time ordering of data,
    essential for financial time series to avoid lookahead bias.

    Usage:
        validator = WalkForwardValidator()
        results = validator.validate(model, X, y, dates)

        print(f"Mean OOS Accuracy: {results['mean_accuracy']:.3f}")
        print(f"Mean OOS F1: {results['mean_f1']:.3f}")
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.results_: List[Dict] = []

    def get_splits(
        self,
        X: np.ndarray,
        dates: pd.DatetimeIndex = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test split indices.

        Args:
            X: Feature matrix
            dates: Optional date index for reporting

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        splits = []

        initial_train_size = int(n_samples * self.config.train_size)
        test_window_size = int(n_samples * self.config.test_size)

        if test_window_size < 10:
            test_window_size = 10

        current_train_end = initial_train_size

        for i in range(self.config.n_splits):
            test_start = current_train_end + self.config.gap
            test_end = min(test_start + test_window_size, n_samples)

            if test_end > n_samples:
                break

            if self.config.expanding:
                # Expanding window: train from start to current point
                train_indices = np.arange(0, current_train_end)
            else:
                # Sliding window: fixed-size training window
                train_start = max(0, current_train_end - initial_train_size)
                train_indices = np.arange(train_start, current_train_end)

            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

            # Move forward
            current_train_end = test_end

        return splits

    def validate(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        dates: pd.DatetimeIndex = None,
        feature_names: List[str] = None,
    ) -> Dict:
        """
        Perform walk-forward validation.

        Args:
            model: Model with fit/predict_proba interface
            X: Feature matrix
            y: Labels
            dates: Optional date index
            feature_names: Optional feature names

        Returns:
            Dictionary with validation results
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.base import clone

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values

        splits = self.get_splits(X, dates)
        self.results_ = []

        print(f"[WalkForward] Running {len(splits)} walk-forward splits...")

        all_predictions = []
        all_actuals = []
        all_probabilities = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit model
            try:
                fold_model = clone(model)
            except Exception:
                # For models that don't support sklearn clone
                fold_model = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})

            fold_model.fit(X_train, y_train)

            # Predict
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }

            if dates is not None:
                fold_result['test_start'] = dates[test_idx[0]]
                fold_result['test_end'] = dates[test_idx[-1]]

            self.results_.append(fold_result)
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            all_probabilities.extend(y_proba)

            print(f"    Fold {fold+1}: Acc={accuracy:.3f}, F1={f1:.3f}, "
                  f"Prec={precision:.3f}, Rec={recall:.3f}")

        # Aggregate metrics
        summary = {
            'n_splits': len(splits),
            'mean_accuracy': np.mean([r['accuracy'] for r in self.results_]),
            'std_accuracy': np.std([r['accuracy'] for r in self.results_]),
            'mean_f1': np.mean([r['f1'] for r in self.results_]),
            'std_f1': np.std([r['f1'] for r in self.results_]),
            'mean_precision': np.mean([r['precision'] for r in self.results_]),
            'mean_recall': np.mean([r['recall'] for r in self.results_]),
            'total_samples': sum(r['test_size'] for r in self.results_),
            'fold_results': self.results_,
            'all_predictions': np.array(all_predictions),
            'all_actuals': np.array(all_actuals),
            'all_probabilities': np.array(all_probabilities),
        }

        print(f"\n[WalkForward] Summary:")
        print(f"    Mean Accuracy: {summary['mean_accuracy']:.3f} +/- {summary['std_accuracy']:.3f}")
        print(f"    Mean F1 Score: {summary['mean_f1']:.3f} +/- {summary['std_f1']:.3f}")
        print(f"    Total OOS samples: {summary['total_samples']}")

        return summary

    def get_report(self) -> str:
        """Generate a validation report"""
        if not self.results_:
            return "No validation results available. Run validate() first."

        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION REPORT",
            "=" * 60,
            "",
            f"Configuration:",
            f"  - Splits: {self.config.n_splits}",
            f"  - Initial train size: {self.config.train_size:.0%}",
            f"  - Test window size: {self.config.test_size:.0%}",
            f"  - Gap: {self.config.gap}",
            f"  - Expanding window: {self.config.expanding}",
            "",
            "Fold Results:",
            "-" * 60,
            f"{'Fold':>4} {'Train':>8} {'Test':>8} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}",
            "-" * 60,
        ]

        for r in self.results_:
            lines.append(
                f"{r['fold']:>4} {r['train_size']:>8} {r['test_size']:>8} "
                f"{r['accuracy']:>10.3f} {r['f1']:>8.3f} {r['precision']:>10.3f} {r['recall']:>8.3f}"
            )

        # Summary stats
        mean_acc = np.mean([r['accuracy'] for r in self.results_])
        std_acc = np.std([r['accuracy'] for r in self.results_])
        mean_f1 = np.mean([r['f1'] for r in self.results_])
        std_f1 = np.std([r['f1'] for r in self.results_])

        lines.extend([
            "-" * 60,
            f"Mean Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}",
            f"Mean F1 Score: {mean_f1:.3f} +/- {std_f1:.3f}",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# Convenience function for full improved training pipeline
# =============================================================================

def train_with_improvements(
    X: pd.DataFrame,
    y: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    use_feature_pruning: bool = True,
    use_ensemble: bool = True,
    use_calibration: bool = True,
    use_walk_forward: bool = True,
    save_path: str = None,
) -> Dict:
    """
    Full training pipeline with all improvements.

    Args:
        X: Training features
        y: Training labels
        X_val: Validation features
        y_val: Validation labels
        use_feature_pruning: Whether to prune features
        use_ensemble: Whether to use enhanced ensemble
        use_calibration: Whether to calibrate probabilities
        use_walk_forward: Whether to run walk-forward validation
        save_path: Path to save the trained model

    Returns:
        Dictionary with model and training results
    """
    results = {
        'original_features': X.shape[1],
        'training_samples': len(X),
    }

    # 1. Feature pruning
    if use_feature_pruning:
        print("\n[Pipeline] Step 1: Feature Pruning...")
        from core.feature_selection import FeatureSelector, FeatureSelectionConfig
        selector = FeatureSelector(FeatureSelectionConfig(
            min_features=30,
            max_features=60,
            use_shap=True,
        ))
        X = selector.fit_transform(X, y)
        if X_val is not None:
            X_val = selector.transform(X_val)
        results['selected_features'] = len(selector.selected_features_)
        results['feature_selector'] = selector
    else:
        print("\n[Pipeline] Step 1: Feature Pruning (skipped)")

    # 2. Model training
    print("\n[Pipeline] Step 2: Model Training...")
    if use_ensemble:
        model = EnhancedEnsemble(EnsembleConfig(
            model_types=['xgboost', 'lightgbm'],
            use_calibration=use_calibration,
        ))
    else:
        if use_calibration:
            model = CalibratedModel(method='isotonic')
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )

    model.fit(X.values if hasattr(X, 'values') else X, y,
              X_val.values if X_val is not None and hasattr(X_val, 'values') else X_val,
              y_val)

    results['model'] = model

    # 3. Walk-forward validation
    if use_walk_forward:
        print("\n[Pipeline] Step 3: Walk-Forward Validation...")
        validator = WalkForwardValidator()
        from xgboost import XGBClassifier
        base_model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
        wf_results = validator.validate(base_model, X, y)
        results['walk_forward'] = wf_results
        print(validator.get_report())
    else:
        print("\n[Pipeline] Step 3: Walk-Forward Validation (skipped)")

    # 4. Save model
    if save_path:
        print(f"\n[Pipeline] Saving model to {save_path}...")
        model.save(save_path)
        results['model_path'] = save_path

    print("\n[Pipeline] Training complete!")
    return results
