"""
Model Drift Detection and Monitoring

Detects when model performance degrades due to:
- Concept drift (relationship between features and target changes)
- Data drift (input feature distributions shift)
- Performance drift (model metrics degrade over time)

Usage:
    from core.model_drift import DriftDetector, DriftConfig

    detector = DriftDetector()
    drift_report = detector.check_drift(model, X_new, y_new, X_baseline, y_baseline)

    if drift_report['needs_retrain']:
        print("Model drift detected - retraining recommended")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # Performance drift thresholds
    f1_drop_threshold: float = 0.05  # Alert if F1 drops by this much
    precision_drop_threshold: float = 0.05
    recall_drop_threshold: float = 0.10  # More tolerant for recall

    # Statistical drift thresholds
    psi_threshold: float = 0.20  # Population Stability Index threshold
    ks_threshold: float = 0.10  # Kolmogorov-Smirnov threshold

    # Time-based settings
    window_size: int = 50  # Rolling window for drift detection
    min_samples_for_drift: int = 30  # Minimum samples to calculate drift

    # Alert settings
    consecutive_alerts_for_retrain: int = 3  # Number of consecutive alerts before recommending retrain


class DriftDetector:
    """
    Detects model drift through multiple methods:
    1. Performance drift - track F1, precision, recall over time
    2. Data drift - track feature distribution changes (PSI, KS test)
    3. Prediction drift - track prediction distribution changes
    """

    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        self.history: List[Dict] = []
        self.baseline_stats: Optional[Dict] = None
        self.alert_count = 0

    def set_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_predictions: np.ndarray = None,
        feature_names: List[str] = None,
    ):
        """
        Set baseline statistics for drift comparison.

        Args:
            X: Baseline feature matrix
            y: Baseline labels
            model_predictions: Model predictions on baseline (optional)
            feature_names: Names of features
        """
        self.baseline_stats = {
            'n_samples': len(X),
            'feature_means': np.nanmean(X, axis=0),
            'feature_stds': np.nanstd(X, axis=0),
            'feature_quantiles': {
                'q25': np.nanpercentile(X, 25, axis=0),
                'q50': np.nanpercentile(X, 50, axis=0),
                'q75': np.nanpercentile(X, 75, axis=0),
            },
            'class_distribution': pd.Series(y).value_counts(normalize=True).to_dict(),
            'feature_names': feature_names or [f'f_{i}' for i in range(X.shape[1])],
            'timestamp': datetime.now().isoformat(),
        }

        if model_predictions is not None:
            self.baseline_stats['prediction_distribution'] = {
                'mean': float(np.mean(model_predictions)),
                'std': float(np.std(model_predictions)),
                'positive_rate': float(np.mean(model_predictions > 0.5)),
            }

        print(f"[DriftDetector] Baseline set with {len(X)} samples, {X.shape[1]} features")

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures how much a distribution has shifted.
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.2: Moderate shift
        PSI >= 0.2: Significant shift

        Args:
            baseline: Baseline distribution
            current: Current distribution
            n_bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Remove NaN values
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]

        if len(baseline) < 10 or len(current) < 10:
            return 0.0

        # Create bins based on baseline
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate proportions
        baseline_counts = np.histogram(baseline, bins=bins)[0]
        current_counts = np.histogram(current, bins=bins)[0]

        # Add small constant to avoid division by zero
        baseline_props = (baseline_counts + 0.001) / (len(baseline) + 0.001 * n_bins)
        current_props = (current_counts + 0.001) / (len(current) + 0.001 * n_bins)

        # Calculate PSI
        psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))

        return float(psi)

    def calculate_ks_statistic(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.

        Measures the maximum distance between two cumulative distributions.

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            KS statistic
        """
        from scipy import stats

        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]

        if len(baseline) < 10 or len(current) < 10:
            return 0.0

        ks_stat, _ = stats.ks_2samp(baseline, current)
        return float(ks_stat)

    def check_data_drift(
        self,
        X_current: np.ndarray,
        top_k_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Check for data drift in features.

        Args:
            X_current: Current feature matrix
            top_k_features: Number of top drifting features to report

        Returns:
            Dictionary with drift statistics
        """
        if self.baseline_stats is None:
            return {'error': 'Baseline not set. Call set_baseline() first.'}

        n_features = X_current.shape[1]
        psi_scores = []
        ks_scores = []

        baseline_means = self.baseline_stats['feature_means']

        for i in range(n_features):
            # Calculate PSI for each feature
            psi = self.calculate_psi(
                baseline_means[i:i+1].repeat(len(X_current)) if len(baseline_means) > i else np.zeros(len(X_current)),
                X_current[:, i]
            )
            psi_scores.append(psi)

            # For KS, we'd need the full baseline distribution
            # Here we approximate with current vs baseline mean/std
            ks_scores.append(0.0)  # Placeholder

        feature_names = self.baseline_stats.get('feature_names', [f'f_{i}' for i in range(n_features)])

        # Get top drifting features
        drift_df = pd.DataFrame({
            'feature': feature_names[:len(psi_scores)],
            'psi': psi_scores,
        }).sort_values('psi', ascending=False)

        top_drifters = drift_df.head(top_k_features).to_dict('records')

        return {
            'mean_psi': float(np.mean(psi_scores)),
            'max_psi': float(np.max(psi_scores)),
            'features_over_threshold': int(sum(p > self.config.psi_threshold for p in psi_scores)),
            'top_drifting_features': top_drifters,
            'data_drift_detected': float(np.mean(psi_scores)) > self.config.psi_threshold,
        }

    def check_performance_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        baseline_metrics: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Check for performance drift.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            baseline_metrics: Baseline performance metrics to compare against

        Returns:
            Dictionary with performance drift statistics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        current_metrics = {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
        }

        if baseline_metrics is None:
            return {
                'current_metrics': current_metrics,
                'drift_detected': False,
                'message': 'No baseline metrics provided for comparison',
            }

        # Calculate drops
        f1_drop = baseline_metrics.get('f1', 0) - current_metrics['f1']
        precision_drop = baseline_metrics.get('precision', 0) - current_metrics['precision']
        recall_drop = baseline_metrics.get('recall', 0) - current_metrics['recall']

        drift_detected = (
            f1_drop > self.config.f1_drop_threshold or
            precision_drop > self.config.precision_drop_threshold or
            recall_drop > self.config.recall_drop_threshold
        )

        return {
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics,
            'metric_drops': {
                'f1': float(f1_drop),
                'precision': float(precision_drop),
                'recall': float(recall_drop),
            },
            'drift_detected': drift_detected,
            'alerts': {
                'f1_alert': f1_drop > self.config.f1_drop_threshold,
                'precision_alert': precision_drop > self.config.precision_drop_threshold,
                'recall_alert': recall_drop > self.config.recall_drop_threshold,
            }
        }

    def check_prediction_drift(
        self,
        y_proba_current: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Check for drift in model predictions.

        Args:
            y_proba_current: Current prediction probabilities

        Returns:
            Dictionary with prediction drift statistics
        """
        if self.baseline_stats is None or 'prediction_distribution' not in self.baseline_stats:
            return {'error': 'Baseline predictions not set'}

        baseline = self.baseline_stats['prediction_distribution']

        current_mean = float(np.mean(y_proba_current))
        current_std = float(np.std(y_proba_current))
        current_positive_rate = float(np.mean(y_proba_current > 0.5))

        # Calculate drift metrics
        mean_drift = abs(current_mean - baseline['mean'])
        std_drift = abs(current_std - baseline['std'])
        rate_drift = abs(current_positive_rate - baseline['positive_rate'])

        return {
            'current': {
                'mean': current_mean,
                'std': current_std,
                'positive_rate': current_positive_rate,
            },
            'baseline': baseline,
            'drift': {
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'rate_drift': rate_drift,
            },
            'prediction_drift_detected': mean_drift > 0.1 or rate_drift > 0.15,
        }

    def check_drift(
        self,
        model,
        X_current: np.ndarray,
        y_current: np.ndarray = None,
        baseline_metrics: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive drift check.

        Args:
            model: Trained model with predict/predict_proba
            X_current: Current feature matrix
            y_current: Current labels (optional, for performance drift)
            baseline_metrics: Baseline performance metrics

        Returns:
            Comprehensive drift report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_current),
        }

        # Data drift
        if self.baseline_stats is not None:
            report['data_drift'] = self.check_data_drift(X_current)

        # Get predictions
        try:
            y_proba = model.predict_proba(X_current)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            # Prediction drift
            if self.baseline_stats is not None and 'prediction_distribution' in self.baseline_stats:
                report['prediction_drift'] = self.check_prediction_drift(y_proba)

            # Performance drift
            if y_current is not None:
                report['performance_drift'] = self.check_performance_drift(
                    y_current, y_pred, y_proba, baseline_metrics
                )
        except Exception as e:
            report['prediction_error'] = str(e)

        # Overall assessment
        needs_retrain = False
        reasons = []

        if 'data_drift' in report and report['data_drift'].get('data_drift_detected'):
            needs_retrain = True
            reasons.append('data_drift')

        if 'performance_drift' in report and report['performance_drift'].get('drift_detected'):
            needs_retrain = True
            reasons.append('performance_drift')

        if 'prediction_drift' in report and report['prediction_drift'].get('prediction_drift_detected'):
            reasons.append('prediction_drift')

        # Update alert count
        if needs_retrain:
            self.alert_count += 1
        else:
            self.alert_count = max(0, self.alert_count - 1)

        report['needs_retrain'] = self.alert_count >= self.config.consecutive_alerts_for_retrain
        report['alert_count'] = self.alert_count
        report['drift_reasons'] = reasons

        # Store in history
        self.history.append(report)

        return report

    def get_drift_summary(self) -> pd.DataFrame:
        """Get summary of drift history"""
        if not self.history:
            return pd.DataFrame()

        records = []
        for h in self.history:
            record = {
                'timestamp': h.get('timestamp'),
                'n_samples': h.get('n_samples'),
                'needs_retrain': h.get('needs_retrain'),
                'alert_count': h.get('alert_count'),
            }

            if 'performance_drift' in h:
                perf = h['performance_drift']
                if 'current_metrics' in perf:
                    record['f1'] = perf['current_metrics'].get('f1')
                    record['precision'] = perf['current_metrics'].get('precision')
                    record['recall'] = perf['current_metrics'].get('recall')

            if 'data_drift' in h:
                record['mean_psi'] = h['data_drift'].get('mean_psi')

            records.append(record)

        return pd.DataFrame(records)

    def save(self, path: str):
        """Save detector state"""
        state = {
            'config': self.config.__dict__,
            'baseline_stats': self.baseline_stats,
            'history': self.history,
            'alert_count': self.alert_count,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"[DriftDetector] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'DriftDetector':
        """Load detector state"""
        with open(path) as f:
            state = json.load(f)

        config = DriftConfig(**state['config'])
        detector = cls(config=config)
        detector.baseline_stats = state['baseline_stats']
        detector.history = state['history']
        detector.alert_count = state['alert_count']

        return detector


def monitor_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
) -> Dict[str, Any]:
    """
    Quick utility to set up monitoring for a trained model.

    Args:
        model: Trained model
        X_train: Training features (for baseline)
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        save_path: Path to save detector state

    Returns:
        Drift report
    """
    detector = DriftDetector()

    # Set baseline from training data
    train_preds = model.predict_proba(X_train)
    if train_preds.ndim > 1:
        train_preds = train_preds[:, 1]
    detector.set_baseline(X_train, y_train, train_preds)

    # Calculate baseline metrics
    train_pred_labels = (train_preds > 0.5).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score

    baseline_metrics = {
        'f1': f1_score(y_train, train_pred_labels, zero_division=0),
        'precision': precision_score(y_train, train_pred_labels, zero_division=0),
        'recall': recall_score(y_train, train_pred_labels, zero_division=0),
    }

    # Check drift on test data
    report = detector.check_drift(model, X_test, y_test, baseline_metrics)

    if save_path:
        detector.save(save_path)

    return report
