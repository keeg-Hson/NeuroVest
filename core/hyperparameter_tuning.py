"""
Bayesian Hyperparameter Tuning with Optuna

Replaces GridSearchCV with more efficient Bayesian optimization.
Explores hyperparameter space more intelligently, finding better
parameters in fewer trials.

Usage:
    from core.hyperparameter_tuning import HyperparameterTuner

    tuner = HyperparameterTuner(n_trials=100)
    best_params = tuner.tune(X_train, y_train, model_type='xgboost')
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score, make_scorer
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Try to import optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[HyperparameterTuning] Optuna not installed. Run: pip install optuna")


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning"""
    n_trials: int = 100  # Number of optimization trials
    n_cv_splits: int = 5  # Cross-validation splits
    timeout: Optional[int] = None  # Max seconds for tuning (None = no limit)
    scoring: str = 'f1'  # Optimization metric
    direction: str = 'maximize'  # Optimization direction
    random_state: int = 42
    verbose: bool = True
    early_stopping_rounds: int = 30  # Stop if no improvement


class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization using Optuna.

    Supports XGBoost, LightGBM, and CatBoost with time-series CV.
    """

    def __init__(self, config: TuningConfig = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.config = config or TuningConfig()
        self.study: Optional[optuna.Study] = None
        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'xgboost',
        sample_weight: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Run Bayesian hyperparameter optimization.

        Args:
            X: Training features
            y: Training labels
            model_type: 'xgboost', 'lightgbm', or 'catboost'
            sample_weight: Optional sample weights

        Returns:
            Dict of best hyperparameters
        """
        print(f"\n{'='*60}")
        print(f"OPTUNA HYPERPARAMETER TUNING - {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Trials: {self.config.n_trials}")
        print(f"CV Splits: {self.config.n_cv_splits}")
        print(f"Metric: {self.config.scoring}")

        # Create objective function
        objective = self._create_objective(X, y, model_type, sample_weight)

        # Create Optuna study
        sampler = TPESampler(seed=self.config.random_state)

        # Suppress Optuna logs if not verbose
        if not self.config.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            study_name=f"{model_type}_tuning",
        )

        # Add early stopping callback
        early_stop_callback = EarlyStoppingCallback(
            self.config.early_stopping_rounds
        )

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[early_stop_callback],
            show_progress_bar=self.config.verbose,
        )

        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        print(f"\n{'='*60}")
        print("TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Best {self.config.scoring}: {self.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in self.best_params_.items():
            print(f"  {param}: {value}")

        return self.best_params_

    def _create_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        sample_weight: np.ndarray = None,
    ) -> Callable:
        """Create Optuna objective function for the specified model type"""

        tscv = TimeSeriesSplit(n_splits=self.config.n_cv_splits)

        if model_type == 'xgboost':
            return self._xgboost_objective(X, y, tscv, sample_weight)
        elif model_type == 'lightgbm':
            return self._lightgbm_objective(X, y, tscv, sample_weight)
        elif model_type == 'catboost':
            return self._catboost_objective(X, y, tscv, sample_weight)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _xgboost_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tscv: TimeSeriesSplit,
        sample_weight: np.ndarray = None,
    ) -> Callable:
        """Objective function for XGBoost"""

        def objective(trial: optuna.Trial) -> float:
            import xgboost as xgb

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.config.random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'verbosity': 0,
            }

            model = xgb.XGBClassifier(**params)

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                sw_train = sample_weight[train_idx] if sample_weight is not None else None

                model.fit(
                    X_train, y_train,
                    sample_weight=sw_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, zero_division=0)
                scores.append(score)

            return np.mean(scores)

        return objective

    def _lightgbm_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tscv: TimeSeriesSplit,
        sample_weight: np.ndarray = None,
    ) -> Callable:
        """Objective function for LightGBM"""

        def objective(trial: optuna.Trial) -> float:
            import lightgbm as lgb

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.config.random_state,
                'verbose': -1,
                'force_col_wise': True,
            }

            model = lgb.LGBMClassifier(**params)

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                sw_train = sample_weight[train_idx] if sample_weight is not None else None

                model.fit(
                    X_train, y_train,
                    sample_weight=sw_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, zero_division=0)
                scores.append(score)

            return np.mean(scores)

        return objective

    def _catboost_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tscv: TimeSeriesSplit,
        sample_weight: np.ndarray = None,
    ) -> Callable:
        """Objective function for CatBoost"""

        def objective(trial: optuna.Trial) -> float:
            from catboost import CatBoostClassifier

            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': self.config.random_state,
                'verbose': False,
                'allow_writing_files': False,
            }

            model = CatBoostClassifier(**params)

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                sw_train = sample_weight[train_idx] if sample_weight is not None else None

                model.fit(
                    X_train, y_train,
                    sample_weight=sw_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False,
                )

                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, zero_division=0)
                scores.append(score)

            return np.mean(scores)

        return objective

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        if self.study is None:
            return pd.DataFrame()

        trials_data = []
        for trial in self.study.trials:
            data = {
                'trial': trial.number,
                'value': trial.value,
                'state': trial.state.name,
            }
            data.update(trial.params)
            trials_data.append(data)

        return pd.DataFrame(trials_data)

    def save(self, path: str):
        """Save tuning results"""
        state = {
            'config': self.config,
            'best_params_': self.best_params_,
            'best_score_': self.best_score_,
            'history': self.get_optimization_history().to_dict(),
        }
        joblib.dump(state, path)
        print(f"[HyperparameterTuner] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'HyperparameterTuner':
        """Load tuning results"""
        state = joblib.load(path)
        instance = cls(config=state['config'])
        instance.best_params_ = state['best_params_']
        instance.best_score_ = state['best_score_']
        return instance


class EarlyStoppingCallback:
    """Callback for early stopping if no improvement"""

    def __init__(self, patience: int = 30):
        self.patience = patience
        self.best_value: Optional[float] = None
        self.no_improvement_count = 0

    def __call__(self, study: Any, trial: Any) -> None:
        if trial.value is None:
            return

        if self.best_value is None or trial.value > self.best_value:
            self.best_value = trial.value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            print(f"\n[EarlyStopping] No improvement for {self.patience} trials. Stopping.")
            study.stop()


def tune_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 100,
    sample_weight: np.ndarray = None,
    save_dir: str = None,
) -> Dict[str, Dict]:
    """
    Tune hyperparameters for all model types.

    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of trials per model
        sample_weight: Optional sample weights
        save_dir: Directory to save results

    Returns:
        Dict mapping model_type to best_params
    """
    config = TuningConfig(n_trials=n_trials)
    tuner = HyperparameterTuner(config)

    results = {}

    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        print(f"\n\nTuning {model_type}...")
        try:
            best_params = tuner.tune(X_train, y_train, model_type, sample_weight)
            results[model_type] = {
                'params': best_params,
                'score': tuner.best_score_,
            }

            if save_dir:
                save_path = Path(save_dir) / f"tuned_params_{model_type}.joblib"
                tuner.save(str(save_path))
        except Exception as e:
            print(f"Error tuning {model_type}: {e}")
            results[model_type] = {'params': {}, 'score': None, 'error': str(e)}

    # Print summary
    print(f"\n{'='*60}")
    print("TUNING SUMMARY")
    print(f"{'='*60}")
    for model_type, result in results.items():
        score = result.get('score', 'N/A')
        if isinstance(score, float):
            score = f"{score:.4f}"
        print(f"{model_type:12s}: F1 = {score}")

    return results
