#!/usr/bin/env python3
"""
Unified Training Script for NeuroVest

This script consolidates functionality from 21+ training scripts into a single,
configuration-driven training pipeline.

Usage:
    # Train XGBoost on SPY
    python train_unified.py --model xgboost --asset SPY

    # Train LSTM with attention
    python train_unified.py --model attention_lstm --asset SPY

    # Train ensemble on multiple assets
    python train_unified.py --model ensemble --assets SPY QQQ IWM

    # Train with custom config
    python train_unified.py --model xgboost --config configs/train_config.yaml

Replaces:
    - train.py, train_improved.py
    - train_lstm_model.py, train_lstm_focal_loss.py, train_lstm_v2_focal.py
    - train_attention_lstm.py, train_cnn_lstm.py
    - train_transformer_model.py
    - train_lightgbm_catboost.py
    - train_multi_asset.py, train_per_asset.py
    - train_ensemble_stacking.py
    - And more...
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, LOGS_DIR, TRAIN_CFG
from core.data_pipeline import DataPipeline, PipelineConfig
from core.data_manager_postgres import DataManager
from core.models import (
    TreeEnsembleModel,
    LSTMModel,
    TransformerModel,
    MetaLearnerModel,
    create_model,
)
from core.feature_selection import (
    FeatureSelector,
    FeatureSelectionConfig,
    select_features_for_training,
)
from core.hyperparameter_tuning import (
    HyperparameterTuner,
    TuningConfig,
    tune_all_models,
    OPTUNA_AVAILABLE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified training script for NeuroVest models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_unified.py --model xgboost
    python train_unified.py --model lstm --epochs 100
    python train_unified.py --model attention_lstm

    # Multi-asset training
    python train_unified.py --model xgboost --assets SPY QQQ IWM --multi-asset

    # Training with 3-class labels
    python train_unified.py --model lightgbm --three-class

    # Full ensemble (trains XGBoost, LightGBM, CatBoost)
    python train_unified.py --model ensemble
        """
    )

    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='xgboost',
        choices=['xgboost', 'lightgbm', 'catboost', 'lstm', 'attention_lstm',
                 'cnn_lstm', 'focal_lstm', 'transformer', 'ensemble', 'meta_learner'],
        help='Model type to train'
    )

    # Data options
    parser.add_argument(
        '--assets', '-a',
        nargs='+',
        default=['SPY'],
        help='Asset(s) to train on'
    )
    parser.add_argument(
        '--multi-asset',
        action='store_true',
        help='Combine multiple assets into single training set'
    )
    parser.add_argument(
        '--three-class',
        action='store_true',
        help='Use 3-class labels (CRASH/NORMAL/SPIKE) instead of binary'
    )

    # Training parameters
    parser.add_argument('--horizon', type=int, default=1, help='Forward return horizon in days')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--threshold', type=float, default=0.005, help='Positive class threshold')

    # Tree model parameters
    parser.add_argument('--max-depth', type=int, default=6, help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--n-estimators', type=int, default=200, help='Number of estimators')

    # LSTM/Transformer parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--sequence-length', type=int, default=20, help='Sequence length for LSTM')

    # Feature selection options
    parser.add_argument('--feature-selection', action='store_true',
                        help='Enable SHAP-based feature selection to reduce overfitting')
    parser.add_argument('--max-features', type=int, default=60,
                        help='Maximum features to keep after selection')
    parser.add_argument('--correlation-threshold', type=float, default=0.75,
                        help='Remove features with correlation above this threshold (tightened from 0.95)')

    # Hyperparameter tuning options (Optuna)
    parser.add_argument('--tune', action='store_true',
                        help='Enable Bayesian hyperparameter tuning with Optuna')
    parser.add_argument('--tune-trials', type=int, default=100,
                        help='Number of Optuna trials for hyperparameter search')
    parser.add_argument('--tune-timeout', type=int, default=None,
                        help='Maximum time in seconds for tuning (None = no limit)')

    # Output options
    parser.add_argument('--output-prefix', type=str, default=None, help='Model output prefix')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser.parse_args()


def train_tree_model(
    model_type: str,
    data: Dict,
    args: argparse.Namespace,
    tuned_params: Dict = None,
) -> TreeEnsembleModel:
    """Train a tree-based model (XGBoost, LightGBM, or CatBoost)"""
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*70}")

    # Use tuned parameters if provided, otherwise use defaults from args
    if tuned_params:
        print(f"Using Optuna-tuned parameters")
        # Map Optuna params to model params
        model_params = {
            'max_depth': tuned_params.get('max_depth', tuned_params.get('depth', args.max_depth)),
            'learning_rate': tuned_params.get('learning_rate', args.learning_rate),
            'n_estimators': tuned_params.get('n_estimators', tuned_params.get('iterations', args.n_estimators)),
            'verbose': 1 if args.verbose else 0,
        }
        # Add model-specific params
        if model_type == 'xgboost':
            model_params.update({
                'min_child_weight': tuned_params.get('min_child_weight', 1),
                'subsample': tuned_params.get('subsample', 0.8),
                'colsample_bytree': tuned_params.get('colsample_bytree', 0.8),
                'gamma': tuned_params.get('gamma', 0),
                'reg_alpha': tuned_params.get('reg_alpha', 0),
                'reg_lambda': tuned_params.get('reg_lambda', 1),
            })
        elif model_type == 'lightgbm':
            model_params.update({
                'num_leaves': tuned_params.get('num_leaves', 31),
                'min_child_samples': tuned_params.get('min_child_samples', 20),
                'subsample': tuned_params.get('subsample', 0.8),
                'colsample_bytree': tuned_params.get('colsample_bytree', 0.8),
                'reg_alpha': tuned_params.get('reg_alpha', 0),
                'reg_lambda': tuned_params.get('reg_lambda', 0),
            })
        elif model_type == 'catboost':
            model_params['max_depth'] = tuned_params.get('depth', args.max_depth)
            model_params['n_estimators'] = tuned_params.get('iterations', args.n_estimators)
            model_params.update({
                'l2_leaf_reg': tuned_params.get('l2_leaf_reg', 3),
                'bagging_temperature': tuned_params.get('bagging_temperature', 1),
                'random_strength': tuned_params.get('random_strength', 1),
            })
    else:
        model_params = {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'verbose': 1 if args.verbose else 0,
        }

    model = create_model(model_type, **model_params)

    model.fit(
        data['X_train'],
        data['y_train'],
        X_val=data['X_test'],
        y_val=data['y_test'],
        sample_weight=data['sample_weights'],
        feature_names=data['feature_cols'],
    )

    # Evaluate
    evaluate_model(model, data, model_type)

    return model


def train_lstm_model(
    variant: str,
    data: Dict,
    args: argparse.Namespace,
) -> LSTMModel:
    """Train an LSTM model variant"""
    print(f"\n{'='*70}")
    print(f"Training {variant.upper()} Model")
    print(f"{'='*70}")

    # Determine LSTM configuration
    use_attention = 'attention' in variant
    use_cnn = 'cnn' in variant
    loss_type = 'focal' if 'focal' in variant else 'binary_crossentropy'

    model = LSTMModel(
        layers=[128, 64, 32],
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_attention=use_attention,
        use_cnn=use_cnn,
        loss_type=loss_type,
        verbose=1 if args.verbose else 0,
    )

    model.fit(
        data['X_train'],
        data['y_train'],
        X_val=data['X_test'],
        y_val=data['y_test'],
        class_weight=data['class_weights'],
    )

    # Evaluate
    evaluate_model(model, data, variant)

    return model


def train_transformer_model(
    data: Dict,
    args: argparse.Namespace,
) -> TransformerModel:
    """Train a Transformer model"""
    print(f"\n{'='*70}")
    print(f"Training TRANSFORMER Model")
    print(f"{'='*70}")

    model = TransformerModel(
        num_heads=4,
        num_blocks=3,
        ff_dim=128,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1 if args.verbose else 0,
    )

    model.fit(
        data['X_train'],
        data['y_train'],
        X_val=data['X_test'],
        y_val=data['y_test'],
        class_weight=data['class_weights'],
    )

    # Evaluate
    evaluate_model(model, data, 'transformer')

    return model


def train_ensemble(
    data: Dict,
    args: argparse.Namespace,
    tuned_params: Dict[str, Dict] = None,
) -> Dict[str, TreeEnsembleModel]:
    """Train full ensemble (XGBoost + LightGBM + CatBoost) with meta-learner stacking"""
    tuned_params = tuned_params or {}
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"\n{'='*70}")
    print(f"Training ENSEMBLE (XGBoost + LightGBM + CatBoost + Meta-Learner)")
    print(f"{'='*70}")

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Step 1: Generate out-of-fold predictions for meta-learner training
    print("\n[Step 1/4] Generating out-of-fold predictions...")
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model_types = ['xgboost', 'lightgbm', 'catboost']
    oof_predictions = {mt: np.zeros(len(X_train)) for mt in model_types}
    test_predictions = {mt: np.zeros(len(X_test)) for mt in model_types}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"  Fold {fold + 1}/{n_splits}...")
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        for model_type in model_types:
            # Train fold model
            fold_model = TreeEnsembleModel(
                model_type=model_type,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
            )
            fold_model.fit(X_fold_train, y_fold_train)

            # Store OOF predictions
            oof_predictions[model_type][val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

            # Accumulate test predictions (average across folds)
            test_predictions[model_type] += fold_model.predict_proba(X_test)[:, 1] / n_splits

    # Step 2: Train final base models on full training set
    print("\n[Step 2/4] Training final base models on full data...")
    models = {}
    for model_type in model_types:
        models[model_type] = train_tree_model(
            model_type, data, args,
            tuned_params=tuned_params.get(model_type)
        )

    # Step 3: Train meta-learner on OOF predictions
    print("\n[Step 3/4] Training meta-learner on out-of-fold predictions...")

    # Stack OOF predictions as meta-features
    meta_train = np.column_stack([oof_predictions[mt] for mt in model_types])
    meta_test = np.column_stack([test_predictions[mt] for mt in model_types])

    # Try to train meta-learner, but fall back to simple averaging if TensorFlow not available
    meta_learner = None
    try:
        meta_learner = MetaLearnerModel(
            hidden_layers=[64, 32],
            dropout=0.3,
            learning_rate=0.001,
            epochs=100,
            batch_size=32,
            verbose=0,
        )

        # Split meta_train for validation
        val_split = int(len(meta_train) * 0.8)
        meta_learner.fit(
            meta_train[:val_split],
            y_train[:val_split],
            X_val=meta_train[val_split:],
            y_val=y_train[val_split:],
        )
        models['meta_learner'] = meta_learner
        print("  Meta-learner trained successfully")
    except Exception as e:
        print(f"  ⚠️  Meta-learner skipped (TensorFlow not installed): {type(e).__name__}")
        print("  Using simple averaging instead - base models still work fine")
        meta_learner = None  # Reset to None so downstream checks work correctly

    # Step 4: Evaluate ensemble with meta-learner
    print(f"\n[Step 4/4] ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    # Simple averaging (baseline)
    simple_avg_probs = np.mean([test_predictions[mt] for mt in model_types], axis=0)
    simple_avg_preds = (simple_avg_probs > 0.5).astype(int)

    print("\n  Simple Averaging (baseline):")
    print(f"    Accuracy:  {accuracy_score(y_test, simple_avg_preds):.4f}")
    print(f"    Precision: {precision_score(y_test, simple_avg_preds, zero_division=0):.4f}")
    print(f"    Recall:    {recall_score(y_test, simple_avg_preds, zero_division=0):.4f}")
    print(f"    F1:        {f1_score(y_test, simple_avg_preds, zero_division=0):.4f}")

    # Meta-learner ensemble (only if available)
    if meta_learner is not None:
        meta_probs = meta_learner.predict_proba(meta_test)[:, 1]
        meta_preds = (meta_probs > 0.5).astype(int)

        print("\n  Meta-Learner Stacking:")
        print(f"    Accuracy:  {accuracy_score(y_test, meta_preds):.4f}")
        print(f"    Precision: {precision_score(y_test, meta_preds, zero_division=0):.4f}")
        print(f"    Recall:    {recall_score(y_test, meta_preds, zero_division=0):.4f}")
        print(f"    F1:        {f1_score(y_test, meta_preds, zero_division=0):.4f}")

        # Calculate improvement
        baseline_f1 = f1_score(y_test, simple_avg_preds, zero_division=0)
        meta_f1 = f1_score(y_test, meta_preds, zero_division=0)
        improvement = (meta_f1 - baseline_f1) * 100
        print(f"\n  Meta-Learner F1 improvement: {improvement:+.2f}%")
    else:
        print("\n  (Meta-learner not available - using simple averaging)")
        print("  Base models (XGBoost, LightGBM, CatBoost) trained successfully")

    return models


def evaluate_model(model, data: Dict, model_name: str):
    """Evaluate model and print metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )

    y_test = data['y_test']
    y_pred = model.predict(data['X_test'])
    y_proba = model.predict_proba(data['X_test'])

    print(f"\n{model_name.upper()} Results:")
    print("-" * 50)
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")


def save_models(
    models: Dict,
    prefix: str,
    data: Dict,
    args: argparse.Namespace,
):
    """Save trained models and metadata"""
    output_dir = Path(MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        # Determine file extension based on model type
        if isinstance(model, (LSTMModel, TransformerModel, MetaLearnerModel)):
            ext = '.joblib'  # Keras models need special handling
        else:
            ext = '.pkl'

        model_path = output_dir / f"{prefix}_{model_name}{ext}"
        model.save(str(model_path))

    # Save scaler and feature names
    if data.get('scaler'):
        scaler_path = output_dir / f"{prefix}_scaler.pkl"
        joblib.dump(data['scaler'], scaler_path)
        print(f"Saved scaler to {scaler_path}")

    if data.get('feature_cols'):
        features_path = output_dir / f"{prefix}_features.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(data['feature_cols']))
        print(f"Saved {len(data['feature_cols'])} features to {features_path}")

    # Save feature selector if used
    if data.get('feature_selector'):
        selector_path = output_dir / f"{prefix}_feature_selector.joblib"
        data['feature_selector'].save(str(selector_path))

    # Save training metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'assets': args.assets,
        'horizon': args.horizon,
        'threshold': args.threshold,
        'test_size': args.test_size,
        'train_samples': len(data['X_train']),
        'test_samples': len(data['X_test']),
        'n_features': len(data['feature_cols']),
        'models': list(models.keys()),
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Save model metadata to PostgreSQL database
    try:
        dm = DataManager()
        for model_name in models.keys():
            dm.save_model_metadata(
                model_name=f"{prefix}_{model_name}",
                model_type=model_name,
                feature_count=len(data['feature_cols']),
                training_samples=len(data['X_train']),
                assets_used=args.assets,
                metrics=data.get('tuned_params', {}).get(model_name),
                hyperparameters={
                    'max_depth': args.max_depth,
                    'learning_rate': args.learning_rate,
                    'n_estimators': args.n_estimators,
                }
            )
        dm.close()
        print(f"✅ Saved model metadata to PostgreSQL database")
    except Exception as e:
        print(f"⚠️  Failed to save to PostgreSQL (non-fatal): {e}")


def main():
    args = parse_args()

    print("=" * 70)
    print("NEUROVEST UNIFIED TRAINING")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Assets:   {args.assets}")
    print(f"Horizon:  {args.horizon} days")
    print(f"3-Class:  {args.three_class}")

    # Determine if we need sequences (for LSTM/Transformer)
    use_sequences = args.model in ['lstm', 'attention_lstm', 'cnn_lstm', 'focal_lstm', 'transformer']

    # Configure data pipeline
    pipeline_config = PipelineConfig(
        horizon=args.horizon,
        pos_threshold=args.threshold,
        test_size=args.test_size,
        use_sequences=use_sequences,
        sequence_length=args.sequence_length,
        use_3_class=args.three_class,
    )

    pipeline = DataPipeline(config=pipeline_config)

    # Prepare training data
    print("\nPreparing training data...")
    data = pipeline.prepare_training_data(
        assets=args.assets,
        multi_asset=args.multi_asset,
    )

    # Apply feature selection if enabled
    feature_selector = None
    if args.feature_selection:
        print(f"\n{'='*70}")
        print("FEATURE SELECTION (SHAP-based)")
        print(f"{'='*70}")

        # Convert to DataFrame for feature selection
        feature_names = data.get('feature_cols', [f"f_{i}" for i in range(data['X_train'].shape[1])])
        X_train_df = pd.DataFrame(data['X_train'], columns=feature_names)
        X_test_df = pd.DataFrame(data['X_test'], columns=feature_names)

        # Configure and run feature selection
        fs_config = FeatureSelectionConfig(
            correlation_threshold=args.correlation_threshold,
            max_features=args.max_features,
            min_features=40,
            shap_sample_size=500,
            use_shap=True,
        )

        X_train_selected, X_test_selected, feature_selector = select_features_for_training(
            X_train_df,
            data['y_train'],
            X_test_df,
            config=fs_config,
        )

        # Update data with selected features
        data['X_train'] = X_train_selected.values
        data['X_test'] = X_test_selected.values
        data['feature_cols'] = feature_selector.selected_features_
        data['feature_selector'] = feature_selector

        print(f"\nFeature reduction: {len(feature_names)} → {len(feature_selector.selected_features_)}")

    # Run hyperparameter tuning if enabled
    tuned_params = {}
    if args.tune and args.model in ['xgboost', 'lightgbm', 'catboost', 'ensemble']:
        if not OPTUNA_AVAILABLE:
            print("\nWarning: Optuna not installed. Skipping hyperparameter tuning.")
            print("Install with: pip install optuna")
        else:
            print(f"\n{'='*70}")
            print("HYPERPARAMETER TUNING (Optuna)")
            print(f"{'='*70}")

            tuning_config = TuningConfig(
                n_trials=args.tune_trials,
                timeout=args.tune_timeout,
                verbose=args.verbose,
            )
            tuner = HyperparameterTuner(tuning_config)

            if args.model == 'ensemble':
                # Tune all tree models
                for model_type in ['xgboost', 'lightgbm', 'catboost']:
                    best_params = tuner.tune(
                        data['X_train'],
                        data['y_train'],
                        model_type=model_type,
                        sample_weight=data.get('sample_weights'),
                    )
                    tuned_params[model_type] = best_params
            else:
                # Tune single model
                best_params = tuner.tune(
                    data['X_train'],
                    data['y_train'],
                    model_type=args.model,
                    sample_weight=data.get('sample_weights'),
                )
                tuned_params[args.model] = best_params

            # Save tuned parameters
            data['tuned_params'] = tuned_params

    # Train model(s)
    models = {}

    if args.model == 'ensemble':
        models = train_ensemble(data, args, tuned_params)
    elif args.model in ['xgboost', 'lightgbm', 'catboost']:
        models[args.model] = train_tree_model(
            args.model, data, args,
            tuned_params=tuned_params.get(args.model)
        )
    elif args.model in ['lstm', 'attention_lstm', 'cnn_lstm', 'focal_lstm']:
        models[args.model] = train_lstm_model(args.model, data, args)
    elif args.model == 'transformer':
        models['transformer'] = train_transformer_model(data, args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Save models
    prefix = args.output_prefix or f"{args.model}_{'_'.join(args.assets)}"
    print(f"\nSaving models with prefix: {prefix}")
    save_models(models, prefix, data, args)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
