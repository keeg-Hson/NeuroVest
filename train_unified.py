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
from core.models import (
    TreeEnsembleModel,
    LSTMModel,
    TransformerModel,
    MetaLearnerModel,
    create_model,
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

    # Output options
    parser.add_argument('--output-prefix', type=str, default=None, help='Model output prefix')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser.parse_args()


def train_tree_model(
    model_type: str,
    data: Dict,
    args: argparse.Namespace,
) -> TreeEnsembleModel:
    """Train a tree-based model (XGBoost, LightGBM, or CatBoost)"""
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*70}")

    model = create_model(
        model_type,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        verbose=1 if args.verbose else 0,
    )

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
) -> Dict[str, TreeEnsembleModel]:
    """Train full ensemble (XGBoost + LightGBM + CatBoost)"""
    print(f"\n{'='*70}")
    print(f"Training ENSEMBLE (XGBoost + LightGBM + CatBoost)")
    print(f"{'='*70}")

    models = {}
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        models[model_type] = train_tree_model(model_type, data, args)

    # Calculate ensemble metrics
    print(f"\n{'='*70}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    # Get ensemble predictions
    all_probs = []
    for model in models.values():
        probs = model.predict_proba(data['X_test'])[:, 1]
        all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_test = data['y_test']
    print(f"  Ensemble Accuracy:  {accuracy_score(y_test, ensemble_preds):.4f}")
    print(f"  Ensemble Precision: {precision_score(y_test, ensemble_preds, zero_division=0):.4f}")
    print(f"  Ensemble Recall:    {recall_score(y_test, ensemble_preds, zero_division=0):.4f}")
    print(f"  Ensemble F1:        {f1_score(y_test, ensemble_preds, zero_division=0):.4f}")

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

    # Train model(s)
    models = {}

    if args.model == 'ensemble':
        models = train_ensemble(data, args)
    elif args.model in ['xgboost', 'lightgbm', 'catboost']:
        models[args.model] = train_tree_model(args.model, data, args)
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
