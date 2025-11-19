"""
Train crypto trading models

Uses same ML architecture as stocks but with crypto-specific parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.train_models import ModelTrainer
import pandas as pd


def main():
    """
    Train models on crypto data
    """
    print("=" * 70)
    print("CRYPTO MODEL TRAINING")
    print("=" * 70)

    # Load BTC data (primary crypto asset)
    print("\nLoading BTC/USDT data...")
    cache_dir = Path('../data_cache')
    btc_file = cache_dir / 'BTC_USDT_1d.csv'

    if not btc_file.exists():
        print(f"✗ BTC data not found at {btc_file}")
        print("  Run crypto/generate_synthetic_data.py first")
        return

    btc = pd.read_csv(btc_file)
    btc['Date'] = pd.to_datetime(btc['Date'])
    btc.set_index('Date', inplace=True)

    print(f"✓ Loaded {len(btc)} days of BTC data")
    print(f"  Date range: {btc.index.min()} to {btc.index.max()}")

    # Initialize trainer
    trainer = ModelTrainer(output_dir='../models')

    # Prepare data with crypto-specific parameters
    # Crypto: 7-day holding (vs 10-day stocks), 3% threshold (vs 2% stocks)
    X, y, df = trainer.prepare_data(
        btc,
        target_days=7,              # Crypto moves faster
        min_return_threshold=0.03   # Higher threshold for crypto volatility
    )

    # Train models
    models = trainer.train_all(X, y, test_size=0.2)

    # Save models with crypto prefix
    trainer.save_models(prefix='crypto')

    # Test ensemble prediction
    print("\n" + "=" * 70)
    print("TESTING ENSEMBLE PREDICTION")
    print("=" * 70)

    # Get predictions on validation set
    val_start = int(len(X) * 0.8)
    X_val = X[val_start:]
    y_val = y[val_start:]

    ensemble_pred, ensemble_proba = trainer.predict_ensemble(X_val, voting_threshold=0.5)

    accuracy = (ensemble_pred == y_val).mean()
    print(f"\nEnsemble validation accuracy: {accuracy:.3f}")

    # Analyze high-confidence predictions
    high_conf_mask = ensemble_proba > 0.6
    if high_conf_mask.sum() > 0:
        high_conf_acc = (ensemble_pred[high_conf_mask] == y_val[high_conf_mask]).mean()
        print(f"High-confidence predictions (>60%): {high_conf_mask.sum()} trades, "
              f"{high_conf_acc:.3f} accuracy")

    print("\n" + "=" * 70)
    print("✓ Crypto model training complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
