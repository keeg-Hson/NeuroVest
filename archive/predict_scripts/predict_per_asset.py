#!/usr/bin/env python3
"""
Per-Asset Prediction Generator

Generates predictions for individual assets using their dedicated models.
More accurate than multi-asset ensemble for asset-specific backtests.

Usage:
    python3 predict_per_asset.py --asset SPY
    python3 predict_per_asset.py --asset BTC/USDT
    python3 predict_per_asset.py --asset-group crypto
    python3 predict_per_asset.py --all

Output:
    logs/predictions/<asset>_predictions.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent / "framework"))
from framework.asset_manager import AssetManager

from config import LOGS_DIR, MODELS_DIR
from utils import add_features, finalize_features


class PerAssetPredictor:
    """Generates predictions for individual assets"""

    def __init__(self):
        self.manager = AssetManager()
        self.data_dir = Path("data_cache")
        self.models_dir = MODELS_DIR
        self.output_dir = LOGS_DIR / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_asset(self, ticker: str) -> bool:
        """Generate predictions for a single asset"""

        print(f"\n{'=' * 80}")
        print(f"PREDICTING: {ticker}")
        print(f"{'=' * 80}")

        # Load asset data
        if '/' in ticker:
            filename = ticker.replace('/', '_') + '_1d.csv'
        else:
            filename = f"{ticker}_1d.csv"

        data_path = self.data_dir / filename

        # Special handling for SPY: prefer data/SPY.csv over data_cache/SPY_1d.csv
        if ticker == 'SPY':
            main_spy_path = Path("data") / "SPY.csv"
            if main_spy_path.exists():
                data_path = main_spy_path
                print(f"   ℹ️  Using main SPY data from {data_path}")

        if not data_path.exists():
            print(f"   ⚠️  Data file not found: {data_path}")
            return False

        # Load models
        model_prefix = ticker.replace('/', '_').lower()
        models = {}

        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            model_path = self.models_dir / f"{model_prefix}_{model_type}.pkl"

            if model_path.exists():
                try:
                    models[model_type] = joblib.load(model_path)
                    print(f"   ✓ Loaded {model_type} model")
                except Exception as e:
                    print(f"   ✗ Failed to load {model_type}: {e}")
            else:
                print(f"   ⚠️  {model_type} model not found: {model_path}")

        if not models:
            print(f"   ❌ No models found for {ticker}")
            print(f"   💡 Run: python3 framework/train_unified.py --asset {ticker}")
            return False

        # Load feature list
        feature_file = self.models_dir / f"{model_prefix}_features.txt"
        if not feature_file.exists():
            # Try generic feature list
            feature_file = self.models_dir / "input_features_fwd.txt"

        if feature_file.exists():
            saved_feats = [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]
            print(f"   Features: {len(saved_feats)}")
        else:
            print(f"   ⚠️  Feature list not found, using all features")
            saved_feats = None

        # Load and prepare data
        print(f"\n📥 Loading {ticker} data...")
        raw = pd.read_csv(data_path, parse_dates=['Date'])
        raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        print(f"   Rows: {len(raw):,}")

        # Build features
        print("\n🔧 Building features...")
        df_feat, all_cols = add_features(raw)

        if saved_feats:
            # Align to saved features
            feature_cols = [c for c in saved_feats if c in df_feat.columns]
            missing = [c for c in saved_feats if c not in df_feat.columns]

            if missing:
                print(f"   ⚠️  Missing {len(missing)} features, filling with 0")
                for feat in missing:
                    df_feat[feat] = 0.0

            X = df_feat[saved_feats].copy()
        else:
            # Use all available features
            feature_cols = [c for c in df_feat.columns if c not in ['Date', 'Label', 'Forward_Return']]
            X = df_feat[feature_cols].copy()

        # Get dates and prices for output
        dates = raw["Date"].values

        # Handle crypto (no Adj Close) vs equity data
        price_cols = ["Open", "High", "Low", "Close"]
        if "Adj Close" in raw.columns:
            price_cols.append("Adj Close")
        if "Volume" in raw.columns:
            price_cols.append("Volume")

        prices = raw[price_cols].values

        # Replace inf/nan
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        print(f"\n🤖 Generating predictions...")
        print(f"   Models: {len(models)}")
        print(f"   Samples: {len(X):,}")

        # Generate predictions from each model
        predictions = {}
        probabilities = {}

        for name, model in models.items():
            try:
                pred = model.predict(X)

                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities[name] = proba
                else:
                    probabilities[name] = None

                predictions[name] = pred
                print(f"   ✓ {name}: {len(pred):,} predictions")

                # Print distribution
                unique, counts = np.unique(pred, return_counts=True)
                dist = dict(zip(unique, counts))
                print(f"      Distribution: {dist}")

            except Exception as e:
                print(f"   ✗ {name} failed: {e}")

        if not predictions:
            print("   ❌ No predictions generated")
            return False

        # Ensemble prediction (majority vote)
        pred_array = np.array(list(predictions.values()))

        if len(pred_array.shape) == 1:
            ensemble_pred = pred_array
        else:
            # Majority vote across models
            ensemble_pred = np.round(pred_array.mean(axis=0)).astype(int)

        print(f"\n📊 Ensemble predictions:")
        unique, counts = np.unique(ensemble_pred, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"   Distribution: {dist}")

        # Calculate confidence (based on agreement)
        if len(predictions) > 1:
            # Agreement score: how many models agree on the majority class
            agreement = np.zeros(len(ensemble_pred))
            for i in range(len(ensemble_pred)):
                votes = pred_array[:, i] if len(pred_array.shape) > 1 else pred_array
                agreement[i] = (votes == ensemble_pred[i]).sum() / len(predictions)
        else:
            agreement = np.ones(len(ensemble_pred))

        # Build output dataframe
        output_data = {
            'Date': dates[:len(ensemble_pred)],
            'Open': prices[:len(ensemble_pred), 0],
            'High': prices[:len(ensemble_pred), 1],
            'Low': prices[:len(ensemble_pred), 2],
            'Close': prices[:len(ensemble_pred), 3],
        }

        # Add optional columns if present
        col_idx = 4
        if "Adj Close" in raw.columns:
            output_data['Adj Close'] = prices[:len(ensemble_pred), col_idx]
            col_idx += 1
        if "Volume" in raw.columns:
            output_data['Volume'] = prices[:len(ensemble_pred), col_idx]

        output_data['Prediction'] = ensemble_pred
        output_data['Confidence'] = agreement

        output = pd.DataFrame(output_data)

        # Add model-specific predictions
        for name, pred in predictions.items():
            output[f'{name}_pred'] = pred[:len(ensemble_pred)]

        # Add probabilities if available
        # For 3-class: 0=CRASH, 1=NORMAL, 2=SPIKE
        if probabilities and probabilities[list(probabilities.keys())[0]] is not None:
            # Average probabilities across models
            all_probas = [p for p in probabilities.values() if p is not None]
            if all_probas:
                avg_proba = np.mean(all_probas, axis=0)

                if avg_proba.shape[1] >= 3:
                    output['Crash_Conf'] = avg_proba[:, 0]
                    output['Spike_Conf'] = avg_proba[:, 2]
                    # Confidence = max probability
                    output['Proba'] = avg_proba.max(axis=1)
                elif avg_proba.shape[1] == 2:
                    # Binary: 0=no-trade, 1=trade
                    output['Crash_Conf'] = 0.0
                    output['Spike_Conf'] = avg_proba[:, 1]
                    output['Proba'] = avg_proba[:, 1]
        else:
            # No probabilities, use agreement as confidence
            output['Proba'] = agreement
            output['Crash_Conf'] = (ensemble_pred == 0).astype(float) * agreement
            output['Spike_Conf'] = (ensemble_pred == 2).astype(float) * agreement

        # Ensure Crash_Conf and Spike_Conf exist
        if 'Crash_Conf' not in output.columns:
            output['Crash_Conf'] = 0.0
        if 'Spike_Conf' not in output.columns:
            output['Spike_Conf'] = 0.0

        # Save predictions
        output_file = self.output_dir / f"{ticker.replace('/', '_')}_predictions.csv"
        output.to_csv(output_file, index=False)
        print(f"\n💾 Saved predictions to: {output_file}")
        print(f"   Rows: {len(output):,}")

        return True

    def predict_group(self, asset_type: str = None, all_assets: bool = False):
        """Generate predictions for multiple assets"""

        if all_assets:
            assets = self.manager.get_all_assets()
            group_name = "ALL"
        elif asset_type:
            assets = [a for a in self.manager.get_all_assets() if a.asset_type == asset_type]
            group_name = asset_type.upper()
        else:
            print("❌ Must specify --asset-group or --all")
            return

        print(f"\n{'=' * 80}")
        print(f"BATCH PREDICTION: {group_name}")
        print(f"{'=' * 80}")
        print(f"Assets: {len(assets)}")
        print(f"{'=' * 80}\n")

        results = {
            'success': [],
            'failed': [],
            'no_data': [],
            'no_models': []
        }

        for i, asset in enumerate(assets, 1):
            print(f"\n[{i}/{len(assets)}] {asset.ticker}...")

            # Check if data exists
            filename = asset.ticker.replace('/', '_') + '_1d.csv'
            if not (self.data_dir / filename).exists():
                print(f"   ⚠️  No data file")
                results['no_data'].append(asset.ticker)
                continue

            # Try to predict
            try:
                success = self.predict_asset(asset.ticker)
                if success:
                    results['success'].append(asset.ticker)
                else:
                    results['no_models'].append(asset.ticker)
            except Exception as e:
                print(f"   ✗ Error: {e}")
                results['failed'].append(asset.ticker)

        # Summary
        print(f"\n{'=' * 80}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'=' * 80}")
        print(f"✓ Success:   {len(results['success'])}")
        print(f"⚠️ No data:   {len(results['no_data'])}")
        print(f"⚠️ No models: {len(results['no_models'])}")
        print(f"✗ Failed:    {len(results['failed'])}")
        print(f"{'=' * 80}\n")

        if results['success']:
            print("Successful predictions:")
            for ticker in results['success']:
                print(f"   ✓ {ticker}")

        if results['no_models']:
            print("\nAssets needing models (run train_unified.py --per-asset):")
            for ticker in results['no_models']:
                print(f"   ⚠️  {ticker}")


def main():
    parser = argparse.ArgumentParser(description="Per-asset prediction generator")
    parser.add_argument('--asset', help="Generate predictions for specific asset (e.g., SPY, BTC/USDT)")
    parser.add_argument('--asset-group', choices=['equity', 'crypto', 'bond', 'commodity'],
                        help="Generate predictions for all assets in group")
    parser.add_argument('--all', action='store_true',
                        help="Generate predictions for all configured assets")

    args = parser.parse_args()

    predictor = PerAssetPredictor()

    if args.asset:
        success = predictor.predict_asset(args.asset)
        sys.exit(0 if success else 1)
    elif args.asset_group or args.all:
        predictor.predict_group(asset_type=args.asset_group, all_assets=args.all)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 predict_per_asset.py --asset SPY")
        print("  python3 predict_per_asset.py --asset BTC/USDT")
        print("  python3 predict_per_asset.py --asset-group crypto")
        print("  python3 predict_per_asset.py --all")


if __name__ == "__main__":
    main()
