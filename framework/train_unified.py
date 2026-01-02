#!/usr/bin/env python3
"""
Unified Training Framework

Trains both per-asset and macro models automatically based on config/assets.yaml

Features:
- Per-asset models: Separate model for each enabled asset
- Macro models: Combined models for groups (all_equities, all_crypto, etc.)
- Saves results in organized structure
- Tracks performance metrics

Usage:
    python framework/train_unified.py --all            # Train everything
    python framework/train_unified.py --per-asset      # Only per-asset models
    python framework/train_unified.py --macro          # Only macro models
    python framework/train_unified.py --asset SPY      # Train single asset
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.asset_manager import AssetManager
from utils import add_features, add_forward_returns_and_labels

# ML imports
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class UnifiedTrainer:
    """Trains all configured models"""

    def __init__(self):
        self.manager = AssetManager()
        self.settings = self.manager.get_settings()

        self.data_dir = Path("data_cache")
        self.results_dir = Path(self.settings.get('results_dir', 'results'))
        self.models_dir = Path(self.settings.get('models_dir', 'models'))

        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        self.per_asset_results = []
        self.macro_results = []

    def train_all(self):
        """Train both per-asset and macro models"""
        print("=" * 80)
        print("UNIFIED TRAINING - ALL MODELS")
        print("=" * 80)

        self.train_per_asset_models()
        self.train_macro_models()

        self._save_combined_results()

    def train_per_asset_models(self):
        """Train separate model for each asset"""
        print("\n" + "=" * 80)
        print("PER-ASSET MODEL TRAINING")
        print("=" * 80)

        assets = self.manager.get_all_assets()
        print(f"\nTraining {len(assets)} assets...")

        for i, asset in enumerate(assets, 1):
            print(f"\n[{i}/{len(assets)}] Training: {asset.ticker} ({asset.name})")
            print("-" * 80)

            try:
                results = self._train_single_asset(asset)
                if results:
                    self.per_asset_results.append(results)
                    print(f"   ✓ {asset.ticker}: Ensemble {results['ensemble_acc']:.1%}")
            except Exception as e:
                print(f"   ✗ Error: {e}")

        # Save per-asset results
        if self.per_asset_results:
            df = pd.DataFrame(self.per_asset_results)
            output_path = self.results_dir / f"per_asset_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_path, index=False)
            print(f"\n✓ Per-asset results saved: {output_path}")

    def train_macro_models(self):
        """Train macro models for asset groups"""
        print("\n" + "=" * 80)
        print("MACRO MODEL TRAINING")
        print("=" * 80)

        macro_groups = self.manager.get_macro_groups()
        print(f"\nTraining {len(macro_groups)} macro groups...")

        for group_name, assets in macro_groups.items():
            print(f"\n[*] Training: {group_name}")
            print(f"    Assets: {len(assets)}")
            print("-" * 80)

            try:
                results = self._train_macro_group(group_name, assets)
                if results:
                    self.macro_results.append(results)
                    print(f"   ✓ {group_name}: Ensemble {results['ensemble_acc']:.1%}")
            except Exception as e:
                print(f"   ✗ Error: {e}")

        # Save macro results
        if self.macro_results:
            df = pd.DataFrame(self.macro_results)
            output_path = self.results_dir / f"macro_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_path, index=False)
            print(f"\n✓ Macro results saved: {output_path}")

    def _train_single_asset(self, asset) -> dict:
        """Train models for a single asset"""
        # Load data
        if asset.asset_type == 'crypto':
            filename = asset.ticker.replace('/', '_') + '_1d.csv'
        else:
            filename = f"{asset.ticker}_1d.csv"

        filepath = self.data_dir / filename

        # Special handling for SPY: prefer data/SPY.csv over data_cache/SPY_1d.csv
        if asset.ticker == 'SPY':
            main_spy_path = Path("data") / "SPY.csv"
            if main_spy_path.exists():
                filepath = main_spy_path
                print(f"   ℹ️  Using main SPY data from {filepath}")

        if not filepath.exists():
            print(f"   ⚠️  Data file not found: {filepath}, skipping")
            return None

        # Load and prepare data
        df = pd.read_csv(filepath, parse_dates=['Date'])

        # Add features (returns tuple: df, feature_cols)
        df, feature_cols = add_features(df)

        # Calculate threshold (auto or from config)
        if self.settings.get('auto_threshold', False):
            # Calculate volatility-based threshold
            returns = df['Close'].pct_change().dropna()
            daily_vol = returns.std()
            vol_multiplier = self.settings.get('vol_multiplier', 0.5)
            threshold = daily_vol * vol_multiplier
            print(f"   ℹ️  Auto-threshold: {threshold:.4f} ({threshold*100:.2f}%) based on {daily_vol*100:.2f}% daily vol")
        else:
            threshold = asset.threshold

        # Add labels
        df = add_forward_returns_and_labels(
            df,
            price_col="Close",
            horizon=self.settings.get('horizon', 1),
            pos_threshold=threshold,
            fee_bps=self.settings.get('fee_bps', 2.0),
            slippage_bps=self.settings.get('slippage_bps', 3.0),
        )

        # Rename 'y' to 'Label' for consistency with training code
        if 'y' in df.columns:
            df = df.rename(columns={'y': 'Label'})

        # Only drop rows where Label is NaN (XGBoost/LightGBM/CatBoost handle NaN features)
        if 'Label' in df.columns:
            df = df.dropna(subset=['Label'])

        if len(df) < 100:
            print(f"   ⚠️  Insufficient data ({len(df)} rows), skipping")
            return None

        print(f"   Dataset: {len(df):,} rows, {len(feature_cols)} features")

        # Train/test split
        split_idx = int(len(df) * self.settings.get('train_test_split', 0.8))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Train models - pass the safe feature_cols from add_features, not all columns
        models, metrics = self._train_models(train_df, test_df, asset.ticker, feature_cols)

        # Save models
        model_prefix = asset.ticker.replace('/', '_').lower()
        for model_name, model in models.items():
            model_path = self.models_dir / f"{model_prefix}_{model_name}.pkl"
            joblib.dump(model, model_path)

        # Return results
        return {
            'asset': asset.ticker,
            'name': asset.name,
            'type': asset.asset_type,
            'category': asset.category,
            'samples': len(df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': len(feature_cols),
            'xgb_acc': metrics['xgboost']['accuracy'],
            'lgb_acc': metrics['lightgbm']['accuracy'],
            'cat_acc': metrics['catboost']['accuracy'],
            'ensemble_acc': metrics['ensemble']['accuracy'],
            'baseline_acc': metrics['baseline']['accuracy'],
            'improvement': metrics['improvement'],
            'threshold': asset.threshold,
            'timestamp': datetime.now().isoformat()
        }

    def _train_macro_group(self, group_name: str, assets: list) -> dict:
        """Train macro model for asset group"""
        # Load and combine data from all assets
        combined_dfs = []
        all_feature_cols = []  # Collect safe feature columns from each asset

        for asset in assets:
            if asset.asset_type == 'crypto':
                filename = asset.ticker.replace('/', '_') + '_1d.csv'
            else:
                filename = f"{asset.ticker}_1d.csv"

            filepath = self.data_dir / filename

            # Special handling for SPY: prefer data/SPY.csv over data_cache/SPY_1d.csv
            if asset.ticker == 'SPY':
                main_spy_path = Path("data") / "SPY.csv"
                if main_spy_path.exists():
                    filepath = main_spy_path

            if not filepath.exists():
                print(f"   ⚠️  {asset.ticker}: File not found at {filepath}, skipping")
                continue

            df = pd.read_csv(filepath, parse_dates=['Date'])
            df, asset_feature_cols = add_features(df)  # Keep feature_cols for safety
            all_feature_cols.append(set(asset_feature_cols))  # Collect for intersection

            # Calculate threshold (auto or from config)
            if self.settings.get('auto_threshold', False):
                returns = df['Close'].pct_change().dropna()
                daily_vol = returns.std()
                vol_multiplier = self.settings.get('vol_multiplier', 0.5)
                threshold = daily_vol * vol_multiplier
            else:
                threshold = asset.threshold

            df = add_forward_returns_and_labels(
                df,
                price_col="Close",
                horizon=self.settings.get('horizon', 1),
                pos_threshold=threshold,
                fee_bps=self.settings.get('fee_bps', 2.0),
                slippage_bps=self.settings.get('slippage_bps', 3.0),
            )

            # Rename 'y' to 'Label' for consistency with training code
            if 'y' in df.columns:
                df = df.rename(columns={'y': 'Label'})

            combined_dfs.append(df)
            print(f"   ✓ {asset.ticker}: {len(df):,} rows")

        if not combined_dfs:
            print(f"   ⚠️  No data files found")
            return None

        # Combine all dataframes
        combined_df = pd.concat(combined_dfs, axis=0, ignore_index=True)

        # Only drop rows where Label is NaN (not all NaN values)
        # Other NaN values in features will be handled by models (XGBoost/LightGBM/CatBoost support NaN)
        if 'Label' in combined_df.columns:
            rows_before = len(combined_df)
            combined_df = combined_df.dropna(subset=['Label'])
            rows_dropped = rows_before - len(combined_df)
            if rows_dropped > 0:
                print(f"   ℹ️  Dropped {rows_dropped} rows with NaN labels")
        else:
            print(f"   ⚠️  Warning: 'Label' column not found in combined data")

        print(f"   Combined: {len(combined_df):,} rows total")

        # Get safe feature columns (intersection of all assets)
        if all_feature_cols:
            safe_feature_cols = list(set.intersection(*all_feature_cols))
            # Only keep features that exist in combined_df
            safe_feature_cols = [c for c in safe_feature_cols if c in combined_df.columns]
            print(f"   ✓ Using {len(safe_feature_cols)} safe features (intersection of all assets)")
        else:
            safe_feature_cols = None  # Fall back to exclusion-based selection

        # Train/test split
        split_idx = int(len(combined_df) * self.settings.get('train_test_split', 0.8))
        train_df = combined_df.iloc[:split_idx]
        test_df = combined_df.iloc[split_idx:]

        # Train models with safe feature columns
        models, metrics = self._train_models(train_df, test_df, group_name, safe_feature_cols)

        # Save models
        for model_name, model in models.items():
            model_path = self.models_dir / f"macro_{group_name}_{model_name}.pkl"
            joblib.dump(model, model_path)

        return {
            'group': group_name,
            'num_assets': len(assets),
            'samples': len(combined_df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': len(safe_feature_cols) if safe_feature_cols else combined_df.shape[1] - 1,
            'xgb_acc': metrics['xgboost']['accuracy'],
            'lgb_acc': metrics['lightgbm']['accuracy'],
            'cat_acc': metrics['catboost']['accuracy'],
            'ensemble_acc': metrics['ensemble']['accuracy'],
            'baseline_acc': metrics['baseline']['accuracy'],
            'improvement': metrics['improvement'],
            'timestamp': datetime.now().isoformat()
        }

    def _train_models(self, train_df, test_df, name: str, feature_cols: list = None):
        """Train XGBoost, LightGBM, CatBoost models"""
        # Use provided feature_cols if available (from add_features, which is safe)
        # Otherwise fall back to deriving from columns (with leakage protection)
        if feature_cols is None:
            # Fallback: Exclude all forward-looking columns to prevent data leakage
            exclude_cols = {
                'Label', 'Date', 'Forward_Return',  # Original exclusions
                'fwd_price', 'fwd_ret_raw', 'fwd_ret_net', 'horizon_forward',  # Future data (leakage!)
                'y',  # Alternative label column name
                'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',  # Raw OHLCV (use derived features)
            }
            feature_cols = [c for c in train_df.columns if c not in exclude_cols]
            print(f"   ℹ️  Using fallback feature selection: {len(feature_cols)} features")

        # Additional safety check - remove any leaky columns that might have slipped through
        leakage_cols = {'fwd_price', 'fwd_ret_raw', 'fwd_ret_net', 'horizon_forward',
                        'y', 'Label', 'Forward_Return', 'Future_Return', 'Next_Close',
                        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
        removed = [c for c in feature_cols if c in leakage_cols or c.startswith('fwd_')]
        if removed:
            print(f"   ⚠️  Removed {len(removed)} leaky/raw columns: {removed[:5]}{'...' if len(removed) > 5 else ''}")
        feature_cols = [c for c in feature_cols if c not in leakage_cols and not c.startswith('fwd_')]

        # Only use columns that exist in train_df
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        print(f"   ✓ Training with {len(feature_cols)} features")

        X_train = train_df[feature_cols].values
        y_train = train_df['Label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['Label'].values

        models = {}
        metrics = {}

        # XGBoost
        print(f"   [1/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = (xgb_pred == y_test).mean()

        models['xgboost'] = xgb_model
        metrics['xgboost'] = {'accuracy': xgb_acc}

        # LightGBM
        print(f"   [2/3] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = (lgb_pred == y_test).mean()

        models['lightgbm'] = lgb_model
        metrics['lightgbm'] = {'accuracy': lgb_acc}

        # CatBoost
        print(f"   [3/3] Training CatBoost...")
        cat_model = cb.CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        cat_pred = cat_model.predict(X_test)
        cat_acc = (cat_pred == y_test).mean()

        models['catboost'] = cat_model
        metrics['catboost'] = {'accuracy': cat_acc}

        # Ensemble (majority vote)
        ensemble_pred = np.round((xgb_pred + lgb_pred + cat_pred) / 3)
        ensemble_acc = (ensemble_pred == y_test).mean()
        metrics['ensemble'] = {'accuracy': ensemble_acc}

        # Calculate baseline accuracy (majority class prediction)
        # This shows what accuracy you'd get by always predicting the most common class
        unique, counts = np.unique(y_test, return_counts=True)
        baseline_acc = counts.max() / len(y_test)
        metrics['baseline'] = {'accuracy': baseline_acc}

        # Improvement over baseline (how much better than random guessing)
        improvement = ensemble_acc - baseline_acc
        metrics['improvement'] = improvement

        return models, metrics

    def _save_combined_results(self):
        """Save combined results summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'per_asset': {
                'count': len(self.per_asset_results),
                'avg_accuracy': np.mean([r['ensemble_acc'] for r in self.per_asset_results]) if self.per_asset_results else 0,
                'best': max(self.per_asset_results, key=lambda x: x['ensemble_acc']) if self.per_asset_results else None
            },
            'macro': {
                'count': len(self.macro_results),
                'avg_accuracy': np.mean([r['ensemble_acc'] for r in self.macro_results]) if self.macro_results else 0,
                'best': max(self.macro_results, key=lambda x: x['ensemble_acc']) if self.macro_results else None
            }
        }

        summary_path = self.results_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Training summary saved: {summary_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nPer-Asset Models:  {summary['per_asset']['count']} trained")
        if summary['per_asset']['best']:
            print(f"   Best: {summary['per_asset']['best']['asset']} - {summary['per_asset']['best']['ensemble_acc']:.1%}")

        print(f"\nMacro Models:      {summary['macro']['count']} trained")
        if summary['macro']['best']:
            print(f"   Best: {summary['macro']['best']['group']} - {summary['macro']['best']['ensemble_acc']:.1%}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Unified model training")
    parser.add_argument('--all', action='store_true', help="Train all models")
    parser.add_argument('--per-asset', action='store_true', help="Train per-asset models only")
    parser.add_argument('--macro', action='store_true', help="Train macro models only")
    parser.add_argument('--asset', help="Train single asset")

    args = parser.parse_args()

    trainer = UnifiedTrainer()

    if args.asset:
        asset = trainer.manager.get_asset(args.asset)
        if not asset:
            print(f"Asset not found: {args.asset}")
            return
        result = trainer._train_single_asset(asset)
        if result:
            imp_sign = '+' if result['improvement'] >= 0 else ''
            print(f"\n✓ {result['asset']}: {result['ensemble_acc']:.1%} accuracy (baseline: {result['baseline_acc']:.1%}, improvement: {imp_sign}{result['improvement']:.1%})")

    elif args.per_asset:
        trainer.train_per_asset_models()

    elif args.macro:
        trainer.train_macro_models()

    else:  # --all or no args
        trainer.train_all()


if __name__ == "__main__":
    main()
