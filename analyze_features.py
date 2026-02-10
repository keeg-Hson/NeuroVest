#!/usr/bin/env python3
"""
Feature Analysis and Evaluation Script
Identifies which features help vs hurt model performance

Run: python analyze_features.py

Outputs:
- Feature importance rankings
- Correlation analysis
- Redundant feature detection
- Features to consider pruning
- Suggestions for new features
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent))


def get_asset_list_from_config():
    """Get asset list from AssetManager config if available."""
    try:
        from framework.asset_manager import AssetManager
        manager = AssetManager()
        return manager.get_yfinance_tickers()
    except Exception as e:
        print(f"Warning: Could not load AssetManager: {e}")
        return None


def download_multi_asset_data():
    """Download data for all assets from yfinance using config or fallback list"""
    import yfinance as yf

    # Try to get tickers from AssetManager
    tickers = get_asset_list_from_config()

    if tickers is None:
        # Fallback to hardcoded list if AssetManager unavailable
        tickers = [
            # Major indices (6)
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM',
            # Sector ETFs (3)
            'XLF', 'XLK', 'XLE',
            # Bonds & Treasury (6)
            'TLT', 'IEF', 'SHY', 'HYG', 'LQD', '^TNX',
            # Dollar (2)
            'DX-Y.NYB', 'UUP',
            # Precious metals (7)
            'GLD', 'SLV', 'GDX', 'GDXJ', 'IAU', 'PPLT', 'PALL',
            # Energy (2)
            'USO', 'UNG',
            # Agriculture (3)
            'DBA', 'CORN', 'WEAT',
            # Volatility
            '^VIX',
            # Crypto proxies
            'BITO', 'ETHE', 'GBTC',
        ]
        print("Using fallback asset list (AssetManager not available)")
    else:
        print(f"Loaded {len(tickers)} assets from config/assets.yaml")

    print(f"Downloading {len(tickers)} assets from yfinance...")
    all_data = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period='5y')
            if len(df) > 0:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                all_data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} rows")
            else:
                print(f"  ✗ {ticker}: no data")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    return all_data


def load_data(use_production_features=False):
    """
    Load and prepare data for analysis.

    Data source priority (same as Streamlit dashboard):
    1. PostgreSQL database (if DATABASE_URL set)
    2. Local CSV files
    3. yfinance download (fallback only)

    Args:
        use_production_features: If True, use utils.add_features() (production pipeline)
                                 If False, use build_feature_table.build_features() (simplified)
    """
    if use_production_features:
        try:
            from utils import add_features as _build_features
            # Wrap to handle tuple return
            def build_features(df):
                result = _build_features(df)
                if isinstance(result, tuple):
                    return result[0]  # Return just the DataFrame
                return result
            print("Using PRODUCTION feature pipeline (utils.add_features)")
        except ImportError:
            print("Warning: Could not import utils.add_features, falling back to build_feature_table")
            from build_feature_table import build_features
    else:
        from build_feature_table import build_features
        print("Using ANALYSIS feature pipeline (build_feature_table.build_features)")

    print("=" * 70)
    print("FEATURE ANALYSIS - Loading Data")
    print("=" * 70)

    spy_data = None
    external_data = {}

    # PRIMARY: Use PostgreSQL database (same as Streamlit dashboard)
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        try:
            from core.data_manager_postgres import DataManager
            dm = DataManager()
            spy_data = dm.get_data('SPY')
            if spy_data is not None and len(spy_data) > 100:
                print(f"✓ Loaded SPY from PostgreSQL database: {len(spy_data)} rows")

                # Get all available assets from database
                try:
                    all_assets = dm.get_all_assets()  # Returns list of (ticker, asset_type) tuples
                    available_tickers = [t[0] for t in all_assets]
                    print(f"  Database has {len(available_tickers)} assets available")

                    for ticker in available_tickers:
                        if ticker == 'SPY':
                            continue
                        try:
                            ext = dm.get_data(ticker)
                            if ext is not None and len(ext) > 100:
                                external_data[ticker] = ext
                        except:
                            pass
                    print(f"  Loaded {len(external_data)} external assets from database")
                except Exception as e:
                    print(f"  Warning: Could not list assets: {e}")
            dm.close()
        except Exception as e:
            print(f"PostgreSQL not available: {e}")
            spy_data = None
    else:
        print("DATABASE_URL not set - cannot load from database")

    # SECONDARY: Try local CSV files
    if spy_data is None or len(spy_data) < 100:
        csv_paths = [
            Path('data/SPY.csv'),
            Path('data/SPY_daily.csv'),
            Path('data/spy_data.csv'),
            Path('data_cache/SPY.csv'),
            Path('data_cache/SPY_1d.csv'),
            Path('logs/spy_data.csv'),
        ]
        for csv_path in csv_paths:
            if csv_path.exists():
                try:
                    temp = pd.read_csv(csv_path)
                    if len(temp) > 100:
                        spy_data = temp
                        print(f"✓ Loaded SPY from CSV: {csv_path} ({len(spy_data)} rows)")
                        break
                except Exception as e:
                    pass

    # TERTIARY: Download from yfinance (fallback only)
    if spy_data is None or len(spy_data) < 100:
        print("⚠️  No database/CSV data available, falling back to yfinance download")
        print("    (This gives less data than the Streamlit dashboard)")
        try:
            all_data = download_multi_asset_data()
            if 'SPY' in all_data:
                spy_data = all_data['SPY']
                external_data = {k: v for k, v in all_data.items() if k != 'SPY'}
        except Exception as e:
            print(f"yfinance download failed: {e}")
            print("Install with: pip install yfinance")

    if spy_data is None or len(spy_data) < 100:
        print("ERROR: Could not load SPY data from any source")
        print("Options:")
        print("  1. Set DATABASE_URL environment variable (same as Streamlit)")
        print("  2. Place SPY data CSV in data/SPY.csv")
        print("  3. Install yfinance: pip install yfinance")
        sys.exit(1)

    # Standardize column names
    spy_data.columns = [c.lower() for c in spy_data.columns]

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in spy_data.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    # Handle date column
    date_col = None
    for col in ['date', 'datetime', 'timestamp', 'index']:
        if col in spy_data.columns:
            date_col = col
            break

    if date_col:
        spy_data['date'] = pd.to_datetime(spy_data[date_col])
    elif isinstance(spy_data.index, pd.DatetimeIndex):
        spy_data['date'] = spy_data.index
        spy_data = spy_data.reset_index(drop=True)

    # Sort by date
    spy_data = spy_data.sort_values('date').reset_index(drop=True)

    print(f"\nSPY date range: {spy_data['date'].min()} to {spy_data['date'].max()}")

    # Save external data for feature building
    if external_data:
        print(f"External assets loaded: {list(external_data.keys())}")
        # Save to temp location for build_features to pick up
        ext_dir = Path('data/external_temp')
        ext_dir.mkdir(parents=True, exist_ok=True)
        for ticker, df in external_data.items():
            df.to_csv(ext_dir / f'{ticker}.csv', index=False)

    # Build feature table
    print("\nBuilding feature table...")

    # Prepare data for utils.add_features() which expects capitalized columns
    if use_production_features:
        # Rename columns to match production pipeline expectations
        spy_data_renamed = spy_data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume', 'date': 'Date'
        })
        df = build_features(spy_data_renamed)
        # Normalize column names back to lowercase
        df.columns = [c.lower() if c not in ['Date'] else c for c in df.columns]
    else:
        df = build_features(spy_data)

    # Handle different close column names
    close_col = 'close' if 'close' in df.columns else 'Close'
    if close_col not in df.columns:
        # Try to find it
        for c in df.columns:
            if c.lower() == 'close':
                close_col = c
                break

    # Create target variable (1 if next day close > today's close)
    if close_col in df.columns:
        df['target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
        df['close'] = df[close_col]  # Ensure lowercase 'close' exists
    else:
        print("Warning: Could not find close column for target creation")
        df['target'] = 0

    df = df.dropna()

    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude meta columns)"""
    # Exclude price, meta, and target columns (both lowercase and titlecase)
    exclude = [
        'date', 'Date', 'datetime', 'timestamp',
        'open', 'Open', 'high', 'High', 'low', 'Low', 'close', 'Close',
        'volume', 'Volume', 'adj_close', 'Adj Close', 'adj close',
        'target', 'Target', 'ticker', 'Ticker', 'asset_type',
        'returns', 'log_returns', 'Daily_Return', 'daily_return',
        'y', 'y_up_fwd', 'y_class_3', 'split', 'fwd_ret', 'fwd_price',
    ]
    exclude_set = set(e.lower() for e in exclude)

    features = []
    for c in df.columns:
        if c.lower() in exclude_set:
            continue
        if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
            features.append(c)

    return features


def analyze_feature_statistics(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Basic statistics for each feature"""
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    stats = []
    for feat in features:
        col = df[feat]
        stats.append({
            'feature': feat,
            'mean': col.mean(),
            'std': col.std(),
            'min': col.min(),
            'max': col.max(),
            'null_pct': col.isna().mean() * 100,
            'zero_pct': (col == 0).mean() * 100,
            'unique_ratio': col.nunique() / len(col) * 100
        })

    stats_df = pd.DataFrame(stats)

    # Flag potential issues
    print("\nPotential Issues:")
    high_null = stats_df[stats_df['null_pct'] > 5]
    if len(high_null) > 0:
        print(f"  High null rate (>5%): {list(high_null['feature'])}")

    low_variance = stats_df[stats_df['std'] < 0.001]
    if len(low_variance) > 0:
        print(f"  Near-zero variance: {list(low_variance['feature'])}")

    high_zero = stats_df[stats_df['zero_pct'] > 90]
    if len(high_zero) > 0:
        print(f"  Mostly zeros (>90%): {list(high_zero['feature'])}")

    return stats_df


def analyze_correlations(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[Tuple]]:
    """Find highly correlated feature pairs"""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Correlation matrix
    corr_matrix = df[features].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if i < j:
                corr = corr_matrix.loc[f1, f2]
                if abs(corr) > 0.95:
                    high_corr_pairs.append((f1, f2, corr))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nHighly correlated pairs (|r| > 0.95): {len(high_corr_pairs)}")
    for f1, f2, corr in high_corr_pairs[:20]:
        print(f"  {f1} <-> {f2}: {corr:.3f}")

    # Correlation with target
    target_corr = df[features + ['target']].corr()['target'].drop('target')
    target_corr = target_corr.abs().sort_values(ascending=False)

    print(f"\nTop 20 features correlated with target:")
    for feat, corr in target_corr.head(20).items():
        print(f"  {feat}: {corr:.4f}")

    print(f"\nBottom 10 features (lowest target correlation):")
    for feat, corr in target_corr.tail(10).items():
        print(f"  {feat}: {corr:.4f}")

    return corr_matrix, high_corr_pairs


def train_baseline_model(X_train, y_train, X_test, y_test, model_type='gradient_boosting'):
    """Train a baseline model and return metrics"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except ImportError:
            print("Warning: XGBoost not installed, falling back to GradientBoosting")
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
    elif model_type == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )
        except ImportError:
            print("Warning: LightGBM not installed, falling back to GradientBoosting")
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'model': model
    }


def compare_model_performance(X_train, y_train, X_test, y_test):
    """Compare performance across different model types"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Production vs Baseline)")
    print("=" * 70)

    models = ['gradient_boosting', 'xgboost', 'lightgbm']
    results = {}

    for model_type in models:
        try:
            print(f"\nTraining {model_type}...")
            metrics = train_baseline_model(X_train, y_train, X_test, y_test, model_type=model_type)
            results[model_type] = metrics
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1:       {metrics['f1']:.4f}")
            print(f"  AUC:      {metrics['auc']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "-" * 70)
    print("COMPARISON SUMMARY:")
    print("-" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 70)
    for model_type, metrics in results.items():
        print(f"{model_type:<20} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc']:>10.4f}")

    # Calculate ensemble prediction
    if len(results) >= 2:
        print("\n" + "-" * 70)
        print("ENSEMBLE (Average prediction):")
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        probs = []
        for model_type, metrics in results.items():
            probs.append(metrics['model'].predict_proba(X_test)[:, 1])

        avg_prob = np.mean(probs, axis=0)
        y_pred_ensemble = (avg_prob >= 0.5).astype(int)

        ens_acc = accuracy_score(y_test, y_pred_ensemble)
        ens_f1 = f1_score(y_test, y_pred_ensemble)
        ens_auc = roc_auc_score(y_test, avg_prob)
        print(f"{'Ensemble':<20} {ens_acc:>10.4f} {ens_f1:>10.4f} {ens_auc:>10.4f}")

    return results


def analyze_feature_importance(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculate feature importance using multiple methods"""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance

    # Prepare data
    X = df[features].fillna(0)
    y = df['target']

    # Train/test split (time-based)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train baseline
    print("\nTraining baseline model (GradientBoosting)...")
    baseline = train_baseline_model(X_train_scaled, y_train, X_test_scaled, y_test, model_type='gradient_boosting')
    print(f"Baseline - Accuracy: {baseline['accuracy']:.4f}, F1: {baseline['f1']:.4f}, AUC: {baseline['auc']:.4f}")

    # Compare with production models
    model_comparison = compare_model_performance(X_train_scaled, y_train, X_test_scaled, y_test)

    # Method 1: Built-in feature importance
    print("\nCalculating built-in importance...")
    builtin_importance = baseline['model'].feature_importances_

    # Method 2: Permutation importance
    print("Calculating permutation importance (this takes a while)...")
    perm_result = permutation_importance(
        baseline['model'], X_test_scaled, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importance = perm_result.importances_mean

    # Combine results
    importance_df = pd.DataFrame({
        'feature': features,
        'builtin_importance': builtin_importance,
        'permutation_importance': perm_importance
    })

    # Normalize to 0-100 scale
    importance_df['builtin_norm'] = (importance_df['builtin_importance'] / importance_df['builtin_importance'].max()) * 100
    importance_df['perm_norm'] = (importance_df['permutation_importance'] / importance_df['permutation_importance'].max()) * 100
    importance_df['combined_score'] = (importance_df['builtin_norm'] + importance_df['perm_norm']) / 2

    importance_df = importance_df.sort_values('combined_score', ascending=False)

    print("\nTop 30 Most Important Features:")
    print("-" * 70)
    for i, row in importance_df.head(30).iterrows():
        print(f"  {row['feature']:40s} | Score: {row['combined_score']:6.2f}")

    print("\nBottom 30 Least Important Features:")
    print("-" * 70)
    for i, row in importance_df.tail(30).iterrows():
        print(f"  {row['feature']:40s} | Score: {row['combined_score']:6.2f}")

    return importance_df, baseline, (X_train_scaled, y_train, X_test_scaled, y_test, scaler)


def analyze_drop_column_importance(df: pd.DataFrame, features: List[str],
                                   baseline: Dict, data_tuple: tuple) -> pd.DataFrame:
    """Test model performance when each feature is dropped"""
    print("\n" + "=" * 70)
    print("DROP-COLUMN ANALYSIS")
    print("=" * 70)
    print("Testing impact of removing each feature...")

    X_train, y_train, X_test, y_test, scaler = data_tuple
    baseline_auc = baseline['auc']

    results = []
    total = len(features)

    for idx, feat in enumerate(features):
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{total}")

        # Get feature index
        feat_idx = features.index(feat)

        # Create data without this feature
        X_train_drop = np.delete(X_train, feat_idx, axis=1)
        X_test_drop = np.delete(X_test, feat_idx, axis=1)

        # Train model without feature
        try:
            metrics = train_baseline_model(X_train_drop, y_train, X_test_drop, y_test)
            auc_change = metrics['auc'] - baseline_auc
            results.append({
                'feature': feat,
                'auc_without': metrics['auc'],
                'auc_change': auc_change,
                'accuracy_without': metrics['accuracy']
            })
        except Exception as e:
            results.append({
                'feature': feat,
                'auc_without': np.nan,
                'auc_change': np.nan,
                'accuracy_without': np.nan
            })

    drop_df = pd.DataFrame(results)
    drop_df = drop_df.sort_values('auc_change', ascending=True)

    print(f"\nBaseline AUC: {baseline_auc:.4f}")

    print("\nFeatures that HURT when removed (most valuable):")
    print("-" * 70)
    hurt_when_removed = drop_df[drop_df['auc_change'] < -0.001].head(20)
    for _, row in hurt_when_removed.iterrows():
        print(f"  {row['feature']:40s} | AUC change: {row['auc_change']:+.4f}")

    print("\nFeatures that HELP when removed (consider pruning):")
    print("-" * 70)
    help_when_removed = drop_df[drop_df['auc_change'] > 0.001].tail(20)
    for _, row in help_when_removed.sort_values('auc_change', ascending=False).iterrows():
        print(f"  {row['feature']:40s} | AUC change: {row['auc_change']:+.4f}")

    return drop_df


def analyze_feature_groups(df: pd.DataFrame, features: List[str],
                           baseline: Dict, data_tuple: tuple) -> Dict:
    """Analyze importance of feature groups"""
    print("\n" + "=" * 70)
    print("FEATURE GROUP ANALYSIS")
    print("=" * 70)

    # Define feature groups
    groups = {
        'momentum': [f for f in features if any(x in f.lower() for x in ['rsi', 'momentum', 'roc', 'macd', 'stoch'])],
        'volatility': [f for f in features if any(x in f.lower() for x in ['volatility', 'atr', 'bbwidth', 'range', 'std'])],
        'trend': [f for f in features if any(x in f.lower() for x in ['sma', 'ema', 'trend', 'adx', 'slope'])],
        'volume': [f for f in features if any(x in f.lower() for x in ['volume', 'obv', 'vwap', 'mfi'])],
        'price': [f for f in features if any(x in f.lower() for x in ['price', 'gap', 'return', 'close', 'high', 'low'])],
        'sentiment': [f for f in features if any(x in f.lower() for x in ['sentiment', 'reddit', 'news', 'social'])],
        'sector': [f for f in features if any(x in f.lower() for x in ['sector', 'xlf', 'xlk', 'xle'])],
        'macro': [f for f in features if any(x in f.lower() for x in ['vix', 'yield', 'spread', 'dxy', 'gold', 'oil'])],
        'calendar': [f for f in features if any(x in f.lower() for x in ['day_of_week', 'month', 'quarter', 'fomc', 'earnings'])],
    }

    X_train, y_train, X_test, y_test, scaler = data_tuple
    baseline_auc = baseline['auc']

    group_results = {}

    for group_name, group_features in groups.items():
        if not group_features:
            continue

        # Find indices of group features
        group_indices = [features.index(f) for f in group_features if f in features]

        if not group_indices:
            continue

        # Create data without this group
        X_train_drop = np.delete(X_train, group_indices, axis=1)
        X_test_drop = np.delete(X_test, group_indices, axis=1)

        try:
            metrics = train_baseline_model(X_train_drop, y_train, X_test_drop, y_test)
            auc_change = metrics['auc'] - baseline_auc
            group_results[group_name] = {
                'feature_count': len(group_features),
                'auc_without': metrics['auc'],
                'auc_change': auc_change,
                'features': group_features[:5]  # Sample features
            }
        except Exception as e:
            print(f"  Error with group {group_name}: {e}")

    print(f"\nBaseline AUC: {baseline_auc:.4f}")
    print("\nGroup Impact (when removed):")
    print("-" * 70)

    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['auc_change'])
    for group_name, data in sorted_groups:
        impact = "VALUABLE" if data['auc_change'] < -0.002 else "NEUTRAL" if abs(data['auc_change']) < 0.002 else "HARMFUL"
        print(f"  {group_name:15s} | {data['feature_count']:3d} features | AUC change: {data['auc_change']:+.4f} | {impact}")

    return group_results


def suggest_improvements(importance_df: pd.DataFrame, drop_df: pd.DataFrame,
                        corr_pairs: List[Tuple], group_results: Dict):
    """Generate actionable recommendations"""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Features to consider removing
    print("\n1. FEATURES TO CONSIDER REMOVING:")
    print("-" * 50)

    # Low importance features
    low_importance = importance_df[importance_df['combined_score'] < 5]['feature'].tolist()

    # Features that improve AUC when removed
    harmful = drop_df[drop_df['auc_change'] > 0.002]['feature'].tolist() if not drop_df.empty else []

    # Combine recommendations
    to_remove = set(low_importance[:20]) | set(harmful[:10])
    for feat in list(to_remove)[:15]:
        print(f"  - {feat}")

    # Redundant features (high correlation)
    print("\n2. REDUNDANT FEATURES (keep one, remove others):")
    print("-" * 50)
    seen = set()
    for f1, f2, corr in corr_pairs[:10]:
        if f1 not in seen and f2 not in seen:
            print(f"  - Keep {f1}, consider removing {f2} (r={corr:.3f})")
            seen.add(f2)

    # Feature groups to improve
    print("\n3. FEATURE GROUPS TO IMPROVE:")
    print("-" * 50)
    for group_name, data in sorted(group_results.items(), key=lambda x: x[1]['auc_change']):
        if data['auc_change'] > 0:
            print(f"  - {group_name}: Consider redesigning or removing ({data['feature_count']} features)")

    # Suggestions for new features
    print("\n4. SUGGESTED NEW FEATURES TO ADD:")
    print("-" * 50)
    suggestions = [
        "Cross-asset momentum (SPY vs QQQ, SPY vs IWM ratios)",
        "Implied volatility term structure (VIX vs VIX3M)",
        "Options flow signals (put/call ratio changes)",
        "Credit spread momentum (HYG/LQD ratio)",
        "Sector rotation signals (XLK/XLF relative strength)",
        "Intraday volatility patterns (if available)",
        "Volume profile features (volume at price levels)",
        "Market breadth (advance/decline, new highs/lows)",
        "Intermarket divergences (bonds vs stocks)",
        "Sentiment momentum (rate of change in sentiment)",
    ]
    for s in suggestions:
        print(f"  + {s}")

    print("\n5. HYPERPARAMETER TUNING SUGGESTIONS:")
    print("-" * 50)
    print("  - Try different lookback periods for momentum (5, 10, 21, 63 days)")
    print("  - Test ensemble weights (currently equal weighting)")
    print("  - Consider asymmetric loss functions for up/down predictions")
    print("  - Test different feature scaling methods (robust scaler, quantile)")


def save_results(importance_df: pd.DataFrame, drop_df: pd.DataFrame,
                 stats_df: pd.DataFrame, output_dir: Path):
    """Save analysis results to files"""
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    importance_df.to_csv(output_dir / f'feature_importance_{timestamp}.csv', index=False)
    drop_df.to_csv(output_dir / f'drop_column_analysis_{timestamp}.csv', index=False)
    stats_df.to_csv(output_dir / f'feature_statistics_{timestamp}.csv', index=False)

    print(f"\nResults saved to {output_dir}/")


def main():
    """Main analysis pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze feature importance and quality')
    parser.add_argument('--production', '-p', action='store_true',
                        help='Use production feature pipeline (utils.add_features) instead of simplified')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare both feature pipelines side by side')
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\nStarting Feature Analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.compare:
        print("\n" + "=" * 70)
        print("COMPARISON MODE: Analyzing both feature pipelines")
        print("=" * 70)

        # Run with simplified features
        print("\n\n>>> SIMPLIFIED FEATURES (build_feature_table.build_features)")
        print("-" * 70)
        df_simple = load_data(use_production_features=False)
        features_simple = get_feature_columns(df_simple)
        print(f"Simplified pipeline: {len(features_simple)} features")

        # Run with production features
        print("\n\n>>> PRODUCTION FEATURES (utils.add_features)")
        print("-" * 70)
        df_prod = load_data(use_production_features=True)
        features_prod = get_feature_columns(df_prod)
        print(f"Production pipeline: {len(features_prod)} features")

        # Compare
        print("\n" + "=" * 70)
        print("FEATURE COMPARISON")
        print("=" * 70)
        simple_set = set(features_simple)
        prod_set = set(features_prod)

        only_in_simple = simple_set - prod_set
        only_in_prod = prod_set - simple_set
        common = simple_set & prod_set

        print(f"\nCommon features: {len(common)}")
        print(f"Only in simplified: {len(only_in_simple)}")
        if only_in_simple:
            for f in sorted(only_in_simple)[:10]:
                print(f"  - {f}")
            if len(only_in_simple) > 10:
                print(f"  ... and {len(only_in_simple) - 10} more")

        print(f"\nOnly in production: {len(only_in_prod)}")
        if only_in_prod:
            for f in sorted(only_in_prod)[:15]:
                print(f"  + {f}")
            if len(only_in_prod) > 15:
                print(f"  ... and {len(only_in_prod) - 15} more")

        # Continue with production features for full analysis
        df = df_prod
        features = features_prod
        print("\n\nContinuing analysis with PRODUCTION features...")
    else:
        # Load data with specified pipeline
        df = load_data(use_production_features=args.production)
        features = get_feature_columns(df)

    print(f"\nTotal features to analyze: {len(features)}")

    # Run analyses
    stats_df = analyze_feature_statistics(df, features)
    corr_matrix, corr_pairs = analyze_correlations(df, features)
    importance_df, baseline, data_tuple = analyze_feature_importance(df, features)
    drop_df = analyze_drop_column_importance(df, features, baseline, data_tuple)
    group_results = analyze_feature_groups(df, features, baseline, data_tuple)

    # Generate recommendations
    suggest_improvements(importance_df, drop_df, corr_pairs, group_results)

    # Save results
    output_dir = Path('outputs/feature_analysis')
    save_results(importance_df, drop_df, stats_df, output_dir)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Features analyzed: {len(features)}")
    print(f"Baseline accuracy: {baseline['accuracy']:.4f}")
    print(f"Baseline AUC: {baseline['auc']:.4f}")
    print(f"\nRun with these results and share the output for specific recommendations.")


if __name__ == '__main__':
    main()
