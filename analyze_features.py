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


def download_multi_asset_data():
    """Download data for all 41 assets from yfinance"""
    import yfinance as yf

    # Full asset list (31 stock/ETF + attempt crypto proxies)
    tickers = [
        # Major indices (6)
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM',
        # Sector ETFs (3)
        'XLF', 'XLK', 'XLE',
        # Bonds & Treasury (6)
        'TLT', 'IEF', 'SHY', 'HYG', 'LQD', '^TNX',
        # Dollar (2)
        'DX-Y.NYB', 'UUP',  # DXY proxy
        # Precious metals (7)
        'GLD', 'SLV', 'GDX', 'GDXJ', 'IAU', 'PPLT', 'PALL',
        # Energy (2)
        'USO', 'UNG',
        # Agriculture (3)
        'DBA', 'CORN', 'WEAT',
        # Volatility
        '^VIX',
        # Crypto proxies (ETFs since yfinance doesn't do spot crypto well)
        'BITO', 'ETHE', 'GBTC',
    ]

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


def load_data():
    """Load and prepare data for analysis"""
    from build_feature_table import build_features

    print("=" * 70)
    print("FEATURE ANALYSIS - Loading Data")
    print("=" * 70)

    spy_data = None
    external_data = {}

    # Option 1: Try PostgreSQL if DATABASE_URL is set
    if os.environ.get('DATABASE_URL'):
        try:
            from core.data_manager_postgres import DataManager
            dm = DataManager()
            spy_data = dm.get_data('SPY')
            if spy_data is not None and len(spy_data) > 100:
                print(f"Loaded SPY from PostgreSQL: {len(spy_data)} rows")
                # Also try to get all other assets
                all_tickers = [
                    'QQQ', 'IWM', 'DIA', 'VTI', 'EEM',
                    'XLF', 'XLK', 'XLE',
                    'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'TNX',
                    'DXY', 'UUP',
                    'GLD', 'SLV', 'GDX', 'GDXJ', 'IAU', 'PPLT', 'PALL',
                    'USO', 'UNG', 'DBA', 'CORN', 'WEAT',
                    'VIX',
                    'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT', 'XRP_USDT',
                    'ADA_USDT', 'DOGE_USDT', 'AVAX_USDT', 'DOT_USDT', 'LINK_USDT',
                ]
                for ticker in all_tickers:
                    try:
                        ext = dm.get_data(ticker)
                        if ext is not None and len(ext) > 100:
                            external_data[ticker] = ext
                    except:
                        pass
                print(f"  Loaded {len(external_data)} external assets from PostgreSQL")
            dm.close()
        except Exception as e:
            print(f"PostgreSQL not available: {e}")

    # Option 2: Try local CSV files
    if spy_data is None or len(spy_data) < 100:
        csv_paths = [
            Path('data/SPY_daily.csv'),
            Path('data/spy_data.csv'),
            Path('data_cache/SPY.csv'),
            Path('logs/spy_data.csv'),
        ]
        for csv_path in csv_paths:
            if csv_path.exists():
                try:
                    temp = pd.read_csv(csv_path)
                    if len(temp) > 100:  # Must have substantial data
                        spy_data = temp
                        print(f"Loaded from {csv_path}: {len(spy_data)} rows")
                        break
                except Exception as e:
                    pass

    # Option 3: Download fresh data from yfinance
    if spy_data is None or len(spy_data) < 100:
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
        print("  1. Set DATABASE_URL environment variable")
        print("  2. Place SPY data CSV in data/SPY_daily.csv")
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
    df = build_features(spy_data)

    # Create target variable (1 if next day close > today's close)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude meta columns)"""
    exclude = ['date', 'Date', 'open', 'high', 'low', 'close', 'volume',
               'target', 'ticker', 'asset_type', 'returns', 'log_returns']
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]


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


def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train a baseline model and return metrics"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
    print("\nTraining baseline model...")
    baseline = train_baseline_model(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Baseline - Accuracy: {baseline['accuracy']:.4f}, F1: {baseline['f1']:.4f}, AUC: {baseline['auc']:.4f}")

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
    start_time = datetime.now()
    print(f"\nStarting Feature Analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()
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
