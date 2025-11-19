#!/usr/bin/env python3
"""
Asset Correlation Analysis

Analyzes correlations between assets to identify:
- Low-correlation pairs for diversification
- Highly correlated assets (avoid redundancy)
- Optimal portfolio combinations
- Diversification scores

Usage:
    python3 analyze_correlations.py --asset-group crypto
    python3 analyze_correlations.py --assets SPY,QQQ,BTC/USDT,ETH/USDT
    python3 analyze_correlations.py --all

Output:
    - Correlation matrix
    - Low-correlation pairs
    - Portfolio diversification recommendations
    - Correlation heatmap
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent / "framework"))
from framework.asset_manager import AssetManager

from utils import load_asset_data
from config import LOGS_DIR


class CorrelationAnalyzer:
    """Analyzes asset correlations for portfolio construction"""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.data_dir = Path("data_cache")
        self.output_dir = LOGS_DIR / "correlation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_returns(self, assets: list) -> pd.DataFrame:
        """Load returns for multiple assets"""

        print(f"\n📥 Loading {len(assets)} assets...")

        returns_dict = {}

        for ticker in assets:
            try:
                df = load_asset_data(ticker)

                # Calculate daily returns
                daily_returns = df['Close'].pct_change().dropna()

                returns_dict[ticker] = daily_returns

                print(f"   ✓ {ticker:15s} {len(daily_returns):,} returns")

            except Exception as e:
                print(f"   ✗ {ticker:15s} Error: {e}")

        if not returns_dict:
            raise ValueError("No asset data loaded")

        # Combine into dataframe with aligned dates
        df_returns = pd.DataFrame(returns_dict)

        # Drop rows with any NaN (ensures aligned time series)
        df_returns = df_returns.dropna()

        print(f"\n📊 Combined returns: {len(df_returns):,} days")
        print(f"   Date range: {df_returns.index.min()} to {df_returns.index.max()}")

        # Trim to lookback period if specified
        if self.lookback_days and len(df_returns) > self.lookback_days:
            df_returns = df_returns.iloc[-self.lookback_days:]
            print(f"   Using last {self.lookback_days} days for correlation")

        return df_returns

    def calculate_correlations(self, df_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""

        print("\n🔧 Calculating correlations...")

        corr_matrix = df_returns.corr()

        print(f"   ✓ Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")

        return corr_matrix

    def find_low_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """Find pairs with low correlation (good for diversification)"""

        print(f"\n🔍 Finding low-correlation pairs (< {threshold})...")

        pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]

                if abs(corr) < threshold:
                    pairs.append({
                        'Asset 1': asset1,
                        'Asset 2': asset2,
                        'Correlation': corr,
                        'Abs Correlation': abs(corr)
                    })

        if pairs:
            df_pairs = pd.DataFrame(pairs).sort_values('Abs Correlation')
        else:
            df_pairs = pd.DataFrame(columns=['Asset 1', 'Asset 2', 'Correlation', 'Abs Correlation'])

        print(f"   Found {len(df_pairs)} low-correlation pairs")

        return df_pairs

    def find_high_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Find highly correlated pairs (redundant assets)"""

        print(f"\n🔍 Finding high-correlation pairs (> {threshold})...")

        pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]

                if corr > threshold:
                    pairs.append({
                        'Asset 1': asset1,
                        'Asset 2': asset2,
                        'Correlation': corr
                    })

        if pairs:
            df_pairs = pd.DataFrame(pairs).sort_values('Correlation', ascending=False)
        else:
            df_pairs = pd.DataFrame(columns=['Asset 1', 'Asset 2', 'Correlation'])

        print(f"   Found {len(df_pairs)} high-correlation pairs")

        return df_pairs

    def calculate_diversification_score(self, corr_matrix: pd.DataFrame) -> dict:
        """Calculate diversification metrics"""

        print("\n📊 Calculating diversification scores...")

        # Average correlation (lower is more diversified)
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

        # Diversification ratio: 1 / avg_abs_corr
        avg_abs_corr = np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).mean()
        div_ratio = 1 / avg_abs_corr if avg_abs_corr > 0 else 0

        # Effective number of independent bets (eigenvalue-based)
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        eigenvalues = eigenvalues[eigenvalues > 0]
        enb = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() if len(eigenvalues) > 0 else 1

        # Per-asset average correlation
        asset_avg_corr = {}
        for asset in corr_matrix.columns:
            # Average correlation with all other assets
            corrs = corr_matrix[asset].drop(asset).values
            asset_avg_corr[asset] = corrs.mean()

        scores = {
            'avg_correlation': avg_corr,
            'avg_abs_correlation': avg_abs_corr,
            'diversification_ratio': div_ratio,
            'effective_n_bets': enb,
            'n_assets': len(corr_matrix),
            'asset_avg_corr': asset_avg_corr
        }

        print(f"   Average correlation:      {avg_corr:.3f}")
        print(f"   Average |correlation|:    {avg_abs_corr:.3f}")
        print(f"   Diversification ratio:    {div_ratio:.2f}")
        print(f"   Effective # of bets:      {enb:.1f} (out of {len(corr_matrix)} assets)")

        return scores

    def recommend_portfolio(self, corr_matrix: pd.DataFrame, n_assets: int = 5) -> list:
        """Recommend diversified portfolio (greedy selection)"""

        print(f"\n💡 Recommending {n_assets}-asset diversified portfolio...")

        selected = []
        remaining = list(corr_matrix.columns)

        # Start with asset that has lowest average correlation
        avg_corrs = corr_matrix.mean(axis=0)
        first_asset = avg_corrs.idxmin()
        selected.append(first_asset)
        remaining.remove(first_asset)

        # Greedily add assets with lowest average correlation to selected
        while len(selected) < n_assets and remaining:
            min_avg_corr = float('inf')
            best_asset = None

            for candidate in remaining:
                # Average correlation with already selected assets
                avg_corr = corr_matrix.loc[selected, candidate].mean()

                if avg_corr < min_avg_corr:
                    min_avg_corr = avg_corr
                    best_asset = candidate

            if best_asset:
                selected.append(best_asset)
                remaining.remove(best_asset)
                print(f"   [{len(selected)}/{n_assets}] Added {best_asset:15s} (avg corr: {min_avg_corr:.3f})")

        return selected

    def plot_heatmap(self, corr_matrix: pd.DataFrame, timestamp: str):
        """Plot correlation heatmap"""

        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'}
        )

        plt.title('Asset Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Asset', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        plot_file = self.output_dir / f"correlation_heatmap_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved heatmap: {plot_file}")

        plt.close()

    def analyze(self, assets: list):
        """Run full correlation analysis"""

        print(f"\n{'=' * 80}")
        print("ASSET CORRELATION ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Assets: {len(assets)}")
        print(f"Lookback: {self.lookback_days} days")
        print(f"{'=' * 80}")

        # Load returns
        df_returns = self.load_returns(assets)

        # Calculate correlations
        corr_matrix = self.calculate_correlations(df_returns)

        # Find low/high correlation pairs
        low_corr_pairs = self.find_low_correlation_pairs(corr_matrix, threshold=0.3)
        high_corr_pairs = self.find_high_correlation_pairs(corr_matrix, threshold=0.8)

        # Calculate diversification scores
        div_scores = self.calculate_diversification_score(corr_matrix)

        # Recommend portfolio
        recommended = self.recommend_portfolio(corr_matrix, n_assets=min(5, len(assets)))

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Correlation matrix
        corr_file = self.output_dir / f"correlation_matrix_{timestamp}.csv"
        corr_matrix.to_csv(corr_file)
        print(f"\n💾 Saved correlation matrix: {corr_file}")

        # Low correlation pairs
        if not low_corr_pairs.empty:
            low_file = self.output_dir / f"low_corr_pairs_{timestamp}.csv"
            low_corr_pairs.to_csv(low_file, index=False)
            print(f"💾 Saved low-correlation pairs: {low_file}")

        # High correlation pairs
        if not high_corr_pairs.empty:
            high_file = self.output_dir / f"high_corr_pairs_{timestamp}.csv"
            high_corr_pairs.to_csv(high_file, index=False)
            print(f"💾 Saved high-correlation pairs: {high_file}")

        # Diversification scores
        scores_file = self.output_dir / f"diversification_scores_{timestamp}.json"
        import json
        with open(scores_file, 'w') as f:
            json.dump(div_scores, f, indent=2, default=str)
        print(f"💾 Saved diversification scores: {scores_file}")

        # Print summary
        print(f"\n{'=' * 80}")
        print("LOW-CORRELATION PAIRS (Best for Diversification)")
        print(f"{'=' * 80}")
        if not low_corr_pairs.empty:
            print(low_corr_pairs.head(10).to_string(index=False))
        else:
            print("No pairs with correlation < 0.3")

        print(f"\n{'=' * 80}")
        print("HIGH-CORRELATION PAIRS (Redundant Assets)")
        print(f"{'=' * 80}")
        if not high_corr_pairs.empty:
            print(high_corr_pairs.head(10).to_string(index=False))
        else:
            print("No pairs with correlation > 0.8")

        print(f"\n{'=' * 80}")
        print("RECOMMENDED DIVERSIFIED PORTFOLIO")
        print(f"{'=' * 80}")
        print(f"Assets: {', '.join(recommended)}")

        # Calculate recommended portfolio metrics
        rec_corr = corr_matrix.loc[recommended, recommended]
        rec_avg = rec_corr.values[np.triu_indices_from(rec_corr.values, k=1)].mean()
        print(f"Average correlation: {rec_avg:.3f}")
        print(f"{'=' * 80}\n")

        # Plot heatmap
        self.plot_heatmap(corr_matrix, timestamp)

        return corr_matrix, div_scores, recommended


def main():
    parser = argparse.ArgumentParser(description="Asset correlation analysis")

    parser.add_argument('--assets', type=str,
                        help="Comma-separated list of assets (e.g., SPY,QQQ,BTC/USDT)")
    parser.add_argument('--asset-group', choices=['equity', 'crypto', 'bond', 'commodity'],
                        help="Analyze all assets from a group")
    parser.add_argument('--all', action='store_true',
                        help="Analyze all available assets")

    parser.add_argument('--lookback', type=int, default=252,
                        help="Lookback period in days (default: 252 = 1 year)")

    args = parser.parse_args()

    # Get asset list
    if args.assets:
        assets = [a.strip() for a in args.assets.split(',')]

    elif args.asset_group or args.all:
        mgr = AssetManager()

        if args.all:
            all_assets = mgr.get_all_assets()
        else:
            all_assets = [a for a in mgr.get_all_assets() if a.asset_type == args.asset_group]

        # Filter to assets with data
        data_dir = Path('data_cache')
        assets = []
        for asset_obj in all_assets:
            filename = asset_obj.ticker.replace('/', '_') + '_1d.csv'
            if (data_dir / filename).exists():
                assets.append(asset_obj.ticker)

        if not assets:
            print(f"❌ No assets with data")
            return

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 analyze_correlations.py --assets SPY,QQQ,BTC/USDT,ETH/USDT")
        print("  python3 analyze_correlations.py --asset-group crypto")
        print("  python3 analyze_correlations.py --all")
        return

    # Run analysis
    analyzer = CorrelationAnalyzer(lookback_days=args.lookback)
    analyzer.analyze(assets)


if __name__ == "__main__":
    main()
