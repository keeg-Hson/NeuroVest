#!/usr/bin/env python3
"""
Results Dashboard - Find and Display Model Performance

Aggregates and displays all training results in an easy-to-read format.
Helps you quickly find:
- Best performing assets
- Model comparisons (per-asset vs macro)
- Performance trends over time
- Which models to use for which assets

Usage:
    python framework/results_dashboard.py                # Show all results
    python framework/results_dashboard.py --top 10       # Show top 10 assets
    python framework/results_dashboard.py --type equity  # Filter by asset type
    python framework/results_dashboard.py --export html  # Export to HTML
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from asset_manager import AssetManager


class ResultsDashboard:
    """Aggregates and displays all training results"""

    def __init__(self):
        self.manager = AssetManager()
        self.results_dir = Path("results")
        self.per_asset_results = None
        self.macro_results = None
        self.summary = None

        self._load_latest_results()

    def _load_latest_results(self):
        """Load latest results files"""
        # Per-asset results
        per_asset_files = list(self.results_dir.glob("per_asset_results_*.csv"))
        if per_asset_files:
            latest = max(per_asset_files, key=lambda x: x.stat().st_mtime)
            self.per_asset_results = pd.read_csv(latest)

        # Macro results
        macro_files = list(self.results_dir.glob("macro_results_*.csv"))
        if macro_files:
            latest = max(macro_files, key=lambda x: x.stat().st_mtime)
            self.macro_results = pd.read_csv(latest)

        # Summary
        summary_files = list(self.results_dir.glob("training_summary_*.json"))
        if summary_files:
            latest = max(summary_files, key=lambda x: x.stat().st_mtime)
            with open(latest) as f:
                self.summary = json.load(f)

    def display_all(self, asset_type: Optional[str] = None, top_n: Optional[int] = None):
        """Display comprehensive dashboard"""
        print("=" * 100)
        print("NEUROVEST RESULTS DASHBOARD")
        print("=" * 100)

        # Overview
        self._display_overview()

        # Per-asset results
        if self.per_asset_results is not None:
            self._display_per_asset(asset_type=asset_type, top_n=top_n)

        # Macro results
        if self.macro_results is not None:
            self._display_macro()

        # Recommendations
        self._display_recommendations()

        print("=" * 100)

    def _display_overview(self):
        """Display high-level overview"""
        print("\n📊 OVERVIEW")
        print("-" * 100)

        if self.summary:
            per_asset = self.summary.get('per_asset', {})
            macro = self.summary.get('macro', {})

            print(f"Training Date: {self.summary.get('timestamp', 'Unknown')}")
            print(f"\nPer-Asset Models:  {per_asset.get('count', 0)} trained")
            print(f"   Avg Accuracy:   {per_asset.get('avg_accuracy', 0):.1%}")
            if per_asset.get('best'):
                best = per_asset['best']
                print(f"   Best Model:     {best['asset']} ({best['ensemble_acc']:.1%})")

            print(f"\nMacro Models:      {macro.get('count', 0)} trained")
            print(f"   Avg Accuracy:   {macro.get('avg_accuracy', 0):.1%}")
            if macro.get('best'):
                best = macro['best']
                print(f"   Best Model:     {best['group']} ({best['ensemble_acc']:.1%})")
        else:
            print("No training summary available. Run training first.")

    def _display_per_asset(self, asset_type: Optional[str] = None, top_n: Optional[int] = None):
        """Display per-asset results table"""
        print("\n\n📈 PER-ASSET MODEL RESULTS")
        print("-" * 100)

        df = self.per_asset_results.copy()

        # Filter by type
        if asset_type:
            df = df[df['type'] == asset_type]
            print(f"Filtered by type: {asset_type}")

        # Sort by ensemble accuracy
        df = df.sort_values('ensemble_acc', ascending=False)

        # Limit to top N
        if top_n:
            df = df.head(top_n)
            print(f"Showing top {top_n} assets\n")

        # Format table
        display_df = df[['asset', 'name', 'type', 'samples', 'xgb_acc', 'lgb_acc', 'cat_acc', 'ensemble_acc']].copy()
        display_df.columns = ['Asset', 'Name', 'Type', 'Samples', 'XGB', 'LGB', 'CAT', 'Ensemble']

        # Format percentages
        for col in ['XGB', 'LGB', 'CAT', 'Ensemble']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")

        print(display_df.to_string(index=False))

        # Stats
        print(f"\nTotal Assets: {len(df)}")
        print(f"Average Ensemble Accuracy: {df['ensemble_acc'].mean():.1%}")
        print(f"Best: {df.iloc[0]['asset']} ({df.iloc[0]['ensemble_acc']:.1%})")
        print(f"Worst: {df.iloc[-1]['asset']} ({df.iloc[-1]['ensemble_acc']:.1%})")

    def _display_macro(self):
        """Display macro model results"""
        print("\n\n🌍 MACRO MODEL RESULTS")
        print("-" * 100)

        df = self.macro_results.copy()
        df = df.sort_values('ensemble_acc', ascending=False)

        # Format table
        display_df = df[['group', 'num_assets', 'samples', 'xgb_acc', 'lgb_acc', 'cat_acc', 'ensemble_acc']].copy()
        display_df.columns = ['Group', '# Assets', 'Samples', 'XGB', 'LGB', 'CAT', 'Ensemble']

        # Format percentages
        for col in ['XGB', 'LGB', 'CAT', 'Ensemble']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")

        print(display_df.to_string(index=False))

        print(f"\nBest Macro Model: {df.iloc[0]['group']} ({df.iloc[0]['ensemble_acc']:.1%})")

    def _display_recommendations(self):
        """Display actionable recommendations"""
        print("\n\n💡 RECOMMENDATIONS")
        print("-" * 100)

        if self.per_asset_results is None:
            print("No results available yet. Train models first.")
            return

        # Find best performers
        top_assets = self.per_asset_results.nlargest(3, 'ensemble_acc')

        print("\n✓ Use These Models for Trading:")
        for idx, row in top_assets.iterrows():
            print(f"   {row['asset']:10s} - {row['ensemble_acc']:.1%} accuracy ({row['name']})")

        # Find underperformers
        print("\n⚠️  Avoid These Models (Low Accuracy):")
        bottom_assets = self.per_asset_results.nsmallest(3, 'ensemble_acc')
        for idx, row in bottom_assets.iterrows():
            if row['ensemble_acc'] < 0.55:  # Below 55% is not much better than random
                print(f"   {row['asset']:10s} - {row['ensemble_acc']:.1%} accuracy (unreliable)")

        # Per-asset vs macro comparison
        if self.macro_results is not None:
            print("\n📊 Per-Asset vs Macro:")
            for _, macro_row in self.macro_results.iterrows():
                group_name = macro_row['group']
                macro_acc = macro_row['ensemble_acc']

                # Find assets in this group
                if 'equity' in group_name.lower():
                    asset_type = 'equity'
                elif 'crypto' in group_name.lower():
                    asset_type = 'crypto'
                else:
                    continue

                type_assets = self.per_asset_results[self.per_asset_results['type'] == asset_type]
                avg_per_asset = type_assets['ensemble_acc'].mean()

                comparison = "✓ Use Macro" if macro_acc > avg_per_asset else "✓ Use Per-Asset"
                print(f"   {group_name:25s}: Macro {macro_acc:.1%} vs Per-Asset Avg {avg_per_asset:.1%} → {comparison}")

    def export_html(self, output_path: str = "results/dashboard.html"):
        """Export dashboard to HTML file"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuroVest Results Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }}
                th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background: #f5f5f5; }}
                .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: white; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                .metric-label {{ font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>NeuroVest Results Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        # Add per-asset table
        if self.per_asset_results is not None:
            html += "<h2>Per-Asset Models</h2>"
            html += self.per_asset_results.to_html(index=False)

        # Add macro table
        if self.macro_results is not None:
            html += "<h2>Macro Models</h2>"
            html += self.macro_results.to_html(index=False)

        html += "</body></html>"

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(html)

        print(f"\n✓ Dashboard exported to: {output_file}")
        print(f"  Open in browser: file://{output_file.absolute()}")

    def get_best_model_for_asset(self, asset: str) -> dict:
        """Find best model for a specific asset"""
        # Check per-asset model
        per_asset_row = self.per_asset_results[self.per_asset_results['asset'] == asset]

        if per_asset_row.empty:
            return {"error": f"No model found for {asset}"}

        per_asset_acc = per_asset_row.iloc[0]['ensemble_acc']
        asset_type = per_asset_row.iloc[0]['type']

        # Check macro model
        macro_acc = None
        if self.macro_results is not None:
            # Find appropriate macro group
            for _, macro_row in self.macro_results.iterrows():
                group_name = macro_row['group']
                if asset_type in group_name.lower() or 'all' in group_name.lower():
                    macro_acc = macro_row['ensemble_acc']
                    macro_group = group_name
                    break

        # Recommendation
        if macro_acc and macro_acc > per_asset_acc:
            recommendation = "macro"
            accuracy = macro_acc
            model_name = macro_group
        else:
            recommendation = "per-asset"
            accuracy = per_asset_acc
            model_name = asset

        return {
            "asset": asset,
            "recommendation": recommendation,
            "model_name": model_name,
            "accuracy": accuracy,
            "per_asset_accuracy": per_asset_acc,
            "macro_accuracy": macro_acc
        }


def main():
    parser = argparse.ArgumentParser(description="NeuroVest Results Dashboard")
    parser.add_argument('--type', choices=['equity', 'bond', 'commodity', 'crypto'],
                        help="Filter by asset type")
    parser.add_argument('--top', type=int, help="Show only top N assets")
    parser.add_argument('--export', choices=['html'], help="Export to format")
    parser.add_argument('--asset', help="Get best model for specific asset")

    args = parser.parse_args()

    dashboard = ResultsDashboard()

    if args.asset:
        result = dashboard.get_best_model_for_asset(args.asset)
        print(json.dumps(result, indent=2))
    elif args.export == 'html':
        dashboard.display_all(asset_type=args.type, top_n=args.top)
        dashboard.export_html()
    else:
        dashboard.display_all(asset_type=args.type, top_n=args.top)


if __name__ == "__main__":
    main()
