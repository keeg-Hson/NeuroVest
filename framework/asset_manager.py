#!/usr/bin/env python3
"""
Asset Manager - Central hub for managing all tradeable assets

This module reads config/assets.yaml and provides:
- List of all enabled assets
- Asset metadata (thresholds, categories, etc.)
- Macro group definitions
- Helper functions for data organization
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Asset:
    """Asset metadata"""
    ticker: str
    name: str
    category: str
    threshold: float
    asset_type: str  # 'equity', 'bond', 'commodity', 'crypto'
    group: str  # which group it belongs to
    exchange: Optional[str] = None  # for crypto
    enabled: bool = True


class AssetManager:
    """
    Manages all assets from configuration file

    Usage:
        manager = AssetManager()

        # Get all enabled assets
        assets = manager.get_all_assets()

        # Get specific groups
        equities = manager.get_assets_by_type('equity')
        cryptos = manager.get_assets_by_type('crypto')

        # Get macro groups
        all_eq = manager.get_macro_group('all_equities')
    """

    def __init__(self, config_path: str = "config/assets.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.assets = self._parse_assets()
        self.settings = self.config.get('settings', {})

    def _load_config(self) -> Dict:
        """Load YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _parse_assets(self) -> List[Asset]:
        """Parse all assets from config"""
        assets = []

        # Asset type mappings
        type_mappings = {
            'equity_major_indices': 'equity',
            'equity_international': 'equity',
            'equity_sectors': 'equity',
            'equity_style': 'equity',
            'equity_thematic': 'equity',
            'bonds': 'bond',
            'commodities': 'commodity',
            'crypto': 'crypto',
        }

        for group_name, asset_type in type_mappings.items():
            group_data = self.config.get(group_name, {})

            for ticker, metadata in group_data.items():
                assets.append(Asset(
                    ticker=ticker,
                    name=metadata['name'],
                    category=metadata['category'],
                    threshold=metadata['threshold'],
                    asset_type=asset_type,
                    group=group_name,
                    exchange=metadata.get('exchange'),
                    enabled=metadata.get('enabled', True)
                ))

        return assets

    def get_all_assets(self, enabled_only: bool = True) -> List[Asset]:
        """Get all assets"""
        if enabled_only:
            return [a for a in self.assets if a.enabled]
        return self.assets

    def get_assets_by_type(self, asset_type: str, enabled_only: bool = True) -> List[Asset]:
        """Get assets by type (equity, bond, commodity, crypto)"""
        assets = [a for a in self.assets if a.asset_type == asset_type]
        if enabled_only:
            assets = [a for a in assets if a.enabled]
        return assets

    def get_assets_by_group(self, group_name: str, enabled_only: bool = True) -> List[Asset]:
        """Get assets by group (equity_major_indices, etc.)"""
        assets = [a for a in self.assets if a.group == group_name]
        if enabled_only:
            assets = [a for a in assets if a.enabled]
        return assets

    def get_asset(self, ticker: str) -> Optional[Asset]:
        """Get single asset by ticker"""
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        return None

    def get_macro_group(self, group_name: str) -> List[Asset]:
        """
        Get assets for a macro model group

        Example: get_macro_group('all_equities') returns all equity assets
        """
        macro_config = self.config.get('macro_groups', {}).get(group_name)
        if not macro_config:
            raise ValueError(f"Macro group not found: {group_name}")

        if not macro_config.get('enabled', True):
            return []

        # Collect all assets from included groups
        assets = []
        for included_group in macro_config['includes']:
            assets.extend(self.get_assets_by_group(included_group))

        return assets

    def get_macro_groups(self, enabled_only: bool = True) -> Dict[str, List[Asset]]:
        """Get all macro groups"""
        groups = {}
        for group_name, config in self.config.get('macro_groups', {}).items():
            if enabled_only and not config.get('enabled', True):
                continue
            groups[group_name] = self.get_macro_group(group_name)
        return groups

    def get_settings(self) -> Dict:
        """Get global settings"""
        return self.settings

    def get_ticker_list(self, asset_type: Optional[str] = None) -> List[str]:
        """Get list of ticker symbols"""
        if asset_type:
            assets = self.get_assets_by_type(asset_type)
        else:
            assets = self.get_all_assets()
        return [a.ticker for a in assets]

    def print_summary(self):
        """Print summary of all configured assets"""
        print("=" * 80)
        print("ASSET MANAGER SUMMARY")
        print("=" * 80)

        # Count by type
        type_counts = {}
        for asset in self.get_all_assets():
            type_counts[asset.asset_type] = type_counts.get(asset.asset_type, 0) + 1

        print(f"\n📊 Total Assets: {len(self.get_all_assets())}")
        for asset_type, count in sorted(type_counts.items()):
            print(f"   {asset_type.capitalize()}: {count}")

        # Macro groups
        print(f"\n📦 Macro Groups:")
        for group_name, assets in self.get_macro_groups().items():
            print(f"   {group_name}: {len(assets)} assets")

        # Settings
        print(f"\n⚙️  Settings:")
        print(f"   Start Date: {self.settings.get('start_date')}")
        print(f"   Models: {', '.join(self.settings.get('models', []))}")
        print(f"   API Enabled: {self.settings.get('api_enabled')}")

        print("=" * 80)


if __name__ == "__main__":
    # Test the asset manager
    manager = AssetManager()
    manager.print_summary()

    # Example usage
    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)

    print(f"\n✓ Equity ETFs: {len(manager.get_assets_by_type('equity'))}")
    print(f"✓ Cryptocurrencies: {len(manager.get_assets_by_type('crypto'))}")
    print(f"✓ Bonds: {len(manager.get_assets_by_type('bond'))}")

    print(f"\n✓ All Equities Macro Group: {len(manager.get_macro_group('all_equities'))} assets")
    print(f"✓ Crypto Macro Group: {len(manager.get_macro_group('all_crypto'))} assets")

    # Show sample asset
    spy = manager.get_asset('SPY')
    if spy:
        print(f"\n✓ SPY Details:")
        print(f"   Name: {spy.name}")
        print(f"   Category: {spy.category}")
        print(f"   Threshold: {spy.threshold * 100}%")
