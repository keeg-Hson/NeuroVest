#!/usr/bin/env python3
"""
Asset-Aware Feature Selection

Provides intelligent feature filtering based on asset type to improve
model accuracy by excluding irrelevant features.

Key insights:
- VIX features are only relevant for US equity markets
- Sector features are only meaningful for equity ETFs
- Credit spreads apply mainly to equity and bond markets
- Macro features (Fed rates) affect all US-traded assets differently
"""

from typing import List, Optional

# Features that should ONLY be used for specific asset types
EQUITY_ONLY_FEATURES = {
    # VIX-related features (US equity fear gauge)
    "VIX",
    "VIX_Percentile",
    "High_Fear",
    "VIX_Spike",
    "VIX_Change",

    # Sector-specific features
    "Sector_MedianRet_20",
    "Sector_Dispersion_20",

    # S&P 500 specific
    "MA_20_50_Cross",
    "Pct_Above_MA20",
}

EQUITY_AND_BOND_FEATURES = {
    # Credit features (corporate bond spreads)
    "Credit_Ratio",
    "Credit_Change_20d",
    "Credit_Stress",
    "Credit_Spread_20",
    "Credit_Ratio_x_Volatility",

    # Treasury yield features
    "Yield_10Y",
    "Yield_Change_20d",
    "High_Yield_Regime",
    "TNX_Change_20",
    "Yield_10Y_x_Price_vs_MA200",

    # Macro features (primarily affect US markets)
    "Macro_10Y_Yield",
    "Rate_Change_3m",
    "Rate_Change_6m",
    "Tightening_Cycle",
    "Easing_Cycle",
    "Recession_Signal",
    "Recovery_Signal",
    "Rate_Change_3m_x_MA200_Slope",
}

# Features that work well for crypto
CRYPTO_ENHANCED_FEATURES = {
    # Momentum features are highly important for crypto
    "Return_Lag1",
    "Return_Lag3",
    "Return_Lag5",
    "Return_Lag7",
    "Return_Lag10",
    "Return_Lag15",
    "RSI",
    "RSI_Delta",
    "RSI_Momentum_5",
    "RSI_ROC_5",
    "Stoch_K",
    "Stoch_D",

    # Volatility features (crypto is highly volatile)
    "BB_Width",
    "BB_Width_x_RSI",
    "BB_Width_x_Return_Lag1",
    "BB_Width_ZScore",
    "Volatility",
    "Volatility_Acceleration",
    "Vol_Percentile",
    "ATR_14",

    # Volume features (important for crypto)
    "OBV",
    "Vol_Ratio",
    "Volume_Momentum_5",
    "OBV_Change_5",
    "Volume_per_ATR",
}


def get_asset_type(ticker: str) -> str:
    """
    Determine asset type from ticker symbol.

    Returns:
        One of: 'crypto', 'equity', 'bond', 'commodity', 'unknown'
    """
    ticker = ticker.upper()

    # Crypto patterns
    if '/USDT' in ticker or '/USD' in ticker or '/BTC' in ticker:
        return 'crypto'
    if ticker in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC']:
        return 'crypto'

    # Bond ETFs
    bond_tickers = {'TLT', 'IEF', 'SHY', 'BND', 'AGG', 'VCIT', 'VCSH', 'LQD', 'HYG', 'JNK'}
    if ticker in bond_tickers:
        return 'bond'

    # Commodity ETFs
    commodity_tickers = {'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'PDBC', 'DBC'}
    if ticker in commodity_tickers:
        return 'commodity'

    # Default to equity for US stocks/ETFs
    return 'equity'


def filter_features_for_asset(
    feature_cols: List[str],
    ticker: str,
    verbose: bool = False
) -> List[str]:
    """
    Filter feature list based on asset type to remove irrelevant features.

    This improves model accuracy by:
    1. Removing features that would be NaN/filled with 0 for the asset
    2. Keeping only economically meaningful features for the asset type

    Args:
        feature_cols: Original list of features
        ticker: Asset ticker symbol
        verbose: Print filtering info

    Returns:
        Filtered list of features appropriate for the asset type
    """
    asset_type = get_asset_type(ticker)

    filtered = []
    removed = []

    for feature in feature_cols:
        should_include = True

        # Apply asset-type specific filtering
        if asset_type == 'crypto':
            # Remove equity-only features for crypto
            if feature in EQUITY_ONLY_FEATURES:
                should_include = False
            # Remove equity/bond specific features
            elif feature in EQUITY_AND_BOND_FEATURES:
                should_include = False

        elif asset_type == 'commodity':
            # Commodities don't have VIX or sectors
            if feature in EQUITY_ONLY_FEATURES:
                should_include = False

        elif asset_type == 'bond':
            # Bonds don't have VIX or sectors (but keep credit/yield features)
            if feature in EQUITY_ONLY_FEATURES:
                should_include = False

        if should_include:
            filtered.append(feature)
        else:
            removed.append(feature)

    if verbose and removed:
        print(f"\n[Asset Features] Filtering for {ticker} ({asset_type}):")
        print(f"  Removed {len(removed)} irrelevant features:")
        for f in removed[:10]:  # Show first 10
            print(f"    - {f}")
        if len(removed) > 10:
            print(f"    ... and {len(removed) - 10} more")
        print(f"  Keeping {len(filtered)} features")

    return filtered


def get_optimal_features_for_asset(ticker: str) -> Optional[List[str]]:
    """
    Get a curated list of optimal features for a specific asset type.

    This can be used instead of filter_features_for_asset when you want
    to use a hand-picked set of features known to work well.

    Returns:
        List of feature names, or None to use default filtered features
    """
    asset_type = get_asset_type(ticker)

    # For now, return None to use the filtered default features
    # In the future, this could return asset-specific optimized feature sets
    # based on feature importance analysis for each asset type
    return None


def get_feature_importance_by_type(asset_type: str) -> dict:
    """
    Get expected feature importance weights by asset type.

    This can be used to weight features during training or to
    prioritize which features to keep when reducing dimensionality.

    Returns:
        Dict mapping feature categories to importance weights
    """
    if asset_type == 'crypto':
        return {
            'momentum': 1.5,      # Very important for crypto
            'volatility': 1.3,   # High volatility = important
            'volume': 1.2,       # Volume spikes matter
            'trend': 1.0,        # Standard
            'macro': 0.3,        # Less relevant
            'sentiment': 0.5,    # Social sentiment matters but noisy
        }
    elif asset_type == 'bond':
        return {
            'momentum': 0.8,     # Less important for bonds
            'volatility': 1.0,   # Standard
            'volume': 0.7,       # Less liquid
            'trend': 1.2,        # Trends matter more
            'macro': 1.5,        # Very important (rates!)
            'credit': 1.5,       # Credit spreads crucial
        }
    elif asset_type == 'commodity':
        return {
            'momentum': 1.2,     # Important for commodities
            'volatility': 1.3,   # Volatile assets
            'volume': 1.0,       # Standard
            'trend': 1.1,        # Trend following works
            'macro': 1.0,        # Dollar strength, rates
        }
    else:  # equity
        return {
            'momentum': 1.0,
            'volatility': 1.2,
            'volume': 1.0,
            'trend': 1.0,
            'macro': 1.1,
            'sentiment': 0.8,
            'vix': 1.3,          # VIX very important for equity
        }


def validate_features_for_asset(
    df,
    feature_cols: List[str],
    ticker: str,
    nan_threshold: float = 0.5
) -> List[str]:
    """
    Validate that features have meaningful data for an asset.

    Removes features that are mostly NaN for this specific asset,
    which indicates the feature isn't relevant or available.

    Args:
        df: DataFrame with feature data
        feature_cols: List of features to validate
        ticker: Asset ticker
        nan_threshold: Remove features with more than this fraction of NaN

    Returns:
        List of validated features
    """
    validated = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        nan_ratio = df[col].isna().sum() / len(df)

        if nan_ratio <= nan_threshold:
            validated.append(col)
        else:
            # Feature has too many NaNs, likely not relevant
            pass

    return validated


if __name__ == "__main__":
    # Test the feature filtering
    from utils import get_feature_list

    all_features = get_feature_list()
    print(f"Total features: {len(all_features)}")

    test_assets = ['SPY', 'BTC/USDT', 'TLT', 'GLD']

    for ticker in test_assets:
        filtered = filter_features_for_asset(all_features, ticker, verbose=True)
        print(f"\n{ticker} ({get_asset_type(ticker)}): {len(filtered)}/{len(all_features)} features")
