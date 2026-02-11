"""
Feature Registry - SINGULAR SOURCE OF TRUTH

This module defines ALL features used across the NeuroVest system.
Any component (dashboard, training, prediction, analysis) should import
features from here to ensure consistency.

Usage:
    from core.feature_registry import FeatureRegistry

    # Get production features
    features = FeatureRegistry.get_production_features()

    # Get features by category
    macro_features = FeatureRegistry.get_macro_features()

    # Get all available features
    all_features = FeatureRegistry.get_all_features()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum


class FeatureCategory(Enum):
    """Feature categories for organization and filtering"""
    CORE_TECHNICAL = "core_technical"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    RETURNS = "returns"
    CROSS_ASSET = "cross_asset"
    MACRO = "macro"
    REGIME = "regime"
    INTERACTIONS = "interactions"
    EXPERIMENTAL = "experimental"


@dataclass
class FeatureDefinition:
    """Definition of a single feature"""
    name: str
    category: FeatureCategory
    description: str = ""
    importance_score: float = 0.0  # From historical analysis
    is_production: bool = True  # Whether actively used in production
    requires_external: bool = False  # Needs external data (macro CSV, etc.)
    lookback_days: int = 0  # Warmup period needed


class FeatureRegistry:
    """
    Central registry for all features in NeuroVest.

    This is the SINGULAR SOURCE OF TRUTH for feature definitions.
    All components should reference this registry.
    """

    # =========================================================================
    # CORE TECHNICAL FEATURES (proven high importance)
    # =========================================================================
    CORE_TECHNICAL = [
        "MA_20",
        "EMA_12",
        "EMA_26",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Width",
        "BB_PctB",
        "ATR_14",
        "Stoch_K",
        "Stoch_D",
        "RSI",
        "RSI_Delta",
        "VWAP_Dev",
        "KC_Width",
        "ADX",
        "Plus_DI",
        "Minus_DI",
    ]

    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================
    MOMENTUM = [
        "Price_Momentum_10",
        "ZMomentum",
        "Acceleration",
        "RSI_Momentum_5",
        "RSI_ROC_5",
        "Return_Momentum_Ratio",
        "Return_Acceleration",
    ]

    # =========================================================================
    # VOLATILITY FEATURES (BB_Width is #1 feature)
    # =========================================================================
    VOLATILITY = [
        "Volatility",
        "Rolling_STD_5",
        "Vol_Percentile",
        "Volatility_Acceleration",
        "BB_Width_Mean_10",
        "BB_Width_Std_10",
        "BB_Width_ZScore",
        "BB_Width_Lag1",
        "BB_Width_Lag3",
        "BB_Width_Change",
        "Vol_Percentile_252",
        "High_Volatility",
        "Ret_Skew_20",
        "Ret_Kurt_20",
    ]

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    VOLUME = [
        "OBV",
        "Vol_Ratio",
        "Volume_per_ATR",
        "Volume_Momentum_5",
        "OBV_Change_5",
        "OBV_Trend",
    ]

    # =========================================================================
    # RETURN/LAG FEATURES (highest importance category)
    # =========================================================================
    RETURNS = [
        "Daily_Return",
        "Return_Lag1",
        "Return_Lag3",
        "Return_Lag5",
        "Return_Lag7",
        "Return_Lag10",
        "Return_Lag15",
        "RSI_Lag_1",
        "RSI_Lag_3",
        "RSI_Lag_5",
        "RSI_Lag7",
        "RSI_Lag10",
        "Return_Lag1_MA5",
        "Return_Lag1_MA10",
        "Return_Volatility_20",
        "Return_Skew_10",
        "Return_Kurt_10",
        "Positive_Return_Streak",
        "Return_1d_vs_10d",
        "Return_3d_vs_10d",
        "Returns_50d",
        "Volatility_50d",
        "RSI_50",
    ]

    # =========================================================================
    # CROSS-ASSET FEATURES (from cross_asset_features.csv)
    # =========================================================================
    CROSS_ASSET = [
        # Credit (HYG/LQD spread)
        "Credit_Ratio",
        "Credit_Change_20d",
        "Credit_Stress",
        # Treasury yields (10Y)
        "Yield_10Y",
        "Yield_Change_20d",
        "High_Yield_Regime",
        # Dollar (DXY)
        "DXY_Level",
        "DXY_Change_20d",
        "Strong_Dollar",
        # Volatility (VIX/realized)
        "Realized_Vol_20",
        "Realized_Vol_60",
        "High_Vol_Regime",
        "Vol_Spike",
    ]

    # =========================================================================
    # MACRO FEATURES (from macro_features.csv)
    # =========================================================================
    MACRO = [
        # Interest rates (continuous - preferred over binary)
        "Macro_10Y_Yield",
        "Rate_Change_3m",
        "Rate_Change_6m",
        # Policy cycle
        "Tightening_Cycle",
        "Easing_Cycle",
        # Economic signals
        "Recession_Signal",
        "Recovery_Signal",
        # Inflation
        "Inflation_Proxy",
        # Financial conditions
        "Financial_Stress",
    ]

    # =========================================================================
    # ADDITIONAL MACRO FEATURES (available but not in production)
    # These can be tested via analyze_features.py
    # =========================================================================
    MACRO_EXTENDED = [
        # Rate regime classifications
        "Macro_Rate_Volatility",
        "Macro_Growth_Momentum_3m",
        "Macro_Growth_Momentum_6m",
        "Macro_Growth_Momentum_1y",
        # Business cycle
        "Macro_Cycle_Position",
        # Financial conditions detailed
        "Macro_Financial_Conditions_Tight",
        "Macro_Financial_Conditions_Loose",
        # Policy risk indicators
        "Macro_Policy_Error_Risk",
        "Macro_Complacency",
    ]

    # =========================================================================
    # REGIME DETECTION FEATURES
    # =========================================================================
    REGIME = [
        # Trend/Bull-Bear
        "MA_200",
        "Price_vs_MA200",
        "MA200_Slope",
        "MA200_Distance_Vol",
        "Near_52w_High",
        "Days_Above_MA20",
        "Trend_Consistency",
        "Regime_Score",
        # VIX-based
        "VIX_Percentile",
        "VIX_Change",
        # Volatility regime
        "Vol_Regime_Low",
        "Vol_Regime_Med",
        "Vol_Regime_High",
        "Vol_Regime_Change",
        "Vol_Mean_Reversion",
        # Risk regime
        "Risk_On_Score",
        "Risk_On_Regime",
        # Trend regime
        "Regime_MeanRevert",
        "Regime_Trending",
        "Regime_StrongTrend",
        "Trend_Direction",
        "ADX_Slope",
        "Trend_Strengthening",
        # Stress indicators
        "Market_Stress_Index",
        "High_Stress_Regime",
        # Cross-asset regimes
        "Dollar_Strengthening",
        "Yields_Rising",
        "Credit_Tightening",
    ]

    # =========================================================================
    # FEATURE INTERACTIONS (regime-dependent relationships)
    # =========================================================================
    INTERACTIONS = [
        # BB_Width interactions (top feature)
        "BB_Width_x_RSI",
        "BB_Width_x_Return_Lag1",
        "BB_Width_x_Vol_Ratio",
        # Return interactions
        "Return_Lag1_x_Return_Lag3",
        "Return_Trend_Strength",
        # RSI interactions
        "RSI_x_Vol_Ratio",
        "RSI_x_Volatility",
        # Volume interactions
        "OBV_x_Return_Lag1",
        "Volume_x_Returns",
        "Volume_x_Volatility",
        # MACD
        "MACD_x_RSI",
        "MACD_divergence",
        # Position x Volatility
        "Near_52w_High_x_Volatility",
        "Near_52w_High_x_KC_Width",
        "Stoch_K_x_Volatility",
        "Return_Lag3_x_Volatility",
        "Return_Lag5_x_ATR",
        "Near_52w_High_x_Return_Lag3",
        "BB_PctB_x_Stoch_K",
        # Cross-asset interactions
        "Credit_Ratio_x_Volatility",
        "Realized_Vol_60_x_Volatility",
        "Rate_Change_3m_x_MA200_Slope",
        "DXY_Level_x_Return_Lag5",
        "Yield_10Y_x_Price_vs_MA200",
        # Regime interactions
        "Trend_x_RiskOn",
        "HighVol_x_Trending",
        "Stress_x_Downtrend",
        "MeanRevert_Opportunity",
        # Trend strength
        "Trend_strength_10",
        "Trend_strength_20",
        "Trend_strength_50",
    ]

    # =========================================================================
    # FEATURES TO EXCLUDE (zero importance or harmful)
    # =========================================================================
    EXCLUDED_FEATURES = [
        # Binary regime indicators (zero importance)
        "RSI_Overbought",
        "RSI_Oversold",
        "RSI_Neutral",
        "Bull_Market",
        "High_Fear",
        "VIX_Spike",
        "MA_20_50_Cross",
        "Pct_Above_MA20",
        "Near_52w_Low",
        "Strong_Trend",
        "Vol_Expanding",
        "Return_Reversal",
        # Macro binary (prefer continuous versions)
        "Low_Rate_Regime",
        "High_Rate_Regime",
        "High_Inflation",
        "Expansion",
        "Contraction",
        # Sentiment (mostly empty)
        "Sent_x_Vol",
        "RSI_x_NewsZ",
        "RSI_x_RedditZ",
        "News_Sent_Z20",
        "Reddit_Sent_Z20",
        # Lead features (future leakage risk)
        "XAsset_High_Vol_lead5",
        "XAsset_Credit_Stress_lead5",
        # Redundant
        "XAsset_Vol_Spike",
        # Temporal (trees handle raw better)
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "Month_sin",
        "Month_cos",
        "Quarter",
    ]

    @classmethod
    def get_production_features(cls) -> List[str]:
        """
        Get the PRODUCTION feature list used for training and prediction.
        This is the primary feature set for the model.
        """
        features = []
        features.extend(cls.CORE_TECHNICAL)
        features.extend(cls.MOMENTUM)
        features.extend(cls.VOLATILITY)
        features.extend(cls.VOLUME)
        features.extend(cls.RETURNS)
        features.extend(cls.CROSS_ASSET)
        features.extend(cls.MACRO)
        features.extend(cls.REGIME)
        features.extend(cls.INTERACTIONS)

        # Remove duplicates while preserving order
        seen = set()
        unique_features = []
        for f in features:
            if f not in seen and f not in cls.EXCLUDED_FEATURES:
                seen.add(f)
                unique_features.append(f)

        return unique_features

    @classmethod
    def get_all_features(cls) -> List[str]:
        """Get ALL features including experimental ones"""
        features = cls.get_production_features()
        features.extend(cls.MACRO_EXTENDED)

        # Remove duplicates
        seen = set()
        unique_features = []
        for f in features:
            if f not in seen:
                seen.add(f)
                unique_features.append(f)

        return unique_features

    @classmethod
    def get_features_by_category(cls, category: FeatureCategory) -> List[str]:
        """Get features for a specific category"""
        mapping = {
            FeatureCategory.CORE_TECHNICAL: cls.CORE_TECHNICAL,
            FeatureCategory.MOMENTUM: cls.MOMENTUM,
            FeatureCategory.VOLATILITY: cls.VOLATILITY,
            FeatureCategory.VOLUME: cls.VOLUME,
            FeatureCategory.RETURNS: cls.RETURNS,
            FeatureCategory.CROSS_ASSET: cls.CROSS_ASSET,
            FeatureCategory.MACRO: cls.MACRO,
            FeatureCategory.REGIME: cls.REGIME,
            FeatureCategory.INTERACTIONS: cls.INTERACTIONS,
        }
        return mapping.get(category, [])

    @classmethod
    def get_cross_asset_features(cls) -> List[str]:
        """Get cross-asset features (for external data loading)"""
        return cls.CROSS_ASSET.copy()

    @classmethod
    def get_macro_features(cls) -> List[str]:
        """Get macro features"""
        return cls.MACRO.copy()

    @classmethod
    def get_macro_extended_features(cls) -> List[str]:
        """Get extended macro features for testing"""
        return cls.MACRO + cls.MACRO_EXTENDED

    @classmethod
    def get_excluded_features(cls) -> List[str]:
        """Get features that should be excluded"""
        return cls.EXCLUDED_FEATURES.copy()

    @classmethod
    def validate_features(cls, features: List[str]) -> Dict[str, List[str]]:
        """
        Validate a list of features against the registry.

        Returns:
            Dict with 'valid', 'excluded', 'unknown' lists
        """
        all_known = set(cls.get_all_features())
        excluded = set(cls.EXCLUDED_FEATURES)

        result = {
            'valid': [],
            'excluded': [],
            'unknown': [],
        }

        for f in features:
            if f in excluded:
                result['excluded'].append(f)
            elif f in all_known:
                result['valid'].append(f)
            else:
                result['unknown'].append(f)

        return result

    @classmethod
    def get_feature_count(cls) -> Dict[str, int]:
        """Get count of features by category"""
        return {
            'core_technical': len(cls.CORE_TECHNICAL),
            'momentum': len(cls.MOMENTUM),
            'volatility': len(cls.VOLATILITY),
            'volume': len(cls.VOLUME),
            'returns': len(cls.RETURNS),
            'cross_asset': len(cls.CROSS_ASSET),
            'macro': len(cls.MACRO),
            'regime': len(cls.REGIME),
            'interactions': len(cls.INTERACTIONS),
            'total_production': len(cls.get_production_features()),
            'total_all': len(cls.get_all_features()),
            'excluded': len(cls.EXCLUDED_FEATURES),
        }


# =========================================================================
# CROSS-ASSET FEATURE MAPPING (CSV column name -> internal name)
# =========================================================================
CROSS_ASSET_COLUMN_MAP = {
    'XAsset_Credit_Ratio': 'Credit_Ratio',
    'XAsset_Credit_Change_20d': 'Credit_Change_20d',
    'XAsset_Credit_Stress': 'Credit_Stress',
    'XAsset_10Y_Yield': 'Yield_10Y',
    'XAsset_10Y_Change_20d': 'Yield_Change_20d',
    'XAsset_High_Yield_Regime': 'High_Yield_Regime',
    'XAsset_DXY': 'DXY_Level',
    'XAsset_DXY_Change_20d': 'DXY_Change_20d',
    'XAsset_Strong_Dollar': 'Strong_Dollar',
    'XAsset_Realized_Vol_20': 'Realized_Vol_20',
    'XAsset_Realized_Vol_60': 'Realized_Vol_60',
    'XAsset_High_Vol_Regime': 'High_Vol_Regime',
    'XAsset_Vol_Spike': 'Vol_Spike',
}

# =========================================================================
# MACRO FEATURE MAPPING (CSV column name -> internal name)
# =========================================================================
MACRO_COLUMN_MAP = {
    'Macro_10Y_Yield': 'Macro_10Y_Yield',
    'Macro_Rate_Change_3m': 'Rate_Change_3m',
    'Macro_Rate_Change_6m': 'Rate_Change_6m',
    'Macro_Tightening_Cycle': 'Tightening_Cycle',
    'Macro_Easing_Cycle': 'Easing_Cycle',
    'Macro_Recession_Signal': 'Recession_Signal',
    'Macro_Recovery_Signal': 'Recovery_Signal',
    'Macro_Inflation_Proxy': 'Inflation_Proxy',
    'Macro_Financial_Stress': 'Financial_Stress',
    # Extended features
    'Macro_Rate_Volatility': 'Macro_Rate_Volatility',
    'Macro_Growth_Momentum_3m': 'Macro_Growth_Momentum_3m',
    'Macro_Growth_Momentum_6m': 'Macro_Growth_Momentum_6m',
    'Macro_Growth_Momentum_1y': 'Macro_Growth_Momentum_1y',
    'Macro_Cycle_Position': 'Macro_Cycle_Position',
    'Macro_Financial_Conditions_Tight': 'Macro_Financial_Conditions_Tight',
    'Macro_Financial_Conditions_Loose': 'Macro_Financial_Conditions_Loose',
    'Macro_Policy_Error_Risk': 'Macro_Policy_Error_Risk',
    'Macro_Complacency': 'Macro_Complacency',
}


def print_feature_summary():
    """Print a summary of the feature registry"""
    counts = FeatureRegistry.get_feature_count()

    print("=" * 60)
    print("FEATURE REGISTRY SUMMARY")
    print("=" * 60)
    print(f"\nProduction Features: {counts['total_production']}")
    print(f"All Features (incl. experimental): {counts['total_all']}")
    print(f"Excluded Features: {counts['excluded']}")
    print("\nBy Category:")
    print(f"  Core Technical: {counts['core_technical']}")
    print(f"  Momentum:       {counts['momentum']}")
    print(f"  Volatility:     {counts['volatility']}")
    print(f"  Volume:         {counts['volume']}")
    print(f"  Returns:        {counts['returns']}")
    print(f"  Cross-Asset:    {counts['cross_asset']}")
    print(f"  Macro:          {counts['macro']}")
    print(f"  Regime:         {counts['regime']}")
    print(f"  Interactions:   {counts['interactions']}")
    print("=" * 60)


if __name__ == "__main__":
    print_feature_summary()

    print("\n\nProduction Features:")
    for i, f in enumerate(FeatureRegistry.get_production_features(), 1):
        print(f"  {i:3d}. {f}")
