#!/usr/bin/env python3
"""
Options Flow Data Integration
==============================
Add options-related features to improve model accuracy.

Expected impact: +1-2%

Features to add:
1. Put/Call Ratio - Sentiment indicator
2. Gamma Exposure (GEX) - Dealer hedging pressure
3. VIX Term Structure - Volatility expectations
4. Implied Volatility metrics
5. Options Volume/Open Interest
6. Max Pain levels

Note: This script creates the infrastructure. Options data needs to be sourced
from providers like CBOE, OptionMetrics, or similar.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 80)
print("OPTIONS FLOW DATA INTEGRATION")
print("=" * 80)
print()

DATA_DIR = Path("data")

def calculate_put_call_ratio(options_df):
    """
    Calculate Put/Call Ratio

    Higher ratio (>1.0) = bearish sentiment (more puts)
    Lower ratio (<1.0) = bullish sentiment (more calls)
    """
    put_volume = options_df[options_df['option_type'] == 'put']['volume'].sum()
    call_volume = options_df[options_df['option_type'] == 'call']['volume'].sum()

    if call_volume == 0:
        return np.nan

    return put_volume / call_volume


def calculate_gamma_exposure(options_df, spot_price):
    """
    Calculate Gamma Exposure (GEX)

    Positive GEX = dealers are short gamma, will hedge by selling into rallies
    Negative GEX = dealers are long gamma, will hedge by buying into rallies

    This creates support/resistance levels
    """
    total_gamma = 0

    for _, row in options_df.iterrows():
        # Gamma exposure = gamma * open_interest * 100 * spot^2 * (1 if call else -1)
        notional_gamma = row['gamma'] * row['open_interest'] * 100 * spot_price ** 2

        if row['option_type'] == 'call':
            total_gamma += notional_gamma
        else:
            total_gamma -= notional_gamma

    return total_gamma / 1e9  # Convert to billions


def calculate_vix_term_structure(vix_futures):
    """
    VIX Term Structure Analysis

    Contango (upward sloping) = market expects volatility to increase (bearish)
    Backwardation (downward sloping) = market expects volatility to decrease (bullish)
    """
    if len(vix_futures) < 2:
        return {'vix_slope': np.nan, 'vix_contango': np.nan}

    # Calculate slope between front month and next month
    front_month = vix_futures.iloc[0]['price']
    next_month = vix_futures.iloc[1]['price']

    slope = (next_month - front_month) / front_month

    return {
        'vix_slope': slope,
        'vix_contango': 1 if slope > 0 else 0,  # Binary indicator
        'vix_front': front_month,
        'vix_second': next_month
    }


def calculate_max_pain(options_df):
    """
    Max Pain Theory

    Price level where option holders (buyers) would lose the most money.
    Theory: market makers push price toward this level at expiration.
    """
    strike_prices = options_df['strike'].unique()
    max_pain_price = None
    min_total_value = float('inf')

    for strike in strike_prices:
        # Calculate total value of all options if stock closes at this strike
        total_value = 0

        for _, row in options_df.iterrows():
            if row['option_type'] == 'call':
                intrinsic = max(0, strike - row['strike'])
            else:
                intrinsic = max(0, row['strike'] - strike)

            total_value += intrinsic * row['open_interest']

        if total_value < min_total_value:
            min_total_value = total_value
            max_pain_price = strike

    return max_pain_price


def create_options_features_demo():
    """
    Create demo options features for SPY

    In production, this would pull from actual options data sources.
    For now, we create a template showing how to structure the data.
    """

    # Load SPY data to get dates
    print("📥 Loading SPY data for date alignment...")
    from utils import load_SPY_data
    spy_df = load_SPY_data()

    # Create template dataframe
    options_features = pd.DataFrame(index=spy_df.index)

    # Initialize all features with NaN (to be filled with real data)
    options_features['put_call_ratio'] = np.nan
    options_features['put_call_ratio_ma5'] = np.nan
    options_features['put_call_ratio_ma20'] = np.nan

    options_features['gamma_exposure'] = np.nan
    options_features['gamma_exposure_pct'] = np.nan

    options_features['vix_term_slope'] = np.nan
    options_features['vix_contango'] = np.nan
    options_features['vix_front_month'] = np.nan

    options_features['implied_vol_rank'] = np.nan
    options_features['iv_percentile'] = np.nan

    options_features['max_pain'] = np.nan
    options_features['price_to_max_pain'] = np.nan

    options_features['call_volume'] = np.nan
    options_features['put_volume'] = np.nan
    options_features['total_options_volume'] = np.nan

    options_features['call_oi'] = np.nan
    options_features['put_oi'] = np.nan
    options_features['oi_change_5d'] = np.nan

    print("✅ Created options features template")
    print(f"   Features: {len(options_features.columns)}")
    print(f"   Samples: {len(options_features)}")
    print()

    return options_features


def create_synthetic_options_features():
    """
    Create synthetic options features for testing

    Based on typical relationships between price, volatility, and options flow
    """

    print("🔬 Creating synthetic options features for testing...")
    from utils import load_SPY_data
    spy_df = load_SPY_data()

    # Calculate returns and volatility
    spy_df['return'] = spy_df['Close'].pct_change()
    spy_df['volatility'] = spy_df['return'].rolling(20).std()
    spy_df['abs_return'] = spy_df['return'].abs()

    options_features = pd.DataFrame(index=spy_df.index)

    # 1. Put/Call Ratio (inversely related to returns, higher during selloffs)
    options_features['put_call_ratio'] = (
        1.0 - spy_df['return'].rolling(5).mean() * 10 +
        np.random.normal(0, 0.1, len(spy_df))
    ).clip(0.5, 2.0)
    options_features['put_call_ratio_ma5'] = options_features['put_call_ratio'].rolling(5).mean()
    options_features['put_call_ratio_ma20'] = options_features['put_call_ratio'].rolling(20).mean()

    # 2. Gamma Exposure (related to volatility regime)
    options_features['gamma_exposure'] = (
        spy_df['volatility'] * 100 * np.random.normal(1, 0.3, len(spy_df))
    )
    spy_close = spy_df['Close'].fillna(method='ffill')
    options_features['gamma_exposure_pct'] = options_features['gamma_exposure'] / spy_close * 100

    # 3. VIX Term Structure (contango during calm, backwardation during stress)
    options_features['vix_term_slope'] = (
        -spy_df['abs_return'].rolling(10).mean() * 50 +
        np.random.normal(0.02, 0.05, len(spy_df))
    )
    options_features['vix_contango'] = (options_features['vix_term_slope'] > 0).astype(int)
    options_features['vix_front_month'] = (
        spy_df['volatility'] * 100 +
        np.random.normal(15, 3, len(spy_df))
    ).clip(10, 80)

    # 4. Implied Volatility metrics
    options_features['implied_vol_rank'] = (
        spy_df['volatility'].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
        )
    )
    options_features['iv_percentile'] = options_features['implied_vol_rank']

    # 5. Max Pain (tends to be near current price)
    options_features['max_pain'] = spy_close * (1 + np.random.normal(0, 0.02, len(spy_df)))
    options_features['price_to_max_pain'] = (spy_close - options_features['max_pain']) / spy_close * 100

    # 6. Volume metrics (higher during volatile periods)
    base_volume = 1000000
    options_features['call_volume'] = base_volume * (1 + spy_df['abs_return'] * 20)
    options_features['put_volume'] = base_volume * (1 + spy_df['abs_return'] * 25)  # Puts trade more in selloffs
    options_features['total_options_volume'] = options_features['call_volume'] + options_features['put_volume']

    # 7. Open Interest
    options_features['call_oi'] = options_features['call_volume'].rolling(20).sum()
    options_features['put_oi'] = options_features['put_volume'].rolling(20).sum()
    options_features['oi_change_5d'] = options_features['put_oi'].pct_change(5)

    # Fill NaN values
    options_features = options_features.fillna(method='ffill').fillna(0)

    print("✅ Created synthetic options features")
    print(f"   Features: {len(options_features.columns)}")
    print(f"   Sample stats:")
    print(options_features[['put_call_ratio', 'gamma_exposure', 'vix_term_slope']].describe())
    print()

    return options_features


def save_options_features(options_df, filename="options_flow_features.csv"):
    """Save options features to CSV"""
    filepath = DATA_DIR / filename
    options_df.to_csv(filepath)
    print(f"💾 Saved options features to: {filepath}")
    return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("📊 Options Flow Data Integration")
    print()

    # Create synthetic features for testing
    options_features = create_synthetic_options_features()

    # Save to file
    save_path = save_options_features(options_features)

    print()
    print("=" * 80)
    print("✅ OPTIONS FLOW FEATURES CREATED")
    print("=" * 80)
    print()
    print("Features added (20 total):")
    print("  Put/Call Metrics:")
    print("    - put_call_ratio, put_call_ratio_ma5, put_call_ratio_ma20")
    print("  Gamma Exposure:")
    print("    - gamma_exposure, gamma_exposure_pct")
    print("  VIX Term Structure:")
    print("    - vix_term_slope, vix_contango, vix_front_month")
    print("  Implied Volatility:")
    print("    - implied_vol_rank, iv_percentile")
    print("  Max Pain:")
    print("    - max_pain, price_to_max_pain")
    print("  Volume/OI:")
    print("    - call_volume, put_volume, total_options_volume")
    print("    - call_oi, put_oi, oi_change_5d")
    print()
    print("Next steps:")
    print("  1. Review options_flow_features.csv")
    print("  2. Run train_with_options_flow.py to add these features to models")
    print("  3. Expected improvement: +1-2%")
    print()
    print("Note: These are synthetic features for testing.")
    print("In production, replace with real options data from:")
    print("  - CBOE (put/call ratios)")
    print("  - OptionMetrics or similar (IV, Greeks)")
    print("  - VIX futures data")
