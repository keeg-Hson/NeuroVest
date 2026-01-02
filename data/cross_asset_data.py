#!/usr/bin/env python3
"""
Cross-Asset Data Fetcher

Downloads and manages cross-asset data for lead-lag analysis:
- HYG: High Yield Corporate Bonds (credit spreads)
- LQD: Investment Grade Corporate Bonds
- TLT: 20Y+ Treasury Bonds (flight to safety)
- GLD: Gold (safe haven)
- VIX: Volatility Index (fear gauge)
- Treasury Yields: 10Y, 5Y, 2Y (yield curve)

These assets LEAD equities by 1-4 weeks and provide critical signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def download_cross_asset_data(ticker, start_date="2000-01-01", force_refresh=False):
    """
    Download cross-asset data with caching

    Args:
        ticker: Ticker symbol (HYG, TLT, GLD, ^VIX, ^TNX, etc.)
        start_date: Start date for data
        force_refresh: Force re-download even if cached

    Returns:
        DataFrame with OHLCV data
    """
    cache_file = CACHE_DIR / f"{ticker.replace('^', '')}_daily.csv"

    # Use cache if available and not forcing refresh
    if cache_file.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"   ✅ Loaded {ticker} from cache ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"   ⚠️  Cache read failed for {ticker}: {e}")

    # Download using yfinance
    try:
        import yfinance as yf
        print(f"   Downloading {ticker}...")
        df = yf.download(ticker, start=start_date, progress=False)

        if not df.empty:
            # Save to cache
            df.to_csv(cache_file)
            print(f"   ✅ Downloaded {ticker} ({len(df)} rows)")
            return df
        else:
            print(f"   ❌ No data for {ticker}")
            return pd.DataFrame()

    except ImportError:
        print(f"   ❌ yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ Download failed for {ticker}: {e}")
        return pd.DataFrame()

def get_all_cross_asset_data(force_refresh=False):
    """
    Download all cross-asset instruments

    Returns:
        Dictionary of DataFrames
    """
    print("=" * 80)
    print("DOWNLOADING CROSS-ASSET DATA")
    print("=" * 80)

    tickers = {
        'SPY': 'S&P 500 ETF',
        'HYG': 'High Yield Corporate Bonds',
        'LQD': 'Investment Grade Corporate Bonds',
        'TLT': '20Y+ Treasury Bonds',
        'IEF': '7-10Y Treasury Bonds',
        'SHY': '1-3Y Treasury Bonds',
        'GLD': 'Gold',
        '^VIX': 'CBOE Volatility Index',
        '^TNX': '10-Year Treasury Yield',
        '^FVX': '5-Year Treasury Yield',
        'DX-Y.NYB': 'US Dollar Index',
    }

    data = {}

    for ticker, name in tickers.items():
        print(f"\n{ticker} ({name})")
        df = download_cross_asset_data(ticker, force_refresh=force_refresh)
        if not df.empty:
            data[ticker] = df

    print(f"\n✅ Downloaded {len(data)} instruments")
    return data

def calculate_credit_spread(hyg_data, lqd_data):
    """
    Calculate credit spread proxy from HYG vs LQD

    High yield spreads widen before equity selloffs (2-4 week lead)
    """
    hyg_close = hyg_data['Close'] if 'Close' in hyg_data.columns else hyg_data['Adj Close']
    lqd_close = lqd_data['Close'] if 'Close' in lqd_data.columns else lqd_data['Adj Close']

    # Ratio of HYG to LQD (lower = credit stress)
    credit_ratio = hyg_close / lqd_close

    # Percentage change over various windows
    credit_spread = pd.DataFrame({
        'Credit_Ratio': credit_ratio,
        'Credit_Change_5d': credit_ratio.pct_change(5),
        'Credit_Change_20d': credit_ratio.pct_change(20),
        'Credit_Stress': (credit_ratio < credit_ratio.rolling(60).quantile(0.25)).astype(int),
        'Credit_Spread_MA20': credit_ratio.rolling(20).mean(),
        'Credit_Spread_Volatility': credit_ratio.pct_change().rolling(20).std()
    })

    return credit_spread

def calculate_yield_curve(tnx_data, fvx_data):
    """
    Calculate yield curve (10Y - 5Y)

    Inversion predicts recession 12-18 months ahead
    """
    tnx = tnx_data['Close'] if 'Close' in tnx_data.columns else tnx_data['Adj Close']
    fvx = fvx_data['Close'] if 'Close' in fvx_data.columns else fvx_data['Adj Close']

    # Yield curve spread
    yield_spread = tnx - fvx

    yield_curve = pd.DataFrame({
        '10Y_Yield': tnx,
        '5Y_Yield': fvx,
        'Yield_Curve': yield_spread,
        'Yield_Curve_Inverted': (yield_spread < 0).astype(int),
        'Yield_Curve_Steepness': yield_spread.rolling(20).mean(),
        'Yield_Curve_Change': yield_spread.diff(20),
        'Yield_10Y_Change': tnx.diff(20),
    })

    return yield_curve

def calculate_vix_features(vix_data):
    """
    Calculate VIX-based fear indicators

    VIX spikes lead equity selloffs by 1-2 weeks
    """
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data['Adj Close']

    vix_features = pd.DataFrame({
        'VIX': vix,
        'VIX_Change_5d': vix.pct_change(5),
        'VIX_Spike': (vix > vix.rolling(20).mean() * 1.5).astype(int),
        'VIX_Percentile_252': vix.rolling(252).rank(pct=True),
        'VIX_MA20': vix.rolling(20).mean(),
        'High_VIX_Regime': (vix > 25).astype(int),
        'VIX_Trend': (vix > vix.rolling(50).mean()).astype(int),
    })

    return vix_features

def calculate_stock_bond_correlation(spy_data, tlt_data, window=60):
    """
    Calculate rolling stock-bond correlation

    Negative correlation = flight to safety = bearish for stocks
    """
    spy_ret = spy_data['Close'].pct_change()
    tlt_ret = tlt_data['Close'].pct_change()

    corr_features = pd.DataFrame({
        'Stock_Bond_Corr_60': spy_ret.rolling(window).corr(tlt_ret),
        'Stock_Bond_Corr_20': spy_ret.rolling(20).corr(tlt_ret),
        'Flight_To_Safety': (spy_ret.rolling(window).corr(tlt_ret) < -0.3).astype(int),
    })

    return corr_features

def calculate_gold_features(spy_data, gld_data):
    """
    Calculate gold safe-haven indicators

    Gold outperformance signals crisis/uncertainty
    """
    spy_ret = spy_data['Close'].pct_change()
    gld_ret = gld_data['Close'].pct_change()

    gold_features = pd.DataFrame({
        'Gold_Outperformance': gld_ret - spy_ret,
        'Gold_Outperformance_20d': (gld_ret - spy_ret).rolling(20).mean(),
        'Gold_Safe_Haven': ((gld_ret > 0) & (spy_ret < 0)).astype(int),
        'Gold_Safe_Haven_Freq': ((gld_ret > 0) & (spy_ret < 0)).astype(int).rolling(20).mean(),
        'Gold_Trend': (gld_data['Close'] > gld_data['Close'].rolling(50).mean()).astype(int),
    })

    return gold_features

def add_lead_lag_features(features_df, lead_days=[5, 10, 20]):
    """
    Add lagged versions of features (these assets LEAD equities)

    Args:
        features_df: DataFrame with cross-asset features
        lead_days: List of lead periods to create

    Returns:
        DataFrame with lagged features
    """
    result = features_df.copy()

    # Features that are known to lead equities
    leading_features = [
        'Credit_Change_5d', 'Credit_Change_20d', 'Credit_Stress',
        'VIX', 'VIX_Spike', 'High_VIX_Regime',
        'Yield_Curve', 'Yield_Curve_Inverted',
        'Stock_Bond_Corr_60', 'Flight_To_Safety',
    ]

    for feat in leading_features:
        if feat in features_df.columns:
            for lag in lead_days:
                result[f'{feat}_lead{lag}'] = features_df[feat].shift(lag)

    return result

def create_cross_asset_features(data, spy_data):
    """
    Create all cross-asset features

    Args:
        data: Dictionary of cross-asset DataFrames
        spy_data: SPY DataFrame

    Returns:
        DataFrame with all cross-asset features
    """
    print("\n" + "=" * 80)
    print("CREATING CROSS-ASSET FEATURES")
    print("=" * 80)

    features = pd.DataFrame(index=spy_data.index)

    # Credit spreads
    if 'HYG' in data and 'LQD' in data:
        print("\n✅ Credit Spread Features")
        credit = calculate_credit_spread(data['HYG'], data['LQD'])
        features = features.join(credit, how='left')

    # Yield curve
    if '^TNX' in data and '^FVX' in data:
        print("✅ Yield Curve Features")
        yield_curve = calculate_yield_curve(data['^TNX'], data['^FVX'])
        features = features.join(yield_curve, how='left')

    # VIX features
    if '^VIX' in data:
        print("✅ VIX Features")
        vix_feat = calculate_vix_features(data['^VIX'])
        features = features.join(vix_feat, how='left')

    # Stock-bond correlation
    if 'TLT' in data:
        print("✅ Stock-Bond Correlation Features")
        corr_feat = calculate_stock_bond_correlation(spy_data, data['TLT'])
        features = features.join(corr_feat, how='left')

    # Gold features
    if 'GLD' in data:
        print("✅ Gold Safe Haven Features")
        gold_feat = calculate_gold_features(spy_data, data['GLD'])
        features = features.join(gold_feat, how='left')

    # Add lead-lag features (these markets LEAD equities)
    print("\n✅ Adding Lead-Lag Features (5, 10, 20 day leads)")
    features = add_lead_lag_features(features, lead_days=[5, 10, 20])

    print(f"\n✅ Total cross-asset features created: {len(features.columns)}")

    return features

if __name__ == "__main__":
    # Test the module
    print("Testing cross-asset data module...")

    # Download data
    data = get_all_cross_asset_data()

    # Load SPY
    if 'SPY' in data:
        spy = data['SPY']

        # Create features
        features = create_cross_asset_features(data, spy)

        # Save
        output_path = CACHE_DIR / "cross_asset_features.csv"
        features.to_csv(output_path)
        print(f"\n💾 Saved to: {output_path}")
        print(f"   Shape: {features.shape}")
        print(f"\nFeature preview:")
        print(features.tail())

    print("\n✅ Cross-asset data module test complete!")
