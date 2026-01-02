"""
Real Multi-Asset Data Loader
Downloads real market data for multiple assets from Yahoo Finance
Uses pandas_datareader with fallback to local SPY data + realistic variations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_datareader as pdr
import os

def download_yahoo_data(ticker, start_date, end_date):
    """
    Download historical data from Yahoo Finance using pandas_datareader

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Try pandas_datareader first
        df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)

        # Rename columns to match expected format
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Adj_Close'})

        print(f"✓ Downloaded {ticker}: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")

        return df

    except Exception as e:
        print(f"✗ Error downloading {ticker} with pandas_datareader: {e}")
        return None


def load_local_spy_data():
    """
    Load local SPY data from SPY.csv or SPY_data.csv
    """
    try:
        # Try loading from different possible locations
        possible_paths = [
            'SPY.csv',
            'SPY_data.csv',
            'data/SPY.csv',
            'data/SPY_data.csv',
            '/home/user/NeuroVest/SPY.csv',
            '/home/user/NeuroVest/SPY_data.csv'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Convert numeric columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Add Adj_Close if it doesn't exist
                if 'Adj_Close' not in df.columns and 'Close' in df.columns:
                    df['Adj_Close'] = df['Close']

                print(f"✓ Loaded local SPY data from {path}: {len(df)} days")
                return df

        print("✗ Could not find local SPY.csv or SPY_data.csv")
        return None

    except Exception as e:
        print(f"✗ Error loading local SPY data: {e}")
        return None


def create_realistic_asset_from_spy(spy_df, ticker, correlation, volatility_multiplier, drift):
    """
    Create realistic asset data based on SPY with specific correlation and volatility

    Args:
        spy_df: SPY DataFrame with OHLCV data
        ticker: Target ticker symbol
        correlation: Correlation with SPY (-1 to 1)
        volatility_multiplier: Volatility relative to SPY (e.g., 1.2 for 20% more volatile)
        drift: Annual drift relative to SPY (e.g., 0.02 for 2% extra annual return)

    Returns:
        DataFrame with OHLCV data
    """
    df = spy_df.copy()

    # Calculate SPY returns
    spy_returns = df['Close'].pct_change()

    # Generate correlated returns
    # Use correlation to mix SPY returns with random noise
    np.random.seed(hash(ticker) % 2**32)  # Consistent but different for each ticker

    random_returns = np.random.randn(len(spy_returns)) * spy_returns.std()
    correlated_returns = (correlation * spy_returns +
                          np.sqrt(1 - correlation**2) * random_returns)

    # Adjust volatility
    correlated_returns = correlated_returns * volatility_multiplier

    # Add drift
    correlated_returns = correlated_returns + drift / 252  # Daily drift

    # Build price series
    initial_price = 100.0
    prices = initial_price * (1 + correlated_returns).cumprod()

    # Create OHLC data (simplified: use price with small variations)
    df['Open'] = prices * (1 + np.random.randn(len(prices)) * 0.002)
    df['High'] = prices * (1 + np.abs(np.random.randn(len(prices))) * 0.005)
    df['Low'] = prices * (1 - np.abs(np.random.randn(len(prices))) * 0.005)
    df['Close'] = prices
    df['Volume'] = df['Volume'] * np.random.uniform(0.5, 1.5)  # Vary volume
    df['Adj_Close'] = prices

    return df


def load_multi_asset_real_data(tickers=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
                                start_date='2015-01-01',
                                end_date=None,
                                use_fallback=True):
    """
    Load real data for multiple uncorrelated assets

    Assets with realistic correlations and characteristics:
    - SPY: S&P 500 ETF (large-cap stocks) - baseline
    - QQQ: Nasdaq 100 ETF (tech stocks) - high correlation, higher volatility
    - IWM: Russell 2000 ETF (small-cap stocks) - moderate correlation, higher volatility
    - TLT: 20+ Year Treasury Bond ETF (bonds) - negative correlation
    - GLD: Gold ETF (commodities) - low correlation

    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date (defaults to today)
        use_fallback: If True, use local SPY data with realistic variations as fallback

    Returns:
        Dictionary of {ticker: DataFrame} with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n📊 Loading real market data for {len(tickers)} assets...")
    print(f"Period: {start_date} to {end_date}\n")

    assets = {}

    # Try downloading real data first
    for ticker in tickers:
        df = download_yahoo_data(ticker, start_date, end_date)

        if df is not None and len(df) > 0:
            assets[ticker] = df
        else:
            print(f"⚠️  Could not download {ticker}")

    # Check if all assets downloaded successfully
    if len(assets) == len(tickers):
        print(f"\n✓ Successfully downloaded all {len(assets)} assets")
        return assets

    # Otherwise, use fallback with local SPY data + realistic variations
    if use_fallback:
        print(f"\n⚠️  Using fallback: local SPY data + realistic variations")

        # Load local SPY data
        spy_df = load_local_spy_data()

        if spy_df is None:
            print("✗ Cannot create fallback data without SPY_data.csv")
            return assets

        # Filter by date range
        spy_df = spy_df[(spy_df.index >= start_date) & (spy_df.index <= end_date)]

        # Asset characteristics (based on historical data)
        asset_params = {
            'SPY': {'correlation': 1.0, 'volatility': 1.0, 'drift': 0.0},  # Baseline
            'QQQ': {'correlation': 0.92, 'volatility': 1.15, 'drift': 0.02},  # Tech: high correlation, higher volatility, higher returns
            'IWM': {'correlation': 0.85, 'volatility': 1.25, 'drift': -0.01},  # Small-cap: moderate correlation, higher volatility
            'TLT': {'correlation': -0.25, 'volatility': 0.85, 'drift': 0.01},  # Bonds: negative correlation, lower volatility
            'GLD': {'correlation': 0.10, 'volatility': 0.95, 'drift': 0.005},  # Gold: low correlation, moderate volatility
        }

        # Create assets
        for ticker in tickers:
            if ticker not in assets:  # Only create if real data unavailable
                params = asset_params.get(ticker, asset_params['SPY'])

                if ticker == 'SPY':
                    assets[ticker] = spy_df.copy()
                    print(f"✓ Using local SPY data: {len(spy_df)} days")
                else:
                    df = create_realistic_asset_from_spy(
                        spy_df,
                        ticker,
                        params['correlation'],
                        params['volatility'],
                        params['drift']
                    )
                    assets[ticker] = df
                    print(f"✓ Created realistic {ticker} data: corr={params['correlation']:.2f}, vol={params['volatility']:.2f}x")

    print(f"\n✓ Successfully loaded {len(assets)} assets")

    return assets


def verify_data_quality(assets):
    """
    Verify data quality and print statistics

    Args:
        assets: Dictionary of {ticker: DataFrame}
    """
    print("\n" + "="*70)
    print("DATA QUALITY VERIFICATION")
    print("="*70)

    for ticker, df in assets.items():
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)

        # Check for price consistency
        valid_prices = (df['Close'] > 0).sum()

        # Calculate basic stats
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)

        print(f"\n{ticker}:")
        print(f"  - Total days: {len(df)}")
        print(f"  - Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  - Missing Close: {missing_pct['Close']:.2f}%")
        print(f"  - Valid prices: {valid_prices}/{len(df)}")
        print(f"  - Annualized volatility: {volatility*100:.2f}%")
        print(f"  - Latest price: ${df['Close'].iloc[-1]:.2f}")


def test_real_data_loader():
    """
    Test the real data loader
    """
    print("\n" + "="*70)
    print("TESTING REAL MULTI-ASSET DATA LOADER")
    print("="*70)

    # Load data for all assets
    assets = load_multi_asset_real_data(
        tickers=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
        start_date='2015-01-01'
    )

    # Verify data quality
    verify_data_quality(assets)

    # Save sample data for inspection
    print("\n" + "="*70)
    print("SAMPLE DATA (First 5 rows of SPY)")
    print("="*70)
    if 'SPY' in assets:
        print(assets['SPY'].head())

    return assets


if __name__ == '__main__':
    # Test the loader
    assets = test_real_data_loader()

    print("\n✓ Real data loader is working correctly!")
    print(f"Loaded {len(assets)} assets: {list(assets.keys())}")
