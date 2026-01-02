"""
Robust data loader for stock market data

Handles multiple data sources with fallback mechanism:
1. Local CSV files (primary)
2. Online APIs (when available)

Supports both single-asset and multi-asset loading
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Unified data loader for stock market data
    """

    def __init__(self, data_dir='./'):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)

    def load_csv(self, ticker, filename=None):
        """
        Load data from local CSV file

        Args:
            ticker: Ticker symbol (e.g., 'SPY')
            filename: Optional custom filename, defaults to {ticker}.csv

        Returns:
            DataFrame with OHLCV data
        """
        if filename is None:
            filename = f"{ticker}.csv"

        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Standardize column names
        df = self._standardize_columns(df)

        # Set date index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # Ensure numeric types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by date
        df.sort_index(inplace=True)

        # Add Adj_Close if missing
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']

        return df

    def _standardize_columns(self, df):
        """
        Standardize column names across different CSV formats
        """
        # Common column name mappings
        column_map = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj_Close',
            'adj_close': 'Adj_Close',
            'adjusted close': 'Adj_Close'
        }

        # Rename columns (case-insensitive)
        df.columns = [column_map.get(col.lower(), col) for col in df.columns]

        return df

    def generate_correlated_asset(self, base_df, ticker, correlation,
                                   volatility_multiplier, drift):
        """
        Generate synthetic asset data with specified correlation to base asset

        Args:
            base_df: Base DataFrame (e.g., SPY)
            ticker: New ticker symbol
            correlation: Target correlation coefficient (-1 to 1)
            volatility_multiplier: Volatility scaling factor
            drift: Annual drift rate (e.g., 0.08 for 8%)

        Returns:
            DataFrame with synthetic OHLCV data
        """
        df = base_df.copy()

        # Calculate base returns
        base_returns = df['Close'].pct_change()

        # Generate correlated returns
        # correlated_returns = correlation * base_returns + sqrt(1 - correlation^2) * random_noise
        random_noise = np.random.randn(len(base_returns)) * base_returns.std()

        correlated_returns = (
            correlation * base_returns +
            np.sqrt(1 - correlation**2) * random_noise
        ) * volatility_multiplier

        # Add drift
        daily_drift = drift / 252  # Annualized to daily
        correlated_returns = correlated_returns + daily_drift

        # Calculate prices
        initial_price = df['Close'].iloc[0]
        prices = initial_price * (1 + correlated_returns).cumprod()

        # Create new DataFrame
        new_df = pd.DataFrame(index=df.index)
        new_df['Close'] = prices

        # Generate OHLC from Close
        # High: Close + random up to 1% above
        # Low: Close - random up to 1% below
        # Open: Previous close + small random move

        daily_range = prices * 0.01  # 1% typical range

        new_df['High'] = prices + np.random.uniform(0, 1, len(prices)) * daily_range
        new_df['Low'] = prices - np.random.uniform(0, 1, len(prices)) * daily_range
        new_df['Open'] = prices.shift(1).fillna(prices.iloc[0]) + np.random.normal(0, daily_range * 0.5)

        # Ensure OHLC constraints
        new_df['High'] = new_df[['High', 'Close', 'Open']].max(axis=1)
        new_df['Low'] = new_df[['Low', 'Close', 'Open']].min(axis=1)

        # Volume: Scale from base asset with some randomness
        new_df['Volume'] = (df['Volume'] * np.random.uniform(0.7, 1.3, len(df))).astype(int)

        # Adj_Close same as Close for synthetic data
        new_df['Adj_Close'] = new_df['Close']

        return new_df

    def load_multi_asset(self, tickers_config, start_date=None, end_date=None):
        """
        Load multiple assets with fallback to synthetic data

        Args:
            tickers_config: Dict mapping ticker to config
                {
                    'SPY': {'file': 'SPY.csv'},
                    'QQQ': {'correlation': 0.92, 'volatility': 1.15, 'drift': 0.10},
                    ...
                }
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dict mapping ticker to DataFrame
        """
        assets = {}
        base_df = None

        # Load base asset first (usually SPY)
        for ticker, config in tickers_config.items():
            if 'file' in config:
                try:
                    df = self.load_csv(ticker, config.get('file'))

                    # Apply date filters
                    if start_date:
                        df = df[df.index >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df.index <= pd.to_datetime(end_date)]

                    assets[ticker] = df
                    if base_df is None:
                        base_df = df

                    print(f"✓ Loaded {ticker}: {len(df)} days")

                except Exception as e:
                    print(f"✗ Error loading {ticker}: {e}")

        # Generate synthetic assets
        if base_df is not None:
            for ticker, config in tickers_config.items():
                if 'correlation' in config and ticker not in assets:
                    try:
                        df = self.generate_correlated_asset(
                            base_df,
                            ticker,
                            config['correlation'],
                            config['volatility'],
                            config['drift']
                        )

                        assets[ticker] = df
                        print(f"✓ Generated {ticker}: corr={config['correlation']:.2f}, "
                              f"vol={config['volatility']:.2f}x")

                    except Exception as e:
                        print(f"✗ Error generating {ticker}: {e}")

        return assets

    def validate_data(self, df, ticker):
        """
        Validate data quality

        Returns:
            tuple: (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check for excessive NaN values
        nan_pct = df[required_cols].isna().sum().sum() / (len(df) * len(required_cols))
        if nan_pct > 0.1:  # More than 10% NaN
            return False, f"Excessive NaN values: {nan_pct:.1%}"

        # Check OHLC constraints
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Close'])
        ).sum()

        if invalid_ohlc > 0:
            return False, f"Invalid OHLC constraints: {invalid_ohlc} rows"

        return True, "Data is valid"


def get_default_multi_asset_config():
    """
    Get default configuration for multi-asset portfolio

    Returns:
        Dict with ticker configurations
    """
    return {
        'SPY': {
            'file': 'SPY.csv',
            'name': 'S&P 500 ETF',
            'asset_class': 'Large Cap Equity'
        },
        'QQQ': {
            'correlation': 0.92,
            'volatility': 1.15,
            'drift': 0.11,
            'name': 'NASDAQ 100 ETF',
            'asset_class': 'Tech Equity'
        },
        'IWM': {
            'correlation': 0.85,
            'volatility': 1.25,
            'drift': 0.09,
            'name': 'Russell 2000 ETF',
            'asset_class': 'Small Cap Equity'
        },
        'TLT': {
            'correlation': -0.25,
            'volatility': 0.85,
            'drift': 0.04,
            'name': '20+ Year Treasury ETF',
            'asset_class': 'Bonds (Hedge)'
        },
        'GLD': {
            'correlation': 0.10,
            'volatility': 0.95,
            'drift': 0.06,
            'name': 'Gold ETF',
            'asset_class': 'Precious Metals'
        }
    }


def get_expanded_asset_config():
    """
    Get expanded configuration with precious metals and commodities

    Returns:
        Dict with ticker configurations for comprehensive portfolio
    """
    return {
        # Equities
        'SPY': {
            'file': 'SPY.csv',
            'name': 'S&P 500 ETF',
            'asset_class': 'Large Cap Equity'
        },
        'QQQ': {
            'correlation': 0.92,
            'volatility': 1.15,
            'drift': 0.11,
            'name': 'NASDAQ 100 ETF',
            'asset_class': 'Tech Equity'
        },
        'IWM': {
            'correlation': 0.85,
            'volatility': 1.25,
            'drift': 0.09,
            'name': 'Russell 2000 ETF',
            'asset_class': 'Small Cap Equity'
        },

        # Bonds
        'TLT': {
            'correlation': -0.25,
            'volatility': 0.85,
            'drift': 0.04,
            'name': '20+ Year Treasury ETF',
            'asset_class': 'Bonds'
        },

        # Precious Metals
        'GLD': {
            'correlation': 0.10,
            'volatility': 0.95,
            'drift': 0.06,
            'name': 'Gold ETF',
            'asset_class': 'Precious Metals - Gold'
        },
        'SLV': {
            'correlation': 0.15,
            'volatility': 1.30,
            'drift': 0.04,
            'name': 'Silver ETF',
            'asset_class': 'Precious Metals - Silver'
        },
        'GDX': {
            'correlation': 0.65,
            'volatility': 1.60,
            'drift': 0.08,
            'name': 'Gold Miners ETF',
            'asset_class': 'Precious Metals - Miners'
        },
        'PPLT': {
            'correlation': 0.08,
            'volatility': 1.40,
            'drift': 0.05,
            'name': 'Platinum ETF',
            'asset_class': 'Precious Metals - Platinum'
        },
        'PALL': {
            'correlation': 0.12,
            'volatility': 1.50,
            'drift': 0.06,
            'name': 'Palladium ETF',
            'asset_class': 'Precious Metals - Palladium'
        },

        # Commodities
        'USO': {
            'correlation': 0.20,
            'volatility': 1.80,
            'drift': 0.03,
            'name': 'Oil ETF',
            'asset_class': 'Commodities - Energy'
        },
        'DBA': {
            'correlation': 0.05,
            'volatility': 1.20,
            'drift': 0.04,
            'name': 'Agriculture ETF',
            'asset_class': 'Commodities - Agriculture'
        }
    }


if __name__ == '__main__':
    # Test the data loader
    print("Testing DataLoader...")
    print("=" * 70)

    loader = DataLoader()

    # Test single asset load
    print("\n1. Loading single asset (SPY)...")
    spy = loader.load_csv('SPY')
    print(f"   Loaded {len(spy)} days")
    print(f"   Date range: {spy.index.min()} to {spy.index.max()}")
    print(f"   Columns: {list(spy.columns)}")

    # Validate
    is_valid, msg = loader.validate_data(spy, 'SPY')
    print(f"   Validation: {msg}")

    # Test multi-asset load
    print("\n2. Loading multi-asset portfolio...")
    config = get_default_multi_asset_config()
    assets = loader.load_multi_asset(config, start_date='2015-01-01')

    print(f"\n   Loaded {len(assets)} assets:")
    for ticker, df in assets.items():
        print(f"   - {ticker}: {len(df)} days, "
              f"{df.index.min().date()} to {df.index.max().date()}")

    # Calculate correlations
    if len(assets) > 1:
        print("\n3. Correlation matrix:")
        close_prices = pd.DataFrame({
            ticker: df['Close'] for ticker, df in assets.items()
        })
        corr_matrix = close_prices.pct_change().corr()
        print(corr_matrix.round(2))

    print("\n" + "=" * 70)
    print("✓ DataLoader test complete")
