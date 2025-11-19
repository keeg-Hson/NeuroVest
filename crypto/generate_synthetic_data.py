"""
Generate synthetic cryptocurrency data for backtesting

Creates realistic crypto price data with appropriate volatility and trends
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_crypto_prices(
    initial_price,
    num_days,
    annual_drift=0.50,  # 50% annual return (crypto avg)
    annual_vol=1.20,    # 120% annualized volatility (crypto avg)
    trend_strength=0.3,
    crash_prob=0.005    # 0.5% chance of flash crash per day
):
    """
    Generate synthetic crypto prices with realistic characteristics

    Args:
        initial_price: Starting price
        num_days: Number of days to generate
        annual_drift: Expected annual return
        annual_vol: Annualized volatility
        trend_strength: Momentum/trend persistence (0-1)
        crash_prob: Probability of flash crash

    Returns:
        DataFrame with OHLCV data
    """
    # Convert to daily parameters
    daily_drift = annual_drift / 365
    daily_vol = annual_vol / np.sqrt(365)

    # Generate returns with momentum
    returns = []
    momentum = 0

    for i in range(num_days):
        # Random shock
        shock = np.random.randn() * daily_vol

        # Add momentum (trend persistence)
        if i > 0:
            momentum = trend_strength * momentum + (1 - trend_strength) * shock
        else:
            momentum = shock

        # Add drift
        daily_return = daily_drift + momentum

        # Random flash crashes (crypto characteristic)
        if np.random.rand() < crash_prob:
            daily_return -= abs(np.random.randn() * 0.15)  # -15% crash

        # Random pumps (crypto characteristic)
        if np.random.rand() < crash_prob:
            daily_return += abs(np.random.randn() * 0.10)  # +10% pump

        returns.append(daily_return)

    # Calculate prices
    returns = np.array(returns)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    dates = pd.date_range(end=datetime.now(), periods=num_days, freq='D')

    df = pd.DataFrame(index=dates)
    df.index.name = 'Date'

    df['Close'] = prices

    # Generate daily high/low with realistic spreads
    daily_range_pct = abs(np.random.randn(num_days)) * daily_vol * 2
    df['High'] = df['Close'] * (1 + daily_range_pct / 2)
    df['Low'] = df['Close'] * (1 - daily_range_pct / 2)

    # Open is close of previous day +/- small gap
    df['Open'] = df['Close'].shift(1) * (1 + np.random.randn(num_days) * daily_vol * 0.3)
    df['Open'].iloc[0] = initial_price

    # Ensure OHLC constraints
    df['High'] = df[['High', 'Close', 'Open']].max(axis=1)
    df['Low'] = df[['Low', 'Close', 'Open']].min(axis=1)

    # Volume (crypto volumes vary widely)
    base_volume = 1000000  # Base daily volume
    volume_multiplier = np.exp(abs(np.random.randn(num_days)) * 0.5)  # Log-normal distribution
    df['Volume'] = base_volume * volume_multiplier

    # Higher volume on big moves
    df['Volume'] = df['Volume'] * (1 + abs(returns) * 10)

    # Adj_Close same as Close for crypto
    df['Adj_Close'] = df['Close']

    return df


def generate_correlated_crypto(base_df, correlation=0.75, vol_multiplier=1.2):
    """
    Generate correlated crypto asset

    Args:
        base_df: Base cryptocurrency DataFrame (e.g., BTC)
        correlation: Correlation to base asset
        vol_multiplier: Volatility multiplier

    Returns:
        DataFrame with OHLCV data
    """
    base_returns = base_df['Close'].pct_change()

    # Generate correlated returns
    random_noise = np.random.randn(len(base_returns)) * base_returns.std()
    correlated_returns = (
        correlation * base_returns +
        np.sqrt(1 - correlation**2) * random_noise
    ) * vol_multiplier

    # Calculate prices
    initial_price = base_df['Close'].iloc[0] * np.random.uniform(0.1, 2.0)
    prices = initial_price * (1 + correlated_returns).cumprod()

    # Create DataFrame
    df = pd.DataFrame(index=base_df.index)
    df['Close'] = prices

    # Generate OHLC
    daily_range = abs(correlated_returns) * 2
    df['High'] = df['Close'] * (1 + daily_range / 2)
    df['Low'] = df['Close'] * (1 - daily_range / 2)
    df['Open'] = df['Close'].shift(1).fillna(initial_price)

    # Ensure OHLC constraints
    df['High'] = df[['High', 'Close', 'Open']].max(axis=1)
    df['Low'] = df[['Low', 'Close', 'Open']].min(axis=1)

    # Volume
    df['Volume'] = base_df['Volume'] * np.random.uniform(0.5, 1.5, len(df))
    df['Adj_Close'] = df['Close']

    return df


def main():
    """
    Generate synthetic crypto data for backtesting
    """
    print("=" * 70)
    print("GENERATING SYNTHETIC CRYPTO DATA")
    print("=" * 70)

    output_dir = Path('../data_cache')
    output_dir.mkdir(exist_ok=True)

    # Bitcoin (base asset)
    print("\nGenerating BTC/USDT...")
    btc = generate_crypto_prices(
        initial_price=10000,  # Start from $10k
        num_days=365 * 3,     # 3 years
        annual_drift=0.50,    # 50% annual return
        annual_vol=1.20,      # 120% volatility
        trend_strength=0.4    # Strong trends
    )

    btc_file = output_dir / 'BTC_USDT_1d.csv'
    btc.to_csv(btc_file)
    print(f"  ✓ Saved to {btc_file}")
    print(f"    Price range: ${btc['Close'].min():,.2f} - ${btc['Close'].max():,.2f}")
    print(f"    Final price: ${btc['Close'].iloc[-1]:,.2f}")
    print(f"    Total return: {(btc['Close'].iloc[-1] / btc['Close'].iloc[0] - 1) * 100:.1f}%")

    # Ethereum (correlated to BTC)
    print("\nGenerating ETH/USDT...")
    eth = generate_correlated_crypto(btc, correlation=0.85, vol_multiplier=1.3)
    eth_file = output_dir / 'ETH_USDT_1d.csv'
    eth.to_csv(eth_file)
    print(f"  ✓ Saved to {eth_file}")
    print(f"    Final price: ${eth['Close'].iloc[-1]:,.2f}")

    # Solana (moderate correlation)
    print("\nGenerating SOL/USDT...")
    sol = generate_correlated_crypto(btc, correlation=0.70, vol_multiplier=1.8)
    sol_file = output_dir / 'SOL_USDT_1d.csv'
    sol.to_csv(sol_file)
    print(f"  ✓ Saved to {sol_file}")
    print(f"    Final price: ${sol['Close'].iloc[-1]:,.2f}")

    # Avalanche (moderate correlation)
    print("\nGenerating AVAX/USDT...")
    avax = generate_correlated_crypto(btc, correlation=0.68, vol_multiplier=1.9)
    avax_file = output_dir / 'AVAX_USDT_1d.csv'
    avax.to_csv(avax_file)
    print(f"  ✓ Saved to {avax_file}")
    print(f"    Final price: ${avax['Close'].iloc[-1]:,.2f}")

    # Polygon (lower correlation)
    print("\nGenerating MATIC/USDT...")
    matic = generate_correlated_crypto(btc, correlation=0.65, vol_multiplier=2.0)
    matic_file = output_dir / 'MATIC_USDT_1d.csv'
    matic.to_csv(matic_file)
    print(f"  ✓ Saved to {matic_file}")
    print(f"    Final price: ${matic['Close'].iloc[-1]:,.2f}")

    # Show correlations
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)

    assets = {
        'BTC': btc['Close'],
        'ETH': eth['Close'],
        'SOL': sol['Close'],
        'AVAX': avax['Close'],
        'MATIC': matic['Close']
    }

    prices_df = pd.DataFrame(assets)
    corr_matrix = prices_df.pct_change().corr()
    print(corr_matrix.round(2))

    print("\n" + "=" * 70)
    print("✓ Synthetic crypto data generated successfully")
    print("=" * 70)


if __name__ == '__main__':
    main()
