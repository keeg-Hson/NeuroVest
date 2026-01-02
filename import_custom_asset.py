#!/usr/bin/env python3
"""
Custom Asset Import Tool

Import custom assets from CSV/Excel files for evaluation with NeuroVest.

Supported formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- TSV files (.tsv)

Expected columns (flexible naming):
- Date/Time/Timestamp
- Open
- High
- Low
- Close
- Volume (optional)

Usage:
    python3 import_custom_asset.py my_stock.csv MYSTK
    python3 import_custom_asset.py portfolio.xlsx CUSTOM --sheet "Daily Prices"
    python3 import_custom_asset.py --validate my_data.csv
    python3 import_custom_asset.py --list
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

# Column name mappings (flexible detection)
DATE_COLUMNS = ['date', 'time', 'timestamp', 'datetime', 'day', 'period']
OPEN_COLUMNS = ['open', 'o', 'open_price', 'opening', 'first']
HIGH_COLUMNS = ['high', 'h', 'high_price', 'max', 'maximum']
LOW_COLUMNS = ['low', 'l', 'low_price', 'min', 'minimum']
CLOSE_COLUMNS = ['close', 'c', 'close_price', 'closing', 'last', 'price', 'adj close', 'adjusted close']
VOLUME_COLUMNS = ['volume', 'vol', 'v', 'shares', 'quantity', 'amount']


def detect_column(df_columns, possible_names):
    """Detect column from possible names (case-insensitive)"""
    df_columns_lower = [c.lower().strip() for c in df_columns]

    for name in possible_names:
        for i, col in enumerate(df_columns_lower):
            if name in col or col in name:
                return df_columns[i]
    return None


def read_file(filepath, sheet_name=None):
    """Read data file (CSV or Excel)"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.suffix.lower()

    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.tsv':
        df = pd.read_csv(filepath, sep='\t')
    elif ext in ['.xlsx', '.xls']:
        if sheet_name:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return df


def standardize_dataframe(df):
    """Standardize DataFrame to expected format"""
    # Detect columns
    date_col = detect_column(df.columns, DATE_COLUMNS)
    open_col = detect_column(df.columns, OPEN_COLUMNS)
    high_col = detect_column(df.columns, HIGH_COLUMNS)
    low_col = detect_column(df.columns, LOW_COLUMNS)
    close_col = detect_column(df.columns, CLOSE_COLUMNS)
    volume_col = detect_column(df.columns, VOLUME_COLUMNS)

    # Check required columns
    if not date_col:
        raise ValueError("Could not detect Date column. Expected: " + ", ".join(DATE_COLUMNS))
    if not close_col:
        raise ValueError("Could not detect Close column. Expected: " + ", ".join(CLOSE_COLUMNS))

    # Build standardized DataFrame
    result = pd.DataFrame()

    # Parse dates
    result['Date'] = pd.to_datetime(df[date_col], errors='coerce')

    # If we don't have OHLC, create from Close
    if open_col:
        result['Open'] = pd.to_numeric(df[open_col], errors='coerce')
    else:
        result['Open'] = pd.to_numeric(df[close_col], errors='coerce')

    if high_col:
        result['High'] = pd.to_numeric(df[high_col], errors='coerce')
    else:
        result['High'] = result['Open']

    if low_col:
        result['Low'] = pd.to_numeric(df[low_col], errors='coerce')
    else:
        result['Low'] = result['Open']

    result['Close'] = pd.to_numeric(df[close_col], errors='coerce')
    result['Adj Close'] = result['Close']

    if volume_col:
        result['Volume'] = pd.to_numeric(df[volume_col], errors='coerce').fillna(0)
    else:
        # Generate synthetic volume based on price movement
        result['Volume'] = 1000000  # Default volume

    # Clean up
    result = result.dropna(subset=['Date', 'Close'])
    result = result.sort_values('Date')
    result = result.drop_duplicates(subset=['Date'], keep='last')
    result = result.reset_index(drop=True)

    return result


def validate_data(df):
    """Validate imported data quality"""
    issues = []
    warnings = []

    # Check minimum rows
    if len(df) < 100:
        warnings.append(f"Only {len(df)} rows - may not be enough for reliable predictions")

    if len(df) < 20:
        issues.append(f"Only {len(df)} rows - need at least 20 for basic analysis")

    # Check for missing dates (gaps)
    df_sorted = df.sort_values('Date')
    date_diff = df_sorted['Date'].diff().dropna()
    large_gaps = date_diff[date_diff > pd.Timedelta(days=5)]

    if len(large_gaps) > 0:
        warnings.append(f"Found {len(large_gaps)} gaps > 5 days in data")

    # Check price validity
    if (df['Close'] <= 0).any():
        issues.append("Found zero or negative prices")

    if (df['High'] < df['Low']).any():
        issues.append("Found High < Low on some days")

    # Check for suspicious data
    daily_returns = df['Close'].pct_change().dropna()

    if (daily_returns.abs() > 0.5).any():
        warnings.append("Found daily returns > 50% - verify data accuracy")

    # Check volume
    if (df['Volume'] == 0).all():
        warnings.append("All volume values are zero - volume-based features will be limited")

    return issues, warnings


def import_asset(filepath, ticker, sheet_name=None, force=False):
    """
    Import custom asset from file.

    Args:
        filepath: Path to CSV/Excel file
        ticker: Ticker symbol to use (e.g., 'MYSTK')
        sheet_name: Sheet name for Excel files
        force: Overwrite existing data
    """
    print(f"\n{'='*60}")
    print(f"  IMPORTING: {ticker}")
    print(f"{'='*60}")

    # Read file
    print(f"\n📂 Reading: {filepath}")
    df = read_file(filepath, sheet_name)
    print(f"   Found {len(df)} rows, {len(df.columns)} columns")

    # Show detected columns
    print(f"\n📋 Original columns: {list(df.columns)}")

    # Standardize
    print("\n🔄 Standardizing format...")
    std_df = standardize_dataframe(df)
    print(f"   Standardized to {len(std_df)} rows")

    # Validate
    print("\n✅ Validating data quality...")
    issues, warnings = validate_data(std_df)

    if issues:
        print("\n❌ Critical issues found:")
        for issue in issues:
            print(f"   - {issue}")
        if not force:
            print("\nUse --force to import anyway")
            return False

    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")

    # Calculate stats
    days = len(std_df)
    years = days / 252

    first_date = std_df['Date'].min()
    last_date = std_df['Date'].max()
    first_price = std_df['Close'].iloc[0]
    last_price = std_df['Close'].iloc[-1]

    total_return = (last_price / first_price - 1) * 100
    if years > 0:
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    else:
        annual_return = 0

    daily_vol = std_df['Close'].pct_change().std() * np.sqrt(252) * 100

    print(f"\n📊 Data Summary:")
    print(f"   Date range: {first_date.date()} to {last_date.date()}")
    print(f"   Trading days: {days} ({years:.1f} years)")
    print(f"   Price range: ${first_price:,.2f} to ${last_price:,.2f}")
    print(f"   Total return: {total_return:,.1f}%")
    print(f"   Annual return: {annual_return:.1f}%")
    print(f"   Annual volatility: {daily_vol:.1f}%")

    # Save to data_cache
    save_path = DATA_DIR / f"{ticker}_1d.csv"

    if save_path.exists() and not force:
        print(f"\n⚠️  File already exists: {save_path}")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Import cancelled")
            return False

    std_df.to_csv(save_path, index=False)
    print(f"\n💾 Saved to: {save_path}")

    print(f"\n✅ Successfully imported {ticker}")
    print("\nNext steps:")
    print(f"  1. Train: python3 train_per_asset.py")
    print(f"  2. Predict: python3 predict.py --asset {ticker}")
    print(f"  3. Backtest: python3 backtest.py --asset {ticker}")

    return True


def list_imported_assets():
    """List all imported assets"""
    print("\n📋 Imported Assets:")
    print("-" * 60)

    assets = []
    for filepath in DATA_DIR.glob("*_1d.csv"):
        ticker = filepath.stem.replace('_1d', '').replace('_', '/')

        try:
            df = pd.read_csv(filepath)
            days = len(df)
            start = df['Date'].min()[:10]
            end = df['Date'].max()[:10]
            assets.append({
                'ticker': ticker,
                'days': days,
                'start': start,
                'end': end
            })
        except Exception:
            pass

    if not assets:
        print("No assets found in data_cache/")
        return

    # Sort by days
    assets.sort(key=lambda x: x['days'], reverse=True)

    for asset in assets:
        print(f"  {asset['ticker']:15s} {asset['days']:5d} days  ({asset['start']} to {asset['end']})")

    print(f"\nTotal: {len(assets)} assets")


def create_sample_csv():
    """Create a sample CSV file to demonstrate expected format"""
    sample_data = {
        'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
        'Open': [100, 101, 99, 102, 103, 101, 104, 105, 103, 106],
        'High': [102, 103, 101, 104, 105, 103, 106, 107, 105, 108],
        'Low': [99, 100, 98, 101, 102, 100, 103, 104, 102, 105],
        'Close': [101, 99, 102, 103, 101, 104, 105, 103, 106, 107],
        'Volume': [1000000, 1200000, 900000, 1100000, 1300000, 1000000, 1400000, 1500000, 1100000, 1600000]
    }

    df = pd.DataFrame(sample_data)
    sample_path = DATA_DIR / "sample_format.csv"
    df.to_csv(sample_path, index=False)

    print(f"\n📝 Created sample file: {sample_path}")
    print("\nSample format:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Import custom assets for NeuroVest evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 import_custom_asset.py my_stock.csv MYSTK
    python3 import_custom_asset.py portfolio.xlsx CUSTOM --sheet "Sheet1"
    python3 import_custom_asset.py --validate my_data.csv
    python3 import_custom_asset.py --list
    python3 import_custom_asset.py --sample
        """
    )

    parser.add_argument('filepath', nargs='?', help='Path to CSV/Excel file')
    parser.add_argument('ticker', nargs='?', help='Ticker symbol for the asset')
    parser.add_argument('--sheet', help='Sheet name for Excel files')
    parser.add_argument('--force', action='store_true', help='Force import even with issues')
    parser.add_argument('--validate', action='store_true', help='Only validate, do not import')
    parser.add_argument('--list', action='store_true', help='List imported assets')
    parser.add_argument('--sample', action='store_true', help='Create sample CSV file')

    args = parser.parse_args()

    print("=" * 60)
    print("  NEUROVEST CUSTOM ASSET IMPORT")
    print("=" * 60)

    if args.list:
        list_imported_assets()
        return

    if args.sample:
        create_sample_csv()
        return

    if not args.filepath:
        parser.print_help()
        print("\n⚠️  Please provide a file path")
        return

    if args.validate:
        # Validation only
        df = read_file(args.filepath, args.sheet)
        std_df = standardize_dataframe(df)
        issues, warnings = validate_data(std_df)

        if issues:
            print("\n❌ Validation failed:")
            for issue in issues:
                print(f"   - {issue}")
        elif warnings:
            print("\n✅ Validation passed with warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        else:
            print("\n✅ Validation passed - data looks good!")
        return

    if not args.ticker:
        print("\n⚠️  Please provide a ticker symbol")
        print("Example: python3 import_custom_asset.py my_data.csv MYSTK")
        return

    # Import the asset
    import_asset(args.filepath, args.ticker.upper(), args.sheet, args.force)


if __name__ == "__main__":
    main()
