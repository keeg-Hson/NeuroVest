#!/bin/bash
# Download extended SPY historical data (1993-2025)
#
# This script needs to be run with a properly configured Python environment
# that has yfinance installed.
#
# Usage:
#   source .venv/bin/activate  # If using virtual environment
#   bash DOWNLOAD_SPY_DATA.sh

python3 << 'PYTHON_SCRIPT'
import yfinance as yf
import pandas as pd

print('Downloading SPY data from 1993-01-29 to present...')
spy = yf.download('SPY', start='1993-01-29', end='2025-12-31', progress=False)

if len(spy) > 0:
    print(f'✓ Downloaded {len(spy)} rows of SPY data')
    print(f'  Date range: {spy.index[0]} to {spy.index[-1]}')
    
    # Save to data/SPY.csv
    spy.to_csv('data/SPY.csv')
    print(f'✓ Saved to data/SPY.csv')
    
    # Show comparison
    print(f'\nData increase:')
    print(f'  Previous: ~3,927 rows (2010-2025)')
    print(f'  Current: {len(spy)} rows (1993-2025)')
    print(f'  Increase: {len(spy) - 3927} rows (~{((len(spy)/3927)-1)*100:.0f}% more data!)')
else:
    print('✗ No data downloaded')
PYTHON_SCRIPT
