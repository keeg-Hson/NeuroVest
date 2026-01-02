# Increase Training Data

## Current Data Problem
- **3,927 rows** (15 years of SPY daily data)
- **5,201 training samples** after feature engineering
- **Insufficient for reliable ML** (need 50K+ samples)

## Solution 1: Extend Historical Data (EASY, HIGH IMPACT)

### Download More SPY History
```python
import yfinance as yf

# Current: 2010-2025 (15 years)
# Target: 1993-2025 (32 years)
spy = yf.download('SPY', start='1993-01-29', end='2025-12-31')
spy.to_csv('data/SPY.csv')
```

**Expected Gain**: 
- From 3,927 → ~8,000 rows
- From 5,201 → ~10,500 training samples
- **2x more data** = significantly better learning

**Pros**: Easy, free, one API call
**Cons**: Still might not be enough

---

## Solution 2: Multi-Asset Training (MEDIUM EFFORT, HIGH IMPACT)

### Train on Multiple Assets Simultaneously

Instead of training only on SPY, train on **all major ETFs**:

```python
tickers = [
    'SPY',   # S&P 500
    'QQQ',   # Nasdaq
    'IWM',   # Russell 2000
    'DIA',   # Dow Jones
    'EFA',   # International
    'EEM',   # Emerging Markets
    'AGG',   # Bonds (for regime detection)
    'GLD',   # Gold
]

# For each ticker, generate features and labels
# Combine all into one training set
combined_data = []
for ticker in tickers:
    df = yf.download(ticker, start='1993-01-01')
    df = add_features(df)
    df = add_forward_returns_and_labels(df, ...)
    df['ticker'] = ticker  # Add as categorical feature
    combined_data.append(df)

train_df = pd.concat(combined_data, ignore_index=True)
```

**Expected Gain**:
- 8 tickers × 8,000 rows = **64,000 training samples**
- **12x more data**
- Models learn **cross-asset patterns** (more robust)

**Pros**: Massive data increase, better generalization
**Cons**: Need to add 'ticker' as feature, assets have different characteristics

---

## Solution 3: Intraday Data (HARD, VERY HIGH IMPACT)

### Use 1-hour or 15-minute bars instead of daily

```python
# Download hourly data
spy_hourly = yf.download('SPY', start='2020-01-01', interval='1h')

# Each trading day = 6.5 hours = 6-7 bars
# 5 years × 252 days × 6.5 bars = ~8,190 samples
```

**Expected Gain**:
- Daily: 3,927 samples
- Hourly: ~8,000 samples (5 years)
- 15-min: ~32,000 samples (5 years)

**Pros**: Massive data increase, capture intraday patterns
**Cons**: 
- Different market dynamics (noise vs signal ratio changes)
- Transaction costs more important
- Need to adapt features for intraday

---

## Solution 4: Alternative Markets (MEDIUM, HIGH IMPACT)

### Trade Crypto Instead (More Inefficiency)

```python
# Crypto trades 24/7, more data available
# Crypto markets less efficient than SPY
import ccxt

exchange = ccxt.binance()
btc_data = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=2000)
eth_data = exchange.fetch_ohlcv('ETH/USDT', '1d', limit=2000)
# ... etc
```

**Why Crypto is Easier**:
- 24/7 trading = more data points
- Less efficient markets = more predictable patterns
- Higher volatility = clearer signals
- Lower barriers = retail can compete

**Trade-offs**:
- Higher volatility = higher risk
- Exchange risk (hacks, insolvency)
- Different market structure

---

## Recommended Approach (BEST ROI)

### Phase 1: Quick Wins (1 hour)
1. ✅ Extend SPY history to 1993 (32 years)
2. ✅ Download QQQ, IWM, DIA (similar assets)

**Expected**: 4x data increase

### Phase 2: Multi-Asset (1 day)
3. ✅ Train on 8 major ETFs combined
4. ✅ Add 'ticker' as categorical feature

**Expected**: 12x data increase

### Phase 3: Alternative (1 week)
5. ✅ Add crypto data (BTC, ETH, top 10)
6. ✅ Train separate crypto models

**Expected**: 20x+ data increase

---

## Implementation Priority

**Do immediately**:
```bash
# 1. Get more SPY history
python -c "
import yfinance as yf
spy = yf.download('SPY', start='1993-01-29', end='2025-12-31')
spy.to_csv('data/SPY.csv')
print(f'Downloaded {len(spy)} rows')
"

# 2. Download other ETFs
python -c "
import yfinance as yf
for ticker in ['QQQ', 'IWM', 'DIA']:
    df = yf.download(ticker, start='1993-01-01')
    df.to_csv(f'data/{ticker}.csv')
    print(f'{ticker}: {len(df)} rows')
"
```

**Expected Immediate Impact**:
- Training samples: 5,201 → 21,000+ (4x increase)
- Model accuracy: +3-5 percentage points
- Overfitting: Significantly reduced
