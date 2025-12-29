# Equity ETF Data Alternatives

## Problem
Yahoo Finance API is blocking automated downloads with 401/429/404 errors. We need alternative data sources to get equity ETF data for multi-asset training.

## Current Results (Without Equity ETFs)

### Per-Asset Model Results
| Asset | Samples | XGB | LGB | CAT | **Ensemble** |
|-------|---------|-----|-----|-----|--------------|
| **SPY** | 6,501 | 70.5% | 68.6% | 71.6% | **70.9%** ✓ |
| BTC_USDT | 1,095 | 63.0% | 64.4% | 63.9% | **63.5%** |
| ETH_USDT | 1,095 | 62.6% | 58.4% | 60.7% | **60.3%** |
| SOL_USDT | 1,095 | 53.4% | 54.3% | 58.4% | **56.6%** |

### Multi-Asset Results (SPY + Crypto)
- **Accuracy: 65.5%** (degraded from 71.0% baseline)
- Conclusion: Crypto hurt SPY performance due to distribution shift

---

## Alternative Data Sources (Ranked by Ease)

### ⭐ Option 1: Alpha Vantage API (FREE, Recommended)
**Pros:**
- Free API key (no credit card required)
- 25 requests/day on free tier (enough for 8 ETFs)
- Already integrated in our codebase

**Cons:**
- Requires API key signup
- 5 calls/minute rate limit (takes ~2 minutes total)

**Setup:**
```bash
# 1. Get free API key
# Visit: https://www.alphavantage.co/support/#api-key
# Enter email, get instant API key

# 2. Set environment variable
export ALPHA_VANTAGE_API_KEY='YOUR_KEY_HERE'

# 3. Run downloader
python3 download_equity_etfs_alternative.py
```

**Expected outcome:**
- All 8 ETFs downloaded (~50,000 total samples)
- Multi-asset accuracy: **73-75%** (projected)
- Training time: ~3-5 minutes

---

### Option 2: Manual CSV Downloads (FREE, Slower)
**Pros:**
- No API key needed
- Direct from Yahoo Finance website
- 100% reliable

**Cons:**
- Manual work (5-10 minutes for 8 ETFs)
- Tedious but straightforward

**Steps for each ETF (QQQ, IWM, DIA, VTI, EEM, XLF, XLK, XLE):**
1. Visit: `https://finance.yahoo.com/quote/[TICKER]/history`
2. Set date range: `Jan 1, 2000` to `Today`
3. Click "Download" button
4. Move downloaded CSV to: `data_cache/[TICKER]_1d.csv`
5. Repeat for all 8 ETFs

**Then run:**
```bash
python3 train_multi_asset.py
python3 compare_approaches.py
```

---

### Option 3: Polygon.io API (FREE tier available)
**Pros:**
- Free tier: 5 API calls/minute
- High-quality data
- Good documentation

**Cons:**
- Requires signup
- Need to write custom downloader

**Setup:**
```bash
# 1. Get free API key
# Visit: https://polygon.io/
# Sign up for free tier

# 2. Install library
pip install polygon-api-client

# 3. Create custom downloader (would need to write this)
```

---

### Option 4: IEX Cloud (FREE tier available)
**Pros:**
- Free tier: 50,000 credits/month
- Real-time and historical data
- Well-documented

**Cons:**
- Requires signup
- Need to write custom downloader

**Setup:**
```bash
# 1. Get free API key
# Visit: https://iexcloud.io/
# Sign up for free tier

# 2. Install library
pip install iexfinance

# 3. Create custom downloader (would need to write this)
```

---

### Option 5: Tiingo API (FREE tier available)
**Pros:**
- Free tier: 1000 requests/day
- Clean API
- Good for backtesting

**Cons:**
- Requires signup
- Need to write custom downloader

**Setup:**
```bash
# 1. Get free API key
# Visit: https://www.tiingo.com/
# Sign up for free tier

# 2. Install library
pip install tiingo

# 3. Create custom downloader (would need to write this)
```

---

## Recommendation

### Best Option: Alpha Vantage (5 minutes setup)
```bash
# Quick start:
export ALPHA_VANTAGE_API_KEY='your-key-from-alphavantage.co'
python3 download_equity_etfs_alternative.py
python3 train_multi_asset.py
```

**Expected improvement:**
- Current baseline: 71.0% (SPY only)
- With equity ETFs: **73-75%** (projected)
- Rationale: Same market structure, no distribution shift, 7x more training data

### Fallback: Manual Downloads (10 minutes)
If you don't want to sign up for anything, just manually download the 8 ETFs from Yahoo Finance website and place them in `data_cache/`.

---

## Why Equity ETFs > Crypto

From our testing results:

| Approach | SPY Accuracy | Reason |
|----------|--------------|--------|
| SPY only (baseline) | 71.0% | ✓ Good baseline |
| SPY + Crypto | 65.5% | ✗ Distribution shift (-5.5 pp) |
| SPY + Equity ETFs | 73-75% (proj.) | ✓ Same market structure |

**Key insight:** Crypto has fundamentally different dynamics (24/7 trading, 5-10x volatility, sentiment-driven). Equity ETFs share SPY's market structure and should improve generalization without distribution shift.

---

## Next Steps

**Option A (5 min):** Get Alpha Vantage API key and run automatic downloader
**Option B (10 min):** Manually download 8 CSV files from Yahoo Finance
**Option C (later):** Keep SPY-only baseline (71.0%) and focus on other improvements

All scripts are ready to use equity ETF data once available.
