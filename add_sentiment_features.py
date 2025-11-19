#!/usr/bin/env python3
"""
Sentiment Analysis Features - Phase 4.2

Adds market sentiment features using:
1. VIX/VVIX ratio (fear gauge)
2. Put/Call ratio (options sentiment)
3. Market breadth (advancing vs declining stocks)
4. News sentiment (if available via financial news APIs)

Expected improvement: +2-4% accuracy
Target: 63-67% → 65-71%
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 80)
print("SENTIMENT ANALYSIS FEATURES (PHASE 4.2)")
print("=" * 80)

DATA_DIR = Path("data")

# ============================================================================
# 1. LOAD SPY DATA
# ============================================================================

print("\n📥 Loading SPY data...")
spy_file = DATA_DIR / "SPY.csv"
spy = pd.read_csv(spy_file, index_col=0, parse_dates=True)
print(f"✅ Loaded SPY: {len(spy)} rows")

# ============================================================================
# 2. VIX SENTIMENT FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("VIX SENTIMENT FEATURES")
print("=" * 80)

print("\n[1/4] VIX Fear Gauge")
print("   Why: VIX measures market fear - spikes predict volatility")

# Load VIX from existing data
try:
    vix_file = DATA_DIR / "VIX.csv"
    if vix_file.exists():
        vix = pd.read_csv(vix_file, index_col=0, parse_dates=True)
        vix_close = vix['Close'].reindex(spy.index, method='ffill')

        sentiment_features = pd.DataFrame(index=spy.index)

        # VIX level
        sentiment_features['Sent_VIX'] = vix_close

        # VIX percentile (historical range position)
        sentiment_features['Sent_VIX_Percentile'] = vix_close.rolling(252).rank(pct=True)

        # VIX change
        sentiment_features['Sent_VIX_Change_5d'] = vix_close.pct_change(5)

        # VIX regimes
        sentiment_features['Sent_Low_Fear'] = (vix_close < 15).astype(int)  # Complacency
        sentiment_features['Sent_Medium_Fear'] = ((vix_close >= 15) & (vix_close < 25)).astype(int)
        sentiment_features['Sent_High_Fear'] = (vix_close >= 25).astype(int)  # Panic

        # VIX spike detection
        vix_ma = vix_close.rolling(20).mean()
        sentiment_features['Sent_VIX_Spike'] = (vix_close > vix_ma * 1.5).astype(int)

        # Fear reversal (VIX declining from high levels = bullish)
        sentiment_features['Sent_Fear_Reversal'] = (
            (vix_close < vix_close.shift(5)) &  # VIX declining
            (vix_close.shift(5) > 25)  # Was recently elevated
        ).astype(int)

        count = 8
        print(f"   ✅ Added {count} VIX sentiment features")
    else:
        print(f"   ⚠️  VIX data file not found")
        sentiment_features = pd.DataFrame(index=spy.index)
except Exception as e:
    print(f"   ❌ Error loading VIX: {e}")
    sentiment_features = pd.DataFrame(index=spy.index)

# ============================================================================
# 3. PUT/CALL RATIO (OPTIONS SENTIMENT)
# ============================================================================

print("\n[2/4] Put/Call Ratio")
print("   Why: High put/call = bearish sentiment, low = bullish")

# Try to get Put/Call ratio data
# Note: This data is not freely available from yfinance, would need CBOE API or Bloomberg
# Creating proxy using SPY options volume if available

print("   ⚠️  Put/Call ratio requires premium data (CBOE)")
print("   Using VIX as proxy for options sentiment")

# Create proxy features using VIX and implied volatility concepts
if 'Sent_VIX' in sentiment_features.columns:
    spy_returns = spy['Close'].pct_change()
    realized_vol = spy_returns.rolling(20).std() * np.sqrt(252) * 100

    # VIX vs Realized Vol = Implied fear premium
    sentiment_features['Sent_Implied_Fear_Premium'] = sentiment_features['Sent_VIX'] - realized_vol

    # High premium = market expects more volatility = bearish
    sentiment_features['Sent_High_Fear_Premium'] = (
        sentiment_features['Sent_Implied_Fear_Premium'] >
        sentiment_features['Sent_Implied_Fear_Premium'].rolling(60).quantile(0.75)
    ).astype(int)

    count = 2
    print(f"   ✅ Added {count} implied sentiment features (VIX-based proxy)")

# ============================================================================
# 4. MARKET BREADTH (ADVANCING VS DECLINING)
# ============================================================================

print("\n[3/4] Market Breadth")
print("   Why: Strong breadth = healthy rally, weak = distribution")

# Download major sector ETFs to calculate breadth
sectors = {
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials'
}

print(f"   Loading {len(sectors)} sector ETFs...")
sector_data = {}
for ticker, name in sectors.items():
    try:
        sector_file = DATA_DIR / f"{ticker}.csv"
        if sector_file.exists():
            df = pd.read_csv(sector_file, index_col=0, parse_dates=True)
            sector_data[ticker] = df['Close']
            print(f"      ✅ {ticker} ({name})")
        else:
            print(f"      ⚠️  {ticker} ({name}) - file not found")
    except:
        print(f"      ❌ {ticker} ({name}) - error loading")

if sector_data:
    # Calculate breadth
    sector_df = pd.DataFrame(sector_data).reindex(spy.index, method='ffill')

    # Calculate returns for each sector
    sector_returns = sector_df.pct_change()

    # Breadth = % of sectors with positive returns
    sentiment_features['Sent_Breadth_1d'] = (sector_returns > 0).sum(axis=1) / len(sectors)
    sentiment_features['Sent_Breadth_5d'] = (sector_df.pct_change(5) > 0).sum(axis=1) / len(sectors)
    sentiment_features['Sent_Breadth_20d'] = (sector_df.pct_change(20) > 0).sum(axis=1) / len(sectors)

    # Strong breadth regimes
    sentiment_features['Sent_Strong_Breadth'] = (sentiment_features['Sent_Breadth_5d'] > 0.7).astype(int)
    sentiment_features['Sent_Weak_Breadth'] = (sentiment_features['Sent_Breadth_5d'] < 0.3).astype(int)

    # Breadth divergence (price up but breadth weak = warning)
    spy_returns_5d = spy['Close'].pct_change(5)
    sentiment_features['Sent_Breadth_Divergence'] = (
        (spy_returns_5d > 0) &  # SPY up
        (sentiment_features['Sent_Breadth_5d'] < 0.4)  # But weak breadth
    ).astype(int)

    # Sector dispersion (high = rotation, low = consensus)
    sentiment_features['Sent_Sector_Dispersion'] = sector_returns.std(axis=1)

    count = 7
    print(f"\n   ✅ Added {count} market breadth features")
else:
    print(f"\n   ❌ No sector data available for breadth calculation")

# ============================================================================
# 5. RISK-ON / RISK-OFF INDICATORS
# ============================================================================

print("\n[4/4] Risk-On / Risk-Off Indicators")
print("   Why: Risk-on favors growth, risk-off favors safety")

# Compare growth vs defensive sectors
if 'XLK' in sector_data and 'XLU' in sector_data:
    xlk = sector_data['XLK'].reindex(spy.index, method='ffill')
    xlu = sector_data['XLU'].reindex(spy.index, method='ffill')

    # Growth/Defensive ratio
    growth_defensive_ratio = xlk / xlu
    sentiment_features['Sent_Growth_Defensive_Ratio'] = growth_defensive_ratio
    sentiment_features['Sent_Growth_Defensive_Change'] = growth_defensive_ratio.pct_change(20)

    # Risk-on regime (growth outperforming)
    sentiment_features['Sent_Risk_On'] = (
        growth_defensive_ratio > growth_defensive_ratio.rolling(60).mean()
    ).astype(int)

    count = 3
    print(f"   ✅ Added {count} risk-on/risk-off features")

# Additional: Safe haven flows (if gold data available)
if (DATA_DIR / "GLD.csv").exists():
    try:
        gld = pd.read_csv(DATA_DIR / "GLD.csv", index_col=0, parse_dates=True)
        gld_close = gld['Close'].reindex(spy.index, method='ffill')
        spy_close = spy['Close']

        # Gold outperformance = risk-off
        sentiment_features['Sent_Gold_Outperformance'] = (
            gld_close.pct_change(20) > spy_close.pct_change(20)
        ).astype(int)

        print(f"   ✅ Added 1 safe-haven sentiment feature")
    except:
        pass

# ============================================================================
# 6. SENTIMENT COMPOSITE SCORE
# ============================================================================

print("\n" + "=" * 80)
print("CREATING COMPOSITE SENTIMENT SCORE")
print("=" * 80)

# Normalize and combine sentiment signals
if len(sentiment_features.columns) > 0:
    # Create bullish sentiment score (0-100)
    bullish_signals = []

    if 'Sent_Low_Fear' in sentiment_features.columns:
        bullish_signals.append(sentiment_features['Sent_Low_Fear'])

    if 'Sent_Fear_Reversal' in sentiment_features.columns:
        bullish_signals.append(sentiment_features['Sent_Fear_Reversal'])

    if 'Sent_Strong_Breadth' in sentiment_features.columns:
        bullish_signals.append(sentiment_features['Sent_Strong_Breadth'])

    if 'Sent_Risk_On' in sentiment_features.columns:
        bullish_signals.append(sentiment_features['Sent_Risk_On'])

    if bullish_signals:
        sentiment_features['Sent_Bullish_Score'] = pd.concat(bullish_signals, axis=1).mean(axis=1) * 100
        print(f"   ✅ Created bullish sentiment score (0-100)")

    # Create bearish sentiment score
    bearish_signals = []

    if 'Sent_High_Fear' in sentiment_features.columns:
        bearish_signals.append(sentiment_features['Sent_High_Fear'])

    if 'Sent_VIX_Spike' in sentiment_features.columns:
        bearish_signals.append(sentiment_features['Sent_VIX_Spike'])

    if 'Sent_Weak_Breadth' in sentiment_features.columns:
        bearish_signals.append(sentiment_features['Sent_Weak_Breadth'])

    if 'Sent_Breadth_Divergence' in sentiment_features.columns:
        bearish_signals.append(sentiment_features['Sent_Breadth_Divergence'])

    if bearish_signals:
        sentiment_features['Sent_Bearish_Score'] = pd.concat(bearish_signals, axis=1).mean(axis=1) * 100
        print(f"   ✅ Created bearish sentiment score (0-100)")

    # Net sentiment
    if 'Sent_Bullish_Score' in sentiment_features.columns and 'Sent_Bearish_Score' in sentiment_features.columns:
        sentiment_features['Sent_Net_Score'] = (
            sentiment_features['Sent_Bullish_Score'] -
            sentiment_features['Sent_Bearish_Score']
        )
        print(f"   ✅ Created net sentiment score (-100 to +100)")

# ============================================================================
# 7. SAVE SENTIMENT FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("SAVING SENTIMENT FEATURES")
print("=" * 80)

sentiment_cols = [col for col in sentiment_features.columns if col.startswith('Sent_')]

print(f"\n✅ Total sentiment features created: {len(sentiment_cols)}")
print(f"   Data range: {sentiment_features.index[0].strftime('%Y-%m-%d')} to {sentiment_features.index[-1].strftime('%Y-%m-%d')}")
print(f"   Total rows: {len(sentiment_features):,}")

# Show features
print(f"\n📊 Sentiment Features:")
for i, col in enumerate(sentiment_cols, 1):
    non_null = sentiment_features[col].notna().sum()
    pct = (non_null / len(sentiment_features)) * 100
    print(f"   {i:2d}. {col:<40s} {non_null:>6,} ({pct:>5.1f}%)")

# Save
output_file = DATA_DIR / "sentiment_features.csv"
sentiment_features[sentiment_cols].to_csv(output_file)
print(f"\n💾 Saved to: {output_file}")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION WITH FORWARD RETURNS")
print("=" * 80)

# Calculate forward returns
spy_close = spy['Close'].reindex(sentiment_features.index, method='ffill')
for horizon in [1, 5, 20]:
    sentiment_features[f'fwd_ret_{horizon}d'] = spy_close.pct_change(horizon).shift(-horizon)

print(f"\n📊 Correlation with Future Returns:")
print(f"{'Feature':<40s} {'1-day':>10s} {'5-day':>10s} {'20-day':>10s}")
print("─" * 75)

correlations = []
for feat in sentiment_cols:
    if sentiment_features[feat].notna().sum() > 100:
        corr_1d = sentiment_features[[feat, 'fwd_ret_1d']].corr().iloc[0, 1]
        corr_5d = sentiment_features[[feat, 'fwd_ret_5d']].corr().iloc[0, 1]
        corr_20d = sentiment_features[[feat, 'fwd_ret_20d']].corr().iloc[0, 1]

        correlations.append({
            'feature': feat,
            'corr_1d': corr_1d,
            'corr_5d': corr_5d,
            'corr_20d': corr_20d,
            'abs_corr_5d': abs(corr_5d)
        })

        print(f"{feat:<40s} {corr_1d:>+.4f}     {corr_5d:>+.4f}     {corr_20d:>+.4f}")

# Save correlations
if correlations:
    corr_df = pd.DataFrame(correlations).sort_values('abs_corr_5d', ascending=False)
    corr_output = DATA_DIR / "sentiment_correlations.csv"
    corr_df.to_csv(corr_output, index=False)

    print(f"\n🎯 Top 5 Most Predictive Sentiment Features:")
    for i, row in corr_df.head(5).iterrows():
        print(f"   {i+1}. {row['feature']:<40s} r={row['corr_5d']:>+.4f}")

    print(f"\n💾 Saved correlation analysis to: {corr_output}")

print("\n" + "=" * 80)
print("✅ SENTIMENT FEATURES CREATED!")
print("=" * 80)

print(f"\n🎯 Summary:")
print(f"   • Created {len(sentiment_cols)} sentiment features")
print(f"   • Covers {len(sentiment_features):,} trading days")
print(f"   • Saved to data/sentiment_features.csv")

print(f"\n📈 Next Steps:")
print(f"   1. Integrate with existing 164 features")
print(f"   2. Retrain models with sentiment features")
print(f"   3. Expected improvement: +2-4% accuracy")
print(f"\n   Current: 63.23%")
print(f"   Target:  65-67% ✨")
