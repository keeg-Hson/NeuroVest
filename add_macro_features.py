#!/usr/bin/env python3
"""
Add Macro-Economic Features

Creates macro-economic indicators from available data sources.
These features capture the economic cycle and policy environment.

Phase 2 of Maximum Accuracy Implementation
Expected improvement: +8-12% accuracy
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("PHASE 2: ADDING MACRO-ECONOMIC FEATURES")
print("=" * 80)

DATA_DIR = Path("data")

# ============================================================================
# 1. LOAD BASE DATA
# ============================================================================

print("\n📥 Loading base data...")

# Load SPY
spy = pd.read_csv(DATA_DIR / "SPY.csv", index_col=0, parse_dates=True)
print(f"   ✅ SPY: {len(spy)} rows")

# Load TNX (10-Year Treasury) - proxy for interest rates
tnx = pd.read_csv(DATA_DIR / "TNX.csv", index_col=0, parse_dates=True)
print(f"   ✅ TNX (10Y Yield): {len(tnx)} rows")

# Create feature dataframe
macro_features = pd.DataFrame(index=spy.index)

feature_count = 0

# ============================================================================
# 2. INTEREST RATE & MONETARY POLICY FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("CREATING MACRO-ECONOMIC FEATURES")
print("=" * 80)

print("\n[1/5] Interest Rate & Monetary Policy (⭐⭐⭐⭐⭐)")
print("   Why: Fed policy drives market cycles")

# Align TNX to SPY index
tnx_yield = tnx['Close'].reindex(spy.index, method='ffill')

# Interest rate level
macro_features['Macro_10Y_Yield'] = tnx_yield

# Interest rate regime
macro_features['Macro_Low_Rate_Regime'] = (tnx_yield < 2.5).astype(int)   # Accommodative
macro_features['Macro_High_Rate_Regime'] = (tnx_yield > 4.0).astype(int)   # Restrictive
macro_features['Macro_Normal_Rate_Regime'] = (
    (tnx_yield >= 2.5) & (tnx_yield <= 4.0)
).astype(int)

# Rate of change (monetary policy direction)
macro_features['Macro_Rate_Change_3m'] = tnx_yield.diff(63)   # 3 months
macro_features['Macro_Rate_Change_6m'] = tnx_yield.diff(126)  # 6 months
macro_features['Macro_Rate_Change_1y'] = tnx_yield.diff(252)  # 1 year

# Tightening/easing cycles
macro_features['Macro_Tightening_Cycle'] = (macro_features['Macro_Rate_Change_6m'] > 0.5).astype(int)
macro_features['Macro_Easing_Cycle'] = (macro_features['Macro_Rate_Change_6m'] < -0.5).astype(int)

# Rate volatility (policy uncertainty)
macro_features['Macro_Rate_Volatility'] = tnx_yield.diff().rolling(60).std()

count = 9
feature_count += count
print(f"   ✅ Added {count} interest rate features")

# ============================================================================
# 3. ECONOMIC GROWTH PROXY (from market behavior)
# ============================================================================

print("\n[2/5] Economic Growth Proxies (⭐⭐⭐⭐)")
print("   Why: Markets discount economic growth 6-12 months ahead")

spy_close = spy['Close']
spy_return = spy_close.pct_change()

# Market-based growth indicators
# Strong market = growth expectations, weak market = recession fears

# Trend strength (sustained growth)
macro_features['Macro_Market_Trend_Strength'] = (
    (spy_close > spy_close.rolling(200).mean()).astype(int)
)

# Breadth indicator (how widespread is growth)
macro_features['Macro_Above_MA50'] = (spy_close > spy_close.rolling(50).mean()).astype(int)
macro_features['Macro_Above_MA200'] = (spy_close > spy_close.rolling(200).mean()).astype(int)

# Growth momentum
macro_features['Macro_Growth_Momentum_3m'] = spy_close.pct_change(63)   # 3 month
macro_features['Macro_Growth_Momentum_6m'] = spy_close.pct_change(126)  # 6 month
macro_features['Macro_Growth_Momentum_1y'] = spy_close.pct_change(252)  # 1 year

# Recession indicator (prolonged decline)
macro_features['Macro_Recession_Signal'] = (
    (spy_close < spy_close.rolling(252).mean()) &
    (macro_features['Macro_Growth_Momentum_6m'] < -0.1)
).astype(int)

# Recovery indicator (bouncing from lows)
macro_features['Macro_Recovery_Signal'] = (
    (spy_close > spy_close.rolling(63).min() * 1.10) &  # 10% off lows
    (macro_features['Macro_Growth_Momentum_3m'] > 0)
).astype(int)

count = 8
feature_count += count
print(f"   ✅ Added {count} growth proxy features")

# ============================================================================
# 4. INFLATION PROXY (from yield and market behavior)
# ============================================================================

print("\n[3/5] Inflation Regime Indicators (⭐⭐⭐⭐)")
print("   Why: Inflation drives Fed policy and market valuations")

# Inflation proxy from nominal yields
# High and rising yields often indicate inflation concerns
macro_features['Macro_Inflation_Proxy'] = tnx_yield  # Absolute level

# Inflation regime
macro_features['Macro_Low_Inflation'] = (tnx_yield < 2.0).astype(int)     # Deflation risk
macro_features['Macro_Target_Inflation'] = (
    (tnx_yield >= 2.0) & (tnx_yield <= 3.0)
).astype(int)  # Fed target range
macro_features['Macro_High_Inflation'] = (tnx_yield > 3.0).astype(int)     # Inflation pressure

# Inflation acceleration (change in yields)
macro_features['Macro_Inflation_Acceleration'] = tnx_yield.diff(63)  # 3-month change

# Inflation volatility (uncertainty)
macro_features['Macro_Inflation_Uncertainty'] = tnx_yield.diff().rolling(60).std()

count = 6
feature_count += count
print(f"   ✅ Added {count} inflation features")

# ============================================================================
# 5. BUSINESS CYCLE INDICATORS
# ============================================================================

print("\n[4/5] Business Cycle Indicators (⭐⭐⭐⭐)")
print("   Why: Different strategies work in different cycle phases")

# Identify business cycle phase from market and rate data
# Expansion: rising market, stable/low rates
# Peak: rising rates, slowing market
# Contraction: falling market, high rates
# Trough: falling rates, market bottoming

spy_ma_200 = spy_close.rolling(200).mean()
rate_ma_60 = tnx_yield.rolling(60).mean()

# Expansion phase
macro_features['Macro_Expansion'] = (
    (spy_close > spy_ma_200) &
    (tnx_yield < 4.0) &
    (macro_features['Macro_Growth_Momentum_6m'] > 0)
).astype(int)

# Peak phase (late cycle)
macro_features['Macro_Peak'] = (
    (spy_close > spy_ma_200) &
    (tnx_yield > 4.0) &
    (macro_features['Macro_Rate_Change_6m'] > 0.3)  # Rising rates
).astype(int)

# Contraction phase
macro_features['Macro_Contraction'] = (
    (spy_close < spy_ma_200) &
    (macro_features['Macro_Growth_Momentum_6m'] < -0.05)
).astype(int)

# Trough phase (early recovery)
macro_features['Macro_Trough'] = (
    (spy_close < spy_ma_200 * 1.05) &  # Near or below MA
    (macro_features['Macro_Rate_Change_3m'] < 0) &  # Easing
    (spy_return.rolling(20).mean() > 0)  # Starting to recover
).astype(int)

# Cycle position (0-1 scale, 0=trough, 1=peak)
# Simplified: based on distance from 200MA and rate level
macro_features['Macro_Cycle_Position'] = (
    0.5 * ((spy_close - spy_ma_200) / spy_ma_200).clip(-0.3, 0.3) / 0.3 + 0.5 +
    0.5 * (tnx_yield / 6.0).clip(0, 1)
)

count = 5
feature_count += count
print(f"   ✅ Added {count} business cycle features")

# ============================================================================
# 6. POLICY & RISK ENVIRONMENT
# ============================================================================

print("\n[5/5] Policy & Risk Environment (⭐⭐⭐⭐)")
print("   Why: Policy mistakes and shocks drive crashes")

# Financial conditions (tight vs loose)
# Proxy: combination of yields, market strength, volatility
spy_vol = spy_return.rolling(20).std() * np.sqrt(252) * 100

macro_features['Macro_Financial_Conditions_Tight'] = (
    (tnx_yield > tnx_yield.rolling(252).quantile(0.75)) |
    (spy_vol > spy_vol.rolling(252).quantile(0.75))
).astype(int)

macro_features['Macro_Financial_Conditions_Loose'] = (
    (tnx_yield < tnx_yield.rolling(252).quantile(0.25)) &
    (spy_vol < spy_vol.rolling(252).quantile(0.5))
).astype(int)

# Stress indicator (high vol + weak market)
macro_features['Macro_Financial_Stress'] = (
    (spy_close < spy_ma_200) &
    (spy_vol > spy_vol.rolling(252).quantile(0.75))
).astype(int)

# Policy error risk (tightening into weakness)
macro_features['Macro_Policy_Error_Risk'] = (
    (macro_features['Macro_Rate_Change_6m'] > 0.5) &  # Tightening
    (macro_features['Macro_Growth_Momentum_6m'] < 0)  # Slowing growth
).astype(int)

# Market complacency (low vol in late cycle)
macro_features['Macro_Complacency'] = (
    (spy_vol < spy_vol.rolling(252).quantile(0.25)) &
    (macro_features['Macro_Cycle_Position'] > 0.7)  # Late cycle
).astype(int)

count = 5
feature_count += count
print(f"   ✅ Added {count} policy/risk features")

# ============================================================================
# 7. SUMMARY & SAVE
# ============================================================================

print("\n" + "=" * 80)
print("MACRO-ECONOMIC FEATURES SUMMARY")
print("=" * 80)

macro_cols = [col for col in macro_features.columns]
print(f"\n✅ Total macro features created: {len(macro_cols)}")
print(f"   Data coverage: {macro_features.index[0].strftime('%Y-%m-%d')} to {macro_features.index[-1].strftime('%Y-%m-%d')}")

# Show features
print("\n📊 Macro features by category:")
categories = {
    'Interest Rates': [c for c in macro_cols if 'Rate' in c or 'Yield' in c],
    'Economic Growth': [c for c in macro_cols if 'Growth' in c or 'Trend' in c or 'Recession' in c or 'Recovery' in c],
    'Inflation': [c for c in macro_cols if 'Inflation' in c],
    'Business Cycle': [c for c in macro_cols if 'Expansion' in c or 'Peak' in c or 'Contraction' in c or 'Trough' in c or 'Cycle' in c],
    'Policy/Risk': [c for c in macro_cols if 'Financial' in c or 'Policy' in c or 'Stress' in c or 'Complacency' in c or 'Tight' in c or 'Loose' in c]
}

for category, features in categories.items():
    print(f"\n{category} ({len(features)} features):")
    for feat in features:
        non_null = macro_features[feat].notna().sum()
        print(f"   • {feat:<45s} {non_null:>6,} values")

# Save
output_file = DATA_DIR / "macro_features.csv"
macro_features.to_csv(output_file)
print(f"\n💾 Saved to: {output_file}")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION WITH FUTURE RETURNS")
print("=" * 80)

# Calculate forward returns
spy_close = spy['Close']
for horizon in [1, 5, 20]:
    macro_features[f'fwd_ret_{horizon}d'] = spy_close.pct_change(horizon).shift(-horizon)

print("\n📊 Correlation with Future Returns:")
print(f"{'Feature':<45s} {'1-day':>10s} {'5-day':>10s} {'20-day':>10s}")
print("─" * 80)

correlations = []
for feat in macro_cols:
    if macro_features[feat].notna().sum() > 100:
        corr_1d = macro_features[[feat, 'fwd_ret_1d']].corr().iloc[0, 1]
        corr_5d = macro_features[[feat, 'fwd_ret_5d']].corr().iloc[0, 1]
        corr_20d = macro_features[[feat, 'fwd_ret_20d']].corr().iloc[0, 1]

        correlations.append({
            'feature': feat,
            'corr_1d': corr_1d,
            'corr_5d': corr_5d,
            'corr_20d': corr_20d,
            'abs_corr_5d': abs(corr_5d)
        })

        print(f"{feat:<45s} {corr_1d:>+.4f}     {corr_5d:>+.4f}     {corr_20d:>+.4f}")

# Sort by absolute 5-day correlation
correlations_df = pd.DataFrame(correlations).sort_values('abs_corr_5d', ascending=False)

print("\n🎯 Top 10 Most Predictive Macro Features (by 5-day correlation):")
for i, row in correlations_df.head(10).iterrows():
    print(f"   {i+1:2d}. {row['feature']:<45s} r={row['corr_5d']:>+.4f}")

# Save
corr_file = DATA_DIR / "macro_correlations.csv"
correlations_df.to_csv(corr_file, index=False)
print(f"\n💾 Saved correlation analysis to: {corr_file}")

print("\n" + "=" * 80)
print("✅ MACRO-ECONOMIC FEATURES CREATED!")
print("=" * 80)

print(f"\n🎯 Summary:")
print(f"   • Created {len(macro_cols)} macro-economic features")
print(f"   • 5 categories: Interest Rates, Growth, Inflation, Cycle, Policy/Risk")
print(f"   • Saved to data/macro_features.csv")

print(f"\n📈 Next Steps:")
print(f"   1. Run retrain_with_macro.py to integrate with existing features")
print(f"   2. Expected improvement: +8-12% accuracy")
print(f"\n   Current: 61.46%")
print(f"   Target:  70%+ ✨")
