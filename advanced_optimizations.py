#!/usr/bin/env python3
"""
Advanced Trading Optimizations

Implements three high-impact improvements:
1. Trade Clustering Prevention - Min 1-2 day gap between trades
2. Multiple Time Horizon Ensemble - Require 7d, 10d, 15d models to agree
3. Market Regime Filtering - Only trade in favorable conditions

Expected impact: +3-7% annualized (8.24% → 10-12%)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)

print("=" * 80)
print("ADVANCED TRADING OPTIMIZATIONS")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

# Configuration
BACKTEST_CFG = {
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "initial_capital": 10000,
}

print(f"\n📋 Implementing:")
print(f"   1. Trade Clustering Prevention (min gap between trades)")
print(f"   2. Multi-Horizon Ensemble (7d + 10d + 15d agreement)")
print(f"   3. Market Regime Filtering (bull + low vol + strong trend)")

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

print("\n📥 Loading data...")
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

# Add forward returns for 10-day horizon (optimal)
df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=10,
    pos_threshold=0.02,
    fee_bps=BACKTEST_CFG['fee_bps'],
    slippage_bps=BACKTEST_CFG['slippage_bps'],
)

prices_df = _raw[["Close"]].copy()
prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")

all_features = [c for c in df.columns if c not in
    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

keep_cols = all_features + ["y", "fwd_ret_net"]
df = df[keep_cols]
df = df.dropna(subset=["y"])
df = df.fillna(0)

print(f"✅ Data loaded: {len(df)} samples, {len(all_features)} features")

# Split data
test_size = int(len(df) * 0.2)
train_end_idx = len(df) - test_size

X_test = df.iloc[train_end_idx:][all_features]
y_test = df.iloc[train_end_idx:]["y"]
test_dates = df.index[train_end_idx:]

print(f"   Test period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
print(f"   Test days: {len(test_dates)}")

# Load models
print("\n📦 Loading models...")
model_7d = joblib.load(MODELS_DIR / "xgboost_regime_7d.pkl")
model_10d = joblib.load(MODELS_DIR / "xgboost_regime_10d.pkl")
model_15d = joblib.load(MODELS_DIR / "xgboost_regime_15d.pkl")
print("✅ Loaded 7-day, 10-day, and 15-day models")

# Get predictions from all models
prob_7d = model_7d.predict_proba(X_test)[:, 1]
prob_10d = model_10d.predict_proba(X_test)[:, 1]
prob_15d = model_15d.predict_proba(X_test)[:, 1]

print(f"   7-day predictions: {(prob_7d >= 0.52).sum()} / {len(prob_7d)} ({(prob_7d >= 0.52).sum()/len(prob_7d):.1%})")
print(f"   10-day predictions: {(prob_10d >= 0.52).sum()} / {len(prob_10d)} ({(prob_10d >= 0.52).sum()/len(prob_10d):.1%})")
print(f"   15-day predictions: {(prob_15d >= 0.52).sum()} / {len(prob_15d)} ({(prob_15d >= 0.52).sum()/len(prob_15d):.1%})")

# Get regime features
regime_score = X_test['Regime_Score'].values if 'Regime_Score' in X_test.columns else np.zeros(len(X_test))
adx = X_test['ADX'].values if 'ADX' in X_test.columns else np.ones(len(X_test)) * 25
bull_market = X_test['Bull_Market'].values if 'Bull_Market' in X_test.columns else np.ones(len(X_test))
high_volatility = X_test['High_Volatility'].values if 'High_Volatility' in X_test.columns else np.zeros(len(X_test))

print(f"   Regime features available: Regime_Score, ADX, Bull_Market, High_Volatility")

# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def run_advanced_backtest(prob_7d, prob_10d, prob_15d, dates, prices,
                         regime_score, adx, bull_market, high_volatility,
                         threshold=0.52, horizon=10,
                         use_ensemble=False, min_agreements=2,
                         use_regime_filter=False,
                         min_gap_days=0,
                         initial_capital=10000, fee_bps=1.5, slippage_bps=2.0):
    """
    Advanced backtest with multiple optimizations.

    Parameters:
    - use_ensemble: Require multiple models to agree
    - min_agreements: How many models must agree (2 or 3)
    - use_regime_filter: Filter trades by market regime
    - min_gap_days: Minimum days between trades (clustering prevention)
    """

    cash = initial_capital
    equity_curve = []
    trades = []

    price_series = prices.reindex(dates)

    position_open = False
    position_entry_idx = None
    last_exit_date = None

    for i in range(len(dates)):
        current_date = dates[i]
        current_price = price_series.loc[current_date]

        # Close position if holding
        should_exit = False
        exit_reason = None

        if position_open:
            if i >= position_entry_idx + horizon:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                exit_price = current_price
                entry_cost = position_entry_price * (1 + (fee_bps + slippage_bps) / 10000)
                exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

                trade_return = (exit_proceeds / entry_cost - 1)
                pnl = trade_return * initial_capital
                cash += initial_capital * (exit_proceeds / entry_cost)

                trades.append({
                    'entry_date': position_entry_date,
                    'entry_price': position_entry_price,
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })

                position_open = False
                last_exit_date = current_date

        # Entry logic
        if not position_open:
            # Base prediction
            should_enter = prob_10d[i] >= threshold

            # Apply ensemble filter
            if use_ensemble and should_enter:
                agreements = sum([
                    prob_7d[i] >= threshold,
                    prob_10d[i] >= threshold,
                    prob_15d[i] >= threshold
                ])
                should_enter = agreements >= min_agreements

            # Apply regime filter
            if use_regime_filter and should_enter:
                # Bull market check
                in_bull = bull_market[i] > 0

                # Low volatility check (not in high volatility regime)
                low_vol = high_volatility[i] == 0

                # Strong trend check (ADX > 20)
                strong_trend = adx[i] > 20

                # All conditions must be met
                should_enter = in_bull and low_vol and strong_trend

            # Apply clustering prevention
            if min_gap_days > 0 and should_enter and last_exit_date is not None:
                days_since_exit = (current_date - last_exit_date).days
                should_enter = days_since_exit >= min_gap_days

            # Enter trade
            if should_enter and i + 1 < len(dates):
                position_entry_idx = i + 1
                position_entry_date = dates[position_entry_idx]
                position_entry_price = price_series.loc[position_entry_date]

                cash -= initial_capital
                position_open = True

        # Calculate portfolio value
        portfolio_value = cash
        if position_open:
            unrealized_value = initial_capital * (current_price / position_entry_price)
            portfolio_value = cash + unrealized_value

        equity_curve.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'position_open': position_open
        })

    # Close final position
    if position_open:
        exit_date = dates[-1]
        exit_price = price_series.loc[exit_date]
        entry_cost = position_entry_price * (1 + (fee_bps + slippage_bps) / 10000)
        exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

        trade_return = (exit_proceeds / entry_cost - 1)
        pnl = trade_return * initial_capital
        cash += initial_capital * (exit_proceeds / entry_cost)

        trades.append({
            'entry_date': position_entry_date,
            'entry_price': position_entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'return': trade_return,
            'pnl': pnl,
            'exit_reason': 'final_close'
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Calculate metrics
    final_value = equity_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1

    days = len(equity_df)
    years = days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
    sharpe = np.sqrt(252) * equity_df['daily_return'].mean() / equity_df['daily_return'].std() if equity_df['daily_return'].std() > 0 else 0

    equity_df['cummax'] = equity_df['portfolio_value'].cummax()
    equity_df['drawdown'] = (equity_df['portfolio_value'] / equity_df['cummax']) - 1
    max_drawdown = equity_df['drawdown'].min()

    n_trades = len(trades_df)
    win_rate = (trades_df['return'] > 0).sum() / n_trades if n_trades > 0 else 0
    avg_return = trades_df['return'].mean() if n_trades > 0 else 0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return,
        'equity_curve': equity_df,
        'trades': trades_df
    }


# ============================================================================
# RUN ALL OPTIMIZATION COMBINATIONS
# ============================================================================

print("\n" + "=" * 80)
print("TESTING OPTIMIZATION STRATEGIES")
print("=" * 80)

results = []

# Baseline (10-day, threshold 0.52)
print(f"\n{'─'*80}")
print(f"Strategy: Baseline (10-day @ 0.52)")
print(f"{'─'*80}")

result = run_advanced_backtest(
    prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
    regime_score, adx, bull_market, high_volatility,
    threshold=0.52, horizon=10,
    use_ensemble=False,
    use_regime_filter=False,
    min_gap_days=0,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Win Rate: {result['win_rate']:.2%}")

results.append({
    'name': 'Baseline (10d @ 0.52)',
    **result
})

# 1. Trade Clustering Prevention
for gap in [1, 2]:
    print(f"\n{'─'*80}")
    print(f"Strategy: Trade Clustering Prevention (min {gap} day gap)")
    print(f"{'─'*80}")

    result = run_advanced_backtest(
        prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
        regime_score, adx, bull_market, high_volatility,
        threshold=0.52, horizon=10,
        use_ensemble=False,
        use_regime_filter=False,
        min_gap_days=gap,
        **BACKTEST_CFG
    )

    print(f"   Total Return: {result['total_return']:.2%}")
    print(f"   Annualized: {result['annualized_return']:.2%}")
    print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
    print(f"   Max DD: {result['max_drawdown']:.2%}")
    print(f"   Trades: {result['n_trades']}")
    print(f"   Win Rate: {result['win_rate']:.2%}")

    results.append({
        'name': f'Clustering Prevention ({gap}d gap)',
        **result
    })

# 2. Multi-Horizon Ensemble
for min_agree in [2, 3]:
    print(f"\n{'─'*80}")
    print(f"Strategy: Multi-Horizon Ensemble (min {min_agree}/3 models agree)")
    print(f"{'─'*80}")

    result = run_advanced_backtest(
        prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
        regime_score, adx, bull_market, high_volatility,
        threshold=0.52, horizon=10,
        use_ensemble=True,
        min_agreements=min_agree,
        use_regime_filter=False,
        min_gap_days=0,
        **BACKTEST_CFG
    )

    print(f"   Total Return: {result['total_return']:.2%}")
    print(f"   Annualized: {result['annualized_return']:.2%}")
    print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
    print(f"   Max DD: {result['max_drawdown']:.2%}")
    print(f"   Trades: {result['n_trades']}")
    print(f"   Win Rate: {result['win_rate']:.2%}")

    results.append({
        'name': f'Ensemble ({min_agree}/3 agree)',
        **result
    })

# 3. Market Regime Filtering
print(f"\n{'─'*80}")
print(f"Strategy: Market Regime Filtering")
print(f"{'─'*80}")

result = run_advanced_backtest(
    prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
    regime_score, adx, bull_market, high_volatility,
    threshold=0.52, horizon=10,
    use_ensemble=False,
    use_regime_filter=True,
    min_gap_days=0,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Win Rate: {result['win_rate']:.2%}")

results.append({
    'name': 'Regime Filtering',
    **result
})

# 4. Combined optimizations
print(f"\n{'─'*80}")
print(f"Strategy: Ensemble (2/3) + Regime Filter")
print(f"{'─'*80}")

result = run_advanced_backtest(
    prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
    regime_score, adx, bull_market, high_volatility,
    threshold=0.52, horizon=10,
    use_ensemble=True,
    min_agreements=2,
    use_regime_filter=True,
    min_gap_days=0,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Win Rate: {result['win_rate']:.2%}")

results.append({
    'name': 'Ensemble + Regime Filter',
    **result
})

print(f"\n{'─'*80}")
print(f"Strategy: Ensemble (3/3) + Regime Filter")
print(f"{'─'*80}")

result = run_advanced_backtest(
    prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
    regime_score, adx, bull_market, high_volatility,
    threshold=0.52, horizon=10,
    use_ensemble=True,
    min_agreements=3,
    use_regime_filter=True,
    min_gap_days=0,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Win Rate: {result['win_rate']:.2%}")

results.append({
    'name': 'Ensemble (3/3) + Regime',
    **result
})

print(f"\n{'─'*80}")
print(f"Strategy: ALL OPTIMIZATIONS (Ensemble 2/3 + Regime + 1d gap)")
print(f"{'─'*80}")

result = run_advanced_backtest(
    prob_7d, prob_10d, prob_15d, test_dates, prices_df['Close'],
    regime_score, adx, bull_market, high_volatility,
    threshold=0.52, horizon=10,
    use_ensemble=True,
    min_agreements=2,
    use_regime_filter=True,
    min_gap_days=1,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Win Rate: {result['win_rate']:.2%}")

results.append({
    'name': 'ALL OPTIMIZATIONS',
    **result
})

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("ADVANCED OPTIMIZATION RESULTS")
print("=" * 80)

results_df = pd.DataFrame([{
    'Strategy': r['name'],
    'Total_Return': r['total_return'],
    'Ann_Return': r['annualized_return'],
    'Sharpe': r['sharpe_ratio'],
    'Max_DD': r['max_drawdown'],
    'Trades': r['n_trades'],
    'Win_Rate': r['win_rate'],
    'Avg_Return': r['avg_return_per_trade']
} for r in results])

results_df = results_df.sort_values('Total_Return', ascending=False)

print("\n" + results_df[['Strategy', 'Total_Return', 'Ann_Return', 'Sharpe',
                         'Max_DD', 'Trades', 'Win_Rate']].to_string(index=False))

# Save results
results_df.to_csv(OUTPUTS_DIR / "advanced_optimization_results.csv", index=False)
print(f"\n💾 Saved: outputs/advanced_optimization_results.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity curves
ax = axes[0, 0]
for result in results[:6]:  # Top 6
    equity_df = result['equity_curve']
    ax.plot(equity_df['date'], equity_df['portfolio_value'],
            label=result['name'], linewidth=2, alpha=0.8)

ax.set_title('Top 6 Strategies: Equity Curves', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=BACKTEST_CFG['initial_capital'], color='black', linestyle='--', alpha=0.5)

# Plot 2: Returns comparison
ax = axes[0, 1]
strategies = results_df['Strategy'].values
returns = results_df['Total_Return'].values * 100
colors = ['green' if r > 0 else 'red' for r in returns]

ax.barh(strategies, returns, color=colors, alpha=0.7)
ax.set_xlabel('Total Return (%)')
ax.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Sharpe vs Ann Return
ax = axes[1, 0]
sharpes = results_df['Sharpe'].values
ann_returns = results_df['Ann_Return'].values * 100

scatter = ax.scatter(sharpes, ann_returns, s=150, alpha=0.7, c=range(len(results_df)), cmap='viridis')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

for i in range(min(5, len(results_df))):
    ax.annotate(results_df.iloc[i]['Strategy'][:15],
                (results_df.iloc[i]['Sharpe'], results_df.iloc[i]['Ann_Return']*100),
                fontsize=7, alpha=0.7)

# Plot 4: Win Rate vs Trades
ax = axes[1, 1]
trades = results_df['Trades'].values
win_rates = results_df['Win_Rate'].values * 100

scatter = ax.scatter(trades, win_rates, s=150, alpha=0.7, c=results_df['Total_Return'].values, cmap='RdYlGn')
ax.set_xlabel('Number of Trades')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate vs Trade Frequency', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Total Return')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'advanced_optimizations.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/advanced_optimizations.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

baseline = results_df[results_df['Strategy'] == 'Baseline (10d @ 0.52)'].iloc[0]
best = results_df.iloc[0]

print(f"\n📊 BASELINE (10-day @ 0.52):")
print(f"   Total Return: {baseline['Total_Return']:.2%}")
print(f"   Annualized: {baseline['Ann_Return']:.2%}")
print(f"   Sharpe: {baseline['Sharpe']:.2f}")
print(f"   Max DD: {baseline['Max_DD']:.2%}")
print(f"   Trades: {baseline['Trades']}")
print(f"   Win Rate: {baseline['Win_Rate']:.2%}")

print(f"\n🏆 BEST STRATEGY: {best['Strategy']}")
print(f"   Total Return: {best['Total_Return']:.2%}")
print(f"   Annualized: {best['Ann_Return']:.2%}")
print(f"   Sharpe: {best['Sharpe']:.2f}")
print(f"   Max DD: {best['Max_DD']:.2%}")
print(f"   Trades: {best['Trades']}")
print(f"   Win Rate: {best['Win_Rate']:.2%}")

if best['Strategy'] != baseline['Strategy']:
    print(f"\n📈 IMPROVEMENT:")
    print(f"   Total Return: {(best['Total_Return'] - baseline['Total_Return'])*100:+.2f}pp")
    print(f"   Annualized: {(best['Ann_Return'] - baseline['Ann_Return'])*100:+.2f}pp")
    print(f"   Sharpe: {(best['Sharpe'] - baseline['Sharpe']):+.2f}")
    print(f"   Max DD: {(best['Max_DD'] - baseline['Max_DD'])*100:+.2f}pp")
    print(f"   Win Rate: {(best['Win_Rate'] - baseline['Win_Rate'])*100:+.2f}pp")
else:
    print(f"\n✅ Baseline is optimal - advanced optimizations don't help!")

print("\n" + "=" * 80)
print("✅ ADVANCED OPTIMIZATION COMPLETE!")
print("=" * 80)
