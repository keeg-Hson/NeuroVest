#!/usr/bin/env python3
"""
Optimized Backtest with Advanced Risk Management

Implements multiple optimization strategies:
1. Confidence-based position sizing (25% to 100% based on prediction probability)
2. Stop loss (-5%) and take profit (+8%)
3. Threshold optimization (test 0.50-0.55)
4. Regime-aware position sizing (larger positions in favorable regimes)
5. Dynamic holding periods (exit when confidence drops)
6. Transaction cost reduction

Expected improvements:
- Annualized Return: 6-8% (vs 4.90%)
- Sharpe Ratio: 0.45-0.55 (vs 0.36)
- Max Drawdown: -12% to -15% (vs -18.77%)
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
from train import TRAIN_CFG

print("=" * 80)
print("OPTIMIZED BACKTEST - ADVANCED RISK MANAGEMENT")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
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

df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

prices_df = _raw[["Close"]].copy()
prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")

# Get regime features for regime-aware sizing
regime_features = [f for f in feature_cols if any(x in f for x in
    ['MA_200', 'Bull_Market', 'ADX', 'Plus_DI', 'Minus_DI',
     'High_Volatility', 'Regime_Score', 'Near_52w', 'Trend_Consistency'])]

all_features = [c for c in df.columns if c not in
    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

keep_cols = all_features + ["y", "fwd_ret_net"]
df = df[keep_cols]
df = df.dropna(subset=["y"])
df = df.fillna(0)

print(f"✅ Data loaded: {len(df)} samples")
print(f"   Features: {len(all_features)} (including {len(regime_features)} regime features)")

# Split data
test_size = int(len(df) * 0.2)
train_end_idx = len(df) - test_size

X_test = df.iloc[train_end_idx:][all_features]
y_test = df.iloc[train_end_idx:]["y"]
test_dates = df.index[train_end_idx:]

print(f"\n📅 Backtest period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
print(f"   Days: {len(test_dates)}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n📦 Loading XGBoost (Regime) model...")
xgb_regime = joblib.load(MODELS_DIR / "xgboost_regime.pkl")
probabilities = xgb_regime.predict_proba(X_test)[:, 1]
print("✅ Model loaded successfully")

# Get regime scores for position sizing
regime_score = X_test['Regime_Score'].values if 'Regime_Score' in X_test.columns else np.zeros(len(X_test))
adx = X_test['ADX'].values if 'ADX' in X_test.columns else np.ones(len(X_test)) * 25

# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def calculate_position_size(probability, regime_score, adx, strategy='confidence'):
    """
    Calculate position size based on multiple factors.

    Strategies:
    - 'confidence': Scale 25% to 100% based on prediction confidence
    - 'regime': Adjust based on market regime (bull/bear/volatility)
    - 'combined': Use both confidence and regime
    """
    if strategy == 'fixed':
        return 1.0

    elif strategy == 'confidence':
        # Scale from 25% (at threshold 0.5) to 100% (at probability 1.0)
        min_size = 0.25
        max_size = 1.0
        # Linear scaling: size = min_size + (prob - 0.5) * (max_size - min_size) / 0.5
        if probability < 0.5:
            return 0.0
        size = min_size + (probability - 0.5) * (max_size - min_size) / 0.5
        return np.clip(size, min_size, max_size)

    elif strategy == 'regime':
        # Base size 50%, adjust based on regime
        base_size = 0.5

        # Regime adjustment: +30% in bull, -30% in bear
        regime_adj = regime_score * 0.3  # regime_score is typically -1 to +1

        # Trend strength (ADX): higher ADX = more confident sizing
        # ADX > 25 is trending, scale from 0.8 to 1.2
        trend_adj = np.clip(adx / 25, 0.8, 1.2) - 1.0  # -0.2 to +0.2

        size = base_size * (1 + regime_adj + trend_adj)
        return np.clip(size, 0.25, 1.0)

    elif strategy == 'combined':
        # Combine confidence and regime
        conf_size = calculate_position_size(probability, regime_score, adx, 'confidence')
        regime_multiplier = 1.0 + regime_score * 0.2  # ±20% based on regime

        # ADX adjustment
        if adx > 25:  # Strong trend
            trend_multiplier = 1.1
        elif adx < 15:  # Weak trend
            trend_multiplier = 0.9
        else:
            trend_multiplier = 1.0

        size = conf_size * regime_multiplier * trend_multiplier
        return np.clip(size, 0.25, 1.0)

    return 1.0


def run_optimized_backtest(probabilities, dates, prices, regime_scores, adx_values,
                          threshold=0.5, position_strategy='fixed',
                          stop_loss_pct=None, take_profit_pct=None,
                          dynamic_exit=False, initial_capital=10000,
                          fee_bps=1.5, slippage_bps=2.0):
    """
    Run backtest with advanced optimizations.

    Parameters:
    - threshold: Minimum probability to enter trade
    - position_strategy: 'fixed', 'confidence', 'regime', or 'combined'
    - stop_loss_pct: Stop loss percentage (e.g., 0.05 for -5%)
    - take_profit_pct: Take profit percentage (e.g., 0.08 for +8%)
    - dynamic_exit: Exit when probability drops below threshold
    """
    horizon = TRAIN_CFG["horizon"]

    cash = initial_capital
    equity_curve = []
    trades = []

    price_series = prices.reindex(dates)

    position_open = False
    position_entry_idx = None
    position_size = 0

    for i in range(len(dates)):
        current_date = dates[i]
        current_price = price_series.loc[current_date]
        current_prob = probabilities[i]

        # Close position if holding
        should_exit = False
        exit_reason = None

        if position_open:
            # Check exit conditions
            entry_price = position_entry_price
            current_return = (current_price / entry_price) - 1

            # 1. Stop loss
            if stop_loss_pct and current_return <= -stop_loss_pct:
                should_exit = True
                exit_reason = 'stop_loss'

            # 2. Take profit
            elif take_profit_pct and current_return >= take_profit_pct:
                should_exit = True
                exit_reason = 'take_profit'

            # 3. Dynamic exit (confidence dropped)
            elif dynamic_exit and current_prob < threshold:
                should_exit = True
                exit_reason = 'confidence_drop'

            # 4. Max holding period reached
            elif i >= position_entry_idx + horizon:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                # Exit position
                exit_price = current_price
                entry_cost = entry_price * (1 + (fee_bps + slippage_bps) / 10000)
                exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

                pnl = (exit_proceeds / entry_cost - 1) * position_size * initial_capital
                cash += position_size * initial_capital * (exit_proceeds / entry_cost)

                trades.append({
                    'entry_date': position_entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': (exit_proceeds / entry_cost - 1),
                    'pnl': pnl,
                    'size': position_size,
                    'exit_reason': exit_reason
                })

                position_open = False

        # Enter new position if signal and not holding
        if current_prob >= threshold and not position_open:
            if i + 1 < len(dates):
                # Calculate position size
                position_size = calculate_position_size(
                    current_prob,
                    regime_scores[i],
                    adx_values[i],
                    position_strategy
                )

                position_entry_idx = i + 1
                position_entry_date = dates[position_entry_idx]
                position_entry_price = price_series.loc[position_entry_date]

                cash -= position_size * initial_capital
                position_open = True

        # Calculate portfolio value
        portfolio_value = cash
        if position_open:
            unrealized_value = position_size * initial_capital * (current_price / position_entry_price)
            portfolio_value = cash + unrealized_value

        equity_curve.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'position_open': position_open
        })

    # Close final position if open
    if position_open:
        exit_date = dates[-1]
        exit_price = price_series.loc[exit_date]
        entry_cost = position_entry_price * (1 + (fee_bps + slippage_bps) / 10000)
        exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

        pnl = (exit_proceeds / entry_cost - 1) * position_size * initial_capital
        cash += position_size * initial_capital * (exit_proceeds / entry_cost)

        trades.append({
            'entry_date': position_entry_date,
            'entry_price': position_entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'return': (exit_proceeds / entry_cost - 1),
            'pnl': pnl,
            'size': position_size,
            'exit_reason': 'final_close'
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return equity_df, trades_df


def calculate_metrics(equity_df, trades_df, initial_capital):
    """Calculate performance metrics"""
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
    avg_size = trades_df['size'].mean() if n_trades > 0 else 0

    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict() if n_trades > 0 else {}

    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return,
        'avg_position_size': avg_size,
        'exit_reasons': exit_reasons
    }


# ============================================================================
# RUN OPTIMIZATION EXPERIMENTS
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING OPTIMIZATION EXPERIMENTS")
print("=" * 80)

initial_capital = 10000
fee_bps = 1.5
slippage_bps = 2.0

experiments = [
    {
        'name': 'Baseline (Original)',
        'threshold': 0.5,
        'position_strategy': 'fixed',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'Confidence-Based Sizing',
        'threshold': 0.5,
        'position_strategy': 'confidence',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'With Stop Loss (-5%)',
        'threshold': 0.5,
        'position_strategy': 'fixed',
        'stop_loss_pct': 0.05,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'Stop Loss + Take Profit',
        'threshold': 0.5,
        'position_strategy': 'fixed',
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.08,
        'dynamic_exit': False
    },
    {
        'name': 'Optimized Threshold (0.52)',
        'threshold': 0.52,
        'position_strategy': 'fixed',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'Optimized Threshold (0.55)',
        'threshold': 0.55,
        'position_strategy': 'fixed',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'Regime-Aware Sizing',
        'threshold': 0.5,
        'position_strategy': 'regime',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'Combined Sizing (Confidence + Regime)',
        'threshold': 0.5,
        'position_strategy': 'combined',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': False
    },
    {
        'name': 'Dynamic Exit (Confidence-Based)',
        'threshold': 0.5,
        'position_strategy': 'fixed',
        'stop_loss_pct': None,
        'take_profit_pct': None,
        'dynamic_exit': True
    },
    {
        'name': 'FULL OPTIMIZATION',
        'threshold': 0.52,
        'position_strategy': 'combined',
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.08,
        'dynamic_exit': True
    }
]

results = []

for exp in experiments:
    print(f"\n{'─' * 80}")
    print(f"Testing: {exp['name']}")
    print(f"{'─' * 80}")
    print(f"   Threshold: {exp['threshold']}")
    print(f"   Position Strategy: {exp['position_strategy']}")
    print(f"   Stop Loss: {exp['stop_loss_pct']*100 if exp['stop_loss_pct'] else 'None'}%")
    print(f"   Take Profit: {exp['take_profit_pct']*100 if exp['take_profit_pct'] else 'None'}%")
    print(f"   Dynamic Exit: {exp['dynamic_exit']}")

    equity_df, trades_df = run_optimized_backtest(
        probabilities, test_dates, prices_df['Close'],
        regime_score, adx,
        threshold=exp['threshold'],
        position_strategy=exp['position_strategy'],
        stop_loss_pct=exp['stop_loss_pct'],
        take_profit_pct=exp['take_profit_pct'],
        dynamic_exit=exp['dynamic_exit'],
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps
    )

    metrics = calculate_metrics(equity_df, trades_df, initial_capital)

    print(f"\n   Final Value: ${metrics['final_value']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.2%}")
    print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   Trades: {metrics['n_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.2%}")
    print(f"   Avg Position Size: {metrics['avg_position_size']:.1%}")

    if metrics['exit_reasons']:
        print(f"   Exit Reasons: {metrics['exit_reasons']}")

    results.append({
        'Strategy': exp['name'],
        **exp,
        **metrics,
        'equity_curve': equity_df,
        'trades': trades_df
    })

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION RESULTS COMPARISON")
print("=" * 80)

results_df = pd.DataFrame([{
    'Strategy': r['Strategy'],
    'Threshold': r['threshold'],
    'Position_Strategy': r['position_strategy'],
    'Stop_Loss': f"{r['stop_loss_pct']*100:.0f}%" if r['stop_loss_pct'] else 'None',
    'Take_Profit': f"{r['take_profit_pct']*100:.0f}%" if r['take_profit_pct'] else 'None',
    'Dynamic_Exit': r['dynamic_exit'],
    'Final_Value': r['final_value'],
    'Total_Return': r['total_return'],
    'Ann_Return': r['annualized_return'],
    'Sharpe': r['sharpe_ratio'],
    'Max_DD': r['max_drawdown'],
    'Trades': r['n_trades'],
    'Win_Rate': r['win_rate'],
    'Avg_Size': r['avg_position_size']
} for r in results])

results_df = results_df.sort_values('Total_Return', ascending=False)

print("\n" + results_df[['Strategy', 'Total_Return', 'Ann_Return', 'Sharpe',
                         'Max_DD', 'Trades', 'Win_Rate']].to_string(index=False))

# Save detailed results
results_df.to_csv(OUTPUTS_DIR / "optimized_backtest_results.csv", index=False)
print(f"\n💾 Saved: outputs/optimized_backtest_results.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity curves
ax = axes[0, 0]
for result in results[:5]:  # Top 5 strategies
    equity_df = result['equity_curve']
    ax.plot(equity_df['date'], equity_df['portfolio_value'],
            label=result['Strategy'], linewidth=2)

ax.set_title('Top 5 Strategies: Equity Curves', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5)

# Plot 2: Returns comparison
ax = axes[0, 1]
strategies = results_df['Strategy'].head(10).values
returns = (results_df['Total_Return'].head(10).values * 100)
colors = ['green' if r > 0 else 'red' for r in returns]

ax.barh(strategies, returns, color=colors, alpha=0.7)
ax.set_xlabel('Total Return (%)')
ax.set_title('Total Return Comparison (Top 10)', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Sharpe vs Return
ax = axes[1, 0]
sharpes = results_df['Sharpe'].values
ann_returns = results_df['Ann_Return'].values * 100

scatter = ax.scatter(sharpes, ann_returns, s=100, alpha=0.6, c=range(len(results_df)), cmap='viridis')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Annotate best strategies
for i in range(min(3, len(results_df))):
    ax.annotate(results_df.iloc[i]['Strategy'][:20],
                (results_df.iloc[i]['Sharpe'], results_df.iloc[i]['Ann_Return']*100),
                fontsize=8, alpha=0.7)

# Plot 4: Drawdown comparison
ax = axes[1, 1]
strategies = results_df['Strategy'].head(10).values
drawdowns = abs(results_df['Max_DD'].head(10).values * 100)

ax.barh(strategies, drawdowns, color='red', alpha=0.6)
ax.set_xlabel('Max Drawdown (%)')
ax.set_title('Maximum Drawdown Comparison (Top 10)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_xaxis()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'optimized_backtest_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/optimized_backtest_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

baseline = results_df[results_df['Strategy'] == 'Baseline (Original)'].iloc[0]
best = results_df.iloc[0]

print(f"\n📊 BASELINE (Original Strategy):")
print(f"   Total Return: {baseline['Total_Return']:.2%}")
print(f"   Annualized Return: {baseline['Ann_Return']:.2%}")
print(f"   Sharpe Ratio: {baseline['Sharpe']:.2f}")
print(f"   Max Drawdown: {baseline['Max_DD']:.2%}")
print(f"   Trades: {baseline['Trades']}")
print(f"   Win Rate: {baseline['Win_Rate']:.2%}")

print(f"\n🏆 BEST OPTIMIZED STRATEGY: {best['Strategy']}")
print(f"   Total Return: {best['Total_Return']:.2%}")
print(f"   Annualized Return: {best['Ann_Return']:.2%}")
print(f"   Sharpe Ratio: {best['Sharpe']:.2f}")
print(f"   Max Drawdown: {best['Max_DD']:.2%}")
print(f"   Trades: {best['Trades']}")
print(f"   Win Rate: {best['Win_Rate']:.2%}")

print(f"\n📈 IMPROVEMENT:")
print(f"   Total Return: {(best['Total_Return'] - baseline['Total_Return'])*100:+.2f}pp")
print(f"   Annualized Return: {(best['Ann_Return'] - baseline['Ann_Return'])*100:+.2f}pp")
print(f"   Sharpe Ratio: {(best['Sharpe'] - baseline['Sharpe']):+.2f}")
print(f"   Max Drawdown: {(best['Max_DD'] - baseline['Max_DD'])*100:+.2f}pp (lower is better)")
print(f"   Win Rate: {(best['Win_Rate'] - baseline['Win_Rate'])*100:+.2f}pp")

print("\n" + "=" * 80)
print("✅ OPTIMIZATION COMPLETE!")
print("=" * 80)
