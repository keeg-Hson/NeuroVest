#!/usr/bin/env python3
"""
Holding Period Optimization

Tests multiple holding periods (3, 7, 10, 15, 20 days) to find optimal trade duration.

This addresses a key limitation: current 5-day holding may be too short for:
- Stop loss/take profit to trigger effectively
- Full market moves to develop
- Optimal risk-adjusted returns

Expected impact: +1-3% annualized return improvement
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)

print("=" * 80)
print("HOLDING PERIOD OPTIMIZATION")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Test these holding periods
HORIZONS = [3, 5, 7, 10, 15, 20]

# Backtest configuration
BACKTEST_CFG = {
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "initial_capital": 10000,
}

print(f"\n📋 Testing holding periods: {HORIZONS} days")
print(f"   Initial Capital: ${BACKTEST_CFG['initial_capital']:,.0f}")
print(f"   Trading Costs: {BACKTEST_CFG['fee_bps'] + BACKTEST_CFG['slippage_bps']} bps per trade")

# ============================================================================
# TRAIN MODELS FOR EACH HORIZON
# ============================================================================

def train_model_for_horizon(horizon, threshold_pos=0.02, threshold_neg=-0.02):
    """
    Train XGBoost model for a specific holding period.

    Parameters:
    - horizon: Number of days to hold
    - threshold_pos: Minimum return to label as positive (after costs)
    - threshold_neg: Maximum return to label as negative (after costs)
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL FOR {horizon}-DAY HORIZON")
    print(f"{'='*80}")

    # Load and prepare data
    df = load_SPY_data()
    df, feature_cols = add_features(df)
    df = finalize_features(df, feature_cols)

    # Get Close prices
    _raw = load_SPY_data()
    _raw_idxed = _raw["Close"].astype(float)
    df.index = pd.to_datetime(df.index, errors="coerce")
    _raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
    df["Close"] = _raw_idxed.reindex(df.index)
    df = df.dropna(subset=["Close"])

    # Add forward returns with this horizon
    df = add_forward_returns_and_labels(
        df,
        price_col="Close",
        horizon=horizon,
        pos_threshold=threshold_pos,
        fee_bps=BACKTEST_CFG['fee_bps'],
        slippage_bps=BACKTEST_CFG['slippage_bps'],
    )

    # Get regime features
    regime_features = [f for f in feature_cols if any(x in f for x in
        ['MA_200', 'Bull_Market', 'ADX', 'Plus_DI', 'Minus_DI',
         'High_Volatility', 'Regime_Score', 'Near_52w', 'Trend_Consistency'])]

    all_features = [c for c in df.columns if c not in
        ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

    # Prepare data
    keep_cols = all_features + ["y", "fwd_ret_net"]
    df = df[keep_cols]
    df = df.dropna(subset=["y"])
    df = df.fillna(0)

    print(f"✅ Data prepared: {len(df)} samples, {len(all_features)} features")

    # Split data (80/20)
    test_size = int(len(df) * 0.2)
    train_end_idx = len(df) - test_size

    X_train = df.iloc[:train_end_idx][all_features]
    y_train = df.iloc[:train_end_idx]["y"]
    X_test = df.iloc[train_end_idx:][all_features]
    y_test = df.iloc[train_end_idx:]["y"]

    print(f"\n📊 Class distribution:")
    print(f"   Train: {y_train.sum()} positive / {len(y_train) - y_train.sum()} negative ({y_train.mean():.1%})")
    print(f"   Test:  {y_test.sum()} positive / {len(y_test) - y_test.sum()} negative ({y_test.mean():.1%})")

    # Train XGBoost with walk-forward validation
    print(f"\n🔧 Training XGBoost with walk-forward validation...")

    N_SPLITS = 10
    TEST_SIZE = len(X_train) // (N_SPLITS + 1)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'
    }

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_fold_train, y_fold_train, verbose=False)

        val_acc = model.score(X_fold_val, y_fold_val)
        fold_scores.append(val_acc)

    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)

    print(f"   Cross-validation accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

    # Train final model on all training data
    print(f"\n🎯 Training final model on full training set...")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X_train, y_train, verbose=False)

    # Test set performance
    test_acc = final_model.score(X_test, y_test)
    print(f"   Test set accuracy: {test_acc:.4f}")

    # Save model
    model_path = MODELS_DIR / f"xgboost_regime_{horizon}d.pkl"
    joblib.dump(final_model, model_path)
    print(f"💾 Model saved: {model_path}")

    return {
        'horizon': horizon,
        'model': final_model,
        'features': all_features,
        'cv_score': mean_cv_score,
        'cv_std': std_cv_score,
        'test_acc': test_acc,
        'df': df,
        'test_start_idx': train_end_idx
    }


# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

def run_backtest_with_optimizations(model_info, threshold=0.52,
                                    stop_loss_pct=None, take_profit_pct=None):
    """
    Run backtest for a specific horizon with optimizations.

    Features:
    - Optimized threshold (0.52)
    - Optional stop loss
    - Optional take profit
    """
    horizon = model_info['horizon']
    model = model_info['model']
    df = model_info['df']
    test_start_idx = model_info['test_start_idx']
    features = model_info['features']

    # Get test data
    X_test = df.iloc[test_start_idx:][features]
    test_dates = df.index[test_start_idx:]

    # Get predictions
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Get prices
    _raw = load_SPY_data()
    prices = _raw["Close"].astype(float)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    price_series = prices.reindex(test_dates)

    # Run backtest
    fee_bps = BACKTEST_CFG['fee_bps']
    slippage_bps = BACKTEST_CFG['slippage_bps']
    initial_capital = BACKTEST_CFG['initial_capital']

    cash = initial_capital
    equity_curve = []
    trades = []

    position_open = False
    position_entry_idx = None

    for i in range(len(test_dates)):
        current_date = test_dates[i]
        current_price = price_series.loc[current_date]

        # Close position if holding
        should_exit = False
        exit_reason = None

        if position_open:
            entry_price = position_entry_price
            current_return = (current_price / entry_price) - 1

            # Check exit conditions
            if stop_loss_pct and current_return <= -stop_loss_pct:
                should_exit = True
                exit_reason = 'stop_loss'
            elif take_profit_pct and current_return >= take_profit_pct:
                should_exit = True
                exit_reason = 'take_profit'
            elif i >= position_entry_idx + horizon:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                exit_price = current_price
                entry_cost = entry_price * (1 + (fee_bps + slippage_bps) / 10000)
                exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

                trade_return = (exit_proceeds / entry_cost - 1)
                pnl = trade_return * initial_capital
                cash += initial_capital * (exit_proceeds / entry_cost)

                trades.append({
                    'entry_date': position_entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'days_held': (current_date - position_entry_date).days
                })

                position_open = False

        # Enter new position if signal
        if predictions[i] == 1 and not position_open:
            if i + 1 < len(test_dates):
                position_entry_idx = i + 1
                position_entry_date = test_dates[position_entry_idx]
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
        exit_date = test_dates[-1]
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
            'exit_reason': 'final_close',
            'days_held': (exit_date - position_entry_date).days
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
    avg_days_held = trades_df['days_held'].mean() if n_trades > 0 else 0

    exit_reasons = trades_df['exit_reason'].value_counts().to_dict() if n_trades > 0 else {}

    return {
        'horizon': horizon,
        'threshold': threshold,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return,
        'avg_days_held': avg_days_held,
        'exit_reasons': exit_reasons,
        'equity_curve': equity_df,
        'trades': trades_df,
        'cv_score': model_info['cv_score'],
        'test_acc': model_info['test_acc']
    }


# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: TRAINING MODELS FOR EACH HORIZON")
print("=" * 80)

trained_models = []

for horizon in HORIZONS:
    model_info = train_model_for_horizon(horizon)
    trained_models.append(model_info)

print("\n" + "=" * 80)
print("STEP 2: BACKTESTING EACH HORIZON")
print("=" * 80)

results = []

for model_info in trained_models:
    horizon = model_info['horizon']

    print(f"\n{'─'*80}")
    print(f"Backtesting {horizon}-day holding period")
    print(f"{'─'*80}")

    # Test with optimized threshold only
    result = run_backtest_with_optimizations(
        model_info,
        threshold=0.52,
        stop_loss_pct=None,
        take_profit_pct=None
    )

    print(f"   Threshold: 0.52")
    print(f"   Final Value: ${result['final_value']:,.2f}")
    print(f"   Total Return: {result['total_return']:.2%}")
    print(f"   Annualized Return: {result['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"   Trades: {result['n_trades']}")
    print(f"   Win Rate: {result['win_rate']:.2%}")
    print(f"   Avg Days Held: {result['avg_days_held']:.1f}")

    results.append({
        'name': f'{horizon}-Day (Threshold 0.52)',
        **result
    })

    # Also test with stop loss + take profit for longer horizons
    if horizon >= 7:
        result_sl_tp = run_backtest_with_optimizations(
            model_info,
            threshold=0.52,
            stop_loss_pct=0.05,
            take_profit_pct=0.08
        )

        print(f"\n   With Stop Loss (-5%) & Take Profit (+8%):")
        print(f"   Total Return: {result_sl_tp['total_return']:.2%}")
        print(f"   Sharpe Ratio: {result_sl_tp['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {result_sl_tp['max_drawdown']:.2%}")
        print(f"   Exit Reasons: {result_sl_tp['exit_reasons']}")

        results.append({
            'name': f'{horizon}-Day (SL/TP)',
            **result_sl_tp
        })

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("HOLDING PERIOD OPTIMIZATION RESULTS")
print("=" * 80)

results_df = pd.DataFrame([{
    'Strategy': r['name'],
    'Horizon': r['horizon'],
    'Threshold': r['threshold'],
    'Stop_Loss': f"{r['stop_loss_pct']*100:.0f}%" if r['stop_loss_pct'] else 'None',
    'Take_Profit': f"{r['take_profit_pct']*100:.0f}%" if r['take_profit_pct'] else 'None',
    'Total_Return': r['total_return'],
    'Ann_Return': r['annualized_return'],
    'Sharpe': r['sharpe_ratio'],
    'Max_DD': r['max_drawdown'],
    'Trades': r['n_trades'],
    'Win_Rate': r['win_rate'],
    'Avg_Days_Held': r['avg_days_held'],
    'CV_Score': r['cv_score'],
    'Test_Acc': r['test_acc']
} for r in results])

results_df = results_df.sort_values('Total_Return', ascending=False)

print("\n" + results_df[['Strategy', 'Total_Return', 'Ann_Return', 'Sharpe',
                         'Max_DD', 'Trades', 'Win_Rate']].to_string(index=False))

# Save results
results_df.to_csv(OUTPUTS_DIR / "holding_period_results.csv", index=False)
print(f"\n💾 Saved: outputs/holding_period_results.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity curves for base strategies (no SL/TP)
ax = axes[0, 0]
for result in results:
    if 'SL/TP' not in result['name']:
        equity_df = result['equity_curve']
        ax.plot(equity_df['date'], equity_df['portfolio_value'],
                label=result['name'], linewidth=2)

ax.set_title('Equity Curves by Holding Period', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=BACKTEST_CFG['initial_capital'], color='black', linestyle='--', alpha=0.5)

# Plot 2: Returns vs Holding Period
ax = axes[0, 1]
base_strategies = results_df[~results_df['Strategy'].str.contains('SL/TP')]
horizons_plot = base_strategies['Horizon'].values
ann_returns = base_strategies['Ann_Return'].values * 100

ax.plot(horizons_plot, ann_returns, 'o-', linewidth=2, markersize=10, color='green')
ax.set_xlabel('Holding Period (Days)')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Annualized Return vs Holding Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=5.46, color='red', linestyle='--', label='Current (5d)', alpha=0.7)
ax.legend()

# Plot 3: Sharpe vs Holding Period
ax = axes[1, 0]
sharpes = base_strategies['Sharpe'].values

ax.plot(horizons_plot, sharpes, 'o-', linewidth=2, markersize=10, color='blue')
ax.set_xlabel('Holding Period (Days)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio vs Holding Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.40, color='red', linestyle='--', label='Current (5d)', alpha=0.7)
ax.legend()

# Plot 4: Max Drawdown vs Holding Period
ax = axes[1, 1]
drawdowns = abs(base_strategies['Max_DD'].values * 100)

ax.plot(horizons_plot, drawdowns, 'o-', linewidth=2, markersize=10, color='red')
ax.set_xlabel('Holding Period (Days)')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Max Drawdown vs Holding Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=14.69, color='blue', linestyle='--', label='Current (5d)', alpha=0.7)
ax.legend()
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'holding_period_optimization.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/holding_period_optimization.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

baseline = results_df[results_df['Strategy'] == '5-Day (Threshold 0.52)'].iloc[0]
best = results_df.iloc[0]

print(f"\n📊 BASELINE (5-day holding):")
print(f"   Total Return: {baseline['Total_Return']:.2%}")
print(f"   Annualized Return: {baseline['Ann_Return']:.2%}")
print(f"   Sharpe Ratio: {baseline['Sharpe']:.2f}")
print(f"   Max Drawdown: {baseline['Max_DD']:.2%}")
print(f"   Trades: {baseline['Trades']}")
print(f"   Win Rate: {baseline['Win_Rate']:.2%}")

print(f"\n🏆 BEST STRATEGY: {best['Strategy']}")
print(f"   Total Return: {best['Total_Return']:.2%}")
print(f"   Annualized Return: {best['Ann_Return']:.2%}")
print(f"   Sharpe Ratio: {best['Sharpe']:.2f}")
print(f"   Max Drawdown: {best['Max_DD']:.2%}")
print(f"   Trades: {best['Trades']}")
print(f"   Win Rate: {best['Win_Rate']:.2%}")

if best['Strategy'] != baseline['Strategy']:
    print(f"\n📈 IMPROVEMENT:")
    print(f"   Total Return: {(best['Total_Return'] - baseline['Total_Return'])*100:+.2f}pp")
    print(f"   Annualized Return: {(best['Ann_Return'] - baseline['Ann_Return'])*100:+.2f}pp")
    print(f"   Sharpe Ratio: {(best['Sharpe'] - baseline['Sharpe']):+.2f}")
    print(f"   Max Drawdown: {(best['Max_DD'] - baseline['Max_DD'])*100:+.2f}pp")
    print(f"   Win Rate: {(best['Win_Rate'] - baseline['Win_Rate'])*100:+.2f}pp")
else:
    print(f"\n✅ Current 5-day holding period is optimal!")

print("\n" + "=" * 80)
print("✅ HOLDING PERIOD OPTIMIZATION COMPLETE!")
print("=" * 80)
