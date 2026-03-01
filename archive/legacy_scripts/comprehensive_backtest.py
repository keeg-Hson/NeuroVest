#!/usr/bin/env python3
"""
Comprehensive Backtest for Market Crash Prediction Models

Simulates realistic trading strategies using trained models.
Compares all model configurations on the same test set.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)
from train import TRAIN_CFG

print("=" * 80)
print("COMPREHENSIVE BACKTEST - ALL MODELS")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Backtest configuration
BACKTEST_CFG = {
    "fee_bps": 1.5,          # Trading fees (basis points)
    "slippage_bps": 2.0,     # Slippage (basis points)
    "initial_capital": 10000,  # Starting capital
    "position_size": 1.0,    # Fraction of capital per trade
}

print(f"\n📋 Backtest Configuration:")
print(f"   Initial Capital: ${BACKTEST_CFG['initial_capital']:,.0f}")
print(f"   Position Size: {BACKTEST_CFG['position_size']:.0%}")
print(f"   Trading Fees: {BACKTEST_CFG['fee_bps']} bps")
print(f"   Slippage: {BACKTEST_CFG['slippage_bps']} bps")

# ============================================================================
# 1. PREPARE DATA
# ============================================================================

print("\n📥 Loading data...")
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

# Reindex Close
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

# Add forward returns
df = add_forward_returns_and_labels(
    df,
    price_col="Close",
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)

# Store prices for backtest
prices_df = _raw[["Close"]].copy()
prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")

# Get features
regime_features = [f for f in feature_cols if any(x in f for x in
    ['MA_200', 'Bull_Market', 'ADX', 'Plus_DI', 'Minus_DI',
     'High_Volatility', 'Regime_Score', 'Near_52w', 'Trend_Consistency'])]
all_features = [c for c in df.columns if c not in ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]
base_features = [f for f in all_features if f not in regime_features]

# Prepare data
keep_cols = all_features + ["y", "fwd_ret_net"]
df = df[keep_cols]
df = df.dropna(subset=["y"])
df = df.fillna(0)

print(f"✅ Data prepared: {len(df)} rows")
print(f"   Total features: {len(all_features)}")
print(f"   Base features: {len(base_features)}")
print(f"   Regime features: {len(regime_features)}")

# Split data (80/20)
test_size = int(len(df) * 0.2)
train_end_idx = len(df) - test_size

X_train = df.iloc[:train_end_idx][all_features]
X_test = df.iloc[train_end_idx:][all_features]
y_test = df.iloc[train_end_idx:]["y"]

# Get test dates
test_dates = df.index[train_end_idx:]

print(f"\n📅 Backtest Period:")
print(f"   Start: {test_dates[0].strftime('%Y-%m-%d')}")
print(f"   End: {test_dates[-1].strftime('%Y-%m-%d')}")
print(f"   Days: {len(test_dates)}")

# ============================================================================
# 2. LOAD MODELS AND GENERATE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING MODELS AND GENERATING PREDICTIONS")
print("=" * 80)

models_to_test = []

# Model 1: XGBoost with Regime Features
try:
    print("\n[1] Loading XGBoost (Regime)...")
    xgb_regime = joblib.load(MODELS_DIR / "xgboost_regime.pkl")
    xgb_regime_proba = xgb_regime.predict_proba(X_test)[:, 1]
    models_to_test.append({
        'name': 'XGBoost (Regime)',
        'probabilities': xgb_regime_proba,
        'threshold': 0.5,
        'features': len(all_features)
    })
    print("✅ Loaded successfully")
except Exception as e:
    print(f"❌ Failed: {e}")

# Model 2: LightGBM with Regime Features
try:
    print("\n[2] Loading LightGBM (Regime)...")
    lgb_regime = joblib.load(MODELS_DIR / "lightgbm_regime.pkl")
    lgb_regime_proba = lgb_regime.predict_proba(X_test)[:, 1]
    models_to_test.append({
        'name': 'LightGBM (Regime)',
        'probabilities': lgb_regime_proba,
        'threshold': 0.5,
        'features': len(all_features)
    })
    print("✅ Loaded successfully")
except Exception as e:
    print(f"❌ Failed: {e}")

# Model 3: LightGBM with Regime (Profit-Optimized)
try:
    print("\n[3] Loading LightGBM (Regime, Profit-Opt @ 0.75)...")
    models_to_test.append({
        'name': 'LightGBM (Regime, Profit-Opt)',
        'probabilities': lgb_regime_proba,  # Same model, different threshold
        'threshold': 0.75,
        'features': len(all_features)
    })
    print("✅ Configured successfully")
except Exception:
    print(f"❌ Failed")

# Model 4: Ensemble
try:
    print("\n[4] Loading Ensemble (XGB + LGB + CatBoost)...")
    cat_regime = joblib.load(MODELS_DIR / "catboost_regime.pkl")
    cat_regime_proba = cat_regime.predict_proba(X_test)[:, 1]
    ensemble_proba = (xgb_regime_proba + lgb_regime_proba + cat_regime_proba) / 3
    models_to_test.append({
        'name': 'Ensemble (Regime)',
        'probabilities': ensemble_proba,
        'threshold': 0.5,
        'features': len(all_features)
    })
    print("✅ Loaded successfully")
except Exception as e:
    print(f"❌ Failed: {e}")

# Model 5: XGBoost Improved (78 features)
try:
    print("\n[5] Loading XGBoost (Improved, 78 features)...")
    xgb_improved_data = joblib.load(MODELS_DIR / "market_crash_model_fwd_improved.pkl")
    if isinstance(xgb_improved_data, dict):
        xgb_improved = xgb_improved_data['model']
        xgb_improved_features = xgb_improved_data['features']
        X_test_improved = X_test[xgb_improved_features]
        xgb_improved_proba = xgb_improved.predict_proba(X_test_improved)[:, 1]

        # Default threshold
        models_to_test.append({
            'name': 'XGBoost (Improved)',
            'probabilities': xgb_improved_proba,
            'threshold': 0.5,
            'features': len(xgb_improved_features)
        })

        # Profit-optimized threshold
        models_to_test.append({
            'name': 'XGBoost (Improved, Profit-Opt)',
            'probabilities': xgb_improved_proba,
            'threshold': 0.65,
            'features': len(xgb_improved_features)
        })
        print("✅ Loaded successfully")
except Exception as e:
    print(f"❌ Failed: {e}")

# Buy and hold baseline
models_to_test.append({
    'name': 'Buy & Hold SPY',
    'probabilities': np.ones(len(X_test)),  # Always buy
    'threshold': 0.5,
    'features': 0
})

print(f"\n✅ Total models to backtest: {len(models_to_test)}")

# ============================================================================
# 3. RUN BACKTEST FOR EACH MODEL
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING BACKTESTS")
print("=" * 80)

def run_backtest(predictions, dates, prices, config):
    """
    Run backtest simulation.

    Strategy:
    - When prediction = 1 (crash predicted), enter position at next day's open
    - Hold for HORIZON days (from TRAIN_CFG)
    - Exit at close on the exit day
    - Account for fees and slippage
    """
    horizon = TRAIN_CFG["horizon"]
    fee_bps = config["fee_bps"]
    slippage_bps = config["slippage_bps"]
    initial_capital = config["initial_capital"]
    position_size = config["position_size"]

    # Track portfolio
    cash = initial_capital
    equity_curve = []
    trades = []

    # Ensure prices and predictions are aligned
    price_series = prices.reindex(dates)

    # Track open positions
    position_open = False
    position_entry_idx = None

    for i in range(len(dates)):
        current_date = dates[i]
        current_price = price_series.loc[current_date]

        # Close position if holding and reached exit
        if position_open and i >= position_entry_idx + horizon:
            # Exit at current close
            exit_price = current_price
            entry_price = position_entry_price

            # Calculate return with fees and slippage
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
                'pnl': pnl
            })

            position_open = False

        # Enter new position if signal and not currently holding
        if predictions[i] == 1 and not position_open:
            # Enter at next day if available
            if i + 1 < len(dates):
                position_entry_idx = i + 1
                position_entry_date = dates[position_entry_idx]
                position_entry_price = price_series.loc[position_entry_date]

                # Deduct position from cash
                cash -= position_size * initial_capital

                position_open = True

        # Calculate portfolio value
        portfolio_value = cash
        if position_open:
            # Mark to market
            unrealized_value = position_size * initial_capital * (current_price / position_entry_price)
            portfolio_value = cash + unrealized_value

        equity_curve.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'position_open': position_open
        })

    # Close any remaining position
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
            'pnl': pnl
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return equity_df, trades_df

def calculate_metrics(equity_df, trades_df, initial_capital):
    """Calculate performance metrics"""
    final_value = equity_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1

    # Annualized return
    days = len(equity_df)
    years = days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe ratio (assuming daily returns)
    equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
    sharpe = np.sqrt(252) * equity_df['daily_return'].mean() / equity_df['daily_return'].std() if equity_df['daily_return'].std() > 0 else 0

    # Max drawdown
    equity_df['cummax'] = equity_df['portfolio_value'].cummax()
    equity_df['drawdown'] = (equity_df['portfolio_value'] / equity_df['cummax']) - 1
    max_drawdown = equity_df['drawdown'].min()

    # Trade statistics
    n_trades = len(trades_df)
    win_rate = (trades_df['return'] > 0).sum() / n_trades if n_trades > 0 else 0
    avg_return = trades_df['return'].mean() if n_trades > 0 else 0
    avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if (trades_df['return'] > 0).any() else 0
    avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if (trades_df['return'] < 0).any() else 0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'days': days
    }

# Run backtests
results = []

for model_cfg in models_to_test:
    print(f"\n{'─' * 80}")
    print(f"Backtesting: {model_cfg['name']}")
    print(f"{'─' * 80}")

    # Generate predictions
    predictions = (model_cfg['probabilities'] >= model_cfg['threshold']).astype(int)

    print(f"   Threshold: {model_cfg['threshold']}")
    print(f"   Signals: {predictions.sum()} / {len(predictions)} ({predictions.sum()/len(predictions):.1%})")

    # Run backtest
    equity_df, trades_df = run_backtest(predictions, test_dates, prices_df['Close'], BACKTEST_CFG)
    metrics = calculate_metrics(equity_df, trades_df, BACKTEST_CFG['initial_capital'])

    print(f"\n   Final Value: ${metrics['final_value']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.2%}")
    print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   Trades: {metrics['n_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.2%}")

    results.append({
        'Model': model_cfg['name'],
        'Features': model_cfg['features'],
        'Threshold': model_cfg['threshold'],
        **metrics,
        'equity_curve': equity_df,
        'trades': trades_df
    })

# ============================================================================
# 4. RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST RESULTS COMPARISON")
print("=" * 80)

results_df = pd.DataFrame([{
    'Model': r['Model'],
    'Features': r['Features'],
    'Threshold': r['Threshold'],
    'Final_Value': r['final_value'],
    'Total_Return': r['total_return'],
    'Annualized_Return': r['annualized_return'],
    'Sharpe_Ratio': r['sharpe_ratio'],
    'Max_Drawdown': r['max_drawdown'],
    'N_Trades': r['n_trades'],
    'Win_Rate': r['win_rate'],
    'Avg_Return_Per_Trade': r['avg_return_per_trade']
} for r in results])

# Sort by total return
results_df = results_df.sort_values('Total_Return', ascending=False)

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv(OUTPUTS_DIR / "backtest_results.csv", index=False)
print(f"\n💾 Saved: outputs/backtest_results.csv")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Plot equity curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for result in results:
    equity_df = result['equity_curve']
    ax1.plot(equity_df['date'], equity_df['portfolio_value'], label=result['Model'], linewidth=2)

ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=BACKTEST_CFG['initial_capital'], color='black', linestyle='--', label='Initial Capital', alpha=0.5)

# Plot returns comparison
models = results_df['Model'].values
returns = results_df['Total_Return'].values * 100
colors = ['green' if r > 0 else 'red' for r in returns]

ax2.barh(models, returns, color=colors, alpha=0.7)
ax2.set_xlabel('Total Return (%)')
ax2.set_title('Total Return Comparison', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'backtest_equity_curves.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/backtest_equity_curves.png")

# Plot drawdowns
fig, ax = plt.subplots(figsize=(14, 6))

for result in results:
    equity_df = result['equity_curve']
    if 'drawdown' in equity_df.columns:
        ax.plot(equity_df['date'], equity_df['drawdown'] * 100, label=result['Model'], linewidth=2)

ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.fill_between(equity_df['date'], 0, equity_df['drawdown'] * 100, alpha=0.1)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'backtest_drawdowns.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/backtest_drawdowns.png")

print("\n" + "=" * 80)
print("✅ BACKTEST COMPLETE!")
print("=" * 80)
print(f"\n📊 Best Model by Total Return: {results_df.iloc[0]['Model']}")
print(f"   Total Return: {results_df.iloc[0]['Total_Return']:.2%}")
print(f"   Sharpe Ratio: {results_df.iloc[0]['Sharpe_Ratio']:.2f}")
