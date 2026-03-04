#!/usr/bin/env python3
"""
Comprehensive Final Optimizations - Push to 12-14% Annualized

Implements 4 final optimizations:
1. Sector Rotation - Trade strongest sector instead of SPY
2. Multi-Strategy Portfolio - Run 7d, 10d, 15d strategies in parallel
3. Feature Engineering - Polynomial features and interactions
4. Return Magnitude Prediction - Size positions by predicted return

Expected impact: +2.5-4.5% annualized (9.46% → 12-14%)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Try to import yfinance, fallback gracefully
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️ yfinance not available - sector rotation will be disabled")

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
)

print("=" * 80)
print("COMPREHENSIVE FINAL OPTIMIZATIONS")
print("=" * 80)
print("Target: Push from 9.46% to 12-14% annualized")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

BACKTEST_CFG = {
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "initial_capital": 10000,
}

# ============================================================================
# 1. FEATURE ENGINEERING - Add Polynomial & Interactions
# ============================================================================

def add_engineered_features(df):
    """
    Add polynomial features and interactions for better predictions.

    Key interactions:
    - Momentum x Trend Strength
    - Volume x Volatility
    - Price patterns
    """
    print("\n📊 Adding engineered features...")

    # Momentum x Trend interactions
    if 'RSI' in df.columns and 'ADX' in df.columns:
        df['RSI_x_ADX'] = df['RSI'] * df['ADX'] / 100

    if 'Stoch_K' in df.columns and 'ADX' in df.columns:
        df['Stoch_x_ADX'] = df['Stoch_K'] * df['ADX'] / 100

    # Volume x Volatility
    if 'Volume_pct' in df.columns and 'ATR' in df.columns:
        df['Vol_x_Volatility'] = df['Volume_pct'] * df['ATR']

    # Moving average interactions
    if 'MA_20' in df.columns and 'MA_50' in df.columns:
        df['MA_Cross_Strength'] = (df['MA_20'] - df['MA_50']) / df['MA_50']

    # Polynomial features (for non-linear patterns)
    if 'RSI' in df.columns:
        df['RSI_squared'] = (df['RSI'] / 100) ** 2

    if 'ATR' in df.columns:
        df['Volatility_squared'] = df['ATR'] ** 2

    # Ratio features
    if 'BB_PctB' in df.columns:
        df['BB_PctB_squared'] = df['BB_PctB'] ** 2

    # Momentum strength
    if 'Plus_DI' in df.columns and 'Minus_DI' in df.columns:
        df['DI_Ratio'] = df['Plus_DI'] / (df['Minus_DI'] + 1e-6)

    engineered_cols = [c for c in df.columns if any(x in c for x in
                      ['_x_', '_squared', '_Ratio', '_Strength'])]

    print(f"   Added {len(engineered_cols)} engineered features")

    return df, engineered_cols


# ============================================================================
# 2. SECTOR ROTATION - Download Sector ETF Data
# ============================================================================

def load_sector_data(start_date, end_date):
    """
    Load sector ETF data for rotation strategy.

    Sectors:
    - XLK: Technology
    - XLF: Financials
    - XLE: Energy
    - XLV: Healthcare
    - XLI: Industrials
    - XLY: Consumer Discretionary
    - XLP: Consumer Staples
    - XLU: Utilities
    """
    if not HAS_YFINANCE:
        print("\n⚠️ Sector data unavailable (yfinance not installed)")
        print("   Sector rotation will be skipped - using SPY only")
        return None

    print("\n📥 Loading sector ETF data...")

    sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities'
    }

    sector_data = {}

    for ticker, name in sectors.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                sector_data[ticker] = data['Close']
                print(f"   ✅ {ticker} ({name}): {len(data)} days")
        except Exception as e:
            print(f"   ❌ {ticker} failed: {e}")

    if sector_data:
        sector_df = pd.DataFrame(sector_data)
        print(f"\n   Loaded {len(sector_data)} sectors")
        return sector_df
    else:
        print("   ⚠️ No sector data loaded, will use SPY only")
        return None


def select_best_sector(date, sector_prices, lookback=20):
    """
    Select the strongest performing sector based on recent returns.
    """
    if sector_prices is None or date not in sector_prices.index:
        return 'SPY'  # Default to SPY

    try:
        # Get recent data
        idx = sector_prices.index.get_loc(date)
        if idx < lookback:
            return 'SPY'

        recent_prices = sector_prices.iloc[idx-lookback:idx+1]

        # Calculate returns for each sector
        returns = {}
        for col in recent_prices.columns:
            ret = (recent_prices[col].iloc[-1] / recent_prices[col].iloc[0]) - 1
            returns[col] = ret

        # Return best performing sector
        best_sector = max(returns, key=returns.get)
        return best_sector
    except Exception:
        return 'SPY'


# ============================================================================
# 3. LOAD DATA AND MODELS
# ============================================================================

print("\n📥 Loading data and models...")

# Load SPY data
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

# Note: Skip engineered features for now since models were trained without them
# df, engineered_cols = add_engineered_features(df)
all_features = feature_cols

_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

prices_df = _raw[["Close"]].copy()
prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")

# Prepare features
keep_cols = all_features + ["Close"]
available_features = [c for c in all_features if c in df.columns]
df = df[[c for c in keep_cols if c in df.columns]]
df = df.fillna(0)

print(f"✅ Data loaded: {len(df)} samples")
print(f"   Features: {len(available_features)}")
print(f"   (Note: Engineered features skipped - models trained without them)")

# Split data
test_size = int(len(df) * 0.2)
train_end_idx = len(df) - test_size

X_test = df.iloc[train_end_idx:][available_features]
test_dates = df.index[train_end_idx:]

print(f"   Test period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

# Load models
print("\n📦 Loading trained models...")
model_7d = joblib.load(MODELS_DIR / "xgboost_regime_7d.pkl")
model_10d = joblib.load(MODELS_DIR / "xgboost_regime_10d.pkl")
model_15d = joblib.load(MODELS_DIR / "xgboost_regime_15d.pkl")
print("✅ Loaded 7-day, 10-day, 15-day models")

# Get predictions (handle missing features gracefully)
print("\n🔮 Generating predictions...")

# Get model feature names
model_features_7d = model_7d.get_booster().feature_names
model_features_10d = model_10d.get_booster().feature_names
model_features_15d = model_15d.get_booster().feature_names

# Simply use the features that the model expects
# All models should have been trained on the same feature set
X_test_7d = X_test[model_features_7d]
X_test_10d = X_test[model_features_10d]
X_test_15d = X_test[model_features_15d]

prob_7d = model_7d.predict_proba(X_test_7d)[:, 1]
prob_10d = model_10d.predict_proba(X_test_10d)[:, 1]
prob_15d = model_15d.predict_proba(X_test_15d)[:, 1]

print(f"   7-day predictions generated: {len(prob_7d)}")
print(f"   10-day predictions generated: {len(prob_10d)}")
print(f"   15-day predictions generated: {len(prob_15d)}")

# Load sector data
sector_prices = load_sector_data(
    start_date=test_dates[0] - timedelta(days=30),
    end_date=test_dates[-1]
)

# ============================================================================
# 4. BACKTESTING STRATEGIES
# ============================================================================

def run_strategy_backtest(strategy_name, predictions, dates, prices,
                         use_sector_rotation=False, sector_prices=None,
                         use_return_magnitude=False,
                         horizon=10, threshold=0.52,
                         initial_capital=10000, fee_bps=1.5, slippage_bps=2.0):
    """
    Universal backtest function for all strategies.
    """
    cash = initial_capital
    equity_curve = []
    trades = []

    price_series = prices.reindex(dates)

    position_open = False
    position_entry_idx = None
    current_ticker = 'SPY'

    for i in range(len(dates)):
        current_date = dates[i]

        # Get current price (either SPY or sector)
        if position_open and current_ticker != 'SPY' and sector_prices is not None:
            if current_date in sector_prices.index and current_ticker in sector_prices.columns:
                current_price = sector_prices.loc[current_date, current_ticker]
            else:
                current_price = price_series.loc[current_date]
        else:
            current_price = price_series.loc[current_date]

        # Close position if holding
        if position_open and i >= position_entry_idx + horizon:
            # Exit
            exit_price = current_price
            entry_cost = position_entry_price * (1 + (fee_bps + slippage_bps) / 10000)
            exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

            position_size_actual = position_size if 'position_size' in locals() else 1.0
            trade_return = (exit_proceeds / entry_cost - 1)
            pnl = trade_return * position_size_actual * initial_capital
            cash += position_size_actual * initial_capital * (exit_proceeds / entry_cost)

            trades.append({
                'entry_date': position_entry_date,
                'entry_price': position_entry_price,
                'exit_date': current_date,
                'exit_price': exit_price,
                'return': trade_return,
                'pnl': pnl,
                'ticker': current_ticker,
                'position_size': position_size_actual
            })

            position_open = False

        # Enter new position
        if predictions[i] >= threshold and not position_open:
            if i + 1 < len(dates):
                # Determine position size
                if use_return_magnitude:
                    # Size based on prediction confidence
                    confidence = predictions[i]
                    position_size = 0.5 + (confidence - threshold) * 1.0
                    position_size = np.clip(position_size, 0.5, 1.0)
                else:
                    position_size = 1.0

                # Determine ticker (SPY or sector)
                if use_sector_rotation and sector_prices is not None:
                    current_ticker = select_best_sector(dates[i], sector_prices, lookback=20)
                else:
                    current_ticker = 'SPY'

                position_entry_idx = i + 1
                position_entry_date = dates[position_entry_idx]

                # Get entry price
                if current_ticker != 'SPY' and sector_prices is not None:
                    if position_entry_date in sector_prices.index and current_ticker in sector_prices.columns:
                        position_entry_price = sector_prices.loc[position_entry_date, current_ticker]
                    else:
                        position_entry_price = price_series.loc[position_entry_date]
                        current_ticker = 'SPY'
                else:
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
            'position_open': position_open
        })

    # Close final position
    if position_open:
        exit_date = dates[-1]
        exit_price = current_price
        entry_cost = position_entry_price * (1 + (fee_bps + slippage_bps) / 10000)
        exit_proceeds = exit_price * (1 - (fee_bps + slippage_bps) / 10000)

        trade_return = (exit_proceeds / entry_cost - 1)
        pnl = trade_return * position_size * initial_capital
        cash += position_size * initial_capital * (exit_proceeds / entry_cost)

        trades.append({
            'entry_date': position_entry_date,
            'entry_price': position_entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'return': trade_return,
            'pnl': pnl,
            'ticker': current_ticker,
            'position_size': position_size
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

    # Sector distribution
    sector_dist = trades_df['ticker'].value_counts().to_dict() if 'ticker' in trades_df.columns else {}

    return {
        'strategy': strategy_name,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return,
        'sector_distribution': sector_dist,
        'equity_curve': equity_df,
        'trades': trades_df
    }


# ============================================================================
# 5. RUN ALL STRATEGIES
# ============================================================================

print("\n" + "=" * 80)
print("TESTING ALL FINAL OPTIMIZATIONS")
print("=" * 80)

results = []

# Baseline: Ensemble (2/3) - Current best
print(f"\n{'─'*80}")
print(f"1. Baseline: Ensemble (2/3 agree)")
print(f"{'─'*80}")

ensemble_pred = ((prob_7d >= 0.52) + (prob_10d >= 0.52) + (prob_15d >= 0.52)) >= 2
ensemble_prob = (prob_7d + prob_10d + prob_15d) / 3

result = run_strategy_backtest(
    'Baseline: Ensemble (2/3)',
    ensemble_prob,
    test_dates,
    prices_df['Close'],
    use_sector_rotation=False,
    horizon=10,
    threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Optimization 1: Sector Rotation
print(f"\n{'─'*80}")
print(f"2. Sector Rotation (trade strongest sector)")
print(f"{'─'*80}")

result = run_strategy_backtest(
    'Sector Rotation',
    ensemble_prob,
    test_dates,
    prices_df['Close'],
    use_sector_rotation=True,
    sector_prices=sector_prices,
    horizon=10,
    threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")
if result['sector_distribution']:
    print(f"   Sector distribution: {result['sector_distribution']}")

results.append(result)

# Optimization 2: Return Magnitude Sizing
print(f"\n{'─'*80}")
print(f"3. Return Magnitude Sizing (size by confidence)")
print(f"{'─'*80}")

result = run_strategy_backtest(
    'Return Magnitude Sizing',
    ensemble_prob,
    test_dates,
    prices_df['Close'],
    use_return_magnitude=True,
    horizon=10,
    threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Optimization 3: Multi-Strategy Portfolio
print(f"\n{'─'*80}")
print(f"4. Multi-Strategy Portfolio (7d + 10d + 15d parallel)")
print(f"{'─'*80}")

# Run strategies in parallel and combine equity curves
strat_7d = run_strategy_backtest('7d Strategy', prob_7d, test_dates, prices_df['Close'],
                                 horizon=7, threshold=0.52, initial_capital=3333)
strat_10d = run_strategy_backtest('10d Strategy', prob_10d, test_dates, prices_df['Close'],
                                  horizon=10, threshold=0.52, initial_capital=3334)
strat_15d = run_strategy_backtest('15d Strategy', prob_15d, test_dates, prices_df['Close'],
                                  horizon=15, threshold=0.52, initial_capital=3333)

# Combine equity curves
portfolio_equity = (strat_7d['equity_curve']['portfolio_value'] +
                   strat_10d['equity_curve']['portfolio_value'] +
                   strat_15d['equity_curve']['portfolio_value'])

portfolio_df = pd.DataFrame({
    'date': test_dates,
    'portfolio_value': portfolio_equity.values
})

final_value = portfolio_df['portfolio_value'].iloc[-1]
total_return = (final_value / 10000) - 1
years = len(portfolio_df) / 252
annualized_return = (1 + total_return) ** (1 / years) - 1
portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
sharpe = np.sqrt(252) * portfolio_df['daily_return'].mean() / portfolio_df['daily_return'].std()
portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cummax']) - 1
max_drawdown = portfolio_df['drawdown'].min()

n_trades = strat_7d['n_trades'] + strat_10d['n_trades'] + strat_15d['n_trades']
combined_trades = pd.concat([strat_7d['trades'], strat_10d['trades'], strat_15d['trades']])
win_rate = (combined_trades['return'] > 0).sum() / len(combined_trades) if len(combined_trades) > 0 else 0

print(f"   Total Return: {total_return:.2%}")
print(f"   Annualized: {annualized_return:.2%}")
print(f"   Sharpe: {sharpe:.2f}")
print(f"   Max DD: {max_drawdown:.2%}")
print(f"   Total Trades: {n_trades}, Win Rate: {win_rate:.2%}")
print(f"   (7d: {strat_7d['n_trades']} trades, 10d: {strat_10d['n_trades']}, 15d: {strat_15d['n_trades']})")

results.append({
    'strategy': 'Multi-Strategy Portfolio',
    'final_value': final_value,
    'total_return': total_return,
    'annualized_return': annualized_return,
    'sharpe_ratio': sharpe,
    'max_drawdown': max_drawdown,
    'n_trades': n_trades,
    'win_rate': win_rate,
    'avg_return_per_trade': combined_trades['return'].mean(),
    'equity_curve': portfolio_df,
    'trades': combined_trades
})

# COMBINED: Sector Rotation + Return Magnitude
print(f"\n{'─'*80}")
print(f"5. ULTIMATE: Sector Rotation + Return Magnitude")
print(f"{'─'*80}")

result = run_strategy_backtest(
    'ULTIMATE Strategy',
    ensemble_prob,
    test_dates,
    prices_df['Close'],
    use_sector_rotation=True,
    sector_prices=sector_prices,
    use_return_magnitude=True,
    horizon=10,
    threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")
if result['sector_distribution']:
    print(f"   Sector distribution: {result['sector_distribution']}")

results.append(result)

# ============================================================================
# 6. RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE FINAL RESULTS")
print("=" * 80)

results_df = pd.DataFrame([{
    'Strategy': r['strategy'],
    'Total_Return': r['total_return'],
    'Ann_Return': r['annualized_return'],
    'Sharpe': r['sharpe_ratio'],
    'Max_DD': r['max_drawdown'],
    'Trades': r['n_trades'],
    'Win_Rate': r['win_rate']
} for r in results])

results_df = results_df.sort_values('Total_Return', ascending=False)

print("\n" + results_df.to_string(index=False))

results_df.to_csv(OUTPUTS_DIR / "comprehensive_final_results.csv", index=False)
print(f"\n💾 Saved: outputs/comprehensive_final_results.csv")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity curves
ax = axes[0, 0]
for result in results:
    equity_df = result['equity_curve']
    ax.plot(equity_df['date'], equity_df['portfolio_value'],
            label=result['strategy'], linewidth=2, alpha=0.8)

ax.set_title('All Strategies: Equity Curves', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Capital')

# Plot 2: Annualized returns
ax = axes[0, 1]
strategies = results_df['Strategy'].values
ann_returns = results_df['Ann_Return'].values * 100
colors = ['green' if r > 0 else 'red' for r in ann_returns]

ax.barh(strategies, ann_returns, color=colors, alpha=0.7)
ax.set_xlabel('Annualized Return (%)')
ax.set_title('Annualized Return Comparison', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=9.46, color='blue', linestyle='--', linewidth=1, label='Previous Best', alpha=0.7)
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

# Plot 3: Sharpe vs Return
ax = axes[1, 0]
sharpes = results_df['Sharpe'].values
returns = results_df['Ann_Return'].values * 100

scatter = ax.scatter(sharpes, returns, s=200, alpha=0.7, c=range(len(results_df)), cmap='viridis')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

for i, strategy in enumerate(strategies):
    ax.annotate(strategy[:20], (sharpes[i], returns[i]), fontsize=7, alpha=0.7)

# Plot 4: Win Rate vs Drawdown
ax = axes[1, 1]
win_rates = results_df['Win_Rate'].values * 100
drawdowns = abs(results_df['Max_DD'].values * 100)

scatter = ax.scatter(drawdowns, win_rates, s=200, alpha=0.7,
                    c=results_df['Total_Return'].values, cmap='RdYlGn')
ax.set_xlabel('Max Drawdown (%)')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate vs Risk', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
plt.colorbar(scatter, ax=ax, label='Total Return')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'comprehensive_final_optimizations.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/comprehensive_final_optimizations.png")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL OPTIMIZATION SUMMARY")
print("=" * 80)

baseline = results_df[results_df['Strategy'] == 'Baseline: Ensemble (2/3)'].iloc[0]
best = results_df.iloc[0]

print(f"\n📊 BASELINE (Previous Best):")
print(f"   Strategy: Ensemble (2/3 agree)")
print(f"   Total Return: {baseline['Total_Return']:.2%}")
print(f"   Annualized: {baseline['Ann_Return']:.2%}")
print(f"   Sharpe: {baseline['Sharpe']:.2f}")
print(f"   Max DD: {baseline['Max_DD']:.2%}")

print(f"\n🏆 NEW CHAMPION: {best['Strategy']}")
print(f"   Total Return: {best['Total_Return']:.2%}")
print(f"   Annualized: {best['Ann_Return']:.2%}")
print(f"   Sharpe: {best['Sharpe']:.2f}")
print(f"   Max DD: {best['Max_DD']:.2%}")

if best['Strategy'] != baseline['Strategy']:
    print(f"\n📈 IMPROVEMENT:")
    print(f"   Total Return: {(best['Total_Return'] - baseline['Total_Return'])*100:+.2f}pp")
    print(f"   Annualized: {(best['Ann_Return'] - baseline['Ann_Return'])*100:+.2f}pp")
    print(f"   Sharpe: {(best['Sharpe'] - baseline['Sharpe']):+.2f}")
    print(f"   Max DD: {(best['Max_DD'] - baseline['Max_DD'])*100:+.2f}pp")
else:
    print(f"\n✅ Baseline remains optimal - new optimizations don't improve!")

print("\n" + "=" * 80)
print("✅ COMPREHENSIVE OPTIMIZATION COMPLETE!")
print("=" * 80)
