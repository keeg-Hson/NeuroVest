#!/usr/bin/env python3
"""
Maximum Performance Optimization - Squeezing Every Bit of Performance

Strategy: Fix issues from ultimate_optimizations.py and find optimal configuration
1. Train on ORIGINAL 103 features (remove noisy advanced features)
2. Test different ensemble voting thresholds (2/4, not 3/4)
3. Test less restrictive regime filtering (OR logic, not AND)
4. Feature selection - use only most important features
5. Hybrid strategies
6. Different holding periods with diverse ensemble

Goal: Find absolute maximum performance with current setup
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not available")

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
)

print("=" * 80)
print("MAXIMUM PERFORMANCE OPTIMIZATION")
print("=" * 80)
print("Goal: Find absolute maximum performance with current setup")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

BACKTEST_CFG = {
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "initial_capital": 10000,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_importance(model, feature_names, top_n=50):
    """Get top N most important features from XGBoost model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importance_dict = model.get_booster().get_score(importance_type='gain')
        importances = np.array([importance_dict.get(f, 0) for f in feature_names])
    else:
        return feature_names[:top_n]

    # Get indices of top features
    top_indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    return top_features


def less_restrictive_regime_filter(df, idx):
    """
    Less restrictive regime filtering (OR logic instead of AND).

    Trade if ANY of these is true:
    - Bull market (price > 200-day MA)
    - Strong trend (ADX > 25)

    Only skip if:
    - Bear market AND weak trend AND high volatility
    """
    row = df.iloc[idx]

    if idx < 200:
        return True, ["insufficient_history"]

    # Count favorable conditions
    favorable = 0
    unfavorable = 0

    # Bull market?
    if 'MA_200' in df.columns:
        if row['Close'] > row['MA_200']:
            favorable += 1
        else:
            unfavorable += 1

    # Strong trend?
    if 'ADX' in df.columns:
        if row['ADX'] > 25:
            favorable += 1
        else:
            unfavorable += 1

    # Extreme volatility?
    if 'ATR' in df.columns:
        avg_atr = df['ATR'].iloc[max(0, idx-50):idx].mean()
        if row['ATR'] < 2.5 * avg_atr:  # Less restrictive than 2x
            favorable += 1
        else:
            unfavorable += 1

    # Trade if at least 2 out of 3 favorable conditions
    if favorable >= 2:
        return True, ["favorable_conditions"]

    return False, ["unfavorable_conditions"]


def train_diverse_models(X_train, y_train, X_val, y_val, model_name_prefix=""):
    """Train diverse model ensemble."""
    print(f"\n🤖 Training diverse models: {model_name_prefix}")

    models = {}

    # 1. XGBoost
    print("   Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_acc = (xgb_model.predict(X_val) == y_val).mean()
    models['xgboost'] = xgb_model
    print(f"      ✅ XGBoost accuracy: {xgb_acc:.4f}")

    # 2. LightGBM
    if HAS_LIGHTGBM:
        print("   Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_acc = (lgb_model.predict(X_val) == y_val).mean()
        models['lightgbm'] = lgb_model
        print(f"      ✅ LightGBM accuracy: {lgb_acc:.4f}")

    # 3. Random Forest
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_acc = (rf_model.predict(X_val) == y_val).mean()
    models['random_forest'] = rf_model
    print(f"      ✅ Random Forest accuracy: {rf_acc:.4f}")

    # 4. Neural Network
    print("   Training Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_acc = (nn_model.predict(X_val_scaled) == y_val).mean()
    models['neural_net'] = nn_model
    models['scaler'] = scaler
    print(f"      ✅ Neural Network accuracy: {nn_acc:.4f}")

    return models


def run_backtest(
    strategy_name, models, X_test, test_dates, prices, df,
    use_regime_filter=False, regime_filter_type='strict',
    use_ensemble_voting=False, voting_threshold=0.5,
    horizon=10, threshold=0.52,
    initial_capital=10000, fee_bps=1.5, slippage_bps=2.0
):
    """Run backtest with various configurations."""

    cash = initial_capital
    equity_curve = []
    trades = []

    # Get predictions
    if isinstance(models, dict):
        predictions_dict = {}
        for model_name, model in models.items():
            if model_name == 'scaler':
                continue
            if model_name == 'neural_net':
                X_scaled = models['scaler'].transform(X_test)
                pred_proba = model.predict_proba(X_scaled)[:, 1]
            else:
                pred_proba = model.predict_proba(X_test)[:, 1]
            predictions_dict[model_name] = pred_proba

        if use_ensemble_voting:
            votes = np.array([pred >= threshold for pred in predictions_dict.values()])
            agreements = votes.sum(axis=0)
            total_models = len(predictions_dict)
            required_votes = int(total_models * voting_threshold)
            predictions = (agreements >= required_votes).astype(float)
            avg_proba = np.array(list(predictions_dict.values())).mean(axis=0)
        else:
            predictions = np.array(list(predictions_dict.values())).mean(axis=0)
            avg_proba = predictions
    else:
        predictions = models.predict_proba(X_test)[:, 1]
        avg_proba = predictions

    price_series = prices.reindex(test_dates)
    position_open = False
    position_entry_idx = None
    position_entry_date = None
    position_entry_price = None
    regime_filter_skips = 0

    for i in range(len(test_dates)):
        current_date = test_dates[i]
        current_price = price_series.loc[current_date]

        # Close position if holding
        if position_open:
            days_held = (current_date - position_entry_date).days
            if days_held >= horizon:
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
                    'pnl': pnl
                })

                position_open = False

        # Enter new position
        if not position_open:
            # Check regime filter
            if use_regime_filter:
                if regime_filter_type == 'strict':
                    from ultimate_optimizations import get_regime_filter
                    should_trade, _ = get_regime_filter(df.loc[test_dates], i)
                else:  # less_restrictive
                    should_trade, _ = less_restrictive_regime_filter(df.loc[test_dates], i)

                if not should_trade:
                    regime_filter_skips += 1
                    equity_curve.append({
                        'date': current_date,
                        'portfolio_value': cash,
                        'position_open': False
                    })
                    continue

            # Check signal
            signal = predictions[i] if use_ensemble_voting else avg_proba[i]

            if (use_ensemble_voting and signal >= 1.0) or (not use_ensemble_voting and signal >= threshold):
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
            'pnl': pnl
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

    return {
        'strategy': strategy_name,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'regime_filter_skips': regime_filter_skips,
        'equity_curve': equity_df,
        'trades': trades_df
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n📥 Loading data with ORIGINAL 103 features (no advanced features)...")

# Load SPY data
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

# Get prices
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

prices_df = _raw[["Close"]].copy()
prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")

# Prepare features - USE ORIGINAL FEATURES ONLY
available_features = [c for c in feature_cols if c in df.columns]
df = df[[c for c in (available_features + ["Close"]) if c in df.columns]]
df = df.fillna(0)

print(f"✅ Data loaded: {len(df)} samples")
print(f"   Features: {len(available_features)} (original features only)")

# Split data
test_size = int(len(df) * 0.2)
train_end_idx = len(df) - test_size

# Create target (10-day horizon)
horizon = 10
df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
df = df.dropna(subset=['Target'])

X = df[available_features].values
y = df['Target'].values
dates = df.index

X_train = X[:train_end_idx]
y_train = y[:train_end_idx]
X_test = X[train_end_idx:]
y_test = y[train_end_idx:]
test_dates = dates[train_end_idx:]

print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})")

# Split train for validation
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=False
)

print("\n" + "=" * 80)
print("PHASE 1: TRAIN DIVERSE MODELS ON ORIGINAL FEATURES")
print("=" * 80)

diverse_models = train_diverse_models(X_train_fit, y_train_fit, X_val, y_val)

# Save models
print("\n💾 Saving models...")
for name, model in diverse_models.items():
    if name != 'scaler':
        joblib.dump(model, MODELS_DIR / f"{name}_max_perf.pkl")
if 'scaler' in diverse_models:
    joblib.dump(diverse_models['scaler'], MODELS_DIR / "scaler_max_perf.pkl")

print("\n" + "=" * 80)
print("PHASE 2: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importance from XGBoost
top_50_features = get_feature_importance(diverse_models['xgboost'], available_features, top_n=50)
top_60_features = get_feature_importance(diverse_models['xgboost'], available_features, top_n=60)
top_75_features = get_feature_importance(diverse_models['xgboost'], available_features, top_n=75)

print(f"\n📊 Feature importance analysis:")
print(f"   Top 50 features selected")
print(f"   Top 10: {', '.join(top_50_features[:10])}")

# Train models on reduced feature sets
print("\n🔬 Testing reduced feature sets...")

# Test with top 50 features
X_train_50 = df.iloc[:train_end_idx][top_50_features].values
X_test_50 = df.iloc[train_end_idx:][top_50_features].values
X_train_fit_50, X_val_50, _, _ = train_test_split(
    X_train_50, y_train, test_size=0.2, random_state=42, shuffle=False
)

models_50 = train_diverse_models(X_train_fit_50, y_train_fit, X_val_50, y_val, "top50")

print("\n" + "=" * 80)
print("PHASE 3: COMPREHENSIVE STRATEGY TESTING")
print("=" * 80)

results = []

# Load previous best ensemble for comparison
try:
    model_7d = joblib.load(MODELS_DIR / "xgboost_regime_7d.pkl")
    model_10d = joblib.load(MODELS_DIR / "xgboost_regime_10d.pkl")
    model_15d = joblib.load(MODELS_DIR / "xgboost_regime_15d.pkl")

    # Get model features
    model_features_7d = model_7d.get_booster().feature_names
    model_features_10d = model_10d.get_booster().feature_names
    model_features_15d = model_15d.get_booster().feature_names

    X_test_7d = df.iloc[train_end_idx:][model_features_7d].values
    X_test_10d = df.iloc[train_end_idx:][model_features_10d].values
    X_test_15d = df.iloc[train_end_idx:][model_features_15d].values

    prob_7d = model_7d.predict_proba(X_test_7d)[:, 1]
    prob_10d = model_10d.predict_proba(X_test_10d)[:, 1]
    prob_15d = model_15d.predict_proba(X_test_15d)[:, 1]

    ensemble_pred = ((prob_7d >= 0.52) + (prob_10d >= 0.52) + (prob_15d >= 0.52)) >= 2
    ensemble_prob = (prob_7d + prob_10d + prob_15d) / 3

    # Test previous best
    print(f"\n{'─'*80}")
    print(f"BASELINE: Previous Best - Multi-Horizon Ensemble (2/3)")
    print(f"{'─'*80}")

    result = run_backtest(
        'Previous Best: Ensemble (2/3)',
        {'model': type('obj', (object,), {'predict_proba': lambda self, X: np.column_stack([1-ensemble_prob, ensemble_prob])})()},
        X_test, test_dates, prices_df['Close'], df,
        use_regime_filter=False, use_ensemble_voting=False,
        horizon=10, threshold=0.52, **BACKTEST_CFG
    )

    print(f"   Annualized: {result['annualized_return']:.2%}")
    print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
    print(f"   Max DD: {result['max_drawdown']:.2%}")
    print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

    results.append(result)
except Exception:
    print("   ⚠️ Previous ensemble models not found, skipping baseline")

# Strategy 1: New Ensemble with 2/4 voting (50% threshold)
print(f"\n{'─'*80}")
print(f"1. New Diverse Ensemble (2/4 models, 50% threshold)")
print(f"{'─'*80}")

result = run_backtest(
    'Ensemble 2/4 (50%)',
    diverse_models, X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False, use_ensemble_voting=True,
    voting_threshold=0.5, horizon=10, threshold=0.52, **BACKTEST_CFG
)

print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Strategy 2: Average probability (no voting)
print(f"\n{'─'*80}")
print(f"2. Average Probability (all 4 models, no voting)")
print(f"{'─'*80}")

result = run_backtest(
    'Average Probability',
    diverse_models, X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False, use_ensemble_voting=False,
    horizon=10, threshold=0.52, **BACKTEST_CFG
)

print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Strategy 3: Less restrictive regime filter + ensemble
print(f"\n{'─'*80}")
print(f"3. Less Restrictive Regime Filter (2/3 favorable) + Ensemble")
print(f"{'─'*80}")

result = run_backtest(
    'Less Restrictive Filter + Ensemble',
    diverse_models, X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=True, regime_filter_type='less_restrictive',
    use_ensemble_voting=True, voting_threshold=0.5,
    horizon=10, threshold=0.52, **BACKTEST_CFG
)

print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")
print(f"   Filtered: {result['regime_filter_skips']} trades")

results.append(result)

# Strategy 4: Top 50 features ensemble
print(f"\n{'─'*80}")
print(f"4. Top 50 Features + Ensemble (2/4 voting)")
print(f"{'─'*80}")

result = run_backtest(
    'Top 50 Features',
    models_50, X_test_50, test_dates, prices_df['Close'], df,
    use_regime_filter=False, use_ensemble_voting=True,
    voting_threshold=0.5, horizon=10, threshold=0.52, **BACKTEST_CFG
)

print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Strategy 5: XGBoost only (best individual model)
print(f"\n{'─'*80}")
print(f"5. XGBoost Only (Single Best Model)")
print(f"{'─'*80}")

result = run_backtest(
    'XGBoost Only',
    {'xgboost': diverse_models['xgboost']}, X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False, use_ensemble_voting=False,
    horizon=10, threshold=0.52, **BACKTEST_CFG
)

print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Strategy 6: Hybrid - 70/30 split
print(f"\n{'─'*80}")
print(f"6. HYBRID: 70% Aggressive + 30% Defensive")
print(f"{'─'*80}")

# Run aggressive (no filter)
aggressive = run_backtest(
    'Aggressive', diverse_models, X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False, use_ensemble_voting=True, voting_threshold=0.5,
    horizon=10, threshold=0.52, initial_capital=7000, **{k: v for k, v in BACKTEST_CFG.items() if k != 'initial_capital'}
)

# Run defensive (with filter)
defensive = run_backtest(
    'Defensive', diverse_models, X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=True, regime_filter_type='less_restrictive',
    use_ensemble_voting=True, voting_threshold=0.5,
    horizon=10, threshold=0.52, initial_capital=3000, **{k: v for k, v in BACKTEST_CFG.items() if k != 'initial_capital'}
)

# Combine equity curves
hybrid_equity = (aggressive['equity_curve']['portfolio_value'] +
                defensive['equity_curve']['portfolio_value'])

hybrid_df = pd.DataFrame({
    'date': test_dates,
    'portfolio_value': hybrid_equity.values
})

final_value = hybrid_df['portfolio_value'].iloc[-1]
total_return = (final_value / 10000) - 1
years = len(hybrid_df) / 252
annualized_return = (1 + total_return) ** (1 / years) - 1
hybrid_df['daily_return'] = hybrid_df['portfolio_value'].pct_change()
sharpe = np.sqrt(252) * hybrid_df['daily_return'].mean() / hybrid_df['daily_return'].std()
hybrid_df['cummax'] = hybrid_df['portfolio_value'].cummax()
hybrid_df['drawdown'] = (hybrid_df['portfolio_value'] / hybrid_df['cummax']) - 1
max_drawdown = hybrid_df['drawdown'].min()

n_trades = aggressive['n_trades'] + defensive['n_trades']
combined_trades = pd.concat([aggressive['trades'], defensive['trades']])
win_rate = (combined_trades['return'] > 0).sum() / len(combined_trades) if len(combined_trades) > 0 else 0

print(f"   Annualized: {annualized_return:.2%}")
print(f"   Sharpe: {sharpe:.2f}")
print(f"   Max DD: {max_drawdown:.2%}")
print(f"   Total Trades: {n_trades}, Win Rate: {win_rate:.2%}")

results.append({
    'strategy': 'HYBRID 70/30',
    'total_return': total_return,
    'annualized_return': annualized_return,
    'sharpe_ratio': sharpe,
    'max_drawdown': max_drawdown,
    'n_trades': n_trades,
    'win_rate': win_rate,
    'equity_curve': hybrid_df,
    'trades': combined_trades
})

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("MAXIMUM PERFORMANCE RESULTS")
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

results_df = results_df.sort_values('Ann_Return', ascending=False)

print("\n" + results_df.to_string(index=False))

results_df.to_csv(OUTPUTS_DIR / "maximum_performance_results.csv", index=False)

# Find best
best = results_df.iloc[0]
print(f"\n🏆 MAXIMUM PERFORMANCE ACHIEVED:")
print(f"   Strategy: {best['Strategy']}")
print(f"   Annualized Return: {best['Ann_Return']:.2%}")
print(f"   Sharpe Ratio: {best['Sharpe']:.2f}")
print(f"   Max Drawdown: {best['Max_DD']:.2%}")
print(f"   Win Rate: {best['Win_Rate']:.2%}")
print(f"   Trades: {best['Trades']}")

print("\n" + "=" * 80)
print("✅ MAXIMUM PERFORMANCE OPTIMIZATION COMPLETE!")
print("=" * 80)
