#!/usr/bin/env python3
"""
Ultimate Trading Optimizations - Aggressive Path to 14-18% Annualized

Implements 5 high-impact optimizations:
1. Market Regime Filtering (+2-4%) - Only trade in favorable conditions
2. Model Ensemble Diversification (+1-3%) - Mix XGBoost, LightGBM, RF, Neural Net
3. Dynamic Exit Strategy (+1-2%) - Adaptive exits vs fixed 10-day
4. Feature Engineering v2 (+0.5-2%) - Market microstructure features
5. Regime-Specific Models (+1.5-3%) - Separate models per regime

Expected combined impact: +7-14% annualized (7.60% → 14-18%)
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

# Try LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not available - will use XGBoost instead")

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
)

print("=" * 80)
print("ULTIMATE TRADING OPTIMIZATIONS - AGGRESSIVE PATH")
print("=" * 80)
print("Target: Push from 7.60% to 14-18% annualized")
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
# 1. ADVANCED FEATURE ENGINEERING
# ============================================================================

def add_advanced_features(df):
    """
    Add market microstructure and regime features.

    Features added:
    - Regime change detection
    - Trend quality metrics
    - Volume-price divergence
    - Momentum acceleration
    - Support/resistance proximity
    - Volatility regime
    """
    print("\n📊 Adding advanced features...")

    engineered = []

    # 1. Regime Change Detection
    if 'Regime' in df.columns:
        df['Regime_Change'] = (df['Regime'] != df['Regime'].shift(1)).astype(int)
        df['Days_In_Regime'] = df.groupby((df['Regime'] != df['Regime'].shift(1)).cumsum()).cumcount() + 1
        engineered.extend(['Regime_Change', 'Days_In_Regime'])

    # 2. Trend Quality (ADX weighted by direction)
    if 'ADX' in df.columns and 'Plus_DI' in df.columns and 'Minus_DI' in df.columns:
        df['Trend_Quality'] = df['ADX'] * np.sign(df['Plus_DI'] - df['Minus_DI'])
        engineered.append('Trend_Quality')

    # 3. DI Ratio (directional strength)
    if 'Plus_DI' in df.columns and 'Minus_DI' in df.columns:
        df['DI_Ratio'] = df['Plus_DI'] / (df['Minus_DI'] + 1e-6)
        engineered.append('DI_Ratio')

    # 4. Volume-Price Divergence
    if 'Volume_pct' in df.columns and 'Close' in df.columns:
        vol_ma = df['Volume_pct'].rolling(20).mean()
        price_ret = df['Close'].pct_change().rolling(20).mean()
        df['Vol_Price_Div'] = vol_ma - price_ret
        engineered.append('Vol_Price_Div')

    # 5. Momentum Acceleration
    if 'RSI' in df.columns:
        df['RSI_Accel'] = df['RSI'].diff()
        df['RSI_Velocity'] = df['RSI'].diff().rolling(5).mean()
        engineered.extend(['RSI_Accel', 'RSI_Velocity'])

    if 'MACD' in df.columns:
        df['MACD_Accel'] = df['MACD'].diff()
        engineered.append('MACD_Accel')

    # 6. Support/Resistance Proximity
    if 'Close' in df.columns:
        df['Dist_To_52W_High'] = (df['Close'] / df['Close'].rolling(252).max()) - 1
        df['Dist_To_52W_Low'] = (df['Close'] / df['Close'].rolling(252).min()) - 1
        df['Dist_To_20D_High'] = (df['Close'] / df['Close'].rolling(20).max()) - 1
        df['Dist_To_20D_Low'] = (df['Close'] / df['Close'].rolling(20).min()) - 1
        engineered.extend(['Dist_To_52W_High', 'Dist_To_52W_Low',
                          'Dist_To_20D_High', 'Dist_To_20D_Low'])

    # 7. Volatility Regime
    if 'ATR' in df.columns:
        df['Vol_Regime'] = (df['ATR'] > df['ATR'].rolling(50).mean()).astype(int)
        df['Vol_Percentile'] = df['ATR'].rolling(252).rank(pct=True)
        engineered.extend(['Vol_Regime', 'Vol_Percentile'])

    # 8. Price Position in Bollinger Bands (if available)
    if 'BB_PctB' in df.columns:
        df['BB_Width'] = df['BB_PctB'].rolling(20).std()
        engineered.append('BB_Width')

    # 9. Moving Average Crossover Strength
    if 'MA_20' in df.columns and 'MA_50' in df.columns and 'MA_200' in df.columns:
        df['MA_Cross_20_50'] = (df['MA_20'] - df['MA_50']) / df['MA_50']
        df['MA_Cross_50_200'] = (df['MA_50'] - df['MA_200']) / df['MA_200']
        engineered.extend(['MA_Cross_20_50', 'MA_Cross_50_200'])

    # 10. Stochastic Momentum
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        df['Stoch_Cross'] = df['Stoch_K'] - df['Stoch_D']
        df['Stoch_Momentum'] = df['Stoch_K'].diff()
        engineered.extend(['Stoch_Cross', 'Stoch_Momentum'])

    # Fill NaN values
    df[engineered] = df[engineered].fillna(0)

    print(f"   ✅ Added {len(engineered)} advanced features")
    print(f"   Features: {', '.join(engineered[:5])}...")

    return df, engineered


# ============================================================================
# 2. MARKET REGIME FILTERING
# ============================================================================

def get_regime_filter(df, idx):
    """
    Determine if current market regime is favorable for trading.

    Filters:
    - Bull market: Price > 200-day MA
    - Moderate volatility: ATR < 2x average
    - Strong trend: ADX > 25
    - Positive regime score

    Returns: (should_trade: bool, filter_reasons: list)
    """
    row = df.iloc[idx]
    reasons = []

    # Check for sufficient history for 200-day MA
    if idx < 200:
        return True, ["insufficient_history"]

    # 1. Bull Market Filter (price > 200-day MA)
    if 'MA_200' in df.columns:
        if row['Close'] < row['MA_200']:
            return False, ["bear_market"]

    # 2. Volatility Filter (ATR < 2x average)
    if 'ATR' in df.columns:
        avg_atr = df['ATR'].iloc[max(0, idx-50):idx].mean()
        if row['ATR'] > 2.0 * avg_atr:
            return False, ["high_volatility"]

    # 3. Trend Strength Filter (ADX > 25)
    if 'ADX' in df.columns:
        if row['ADX'] < 25:
            return False, ["weak_trend"]

    # 4. Regime Filter (avoid bearish regimes)
    if 'Regime' in df.columns:
        if row['Regime'] == 'Bear':
            return False, ["bear_regime"]

    return True, ["all_passed"]


# ============================================================================
# 3. DYNAMIC EXIT STRATEGY
# ============================================================================

def should_exit_dynamic(entry_date, current_date, entry_price, current_price,
                       entry_idx, current_idx, df, min_hold=5, max_hold=15):
    """
    Dynamic exit conditions based on market state.

    Exit triggers:
    - Take profit: 5%+ gain AND RSI > 75 (overbought)
    - Stop loss: 2.5%+ loss AND ADX < 20 (trend broken)
    - Regime change: Entry regime != current regime (after min hold)
    - Max hold: 15 days maximum
    - Default: 10 days

    Returns: (should_exit: bool, exit_reason: str)
    """
    days_held = (current_date - entry_date).days
    current_return = (current_price / entry_price) - 1

    # Minimum holding period (avoid whipsaw)
    if days_held < min_hold:
        return False, None

    current_row = df.iloc[current_idx]

    # Take Profit: Strong gain + overbought
    if current_return >= 0.05:  # 5% gain
        if 'RSI' in df.columns and current_row['RSI'] > 75:
            return True, 'take_profit_overbought'
        # Take profit on very large gains regardless
        if current_return >= 0.08:  # 8% gain
            return True, 'take_profit_large'

    # Stop Loss: Loss + trend broken
    if current_return <= -0.025:  # 2.5% loss
        if 'ADX' in df.columns and current_row['ADX'] < 20:
            return True, 'stop_loss_trend_broken'

    # Hard stop loss at 4%
    if current_return <= -0.04:
        return True, 'stop_loss_hard'

    # Regime Change Exit
    if 'Regime' in df.columns and entry_idx < len(df):
        entry_regime = df.iloc[entry_idx]['Regime']
        current_regime = current_row['Regime']
        if entry_regime != current_regime and days_held >= 7:
            return True, 'regime_change'

    # Momentum Exhaustion (RSI reversal)
    if 'RSI' in df.columns and current_return > 0.02:  # In profit
        if current_row['RSI'] < 30:  # Became oversold
            return True, 'momentum_exhausted'

    # Maximum holding period
    if days_held >= max_hold:
        return True, 'max_hold'

    # Default exit at 10 days
    if days_held >= 10:
        return True, 'time_exit_10d'

    return False, None


# ============================================================================
# 4. MODEL TRAINING - DIVERSE ENSEMBLE
# ============================================================================

def train_diverse_models(X_train, y_train, X_val, y_val, model_name_prefix=""):
    """
    Train diverse model ensemble:
    - XGBoost (gradient boosting)
    - LightGBM (fast gradient boosting)
    - Random Forest (bagging)
    - Neural Network (deep learning)

    Returns: dict of trained models
    """
    print(f"\n🤖 Training diverse model ensemble: {model_name_prefix}")

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
    # Scale features for neural network
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
    models['scaler'] = scaler  # Save scaler for neural net
    print(f"      ✅ Neural Network accuracy: {nn_acc:.4f}")

    # Calculate ensemble accuracy (majority vote)
    ensemble_preds = []
    for name, model in models.items():
        if name == 'scaler':
            continue
        if name == 'neural_net':
            preds = model.predict(X_val_scaled)
        else:
            preds = model.predict(X_val)
        ensemble_preds.append(preds)

    ensemble_preds = np.array(ensemble_preds)
    majority_vote = (ensemble_preds.sum(axis=0) >= len(models)/2).astype(int)
    ensemble_acc = (majority_vote == y_val).mean()
    print(f"   🎯 Ensemble (majority vote) accuracy: {ensemble_acc:.4f}")

    return models


# ============================================================================
# 5. REGIME-SPECIFIC MODELS
# ============================================================================

def train_regime_specific_models(df, feature_cols, horizon=10):
    """
    Train separate models for each market regime.

    Regimes:
    - Bull + Low Vol
    - Bull + High Vol
    - Bear
    - Sideways
    """
    print(f"\n🎭 Training regime-specific models (horizon={horizon})...")

    # Create target
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    df = df.dropna(subset=['Target'])

    # Identify regimes
    if 'ATR' not in df.columns:
        print("   ⚠️ ATR not available, using single regime")
        return None

    atr_median = df['ATR'].median()

    regime_models = {}

    # Define regime filters
    regimes = {
        'bull_low_vol': (df['Regime'] == 'Bull') & (df['ATR'] <= atr_median),
        'bull_high_vol': (df['Regime'] == 'Bull') & (df['ATR'] > atr_median),
        'bear': df['Regime'] == 'Bear',
        'sideways': df['Regime'] == 'Sideways'
    }

    for regime_name, regime_mask in regimes.items():
        regime_data = df[regime_mask]

        if len(regime_data) < 500:  # Need minimum samples
            print(f"   ⚠️ {regime_name}: Insufficient data ({len(regime_data)} samples)")
            continue

        print(f"   Training {regime_name} models ({len(regime_data)} samples)...")

        # Prepare data
        X = regime_data[feature_cols].values
        y = regime_data['Target'].values

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Train models for this regime
        models = train_diverse_models(X_train, y_train, X_val, y_val,
                                     model_name_prefix=regime_name)

        regime_models[regime_name] = models

    return regime_models


# ============================================================================
# 6. COMPREHENSIVE BACKTEST ENGINE
# ============================================================================

def run_comprehensive_backtest(
    strategy_name,
    models,
    X_test,
    test_dates,
    prices,
    df,
    use_regime_filter=False,
    use_dynamic_exits=False,
    use_ensemble_voting=False,
    voting_threshold=0.5,
    horizon=10,
    threshold=0.52,
    initial_capital=10000,
    fee_bps=1.5,
    slippage_bps=2.0
):
    """
    Comprehensive backtesting with all optimizations.
    """
    cash = initial_capital
    equity_curve = []
    trades = []

    # Get predictions from models
    if isinstance(models, dict):
        # Ensemble of different model types
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

        # Ensemble voting
        if use_ensemble_voting:
            # Count how many models predict positive
            votes = np.array([pred >= threshold for pred in predictions_dict.values()])
            agreements = votes.sum(axis=0)
            total_models = len(predictions_dict)

            # Require majority (e.g., 3 out of 4 models)
            required_votes = int(total_models * voting_threshold)
            predictions = (agreements >= required_votes).astype(float)
            # Use average probability for position sizing
            avg_proba = np.array(list(predictions_dict.values())).mean(axis=0)
        else:
            # Use average probability
            predictions = np.array(list(predictions_dict.values())).mean(axis=0)
            avg_proba = predictions
    else:
        # Single model
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
            # Check dynamic exit conditions
            if use_dynamic_exits:
                should_exit, exit_reason = should_exit_dynamic(
                    position_entry_date, current_date,
                    position_entry_price, current_price,
                    position_entry_idx, i, df.loc[test_dates]
                )
            else:
                # Fixed horizon exit
                days_held = (current_date - position_entry_date).days
                should_exit = days_held >= horizon
                exit_reason = f'time_exit_{horizon}d'

            if should_exit:
                # Execute exit
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

        # Enter new position
        if not position_open:
            # Check regime filter
            if use_regime_filter:
                should_trade, filter_reasons = get_regime_filter(df.loc[test_dates], i)
                if not should_trade:
                    regime_filter_skips += 1
                    equity_curve.append({
                        'date': current_date,
                        'portfolio_value': cash,
                        'position_open': False
                    })
                    continue

            # Check prediction signal
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

    # Close final position if open
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

    # Exit reason distribution
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict() if 'exit_reason' in trades_df.columns else {}

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
        'exit_reasons': exit_reasons,
        'equity_curve': equity_df,
        'trades': trades_df
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n📥 Loading data...")

# Load SPY data
df = load_SPY_data()
df, feature_cols = add_features(df)
df = finalize_features(df, feature_cols)

# Add advanced features
df, advanced_cols = add_advanced_features(df)
all_features = feature_cols + advanced_cols

# Get prices
_raw = load_SPY_data()
_raw_idxed = _raw["Close"].astype(float)
df.index = pd.to_datetime(df.index, errors="coerce")
_raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
df["Close"] = _raw_idxed.reindex(df.index)
df = df.dropna(subset=["Close"])

prices_df = _raw[["Close"]].copy()
prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")

# Prepare features
available_features = [c for c in all_features if c in df.columns]
df = df[[c for c in (available_features + ["Close"]) if c in df.columns]]
df = df.fillna(0)

print(f"✅ Data loaded: {len(df)} samples")
print(f"   Original features: {len(feature_cols)}")
print(f"   Advanced features: {len(advanced_cols)}")
print(f"   Total features: {len(available_features)}")

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

# Further split train for validation
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=False
)

print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

# Train diverse model ensemble
diverse_models = train_diverse_models(X_train_fit, y_train_fit, X_val, y_val)

# Save models
print("\n💾 Saving models...")
for name, model in diverse_models.items():
    if name != 'scaler':
        joblib.dump(model, MODELS_DIR / f"{name}_ultimate.pkl")
        print(f"   ✅ Saved {name}_ultimate.pkl")
if 'scaler' in diverse_models:
    joblib.dump(diverse_models['scaler'], MODELS_DIR / "scaler_ultimate.pkl")
    print(f"   ✅ Saved scaler_ultimate.pkl")

print("\n" + "=" * 80)
print("RUNNING COMPREHENSIVE BACKTESTS")
print("=" * 80)

results = []

# Strategy 1: Baseline (XGBoost only, no optimizations)
print(f"\n{'─'*80}")
print(f"1. Baseline: XGBoost only (no optimizations)")
print(f"{'─'*80}")

result = run_comprehensive_backtest(
    'Baseline: XGBoost',
    {'xgboost': diverse_models['xgboost']},
    X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False,
    use_dynamic_exits=False,
    use_ensemble_voting=False,
    horizon=10, threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Strategy 2: Model Ensemble (3/4 models must agree)
print(f"\n{'─'*80}")
print(f"2. Model Ensemble Diversification (3/4 models agree)")
print(f"{'─'*80}")

result = run_comprehensive_backtest(
    'Model Ensemble (3/4)',
    diverse_models,
    X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False,
    use_dynamic_exits=False,
    use_ensemble_voting=True,
    voting_threshold=0.75,  # 3 out of 4
    horizon=10, threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")

results.append(result)

# Strategy 3: Market Regime Filtering
print(f"\n{'─'*80}")
print(f"3. Market Regime Filtering (bull + low vol + strong trend)")
print(f"{'─'*80}")

result = run_comprehensive_backtest(
    'Regime Filtering',
    diverse_models,
    X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=True,
    use_dynamic_exits=False,
    use_ensemble_voting=True,
    voting_threshold=0.75,
    horizon=10, threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")
print(f"   Trades skipped by filter: {result['regime_filter_skips']}")

results.append(result)

# Strategy 4: Dynamic Exits
print(f"\n{'─'*80}")
print(f"4. Dynamic Exit Strategy (take profit, stop loss, regime change)")
print(f"{'─'*80}")

result = run_comprehensive_backtest(
    'Dynamic Exits',
    diverse_models,
    X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=False,
    use_dynamic_exits=True,
    use_ensemble_voting=True,
    voting_threshold=0.75,
    horizon=10, threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")
print(f"   Exit reasons: {result['exit_reasons']}")

results.append(result)

# Strategy 5: ULTIMATE - All Optimizations Combined
print(f"\n{'─'*80}")
print(f"5. ULTIMATE: All Optimizations Combined")
print(f"{'─'*80}")

result = run_comprehensive_backtest(
    'ULTIMATE: All Combined',
    diverse_models,
    X_test, test_dates, prices_df['Close'], df,
    use_regime_filter=True,
    use_dynamic_exits=True,
    use_ensemble_voting=True,
    voting_threshold=0.75,
    horizon=10, threshold=0.52,
    **BACKTEST_CFG
)

print(f"   Total Return: {result['total_return']:.2%}")
print(f"   Annualized: {result['annualized_return']:.2%}")
print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
print(f"   Max DD: {result['max_drawdown']:.2%}")
print(f"   Trades: {result['n_trades']}, Win Rate: {result['win_rate']:.2%}")
print(f"   Trades skipped by filter: {result['regime_filter_skips']}")
print(f"   Exit reasons: {result['exit_reasons']}")

results.append(result)

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS COMPARISON")
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

results_df.to_csv(OUTPUTS_DIR / "ultimate_optimizations_results.csv", index=False)
print(f"\n💾 Saved: outputs/ultimate_optimizations_results.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

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
ax.axhline(y=10000, color='black', linestyle='--', alpha=0.5)

# Plot 2: Annualized returns
ax = axes[0, 1]
strategies = results_df['Strategy'].values
ann_returns = results_df['Ann_Return'].values * 100
colors = ['darkgreen' if r > 10 else 'green' if r > 7.6 else 'orange' for r in ann_returns]

bars = ax.barh(strategies, ann_returns, color=colors, alpha=0.7)
ax.set_xlabel('Annualized Return (%)')
ax.set_title('Annualized Return Comparison', fontsize=12, fontweight='bold')
ax.axvline(x=7.6, color='red', linestyle='--', linewidth=1, label='Previous Best (7.6%)', alpha=0.7)
ax.axvline(x=14, color='blue', linestyle='--', linewidth=1, label='Target (14%)', alpha=0.7)
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

# Plot 3: Sharpe ratio
ax = axes[0, 2]
sharpes = results_df['Sharpe'].values

bars = ax.barh(strategies, sharpes, color='steelblue', alpha=0.7)
ax.set_xlabel('Sharpe Ratio')
ax.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
ax.axvline(x=0.63, color='red', linestyle='--', linewidth=1, label='Previous Best', alpha=0.7)
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

# Plot 4: Win rate comparison
ax = axes[1, 0]
win_rates = results_df['Win_Rate'].values * 100

bars = ax.barh(strategies, win_rates, color='mediumseagreen', alpha=0.7)
ax.set_xlabel('Win Rate (%)')
ax.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
ax.axvline(x=63.64, color='red', linestyle='--', linewidth=1, label='Previous Best', alpha=0.7)
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

# Plot 5: Max Drawdown
ax = axes[1, 1]
drawdowns = abs(results_df['Max_DD'].values * 100)

bars = ax.barh(strategies, drawdowns, color='coral', alpha=0.7)
ax.set_xlabel('Max Drawdown (%)')
ax.set_title('Maximum Drawdown (Lower is Better)', fontsize=12, fontweight='bold')
ax.invert_xaxis()
ax.grid(True, alpha=0.3, axis='x')

# Plot 6: Return vs Risk
ax = axes[1, 2]
sharpes = results_df['Sharpe'].values
returns = results_df['Ann_Return'].values * 100

scatter = ax.scatter(sharpes, returns, s=300, alpha=0.7,
                    c=range(len(results_df)), cmap='viridis')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Return vs Risk (Top Right = Best)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

for i, strategy in enumerate(strategies):
    ax.annotate(strategy[:15], (sharpes[i], returns[i]),
               fontsize=7, alpha=0.8, ha='center')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'ultimate_optimizations.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/ultimate_optimizations.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ULTIMATE OPTIMIZATION SUMMARY")
print("=" * 80)

baseline = results_df[results_df['Strategy'] == 'Baseline: XGBoost'].iloc[0]
best = results_df.iloc[0]

print(f"\n📊 BASELINE:")
print(f"   Strategy: {baseline['Strategy']}")
print(f"   Annualized: {baseline['Ann_Return']:.2%}")
print(f"   Sharpe: {baseline['Sharpe']:.2f}")
print(f"   Win Rate: {baseline['Win_Rate']:.2%}")

print(f"\n🏆 BEST STRATEGY: {best['Strategy']}")
print(f"   Total Return: {best['Total_Return']:.2%}")
print(f"   Annualized: {best['Ann_Return']:.2%}")
print(f"   Sharpe: {best['Sharpe']:.2f}")
print(f"   Max DD: {best['Max_DD']:.2%}")
print(f"   Trades: {best['Trades']}, Win Rate: {best['Win_Rate']:.2%}")

print(f"\n📈 IMPROVEMENT vs PREVIOUS BEST (7.60%):")
print(f"   Annualized Return: {(best['Ann_Return'] - 0.076)*100:+.2f}pp")
print(f"   ({best['Ann_Return']:.2%} vs 7.60%)")

if best['Ann_Return'] >= 0.14:
    print(f"\n🎯 TARGET ACHIEVED! Reached {best['Ann_Return']:.2%} (target: 14%+)")
elif best['Ann_Return'] >= 0.10:
    print(f"\n✅ Strong improvement! Reached {best['Ann_Return']:.2%} (target: 14%)")
else:
    print(f"\n⚠️ Results: {best['Ann_Return']:.2%} (target: 14%)")

print("\n" + "=" * 80)
print("✅ ULTIMATE OPTIMIZATION COMPLETE!")
print("=" * 80)
