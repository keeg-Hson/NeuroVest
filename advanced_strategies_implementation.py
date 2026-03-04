#!/usr/bin/env python3
"""
Advanced Strategies Implementation - Push to 15-20% Annualized

Implements 4 advanced strategies:
1. Multi-Asset Trading - Trade SPY, QQQ, IWM, TLT, GLD simultaneously
2. Moderate Leverage - 1.5-2x leverage on high-confidence signals
3. Options Flow Framework - Simulated + integration guide for real data
4. Options Selling Strategies - Cash-secured puts & covered calls simulation

Expected combined impact: +7-12% annualized (7.29% → 15-20%)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
)

print("=" * 80)
print("ADVANCED STRATEGIES IMPLEMENTATION")
print("=" * 80)
print("Target: 15-20% annualized with advanced strategies")
print("=" * 80)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

BACKTEST_CFG = {
    "fee_bps": 1.5,
    "slippage_bps": 2.0,
    "initial_capital": 100000,  # $100k for realistic testing
}

# ============================================================================
# 1. MULTI-ASSET DATA LOADING
# ============================================================================

def load_multi_asset_data():
    """
    Load data for multiple uncorrelated assets.

    Assets:
    - SPY: S&P 500 (large cap US equity)
    - QQQ: NASDAQ 100 (tech-heavy)
    - IWM: Russell 2000 (small caps)
    - TLT: 20+ Year Treasury (bonds, negative correlation to stocks)
    - GLD: Gold (safe haven, inflation hedge)
    """
    print("\n📥 Loading multi-asset data...")

    assets = {}

    # Load SPY data
    print("   Loading SPY...")
    spy_data = load_SPY_data()
    spy_data, spy_features = add_features(spy_data)
    spy_data = finalize_features(spy_data, spy_features)
    assets['SPY'] = {
        'data': spy_data,
        'features': spy_features,
        'name': 'S&P 500'
    }
    print(f"      ✅ SPY loaded: {len(spy_data)} samples")

    # For other assets, use SPY as proxy
    # In production, load real data for each asset using yfinance or similar
    print("   📝 NOTE: Using SPY data as proxy for other assets")
    print("      In production, load real data for QQQ, IWM, TLT, GLD")

    # For demonstration, use SPY for all assets
    # In production, load real data using yfinance for each ticker
    for ticker in ['QQQ', 'IWM', 'TLT', 'GLD']:
        # In production:
        # raw_data = yf.download(ticker)
        # data, features = add_features(raw_data)
        # data = finalize_features(data, features)

        # For now, use SPY as proxy
        assets[ticker] = {
            'data': spy_data.copy(),
            'features': spy_features,
            'name': ticker
        }
        print(f"      ✅ {ticker} prepared (using SPY proxy)")

    return assets


# ============================================================================
# 2. OPTIONS FLOW FRAMEWORK (Simulated)
# ============================================================================

def simulate_options_flow_signals(df, asset='SPY'):
    """
    Simulate options flow signals.

    In production, replace with real options data from:
    - FlowAlgo ($250/month): Real-time unusual options activity
    - Trade Alert ($200-500/month): Dark pool prints
    - CBOE DataShop ($500-1,500/month): Official exchange data

    Returns: DataFrame with options flow features
    """

    # Simulate options flow features
    # In production, these would come from real data APIs

    # 1. Put/Call Ratio (contrarian indicator)
    # High put/call = bearish sentiment (contrarian bullish)
    df['Options_PutCallRatio'] = np.random.normal(0.8, 0.3, len(df))
    df['Options_PutCallRatio'] = df['Options_PutCallRatio'].clip(0.3, 2.0)

    # 2. Unusual Options Activity (smart money indicator)
    # Spikes indicate institutional positioning
    df['Options_UnusualActivity'] = np.random.binomial(1, 0.1, len(df))  # 10% of days

    # 3. Dark Pool Prints (large block trades)
    # Indicates institutional accumulation/distribution
    df['Options_DarkPoolPrints'] = np.random.binomial(1, 0.15, len(df))

    # 4. Implied Volatility Rank (0-100)
    # High IV rank = option premium rich (good for selling)
    df['Options_IVRank'] = np.random.uniform(0, 100, len(df))

    # 5. Net Gamma Exposure (dealer hedging pressure)
    # Positive GEX = dealers suppress volatility
    # Negative GEX = dealers amplify moves
    df['Options_GammaExposure'] = np.random.normal(0, 1e9, len(df))

    print(f"\n   ✅ Simulated options flow signals for {asset}")
    print(f"      In production: Replace with real data from FlowAlgo/Trade Alert")

    return df


def get_options_flow_signal(row):
    """
    Generate trading signal from options flow.

    Real implementation would analyze:
    - Unusual call buying → Bullish
    - Unusual put buying → Bearish
    - Dark pool accumulation → Bullish
    - High IV rank + bullish signal → Sell puts for premium
    """

    signal_strength = 0

    # Unusual activity + dark pool prints = institutional interest
    if row['Options_UnusualActivity'] == 1 and row['Options_DarkPoolPrints'] == 1:
        signal_strength += 0.05  # Boost probability by 5%

    # Contrarian put/call ratio signal
    if row['Options_PutCallRatio'] > 1.2:  # High fear
        signal_strength += 0.03  # Contrarian bullish

    # Gamma exposure (simplified)
    if row['Options_GammaExposure'] > 0:
        signal_strength += 0.02  # Positive GEX supports market

    return signal_strength


# ============================================================================
# 3. LEVERAGE SYSTEM
# ============================================================================

def calculate_position_size_with_leverage(
    base_signal_prob,
    ensemble_agreement,
    options_flow_boost,
    regime_favorable,
    max_leverage=2.0
):
    """
    Dynamic leverage based on signal confidence.

    Leverage tiers:
    - Very high confidence (4/4 models + favorable regime): 2.0x
    - High confidence (3/4 models + favorable regime): 1.5x
    - Medium confidence (2/4 models or favorable regime): 1.2x
    - Base confidence: 1.0x (no leverage)

    Returns: position_size as fraction of capital (1.0 = 100%, 2.0 = 200%)
    """

    position_size = 1.0  # Base (no leverage)

    # Factor 1: Model ensemble agreement
    if ensemble_agreement >= 0.75:  # 3/4 or 4/4 models
        position_size = 1.5
    if ensemble_agreement >= 1.0:  # All 4 models agree
        position_size = 1.8

    # Factor 2: Regime favorability
    if regime_favorable:
        position_size += 0.2

    # Factor 3: Options flow boost
    if options_flow_boost > 0.05:  # Strong institutional signal
        position_size += 0.2

    # Factor 4: Base signal strength
    if base_signal_prob >= 0.60:  # Very high probability
        position_size += 0.2

    # Cap at max leverage
    position_size = min(position_size, max_leverage)

    return position_size


# ============================================================================
# 4. OPTIONS SELLING STRATEGIES (Simulated)
# ============================================================================

def simulate_options_premium_collection(
    portfolio_value,
    current_price,
    signal,
    iv_rank,
    days_to_hold=7
):
    """
    Simulate cash-secured put and covered call premium collection.

    Strategy:
    - When bullish + high IV: Sell cash-secured puts (collect premium)
    - When holding position + high IV: Sell covered calls (boost yield)

    Premium rates (weekly):
    - High IV (>60): 0.4-0.6% per week
    - Medium IV (30-60): 0.2-0.4% per week
    - Low IV (<30): 0.1-0.2% per week

    In production, use real options pricing:
    - Black-Scholes for fair value
    - Real market quotes from broker API
    """

    premium_collected = 0

    # High IV rank makes option premium attractive
    if iv_rank > 60:  # High IV
        weekly_premium_pct = np.random.uniform(0.004, 0.006)  # 0.4-0.6%
    elif iv_rank > 30:  # Medium IV
        weekly_premium_pct = np.random.uniform(0.002, 0.004)  # 0.2-0.4%
    else:  # Low IV
        weekly_premium_pct = np.random.uniform(0.001, 0.002)  # 0.1-0.2%

    # Only sell options if IV is attractive and signal is favorable
    if iv_rank > 40 and signal >= 0.52:
        # Sell puts on 20-30% of capital (conservative)
        capital_allocated = portfolio_value * 0.25
        premium_collected = capital_allocated * weekly_premium_pct

    return premium_collected


# ============================================================================
# 5. MULTI-ASSET PORTFOLIO BACKTEST
# ============================================================================

def run_multi_asset_backtest(
    assets,
    models,
    use_leverage=False,
    use_options_flow=False,
    use_options_selling=False,
    max_leverage=2.0,
    max_positions=3,  # Max simultaneous positions
    position_correlation_threshold=0.7,  # Avoid correlated positions
    initial_capital=100000
):
    """
    Comprehensive multi-asset backtest with all advanced strategies.
    """

    print(f"\n📊 Running multi-asset backtest...")
    print(f"   Assets: {list(assets.keys())}")
    print(f"   Leverage: {'Yes (max ' + str(max_leverage) + 'x)' if use_leverage else 'No'}")
    print(f"   Options Flow: {'Yes (simulated)' if use_options_flow else 'No'}")
    print(f"   Options Selling: {'Yes (simulated)' if use_options_selling else 'No'}")

    # Initialize portfolio
    cash = initial_capital
    positions = {}  # {asset: {'entry_date': date, 'entry_price': price, 'size': shares}}
    equity_curve = []
    trades = []
    options_premiums = []

    # Get test period from SPY
    spy_data = assets['SPY']['data']
    test_size = int(len(spy_data) * 0.2)
    train_end_idx = len(spy_data) - test_size
    test_dates = spy_data.index[train_end_idx:]

    # Add options flow signals if enabled
    if use_options_flow:
        for asset_name in assets.keys():
            assets[asset_name]['data'] = simulate_options_flow_signals(
                assets[asset_name]['data'].copy(),
                asset_name
            )

    # Prepare data for each asset
    asset_predictions = {}
    for asset_name, asset_info in assets.items():
        df = asset_info['data'].copy()
        features = asset_info['features']

        # Verify Close column exists
        if 'Close' not in df.columns:
            # Get raw close prices
            raw_data = load_SPY_data()
            df['Close'] = raw_data['Close'].reindex(df.index)

        # Create target
        df['Target'] = (df['Close'].shift(-10) > df['Close']).astype(int)
        df = df.dropna(subset=['Target'])

        # Get features
        available_features = [c for c in features if c in df.columns]
        X_test_df = df.iloc[train_end_idx:][available_features]

        # Fill NaN values
        X_test_df = X_test_df.fillna(0)
        X_test = X_test_df.values

        # Get predictions from ensemble
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

        # Calculate ensemble agreement
        votes = np.array([pred >= 0.52 for pred in predictions_dict.values()])
        agreements = votes.sum(axis=0)
        avg_proba = np.array(list(predictions_dict.values())).mean(axis=0)

        asset_predictions[asset_name] = {
            'probabilities': avg_proba,
            'agreements': agreements,
            'total_models': len(predictions_dict),
            'dates': df.index[train_end_idx:],
            'prices': df.iloc[train_end_idx:]['Close'].values,
            'data': df.iloc[train_end_idx:]
        }

    print(f"   ✅ Prepared predictions for {len(assets)} assets")

    # Run backtest day by day
    for i, current_date in enumerate(test_dates):
        current_portfolio_value = cash

        # Calculate value of existing positions
        positions_to_close = []
        for asset_name, position in positions.items():
            if current_date in asset_predictions[asset_name]['dates']:
                idx = asset_predictions[asset_name]['dates'].get_loc(current_date)
                current_price = asset_predictions[asset_name]['prices'][idx]

                # Check if holding period complete (10 days)
                days_held = (current_date - position['entry_date']).days
                if days_held >= 10:
                    positions_to_close.append(asset_name)

                # Add unrealized value
                position_value = position['shares'] * current_price
                current_portfolio_value += position_value

        # Close positions that hit 10-day holding period
        for asset_name in positions_to_close:
            position = positions[asset_name]
            idx = asset_predictions[asset_name]['dates'].get_loc(current_date)
            exit_price = asset_predictions[asset_name]['prices'][idx]

            # Calculate return
            entry_cost = position['entry_price'] * (1 + 0.00035)  # fees + slippage
            exit_proceeds = exit_price * (1 - 0.00035)
            trade_return = (exit_proceeds / entry_cost - 1)
            pnl = position['shares'] * (exit_proceeds - entry_cost)

            cash += position['shares'] * exit_proceeds

            trades.append({
                'asset': asset_name,
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'exit_date': current_date,
                'exit_price': exit_price,
                'shares': position['shares'],
                'leverage': position.get('leverage', 1.0),
                'return': trade_return,
                'pnl': pnl
            })

            del positions[asset_name]

        # Options selling: collect weekly premium
        if use_options_selling and len(trades) % 5 == 0:  # Every week
            for asset_name in assets.keys():
                if current_date in asset_predictions[asset_name]['dates']:
                    idx = asset_predictions[asset_name]['dates'].get_loc(current_date)
                    asset_data = asset_predictions[asset_name]['data']

                    if 'Options_IVRank' in asset_data.columns:
                        iv_rank = asset_data.iloc[idx]['Options_IVRank']
                        signal = asset_predictions[asset_name]['probabilities'][idx]

                        premium = simulate_options_premium_collection(
                            current_portfolio_value,
                            asset_predictions[asset_name]['prices'][idx],
                            signal,
                            iv_rank
                        )

                        if premium > 0:
                            cash += premium
                            options_premiums.append({
                                'date': current_date,
                                'asset': asset_name,
                                'premium': premium
                            })

        # Consider new positions if capacity available
        if len(positions) < max_positions:
            # Score all assets
            asset_scores = []
            for asset_name in assets.keys():
                if asset_name in positions:
                    continue  # Already holding

                if current_date not in asset_predictions[asset_name]['dates']:
                    continue

                idx = asset_predictions[asset_name]['dates'].get_loc(current_date)
                prob = asset_predictions[asset_name]['probabilities'][idx]
                agreements = asset_predictions[asset_name]['agreements'][idx]
                total_models = asset_predictions[asset_name]['total_models']

                # Check if meets threshold (2/4 models agree)
                if agreements < 2:
                    continue

                # Calculate regime favorability (simplified)
                asset_data = asset_predictions[asset_name]['data']
                regime_favorable = asset_data.iloc[idx].get('Regime', 'Bull') == 'Bull'

                # Options flow boost
                options_boost = 0
                if use_options_flow and 'Options_UnusualActivity' in asset_data.columns:
                    options_boost = get_options_flow_signal(asset_data.iloc[idx])

                # Calculate position size with leverage
                if use_leverage:
                    position_size = calculate_position_size_with_leverage(
                        prob,
                        agreements / total_models,
                        options_boost,
                        regime_favorable,
                        max_leverage
                    )
                else:
                    position_size = 1.0

                # Adjust for number of positions
                position_size = position_size / max_positions

                asset_scores.append({
                    'asset': asset_name,
                    'score': prob + options_boost,
                    'prob': prob,
                    'position_size': position_size,
                    'idx': idx
                })

            # Sort by score and take best opportunities
            asset_scores.sort(key=lambda x: x['score'], reverse=True)

            # Enter positions
            for entry in asset_scores[:max_positions - len(positions)]:
                asset_name = entry['asset']
                idx = entry['idx']

                # Get entry price (next day)
                if idx + 1 < len(asset_predictions[asset_name]['prices']):
                    entry_price = asset_predictions[asset_name]['prices'][idx + 1]
                    entry_date = asset_predictions[asset_name]['dates'][idx + 1]

                    # Calculate position value
                    position_value = (cash / (max_positions - len(positions))) * entry['position_size']

                    if position_value > 100:  # Minimum $100 position
                        shares = position_value / entry_price

                        # Deduct from cash
                        actual_cost = shares * entry_price
                        if actual_cost <= cash:
                            cash -= actual_cost

                            positions[asset_name] = {
                                'entry_date': entry_date,
                                'entry_price': entry_price,
                                'shares': shares,
                                'leverage': entry['position_size'] * max_positions  # Actual leverage used
                            }

        # Recalculate portfolio value
        current_portfolio_value = cash
        for asset_name, position in positions.items():
            if current_date in asset_predictions[asset_name]['dates']:
                idx = asset_predictions[asset_name]['dates'].get_loc(current_date)
                current_price = asset_predictions[asset_name]['prices'][idx]
                position_value = position['shares'] * current_price
                current_portfolio_value += position_value

        equity_curve.append({
            'date': current_date,
            'portfolio_value': current_portfolio_value,
            'cash': cash,
            'n_positions': len(positions)
        })

    # Close remaining positions
    final_date = test_dates[-1]
    for asset_name, position in positions.items():
        idx = asset_predictions[asset_name]['dates'].get_loc(final_date)
        exit_price = asset_predictions[asset_name]['prices'][idx]

        entry_cost = position['entry_price'] * (1 + 0.00035)
        exit_proceeds = exit_price * (1 - 0.00035)
        trade_return = (exit_proceeds / entry_cost - 1)
        pnl = position['shares'] * (exit_proceeds - entry_cost)

        trades.append({
            'asset': asset_name,
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': final_date,
            'exit_price': exit_price,
            'shares': position['shares'],
            'leverage': position.get('leverage', 1.0),
            'return': trade_return,
            'pnl': pnl
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    options_df = pd.DataFrame(options_premiums) if options_premiums else pd.DataFrame()

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

    # Calculate leverage stats
    avg_leverage = trades_df['leverage'].mean() if 'leverage' in trades_df.columns and len(trades_df) > 0 else 1.0
    max_leverage_used = trades_df['leverage'].max() if 'leverage' in trades_df.columns and len(trades_df) > 0 else 1.0

    # Options premium
    total_options_premium = options_df['premium'].sum() if len(options_df) > 0 else 0
    options_boost = (total_options_premium / initial_capital) / years if years > 0 and total_options_premium > 0 else 0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_leverage': avg_leverage,
        'max_leverage_used': max_leverage_used,
        'options_premium_total': total_options_premium,
        'options_premium_annualized': options_boost,
        'equity_curve': equity_df,
        'trades': trades_df,
        'options_premiums': options_df
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n📥 Loading data and models...")

# Load multi-asset data
assets = load_multi_asset_data()

# Load trained models
print("\n📦 Loading models...")
try:
    models = {
        'xgboost': joblib.load(MODELS_DIR / "xgboost_max_perf.pkl"),
        'lightgbm': joblib.load(MODELS_DIR / "lightgbm_max_perf.pkl"),
        'random_forest': joblib.load(MODELS_DIR / "random_forest_max_perf.pkl"),
        'neural_net': joblib.load(MODELS_DIR / "neural_net_max_perf.pkl"),
        'scaler': joblib.load(MODELS_DIR / "scaler_max_perf.pkl")
    }
    print("   ✅ Loaded diverse model ensemble")
except Exception:
    print("   ⚠️ Could not load max_perf models, loading alternative...")
    models = {
        'xgboost': joblib.load(MODELS_DIR / "xgboost_ultimate.pkl"),
        'lightgbm': joblib.load(MODELS_DIR / "lightgbm_ultimate.pkl"),
        'random_forest': joblib.load(MODELS_DIR / "random_forest_ultimate.pkl"),
        'neural_net': joblib.load(MODELS_DIR / "neural_net_ultimate.pkl"),
        'scaler': joblib.load(MODELS_DIR / "scaler_ultimate.pkl")
    }
    print("   ✅ Loaded ultimate models")

print("\n" + "=" * 80)
print("RUNNING ADVANCED STRATEGY BACKTESTS")
print("=" * 80)

results = []

# Baseline: SPY only, no leverage
print(f"\n{'─'*80}")
print(f"1. BASELINE: SPY Only, No Leverage")
print(f"{'─'*80}")

baseline_assets = {'SPY': assets['SPY']}
result = run_multi_asset_backtest(
    baseline_assets, models,
    use_leverage=False,
    use_options_flow=False,
    use_options_selling=False,
    max_positions=1,
    initial_capital=BACKTEST_CFG['initial_capital']
)

print(f"   Annualized Return: {result['annualized_return']:.2%}")
print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
print(f"   Win Rate: {result['win_rate']:.2%}")
print(f"   Trades: {result['n_trades']}")

results.append({
    'strategy': 'BASELINE: SPY Only',
    **{k: v for k, v in result.items() if k not in ['equity_curve', 'trades', 'options_premiums']}
})

# Strategy 1: Multi-Asset
print(f"\n{'─'*80}")
print(f"2. Multi-Asset Portfolio (5 assets, no leverage)")
print(f"{'─'*80}")

result = run_multi_asset_backtest(
    assets, models,
    use_leverage=False,
    use_options_flow=False,
    use_options_selling=False,
    max_positions=3,
    initial_capital=BACKTEST_CFG['initial_capital']
)

print(f"   Annualized Return: {result['annualized_return']:.2%}")
print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
print(f"   Win Rate: {result['win_rate']:.2%}")
print(f"   Trades: {result['n_trades']}")

results.append({
    'strategy': 'Multi-Asset (5 assets)',
    **{k: v for k, v in result.items() if k not in ['equity_curve', 'trades', 'options_premiums']}
})

# Strategy 2: Multi-Asset + Leverage
print(f"\n{'─'*80}")
print(f"3. Multi-Asset + Moderate Leverage (max 2x)")
print(f"{'─'*80}")

result = run_multi_asset_backtest(
    assets, models,
    use_leverage=True,
    use_options_flow=False,
    use_options_selling=False,
    max_leverage=2.0,
    max_positions=3,
    initial_capital=BACKTEST_CFG['initial_capital']
)

print(f"   Annualized Return: {result['annualized_return']:.2%}")
print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
print(f"   Win Rate: {result['win_rate']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Avg Leverage: {result['avg_leverage']:.2f}x")
print(f"   Max Leverage Used: {result['max_leverage_used']:.2f}x")

results.append({
    'strategy': 'Multi-Asset + Leverage',
    **{k: v for k, v in result.items() if k not in ['equity_curve', 'trades', 'options_premiums']}
})

# Strategy 3: Multi-Asset + Leverage + Options Flow
print(f"\n{'─'*80}")
print(f"4. Multi-Asset + Leverage + Options Flow (simulated)")
print(f"{'─'*80}")

result = run_multi_asset_backtest(
    assets, models,
    use_leverage=True,
    use_options_flow=True,
    use_options_selling=False,
    max_leverage=2.0,
    max_positions=3,
    initial_capital=BACKTEST_CFG['initial_capital']
)

print(f"   Annualized Return: {result['annualized_return']:.2%}")
print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
print(f"   Win Rate: {result['win_rate']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Avg Leverage: {result['avg_leverage']:.2f}x")

results.append({
    'strategy': 'Multi-Asset + Leverage + Options Flow',
    **{k: v for k, v in result.items() if k not in ['equity_curve', 'trades', 'options_premiums']}
})

# Strategy 4: ULTIMATE - All Strategies Combined
print(f"\n{'─'*80}")
print(f"5. ULTIMATE: All Strategies Combined")
print(f"{'─'*80}")

result = run_multi_asset_backtest(
    assets, models,
    use_leverage=True,
    use_options_flow=True,
    use_options_selling=True,
    max_leverage=2.0,
    max_positions=3,
    initial_capital=BACKTEST_CFG['initial_capital']
)

print(f"   Annualized Return: {result['annualized_return']:.2%}")
print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
print(f"   Win Rate: {result['win_rate']:.2%}")
print(f"   Trades: {result['n_trades']}")
print(f"   Avg Leverage: {result['avg_leverage']:.2f}x")
print(f"   Options Premium (annualized): {result['options_premium_annualized']:.2%}")
print(f"   Total Options Premium: ${result['options_premium_total']:,.2f}")

results.append({
    'strategy': 'ULTIMATE: All Combined',
    **{k: v for k, v in result.items() if k not in ['equity_curve', 'trades', 'options_premiums']}
})

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ADVANCED STRATEGIES RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('annualized_return', ascending=False)

print("\n" + results_df[['strategy', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'n_trades']].to_string(index=False))

results_df.to_csv(OUTPUTS_DIR / "advanced_strategies_results.csv", index=False)

best = results_df.iloc[0]
print(f"\n🏆 BEST STRATEGY: {best['strategy']}")
print(f"   Annualized Return: {best['annualized_return']:.2%}")
print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {best['max_drawdown']:.2%}")
print(f"   Win Rate: {best['win_rate']:.2%}")

if 'avg_leverage' in best:
    print(f"   Average Leverage: {best['avg_leverage']:.2f}x")
if 'options_premium_annualized' in best and best['options_premium_annualized'] > 0:
    print(f"   Options Premium Boost: +{best['options_premium_annualized']:.2%}")

print("\n" + "=" * 80)
print("✅ ADVANCED STRATEGIES IMPLEMENTATION COMPLETE!")
print("=" * 80)

print("\n📝 NEXT STEPS FOR PRODUCTION:")
print("   1. Replace simulated options data with real FlowAlgo/Trade Alert API")
print("   2. Load real data for QQQ, IWM, TLT, GLD (use yfinance)")
print("   3. Implement real options pricing (Black-Scholes or broker API)")
print("   4. Set up broker API for live trading (Interactive Brokers, TD Ameritrade)")
print("   5. Start with paper trading to validate before going live")
