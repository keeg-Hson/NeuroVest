"""
Multi-Asset Backtest with Real Data
Uses real/realistic data for multiple uncorrelated assets
Tests the advanced strategies with true diversification benefits
"""

import pandas as pd
import numpy as np
import pickle
from real_data_loader import load_multi_asset_real_data, verify_data_quality
from utils import add_features, finalize_features
import warnings
warnings.filterwarnings('ignore')


def load_spy_models():
    """
    Load pre-trained models from previous experiments
    Try multiple model sets in order of preference
    """
    model_sets = [
        ('ultimate', 'models/xgboost_ultimate.pkl'),
        ('max_perf', 'models/xgboost_max_perf.pkl'),
        ('regime', 'models/xgboost_regime.pkl')
    ]

    for model_name, xgb_path in model_sets:
        try:
            base_name = model_name
            model_dir = 'models'

            with open(f'{model_dir}/xgboost_{base_name}.pkl', 'rb') as f:
                xgb_model = pickle.load(f)
            with open(f'{model_dir}/lightgbm_{base_name}.pkl', 'rb') as f:
                lgb_model = pickle.load(f)
            with open(f'{model_dir}/random_forest_{base_name}.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            with open(f'{model_dir}/neural_net_{base_name}.pkl', 'rb') as f:
                nn_model = pickle.load(f)
            with open(f'{model_dir}/scaler_{base_name}.pkl', 'rb') as f:
                scaler = pickle.load(f)

            print(f"✓ Loaded {base_name} models")
            return xgb_model, lgb_model, rf_model, nn_model, scaler

        except Exception as e:
            print(f"⚠️  Could not load {model_name} models: {e}")
            continue

    print("✗ Could not load any models")
    print("⚠️  Will need to train new models")
    return None, None, None, None, None


def prepare_asset_data(asset_df):
    """
    Prepare asset data with features for ML prediction
    """
    # Add features
    df, features = add_features(asset_df.copy())
    df = finalize_features(df, features)

    return df, features


def multi_asset_backtest(
    assets,
    models,
    start_date='2020-09-03',
    initial_capital=100000,
    holding_period=10,
    use_leverage=True,
    max_leverage=2.0
):
    """
    Run backtest across multiple assets with portfolio optimization

    Args:
        assets: Dictionary of {ticker: DataFrame}
        models: Tuple of (xgb, lgb, rf, nn, scaler)
        start_date: Backtest start date
        initial_capital: Starting capital
        holding_period: Days to hold each position
        use_leverage: Whether to use dynamic leverage
        max_leverage: Maximum leverage allowed

    Returns:
        Dictionary with backtest results
    """
    xgb_model, lgb_model, rf_model, nn_model, scaler = models

    # Initialize portfolio
    portfolio_value = initial_capital
    cash = initial_capital
    positions = {}  # {ticker: {'shares': N, 'entry_price': P, 'entry_date': D}}
    trades = []
    daily_values = []

    # Prepare data for all assets
    print("\n📊 Preparing asset data with ML features...")
    prepared_assets = {}
    for ticker, df in assets.items():
        prep_df, features = prepare_asset_data(df)
        prepared_assets[ticker] = {'data': prep_df, 'features': features}
        print(f"✓ {ticker}: {len(prep_df)} days with {len(features)} features")

    # Get common date range (intersection of all assets)
    date_ranges = [df['data'].index for df in prepared_assets.values()]
    common_dates = date_ranges[0]
    for dates in date_ranges[1:]:
        common_dates = common_dates.intersection(dates)

    common_dates = common_dates[common_dates >= start_date].sort_values()
    print(f"\n✓ Common date range: {len(common_dates)} days from {common_dates[0].date()} to {common_dates[-1].date()}")

    # Run backtest
    print("\n🔄 Running multi-asset backtest...")

    for i, current_date in enumerate(common_dates):
        # Update portfolio value based on current prices
        portfolio_value = cash
        for ticker, pos in positions.items():
            current_price = prepared_assets[ticker]['data'].loc[current_date, 'Close']
            portfolio_value += pos['shares'] * current_price

        # Check existing positions for exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            days_held = (current_date - pos['entry_date']).days

            if days_held >= holding_period:
                # Exit position
                exit_price = prepared_assets[ticker]['data'].loc[current_date, 'Close']
                exit_value = pos['shares'] * exit_price
                cash += exit_value

                pnl = exit_value - pos['entry_value']
                pnl_pct = (pnl / pos['entry_value']) * 100

                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held
                })

                del positions[ticker]

        # Generate signals for all assets
        signals = {}

        for ticker, asset in prepared_assets.items():
            # Skip if already in position
            if ticker in positions:
                continue

            # Get data up to current date
            asset_data = asset['data'].loc[:current_date]
            if len(asset_data) < 200:  # Need enough history
                continue

            # Get features for prediction
            try:
                X = asset_data[asset['features']].iloc[-1:].fillna(0).values

                # Get predictions from all models
                xgb_prob = xgb_model.predict_proba(X)[0, 1]
                lgb_prob = lgb_model.predict_proba(X)[0, 1]
                rf_prob = rf_model.predict_proba(X)[0, 1]

                X_scaled = scaler.transform(X)
                nn_prob = nn_model.predict_proba(X_scaled)[0, 1]

                # Ensemble voting (2/4 threshold)
                predictions = [xgb_prob, lgb_prob, rf_prob, nn_prob]
                votes = sum([1 for p in predictions if p > 0.5])

                if votes >= 2:  # At least 2 models agree
                    avg_prob = np.mean(predictions)
                    ensemble_agreement = votes / 4.0

                    signals[ticker] = {
                        'prob': avg_prob,
                        'agreement': ensemble_agreement,
                        'price': asset_data['Close'].iloc[-1]
                    }

            except Exception as e:
                continue

        # Execute signals (allocate capital across multiple assets)
        if signals:
            # Sort by probability (best signals first)
            sorted_signals = sorted(signals.items(), key=lambda x: x[1]['prob'], reverse=True)

            # Determine number of positions to open (max 3 at a time)
            max_positions = min(3, len(sorted_signals))
            capital_per_position = cash / max(max_positions, 1)

            for ticker, signal in sorted_signals[:max_positions]:
                if cash < capital_per_position * 0.5:  # Not enough cash
                    break

                # Calculate position size
                base_size = capital_per_position

                # Dynamic leverage based on signal strength
                if use_leverage:
                    leverage = 1.0

                    if signal['agreement'] >= 0.75:  # 3/4 models
                        leverage = 1.5
                    if signal['agreement'] >= 1.0:  # All 4 models
                        leverage = 1.8
                    if signal['prob'] >= 0.60:
                        leverage += 0.2

                    leverage = min(leverage, max_leverage)
                    position_value = base_size * leverage
                else:
                    position_value = base_size

                # Limit to available cash
                position_value = min(position_value, cash)

                if position_value >= 1000:  # Minimum position size
                    shares = position_value / signal['price']

                    positions[ticker] = {
                        'shares': shares,
                        'entry_price': signal['price'],
                        'entry_value': position_value,
                        'entry_date': current_date,
                        'leverage': leverage if use_leverage else 1.0,
                        'signal_prob': signal['prob'],
                        'agreement': signal['agreement']
                    }

                    cash -= position_value

        # Record daily portfolio value
        daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'num_positions': len(positions)
        })

        # Progress update
        if i % 252 == 0:
            years = i / 252
            print(f"Year {years:.1f}: Portfolio = ${portfolio_value:,.0f}, Positions = {len(positions)}")

    # Calculate final results
    final_value = daily_values[-1]['portfolio_value']
    total_return = ((final_value / initial_capital) - 1) * 100

    # Calculate annualized return
    days = len(daily_values)
    years = days / 252
    annualized_return = (((final_value / initial_capital) ** (1/years)) - 1) * 100

    # Calculate Sharpe ratio
    df_values = pd.DataFrame(daily_values)
    df_values['returns'] = df_values['portfolio_value'].pct_change()
    sharpe = (df_values['returns'].mean() / df_values['returns'].std()) * np.sqrt(252)

    # Calculate max drawdown
    df_values['cummax'] = df_values['portfolio_value'].cummax()
    df_values['drawdown'] = (df_values['portfolio_value'] / df_values['cummax'] - 1) * 100
    max_drawdown = df_values['drawdown'].min()

    # Trade statistics
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100 if len(trades_df) > 0 else 0

    results = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'num_trades': len(trades_df),
        'win_rate_pct': win_rate,
        'years': years,
        'trades': trades_df,
        'daily_values': df_values,
        'use_leverage': use_leverage,
        'max_leverage': max_leverage if use_leverage else 1.0
    }

    return results


def print_results(results, strategy_name="Multi-Asset Strategy"):
    """
    Print backtest results
    """
    print("\n" + "="*70)
    print(f"{strategy_name} - BACKTEST RESULTS")
    print("="*70)

    print(f"\n💰 RETURNS:")
    print(f"  Initial Capital:    ${results['initial_capital']:,.0f}")
    print(f"  Final Value:        ${results['final_value']:,.0f}")
    print(f"  Total Return:       {results['total_return_pct']:.2f}%")
    print(f"  Annualized Return:  {results['annualized_return_pct']:.2f}%")

    print(f"\n📊 RISK METRICS:")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")

    print(f"\n📈 TRADING STATS:")
    print(f"  Total Trades:       {results['num_trades']}")
    print(f"  Win Rate:           {results['win_rate_pct']:.2f}%")
    print(f"  Test Period:        {results['years']:.2f} years")

    if results['use_leverage']:
        print(f"\n⚡ LEVERAGE:")
        print(f"  Max Leverage:       {results['max_leverage']:.1f}x")
        avg_leverage = results['trades']['leverage'].mean() if 'leverage' in results['trades'].columns else 1.0
        print(f"  Avg Leverage:       {avg_leverage:.2f}x")

    # Asset breakdown
    if not results['trades'].empty:
        print(f"\n🎯 TRADES BY ASSET:")
        asset_stats = results['trades'].groupby('ticker').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        asset_stats.columns = ['Trades', 'Total PnL', 'Win Rate %']
        print(asset_stats.to_string())


def main():
    """
    Main function to run multi-asset backtest
    """
    print("="*70)
    print("MULTI-ASSET BACKTEST WITH REAL DATA")
    print("="*70)

    # Load real multi-asset data
    assets = load_multi_asset_real_data(
        tickers=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
        start_date='2015-01-01'
    )

    verify_data_quality(assets)

    # Load pre-trained models
    models = load_spy_models()

    if None in models:
        print("\n✗ Cannot run backtest without trained models")
        print("Please run advanced_strategies_implementation.py first to train models")
        return

    # Run backtest WITH leverage
    print("\n" + "="*70)
    print("STRATEGY 1: Multi-Asset with Dynamic Leverage (1.0-2.0x)")
    print("="*70)

    results_leverage = multi_asset_backtest(
        assets=assets,
        models=models,
        start_date='2020-09-03',
        initial_capital=100000,
        holding_period=10,
        use_leverage=True,
        max_leverage=2.0
    )

    print_results(results_leverage, "Multi-Asset + Leverage")

    # Run backtest WITHOUT leverage (baseline)
    print("\n" + "="*70)
    print("STRATEGY 2: Multi-Asset Baseline (No Leverage)")
    print("="*70)

    results_baseline = multi_asset_backtest(
        assets=assets,
        models=models,
        start_date='2020-09-03',
        initial_capital=100000,
        holding_period=10,
        use_leverage=False,
        max_leverage=1.0
    )

    print_results(results_baseline, "Multi-Asset Baseline")

    # Comparison
    print("\n" + "="*70)
    print("📊 STRATEGY COMPARISON")
    print("="*70)

    comparison = pd.DataFrame({
        'Baseline (No Leverage)': [
            f"{results_baseline['annualized_return_pct']:.2f}%",
            f"{results_baseline['sharpe_ratio']:.2f}",
            f"{results_baseline['max_drawdown_pct']:.2f}%",
            f"{results_baseline['win_rate_pct']:.2f}%",
            results_baseline['num_trades']
        ],
        'With Leverage (1-2x)': [
            f"{results_leverage['annualized_return_pct']:.2f}%",
            f"{results_leverage['sharpe_ratio']:.2f}",
            f"{results_leverage['max_drawdown_pct']:.2f}%",
            f"{results_leverage['win_rate_pct']:.2f}%",
            results_leverage['num_trades']
        ]
    }, index=['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades'])

    print(comparison.to_string())

    print("\n✓ Multi-asset backtest complete!")

    return results_leverage, results_baseline


if __name__ == '__main__':
    results_leverage, results_baseline = main()
