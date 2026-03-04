#!/usr/bin/env python3
"""
Multi-Horizon Economic Forecaster

Predicts market returns across multiple time horizons with uncertainty quantification.
This is the first step toward comprehensive economic forecasting.

Horizons:
- 1 day: Very short-term (day trading)
- 5 days: Short-term (weekly swing)
- 21 days: Medium-term (monthly)
- 63 days: Long-term (quarterly)

For each horizon, predicts:
- Point estimate (expected return)
- Confidence intervals (10th, 50th, 90th percentile)
- Direction probability (up vs down)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
)

print("=" * 80)
print("MULTI-HORIZON ECONOMIC FORECASTER")
print("=" * 80)

MODELS_DIR = Path("models/multi_horizon")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Horizons to predict (in trading days)
HORIZONS = {
    '1d': 1,      # Next day
    '1w': 5,      # 1 week
    '1m': 21,     # 1 month
    '3m': 63,     # 3 months
}

# ============================================================================
# 1. PREPARE DATA WITH MULTIPLE TARGETS
# ============================================================================

print("\n📥 Loading data...")
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

print(f"✅ Data loaded: {len(df)} rows")
print(f"   Features: {len(feature_cols)}")

# ============================================================================
# 2. GENERATE FORWARD RETURNS FOR EACH HORIZON
# ============================================================================

print("\n📊 Generating forward returns for each horizon...")

for name, days in HORIZONS.items():
    # Forward return
    df[f'fwd_ret_{name}'] = df['Close'].pct_change(days).shift(-days)

    # Forward volatility (realized vol over the horizon)
    df[f'fwd_vol_{name}'] = df['Close'].pct_change().rolling(days).std().shift(-days) * np.sqrt(252)

    # Direction (1 if positive return, 0 if negative)
    df[f'fwd_dir_{name}'] = (df[f'fwd_ret_{name}'] > 0).astype(int)

    valid_count = df[f'fwd_ret_{name}'].notna().sum()
    print(f"   {name:3s} ({days:2d} days): {valid_count:,} valid samples")

# ============================================================================
# 3. PREPARE FEATURES
# ============================================================================

# Get all features
all_features = [c for c in df.columns if c not in
                ['Close', 'fwd_ret_1d', 'fwd_ret_1w', 'fwd_ret_1m', 'fwd_ret_3m',
                 'fwd_vol_1d', 'fwd_vol_1w', 'fwd_vol_1m', 'fwd_vol_3m',
                 'fwd_dir_1d', 'fwd_dir_1w', 'fwd_dir_1m', 'fwd_dir_3m',
                 'y', 'fwd_ret_net', 'fwd_ret_raw', 'fwd_price', 'horizon_forward']]

# Fill NaN in features
df[all_features] = df[all_features].fillna(0)

print(f"\n✅ Feature preparation complete")
print(f"   Total features: {len(all_features)}")

# ============================================================================
# 4. SPLIT DATA
# ============================================================================

# Use 80/20 split
test_size = int(len(df) * 0.2)
train_end_idx = len(df) - test_size

X_train = df.iloc[:train_end_idx][all_features]
X_test = df.iloc[train_end_idx:][all_features]

print(f"\n📅 Data split:")
print(f"   Train: {len(X_train)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[train_end_idx-1].strftime('%Y-%m-%d')})")
print(f"   Test:  {len(X_test)} rows ({df.index[train_end_idx].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

# ============================================================================
# 5. TRAIN MODELS FOR EACH HORIZON
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING MULTI-HORIZON MODELS")
print("=" * 80)

results = {}

for horizon_name, horizon_days in HORIZONS.items():
    print(f"\n{'─' * 80}")
    print(f"HORIZON: {horizon_name} ({horizon_days} days)")
    print(f"{'─' * 80}")

    # Get targets
    y_train = df.iloc[:train_end_idx][f'fwd_ret_{horizon_name}']
    y_test = df.iloc[train_end_idx:][f'fwd_ret_{horizon_name}']

    # Remove NaN targets
    train_valid = y_train.notna()
    test_valid = y_test.notna()

    X_train_valid = X_train[train_valid]
    y_train_valid = y_train[train_valid]
    X_test_valid = X_test[test_valid]
    y_test_valid = y_test[test_valid]

    print(f"\n   Valid samples: {len(y_train_valid)} train, {len(y_test_valid)} test")

    # Train 3 models: median (q50), lower bound (q10), upper bound (q90)
    models = {}

    for quantile, alpha in [('q10', 0.10), ('q50', 0.50), ('q90', 0.90)]:
        print(f"\n   [{quantile}] Training quantile regression (alpha={alpha})...")

        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.02,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
        )

        start = datetime.now()
        model.fit(X_train_valid, y_train_valid)
        elapsed = (datetime.now() - start).total_seconds()

        # Predict on test
        y_pred = model.predict(X_test_valid)

        # Calculate metrics
        mse = mean_squared_error(y_test_valid, y_pred)
        mae = mean_absolute_error(y_test_valid, y_pred)
        rmse = np.sqrt(mse)

        print(f"        Completed in {elapsed:.1f}s")
        print(f"        MAE: {mae:.6f} ({mae*100:.4f}%)")
        print(f"        RMSE: {rmse:.6f} ({rmse*100:.4f}%)")

        # Save model
        model_path = MODELS_DIR / f"model_{horizon_name}_{quantile}.pkl"
        joblib.dump(model, model_path)

        models[quantile] = {
            'model': model,
            'predictions': y_pred,
            'mae': mae,
            'rmse': rmse
        }

    # Combine predictions for analysis
    predictions = pd.DataFrame({
        'actual': y_test_valid.values,
        'pred_q10': models['q10']['predictions'],
        'pred_median': models['q50']['predictions'],
        'pred_q90': models['q90']['predictions'],
    })

    # Calculate prediction interval coverage (should be ~80%)
    predictions['in_interval'] = (
        (predictions['actual'] >= predictions['pred_q10']) &
        (predictions['actual'] <= predictions['pred_q90'])
    )
    coverage = predictions['in_interval'].mean()

    # Direction accuracy
    predictions['actual_dir'] = (predictions['actual'] > 0).astype(int)
    predictions['pred_dir'] = (predictions['pred_median'] > 0).astype(int)
    direction_acc = (predictions['actual_dir'] == predictions['pred_dir']).mean()

    # R-squared
    r2 = r2_score(predictions['actual'], predictions['pred_median'])

    print(f"\n   ✅ Results Summary:")
    print(f"      R² Score: {r2:.4f}")
    print(f"      Direction Accuracy: {direction_acc:.2%}")
    print(f"      80% Interval Coverage: {coverage:.2%} (target: 80%)")
    print(f"      Median MAE: {models['q50']['mae']*100:.4f}%")

    # Save results
    results[horizon_name] = {
        'horizon_days': horizon_days,
        'r2': r2,
        'direction_accuracy': direction_acc,
        'interval_coverage': coverage,
        'mae': models['q50']['mae'],
        'rmse': models['q50']['rmse'],
        'predictions': predictions,
        'models': models
    }

    # Save predictions
    predictions.to_csv(OUTPUTS_DIR / f"predictions_{horizon_name}.csv", index=False)

# ============================================================================
# 6. RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MULTI-HORIZON PERFORMANCE COMPARISON")
print("=" * 80)

comparison = pd.DataFrame([
    {
        'Horizon': name,
        'Days': HORIZONS[name],
        'R²': results[name]['r2'],
        'Direction_Acc': results[name]['direction_accuracy'],
        'MAE_%': results[name]['mae'] * 100,
        'RMSE_%': results[name]['rmse'] * 100,
        'Interval_Coverage': results[name]['interval_coverage'],
    }
    for name in HORIZONS.keys()
])

print("\n" + comparison.to_string(index=False))

comparison.to_csv(OUTPUTS_DIR / "multi_horizon_performance.csv", index=False)
print(f"\n💾 Saved: outputs/multi_horizon_performance.csv")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Plot 1: Prediction accuracy vs horizon
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² vs Horizon
ax = axes[0, 0]
ax.plot(comparison['Days'], comparison['R²'], 'o-', linewidth=2, markersize=8, color='blue')
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('R² Score')
ax.set_title('Prediction Quality vs Horizon', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Direction Accuracy vs Horizon
ax = axes[0, 1]
ax.plot(comparison['Days'], comparison['Direction_Acc'] * 100, 'o-', linewidth=2, markersize=8, color='green')
ax.axhline(y=50, color='red', linestyle='--', label='Random (50%)', alpha=0.5)
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Direction Accuracy (%)')
ax.set_title('Direction Prediction Accuracy', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# MAE vs Horizon
ax = axes[1, 0]
ax.plot(comparison['Days'], comparison['MAE_%'], 'o-', linewidth=2, markersize=8, color='orange')
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Mean Absolute Error (%)')
ax.set_title('Prediction Error vs Horizon', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Interval Coverage
ax = axes[1, 1]
ax.bar(comparison['Horizon'], comparison['Interval_Coverage'] * 100, color='purple', alpha=0.7)
ax.axhline(y=80, color='red', linestyle='--', label='Target (80%)', linewidth=2)
ax.set_xlabel('Horizon')
ax.set_ylabel('Coverage (%)')
ax.set_title('80% Prediction Interval Coverage', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'multi_horizon_performance.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/multi_horizon_performance.png")

# Plot 2: Actual vs Predicted for each horizon
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    preds = result['predictions']

    # Scatter plot
    ax.scatter(preds['actual'] * 100, preds['pred_median'] * 100,
              alpha=0.3, s=10, label='Predictions')

    # Perfect prediction line
    lim = max(abs(preds['actual'].min()), abs(preds['actual'].max())) * 100
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Perfect', alpha=0.7)

    # Prediction interval
    sample_indices = np.random.choice(len(preds), min(200, len(preds)), replace=False)
    for i in sample_indices:
        row = preds.iloc[i]
        ax.plot([row['actual'] * 100, row['actual'] * 100],
               [row['pred_q10'] * 100, row['pred_q90'] * 100],
               color='gray', alpha=0.1, linewidth=1)

    ax.set_xlabel('Actual Return (%)')
    ax.set_ylabel('Predicted Return (%)')
    ax.set_title(f'{name} ({HORIZONS[name]} days) - R²={result["r2"]:.3f}',
                fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'multi_horizon_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/multi_horizon_predictions.png")

# ============================================================================
# 8. EXAMPLE FORECAST
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE: LATEST FORECAST")
print("=" * 80)

print("\nAs of:", df.index[-1].strftime('%Y-%m-%d'))
print(f"Current SPY Close: ${df['Close'].iloc[-1]:.2f}")

latest_features = X_test.iloc[-1:]

print("\nForecasted Returns:")
print(f"{'Horizon':<10} {'Lower (10%)':<15} {'Expected (50%)':<18} {'Upper (90%)':<15} {'Price Target'}")
print("─" * 85)

for name in HORIZONS.keys():
    q10_model = results[name]['models']['q10']['model']
    q50_model = results[name]['models']['q50']['model']
    q90_model = results[name]['models']['q90']['model']

    q10_pred = q10_model.predict(latest_features)[0]
    q50_pred = q50_model.predict(latest_features)[0]
    q90_pred = q90_model.predict(latest_features)[0]

    current_price = df['Close'].iloc[-1]
    target_price = current_price * (1 + q50_pred)

    print(f"{name:<10} {q10_pred:>+.4f} ({q10_pred*100:>+6.2f}%)   "
          f"{q50_pred:>+.4f} ({q50_pred*100:>+6.2f}%)   "
          f"{q90_pred:>+.4f} ({q90_pred*100:>+6.2f}%)   ${target_price:.2f}")

print("\n" + "=" * 80)
print("✅ MULTI-HORIZON TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📁 Models saved to: {MODELS_DIR}/")
print(f"📁 Results saved to: {OUTPUTS_DIR}/")
print("\nKey Findings:")
print(f"   Best R²: {comparison['R²'].max():.4f} ({comparison.loc[comparison['R²'].idxmax(), 'Horizon']} horizon)")
print(f"   Best Direction Acc: {comparison['Direction_Acc'].max():.2%} ({comparison.loc[comparison['Direction_Acc'].idxmax(), 'Horizon']} horizon)")
print(f"   Avg Interval Coverage: {comparison['Interval_Coverage'].mean():.2%}")
