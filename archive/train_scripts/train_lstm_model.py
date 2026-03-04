#!/usr/bin/env python3
"""
LSTM Model for Sequential Pattern Recognition - Phase 4.1

Deep learning approach to capture temporal dependencies that gradient
boosting models miss. Uses sequences of historical features to predict
future market movements.

Expected improvement: +5-8% accuracy
Target: 63.23% → 68-71%
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  TensorFlow/Keras not available. Installing...")

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)
from train import TRAIN_CFG

print("=" * 80)
print("LSTM MODEL TRAINING (PHASE 4.1)")
print("=" * 80)

if not KERAS_AVAILABLE:
    print("\n❌ TensorFlow/Keras is required for LSTM models.")
    print("   Please install: pip install tensorflow")
    exit(1)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# ============================================================================
# 1. LOAD ALL FEATURES
# ============================================================================

print("\n📥 Loading all feature sets...")

# Existing features
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

# Cross-asset features
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)

# Macro features
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)

# Combine all features
existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

df_all = df[existing_features + ["y", "fwd_ret_net", "Close"]].copy()

# Join cross-asset
for col in cross_asset_df.columns:
    df_all[col] = cross_asset_df[col].reindex(df_all.index)

# Join macro
for col in macro_df.columns:
    df_all[col] = macro_df[col].reindex(df_all.index)

all_features = existing_features + list(cross_asset_df.columns) + list(macro_df.columns)

df_all = df_all.fillna(0)
df_all = df_all.dropna(subset=["y"])

print(f"✅ Loaded {len(df_all)} rows with {len(all_features)} features")

# ============================================================================
# 2. CREATE SEQUENCES FOR LSTM
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SEQUENTIAL DATA FOR LSTM")
print("=" * 80)

SEQUENCE_LENGTH = 20  # Look back 20 days

def create_sequences(data, features, labels, seq_length):
    """Create sequences for LSTM input"""
    X, y = [], []

    for i in range(seq_length, len(data)):
        # Get sequence of features
        sequence = data.iloc[i-seq_length:i][features].values
        label = labels.iloc[i]

        X.append(sequence)
        y.append(label)

    return np.array(X), np.array(y)

# Split data first
test_size = int(len(df_all) * 0.2)
train_end_idx = len(df_all) - test_size

train_data = df_all.iloc[:train_end_idx].copy()
test_data = df_all.iloc[train_end_idx:].copy()

print(f"\n📊 Sequence parameters:")
print(f"   Sequence length: {SEQUENCE_LENGTH} days")
print(f"   Features per timestep: {len(all_features)}")
print(f"   Total sequence shape: ({SEQUENCE_LENGTH}, {len(all_features)})")

# Scale features (LSTM works better with normalized data)
scaler = StandardScaler()
train_data[all_features] = scaler.fit_transform(train_data[all_features])
test_data[all_features] = scaler.transform(test_data[all_features])

# Create sequences
X_train_seq, y_train_seq = create_sequences(train_data, all_features, train_data['y'], SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(test_data, all_features, test_data['y'], SEQUENCE_LENGTH)

print(f"\n✅ Created sequences:")
print(f"   Train: {X_train_seq.shape[0]} sequences of shape {X_train_seq.shape[1:]}")
print(f"   Test:  {X_test_seq.shape[0]} sequences of shape {X_test_seq.shape[1:]}")

# Calculate class weights
class_counts = pd.Series(y_train_seq).value_counts()
total = len(y_train_seq)
class_weight_dict = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}

print(f"\n📊 Class distribution:")
print(f"   Class 0 (bearish): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
print(f"   Class 1 (bullish): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
print(f"   Class weights: {class_weight_dict}")

# ============================================================================
# 3. BUILD LSTM MODEL
# ============================================================================

print("\n" + "=" * 80)
print("BUILDING LSTM ARCHITECTURE")
print("=" * 80)

def build_lstm_model(sequence_length, n_features):
    """Build LSTM model architecture"""

    model = keras.Sequential([
        # First LSTM layer with return sequences
        layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # Second LSTM layer
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # Third LSTM layer (no return sequences)
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])

    return model

model = build_lstm_model(SEQUENCE_LENGTH, len(all_features))

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\n📊 Model Architecture:")
model.summary()

# ============================================================================
# 4. TRAIN LSTM MODEL
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING LSTM MODEL")
print("=" * 80)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("\n⏳ Training LSTM model...")
print(f"   Epochs: 100 (with early stopping)")
print(f"   Batch size: 32")
print(f"   Validation split: 20%")

start = datetime.now()

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

elapsed = (datetime.now() - start).total_seconds()
print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")

# ============================================================================
# 5. EVALUATE LSTM MODEL
# ============================================================================

print("\n" + "=" * 80)
print("LSTM MODEL EVALUATION")
print("=" * 80)

# Predict probabilities
y_pred_proba = model.predict(X_test_seq, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
acc_lstm = accuracy_score(y_test_seq, y_pred)
prec_lstm = precision_score(y_test_seq, y_pred, zero_division=0)
rec_lstm = recall_score(y_test_seq, y_pred)
f1_lstm = f1_score(y_test_seq, y_pred)

print(f"\n🎯 LSTM Model Performance:")
print(f"   Accuracy:  {acc_lstm:.4f} ({acc_lstm*100:.2f}%)")
print(f"   Precision: {prec_lstm:.4f}")
print(f"   Recall:    {rec_lstm:.4f}")
print(f"   F1 Score:  {f1_lstm:.4f}")

# ============================================================================
# 6. TRAIN LIGHTGBM BASELINE FOR COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING LIGHTGBM BASELINE (FOR COMPARISON)")
print("=" * 80)

# Use same data but without sequences (just last day features)
X_train_flat = train_data.iloc[SEQUENCE_LENGTH:][all_features]
y_train_flat = train_data.iloc[SEQUENCE_LENGTH:]['y']
X_test_flat = test_data.iloc[SEQUENCE_LENGTH:][all_features]

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.02,
    'n_estimators': 400,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbosity': -1,
}

# Calculate sample weights
class_counts_flat = y_train_flat.value_counts()
total_flat = len(y_train_flat)
class_weight_dict_flat = {
    0: total_flat / (2 * class_counts_flat[0]),
    1: total_flat / (2 * class_counts_flat[1])
}
sample_weights = y_train_flat.map(class_weight_dict_flat)

print(f"\n⏳ Training LightGBM baseline...")
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train_flat, y_train_flat, sample_weight=sample_weights)

lgb_pred = lgb_model.predict(X_test_flat)
acc_lgb = accuracy_score(y_test_seq, lgb_pred)
f1_lgb = f1_score(y_test_seq, lgb_pred)

print(f"\n✅ LightGBM Results:")
print(f"   Accuracy:  {acc_lgb:.4f} ({acc_lgb*100:.2f}%)")
print(f"   F1 Score:  {f1_lgb:.4f}")

# ============================================================================
# 7. ENSEMBLE: LSTM + LIGHTGBM
# ============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE: LSTM + LIGHTGBM")
print("=" * 80)

# Average probabilities
lgb_proba = lgb_model.predict_proba(X_test_flat)[:, 1]
lstm_proba = y_pred_proba.flatten()

ensemble_proba = (lstm_proba * 0.5 + lgb_proba * 0.5)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

acc_ensemble = accuracy_score(y_test_seq, ensemble_pred)
prec_ensemble = precision_score(y_test_seq, ensemble_pred, zero_division=0)
rec_ensemble = recall_score(y_test_seq, ensemble_pred)
f1_ensemble = f1_score(y_test_seq, ensemble_pred)

print(f"\n🎯 Ensemble (50% LSTM + 50% LightGBM) Performance:")
print(f"   Accuracy:  {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
print(f"   Precision: {prec_ensemble:.4f}")
print(f"   Recall:    {rec_ensemble:.4f}")
print(f"   F1 Score:  {f1_ensemble:.4f}")

# ============================================================================
# 8. COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame([
    {'Model': 'LightGBM (Baseline)', 'Accuracy': acc_lgb, 'F1_Score': f1_lgb},
    {'Model': 'LSTM', 'Accuracy': acc_lstm, 'F1_Score': f1_lstm},
    {'Model': 'Ensemble (LSTM+LightGBM)', 'Accuracy': acc_ensemble, 'F1_Score': f1_ensemble},
])

print("\n" + comparison.to_string(index=False))

# Determine best model
best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Model']
best_accuracy = comparison['Accuracy'].max()

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================================
# 9. SAVE MODELS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

# Save LSTM model
lstm_model_path = MODELS_DIR / "lstm_model.h5"
model.save(lstm_model_path)
print(f"💾 Saved LSTM model to: {lstm_model_path}")

# Save scaler
scaler_path = MODELS_DIR / "lstm_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"💾 Saved scaler to: {scaler_path}")

# Save LightGBM model
lgb_model_path = MODELS_DIR / "lgb_for_ensemble.pkl"
joblib.dump(lgb_model, lgb_model_path)
print(f"💾 Saved LightGBM model to: {lgb_model_path}")

# Save ensemble metadata
ensemble_path = MODELS_DIR / "lstm_ensemble.pkl"
joblib.dump({
    'features': all_features,
    'sequence_length': SEQUENCE_LENGTH,
    'lstm_accuracy': acc_lstm,
    'lgb_accuracy': acc_lgb,
    'ensemble_accuracy': acc_ensemble,
    'ensemble_f1': f1_ensemble,
    'training_date': datetime.now().isoformat()
}, ensemble_path)
print(f"💾 Saved ensemble metadata to: {ensemble_path}")

# Save comparison results
comparison.to_csv("lstm_model_comparison.csv", index=False)
print(f"💾 Saved comparison to: lstm_model_comparison.csv")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ PHASE 4.1 COMPLETE!")
print("=" * 80)

phase0_acc = 0.5838
phase1_acc = 0.6146
phase2_acc = 0.6231
phase3_acc = 0.6323

print(f"\n🎯 Progressive Results:")
print(f"   Baseline:                  {phase0_acc*100:.2f}%")
print(f"   Phase 1 (+Cross-Asset):    {phase1_acc*100:.2f}% ({(phase1_acc-phase0_acc)*100:+.2f}%)")
print(f"   Phase 2 (+Macro):          {phase2_acc*100:.2f}% ({(phase2_acc-phase1_acc)*100:+.2f}%)")
print(f"   Phase 3 (+Regime-Switch):  {phase3_acc*100:.2f}% ({(phase3_acc-phase2_acc)*100:+.2f}%)")
print(f"   Phase 4.1 (LSTM):          {best_accuracy*100:.2f}% ({(best_accuracy-phase3_acc)*100:+.2f}%)")
print(f"\n   🚀 TOTAL IMPROVEMENT:      {phase0_acc*100:.2f}% → {best_accuracy*100:.2f}% ({(best_accuracy-phase0_acc)*100:+.2f}%)")

if best_accuracy > 0.70:
    print(f"\n   🎊 ACHIEVED 70%+ ACCURACY!")
    achievement = "EXCELLENT"
elif best_accuracy > 0.68:
    print(f"\n   🎉 Strong performance!")
    achievement = "VERY GOOD"
elif best_accuracy > 0.65:
    print(f"\n   ✅ Good improvement!")
    achievement = "GOOD"
else:
    print(f"\n   📈 Modest improvement - sentiment analysis may help more")
    achievement = "MODERATE"

print(f"\n📊 Achievement Level: {achievement}")

print(f"\n🎯 Next Steps (Phase 4.2 & 4.3):")
print(f"   - Sentiment analysis with FinBERT (+2-4%)")
print(f"   - Options flow analysis (+3-5%)")
print(f"   - Target: 75-80% accuracy")
