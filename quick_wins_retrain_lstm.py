#!/usr/bin/env python3
"""
QUICK WINS: LSTM with Focal Loss + Optimized Hyperparameters
==============================================================
Combines multiple quick wins into a single retrain:
1. Focal loss for class imbalance (73% bearish / 27% bullish)
2. Tuned hyperparameters (learning rate, dropout, architecture)
3. Better regularization

Expected improvement: +1-2% over current LSTM (67.81% → 68.81-69.81%)
This should push us toward the 70% target.
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    add_forward_returns_and_labels,
)
from train import TRAIN_CFG
# from create_sector_features import create_sector_features

print("=" * 80)
print("QUICK WINS: LSTM WITH FOCAL LOSS + OPTIMIZED HYPERPARAMETERS")
print("=" * 80)
print()

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.27):
    """
    Focal Loss for handling class imbalance
    
    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: weight for positive class (should match positive class proportion)
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Focal loss formula
        pos_loss = -alpha * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
        neg_loss = -(1.0 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        
        loss = y_true * pos_loss + (1.0 - y_true) * neg_loss
        return tf.reduce_mean(loss)
    
    return loss_fn

# ============================================================================
# 1. LOAD ALL FEATURES (same as original LSTM)
# ============================================================================

print("\n📥 Loading all feature sets...")

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

for col in cross_asset_df.columns:
    if col not in df_all.columns:
        df_all[col] = cross_asset_df[col]

for col in macro_df.columns:
    if col not in df_all.columns:
        df_all[col] = macro_df[col]

## Add sector features
#try:
#    sector_features = create_sector_features()
#    for col in sector_features.columns:
#        if col not in df_all.columns:
#            df_all[col] = sector_features[col]
#except:
#    pass

df_all = df_all.dropna()

print(f"✅ Loaded {len(df_all)} rows with {len(df_all.columns)-3} features")

# Get feature columns
feature_cols = [c for c in df_all.columns if c not in ["y", "fwd_ret_net", "Close"]]
print(f"📊 Using {len(feature_cols)} features")

# ============================================================================
# 2. CREATE SEQUENCES
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SEQUENTIAL DATA")
print("=" * 80)
print()

sequence_length = 20

# Split data (80/20)
split_idx = int(len(df_all) * 0.8)
train_df = df_all.iloc[:split_idx]
test_df = df_all.iloc[split_idx:]

X_train = train_df[feature_cols].values
y_train = train_df['y'].values
X_test = test_df[feature_cols].values
y_test = test_df['y'].values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sequences
def create_sequences(X, y, seq_len):
    sequences = []
    labels = []
    for i in range(seq_len, len(X)):
        sequences.append(X[i-seq_len:i])
        labels.append(y[i])
    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

print(f"✅ Created sequences:")
print(f"   Train: {len(X_train_seq)} sequences of shape {X_train_seq.shape[1:]}")
print(f"   Test:  {len(X_test_seq)} sequences of shape {X_test_seq.shape[1:]}")

# Class distribution
unique, counts = np.unique(y_train_seq, return_counts=True)
class_counts = dict(zip(unique, counts))
bullish_prop = class_counts.get(1, 0) / len(y_train_seq)
bearish_prop = class_counts.get(0, 0) / len(y_train_seq)

print(f"\n📊 Class distribution:")
print(f"   Class 0 (bearish): {class_counts.get(0, 0)} ({bearish_prop*100:.1f}%)")
print(f"   Class 1 (bullish): {class_counts.get(1, 0)} ({bullish_prop*100:.1f}%)")
print()

# ============================================================================
# 3. BUILD OPTIMIZED LSTM WITH FOCAL LOSS
# ============================================================================

print("=" * 80)
print("BUILDING OPTIMIZED LSTM ARCHITECTURE")
print("=" * 80)
print()

print("🎯 Quick Wins Applied:")
print("   1. Focal Loss (gamma=2.0, alpha={:.3f})".format(bullish_prop))
print("   2. Optimized learning rate (0.0005 vs 0.001)")
print("   3. Stronger regularization (dropout 0.4 vs 0.3)")
print("   4. Batch normalization for stability")
print()

def build_optimized_lstm(sequence_length, n_features):
    """Optimized LSTM with better hyperparameters"""
    model = keras.Sequential([
        # First LSTM layer - increased units for more capacity
        layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.4),  # Stronger regularization
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        # Third LSTM layer
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

model = build_optimized_lstm(sequence_length, len(feature_cols))

# Compile with Focal Loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR for stability
    loss=focal_loss(gamma=2.0, alpha=bullish_prop),
    metrics=['accuracy', keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall')]
)

print(model.summary())
print()

# ============================================================================
# 4. TRAIN WITH EARLY STOPPING
# ============================================================================

print("=" * 80)
print("TRAINING OPTIMIZED LSTM")
print("=" * 80)
print()

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

print("⏳ Training model with focal loss...")
print("   Epochs: 100 (with early stopping)")
print("   Batch size: 32")
print("   Validation split: 20%")
print()

history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 5. EVALUATE AND COMPARE
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)
print()

# Predictions
y_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test_seq, y_pred)
prec = precision_score(y_test_seq, y_pred, zero_division=0)
rec = recall_score(y_test_seq, y_pred, zero_division=0)
f1 = f1_score(y_test_seq, y_pred, zero_division=0)

print("Test Results:")
print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print()

# Compare with original LSTM
original_acc = 0.6781
improvement = acc - original_acc

print("📊 Comparison:")
print(f"   Original LSTM (binary cross-entropy): 67.81%")
print(f"   Quick Wins LSTM (focal loss + optimized): {acc*100:.2f}%")
if improvement > 0:
    print(f"   ✅ Improvement: +{improvement*100:.2f}%")
else:
    print(f"   Change: {improvement*100:.2f}%")
print()

# ============================================================================
# 6. SAVE MODEL
# ============================================================================

print("=" * 80)
print("SAVING MODEL")
print("=" * 80)
print()

model.save(MODELS_DIR / "lstm_quickwins.h5")
joblib.dump(scaler, MODELS_DIR / "lstm_quickwins_scaler.pkl")

print(f"💾 Saved model: {MODELS_DIR / 'lstm_quickwins.h5'}")
print(f"💾 Saved scaler: {MODELS_DIR / 'lstm_quickwins_scaler.pkl'}")

# Save comparison
comparison_df = pd.DataFrame({
    'Model': ['LSTM (Original)', 'LSTM (Quick Wins)'],
    'Accuracy': [original_acc, acc],
    'Precision': [0.2954, prec],
    'Recall': [0.3707, rec],
    'F1': [0.3220, f1],
    'Loss_Function': ['Binary Cross-Entropy', 'Focal Loss'],
    'Learning_Rate': [0.001, 0.0005],
    'Dropout': [0.3, 0.4]
})

comparison_df.to_csv("lstm_quickwins_comparison.csv", index=False)
print(f"💾 Saved comparison: lstm_quickwins_comparison.csv")
print()

print("=" * 80)
print("✅ QUICK WINS LSTM TRAINING COMPLETE")
print("=" * 80)
print()

# Calculate journey
baseline = 0.5838
lstm_orig = 0.6781
lstm_new = acc

print("🎯 Complete Journey:")
print(f"   Baseline:                        58.38%")
print(f"   Phase 4 (Original LSTM):         67.81% (+9.43%)")
print(f"   Quick Wins (Focal Loss + Opt):   {lstm_new*100:.2f}% (+{(lstm_new-baseline)*100:.2f}%)")
print()

if acc >= 0.70:
    print("🎊 TARGET ACHIEVED: 70%+ accuracy!")
elif acc >= 0.69:
    print("🎉 Nearly there! Just 1% from 70% target!")
else:
    print(f"📈 Progress: {((acc-lstm_orig)/(0.70-lstm_orig))*100:.1f}% toward 70% target")

print()
print("💡 Next Steps:")
print("   1. Retrain Transformer with focal loss")
print("   2. Update ensemble with new models")
print("   3. Run full hyperparameter optimization with Optuna")
