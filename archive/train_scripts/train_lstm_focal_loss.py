#!/usr/bin/env python3
"""
Retrain LSTM with Focal Loss - Quick Win #3
============================================
Retrain LSTM using focal loss instead of binary cross-entropy to better
handle the 73% bearish / 27% bullish class imbalance.

Expected improvement: +0.5-1% over current LSTM (67.81% → 68.31-68.81%)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import focal loss
from focal_loss_utils import FocalLoss, binary_focal_loss

# Import data loading
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels
from train import TRAIN_CFG

print("=" * 80)
print("LSTM WITH FOCAL LOSS - QUICK WIN #3")
print("=" * 80)
print()

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("📥 Loading data and features...")
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

# Load additional features
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)

existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

df_all = df[existing_features + ["y", "fwd_ret_net", "Close"]].copy()

for col in cross_asset_df.columns:
    if col not in df_all.columns:
        df_all[col] = cross_asset_df[col]

for col in macro_df.columns:
    if col not in df_all.columns:
        df_all[col] = macro_df[col]

df_all = df_all.dropna()

print(f"✅ Loaded {len(df_all)} rows")

# Get feature columns
feature_cols = [c for c in df_all.columns if c not in ["y", "fwd_ret_net", "Close"]]
print(f"📊 Using {len(feature_cols)} features")
print()

# ============================================================================
# 2. PREPARE SEQUENTIAL DATA
# ============================================================================

print("=" * 80)
print("CREATING SEQUENCES FOR LSTM")
print("=" * 80)
print()

sequence_length = 20

# Split data
split_idx = int(len(df_all) * 0.8)
X_train = df_all[feature_cols].iloc[:split_idx].values
y_train = df_all['y'].iloc[:split_idx].values
X_test = df_all[feature_cols].iloc[split_idx:].values
y_test = df_all['y'].iloc[split_idx:].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sequences
train_sequences = []
y_train_seq = []
for i in range(sequence_length, len(X_train_scaled)):
    train_sequences.append(X_train_scaled[i-sequence_length:i])
    y_train_seq.append(y_train[i])

train_sequences = np.array(train_sequences)
y_train_seq = np.array(y_train_seq)

test_sequences = []
y_test_seq = []
for i in range(sequence_length, len(X_test_scaled)):
    test_sequences.append(X_test_scaled[i-sequence_length:i])
    y_test_seq.append(y_test[i])

test_sequences = np.array(test_sequences)
y_test_seq = np.array(y_test_seq)

print(f"✅ Created sequences:")
print(f"   Train: {train_sequences.shape}")
print(f"   Test:  {test_sequences.shape}")
print()

# Class distribution
unique, counts = np.unique(y_train_seq, return_counts=True)
class_dist = dict(zip(unique, counts))
bullish_prop = class_dist.get(1, 0) / len(y_train_seq)

print(f"📊 Class distribution:")
print(f"   Bearish (0): {class_dist.get(0, 0)} ({100*(1-bullish_prop):.1f}%)")
print(f"   Bullish (1): {class_dist.get(1, 0)} ({100*bullish_prop:.1f}%)")
print()

# ============================================================================
# 3. BUILD LSTM WITH FOCAL LOSS
# ============================================================================

print("=" * 80)
print("BUILDING LSTM WITH FOCAL LOSS")
print("=" * 80)
print()

def build_lstm_model(sequence_length, n_features):
    """Build LSTM architecture"""
    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
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

model = build_lstm_model(sequence_length, len(feature_cols))

# Compile with Focal Loss
print("Compiling model with Focal Loss...")
print(f"   gamma=2.0 (focus on hard examples)")
print(f"   alpha={bullish_prop:.3f} (bullish class weight)")
print()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=binary_focal_loss(gamma=2.0, alpha=bullish_prop),
    metrics=['accuracy', keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall')]
)

print(model.summary())
print()

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================

print("=" * 80)
print("TRAINING LSTM WITH FOCAL LOSS")
print("=" * 80)
print()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
]

print("Training...")
history = model.fit(
    train_sequences, y_train_seq,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 5. EVALUATE
# ============================================================================

print()
print("=" * 80)
print("EVALUATION")
print("=" * 80)
print()

# Predictions
y_pred_proba = model.predict(test_sequences, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test_seq, y_pred)
prec = precision_score(y_test_seq, y_pred, zero_division=0)
rec = recall_score(y_test_seq, y_pred, zero_division=0)
f1 = f1_score(y_test_seq, y_pred, zero_division=0)

print(f"Test Results:")
print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print()

# Compare with original LSTM (67.81%)
original_acc = 0.6781
improvement = acc - original_acc

print(f"📊 Comparison:")
print(f"   Original LSTM (binary cross-entropy): 67.81%")
print(f"   LSTM with Focal Loss: {acc*100:.2f}%")
if improvement > 0:
    print(f"   Improvement: +{improvement*100:.2f}%")
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

model.save(MODELS_DIR / "lstm_focal_loss.h5")
joblib.dump(scaler, MODELS_DIR / "lstm_focal_scaler.pkl")

print(f"💾 Saved model: {MODELS_DIR / 'lstm_focal_loss.h5'}")
print(f"💾 Saved scaler: {MODELS_DIR / 'lstm_focal_scaler.pkl'}")
print()

# Save comparison
comparison = pd.DataFrame({
    'Model': ['LSTM (Binary Cross-Entropy)', 'LSTM (Focal Loss)'],
    'Accuracy': [original_acc, acc],
    'Precision': [0.2954, prec],  # From previous training
    'Recall': [0.3707, rec],
    'F1': [0.3220, f1]
})

comparison.to_csv("lstm_focal_loss_comparison.csv", index=False)
print(f"💾 Saved comparison: lstm_focal_loss_comparison.csv")
print()

print("=" * 80)
print("✅ LSTM WITH FOCAL LOSS COMPLETE")
print("=" * 80)
print()

print("💡 Next Step: Run hyperparameter optimization with Optuna")
