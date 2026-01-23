#!/usr/bin/env python3
"""
CNN-LSTM Hybrid Model
=====================
CNN for pattern extraction + LSTM for temporal sequences

Architecture:
1. 1D CNN layers to extract local patterns from sequences
2. Max pooling to reduce dimensionality
3. LSTM layers to capture temporal dependencies
4. Dense layers for classification

Expected impact: +1-2%

Advantages:
- CNN extracts local patterns (support/resistance, chart patterns)
- LSTM captures long-term dependencies
- Combination leverages both spatial and temporal features
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("=" * 80)
print("CNN-LSTM HYBRID MODEL")
print("=" * 80)
print()

np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
SEQUENCE_LENGTH = 20

# Load selected features
print("📥 Loading selected features...")
with open("selected_features.txt", 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"✅ Using {len(selected_features)} selected features")
print()

# Load data
print("📥 Loading SPY data...")
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels
from train import TRAIN_CFG

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

# Load cross-asset and macro
cross_asset_df = pd.read_csv(DATA_DIR / "cross_asset_features.csv", index_col=0, parse_dates=True)
macro_df = pd.read_csv(DATA_DIR / "macro_features.csv", index_col=0, parse_dates=True)

existing_features = [c for c in df.columns if c not in
                    ["y", "fwd_ret_net", "fwd_ret_raw", "fwd_price", "horizon_forward", "Close"]]

df_all = df[existing_features + ["y", "Close"]].copy()

for col in cross_asset_df.columns:
    df_all[col] = cross_asset_df[col].reindex(df_all.index)

for col in macro_df.columns:
    df_all[col] = macro_df[col].reindex(df_all.index)

df_all = df_all.fillna(0)
df_all = df_all.dropna(subset=["y"])

# Filter to selected features
available_features = [f for f in selected_features if f in df_all.columns]
X = df_all[available_features].values
y = df_all["y"].values

print(f"✅ Loaded {len(X)} samples with {len(available_features)} features")
print()

# Create sequences
print("🔄 Creating sequences...")

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)
print(f"✅ Created {len(X_seq)} sequences of shape {X_seq.shape[1:]}")
print()

# Train/test split
split_idx = int(len(X_seq) * 0.8)
X_train_seq = X_seq[:split_idx]
y_train_seq = y_seq[:split_idx]
X_test_seq = X_seq[split_idx:]
y_test_seq = y_seq[split_idx:]

print(f"📊 Data split:")
print(f"   Train: {len(X_train_seq)} sequences")
print(f"   Test:  {len(X_test_seq)} sequences")
print()

# Scale features
print("⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, len(available_features)))
X_train_seq = X_train_scaled.reshape(X_train_seq.shape)

X_test_scaled = scaler.transform(X_test_seq.reshape(-1, len(available_features)))
X_test_seq = X_test_scaled.reshape(X_test_seq.shape)

# Class weights
class_counts = pd.Series(y_train_seq).value_counts()
total = len(y_train_seq)
class_weight_dict = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}

print(f"📊 Class distribution:")
print(f"   Bearish: {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
print(f"   Bullish: {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
print()

# ============================================================================
# BUILD CNN-LSTM HYBRID MODEL
# ============================================================================

print("=" * 80)
print("BUILDING CNN-LSTM HYBRID ARCHITECTURE")
print("=" * 80)
print()

def build_cnn_lstm(sequence_length, n_features):
    """
    CNN-LSTM Hybrid Architecture

    1. 1D Convolutional layers for local pattern extraction
    2. Max pooling for dimensionality reduction
    3. LSTM layers for temporal sequence modeling
    4. Dense layers for final classification

    Advantages over pure LSTM:
    - CNN captures local patterns (chart patterns, support/resistance)
    - LSTM captures long-term temporal dependencies
    - More parameter efficient
    """

    inputs = layers.Input(shape=(sequence_length, n_features))

    # CNN layers for feature extraction
    # Use 1D convolutions along the time dimension
    conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2)(conv1)
    dropout1 = layers.Dropout(0.2)(pool1)

    conv2 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(dropout1)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling1D(pool_size=2)(conv2)
    dropout2 = layers.Dropout(0.2)(pool2)

    # LSTM layers for temporal dependencies
    lstm1 = layers.LSTM(64, return_sequences=True)(dropout2)
    lstm1 = layers.Dropout(0.3)(lstm1)
    lstm1 = layers.BatchNormalization()(lstm1)

    lstm2 = layers.LSTM(32, return_sequences=False)(lstm1)
    lstm2 = layers.Dropout(0.3)(lstm2)
    lstm2 = layers.BatchNormalization()(lstm2)

    # Dense layers for classification
    dense1 = layers.Dense(32, activation='relu')(lstm2)
    dense1 = layers.Dropout(0.2)(dense1)

    outputs = layers.Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_cnn_lstm(SEQUENCE_LENGTH, len(available_features))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("📊 Model Architecture:")
model.summary()
print()

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("=" * 80)
print("TRAINING CNN-LSTM HYBRID")
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
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

print("⏳ Training...")
print(f"   Epochs: 100 (with early stopping)")
print(f"   Batch size: 32")
print(f"   Validation split: 20%")
print()

start_time = datetime.now()

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
print()

# ============================================================================
# EVALUATE
# ============================================================================

print("=" * 80)
print("EVALUATION")
print("=" * 80)
print()

y_pred_proba = model.predict(X_test_seq, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

acc = accuracy_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred)
recall = recall_score(y_test_seq, y_pred)
f1 = f1_score(y_test_seq, y_pred)

print(f"🎯 CNN-LSTM Performance:")
print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print()

# Comparison with baseline LSTM
print("📊 Comparison with Baseline LSTM (67.81%):")
baseline_lstm_acc = 0.6781
improvement = (acc - baseline_lstm_acc) * 100

if improvement > 0:
    print(f"✅ IMPROVEMENT: +{improvement:.2f}pp")
else:
    print(f"❌ REGRESSION: {improvement:.2f}pp")
print()

# Save model
model_path = MODELS_DIR / "cnn_lstm.h5"
model.save(model_path)
print(f"💾 Saved model to: {model_path}")

scaler_path = MODELS_DIR / "cnn_lstm_scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"💾 Saved scaler to: {scaler_path}")
print()

# Save results
results = {
    'Model': 'CNN-LSTM',
    'Accuracy': acc,
    'F1_Score': f1,
    'Precision': precision,
    'Recall': recall,
    'Improvement_vs_LSTM': improvement
}

results_df = pd.DataFrame([results])
results_df.to_csv("cnn_lstm_results.csv", index=False)
print(f"💾 Saved results to: cnn_lstm_results.csv")
print()

print("=" * 80)
print("✅ CNN-LSTM HYBRID TRAINING COMPLETE")
print("=" * 80)
