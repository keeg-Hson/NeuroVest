#!/usr/bin/env python3
"""
Simple Neural Meta-Learner - Uses existing model predictions

Loads pre-trained base models, generates predictions, then trains
a neural network to optimally combine them.

Expected: +0.5-1% over simple weighted average
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 80)
print("SIMPLE NEURAL META-LEARNER")
print("=" * 80)
print()

np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = Path("models")

# Load test predictions from final_model_comparison
# These are predictions on the same test set from each model

print("📥 Loading pre-computed predictions...")

# Load actual models and generate predictions on proper train/test split
# Load comparison results to identify available models
comparison_df = pd.read_csv("final_model_comparison.csv")
print(f"✅ Available models: {list(comparison_df['Model'])}")
print()

# Load actual predictions from the models
# Use train_transformer.py approach with stored predictions

# Create simpler approach:
# 1. Load lstm_model_comparison.csv which has test data predictions
# 2. Generate simple weighted average
# 3. Train neural meta-learner on those same predictions

print("📊 Loading LSTM model comparison data...")
lstm_comp = pd.read_csv("lstm_model_comparison.csv")
print(f"Models in comparison: {list(lstm_comp['Model'])}")
print()

# Check for saved predictions
pred_files = list(MODELS_DIR.glob("*_predictions.npy"))
print(f"Found {len(pred_files)} prediction files: {[f.name for f in pred_files]}")

if len(pred_files) == 0:
    print("⚠️  No prediction files found. Generating from models...")
    print("This requires loading the full data pipeline - postponing for now.")
    print()
    print("ALTERNATIVE APPROACH:")
    print("Since we have accuracy numbers for each model, we can create")
    print("synthetic predictions to demonstrate the meta-learner concept.")
    print()
    print("For production use, run base models first to generate predictions.")
    exit(0)

# Load available predictions
print("📦 Loading predictions...")
predictions = {}
for pred_file in pred_files:
    model_name = pred_file.stem.replace("_predictions", "")
    predictions[model_name] = np.load(pred_file)
    print(f"   {model_name}: {predictions[model_name].shape}")

# Load ground truth labels
y_true_file = MODELS_DIR / "test_labels.npy"
if not y_true_file.exists():
    print("⚠️  test_labels.npy not found. Cannot train meta-learner without ground truth.")
    exit(1)

y_true = np.load(y_true_file)
print(f"✅ Loaded {len(y_true)} test labels")
print()

# Create meta-features (base model predictions)
model_names = list(predictions.keys())
X_meta = np.column_stack([predictions[name] for name in model_names])
print(f"📊 Meta-features shape: {X_meta.shape}")
print(f"   {len(model_names)} base models x {len(y_true)} predictions")
print()

# Split into train/val for meta-learner
X_meta_train, X_meta_val, y_train, y_val = train_test_split(
    X_meta, y_true, test_size=0.3, random_state=42, stratify=y_true
)

print(f"Meta-learner splits:")
print(f"   Train: {len(X_meta_train)} samples")
print(f"   Val:   {len(X_meta_val)} samples")
print()

# Build meta-learner
print("🏗️  Building neural meta-learner...")
meta_model = keras.Sequential([
    layers.Input(shape=(len(model_names),)),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(8, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(1, activation='sigmoid')
])

meta_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("📊 Meta-learner architecture:")
meta_model.summary()
print()

# Train meta-learner
print("⏳ Training meta-learner...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = meta_model.fit(
    X_meta_train, y_train,
    validation_data=(X_meta_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

# Meta-learner predictions
meta_pred_proba = meta_model.predict(X_meta, verbose=0)
meta_pred = (meta_pred_proba > 0.5).astype(int).flatten()

meta_acc = accuracy_score(y_true, meta_pred)
meta_f1 = f1_score(y_true, meta_pred)

print(f"\n🎯 Neural Meta-Learner:")
print(f"   Accuracy:  {meta_acc:.4f} ({meta_acc*100:.2f}%)")
print(f"   F1 Score:  {meta_f1:.4f}")
print()

# Compare with simple average
simple_avg = X_meta.mean(axis=1)
simple_pred = (simple_avg > 0.5).astype(int)
simple_acc = accuracy_score(y_true, simple_pred)
simple_f1 = f1_score(y_true, simple_pred)

print(f"📊 Simple Average Baseline:")
print(f"   Accuracy:  {simple_acc:.4f} ({simple_acc*100:.2f}%)")
print(f"   F1 Score:  {simple_f1:.4f}")
print()

improvement = (meta_acc - simple_acc) * 100
print(f"💡 Improvement: {improvement:+.2f}pp")
print()

# Save meta-learner
meta_model_path = MODELS_DIR / "neural_meta_learner.h5"
meta_model.save(meta_model_path)
print(f"💾 Saved meta-learner to: {meta_model_path}")
print()

print("✅ NEURAL META-LEARNER TRAINING COMPLETE")
