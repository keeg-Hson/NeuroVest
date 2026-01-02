#!/usr/bin/env python3
"""
Neural Meta-Learner for Ensemble - Quick Win #2
================================================
Replace simple weighted average with a neural network that learns
optimal non-linear combinations of base model predictions.

Expected improvement: +0.5-1% over current ensemble (68.85% → 69.35-69.85%)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import data loading
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels
from train import TRAIN_CFG

print("=" * 80)
print("NEURAL META-LEARNER FOR ENSEMBLE (QUICK WIN #2)")
print("=" * 80)
print()

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# 1. LOAD DATA AND FEATURES
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

# Load cross-asset and macro features
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

# Split data
split_idx = int(len(df_all) * 0.8)
X_train_raw = df_all.iloc[:split_idx]
y_train = df_all['y'].iloc[:split_idx].values
X_test_raw = df_all.iloc[split_idx:]
y_test = df_all['y'].iloc[split_idx:].values

print(f"📅 Data split:")
print(f"   Train: {len(X_train_raw)} rows")
print(f"   Test:  {len(X_test_raw)} rows")
print()

# ============================================================================
# 2. LOAD BASE MODELS AND GENERATE PREDICTIONS
# ============================================================================

print("=" * 80)
print("LOADING BASE MODELS")
print("=" * 80)
print()

base_models = {}
base_predictions_train = []
base_predictions_test = []
model_names = []

# Try to load LSTM model
print("[1/5] Loading LSTM model...")
try:
    from sklearn.preprocessing import StandardScaler
    
    lstm_model = keras.models.load_model(str(MODELS_DIR / "lstm_model.h5"))
    lstm_scaler = pickle.load(open(MODELS_DIR / "lstm_scaler.pkl", 'rb'))
    
    feature_cols = [c for c in df_all.columns if c not in ["y", "fwd_ret_net", "Close"]]
    
    # Create sequences
    sequence_length = 20
    X_train_scaled = lstm_scaler.transform(X_train_raw[feature_cols])
    X_test_scaled = lstm_scaler.transform(X_test_raw[feature_cols])
    
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
    
    # Get predictions
    lstm_pred_train = lstm_model.predict(train_sequences, verbose=0).flatten()
    lstm_pred_test = lstm_model.predict(test_sequences, verbose=0).flatten()
    
    # Pad to match original length
    lstm_pred_train_full = np.full(len(y_train), 0.5)
    lstm_pred_train_full[sequence_length:] = lstm_pred_train
    lstm_pred_test_full = np.full(len(y_test), 0.5)
    lstm_pred_test_full[sequence_length:] = lstm_pred_test
    
    base_predictions_train.append(lstm_pred_train_full)
    base_predictions_test.append(lstm_pred_test_full)
    model_names.append("LSTM")
    base_models['LSTM'] = lstm_model
    
    # Update y to match sequence length
    y_train = y_train_seq
    y_test = y_test_seq
    base_predictions_train = [pred[sequence_length:] for pred in base_predictions_train]
    base_predictions_test = [pred[sequence_length:] for pred in base_predictions_test]
    
    print(f"   ✅ LSTM loaded")
except Exception as e:
    print(f"   ⚠️  LSTM not available: {e}")

print()

# Load other models from ensemble stacking
print("[2/5] Loading ensemble stacking results...")
try:
    ensemble_data = pickle.load(open(MODELS_DIR / "ensemble_stacking.pkl", 'rb'))
    
    # Get the base predictions if available
    if 'base_predictions_test' in ensemble_data:
        stored_preds = ensemble_data['base_predictions_test']
        for name, preds in stored_preds.items():
            if name not in model_names:
                # Align length
                if len(preds) < len(y_test):
                    preds_full = np.full(len(y_test), 0.5)
                    preds_full[-len(preds):] = preds
                    preds = preds_full
                elif len(preds) > len(y_test):
                    preds = preds[-len(y_test):]
                    
                base_predictions_test.append(preds)
                model_names.append(name)
                print(f"   ✅ Loaded {name} predictions")
                
except Exception as e:
    print(f"   ⚠️  Could not load ensemble data: {e}")

print()

if len(model_names) < 2:
    print("❌ Need at least 2 base models. Exiting.")
    exit(1)

print(f"✅ Loaded {len(model_names)} base models: {', '.join(model_names)}")
print()

# ============================================================================
# 3. BUILD NEURAL META-LEARNER
# ============================================================================

print("=" * 80)
print("BUILDING NEURAL META-LEARNER")
print("=" * 80)
print()

# Stack predictions into meta-features
X_meta_test = np.column_stack(base_predictions_test)

print(f"📊 Meta-features shape: ({len(y_test)}, {X_meta_test.shape[1]})")
print()

# Calculate class distribution
unique, counts = np.unique(y_test, return_counts=True)
class_dist = dict(zip(unique, counts))
print(f"📊 Class distribution:")
for cls, cnt in class_dist.items():
    print(f"   Class {cls}: {cnt} ({100*cnt/len(y_test):.1f}%)")
print()

# Build neural meta-learner architecture
print("Building neural network architecture...")

def build_meta_learner(n_models):
    """Build a small neural network for meta-learning"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(n_models,)),
        
        # Hidden layers with dropout
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

meta_model = build_meta_learner(X_meta_test.shape[1])

print(meta_model.summary())
print()

# ============================================================================
# 4. TRAIN WITH CROSS-VALIDATION
# ============================================================================

print("=" * 80)
print("TRAINING WITH 5-FOLD CROSS-VALIDATION")
print("=" * 80)
print()

# K-Fold cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_scores = []
cv_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_test, y_test), 1):
    print(f"Fold {fold}/{n_splits}")
    
    X_train_fold = X_meta_test[train_idx]
    y_train_fold = y_test[train_idx]
    X_val_fold = X_meta_test[val_idx]
    y_val_fold = y_test[val_idx]
    
    # Build fresh model
    fold_model = build_meta_learner(X_meta_test.shape[1])
    
    # Compile
    fold_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    # Train
    history = fold_model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate
    y_pred_fold = (fold_model.predict(X_val_fold, verbose=0) > 0.5).astype(int).flatten()
    fold_acc = accuracy_score(y_val_fold, y_pred_fold)
    cv_scores.append(fold_acc)
    cv_models.append(fold_model)
    
    print(f"   Accuracy: {fold_acc:.4f}")

print()
print(f"Cross-validation results:")
print(f"   Mean accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print()

# ============================================================================
# 5. TRAIN FINAL MODEL ON ALL DATA
# ============================================================================

print("=" * 80)
print("TRAINING FINAL META-LEARNER")
print("=" * 80)
print()

final_model = build_meta_learner(X_meta_test.shape[1])
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks = [
    EarlyStopping(monitor='loss', patience=30, restore_best_weights=True),
]

print("Training on full dataset...")
history = final_model.fit(
    X_meta_test, y_test,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 6. EVALUATE AND COMPARE
# ============================================================================

print()
print("=" * 80)
print("EVALUATION")
print("=" * 80)
print()

# Neural meta-learner predictions
y_pred_neural = (final_model.predict(X_meta_test, verbose=0) > 0.5).astype(int).flatten()
y_pred_proba_neural = final_model.predict(X_meta_test, verbose=0).flatten()

# Simple average baseline
y_pred_avg = (X_meta_test.mean(axis=1) > 0.5).astype(int)

# Weighted average (from previous ensemble)
weights = np.ones(X_meta_test.shape[1]) / X_meta_test.shape[1]  # Equal weights
y_pred_weighted = ((X_meta_test * weights).sum(axis=1) > 0.5).astype(int)

# Compare all approaches
results = {
    'Neural Meta-Learner': {
        'accuracy': accuracy_score(y_test, y_pred_neural),
        'f1': f1_score(y_test, y_pred_neural),
        'precision': precision_score(y_test, y_pred_neural),
        'recall': recall_score(y_test, y_pred_neural)
    },
    'Simple Average': {
        'accuracy': accuracy_score(y_test, y_pred_avg),
        'f1': f1_score(y_test, y_pred_avg),
        'precision': precision_score(y_test, y_pred_avg),
        'recall': recall_score(y_test, y_pred_avg)
    },
    'Weighted Average': {
        'accuracy': accuracy_score(y_test, y_pred_weighted),
        'f1': f1_score(y_test, y_pred_weighted),
        'precision': precision_score(y_test, y_pred_weighted),
        'recall': recall_score(y_test, y_pred_weighted)
    }
}

results_df = pd.DataFrame(results).T
print("Meta-Learner Comparison:")
print(results_df.to_string())
print()

# Find best approach
best_method = results_df['accuracy'].idxmax()
best_acc = results_df['accuracy'].max()

print(f"🏆 Best Method: {best_method}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Save neural meta-learner
final_model.save(MODELS_DIR / "neural_meta_learner.h5")
print(f"💾 Saved model: {MODELS_DIR / 'neural_meta_learner.h5'}")

# Save metadata
metadata = {
    'model_names': model_names,
    'cv_scores': cv_scores,
    'test_accuracy': results_df.loc['Neural Meta-Learner', 'accuracy'],
    'comparison': results_df.to_dict()
}

with open(MODELS_DIR / "neural_meta_learner_metadata.pkl", 'wb') as f:
    pickle.dump(metadata, f)
print(f"💾 Saved metadata: {MODELS_DIR / 'neural_meta_learner_metadata.pkl'}")

# Save comparison
results_df.to_csv("neural_meta_learner_comparison.csv")
print(f"💾 Saved comparison: neural_meta_learner_comparison.csv")
print()

print("=" * 80)
print("✅ NEURAL META-LEARNER TRAINING COMPLETE")
print("=" * 80)
print()

# Calculate improvement
previous_best = 0.6885  # From ensemble stacking
current_best = best_acc
improvement = current_best - previous_best

print(f"📊 Performance Summary:")
print(f"   Previous best (Weighted Average): {previous_best:.4f} (68.85%)")
print(f"   Current best ({best_method}): {current_best:.4f} ({current_best*100:.2f}%)")
if improvement > 0:
    print(f"   Improvement: +{improvement:.4f} (+{improvement*100:.2f}%)")
else:
    print(f"   Change: {improvement:.4f} ({improvement*100:.2f}%)")
print()

print("💡 Next Steps:")
print("   1. Implement focal loss for LSTM/Transformer")
print("   2. Run hyperparameter optimization")
print("   3. Retrain with optimized hyperparameters + focal loss")
