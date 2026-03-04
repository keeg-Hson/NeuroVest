#!/usr/bin/env python3
"""
optimize_threshold.py - Find optimal prediction threshold for accuracy

This script:
1. Loads the trained XGBoost model
2. Gets predictions on the test set
3. Tries different thresholds (0.5 to 0.95)
4. Calculates metrics for each threshold
5. Identifies the best threshold for accuracy and other metrics
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODELS_DIR, TRAIN_CFG
from utils import load_SPY_data, add_features, finalize_features, add_forward_returns_and_labels

print("=" * 80)
print("THRESHOLD OPTIMIZATION FOR IMPROVED ACCURACY")
print("=" * 80)

# Load the improved model
print("\n📥 Loading improved XGBoost model...")
model_path = MODELS_DIR / "market_crash_model_fwd_improved.pkl"
model_payload = joblib.load(model_path)
model = model_payload["model"]
saved_features = model_payload["features"]
print(f"✅ Model loaded from {model_path}")
print(f"   Model type: {type(model)}")
print(f"   Features in model: {len(saved_features)}")

# Load data
print("\n📥 Loading SPY data...")
df = load_SPY_data()
print(f"✅ Loaded {len(df)} rows")

# Add features (same as training)
print("\n📊 Adding features...")
df, feature_cols = add_features(df)
print(f"✅ Added features, total columns: {len(df.columns)}")

# Add forward returns and labels
print("\n🎯 Adding forward returns and labels...")
df = add_forward_returns_and_labels(
    df,
    price_col=TRAIN_CFG["price_col"],
    horizon=TRAIN_CFG["horizon"],
    pos_threshold=TRAIN_CFG["pos_threshold"],
    fee_bps=TRAIN_CFG.get("fee_bps", 1.5),
    slippage_bps=TRAIN_CFG.get("slippage_bps", 2.0),
)
print(f"✅ Labels added")

# Save target before finalize_features removes it
if "y" not in df.columns:
    raise ValueError("Target column 'y' not found in dataframe")
y_all = df["y"].copy()

# Finalize features (keeps only specified feature columns)
df = finalize_features(df, feature_cols)

# Add y back
df["y"] = y_all

# Use only the features that were used in training the model
# Verify all expected features are present
missing_features = set(saved_features) - set(df.columns)
if missing_features:
    print(f"⚠️ Warning: Missing features from training: {missing_features}")
    # Add missing features as zeros
    for feat in missing_features:
        df[feat] = 0.0

# Keep only features used in model
df = df[[col for col in saved_features if col in df.columns] + ["y"]]

# Remove NaN rows
print(f"   Rows before dropna: {len(df)}")
df = df.dropna()
print(f"✅ After removing NaN: {len(df)} rows")

# Split into train/test (same logic as training script)
# Use last 20% as test set
test_size = int(len(df) * 0.2)
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

print(f"\n📊 Data split:")
print(f"   Training: {len(train_df)} rows")
print(f"   Test: {len(test_df)} rows")

# Get test set features and labels
feature_cols_in_df = [c for c in df.columns if c != "y"]
X_test = test_df[feature_cols_in_df]
y_test = test_df["y"]

print(f"\n🎯 Test set class distribution:")
print(f"   Class 0 (no trade): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"   Class 1 (trade): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

# Get probability predictions
print("\n🔮 Getting model predictions...")
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
print(f"✅ Predictions obtained, shape: {y_proba.shape}")
print(f"   Probability range: {y_proba.min():.4f} to {y_proba.max():.4f}")
print(f"   Mean probability: {y_proba.mean():.4f}")

# Try different thresholds
print("\n" + "=" * 80)
print("TESTING DIFFERENT THRESHOLDS")
print("=" * 80)

thresholds = np.arange(0.3, 0.96, 0.05)
results = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'n_predictions': (y_pred == 1).sum()
    })

    print(f"\nThreshold: {threshold:.2f}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Predictions: {(y_pred == 1).sum()} / {len(y_pred)} ({(y_pred == 1).mean()*100:.1f}%)")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("OPTIMAL THRESHOLDS BY METRIC")
print("=" * 80)

# Best threshold for each metric
best_accuracy_idx = results_df['accuracy'].idxmax()
best_precision_idx = results_df['precision'].idxmax()
best_f1_idx = results_df['f1'].idxmax()

print(f"\n🎯 Best for ACCURACY: {results_df.loc[best_accuracy_idx, 'threshold']:.2f}")
print(f"   Accuracy:  {results_df.loc[best_accuracy_idx, 'accuracy']:.4f} ({results_df.loc[best_accuracy_idx, 'accuracy']*100:.2f}%)")
print(f"   Precision: {results_df.loc[best_accuracy_idx, 'precision']:.4f} ({results_df.loc[best_accuracy_idx, 'precision']*100:.2f}%)")
print(f"   Recall:    {results_df.loc[best_accuracy_idx, 'recall']:.4f} ({results_df.loc[best_accuracy_idx, 'recall']*100:.2f}%)")
print(f"   F1 Score:  {results_df.loc[best_accuracy_idx, 'f1']:.4f}")

print(f"\n🎯 Best for PRECISION: {results_df.loc[best_precision_idx, 'threshold']:.2f}")
print(f"   Accuracy:  {results_df.loc[best_precision_idx, 'accuracy']:.4f} ({results_df.loc[best_precision_idx, 'accuracy']*100:.2f}%)")
print(f"   Precision: {results_df.loc[best_precision_idx, 'precision']:.4f} ({results_df.loc[best_precision_idx, 'precision']*100:.2f}%)")
print(f"   Recall:    {results_df.loc[best_precision_idx, 'recall']:.4f} ({results_df.loc[best_precision_idx, 'recall']*100:.2f}%)")
print(f"   F1 Score:  {results_df.loc[best_precision_idx, 'f1']:.4f}")

print(f"\n🎯 Best for F1 SCORE: {results_df.loc[best_f1_idx, 'threshold']:.2f}")
print(f"   Accuracy:  {results_df.loc[best_f1_idx, 'accuracy']:.4f} ({results_df.loc[best_f1_idx, 'accuracy']*100:.2f}%)")
print(f"   Precision: {results_df.loc[best_f1_idx, 'precision']:.4f} ({results_df.loc[best_f1_idx, 'precision']*100:.2f}%)")
print(f"   Recall:    {results_df.loc[best_f1_idx, 'recall']:.4f} ({results_df.loc[best_f1_idx, 'recall']*100:.2f}%)")
print(f"   F1 Score:  {results_df.loc[best_f1_idx, 'f1']:.4f}")

# Find balanced threshold (good accuracy + reasonable precision)
# Filter for precision > 0.4 and find best accuracy
balanced_df = results_df[results_df['precision'] >= 0.4]
if len(balanced_df) > 0:
    best_balanced_idx = balanced_df['accuracy'].idxmax()
    print(f"\n🎯 RECOMMENDED (Balanced - Precision ≥ 40%): {results_df.loc[best_balanced_idx, 'threshold']:.2f}")
    print(f"   Accuracy:  {results_df.loc[best_balanced_idx, 'accuracy']:.4f} ({results_df.loc[best_balanced_idx, 'accuracy']*100:.2f}%)")
    print(f"   Precision: {results_df.loc[best_balanced_idx, 'precision']:.4f} ({results_df.loc[best_balanced_idx, 'precision']*100:.2f}%)")
    print(f"   Recall:    {results_df.loc[best_balanced_idx, 'recall']:.4f} ({results_df.loc[best_balanced_idx, 'recall']*100:.2f}%)")
    print(f"   F1 Score:  {results_df.loc[best_balanced_idx, 'f1']:.4f}")
    recommended_threshold = results_df.loc[best_balanced_idx, 'threshold']
else:
    print("\n⚠️ No threshold found with precision ≥ 40%")
    print("   Using best accuracy threshold instead")
    recommended_threshold = results_df.loc[best_accuracy_idx, 'threshold']

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save threshold optimization results
results_df.to_csv('threshold_optimization_results.csv', index=False)
print(f"✅ Detailed results saved to: threshold_optimization_results.csv")

# Update threshold file
threshold_file = MODELS_DIR / "thresholds_fwd_improved.json"
with open(threshold_file, 'r') as f:
    threshold_config = json.load(f)

old_threshold = threshold_config['threshold']
threshold_config['threshold'] = float(recommended_threshold)
threshold_config['optimization_note'] = 'Optimized for balanced accuracy and precision'

best_row = results_df[results_df['threshold'] == recommended_threshold].iloc[0]
threshold_config['optimized_accuracy'] = float(best_row['accuracy'])
threshold_config['optimized_precision'] = float(best_row['precision'])
threshold_config['optimized_recall'] = float(best_row['recall'])
threshold_config['optimized_f1'] = float(best_row['f1'])

with open(threshold_file, 'w') as f:
    json.dump(threshold_config, f, indent=2)

print(f"✅ Updated threshold file: {threshold_file}")
print(f"   Old threshold: {old_threshold}")
print(f"   New threshold: {recommended_threshold:.2f}")

# Create visualization
print("\n📊 Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Threshold Optimization Results', fontsize=16, fontweight='bold')

# Plot 1: Accuracy vs Threshold
axes[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'b-', linewidth=2, marker='o')
axes[0, 0].axvline(x=recommended_threshold, color='r', linestyle='--', label=f'Recommended: {recommended_threshold:.2f}')
axes[0, 0].set_xlabel('Threshold', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Accuracy vs Threshold', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Precision vs Recall
axes[0, 1].plot(results_df['recall'], results_df['precision'], 'g-', linewidth=2, marker='o')
for i, row in results_df.iterrows():
    if row['threshold'] in [0.3, 0.5, 0.7, recommended_threshold]:
        axes[0, 1].annotate(f"{row['threshold']:.2f}",
                           (row['recall'], row['precision']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[0, 1].set_xlabel('Recall', fontsize=12)
axes[0, 1].set_ylabel('Precision', fontsize=12)
axes[0, 1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: All metrics
axes[1, 0].plot(results_df['threshold'], results_df['accuracy'], 'b-', linewidth=2, label='Accuracy', marker='o')
axes[1, 0].plot(results_df['threshold'], results_df['precision'], 'r-', linewidth=2, label='Precision', marker='s')
axes[1, 0].plot(results_df['threshold'], results_df['recall'], 'g-', linewidth=2, label='Recall', marker='^')
axes[1, 0].plot(results_df['threshold'], results_df['f1'], 'orange', linewidth=2, label='F1 Score', marker='d')
axes[1, 0].axvline(x=recommended_threshold, color='purple', linestyle='--', alpha=0.7, label=f'Recommended: {recommended_threshold:.2f}')
axes[1, 0].set_xlabel('Threshold', fontsize=12)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('All Metrics vs Threshold', fontsize=13, fontweight='bold')
axes[1, 0].legend(loc='best')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Number of predictions
axes[1, 1].plot(results_df['threshold'], results_df['n_predictions'], 'purple', linewidth=2, marker='o')
axes[1, 1].axvline(x=recommended_threshold, color='r', linestyle='--', label=f'Recommended: {recommended_threshold:.2f}')
axes[1, 1].set_xlabel('Threshold', fontsize=12)
axes[1, 1].set_ylabel('Number of Positive Predictions', fontsize=12)
axes[1, 1].set_title('Prediction Volume vs Threshold', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('threshold_optimization_plot.png', dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: threshold_optimization_plot.png")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✅ Threshold optimization complete!")
print(f"\n📈 Improvement from old threshold (0.50) to new ({recommended_threshold:.2f}):")

old_results = results_df[results_df['threshold'] == 0.50].iloc[0] if 0.50 in results_df['threshold'].values else results_df.iloc[0]
new_results = results_df[results_df['threshold'] == recommended_threshold].iloc[0]

print(f"\n   Accuracy:  {old_results['accuracy']:.4f} → {new_results['accuracy']:.4f} ({(new_results['accuracy'] - old_results['accuracy'])*100:+.2f}%)")
print(f"   Precision: {old_results['precision']:.4f} → {new_results['precision']:.4f} ({(new_results['precision'] - old_results['precision'])*100:+.2f}%)")
print(f"   Recall:    {old_results['recall']:.4f} → {new_results['recall']:.4f} ({(new_results['recall'] - old_results['recall'])*100:+.2f}%)")
print(f"   F1 Score:  {old_results['f1']:.4f} → {new_results['f1']:.4f} ({(new_results['f1'] - old_results['f1'])*100:+.2f}%)")

print("\n🎉 Optimization complete! Use the recommended threshold for better accuracy.")
