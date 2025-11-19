#!/usr/bin/env python3
"""
Detailed analysis of walk-forward CV behavior
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/NeuroVest')

from train import PurgedWalkForwardSplit

print("=" * 80)
print("DETAILED WALK-FORWARD CV ANALYSIS")
print("=" * 80)

n_samples = 5201

# Current implementation
cv_current = PurgedWalkForwardSplit(
    n_splits=5,
    min_train_size=1040,
    test_size=832,
    embargo=10
)

print(f"\nParameters:")
print(f"  Total samples: {n_samples}")
print(f"  Requested splits: {cv_current.n_splits}")
print(f"  Min train size: {cv_current.min_train_size}")
print(f"  Test size: {cv_current.test_size}")
print(f"  Embargo: {cv_current.embargo}")

print(f"\nCurrent implementation:")
X_dummy = np.arange(n_samples)
splits = list(cv_current.split(X_dummy))

print(f"  Actual splits generated: {len(splits)}/5")
print(f"  Lost splits: {5 - len(splits)}")
print()

total_train_samples = 0
total_test_samples = 0

for i, (train_idx, test_idx) in enumerate(splits, 1):
    train_range = f"[{train_idx[0]}:{train_idx[-1]+1}]"
    test_range = f"[{test_idx[0]}:{test_idx[-1]+1}]"
    embargo_start = train_idx[-1] + 1
    embargo_end = test_idx[0]
    embargo_gap = embargo_end - embargo_start

    print(f"  Split {i}:")
    print(f"    Train: {train_range} = {len(train_idx)} samples")
    print(f"    Embargo gap: [{embargo_start}:{embargo_end}] = {embargo_gap} samples")
    print(f"    Test: {test_range} = {len(test_idx)} samples")

    total_train_samples += len(train_idx)
    total_test_samples += len(test_idx)

print(f"\n  Total unique train samples: {total_train_samples} (with overlap)")
print(f"  Total test samples: {total_test_samples}")
print(f"  Unused samples: {n_samples - splits[-1][1][-1] - 1}")

# Show what SHOULD happen with fixed implementation
print(f"\n" + "=" * 80)
print("PROPOSED FIX:")
print("=" * 80)

print(f"\nFix: start_test = min_train_size + embargo instead of min_train_size")
print(f"     This ensures first split has exactly min_train_size training samples\n")

# Manually simulate the fixed behavior
start_test = cv_current.min_train_size + cv_current.embargo  # FIX HERE
test_size = cv_current.test_size
embargo = cv_current.embargo
min_train = cv_current.min_train_size

print(f"Fixed implementation simulation:")
fixed_splits = []

for split_num in range(cv_current.n_splits):
    test_start = start_test
    test_end = min(n_samples, test_start + test_size)

    if test_end - test_start < 1:
        break

    train_end = max(0, test_start - embargo)
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, test_end)

    if len(train_idx) >= min_train:
        fixed_splits.append((train_idx, test_idx))
        train_range = f"[{train_idx[0]}:{train_idx[-1]+1}]"
        test_range = f"[{test_idx[0]}:{test_idx[-1]+1}]"
        embargo_start = train_idx[-1] + 1
        embargo_end = test_idx[0]
        embargo_gap = embargo_end - embargo_start

        print(f"  Split {len(fixed_splits)}:")
        print(f"    Train: {train_range} = {len(train_idx)} samples")
        print(f"    Embargo gap: [{embargo_start}:{embargo_end}] = {embargo_gap} samples")
        print(f"    Test: {test_range} = {len(test_idx)} samples")

    start_test = test_end

    if len(fixed_splits) >= cv_current.n_splits:
        break

print(f"\n  Fixed splits generated: {len(fixed_splits)}/5")
print(f"  Improvement: +{len(fixed_splits) - len(splits)} split(s)")

if len(fixed_splits) > len(splits):
    print(f"\n✅ FIX VALIDATED: Generates all {len(fixed_splits)} requested splits")
else:
    print(f"\n⚠️  FIX INCOMPLETE: Still only {len(fixed_splits)} splits")

print("\n" + "=" * 80)
