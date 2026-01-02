#!/usr/bin/env python3
"""
Test script to verify walk-forward CV bug and fix

This verifies that PurgedWalkForwardSplit is broken and generates 0 splits.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/NeuroVest')

from train import PurgedWalkForwardSplit, _build_adaptive_cv, _n_splits

print("=" * 80)
print("TESTING WALK-FORWARD CV BUG")
print("=" * 80)

# Test with realistic data size (5201 samples like SPY)
n_samples = 5201

print(f"\nTest data size: {n_samples} samples")

# Test current implementation
print("\n1. Testing _build_adaptive_cv() output:")
cv = _build_adaptive_cv(n_samples)
print(f"   CV object: {cv}")
print(f"   n_splits requested: {cv.n_splits}")
print(f"   min_train_size: {cv.min_train_size}")
print(f"   test_size: {cv.test_size}")
print(f"   embargo: {cv.embargo}")

# Test how many splits it actually generates
print("\n2. Testing actual split generation:")
X_dummy = np.arange(n_samples)
splits = list(cv.split(X_dummy))
print(f"   Actual splits generated: {len(splits)}")

if len(splits) == 0:
    print("   ❌ BUG CONFIRMED: Zero splits generated!")
else:
    print(f"   ✓ {len(splits)} splits generated")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"   Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")

# Test _n_splits helper
print("\n3. Testing _n_splits() helper:")
n = _n_splits(cv, n_samples)
print(f"   _n_splits() returned: {n}")

# Diagnose why it's failing
print("\n4. Diagnosis:")
print(f"   start_test initialized to: {cv.min_train_size}")
print(f"   embargo: {cv.embargo}")
print(f"   train_end would be: max(0, {cv.min_train_size} - {cv.embargo}) = {max(0, cv.min_train_size - cv.embargo)}")
print(f"   Check: train_end >= min_train_size?")
print(f"         {max(0, cv.min_train_size - cv.embargo)} >= {cv.min_train_size}? = {max(0, cv.min_train_size - cv.embargo) >= cv.min_train_size}")

if max(0, cv.min_train_size - cv.embargo) < cv.min_train_size:
    print("   ❌ FAILS: train_end < min_train_size, so no splits are yielded")
    print("\n5. Root cause:")
    print("   - start_test = min_train_size")
    print("   - train_end = start_test - embargo")
    print("   - This makes train_end < min_train_size")
    print("\n6. Fix:")
    print("   - start_test should be: min_train_size + embargo")
    print("   - Then train_end = (min_train_size + embargo) - embargo = min_train_size ✓")

print("\n" + "=" * 80)
