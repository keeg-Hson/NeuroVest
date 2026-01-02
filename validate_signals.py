#!/usr/bin/env python3
"""
Signal Validation & Quality Check

Validates crash/spike triggers, checks for false signals, and ensures
color coding matches signal types correctly.

Usage:
    python3 validate_signals.py
    python3 validate_signals.py --detailed
    python3 validate_signals.py --fix-colors
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def print_header(title):
    """Print styled header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def load_predictions():
    """Load prediction data"""
    pred_file = Path("logs/daily_predictions.csv")

    if not pred_file.exists():
        print(f"❌ Predictions not found: {pred_file}")
        print("   Run: python3 predict_multi_asset_ensemble.py")
        return None

    df = pd.read_csv(pred_file)
    df['Date'] = pd.to_datetime(df['Date'])

    return df


def validate_signal_distribution(df):
    """Check if signal distribution is reasonable"""
    print_header("SIGNAL DISTRIBUTION VALIDATION")

    pred_counts = df['Prediction'].value_counts()
    total = len(df)

    crash_pct = pred_counts.get(0, 0) / total * 100
    normal_pct = pred_counts.get(1, 0) / total * 100
    spike_pct = pred_counts.get(2, 0) / total * 100

    print(f"\nTotal predictions: {total}")
    print(f"\nSignal breakdown:")
    print(f"  CRASH (0):  {pred_counts.get(0, 0):5d} ({crash_pct:5.1f}%)")
    print(f"  NORMAL (1): {pred_counts.get(1, 0):5d} ({normal_pct:5.1f}%)")
    print(f"  SPIKE (2):  {pred_counts.get(2, 0):5d} ({spike_pct:5.1f}%)")

    # Validation checks
    issues = []

    # Check for balanced distribution (should be ~30/40/30)
    if crash_pct < 20 or crash_pct > 40:
        issues.append(f"⚠️  CRASH signals outside expected range (20-40%): {crash_pct:.1f}%")

    if normal_pct < 30 or normal_pct > 50:
        issues.append(f"⚠️  NORMAL signals outside expected range (30-50%): {normal_pct:.1f}%")

    if spike_pct < 20 or spike_pct > 40:
        issues.append(f"⚠️  SPIKE signals outside expected range (20-40%): {spike_pct:.1f}%")

    # Check for extreme imbalance
    if crash_pct > 60 or spike_pct > 60:
        issues.append(f"❌ SEVERE IMBALANCE: One signal dominates (>60%)")

    if normal_pct < 10:
        issues.append(f"❌ Too few NORMAL signals (<10%)")

    if issues:
        print("\n❌ Distribution Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ Signal distribution looks healthy")
        return True


def validate_confidence_values(df):
    """Check if confidence values are reasonable"""
    print_header("CONFIDENCE VALUE VALIDATION")

    if 'Confidence' not in df.columns:
        print("⚠️  No Confidence column found")
        return True

    conf = df['Confidence']

    print(f"\nConfidence statistics:")
    print(f"  Mean:   {conf.mean():.3f}")
    print(f"  Median: {conf.median():.3f}")
    print(f"  Min:    {conf.min():.3f}")
    print(f"  Max:    {conf.max():.3f}")
    print(f"  Std:    {conf.std():.3f}")

    issues = []

    # Check for invalid ranges
    if conf.min() < 0 or conf.max() > 1:
        issues.append(f"❌ Confidence outside valid range [0, 1]")

    # Check for constant values (red flag)
    if conf.std() < 0.01:
        issues.append(f"❌ Confidence values are nearly constant (std={conf.std():.4f})")

    # Check for reasonable distribution
    if conf.mean() < 0.2 or conf.mean() > 0.8:
        issues.append(f"⚠️  Confidence mean unusual: {conf.mean():.3f}")

    # Check SPIKE/CRASH confidence alignment
    if 'Spike_Conf' in df.columns and 'Crash_Conf' in df.columns:
        spike_signals = df[df['Prediction'] == 2]
        crash_signals = df[df['Prediction'] == 0]

        if len(spike_signals) > 0:
            avg_spike_conf = spike_signals['Spike_Conf'].mean()
            if avg_spike_conf < 0.5:
                issues.append(f"⚠️  SPIKE signals have low confidence: {avg_spike_conf:.3f}")

        if len(crash_signals) > 0:
            avg_crash_conf = crash_signals['Crash_Conf'].mean()
            if avg_crash_conf < 0.5:
                issues.append(f"⚠️  CRASH signals have low confidence: {avg_crash_conf:.3f}")

    if issues:
        print("\n❌ Confidence Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ Confidence values look reasonable")
        return True


def detect_false_signals(df):
    """Detect potential false signals"""
    print_header("FALSE SIGNAL DETECTION")

    if 'Close' not in df.columns:
        print("⚠️  No price data to validate against")
        return True

    # Calculate actual returns (forward 1-day)
    df['Actual_Return'] = df['Close'].pct_change().shift(-1)

    # Define what constitutes a "correct" signal
    # SPIKE should predict positive return
    # CRASH should predict negative return
    # NORMAL can be either

    spike_signals = df[df['Prediction'] == 2].copy()
    crash_signals = df[df['Prediction'] == 0].copy()

    print(f"\nAnalyzing {len(spike_signals)} SPIKE signals...")
    print(f"Analyzing {len(crash_signals)} CRASH signals...")

    false_spikes = 0
    false_crashes = 0

    if len(spike_signals) > 0:
        # SPIKE signals that had negative returns
        false_spike_mask = spike_signals['Actual_Return'] < -0.005  # < -0.5%
        false_spikes = false_spike_mask.sum()
        false_spike_rate = false_spikes / len(spike_signals) * 100

        print(f"\nSPIKE Signal Analysis:")
        print(f"  Total SPIKE signals: {len(spike_signals)}")
        print(f"  False signals: {false_spikes} ({false_spike_rate:.1f}%)")

        if false_spike_rate > 50:
            print(f"  ❌ High false positive rate for SPIKE signals")
        elif false_spike_rate > 40:
            print(f"  ⚠️  Moderate false positive rate for SPIKE signals")
        else:
            print(f"  ✅ Acceptable false positive rate")

    if len(crash_signals) > 0:
        # CRASH signals that had positive returns
        false_crash_mask = crash_signals['Actual_Return'] > 0.005  # > +0.5%
        false_crashes = false_crash_mask.sum()
        false_crash_rate = false_crashes / len(crash_signals) * 100

        print(f"\nCRASH Signal Analysis:")
        print(f"  Total CRASH signals: {len(crash_signals)}")
        print(f"  False signals: {false_crashes} ({false_crash_rate:.1f}%)")

        if false_crash_rate > 50:
            print(f"  ❌ High false positive rate for CRASH signals")
        elif false_crash_rate > 40:
            print(f"  ⚠️  Moderate false positive rate for CRASH signals")
        else:
            print(f"  ✅ Acceptable false positive rate")

    return false_spikes + false_crashes


def check_signal_consistency(df):
    """Check for signal consistency issues"""
    print_header("SIGNAL CONSISTENCY CHECK")

    issues = []

    # Check for rapid signal changes (whipsaws)
    signal_changes = (df['Prediction'].diff() != 0).sum()
    change_rate = signal_changes / len(df) * 100

    print(f"\nSignal stability:")
    print(f"  Total signal changes: {signal_changes}")
    print(f"  Change rate: {change_rate:.1f}%")

    if change_rate > 60:
        issues.append(f"⚠️  High signal volatility ({change_rate:.1f}% changes)")
        issues.append("   Signals may be too sensitive/noisy")

    # Check for signal streaks
    df['Signal_Streak'] = (df['Prediction'] != df['Prediction'].shift()).cumsum()
    streak_lengths = df.groupby('Signal_Streak').size()

    max_streak = streak_lengths.max()
    avg_streak = streak_lengths.mean()

    print(f"\nSignal streaks:")
    print(f"  Longest streak: {max_streak} days")
    print(f"  Average streak: {avg_streak:.1f} days")

    if max_streak > 100:
        issues.append(f"⚠️  Very long signal streak ({max_streak} days)")
        issues.append("   Model may be stuck in one prediction")

    if issues:
        print("\n⚠️  Consistency Issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ Signal consistency looks good")
        return True


def check_color_coding():
    """Check if color coding in dashboard matches signals"""
    print_header("COLOR CODING VALIDATION")

    print("\nExpected color scheme:")
    print("  CRASH (0):  🔴 Red    (bearish, danger)")
    print("  NORMAL (1): 🟡 Yellow (neutral, caution)")
    print("  SPIKE (2):  🟢 Green  (bullish, success)")

    # Check dashboard.py for color assignments
    dashboard_file = Path("dashboard.py")

    if not dashboard_file.exists():
        print("\n⚠️  dashboard.py not found, skipping color check")
        return True

    content = dashboard_file.read_text()

    issues = []

    # Look for common color assignment patterns
    if 'color' in content.lower() or 'background' in content.lower():
        print("\n✅ Color definitions found in dashboard.py")

        # Check for potential mismatches
        if '== 2' in content and 'red' in content.lower():
            issues.append("⚠️  Possible mismatch: SPIKE (2) may be colored red")

        if '== 0' in content and 'green' in content.lower():
            issues.append("⚠️  Possible mismatch: CRASH (0) may be colored green")

    if issues:
        print("\n⚠️  Potential color mismatches:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   Review dashboard.py signal color assignments")
        return False
    else:
        print("✅ No obvious color mismatches detected")
        return True


def run_comprehensive_validation(detailed=False):
    """Run all validation checks"""
    print_header("COMPREHENSIVE SIGNAL VALIDATION")

    # Load data
    df = load_predictions()
    if df is None:
        return False

    results = {
        'distribution': validate_signal_distribution(df),
        'confidence': validate_confidence_values(df),
        'consistency': check_signal_consistency(df),
        'colors': check_color_coding()
    }

    # Detect false signals if detailed
    if detailed:
        false_count = detect_false_signals(df)
        results['false_signals'] = false_count < (len(df) * 0.4)  # <40% false rate

    # Summary
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")
    print()

    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test.upper():20s}: {status}")

    if passed == total:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
        print("\nSignals appear to be working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} VALIDATION CHECKS FAILED")
        print("\nRecommendations:")

        if not results['distribution']:
            print("  • Re-run predictions with correct thresholds")
            print("  • Check percentile-based threshold calculation")

        if not results['confidence']:
            print("  • Review confidence calculation in predict_multi_asset_ensemble.py")
            print("  • Ensure confidence scales with signal strength")

        if not results['consistency']:
            print("  • Consider smoothing signals (e.g., 3-day consensus)")
            print("  • Adjust prediction thresholds to reduce noise")

        if not results.get('false_signals', True):
            print("  • Review feature engineering for predictive power")
            print("  • Consider retraining models with better data")

        if not results['colors']:
            print("  • Fix color assignments in dashboard.py")
            print("  • Ensure CRASH=red, NORMAL=yellow, SPIKE=green")

        return False


def main():
    parser = argparse.ArgumentParser(description="Signal Validation & Quality Check")
    parser.add_argument("--detailed", action="store_true",
                        help="Run detailed analysis including false signal detection")
    parser.add_argument("--fix-colors", action="store_true",
                        help="Suggest color coding fixes")

    args = parser.parse_args()

    success = run_comprehensive_validation(detailed=args.detailed)

    if args.fix_colors and not check_color_coding():
        print("\n" + "="*70)
        print("SUGGESTED COLOR FIXES")
        print("="*70)
        print("""
In dashboard.py, ensure color mapping:

def get_signal_color(prediction):
    if prediction == 0:
        return 'red'      # CRASH
    elif prediction == 1:
        return 'yellow'   # NORMAL
    else:  # prediction == 2
        return 'green'    # SPIKE

# For Plotly charts
color_map = {0: 'red', 1: 'yellow', 2: 'green'}

# For Streamlit
if prediction == 2:
    st.success("SPIKE - Bullish")  # Green
elif prediction == 0:
    st.error("CRASH - Bearish")    # Red
else:
    st.warning("NORMAL - Neutral") # Yellow
        """)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
