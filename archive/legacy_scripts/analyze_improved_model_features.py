#!/usr/bin/env python3
"""Analyze feature importance for the improved XGBoost model"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the improved model
print("Loading improved XGBoost model...")
model_path = Path("models/market_crash_model_fwd_improved.pkl")
payload = joblib.load(model_path)
model = payload["model"]
features = payload["features"]

print(f"Model type: {type(model)}")
print(f"Number of features: {len(features)}")
print(f"\nFeatures used in model:")
for i, f in enumerate(features, 1):
    print(f"  {i}. {f}")

# Extract the base estimator from the calibrated classifier
if hasattr(model, 'calibrated_classifiers_'):
    print("\nModel is calibrated. Extracting base estimator...")
    base_model = model.calibrated_classifiers_[0].estimator

    # Get the pipeline
    if hasattr(base_model, 'named_steps'):
        print(f"Pipeline steps: {list(base_model.named_steps.keys())}")

        # Get the actual classifier
        clf = base_model.named_steps['clf']
        print(f"Classifier type: {type(clf)}")

        # Get feature importances from XGBoost
        if hasattr(clf, 'feature_importances_'):
            # Get selected features after SelectKBest
            if 'kbest' in base_model.named_steps:
                kbest = base_model.named_steps['kbest']
                selected_mask = kbest.get_support()
                selected_features = [f for f, selected in zip(features, selected_mask) if selected]
                print(f"\nFeatures selected by SelectKBest: {len(selected_features)} out of {len(features)}")
            else:
                selected_features = features

            importances = clf.feature_importances_

            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': selected_features[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("\n" + "="*80)
            print("TOP 20 FEATURES BY IMPORTANCE (XGBoost)")
            print("="*80)
            print(importance_df.head(20).to_string(index=False))

            # Save to CSV
            importance_df.to_csv('improved_model_feature_importance.csv', index=False)
            print(f"\n✅ Full importance saved to: improved_model_feature_importance.csv")

            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 10))
            top_20 = importance_df.head(20)
            ax.barh(range(len(top_20)), top_20['importance'])
            ax.set_yticks(range(len(top_20)))
            ax.set_yticklabels(top_20['feature'])
            ax.set_xlabel('Importance (Gain)')
            ax.set_title('Top 20 Features - Improved XGBoost Model', fontweight='bold', fontsize=14)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig('improved_model_feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: improved_model_feature_importance.png")

            # Analyze new engineered features
            print("\n" + "="*80)
            print("ANALYSIS OF NEW ENGINEERED FEATURES")
            print("="*80)

            new_feature_patterns = [
                'BB_Width_x_',  # Feature interactions
                'BB_Width_Lag',  # BB Width lags
                'RSI_Lag7', 'RSI_Lag10',  # New RSI lags
                'Return_Lag7', 'Return_Lag10', 'Return_Lag15',  # New return lags
                'Vol_Expanding', 'Vol_Percentile', 'Volatility_Acceleration',  # Volatility features
                'Return_Trend_Strength',  # Trend features
                'OBV_x_'  # OBV interactions
            ]

            new_features = importance_df[importance_df['feature'].str.contains('|'.join(new_feature_patterns), case=False, regex=True)]

            if len(new_features) > 0:
                print(f"\nFound {len(new_features)} of the 47 new engineered features in model:")
                print(new_features.to_string(index=False))

                # Calculate percentage of total importance
                total_importance = importance_df['importance'].sum()
                new_features_importance = new_features['importance'].sum()
                pct = (new_features_importance / total_importance) * 100

                print(f"\n📊 New features contribute {pct:.2f}% of total model importance")

                if pct > 20:
                    print("✅ New features are HIGHLY valuable!")
                elif pct > 10:
                    print("✅ New features are moderately valuable")
                else:
                    print("⚠️ New features have limited impact")
            else:
                print("\n⚠️ No new engineered features found in selected features")
                print("   This means SelectKBest filtered them out")

            # Check what features are being used
            print("\n" + "="*80)
            print("FEATURE CATEGORIES")
            print("="*80)

            categories = {
                'Original Core': ['MA_20', 'EMA_12', 'EMA_26', 'MACD', 'BB_Width', 'RSI', 'OBV'],
                'Return Lags': [f for f in selected_features if 'Return_Lag' in f],
                'RSI Lags': [f for f in selected_features if 'RSI_Lag' in f],
                'Volatility': [f for f in selected_features if 'Vol' in f or 'Volatility' in f],
                'Interactions': [f for f in selected_features if '_x_' in f],
                'Trend': [f for f in selected_features if 'Trend' in f or 'Momentum' in f],
            }

            for cat, feats in categories.items():
                matching = [f for f in feats if f in importance_df['feature'].values]
                if matching:
                    cat_importance = importance_df[importance_df['feature'].isin(matching)]['importance'].sum()
                    print(f"\n{cat}: {len(matching)} features")
                    print(f"  Total importance: {cat_importance:.4f}")
                    print(f"  Top 3: {', '.join(matching[:3])}")
        else:
            print("No feature importances available for this model")
else:
    print("Model is not calibrated")

print("\n🎉 Analysis complete!")
