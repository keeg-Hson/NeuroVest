# analyze_signals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import load_SPY_data, add_features
import warnings
warnings.filterwarnings("ignore", message=".*no_silent_downcasting.*")


def analyze_signals():
    print("Loading and processing SPY data...")
    df = load_SPY_data()
    df, feature_cols = add_features(df)

    print("Forward filling missing values...")
    df = df.infer_objects(copy=False)
    df.ffill(inplace=True)

    # Only keep valid numeric feature columns
    valid_feature_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # 1) Full feature-to-feature correlation (for heatmap)
    print("Computing feature-to-feature correlation (no Close)...")
    feat_corr = df[valid_feature_cols].corr(min_periods=30)

    feat_corr.to_csv("logs/signal_feature_corr_matrix.csv")

    plt.figure(figsize=(16, 12))
    sns.heatmap(feat_corr, cmap="coolwarm", annot=False, center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("logs/correlation_heatmap.png")
    print("Saved: logs/correlation_heatmap.png")
    plt.close()

    # 2) Correlation of features vs. NEXT-DAY RETURN 
    print("Computing correlation to next-day return...")
    next_ret = df["Close"].pct_change().shift(-1)

    # keep only rows where target is finite
    mask = np.isfinite(next_ret.values)
    next_ret = next_ret[mask]

    # build a safe feature dict with enough observations
    safe_feats = {}
    for c in valid_feature_cols:
        s = df[c]
        s = s[mask].replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() >= 30:  # require a minimum sample size
            safe_feats[c] = s

    if safe_feats:
        corr_series = pd.DataFrame(safe_feats).corrwith(next_ret, method="pearson")
        corr_series = corr_series.dropna().sort_values(ascending=False)
        corr_series.to_csv("logs/signal_corr_to_nextday_return.csv", header=["corr"])
        print("Saved: logs/signal_corr_to_nextday_return.csv")
    else:
        print("⚠️ Not enough valid data to compute correlations to next-day return.")


    # Example overlays (no emojis in titles)
    example_signals = ['RSI', 'MACD', 'MACD_Signal', 'VIX', 'News_Sentiment', 'Reddit_Sentiment']
    for signal in example_signals:
        if signal in df.columns:
            plt.figure(figsize=(14, 6))
            ax1 = df['Close'].plot(label='Close', linewidth=2)
            ax2 = df[signal].plot(secondary_y=True, label=signal, alpha=0.6)

            ax1.set_ylabel("Close Price")
            ax2.set_ylabel(f"{signal}")

            plt.title(f"Close vs {signal}")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"logs/close_vs_{signal}.png")
            print(f"Saved: logs/close_vs_{signal}.png")
            plt.close()


if __name__ == "__main__":
    analyze_signals()
