# analyze_signals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_SPY_data, add_features

def analyze_signals():
    print("📈 Loading and processing SPY data...")
    df = load_SPY_data()
    df, feature_cols = add_features(df)

    print("🧹 Forward filling missing values...")
    df.ffill(inplace=True)


    print("📊 Computing feature correlation matrix...")
    valid_feature_cols = [col for col in feature_cols if col in df.columns]
    corr_matrix = df[valid_feature_cols + ['Close']].corr()


    # Save correlation matrix as CSV
    corr_matrix.to_csv("logs/signal_correlations.csv")
    print("✅ Correlation matrix saved to logs/signal_correlations.csv")

    # Visualize full correlation heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, center=0)
    plt.title("Feature Correlation Heatmap (incl. Close Price)")
    plt.tight_layout()
    plt.savefig("logs/correlation_heatmap.png")
    print("📸 Heatmap saved to logs/correlation_heatmap.png")
    plt.close()

    # Plot example signals over Close price
    example_signals = ['RSI', 'MACD', 'MACD_Signal', 'VIX', 'News_Sentiment', 'Reddit_Sentiment']

    for signal in example_signals:
        if signal in df.columns:
            plt.figure(figsize=(14, 6))
            ax1 = df['Close'].plot(label='Close', color='black', linewidth=2)
            ax2 = df[signal].plot(secondary_y=True, label=signal, color='blue', alpha=0.6)

            ax1.set_ylabel("Close Price")
            ax2.set_ylabel(f"{signal} Value")

            plt.title(f"📊 {signal} vs Close Price")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"logs/close_vs_{signal}.png")
            print(f"📸 Plot saved: logs/close_vs_{signal}.png")
            plt.close()

if __name__ == "__main__":
    analyze_signals()
