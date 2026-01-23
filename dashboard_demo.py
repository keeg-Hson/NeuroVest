#!/usr/bin/env python3
"""
NeuroVest Quick Demo Dashboard

Simple Streamlit dashboard for quick testing and debugging.
Designed for developers and internal testing.

Usage:
    streamlit run dashboard_demo.py
    streamlit run dashboard_demo.py --server.port 8502

Features:
- Quick prediction overview
- Model status check
- Data health check
- Recent signals
- Backtest results summary
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install streamlit plotly pandas")
    sys.exit(1)

# Configure page
st.set_page_config(
    page_title="NeuroVest Quick Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Directories
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")

# Hero section
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">⚡ NeuroVest Quick Demo</h1>
    <p style="font-size: 1.3rem; color: #666;">Fast Testing & Debugging Interface</p>
</div>
""", unsafe_allow_html=True)

# Value proposition - compact version
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; margin: 1rem 0;">
    <p style="font-size: 1.1rem; line-height: 1.5; margin-bottom: 0;">
        <b>Market Forecasting API</b> using ensemble ML (XGBoost + LightGBM + CatBoost) for 3-class predictions (CRASH/NORMAL/SPIKE)
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs for organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🤖 Models", "📈 Predictions", "💰 Backtest", "📁 Data"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.markdown("### 🔧 System Status")

    # Quick status cards with better colors
    col1, col2, col3, col4 = st.columns(4)

    # Check data files
    spy_exists = (DATA_DIR / "SPY.csv").exists()
    pred_exists = (LOGS_DIR / "labeled_predictions.csv").exists()
    backtest_exists = (LOGS_DIR / "backtest_results.csv").exists()

    # Count models
    model_count = 0
    for model_file in ['xgboost_multi_asset.pkl', 'lightgbm_multi_asset.pkl', 'catboost_multi_asset.pkl']:
        if (MODELS_DIR / model_file).exists():
            model_count += 1

    with col1:
        if spy_exists:
            st.success("🟢 SPY Data Ready")
        else:
            st.error("🔴 SPY Data Missing")

    with col2:
        if model_count == 3:
            st.success(f"🟢 All {model_count} Models")
        elif model_count > 0:
            st.warning(f"🟡 {model_count}/3 Models")
        else:
            st.error(f"🔴 No Models")

    with col3:
        if pred_exists:
            st.success("🟢 Predictions Ready")
        else:
            st.error("🔴 No Predictions")

    with col4:
        if backtest_exists:
            st.success("🟢 Backtest Complete")
        else:
            st.warning("🟡 No Backtest")

    st.markdown("---")

    # System health
    st.subheader("🏥 System Health")

    health_issues = []

    if not spy_exists:
        health_issues.append("SPY.csv missing - Run: `python3 update_spy_data.py`")

    if model_count == 0:
        health_issues.append("No models found - Run: `python3 train_multi_asset.py`")

    if not pred_exists:
        health_issues.append("No predictions - Run: `python3 predict_multi_asset_ensemble.py`")

    if health_issues:
        st.warning("**Issues Detected:**")
        for issue in health_issues:
            st.markdown(f"- {issue}")
    else:
        st.success("✅ All systems operational!")

    # Quick stats
    if pred_exists:
        st.markdown("---")
        st.subheader("📊 Quick Stats")

        try:
            pred_df = pd.read_csv(LOGS_DIR / "labeled_predictions.csv")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Predictions", f"{len(pred_df):,}")

            with col2:
                if 'Prediction' in pred_df.columns:
                    spike_pct = (pred_df['Prediction'] == 2).sum() / len(pred_df) * 100
                    st.metric("Spike Signals %", f"{spike_pct:.1f}%")

            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")

        except Exception as e:
            st.error(f"Error loading predictions: {e}")

# ============================================================================
# TAB 2: MODELS
# ============================================================================
with tab2:
    st.header("🤖 Model Status")

    model_files = {
        'XGBoost': 'xgboost_multi_asset.pkl',
        'LightGBM': 'lightgbm_multi_asset.pkl',
        'CatBoost': 'catboost_multi_asset.pkl'
    }

    model_data = []

    for name, filename in model_files.items():
        filepath = MODELS_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(filepath.stat().st_mtime)
            model_data.append({
                'Model': name,
                'Status': '✅ Loaded',
                'Size (MB)': f"{size_mb:.2f}",
                'Last Modified': modified.strftime('%Y-%m-%d %H:%M')
            })
        else:
            model_data.append({
                'Model': name,
                'Status': '❌ Missing',
                'Size (MB)': '-',
                'Last Modified': '-'
            })

    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)

    # Feature list
    st.markdown("---")
    st.subheader("📋 Features")

    feature_file = MODELS_DIR / "multi_asset_features.txt"
    if feature_file.exists():
        features = [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]
        st.info(f"✅ {len(features)} features configured")

        with st.expander("View Feature List"):
            st.text("\n".join(features[:50]))  # Show first 50
            if len(features) > 50:
                st.text(f"... and {len(features) - 50} more")
    else:
        st.warning("⚠️ Feature list not found")

    # Training commands
    st.markdown("---")
    st.subheader("💻 Training Commands")

    st.code("""
# Train all models
python3 train_multi_asset.py

# Train with weight optimization
python3 train_multi_asset.py --optimize-weights

# Train specific model
python3 train_multi_asset.py --model xgboost
    """)

# ============================================================================
# TAB 3: PREDICTIONS
# ============================================================================
with tab3:
    st.header("📈 Prediction Analysis")

    pred_file = LOGS_DIR / "labeled_predictions.csv"

    if not pred_file.exists():
        st.warning("No predictions found. Run: `python3 predict_multi_asset_ensemble.py`")
    else:
        try:
            pred_df = pd.read_csv(pred_file)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'], errors='coerce')
            pred_df = pred_df.dropna(subset=['Date'])

            # Signal distribution
            st.subheader("📊 Signal Distribution")

            if 'Prediction' in pred_df.columns:
                signal_counts = pred_df['Prediction'].value_counts().sort_index()
                signal_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}

                col1, col2, col3 = st.columns(3)

                for signal_id, col in zip([0, 1, 2], [col1, col2, col3]):
                    count = signal_counts.get(signal_id, 0)
                    pct = (count / len(pred_df)) * 100
                    signal_name = signal_map[signal_id]

                    with col:
                        st.metric(signal_name, f"{count:,}", delta=f"{pct:.1f}%")

                # Chart
                fig = go.Figure(data=[go.Bar(
                    x=['CRASH', 'NORMAL', 'SPIKE'],
                    y=[signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(2, 0)],
                    marker_color=['red', 'yellow', 'green']
                )])
                fig.update_layout(title="Signal Distribution", xaxis_title="Signal", yaxis_title="Count", height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Recent predictions
            st.markdown("---")
            st.subheader("📅 Recent Predictions (Last 20)")

            recent = pred_df.tail(20).copy()

            if 'Prediction' in recent.columns:
                recent['Signal'] = recent['Prediction'].map({0: '🔴 CRASH', 1: '🟡 NORMAL', 2: '🟢 SPIKE'})

            display_cols = ['Date', 'Signal', 'Proba', 'Confidence', 'Spike_Conf', 'Crash_Conf']
            available_cols = [col for col in display_cols if col in recent.columns]

            if available_cols:
                st.dataframe(recent[available_cols].sort_values('Date', ascending=False), use_container_width=True)

            # Confidence histogram
            if 'Confidence' in pred_df.columns:
                st.markdown("---")
                st.subheader("🎯 Confidence Distribution")

                fig = go.Figure(data=[go.Histogram(x=pred_df['Confidence'], nbinsx=50, marker_color='blue')])
                fig.update_layout(title="Confidence Values", xaxis_title="Confidence", yaxis_title="Frequency", height=300)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading predictions: {e}")

# ============================================================================
# TAB 4: BACKTEST
# ============================================================================
with tab4:
    st.header("💰 Backtest Results")

    backtest_file = LOGS_DIR / "backtest_results.csv"

    if not backtest_file.exists():
        st.warning("No backtest results found. Run: `python3 backtest.py`")

        st.markdown("---")
        st.subheader("💻 Run Backtest")

        st.code("""
# Run backtest with default config
python3 backtest.py

# Run with specific profile
python3 backtest.py --profile moderate

# Run with custom config
python3 backtest.py --config configs/backtest_optimized.json
        """)
    else:
        try:
            bt_df = pd.read_csv(backtest_file)

            st.success(f"✅ Loaded {len(bt_df):,} trades")

            # Summary metrics
            if 'profit_pct' in bt_df.columns and len(bt_df) > 0:
                st.markdown("---")
                st.subheader("📊 Performance Summary")

                total_return = bt_df['profit_pct'].sum()
                wins = (bt_df['profit_pct'] > 0).sum()
                losses = (bt_df['profit_pct'] < 0).sum()
                win_rate = (wins / len(bt_df)) * 100 if len(bt_df) > 0 else 0

                avg_win = bt_df[bt_df['profit_pct'] > 0]['profit_pct'].mean() if wins > 0 else 0
                avg_loss = bt_df[bt_df['profit_pct'] < 0]['profit_pct'].mean() if losses > 0 else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%", delta="Cumulative")

                with col2:
                    st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins}/{len(bt_df)}")

                with col3:
                    st.metric("Avg Win", f"{avg_win:.2f}%")

                with col4:
                    st.metric("Avg Loss", f"{avg_loss:.2f}%")

                # Equity curve
                st.markdown("---")
                st.subheader("📈 Equity Curve")

                if 'cumulative_pnl' in bt_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=bt_df['cumulative_pnl'], mode='lines', name='Equity',
                                           line=dict(color='blue', width=2)))
                    fig.update_layout(title="Cumulative P&L", xaxis_title="Trade Number", yaxis_title="P&L ($)", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Trade distribution
                st.markdown("---")
                st.subheader("📊 Trade Distribution")

                fig = go.Figure(data=[go.Histogram(x=bt_df['profit_pct'], nbinsx=50, marker_color='purple')])
                fig.update_layout(title="Profit/Loss Distribution", xaxis_title="Profit %", yaxis_title="Frequency", height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Recent trades
                st.markdown("---")
                st.subheader("📅 Recent Trades (Last 10)")

                recent_trades = bt_df.tail(10)
                display_cols = ['entry_date', 'exit_date', 'signal', 'profit_pct', 'cumulative_pnl']
                available_cols = [col for col in display_cols if col in recent_trades.columns]

                if available_cols:
                    st.dataframe(recent_trades[available_cols], use_container_width=True)

        except Exception as e:
            st.error(f"Error loading backtest results: {e}")

# ============================================================================
# TAB 5: DATA
# ============================================================================
with tab5:
    st.header("📁 Data Health Check")

    # SPY data
    st.subheader("📊 SPY Data")

    spy_file = DATA_DIR / "SPY.csv"

    if not spy_file.exists():
        st.error("❌ SPY.csv not found")
        st.code("python3 update_spy_data.py")
    else:
        try:
            spy_df = pd.read_csv(spy_file)

            if len(spy_df) == 0:
                st.error("❌ SPY.csv is empty!")
                st.code("python3 update_spy_data.py")
            else:
                st.success(f"✅ {len(spy_df):,} rows")

                # Convert to numeric
                for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    if col in spy_df.columns:
                        spy_df[col] = pd.to_numeric(spy_df[col], errors='coerce')

                spy_df['Date'] = pd.to_datetime(spy_df['Date'], errors='coerce')
                spy_df = spy_df.dropna(subset=['Date'])

                if len(spy_df) > 0:
                    first_date = spy_df['Date'].min()
                    last_date = spy_df['Date'].max()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("First Date", first_date.strftime('%Y-%m-%d'))

                    with col2:
                        st.metric("Last Date", last_date.strftime('%Y-%m-%d'))

                    with col3:
                        days = (last_date - first_date).days
                        st.metric("Days of Data", f"{days:,}")

                    # Quick chart
                    st.markdown("---")
                    st.subheader("📈 SPY Price (Last 90 Days)")

                    recent_spy = spy_df.tail(90)
                    fig = go.Figure(data=[go.Scatter(x=recent_spy['Date'], y=recent_spy['Close'],
                                                     mode='lines', name='SPY', line=dict(color='blue', width=2))])
                    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)", height=400)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading SPY data: {e}")

    # Crypto data
    st.markdown("---")
    st.subheader("💎 Crypto Data (Optional)")

    crypto_files = ['BTC.csv', 'ETH.csv', 'SOL.csv']
    crypto_found = []

    for crypto_file in crypto_files:
        filepath = DATA_DIR / crypto_file
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                crypto_found.append((crypto_file.replace('.csv', ''), len(df)))
            except Exception:
                pass

    if crypto_found:
        for ticker, rows in crypto_found:
            st.success(f"✅ {ticker}: {rows:,} rows")
    else:
        st.info("No crypto data found (optional)")
        st.code("python3 download_crypto_enhanced.py")

    # Directory summary
    st.markdown("---")
    st.subheader("📂 Directory Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        if DATA_DIR.exists():
            csv_count = len(list(DATA_DIR.glob("*.csv")))
            st.metric("data/", f"{csv_count} CSV files")

    with col2:
        if LOGS_DIR.exists():
            log_count = len(list(LOGS_DIR.glob("*.csv")))
            st.metric("logs/", f"{log_count} files")

    with col3:
        if MODELS_DIR.exists():
            model_count = len(list(MODELS_DIR.glob("*.pkl")))
            st.metric("models/", f"{model_count} models")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <b>NeuroVest Quick Demo Dashboard</b><br>
    For comprehensive features, run: <code>streamlit run dashboard_comprehensive.py</code><br>
    For CLI demos, run: <code>python3 demo.py</code> or <code>python3 demo_comprehensive.py</code>
</div>
""", unsafe_allow_html=True)
