#!/usr/bin/env python3
"""
NeuroVest Web Dashboard

Interactive web interface for NeuroVest using Streamlit.

Usage:
    streamlit run dashboard.py

    Or with custom port:
    streamlit run dashboard.py --server.port 8080

Requirements:
    pip install streamlit plotly pandas

Features:
- Asset overview and data visualization
- Interactive predictions
- Backtest results viewer
- Custom asset import
- LLM analysis integration
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install streamlit plotly pandas")
    sys.exit(1)

# Configure page
st.set_page_config(
    page_title="NeuroVest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data directories
DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")


def get_available_assets():
    """Get list of available assets"""
    assets = set()

    # Check data directory
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.csv"):
            if f.stem not in ['cross_asset_features', 'macro_features', 'sentiment_features']:
                assets.add(f.stem)

    # Check cache directory
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*_1d.csv"):
            ticker = f.stem.replace('_1d', '').replace('_', '/')
            assets.add(ticker)

    return sorted(list(assets))


def load_asset_data(ticker):
    """Load asset data from file"""
    # Handle different file locations
    if ticker == 'SPY':
        filepath = DATA_DIR / "SPY.csv"
    elif '/' in ticker:
        filename = ticker.replace('/', '_') + '_1d.csv'
        filepath = CACHE_DIR / filename
    else:
        filepath = CACHE_DIR / f"{ticker}_1d.csv"
        if not filepath.exists():
            filepath = DATA_DIR / f"{ticker}.csv"

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    return df


def load_predictions(ticker):
    """Load latest predictions for an asset"""
    pred_path = LOGS_DIR / "predictions" / f"{ticker.replace('/', '_')}_predictions.csv"

    if pred_path.exists():
        return pd.read_csv(pred_path)

    return None


def load_backtest_results():
    """Load backtest results if available"""
    results_path = LOGS_DIR / "backtest_results.csv"

    if results_path.exists():
        return pd.read_csv(results_path)

    return None


def create_price_chart(df, ticker):
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f'{ticker} Price', 'Volume', 'RSI']
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add moving averages
    if len(df) >= 20:
        df['MA20'] = df['Close'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA20'], name='MA20',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )

    if len(df) >= 50:
        df['MA50'] = df['Close'].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA50'], name='MA50',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )

    # Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green'
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
               marker_color=colors, showlegend=False),
        row=2, col=1
    )

    # RSI
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        fig.add_trace(
            go.Scatter(x=df['Date'], y=rsi, name='RSI',
                      line=dict(color='purple', width=1)),
            row=3, col=1
        )

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_returns_chart(df):
    """Create returns distribution chart"""
    returns = df['Close'].pct_change().dropna() * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker_color='steelblue'
    ))

    # Add normal distribution overlay
    mean = returns.mean()
    std = returns.std()

    fig.add_vline(x=mean, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean:.2f}%")

    fig.update_layout(
        title='Daily Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        height=400
    )

    return fig


def calculate_metrics(df):
    """Calculate key metrics for an asset"""
    if len(df) < 2:
        return {}

    returns = df['Close'].pct_change().dropna()

    # Basic metrics
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

    # Annualized metrics
    days = (df['Date'].max() - df['Date'].min()).days
    years = days / 365.25

    if years > 0:
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        annual_vol = returns.std() * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    else:
        annual_return = 0
        annual_vol = 0
        sharpe = 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    return {
        'Total Return': f"{total_return:.1f}%",
        'Annual Return': f"{annual_return:.1f}%",
        'Annual Volatility': f"{annual_vol:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown:.1f}%",
        'Trading Days': len(df),
        'Date Range': f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    }


def main():
    # Sidebar
    st.sidebar.title("NeuroVest")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Asset Analysis", "Predictions", "Backtest", "Import Data", "Settings"]
    )

    # Asset selection
    assets = get_available_assets()
    if assets:
        selected_asset = st.sidebar.selectbox("Select Asset", assets, index=0)
    else:
        selected_asset = None
        st.sidebar.warning("No assets found. Import data first.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Main content
    if page == "Dashboard":
        show_dashboard(selected_asset)
    elif page == "Asset Analysis":
        show_asset_analysis(selected_asset)
    elif page == "Predictions":
        show_predictions(selected_asset)
    elif page == "Backtest":
        show_backtest()
    elif page == "Import Data":
        show_import()
    elif page == "Settings":
        show_settings()


def show_dashboard(selected_asset):
    """Main dashboard view"""
    st.title("NeuroVest Dashboard")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    assets = get_available_assets()

    with col1:
        st.metric("Total Assets", len(assets))

    with col2:
        crypto = len([a for a in assets if '/' in a])
        st.metric("Crypto Assets", crypto)

    with col3:
        etfs = len([a for a in assets if '/' not in a])
        st.metric("ETF Assets", etfs)

    with col4:
        # Check for recent predictions
        pred_count = 0
        if LOGS_DIR.exists():
            pred_files = list(LOGS_DIR.glob("*predictions*.csv"))
            pred_count = len(pred_files)
        st.metric("Predictions", pred_count)

    st.markdown("---")

    # Asset overview table
    st.subheader("Asset Overview")

    if assets:
        overview_data = []

        for asset in assets[:10]:  # Show first 10
            df = load_asset_data(asset)
            if df is not None and len(df) > 0:
                returns = df['Close'].pct_change().dropna()
                total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                daily_return = returns.iloc[-1] * 100 if len(returns) > 0 else 0

                overview_data.append({
                    'Asset': asset,
                    'Last Price': f"${df['Close'].iloc[-1]:,.2f}",
                    'Daily Return': f"{daily_return:+.2f}%",
                    'Total Return': f"{total_return:,.1f}%",
                    'Days': len(df)
                })

        if overview_data:
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)

    # Quick chart for selected asset
    if selected_asset:
        st.markdown("---")
        st.subheader(f"{selected_asset} Quick View")

        df = load_asset_data(selected_asset)
        if df is not None:
            # Show last 90 days
            recent = df.tail(90)
            fig = create_price_chart(recent, selected_asset)
            st.plotly_chart(fig, use_container_width=True)


def show_asset_analysis(selected_asset):
    """Detailed asset analysis view"""
    st.title(f"Asset Analysis: {selected_asset}")

    if not selected_asset:
        st.warning("Please select an asset from the sidebar")
        return

    df = load_asset_data(selected_asset)

    if df is None:
        st.error(f"Could not load data for {selected_asset}")
        return

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['Date'].min())
    with col2:
        end_date = st.date_input("End Date", df['Date'].max())

    # Filter data
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    filtered_df = df[mask]

    # Metrics
    st.subheader("Key Metrics")
    metrics = calculate_metrics(filtered_df)

    cols = st.columns(4)
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % 4]:
            st.metric(key, value)

    st.markdown("---")

    # Charts
    tab1, tab2, tab3 = st.tabs(["Price Chart", "Returns", "Statistics"])

    with tab1:
        fig = create_price_chart(filtered_df, selected_asset)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = create_returns_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Detailed statistics
        returns = filtered_df['Close'].pct_change().dropna()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Return Statistics**")
            st.write(f"Mean Daily Return: {returns.mean()*100:.3f}%")
            st.write(f"Median Daily Return: {returns.median()*100:.3f}%")
            st.write(f"Std Dev: {returns.std()*100:.3f}%")
            st.write(f"Skewness: {returns.skew():.3f}")
            st.write(f"Kurtosis: {returns.kurtosis():.3f}")

        with col2:
            st.markdown("**Tail Risk**")
            st.write(f"5% VaR: {returns.quantile(0.05)*100:.2f}%")
            st.write(f"1% VaR: {returns.quantile(0.01)*100:.2f}%")
            st.write(f"Best Day: {returns.max()*100:.2f}%")
            st.write(f"Worst Day: {returns.min()*100:.2f}%")


def show_predictions(selected_asset):
    """Predictions view"""
    st.title("Predictions")

    if not selected_asset:
        st.warning("Please select an asset from the sidebar")
        return

    st.subheader(f"Generate Prediction for {selected_asset}")

    col1, col2 = st.columns(2)

    with col1:
        horizon = st.selectbox("Prediction Horizon", [1, 5, 10, 20], index=1)

    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.7)

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Running prediction..."):
            # Here you would call the actual prediction code
            # For now, show placeholder
            st.info("Prediction functionality would run here")
            st.write("To run predictions from command line:")
            st.code(f"python3 predict.py --asset {selected_asset}")

    st.markdown("---")

    # Show existing predictions if available
    st.subheader("Recent Predictions")

    signals_path = LOGS_DIR / "signals.csv"
    if signals_path.exists():
        signals = pd.read_csv(signals_path)
        if len(signals) > 0:
            recent = signals.tail(20)
            st.dataframe(recent, use_container_width=True)
        else:
            st.info("No predictions found")
    else:
        st.info("No prediction history available. Run predictions first.")


def show_backtest():
    """Backtest results view"""
    st.title("Backtest Results")

    # Configuration
    st.subheader("Backtest Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        config = st.selectbox(
            "Strategy",
            ["Optimized (1.25x ATR)", "High Profit (1.75x ATR)", "Aggressive (2.5x ATR)"]
        )

    with col2:
        initial_capital = st.number_input("Initial Capital", value=10000, min_value=1000)

    with col3:
        asset = st.selectbox("Asset", get_available_assets())

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            st.info("Backtest would run here")
            st.write("To run from command line:")
            st.code(f"python3 backtest.py --asset {asset}")

    st.markdown("---")

    # Show results if available
    results = load_backtest_results()

    if results is not None:
        st.subheader("Latest Results")
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No backtest results available. Run a backtest first.")


def show_import():
    """Data import view"""
    st.title("Import Custom Data")

    st.markdown("""
    Import your own asset data from CSV or Excel files.

    **Required columns:**
    - Date (or Time, Timestamp)
    - Close (or Price)

    **Optional columns:**
    - Open, High, Low
    - Volume
    """)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file:
        # Preview data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {list(df.columns)}")

            # Import options
            ticker = st.text_input("Ticker Symbol", value="CUSTOM")

            if st.button("Import Data", type="primary"):
                # Save to data_cache
                save_path = CACHE_DIR / f"{ticker}_1d.csv"

                # Standardize columns
                # (simplified version - full logic in import_custom_asset.py)
                df.to_csv(save_path, index=False)
                st.success(f"Imported as {ticker}")
                st.info(f"Saved to: {save_path}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")

    # Show command line option
    st.subheader("Command Line Import")
    st.code("""
# Import from command line with full validation:
python3 import_custom_asset.py my_data.csv MYTICKER

# Create sample template:
python3 import_custom_asset.py --sample
    """)


def show_settings():
    """Settings view"""
    st.title("Settings")

    st.subheader("API Keys")

    with st.expander("OpenAI"):
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.info("API key configured (not saved)")

    with st.expander("Telegram Notifications"):
        telegram_token = st.text_input("Bot Token", type="password")
        chat_id = st.text_input("Chat ID")

    st.markdown("---")

    st.subheader("Model Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Default Model", ["XGBoost", "LightGBM", "CatBoost", "Ensemble"])

    with col2:
        st.number_input("Prediction Horizon (days)", value=5, min_value=1, max_value=30)

    st.markdown("---")

    st.subheader("System Information")

    # Show system info
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Data Directory:** {DATA_DIR}")
        st.write(f"**Cache Directory:** {CACHE_DIR}")
        st.write(f"**Models Directory:** {MODELS_DIR}")

    with col2:
        # Check for installed packages
        packages = {
            'streamlit': st.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
        }

        for pkg, ver in packages.items():
            st.write(f"**{pkg}:** {ver}")


if __name__ == "__main__":
    main()
