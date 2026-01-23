#!/usr/bin/env python3
"""
NeuroVest API Dashboard

Interactive web interface for NeuroVest Market Forecasting API.
View predictions, analyze forecast accuracy, and monitor API performance.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py --server.port 8080

Requirements:
    pip install streamlit plotly pandas

Features:
- Asset data visualization
- API forecast results viewer
- Prediction accuracy metrics
- Historical forecast performance
- Custom data import for forecasting
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
    page_title="NeuroVest Forecasting API",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data directories
DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")


def get_available_assets():
    """Get list of available assets for forecasting"""
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

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Error reading {ticker}: {e}")
        return None

    # Check if file is empty
    if len(df) == 0:
        st.warning(f"{ticker} data file is empty. Run: python3 update_spy_data.py")
        return None

    df = df.copy()  # Avoid SettingWithCopyWarning

    # Handle different date column names
    date_col = None
    for col in ['Date', 'date', 'Timestamp', 'timestamp', 'datetime', 'time']:
        if col in df.columns:
            date_col = col
            break

    # Check if index might be the date
    if date_col is None:
        if df.index.name in ['Date', 'date', 'Timestamp']:
            df = df.reset_index()
            date_col = df.columns[0]
        elif len(df.columns) > 0:
            # Try first column if it looks like dates
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0])
                date_col = first_col
            except Exception:
                return None  # Can't find date column

    if date_col is None:
        return None

    df['Date'] = pd.to_datetime(df[date_col])
    df = df.sort_values('Date')

    # Convert numeric columns to proper types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where Close is NaN (essential for calculations)
    if 'Close' in df.columns:
        df = df.dropna(subset=['Close'])

    return df


def load_predictions(ticker):
    """Load API forecast results for an asset"""
    pred_path = LOGS_DIR / "predictions" / f"{ticker.replace('/', '_')}_predictions.csv"

    if pred_path.exists():
        return pd.read_csv(pred_path)

    return None


def load_forecast_results():
    """Load forecast results if available"""
    results_path = LOGS_DIR / "backtest_results.csv"

    if results_path.exists():
        return pd.read_csv(results_path)

    return None


def create_price_chart(df, ticker):
    """Create interactive price chart"""
    df = df.copy()
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price', 'Volume')
    )

    # Candlestick if OHLC available
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            ),
            row=1, col=1
        )
    else:
        # Just close price
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name=ticker,
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

    # Volume
    if 'Volume' in df.columns:
        colors = ['red' if df['Close'].iloc[i] < df['Close'].iloc[i-1] else 'green'
                 for i in range(1, len(df))]
        colors = ['gray'] + colors  # First bar

        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

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
    st.sidebar.title("NeuroVest API")
    st.sidebar.markdown("*Market Forecasting Tool*")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Asset Analysis", "API Forecasts", "Forecast Performance", "Import Data", "Settings"]
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
    elif page == "API Forecasts":
        show_forecasts(selected_asset)
    elif page == "Forecast Performance":
        show_performance()
    elif page == "Import Data":
        show_import()
    elif page == "Settings":
        show_settings()


def show_dashboard(selected_asset):
    """Main dashboard view - Deployment-ready homepage"""
    # Hero section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">📊 NeuroVest Forecasting API</h1>
        <p style="font-size: 1.5rem; color: #666;">AI-Powered Market Predictions & Economic Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Value proposition
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
        <h2 style="color: white; margin-top: 0;">🎯 What is NeuroVest?</h2>
        <p style="font-size: 1.2rem; line-height: 1.6;">
            <b>NeuroVest is a Market Forecasting API</b> that predicts price movements using ensemble machine learning (XGBoost + LightGBM + CatBoost).
        </p>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            ✨ <b>Primary Function:</b> Generate 3-class forecasts (CRASH / NORMAL / SPIKE) for stocks, ETFs, crypto, and precious metals
        </p>
        <p style="font-size: 1.0rem; line-height: 1.6; margin-bottom: 0;">
            ⚠️ <b>Note:</b> This is a forecasting tool, NOT a trading system. It provides predictions and analysis, not trade execution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    assets = get_available_assets()
    crypto = len([a for a in assets if '/' in a])
    etfs = len([a for a in assets if '/' not in a])

    with col1:
        st.metric("📈 Total Assets", len(assets), help="All available assets for forecasting")

    with col2:
        st.metric("💎 Crypto Assets", crypto, help="Cryptocurrency pairs")

    with col3:
        st.metric("📊 Stock/ETF Assets", etfs, help="Traditional market assets")

    with col4:
        pred_count = 0
        if LOGS_DIR.exists():
            pred_files = list(LOGS_DIR.glob("*predictions*.csv"))
            pred_count = len(pred_files)
        st.metric("🔮 Forecast Files", pred_count, help="Generated prediction files")

    # Use Cases
    st.markdown("---")
    st.markdown("### 🎬 Use Cases")

    use_case_col1, use_case_col2, use_case_col3 = st.columns(3)

    with use_case_col1:
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 1.5rem; border-radius: 10px; height: 100%;">
            <h4 style="color: #2E7D32;">💼 Institutional Research</h4>
            <p style="color: #1B5E20;">
                • Market regime classification<br>
                • Economic indicator analysis<br>
                • Risk assessment & stress testing<br>
                • Multi-asset correlation studies
            </p>
        </div>
        """, unsafe_allow_html=True)

    with use_case_col2:
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 1.5rem; border-radius: 10px; height: 100%;">
            <h4 style="color: #1565C0;">📊 Portfolio Management</h4>
            <p style="color: #0D47A1;">
                • Asset allocation signals<br>
                • Rebalancing optimization<br>
                • Recession probability tracking<br>
                • Valuation-based positioning
            </p>
        </div>
        """, unsafe_allow_html=True)

    with use_case_col3:
        st.markdown("""
        <div style="background-color: #FCE4EC; padding: 1.5rem; border-radius: 10px; height: 100%;">
            <h4 style="color: #C2185B;">🔬 Research & Development</h4>
            <p style="color: #880E4F;">
                • ML model benchmarking<br>
                • Feature importance analysis<br>
                • Prediction accuracy studies<br>
                • Custom model integration
            </p>
        </div>
        """, unsafe_allow_html=True)

    # System Status
    st.markdown("---")
    st.markdown("### 🔧 System Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown("**📁 Data Files**")
        spy_exists = (DATA_DIR / "SPY.csv").exists()
        spy_status = "🟢 Ready" if spy_exists else "🔴 Missing"

        if spy_exists:
            spy_df = load_asset_data('SPY')
            rows = len(spy_df) if spy_df is not None else 0
            st.markdown(f"{spy_status} SPY.csv ({rows:,} rows)")
        else:
            st.markdown(f"{spy_status} SPY.csv")

        st.markdown(f"Total assets: {len(assets)}")

    with status_col2:
        st.markdown("**🤖 ML Models**")
        for model in ['xgboost', 'lightgbm', 'catboost']:
            exists = (MODELS_DIR / f"{model}_multi_asset.pkl").exists()
            status = "🟢" if exists else "🔴"
            st.markdown(f"{status} {model.capitalize()}")

    with status_col3:
        st.markdown("**🔮 Forecasts**")
        pred_file = LOGS_DIR / "labeled_predictions.csv"
        if pred_file.exists():
            try:
                df = pd.read_csv(pred_file)
                st.markdown(f"🟢 {len(df):,} predictions")
            except Exception:
                st.markdown("🟡 File exists (parse error)")
        else:
            st.markdown("🔴 No predictions yet")

    # Quick chart for selected asset
    if selected_asset:
        st.markdown("---")
        st.markdown(f"### 📈 {selected_asset} Quick View")

        df = load_asset_data(selected_asset)
        if df is not None and len(df) > 0:
            # Show last 90 days
            recent = df.tail(90)
            fig = create_price_chart(recent, selected_asset)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for this asset")
    else:
        st.info("💡 Select an asset from the sidebar to view its price chart")


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

    # Price chart
    st.markdown("---")
    st.subheader("Price Chart")

    fig = create_price_chart(filtered_df, selected_asset)
    st.plotly_chart(fig, use_container_width=True)


def show_forecasts(selected_asset):
    """API forecasts view"""
    st.title("API Forecast Results")
    st.markdown("*View predictions generated by the forecasting models*")

    st.subheader("Recent Forecasts")

    signals_path = LOGS_DIR / "labeled_predictions.csv"
    if signals_path.exists():
        signals = pd.read_csv(signals_path)
        if len(signals) > 0:
            recent = signals.tail(20)

            # Map predictions to labels
            if 'Prediction' in recent.columns:
                recent = recent.copy()
                recent['Forecast'] = recent['Prediction'].map({
                    0: '🔴 CRASH',
                    1: '🟡 NORMAL',
                    2: '🟢 SPIKE'
                })

            st.dataframe(recent, use_container_width=True)
        else:
            st.info("No forecast results found")
    else:
        st.info("No forecast history available. Run: `python3 predict_multi_asset_ensemble.py`")


def show_performance():
    """Forecast performance analysis"""
    st.title("Forecast Performance Analysis")

    # Load results
    results = load_forecast_results()

    if results is not None:
        st.subheader("Latest Results")
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No performance results available. Run forecast validation first.")

    st.markdown("---")
    st.subheader("Performance Metrics")
    st.code("""
# Extract performance metrics
python3 extract_metrics.py --comprehensive

# Validate forecast signals
python3 validate_signals.py --detailed
    """)


def show_import():
    """Data import view"""
    st.title("Import Custom Data")

    st.markdown("""
    Import your own asset data to generate forecasts.

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
                df.to_csv(save_path, index=False)
                st.success(f"Imported as {ticker}")
                st.info(f"Saved to: {save_path}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")
    st.subheader("Command Line Import")
    st.code("""
# Import from command line with validation:
python3 import_custom_asset.py my_data.csv MYTICKER

# Create sample template:
python3 import_custom_asset.py --sample
    """)


def show_settings():
    """Settings view"""
    st.title("API Settings")

    st.subheader("API Keys")

    with st.expander("OpenAI"):
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.info("API key configured (not saved)")

    with st.expander("NewsAPI"):
        news_key = st.text_input("News API Key", type="password")

    st.markdown("---")

    st.subheader("Model Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Default Model", ["XGBoost", "LightGBM", "CatBoost", "Ensemble"])

    with col2:
        st.number_input("Forecast Horizon (days)", value=5, min_value=1, max_value=30)

    st.markdown("---")

    st.subheader("System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Data Directory:** {DATA_DIR}")
        st.write(f"**Cache Directory:** {CACHE_DIR}")
        st.write(f"**Models Directory:** {MODELS_DIR}")

    with col2:
        packages = {
            'streamlit': st.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
        }

        for pkg, ver in packages.items():
            st.write(f"**{pkg}:** {ver}")


if __name__ == "__main__":
    main()
