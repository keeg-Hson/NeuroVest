#!/usr/bin/env python3
"""
NeuroVest Comprehensive Forecasting Dashboard

Full-featured interface showcasing ALL NeuroVest capabilities:
- 41 assets (stocks, ETFs, crypto, precious metals)
- Recession probability analysis
- Asset valuation detection
- LLM-powered market analysis
- Portfolio rebalancing optimization
- Custom data imports
- Full pipeline automation

Usage:
    streamlit run dashboard_comprehensive.py
    streamlit run dashboard_comprehensive.py --server.port 8501
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
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

# Directories
DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

# All supported assets
STOCK_ETFS = {
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
    'DIA': 'Dow Jones', 'VTI': 'Total Stock Market', 'EEM': 'Emerging Markets',
    'XLF': 'Financials', 'XLK': 'Technology', 'XLE': 'Energy',
    'DXY': 'US Dollar', 'HYG': 'High Yield Bonds', 'LQD': 'Investment Grade Bonds',
    'TNX': '10Y Treasury', 'UUP': 'US Dollar Bull'
}

PRECIOUS_METALS = {
    'GLD': 'Gold Trust', 'SLV': 'Silver Trust', 'GDX': 'Gold Miners',
    'GDXJ': 'Junior Gold Miners', 'IAU': 'iShares Gold',
    'PPLT': 'Platinum', 'PALL': 'Palladium'
}

CRYPTO_ASSETS = {
    'BTC/USDT': 'Bitcoin', 'ETH/USDT': 'Ethereum', 'SOL/USDT': 'Solana',
    'BNB/USDT': 'Binance Coin', 'XRP/USDT': 'Ripple', 'ADA/USDT': 'Cardano',
    'DOGE/USDT': 'Dogecoin', 'AVAX/USDT': 'Avalanche',
    'MATIC/USDT': 'Polygon', 'LINK/USDT': 'Chainlink'
}

def check_asset_status(ticker):
    """Check if asset data is downloaded"""
    # Check data/ directory
    if (DATA_DIR / f"{ticker}.csv").exists():
        return "downloaded"

    # Check data_cache/ directory
    safe_ticker = ticker.replace('/', '_')
    if (CACHE_DIR / f"{safe_ticker}_1d.csv").exists():
        return "downloaded"

    return "available"

def load_asset_data(ticker):
    """Load asset data if available"""
    # Try data/ first
    filepath = DATA_DIR / f"{ticker}.csv"
    if not filepath.exists():
        # Try data_cache/
        safe_ticker = ticker.replace('/', '_')
        filepath = CACHE_DIR / f"{safe_ticker}_1d.csv"

    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return None

        # Convert date column
        for col in ['Date', 'date', 'Timestamp']:
            if col in df.columns:
                df['Date'] = pd.to_datetime(df[col])
                break

        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])

        return df.sort_values('Date') if 'Date' in df.columns else df

    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None

def main():
    # Sidebar
    st.sidebar.title("NeuroVest API")
    st.sidebar.markdown("*Market Forecasting Tool*")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "📊 Overview",
            "📥 Asset Manager",
            "📉 Recession Indicator",
            "💰 Valuation Detector",
            "🤖 LLM Analysis",
            "🔄 Portfolio Rebalancing",
            "📈 Forecast Results",
            "🚀 Automation",
            "📁 Custom Imports"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")

    if st.sidebar.button("🔄 Refresh Display"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")

    # Count downloaded assets
    downloaded = 0
    for ticker in list(STOCK_ETFS.keys()) + list(PRECIOUS_METALS.keys()) + list(CRYPTO_ASSETS.keys()):
        if check_asset_status(ticker) == "downloaded":
            downloaded += 1

    total_assets = len(STOCK_ETFS) + len(PRECIOUS_METALS) + len(CRYPTO_ASSETS)
    st.sidebar.metric("Assets Downloaded", f"{downloaded}/{total_assets}")

    # Check models
    model_count = sum(1 for f in ['xgboost_multi_asset.pkl', 'lightgbm_multi_asset.pkl', 'catboost_multi_asset.pkl']
                     if (MODELS_DIR / f).exists())
    st.sidebar.metric("Models Trained", f"{model_count}/3")

    # Route to pages
    if page == "📊 Overview":
        show_overview()
    elif page == "📥 Asset Manager":
        show_asset_manager()
    elif page == "📉 Recession Indicator":
        show_recession_indicator()
    elif page == "💰 Valuation Detector":
        show_valuation_detector()
    elif page == "🤖 LLM Analysis":
        show_llm_analysis()
    elif page == "🔄 Portfolio Rebalancing":
        show_portfolio_rebalancing()
    elif page == "📈 Forecast Results":
        show_forecast_results()
    elif page == "🚀 Automation":
        show_automation()
    elif page == "📁 Custom Imports":
        show_custom_imports()


def show_overview():
    """Overview page"""
    st.title("📊 NeuroVest Forecasting API")
    st.markdown("### AI-Powered Market Predictions & Economic Analysis")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h4>🎯 What NeuroVest Does</h4>
    <p><b>NeuroVest is a Market Forecasting API</b> that predicts price movements using ensemble machine learning.</p>

    <p><b>Primary Function:</b> Generate 3-class forecasts (CRASH / NORMAL / SPIKE) for stocks, ETFs, crypto, and precious metals</p>

    <p><b>Not a Trading System:</b> This is a forecasting tool. It provides predictions and analysis, not trade execution.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)

    total_assets = len(STOCK_ETFS) + len(PRECIOUS_METALS) + len(CRYPTO_ASSETS)
    downloaded = sum(1 for t in list(STOCK_ETFS.keys()) + list(PRECIOUS_METALS.keys()) + list(CRYPTO_ASSETS.keys())
                    if check_asset_status(t) == "downloaded")

    with col1:
        st.metric("Total Assets Supported", total_assets)

    with col2:
        st.metric("Assets Downloaded", downloaded)

    with col3:
        pred_count = len(list(LOGS_DIR.glob("*predictions*.csv"))) if LOGS_DIR.exists() else 0
        st.metric("Forecast Files", pred_count)

    with col4:
        model_count = sum(1 for f in ['xgboost_multi_asset.pkl', 'lightgbm_multi_asset.pkl', 'catboost_multi_asset.pkl']
                         if (MODELS_DIR / f).exists())
        st.metric("Models Trained", f"{model_count}/3")

    # Feature showcase
    st.markdown("---")
    st.subheader("🎯 Core Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Forecasting & Analysis:**
        - 📈 Multi-asset ensemble predictions
        - 📉 Recession probability indicator
        - 💰 Asset valuation detector
        - 🤖 LLM-powered market analysis
        - 📊 Signal validation & backtesting
        """)

    with col2:
        st.markdown("""
        **Asset Coverage:**
        - 📊 14 Stock/ETF assets
        - 🥇 7 Precious metals (Gold, Silver, etc.)
        - 💎 10 Cryptocurrencies
        - 📁 Custom data imports
        - 🔄 Multi-asset portfolio analysis
        """)

    # Quick status
    st.markdown("---")
    st.subheader("📊 System Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown("**Data Files:**")
        spy_status = "✅" if (DATA_DIR / "SPY.csv").exists() else "❌"
        st.markdown(f"{spy_status} SPY.csv")

        crypto_downloaded = sum(1 for t in CRYPTO_ASSETS.keys() if check_asset_status(t) == "downloaded")
        st.markdown(f"💎 Crypto: {crypto_downloaded}/10 downloaded")

    with status_col2:
        st.markdown("**Models:**")
        for model in ['xgboost', 'lightgbm', 'catboost']:
            exists = (MODELS_DIR / f"{model}_multi_asset.pkl").exists()
            icon = "✅" if exists else "❌"
            st.markdown(f"{icon} {model.capitalize()}")

    with status_col3:
        st.markdown("**Forecasts:**")
        pred_file = LOGS_DIR / "labeled_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            st.markdown(f"✅ {len(df):,} predictions")
        else:
            st.markdown("❌ No predictions")


def show_asset_manager():
    """Asset download manager"""
    st.title("📥 Asset Manager")
    st.markdown("*Download and manage data for all 41 supported assets*")

    # Asset category tabs
    tab1, tab2, tab3 = st.tabs(["📊 Stocks & ETFs", "🥇 Precious Metals", "💎 Crypto"])

    with tab1:
        st.subheader(f"Stock & ETF Assets ({len(STOCK_ETFS)} available)")

        asset_data = []
        for ticker, name in STOCK_ETFS.items():
            status = check_asset_status(ticker)
            df = load_asset_data(ticker)
            rows = len(df) if df is not None else 0

            asset_data.append({
                'Ticker': ticker,
                'Name': name,
                'Status': '✅ Downloaded' if status == 'downloaded' else '⬇️ Available',
                'Rows': rows if rows > 0 else '-'
            })

        df_assets = pd.DataFrame(asset_data)
        st.dataframe(df_assets, use_container_width=True)

        st.markdown("**Download Command:**")
        st.code("python3 update_spy_data.py  # For SPY\npython3 download_equity_etfs.py  # For all ETFs")

    with tab2:
        st.subheader(f"Precious Metals ({len(PRECIOUS_METALS)} available)")

        metal_data = []
        for ticker, name in PRECIOUS_METALS.items():
            status = check_asset_status(ticker)
            df = load_asset_data(ticker)
            rows = len(df) if df is not None else 0

            metal_data.append({
                'Ticker': ticker,
                'Name': name,
                'Status': '✅ Downloaded' if status == 'downloaded' else '⬇️ Available',
                'Rows': rows if rows > 0 else '-'
            })

        df_metals = pd.DataFrame(metal_data)
        st.dataframe(df_metals, use_container_width=True)

        st.markdown("**Download Command:**")
        st.code("python3 framework/download_all_assets.py --asset GLD\npython3 framework/download_all_assets.py --asset SLV")

    with tab3:
        st.subheader(f"Cryptocurrency Assets ({len(CRYPTO_ASSETS)} available)")

        crypto_data = []
        for ticker, name in CRYPTO_ASSETS.items():
            status = check_asset_status(ticker)
            df = load_asset_data(ticker)
            rows = len(df) if df is not None else 0

            crypto_data.append({
                'Ticker': ticker,
                'Name': name,
                'Status': '✅ Downloaded' if status == 'downloaded' else '⬇️ Available',
                'Rows': rows if rows > 0 else '-'
            })

        df_crypto = pd.DataFrame(crypto_data)
        st.dataframe(df_crypto, use_container_width=True)

        st.markdown("**Download Command:**")
        st.code("python3 download_crypto_enhanced.py  # Downloads all 10 crypto assets")

    # Bulk download section
    st.markdown("---")
    st.subheader("📦 Bulk Downloads")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Download All Stocks/ETFs:**")
        st.code("python3 download_equity_etfs.py")

    with col2:
        st.markdown("**Download All Crypto:**")
        st.code("python3 download_crypto_enhanced.py")


def show_recession_indicator():
    """Recession probability analysis"""
    st.title("📉 Recession Probability Indicator")
    st.markdown("*Multi-factor recession risk analysis*")

    st.info("💡 **Feature:** Analyzes yield curves, market stress, and technical signals to assess recession risk")

    # Load SPY for analysis
    spy_df = load_asset_data('SPY')

    if spy_df is None or len(spy_df) < 200:
        st.warning("⚠️ Insufficient SPY data. Download data first:")
        st.code("python3 update_spy_data.py")
        return

    # Calculate recession indicators
    recent = spy_df.tail(252)
    latest = recent.iloc[-1]

    # Market stress metrics
    returns = recent['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = ((cumulative - rolling_max) / rolling_max * 100).min()

    # Moving averages
    ma_50 = recent['Close'].rolling(50).mean().iloc[-1]
    ma_200 = recent['Close'].rolling(200).mean().iloc[-1]
    death_cross = ma_50 < ma_200

    # Recession score
    recession_score = 0
    if death_cross:
        recession_score += 25
    if latest['Close'] < ma_200:
        recession_score += 20
    if volatility > 25:
        recession_score += 20
    if drawdown < -15:
        recession_score += 25
    if volatility > 20 and drawdown < -10:
        recession_score += 10

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Price", f"${latest['Close']:.2f}")

    with col2:
        below_200ma = latest['Close'] < ma_200
        st.metric("vs 200-MA", f"${ma_200:.2f}",
                 delta="Below" if below_200ma else "Above",
                 delta_color="inverse" if below_200ma else "normal")

    with col3:
        st.metric("Volatility", f"{volatility:.1f}%",
                 delta="High" if volatility > 25 else "Normal")

    with col4:
        st.metric("Max Drawdown", f"{drawdown:.1f}%", delta_color="inverse")

    # Recession assessment
    st.markdown("---")
    st.subheader("🚨 Recession Risk Assessment")

    if recession_score > 70:
        st.error(f"⚠️ **HIGH RECESSION RISK** (Score: {recession_score}/100)")
        st.markdown("""
        **Recommended Actions:**
        - Reduce equity exposure
        - Increase cash/bond allocation
        - Consider defensive sectors
        - Hedge positions if appropriate
        """)
    elif recession_score > 40:
        st.warning(f"⚠️ **MODERATE RECESSION RISK** (Score: {recession_score}/100)")
        st.markdown("""
        **Recommended Actions:**
        - Monitor indicators closely
        - Reduce position sizes
        - Maintain cash reserves
        - Avoid aggressive strategies
        """)
    else:
        st.success(f"✅ **LOW RECESSION RISK** (Score: {recession_score}/100)")
        st.markdown("""
        **Market Conditions:**
        - Normal market conditions
        - Standard strategies appropriate
        - Maintain diversified portfolio
        """)

    # Signals
    st.markdown("---")
    st.subheader("📊 Technical Signals")

    col1, col2 = st.columns(2)

    with col1:
        if death_cross:
            st.markdown("🔴 **Death Cross** - 50-MA below 200-MA (Bearish)")
        else:
            st.markdown("🟢 **Golden Cross** - 50-MA above 200-MA (Bullish)")

        if latest['Close'] < ma_200:
            st.markdown("🔴 **Below 200-MA** - Bearish trend")
        else:
            st.markdown("🟢 **Above 200-MA** - Bullish trend")

    with col2:
        st.markdown(f"**Stress Score:** {min(100, (volatility * 2 + abs(drawdown)) / 2):.1f}/100")

        if volatility > 25:
            st.markdown("🔴 **High Volatility** - Market stress")
        else:
            st.markdown("🟢 **Normal Volatility**")

    # Price chart
    st.markdown("---")
    st.subheader("📈 Price vs Moving Averages")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recent['Date'], y=recent['Close'],
                             mode='lines', name='SPY Price',
                             line=dict(color='black', width=2)))

    fig.add_trace(go.Scatter(x=recent['Date'], y=recent['Close'].rolling(50).mean(),
                             mode='lines', name='50-Day MA',
                             line=dict(color='blue', width=1, dash='dash')))

    fig.add_trace(go.Scatter(x=recent['Date'], y=recent['Close'].rolling(200).mean(),
                             mode='lines', name='200-Day MA',
                             line=dict(color='red', width=1, dash='dash')))

    fig.update_layout(title="SPY with Moving Averages",
                     xaxis_title="Date",
                     yaxis_title="Price ($)",
                     height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Command to run full analysis
    st.markdown("---")
    st.markdown("**Run Full Analysis:**")
    st.code("python3 recession_indicator.py --save")


def show_valuation_detector():
    """Asset valuation analysis"""
    st.title("💰 Valuation Detector")
    st.markdown("*Identify over/undervalued assets using technical indicators*")

    st.info("💡 **Feature:** Uses RSI, Z-Score, Bollinger Bands, and MA deviation to classify asset valuation")

    # Asset selector
    all_assets = list(STOCK_ETFS.keys()) + list(PRECIOUS_METALS.keys()) + list(CRYPTO_ASSETS.keys())
    downloaded_assets = [a for a in all_assets if check_asset_status(a) == "downloaded"]

    if not downloaded_assets:
        st.warning("⚠️ No assets downloaded. Visit Asset Manager to download data.")
        return

    selected_asset = st.selectbox("Select Asset", downloaded_assets)

    df = load_asset_data(selected_asset)

    if df is None or len(df) < 100:
        st.warning(f"⚠️ Insufficient data for {selected_asset}")
        return

    # Calculate valuation metrics
    recent = df.tail(252)
    latest = recent.iloc[-1]

    # RSI
    delta = recent['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1]

    # Z-Score
    mean_price = recent['Close'].mean()
    std_price = recent['Close'].std()
    z_score = (latest['Close'] - mean_price) / std_price if std_price > 0 else 0

    # Bollinger Bands
    sma_20 = recent['Close'].rolling(20).mean().iloc[-1]
    std_20 = recent['Close'].rolling(20).std().iloc[-1]
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_position = ((latest['Close'] - lower_band) / (upper_band - lower_band)) * 100 if upper_band > lower_band else 50

    # MA deviation
    ma_200 = recent['Close'].rolling(200).mean().iloc[-1] if len(recent) >= 200 else recent['Close'].mean()
    ma_deviation = ((latest['Close'] - ma_200) / ma_200) * 100

    # Valuation score
    valuation_score = 0
    if latest_rsi > 70:
        valuation_score += 0.3
    elif latest_rsi < 30:
        valuation_score -= 0.3

    if z_score > 2:
        valuation_score += 0.3
    elif z_score < -2:
        valuation_score -= 0.3

    if bb_position > 80:
        valuation_score += 0.2
    elif bb_position < 20:
        valuation_score -= 0.2

    if ma_deviation > 20:
        valuation_score += 0.2
    elif ma_deviation < -20:
        valuation_score -= 0.2

    # Classification
    if valuation_score > 0.5:
        classification = "OVERVALUED"
        color = "error"
    elif valuation_score < -0.5:
        classification = "UNDERVALUED"
        color = "success"
    else:
        classification = "FAIRLY VALUED"
        color = "info"

    # Display classification
    st.markdown("---")
    if color == "error":
        st.error(f"### 🔴 {classification}")
        st.markdown("**Recommendation:** Monitor for exit opportunities, consider taking profits")
    elif color == "success":
        st.success(f"### 🟢 {classification}")
        st.markdown("**Recommendation:** Consider accumulating or opening positions")
    else:
        st.info(f"### 🟡 {classification}")
        st.markdown("**Recommendation:** Hold current positions, wait for better entry/exit")

    st.markdown(f"**Valuation Score:** {valuation_score:.2f} (Range: -1.0 to +1.0)")

    # Metrics
    st.markdown("---")
    st.subheader("📊 Technical Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RSI (14)", f"{latest_rsi:.1f}")
        if latest_rsi > 70:
            st.markdown("🔴 Overbought (>70)")
        elif latest_rsi < 30:
            st.markdown("🟢 Oversold (<30)")
        else:
            st.markdown("🟡 Neutral (30-70)")

    with col2:
        st.metric("Z-Score", f"{z_score:.2f}")
        if z_score > 2:
            st.markdown("🔴 Expensive (>2)")
        elif z_score < -2:
            st.markdown("🟢 Cheap (<-2)")
        else:
            st.markdown("🟡 Normal (-2 to 2)")

    with col3:
        st.metric("Bollinger %", f"{bb_position:.1f}%")
        if bb_position > 80:
            st.markdown("🔴 Upper Band")
        elif bb_position < 20:
            st.markdown("🟢 Lower Band")
        else:
            st.markdown("🟡 Middle Range")

    # Additional metrics
    col4, col5 = st.columns(2)

    with col4:
        st.metric("200-MA Deviation", f"{ma_deviation:+.1f}%")

    with col5:
        roc_30 = ((latest['Close'] - recent.iloc[-30]['Close']) / recent.iloc[-30]['Close']) * 100 if len(recent) >= 30 else 0
        st.metric("30-Day ROC", f"{roc_30:+.1f}%")

    # Chart
    st.markdown("---")
    st.subheader("📈 Price with Bollinger Bands")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recent['Date'], y=recent['Close'],
                             mode='lines', name='Price',
                             line=dict(color='black', width=2)))

    # Bollinger bands
    sma = recent['Close'].rolling(20).mean()
    std = recent['Close'].rolling(20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)

    fig.add_trace(go.Scatter(x=recent['Date'], y=upper,
                             mode='lines', name='Upper BB',
                             line=dict(color='red', width=1, dash='dash')))

    fig.add_trace(go.Scatter(x=recent['Date'], y=sma,
                             mode='lines', name='SMA(20)',
                             line=dict(color='blue', width=1)))

    fig.add_trace(go.Scatter(x=recent['Date'], y=lower,
                             mode='lines', name='Lower BB',
                             line=dict(color='green', width=1, dash='dash')))

    fig.update_layout(title=f"{selected_asset} Valuation Analysis",
                     xaxis_title="Date",
                     yaxis_title="Price ($)",
                     height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Command
    st.markdown("---")
    st.markdown("**Run Valuation Analysis:**")
    st.code(f"python3 valuation_detector.py --asset {selected_asset}")


def show_llm_analysis():
    """LLM-powered market analysis"""
    st.title("🤖 LLM Market Analysis")
    st.markdown("*AI-powered market commentary using GPT-4 or Claude*")

    st.info("💡 **Feature:** Integrates with OpenAI/Anthropic to generate market analysis based on predictions, price data, and news")

    # Configuration
    st.subheader("⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Required Environment Variables:**")
        st.code("""
OPENAI_API_KEY=sk-your-key-here
# Or
ANTHROPIC_API_KEY=sk-ant-your-key-here

# For news integration
NEWS_API_KEY=your-newsapi-key
        """)

    with col2:
        st.markdown("**Providers Supported:**")
        st.markdown("- OpenAI GPT-4")
        st.markdown("- Anthropic Claude")
        st.markdown("- NewsAPI integration")

    # Sample analysis
    st.markdown("---")
    st.subheader("📊 Sample Analysis Output")

    # Load latest prediction
    pred_file = LOGS_DIR / "labeled_predictions.csv"
    if pred_file.exists():
        preds = pd.read_csv(pred_file)
        if len(preds) > 0:
            latest = preds.iloc[-1]

            signal_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}
            signal = signal_map.get(latest.get('Prediction', 1), 'NORMAL')
            confidence = latest.get('Confidence', 0.5) * 100 if pd.notna(latest.get('Confidence')) else 50

            st.markdown(f"""
            **Asset:** SPY
            **Date:** {latest.get('Date', 'N/A')}
            **Signal:** {signal}
            **Confidence:** {confidence:.1f}%
            """)

            if signal == 'SPIKE':
                st.success(f"""
**AI Analysis Example:**

The model shows strong bullish momentum with high conviction. Current market conditions favor upside movement.

**Key Factors:**
- Model confidence: {confidence:.0f}%
- Technical setup supports continuation
- Risk-reward favors long positioning

**Scenario Likelihoods:**
- CRASH (Bearish): 15% - Minor downside risk
- NORMAL (Neutral): 20% - Consolidation possible
- SPIKE (Bullish): 65% - Primary scenario

**Recommendation:**
Monitor for entry opportunities with appropriate risk management.
                """)
            elif signal == 'CRASH':
                st.error(f"""
**AI Analysis Example:**

The model indicates significant downside risk with {confidence:.0f}% confidence. Defensive positioning recommended.

**Key Factors:**
- High crash probability
- Technical weakness present
- Risk management priority

**Scenario Likelihoods:**
- CRASH (Bearish): 60% - Primary concern
- NORMAL (Neutral): 25% - Some support
- SPIKE (Bullish): 15% - Low probability

**Recommendation:**
Reduce exposure, raise cash levels, consider hedging strategies.
                """)
            else:
                st.info(f"""
**AI Analysis Example:**

Neutral forecast suggests mixed signals. Market in consolidation phase.

**Recommendation:**
Wait for clearer directional bias before taking action.
                """)

    # Commands
    st.markdown("---")
    st.subheader("💻 Run LLM Analysis")

    st.code("""
# Single asset analysis with OpenAI
python3 llm_forecast.py --asset SPY --provider openai

# Multi-asset summary with Anthropic
python3 llm_forecast.py --all --provider anthropic

# Generate newsletter
python3 newsletter_generator.py --send --assets SPY,BTC/USDT
    """)


def show_portfolio_rebalancing():
    """Portfolio rebalancing optimizer"""
    st.title("🔄 Portfolio Rebalancing Optimizer")
    st.markdown("*Find optimal rebalancing frequency for your portfolio*")

    st.info("💡 **Feature:** Tests different rebalancing frequencies (daily, weekly, monthly, etc.) and identifies the optimal strategy based on returns, Sharpe ratio, and transaction costs")

    st.markdown("""
    ### How It Works

    The optimizer tests multiple rebalancing strategies:
    - **Daily** - Rebalance every trading day
    - **Weekly** - Rebalance weekly
    - **Monthly** - Rebalance monthly (often optimal)
    - **Quarterly** - Rebalance every 3 months
    - **Semi-Annual** - Rebalance twice per year
    - **Annual** - Rebalance once per year
    - **Buy & Hold** - Never rebalance (baseline)

    It accounts for:
    - Transaction costs (customizable)
    - Sharpe ratio (risk-adjusted returns)
    - Maximum drawdown
    - Total net returns
    """)

    # Example results
    st.markdown("---")
    st.subheader("📊 Example Results")

    example_data = {
        'Strategy': ['Buy & Hold', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annual', 'Annual'],
        'Gross Return': ['45.2%', '52.1%', '51.3%', '50.8%', '48.5%', '46.8%', '45.9%'],
        'Sharpe Ratio': [1.85, 1.92, 1.95, 2.10, 2.08, 1.98, 1.88],
        'Max Drawdown': ['-8.2%', '-7.1%', '-7.3%', '-7.0%', '-7.5%', '-8.0%', '-8.3%'],
        'Rebalances': [0, 1260, 260, 60, 20, 10, 5],
        'Net Return': ['45.2%', '27.6%', '41.9%', '49.6%', '48.1%', '46.5%', '45.7%'],
        'Recommended': ['', '', '', '✅', '', '', '']
    }

    df_example = pd.DataFrame(example_data)
    st.dataframe(df_example, use_container_width=True)

    st.success("**Optimal Strategy:** Monthly rebalancing provides best risk-adjusted returns")

    # Commands
    st.markdown("---")
    st.subheader("💻 Run Rebalancing Analysis")

    st.code("""
# Find optimal rebalancing period
python3 portfolio_rebalancer.py --find-optimal \\
    --assets SPY,GLD,TLT --weights 0.6,0.3,0.1

# Execute rebalancing with specific profile
python3 portfolio_rebalancer.py \\
    --assets SPY,GLD,TLT \\
    --weights 0.6,0.3,0.1 \\
    --profile moderate
    """)


def show_forecast_results():
    """Display forecast results"""
    st.title("📈 Forecast Results")
    st.markdown("*View predictions generated by the forecasting API*")

    # Load predictions
    pred_file = LOGS_DIR / "labeled_predictions.csv"

    if not pred_file.exists():
        st.warning("⚠️ No forecast results available. Generate predictions first:")
        st.code("python3 predict_multi_asset_ensemble.py")
        return

    df = pd.read_csv(pred_file)

    if len(df) == 0:
        st.info("No predictions found")
        return

    # Signal distribution
    st.subheader("📊 Signal Distribution")

    if 'Prediction' in df.columns:
        signal_counts = df['Prediction'].value_counts().sort_index()

        col1, col2, col3 = st.columns(3)

        total = len(df)
        crash_count = signal_counts.get(0, 0)
        normal_count = signal_counts.get(1, 0)
        spike_count = signal_counts.get(2, 0)

        with col1:
            st.metric("CRASH Signals", f"{crash_count:,}", delta=f"{(crash_count/total)*100:.1f}%")

        with col2:
            st.metric("NORMAL Signals", f"{normal_count:,}", delta=f"{(normal_count/total)*100:.1f}%")

        with col3:
            st.metric("SPIKE Signals", f"{spike_count:,}", delta=f"{(spike_count/total)*100:.1f}%")

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['CRASH', 'NORMAL', 'SPIKE'],
            values=[crash_count, normal_count, spike_count],
            marker=dict(colors=['red', 'yellow', 'green'])
        )])
        fig.update_layout(title="Signal Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Recent predictions
    st.markdown("---")
    st.subheader("📅 Recent Forecasts (Last 20)")

    recent = df.tail(20).copy()
    if 'Prediction' in recent.columns:
        recent['Signal'] = recent['Prediction'].map({0: '🔴 CRASH', 1: '🟡 NORMAL', 2: '🟢 SPIKE'})

    display_cols = ['Date', 'Signal', 'Proba', 'Confidence']
    available_cols = [col for col in display_cols if col in recent.columns]

    if available_cols:
        st.dataframe(recent[available_cols].sort_values('Date', ascending=False), use_container_width=True)


def show_automation():
    """Automation and pipeline execution"""
    st.title("🚀 Automation & Pipeline")
    st.markdown("*Automate data downloads, training, and forecasting*")

    st.markdown("""
    ### Full Pipeline

    Run the complete NeuroVest workflow automatically:
    1. Download/update data (SPY + crypto)
    2. Train models with optimized weights
    3. Generate predictions
    4. Run backtests
    5. Generate LLM analysis (optional)

    **Estimated Time:** 20-35 minutes
    """)

    # Full pipeline command
    st.subheader("🚀 Run Complete Pipeline")

    st.code("""
python3 main.py
# Then select "Option R" - Run Full Pipeline
    """)

    st.markdown("---")

    # Individual commands
    st.subheader("🔧 Individual Commands")

    tab1, tab2, tab3, tab4 = st.tabs(["📥 Download Data", "🤖 Train Models", "📈 Generate Forecasts", "📊 Validate"])

    with tab1:
        st.markdown("**Download Commands:**")
        st.code("""
# Update SPY data
python3 update_spy_data.py

# Download all stock ETFs
python3 download_equity_etfs.py

# Download all crypto
python3 download_crypto_enhanced.py

# Download specific precious metal
python3 framework/download_all_assets.py --asset GLD
        """)

    with tab2:
        st.markdown("**Training Commands:**")
        st.code("""
# Standard training
python3 train_multi_asset.py

# With optimized weights (RECOMMENDED)
python3 train_multi_asset.py --optimize-weights

# With hyperparameter tuning
python3 train_multi_asset.py --tune

# With feature selection
python3 train_multi_asset.py --optimize-weights --feature-select
        """)

    with tab3:
        st.markdown("**Prediction Commands:**")
        st.code("""
# Generate ensemble predictions
python3 predict_multi_asset_ensemble.py

# Generate per-asset predictions
python3 predict_per_asset.py --all

# Specific asset
python3 predict_per_asset.py --asset BTC/USDT
        """)

    with tab4:
        st.markdown("**Validation Commands:**")
        st.code("""
# Extract performance metrics
python3 extract_metrics.py --comprehensive

# Validate signal quality
python3 validate_signals.py --detailed

# Run backtest
python3 backtest.py
        """)


def show_custom_imports():
    """Custom data import interface"""
    st.title("📁 Custom Data Imports")
    st.markdown("*Import your own asset data for forecasting*")

    st.markdown("""
    Import custom CSV or Excel files to generate forecasts for any asset.

    **Required Columns:**
    - Date (or Time, Timestamp)
    - Close (or Price)

    **Optional Columns:**
    - Open, High, Low
    - Volume
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {list(df.columns)}")

            # Import settings
            st.subheader("⚙️ Import Settings")

            col1, col2 = st.columns(2)

            with col1:
                ticker = st.text_input("Ticker Symbol", value="CUSTOM")

            with col2:
                asset_type = st.selectbox("Asset Type", ["Stock", "ETF", "Crypto", "Other"])

            if st.button("✅ Import Data", type="primary"):
                CACHE_DIR.mkdir(exist_ok=True)
                safe_ticker = ticker.replace('/', '_')
                save_path = CACHE_DIR / f"{safe_ticker}_1d.csv"

                df.to_csv(save_path, index=False)

                st.success(f"✅ Successfully imported as {ticker}")
                st.info(f"📁 Saved to: {save_path}")
                st.markdown("**Next Steps:**")
                st.code(f"""
# Train model for this asset
python3 train_per_asset.py --asset {ticker}

# Generate predictions
python3 predict_per_asset.py --asset {ticker}
                """)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # CLI import
    st.markdown("---")
    st.subheader("💻 Command Line Import")

    st.code("""
# Import with validation
python3 import_custom_asset.py my_data.csv MYTICKER

# Create sample template
python3 import_custom_asset.py --sample
    """)


if __name__ == "__main__":
    main()
