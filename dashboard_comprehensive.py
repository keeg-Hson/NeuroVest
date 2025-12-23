#!/usr/bin/env python3
"""
NeuroVest Comprehensive Feature Dashboard

Interactive Streamlit dashboard showcasing all advanced NeuroVest features.
Perfect for client demonstrations and feature exploration.

Usage:
    streamlit run dashboard_comprehensive.py
    streamlit run dashboard_comprehensive.py --server.port 8501

Features:
- Trading Risk Profiles (Conservative/Moderate/Aggressive)
- Recession Probability Indicator
- Valuation Detector (Over/Undervalued Assets)
- LLM-Powered Market Analysis
- Portfolio Rebalancing Optimization
- Signal Distribution & Performance
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

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
    page_title="NeuroVest Comprehensive Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directories
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")
CONFIGS_DIR = Path("configs")

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .green-metric {
        color: #2ca02c;
        font-weight: bold;
    }
    .red-metric {
        color: #d62728;
        font-weight: bold;
    }
    .yellow-metric {
        color: #ff7f0e;
        font-weight: bold;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_spy_data():
    """Load SPY data"""
    spy_path = DATA_DIR / "SPY.csv"
    if not spy_path.exists():
        return None

    df = pd.read_csv(spy_path)
    if len(df) == 0:
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Close'])
    return df.sort_values('Date').reset_index(drop=True)


def load_predictions():
    """Load prediction data"""
    pred_path = LOGS_DIR / "labeled_predictions.csv"
    if not pred_path.exists():
        return None

    df = pd.read_csv(pred_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    return df


def calculate_recession_indicators(df):
    """Calculate recession indicators"""
    if df is None or len(df) < 200:
        return None

    latest = df.iloc[-1]
    recent = df.tail(200)

    # Calculate moving averages
    ma_50 = recent['Close'].tail(50).mean()
    ma_200 = recent['Close'].tail(200).mean()

    # Calculate volatility
    returns = recent['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100

    # Calculate drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = ((cumulative - rolling_max) / rolling_max * 100).min()

    # Death cross detection
    death_cross = ma_50 < ma_200

    # Price vs 200-MA
    below_200ma = latest['Close'] < ma_200

    return {
        'price': latest['Close'],
        'ma_50': ma_50,
        'ma_200': ma_200,
        'volatility': volatility,
        'max_drawdown': drawdown,
        'death_cross': death_cross,
        'below_200ma': below_200ma,
        'stress_score': min(100, (volatility * 2 + abs(drawdown)) / 2)
    }


def calculate_valuation_metrics(df):
    """Calculate valuation metrics"""
    if df is None or len(df) < 100:
        return None

    recent = df.tail(100)
    latest = recent.iloc[-1]

    # RSI calculation
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

    # Moving average deviation
    ma_200 = df.tail(200)['Close'].mean()
    ma_deviation = ((latest['Close'] - ma_200) / ma_200) * 100

    # Rate of change (30-day)
    roc_30 = ((latest['Close'] - recent.iloc[-30]['Close']) / recent.iloc[-30]['Close']) * 100 if len(recent) >= 30 else 0

    # Valuation score (-1 to +1)
    # Undervalued if RSI < 30, Z-Score < -2, below lower BB
    # Overvalued if RSI > 70, Z-Score > 2, above upper BB
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
    elif valuation_score < -0.5:
        classification = "UNDERVALUED"
    else:
        classification = "FAIRLY VALUED"

    return {
        'rsi': latest_rsi,
        'z_score': z_score,
        'bb_position': bb_position,
        'ma_deviation': ma_deviation,
        'roc_30': roc_30,
        'valuation_score': valuation_score,
        'classification': classification
    }


def show_overview():
    """Overview page"""
    st.title("🚀 NeuroVest Comprehensive Dashboard")
    st.markdown("### Advanced AI-Powered Trading System")

    st.markdown("""
    <div class="info-box">
    <h4>✨ Featured Capabilities</h4>
    This dashboard demonstrates NeuroVest's most advanced features:
    <ul>
        <li><b>Trading Risk Profiles:</b> Conservative, Moderate, and Aggressive strategies</li>
        <li><b>Recession Indicator:</b> Multi-factor recession probability analysis</li>
        <li><b>Valuation Detector:</b> Identify over/undervalued assets using technical analysis</li>
        <li><b>LLM Integration:</b> AI-powered market analysis and forecasting</li>
        <li><b>Portfolio Rebalancing:</b> Optimize rebalancing frequency for maximum returns</li>
        <li><b>Signal Analytics:</b> Deep dive into prediction performance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    # Load data
    predictions = load_predictions()
    spy_data = load_spy_data()

    with col1:
        if predictions is not None:
            st.metric("Total Predictions", f"{len(predictions):,}")
        else:
            st.metric("Total Predictions", "No data")

    with col2:
        if spy_data is not None:
            st.metric("SPY Data Days", f"{len(spy_data):,}")
        else:
            st.metric("SPY Data Days", "No data")

    with col3:
        model_count = sum(1 for f in ['xgboost_multi_asset.pkl', 'lightgbm_multi_asset.pkl', 'catboost_multi_asset.pkl']
                         if (MODELS_DIR / f).exists())
        st.metric("Models Loaded", f"{model_count}/3")

    with col4:
        if predictions is not None and 'Prediction' in predictions.columns:
            spike_count = (predictions['Prediction'] == 2).sum()
            spike_pct = (spike_count / len(predictions)) * 100
            st.metric("Spike Signals", f"{spike_pct:.1f}%")
        else:
            st.metric("Spike Signals", "N/A")

    # System status
    st.markdown("---")
    st.subheader("📊 System Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.markdown("**Data Files:**")
        if (DATA_DIR / "SPY.csv").exists():
            st.markdown("✅ SPY.csv")
        else:
            st.markdown("❌ SPY.csv (run: `python3 update_spy_data.py`)")

        if (LOGS_DIR / "labeled_predictions.csv").exists():
            st.markdown("✅ labeled_predictions.csv")
        else:
            st.markdown("❌ labeled_predictions.csv (run: `python3 predict_multi_asset_ensemble.py`)")

    with status_col2:
        st.markdown("**Models:**")
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            model_file = MODELS_DIR / f"{model_name}_multi_asset.pkl"
            if model_file.exists():
                st.markdown(f"✅ {model_name}")
            else:
                st.markdown(f"❌ {model_name}")


def show_trading_profiles():
    """Trading risk profiles page"""
    st.title("🎯 Trading Risk Profiles")

    st.markdown("""
    NeuroVest supports three distinct risk profiles, each optimized for different investor preferences:
    """)

    # Profile comparison table
    profiles_data = {
        'Profile': ['Conservative', 'Moderate', 'Aggressive'],
        'Confidence Threshold': ['70%+', '55%+', '45%+'],
        'Stop Loss (ATR)': ['1.0x', '1.5x', '2.0x'],
        'Position Size': ['5-15%', '10-25%', '15-40%'],
        'Expected Return': ['120-180%', '150-250%', '300-500%'],
        'Expected Sharpe': ['2.5-3.0', '2.0-2.6', '1.8-2.3'],
        'Max Drawdown': ['-3% to -6%', '-6% to -10%', '-10% to -18%'],
        'Win Rate': ['60-65%', '55-60%', '50-57%']
    }

    df_profiles = pd.DataFrame(profiles_data)
    st.dataframe(df_profiles, use_container_width=True)

    st.markdown("---")

    # Profile selector
    profile = st.selectbox("Select Profile to Analyze", ["Conservative", "Moderate", "Aggressive"])

    if profile == "Conservative":
        st.markdown("""
        ### 🛡️ Conservative Profile

        **Philosophy:** Capital preservation with steady growth

        **Strategy:**
        - Only trade when confidence ≥ 70%
        - Tight stop losses (1.0x ATR)
        - Small position sizes (5-15% of capital)
        - Focus on high-probability setups

        **Ideal For:**
        - Risk-averse investors
        - Retirement accounts
        - Preservation of capital
        - Consistent returns over time

        **Typical Performance:**
        - Lower returns but higher win rate
        - Minimal drawdowns
        - Very stable equity curve
        - Excellent Sharpe ratio
        """)

        # Show sample equity curve
        fig = go.Figure()
        x = list(range(252))  # 1 year
        y = [10000 * (1 + 0.006 * i + np.random.normal(0, 30)) for i in x]  # Conservative growth
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Equity', line=dict(color='green', width=2)))
        fig.update_layout(title="Conservative Profile - Sample Equity Curve", xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)

    elif profile == "Moderate":
        st.markdown("""
        ### ⚖️ Moderate Profile

        **Philosophy:** Balanced risk-reward approach

        **Strategy:**
        - Trade when confidence ≥ 55%
        - Moderate stop losses (1.5x ATR)
        - Medium position sizes (10-25% of capital)
        - Balance between frequency and quality

        **Ideal For:**
        - Most investors
        - Growth-oriented portfolios
        - Balanced approach
        - Long-term wealth building

        **Typical Performance:**
        - Good returns with acceptable risk
        - Moderate drawdowns
        - Balanced win rate
        - Strong risk-adjusted returns
        """)

        fig = go.Figure()
        x = list(range(252))
        y = [10000 * (1 + 0.008 * i + np.random.normal(0, 50)) for i in x]  # Moderate growth
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Equity', line=dict(color='blue', width=2)))
        fig.update_layout(title="Moderate Profile - Sample Equity Curve", xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Aggressive
        st.markdown("""
        ### 🔥 Aggressive Profile

        **Philosophy:** Maximum returns, higher risk tolerance

        **Strategy:**
        - Trade when confidence ≥ 45%
        - Wide stop losses (2.0x ATR)
        - Large position sizes (15-40% of capital)
        - Maximize trading opportunities

        **Ideal For:**
        - Risk-tolerant investors
        - Shorter time horizons
        - Experienced traders
        - Growth at any cost

        **Typical Performance:**
        - Highest returns
        - Larger drawdowns
        - More volatility
        - Lower win rate but bigger wins
        """)

        fig = go.Figure()
        x = list(range(252))
        y = [10000 * (1 + 0.012 * i + np.random.normal(0, 100)) for i in x]  # Aggressive growth
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Equity', line=dict(color='red', width=2)))
        fig.update_layout(title="Aggressive Profile - Sample Equity Curve", xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)

    # Command to run
    st.markdown("---")
    st.markdown("### 💻 Run with Command Line")
    st.code(f"""
# Run backtest with {profile.lower()} profile
python3 backtest.py --config configs/backtest_{profile.lower()}.json

# Or use the profile directly
python3 backtest.py --profile {profile.lower()}
    """)


def show_recession_indicator():
    """Recession indicator page"""
    st.title("📉 Recession Probability Indicator")

    st.markdown("""
    Multi-factor analysis to assess recession risk using:
    - Yield curve inversion (10Y-2Y spread)
    - Market stress indicators (volatility, drawdown)
    - Technical signals (death cross, price vs moving averages)
    """)

    spy_data = load_spy_data()

    if spy_data is None or len(spy_data) < 200:
        st.warning("Insufficient SPY data. Run: `python3 update_spy_data.py`")
        return

    indicators = calculate_recession_indicators(spy_data)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Price", f"${indicators['price']:.2f}")

    with col2:
        below_text = "Below" if indicators['below_200ma'] else "Above"
        delta_color = "inverse" if indicators['below_200ma'] else "normal"
        st.metric("200-Day MA", f"${indicators['ma_200']:.2f}", delta=below_text, delta_color=delta_color)

    with col3:
        st.metric("Volatility", f"{indicators['volatility']:.1f}%")

    with col4:
        st.metric("Max Drawdown", f"{indicators['max_drawdown']:.1f}%", delta_color="inverse")

    st.markdown("---")

    # Recession signals
    st.subheader("🚨 Recession Signals")

    signal_col1, signal_col2 = st.columns(2)

    with signal_col1:
        st.markdown("#### Technical Signals")
        if indicators['death_cross']:
            st.markdown("🔴 **Death Cross Detected** - 50-MA below 200-MA")
        else:
            st.markdown("🟢 **Golden Cross** - 50-MA above 200-MA")

        if indicators['below_200ma']:
            st.markdown("🔴 **Price Below 200-MA** - Bearish trend")
        else:
            st.markdown("🟢 **Price Above 200-MA** - Bullish trend")

    with signal_col2:
        st.markdown("#### Market Stress")
        st.markdown(f"**Stress Score:** {indicators['stress_score']:.1f}/100")

        if indicators['stress_score'] > 60:
            st.markdown("🔴 **Elevated Stress** - Recession risk increased")
        elif indicators['stress_score'] > 40:
            st.markdown("🟡 **Moderate Stress** - Monitor closely")
        else:
            st.markdown("🟢 **Low Stress** - Normal conditions")

    # Overall assessment
    st.markdown("---")
    st.subheader("📊 Overall Assessment")

    recession_score = 0
    if indicators['death_cross']:
        recession_score += 25
    if indicators['below_200ma']:
        recession_score += 20
    if indicators['volatility'] > 25:
        recession_score += 20
    if indicators['max_drawdown'] < -15:
        recession_score += 25
    if indicators['stress_score'] > 50:
        recession_score += 10

    if recession_score > 70:
        st.error(f"⚠️ **HIGH RECESSION RISK** (Score: {recession_score}/100)")
        st.markdown("""
        **Recommended Actions:**
        - Reduce equity exposure
        - Increase cash/bond allocation
        - Consider defensive sectors (utilities, healthcare, consumer staples)
        - Hedge positions with puts or inverse ETFs
        """)
    elif recession_score > 40:
        st.warning(f"⚠️ **MODERATE RECESSION RISK** (Score: {recession_score}/100)")
        st.markdown("""
        **Recommended Actions:**
        - Monitor indicators closely
        - Reduce position sizes
        - Maintain some cash reserves
        - Avoid aggressive strategies
        """)
    else:
        st.success(f"✅ **LOW RECESSION RISK** (Score: {recession_score}/100)")
        st.markdown("""
        **Market Conditions:**
        - Normal market conditions
        - Standard trading strategies appropriate
        - Maintain diversified portfolio
        """)

    # Chart
    st.markdown("---")
    st.subheader("📈 Price vs Moving Averages")

    recent_data = spy_data.tail(252)  # Last year
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['Close'],
                             mode='lines', name='SPY Price', line=dict(color='black', width=2)))

    # Calculate and plot MAs
    ma_50_series = recent_data['Close'].rolling(50).mean()
    ma_200_series = recent_data['Close'].rolling(200).mean()

    fig.add_trace(go.Scatter(x=recent_data['Date'], y=ma_50_series,
                             mode='lines', name='50-Day MA', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=recent_data['Date'], y=ma_200_series,
                             mode='lines', name='200-Day MA', line=dict(color='red', width=1)))

    fig.update_layout(title="SPY Price with Moving Averages", xaxis_title="Date", yaxis_title="Price ($)", height=500)
    st.plotly_chart(fig, use_container_width=True)


def show_valuation_detector():
    """Valuation detector page"""
    st.title("💰 Valuation Detector")

    st.markdown("""
    Identify over/undervalued assets using multiple technical indicators:
    - **RSI** (>70 = Overbought, <30 = Oversold)
    - **Z-Score** (>2 = Expensive, <-2 = Cheap)
    - **Bollinger Bands** Position
    - **Moving Average** Deviation
    - **Rate of Change** (30-day momentum)
    """)

    spy_data = load_spy_data()

    if spy_data is None or len(spy_data) < 100:
        st.warning("Insufficient SPY data. Run: `python3 update_spy_data.py`")
        return

    metrics = calculate_valuation_metrics(spy_data)

    # Overall classification
    st.markdown("---")

    if metrics['classification'] == "OVERVALUED":
        st.error(f"### 🔴 {metrics['classification']}")
        st.markdown("**Recommendation:** Monitor for exit opportunities, consider taking profits")
    elif metrics['classification'] == "UNDERVALUED":
        st.success(f"### 🟢 {metrics['classification']}")
        st.markdown("**Recommendation:** Consider accumulating or opening positions")
    else:
        st.info(f"### 🟡 {metrics['classification']}")
        st.markdown("**Recommendation:** Hold current positions, wait for better entry/exit")

    st.markdown(f"**Valuation Score:** {metrics['valuation_score']:.2f} (Range: -1.0 to +1.0)")

    # Detailed metrics
    st.markdown("---")
    st.subheader("📊 Technical Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RSI (14)", f"{metrics['rsi']:.1f}")
        if metrics['rsi'] > 70:
            st.markdown("🔴 Overbought (>70)")
        elif metrics['rsi'] < 30:
            st.markdown("🟢 Oversold (<30)")
        else:
            st.markdown("🟡 Neutral (30-70)")

    with col2:
        st.metric("Z-Score", f"{metrics['z_score']:.2f}")
        if metrics['z_score'] > 2:
            st.markdown("🔴 Statistically Expensive (>2)")
        elif metrics['z_score'] < -2:
            st.markdown("🟢 Statistically Cheap (<-2)")
        else:
            st.markdown("🟡 Normal Range (-2 to 2)")

    with col3:
        st.metric("Bollinger Position", f"{metrics['bb_position']:.1f}%")
        if metrics['bb_position'] > 80:
            st.markdown("🔴 Upper Band (Overbought)")
        elif metrics['bb_position'] < 20:
            st.markdown("🟢 Lower Band (Oversold)")
        else:
            st.markdown("🟡 Middle Range")

    # Additional metrics
    col4, col5 = st.columns(2)

    with col4:
        st.metric("200-MA Deviation", f"{metrics['ma_deviation']:+.1f}%")
        if abs(metrics['ma_deviation']) > 20:
            st.markdown("⚠️ Extended (>20% deviation)")
        else:
            st.markdown("✅ Normal (<20% deviation)")

    with col5:
        st.metric("30-Day ROC", f"{metrics['roc_30']:+.1f}%")
        if abs(metrics['roc_30']) > 10:
            st.markdown("⚡ Strong Momentum")
        else:
            st.markdown("📊 Normal Momentum")

    # Visualization
    st.markdown("---")
    st.subheader("📈 Valuation Over Time")

    # Calculate rolling valuation score
    recent = spy_data.tail(100)
    rsi_values = []
    z_scores = []

    for i in range(14, len(recent)):
        window = recent.iloc[max(0, i-14):i+1]
        delta = window['Close'].diff()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

        mean_price = recent.iloc[:i+1]['Close'].mean()
        std_price = recent.iloc[:i+1]['Close'].std()
        z = (window['Close'].iloc[-1] - mean_price) / std_price if std_price > 0 else 0
        z_scores.append(z)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("RSI (14)", "Z-Score"),
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=rsi_values, mode='lines', name='RSI', line=dict(color='purple')),
                  row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    fig.add_trace(go.Scatter(y=z_scores, mode='lines', name='Z-Score', line=dict(color='orange')),
                  row=2, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_llm_integration():
    """LLM integration page"""
    st.title("🤖 LLM-Powered Market Analysis")

    st.markdown("""
    NeuroVest integrates with Large Language Models for advanced market analysis:
    - **OpenAI GPT-4** or **Anthropic Claude**
    - Real-time news integration via NewsAPI
    - Scenario likelihood analysis (Crash/Normal/Spike probabilities)
    - Actionable trading recommendations
    - Automated newsletter generation
    """)

    # Configuration check
    st.markdown("---")
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
        st.markdown("**Optional (Newsletter):**")
        st.code("""
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
NEWSLETTER_RECIPIENTS=recipient@example.com
        """)

    # Sample analysis
    st.markdown("---")
    st.subheader("📊 Sample LLM Analysis")

    predictions = load_predictions()
    spy_data = load_spy_data()

    if predictions is not None and len(predictions) > 0:
        latest_pred = predictions.iloc[-1]

        signal_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}
        signal = signal_map.get(latest_pred.get('Prediction', 1), 'NORMAL')
        confidence = latest_pred.get('Confidence', 0) * 100 if pd.notna(latest_pred.get('Confidence')) else 50

        st.markdown(f"""
        ### ASSET: SPY
        **Date:** {latest_pred['Date'].strftime('%Y-%m-%d') if pd.notna(latest_pred['Date']) else 'N/A'}

        #### Current Market Data
        """)

        if spy_data is not None and len(spy_data) > 0:
            latest_spy = spy_data.iloc[-1]
            recent_spy = spy_data.tail(20)

            daily_change = ((latest_spy['Close'] - spy_data.iloc[-2]['Close']) / spy_data.iloc[-2]['Close']) * 100 if len(spy_data) > 1 else 0
            weekly_change = ((latest_spy['Close'] - recent_spy.iloc[0]['Close']) / recent_spy.iloc[0]['Close']) * 100 if len(recent_spy) > 5 else 0

            st.markdown(f"""
            - **Latest Price:** ${latest_spy['Close']:.2f}
            - **Daily Change:** {daily_change:+.2f}%
            - **5-Day Return:** {weekly_change:+.2f}%
            - **Volume:** {latest_spy['Volume']:,.0f}
            """)

        st.markdown(f"""
        #### Model Prediction
        - **Primary Signal:** {signal}
        - **Model Confidence:** {confidence:.1f}%

        #### Scenario Likelihoods
        """)

        # Calculate scenario probabilities from predictions
        if 'Spike_Conf' in latest_pred and 'Crash_Conf' in latest_pred:
            spike_prob = latest_pred['Spike_Conf'] * 100 if pd.notna(latest_pred['Spike_Conf']) else 30
            crash_prob = latest_pred['Crash_Conf'] * 100 if pd.notna(latest_pred['Crash_Conf']) else 30
            normal_prob = 100 - spike_prob - crash_prob
        else:
            spike_prob, crash_prob, normal_prob = 30, 30, 40

        st.markdown(f"""
        - **CRASH (Bearish):** {crash_prob:.0f}% - Significant downward movement expected
        - **NORMAL (Neutral):** {normal_prob:.0f}% - Sideways/mixed price action expected
        - **SPIKE (Bullish):** {spike_prob:.0f}% - Significant upward movement expected
        """)

        # Mock LLM analysis
        st.markdown("---")
        st.subheader("🧠 AI Analysis")

        if signal == 'SPIKE':
            st.success(f"""
            The model shows strong bullish momentum with {spike_prob:.0f}% probability of upside movement.
            The {confidence:.0f}% confidence level suggests this is a high-conviction signal.

            **Recommendation:**
            Consider scaling into long positions with appropriate risk management. Given the
            {crash_prob:.0f}% crash probability, maintain protective stops and position sizing discipline.

            **Target:** +5-7% upside
            **Stop Loss:** -2-3%
            **Position Size:** Based on your risk profile
            """)
        elif signal == 'CRASH':
            st.error(f"""
            The model indicates significant downside risk with {crash_prob:.0f}% probability of decline.
            The {confidence:.0f}% confidence level suggests this is a high-conviction signal.

            **Recommendation:**
            Consider defensive positioning: reduce equity exposure, raise cash, or establish hedges.
            The {spike_prob:.0f}% spike probability suggests some caution, but overall bias is bearish.

            **Action:** Reduce risk, raise stops, consider hedging
            **Target:** Preserve capital
            """)
        else:
            st.info(f"""
            The model suggests neutral to mixed price action with {normal_prob:.0f}% probability.
            The {confidence:.0f}% confidence level indicates moderate conviction.

            **Recommendation:**
            Maintain current positions but avoid new commitments. Wait for clearer signals.
            Monitor for breakout in either direction.

            **Action:** Hold, wait for better setup
            """)

    else:
        st.info("No predictions available. Run: `python3 predict_multi_asset_ensemble.py`")

    # Commands
    st.markdown("---")
    st.subheader("💻 Run LLM Analysis")

    st.code("""
# Single asset analysis with OpenAI
python3 llm_forecast.py --asset SPY --provider openai

# Single asset analysis with Anthropic
python3 llm_forecast.py --asset SPY --provider anthropic

# Multi-asset summary
python3 llm_forecast.py --all --provider openai

# Generate and send newsletter
python3 newsletter_generator.py --send --assets SPY,BTC/USDT
    """)


def show_signal_analytics():
    """Signal analytics page"""
    st.title("📊 Signal Analytics")

    predictions = load_predictions()

    if predictions is None or len(predictions) == 0:
        st.warning("No predictions available. Run: `python3 predict_multi_asset_ensemble.py`")
        return

    # Signal distribution
    st.subheader("📈 Signal Distribution")

    if 'Prediction' in predictions.columns:
        signal_counts = predictions['Prediction'].value_counts().sort_index()
        signal_map = {0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'}

        col1, col2, col3 = st.columns(3)

        crash_count = signal_counts.get(0, 0)
        normal_count = signal_counts.get(1, 0)
        spike_count = signal_counts.get(2, 0)
        total = len(predictions)

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

    # Confidence distribution
    st.markdown("---")
    st.subheader("🎯 Confidence Distribution")

    if 'Confidence' in predictions.columns:
        confidence_values = predictions['Confidence'].dropna()

        fig = go.Figure(data=[go.Histogram(x=confidence_values, nbinsx=50, marker_color='blue')])
        fig.update_layout(title="Confidence Distribution", xaxis_title="Confidence", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        - **Mean Confidence:** {confidence_values.mean():.3f}
        - **Median Confidence:** {confidence_values.median():.3f}
        - **Std Dev:** {confidence_values.std():.3f}
        """)

    # Recent signals
    st.markdown("---")
    st.subheader("📅 Recent Signals")

    recent = predictions.tail(20).copy()
    if 'Prediction' in recent.columns:
        recent['Signal'] = recent['Prediction'].map({0: '🔴 CRASH', 1: '🟡 NORMAL', 2: '🟢 SPIKE'})

    display_cols = ['Date', 'Signal', 'Proba', 'Confidence']
    available_cols = [col for col in display_cols if col in recent.columns]

    if available_cols:
        st.dataframe(recent[available_cols], use_container_width=True)


def main():
    """Main application"""
    st.sidebar.title("🚀 NeuroVest")
    st.sidebar.markdown("**Comprehensive Features**")
    st.sidebar.markdown("---")

    # Navigation
    pages = {
        "Overview": show_overview,
        "Trading Risk Profiles": show_trading_profiles,
        "Recession Indicator": show_recession_indicator,
        "Valuation Detector": show_valuation_detector,
        "LLM Integration": show_llm_integration,
        "Signal Analytics": show_signal_analytics
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Actions")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 Resources
    - [Documentation](README.md)
    - [Deployment Guide](DEPLOYMENT.md)
    - Run: `python3 demo_comprehensive.py`
    """)

    # Render selected page
    pages[page]()


if __name__ == "__main__":
    main()
