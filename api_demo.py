#!/usr/bin/env python3
"""
NeuroVest API Demo - Customer-Facing Version

Professional demonstration of the NeuroVest Forecasting API for prospective customers.
"""

import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Install required packages: pip install streamlit plotly pandas")
    sys.exit(1)

st.set_page_config(
    page_title="NeuroVest API - Market Forecasting",
    page_icon="assets/neurovest_logo.png" if Path("assets/neurovest_logo.png").exists() else "📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://neurovestdemo.streamlit.app/',
        'About': "NeuroVest Forecasting API - AI-Powered Market Intelligence"
    }
)

# Dark theme with READABLE text - dark cards with light text
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
    }

    h1, h2, h3 { color: #ffffff; }
    p { color: #e0e0e0; }

    /* Metric boxes - dark cards with light text */
    .metric-box {
        background: #1e2530;
        border: 1px solid #3498db;
        border-radius: 8px;
        padding: 2rem 1.5rem;
        text-align: center;
    }
    .metric-box h2 {
        color: #3498db;
        font-size: 2.5rem;
        margin: 0;
    }
    .metric-box p {
        color: #ffffff;
        margin: 0.5rem 0 0 0;
    }

    /* Info cards */
    .info-card {
        background: #1e2530;
        border: 1px solid #3498db;
        border-radius: 8px;
        padding: 2rem;
    }
    .info-card h3 {
        color: #3498db;
        margin-top: 0;
    }
    .info-card p {
        color: #e0e0e0;
        line-height: 1.7;
    }

    /* Pricing boxes */
    .pricing-box {
        background: #1e2530;
        border: 2px solid #2d3748;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        height: 100%;
    }
    .pricing-box:hover {
        border-color: #3498db;
    }
    .pricing-box h3 {
        color: #ffffff;
        margin: 0 0 1rem 0;
    }
    .pricing-box h2 {
        color: #3498db;
        font-size: 2rem;
        margin: 0 0 1.5rem 0;
    }
    .pricing-box ul {
        color: #e0e0e0;
        text-align: left;
        list-style: none;
        padding: 0;
    }
    .pricing-box li {
        margin: 0.5rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Logo
logo_path = Path("assets/neurovest_logo.png")
if logo_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(logo_path), width=200)

st.markdown("# NeuroVest Forecasting API")
st.markdown("### AI-Powered Market Intelligence for Quantitative Developers")
st.markdown("---")

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-box">
        <h2>191%</h2>
        <p>Total Return<br>25-year backtest</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <h2>2.55</h2>
        <p>Sharpe Ratio<br>Risk-adjusted</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <h2>-5.4%</h2>
        <p>Max Drawdown<br>vs -55% buy-hold</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box">
        <h2>59+</h2>
        <p>Assets<br>Stocks, ETFs, Crypto</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("## What is NeuroVest?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🔬 How It Works</h3>
        <p>
            Ensemble ML forecasting platform predicting market movements across 59 assets.
            Combines XGBoost, LightGBM, and CatBoost trained on 126+ features.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>💡 Why It Matters</h3>
        <p>
            Most APIs give raw prices. NeuroVest provides processed intelligence with
            quantified confidence levels. Build systems that adjust based on signal strength.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Track Record</h3>
        <p>
            <strong>25-year SPY backtest:</strong><br>
            • 69.85% model accuracy<br>
            • 2.55 Sharpe ratio<br>
            • 467% better than buy-hold
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("## Core API Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Real-Time Predictions")
    st.markdown("Single-asset forecasts with three-class output. Response times under 500ms.")

with col2:
    st.markdown("### 📊 Batch Analysis")
    st.markdown("Multi-asset requests (up to 50 tickers) processed in parallel.")

with col3:
    st.markdown("### 📈 Custom Backtests")
    st.markdown("Test signal performance with configurable parameters.")

st.markdown("---")
st.markdown("## 💳 API Pricing")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="pricing-box">
        <h3>Free Tier</h3>
        <h2>$0/mo</h2>
        <ul>
            <li>✓ 1,000 requests/month</li>
            <li>✓ All 59 assets</li>
            <li>✓ Basic support</li>
            <li>✓ API documentation</li>
        </ul>
        <p style="margin-top: 1rem; color: #3498db;"><strong>For:</strong> Testing</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="pricing-box">
        <h3>Developer</h3>
        <h2>$99/mo</h2>
        <ul>
            <li>✓ 50,000 requests/month</li>
            <li>✓ All features</li>
            <li>✓ Priority support</li>
            <li>✓ Backtesting API</li>
        </ul>
        <p style="margin-top: 1rem; color: #3498db;"><strong>For:</strong> Traders</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pricing-box" style="border: 2px solid #3498db;">
        <h3>Professional</h3>
        <h2>$499/mo</h2>
        <ul>
            <li>✓ 500,000 requests/month</li>
            <li>✓ Dedicated support</li>
            <li>✓ WebSocket streaming</li>
            <li>✓ Custom models</li>
        </ul>
        <p style="margin-top: 1rem; color: #3498db;"><strong>For:</strong> Hedge funds</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="pricing-box">
        <h3>Enterprise</h3>
        <h2>Custom</h2>
        <ul>
            <li>✓ Unlimited requests</li>
            <li>✓ White-glove support</li>
            <li>✓ On-premise deployment</li>
            <li>✓ Custom integrations</li>
        </ul>
        <p style="margin-top: 1rem; color: #3498db;"><strong>For:</strong> Institutions</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("## Ready to Get Started?")
st.markdown("[Start Free Trial](https://neurovestdemo.streamlit.app/) | [Schedule Demo](https://neurovestdemo.streamlit.app/)")
