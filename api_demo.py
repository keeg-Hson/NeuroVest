#!/usr/bin/env python3
"""
NeuroVest API Demo - Customer-Facing Version

Professional demonstration of the NeuroVest Forecasting API for prospective customers.
Showcases API capabilities, integration examples, and business value.

Usage:
    streamlit run api_demo.py
    streamlit run api_demo.py --server.port 8501
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
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

# Dark theme with readable text
st.markdown("""
<style>
    /* Dark background */
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
    }

    /* Typography - white text on dark background */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }

    p {
        color: #ffffff;
    }

    /* Metric cards - white bg with DARK text */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        margin: 1rem 0;
    }

    .metric-box h2 {
        color: #1a1a1a !important;
        font-size: 2.5rem;
        margin: 0;
    }

    .metric-box p {
        color: #1a1a1a !important;
        margin: 0.5rem 0 0 0;
    }

    .metric-box small {
        color: #555 !important;
    }

    /* Info cards - white bg with DARK text */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        height: 100%;
    }

    .info-card h3 {
        color: #1a1a1a !important;
        margin-top: 0;
    }

    .info-card p, .info-card strong {
        color: #1a1a1a !important;
        line-height: 1.7;
    }

    /* Buttons */
    .stButton > button {
        background: #3498db;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background: #2980b9;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def show_hero():
    """Hero section"""
    # Logo
    logo_path = Path("assets/neurovest_logo.png")
    if logo_path.exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(logo_path), width=200)

    st.markdown("# NeuroVest Forecasting API")
    st.markdown("### AI-Powered Market Intelligence for Quantitative Developers")
    st.markdown("Real-time predictions for 59+ assets | Sub-500ms latency | Proven 25-year track record")
    st.markdown("---")


def show_quick_stats():
    """Key performance metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <h2>191%</h2>
            <p>Total Return<br><small>25-year SPY backtest</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <h2>2.55</h2>
            <p>Sharpe Ratio<br><small>Risk-adjusted returns</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <h2>-5.4%</h2>
            <p>Max Drawdown<br><small>vs -55% buy-hold</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
            <h2>59+</h2>
            <p>Assets Covered<br><small>Stocks, ETFs, Crypto</small></p>
        </div>
        """, unsafe_allow_html=True)


def show_value_proposition():
    """Value proposition"""
    st.markdown("## What is NeuroVest?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 How It Works</h3>
            <p>
                Ensemble ML forecasting platform that predicts market movements across 59 assets.
                Combines XGBoost, LightGBM, and CatBoost models trained on 126+ features to
                generate three-class predictions (CRASH/NORMAL/SPIKE) with probability distributions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>💡 Why It Matters</h3>
            <p>
                Most market data APIs give you raw prices. NeuroVest provides processed intelligence
                with quantified confidence levels. Build systems that adjust position sizing based on
                signal strength, filter low-confidence trades, or trigger alerts for high-probability setups.
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
                • 467% better than buy-hold<br>
                • -5.4% max drawdown<br>
                Performance proven across multiple market regimes including 2008 and 2020 crashes.
            </p>
        </div>
        """, unsafe_allow_html=True)


def show_features():
    """Core features"""
    st.markdown("---")
    st.markdown("## Core API Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🎯 Real-Time Predictions
        Single-asset forecasts with three-class output and probability distributions.
        Response times under 500ms.

        **Typical use:** Algorithmic entry/exit signals, risk alerts, portfolio triggers
        """)

    with col2:
        st.markdown("""
        ### 📊 Batch Analysis
        Multi-asset requests (up to 50 tickers) processed in parallel with structured JSON output.

        **Typical use:** Cross-sectional screening, sector rotation, daily reports
        """)

    with col3:
        st.markdown("""
        ### 📈 Custom Backtests
        Test signal performance with configurable parameters: stops, targets, sizing, costs.

        **Typical use:** Strategy validation, parameter optimization, compliance reporting
        """)


def show_api_playground():
    """API examples"""
    st.markdown("---")
    st.markdown("## 🔌 API Examples")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])

    with tab1:
        st.markdown("### Get Single Asset Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Request:**")
            st.code("""GET /predict/SPY

Headers:
  Authorization: Bearer {API_KEY}
  Content-Type: application/json""", language="http")

        with col2:
            st.markdown("**Response:**")
            st.json({
                "asset": "SPY",
                "prediction": 2,
                "signal": "SPIKE",
                "probability": 0.78,
                "confidence": "high",
                "models_agree": True
            })

    with tab2:
        st.markdown("### Batch Asset Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Request:**")
            st.code("""POST /batch-predict

Body:
{
  "assets": ["SPY", "QQQ", "BTC/USDT"]
}""", language="http")

        with col2:
            st.markdown("**Response:**")
            st.json({
                "total_assets": 3,
                "results": [
                    {"asset": "SPY", "signal": "SPIKE", "probability": 0.78},
                    {"asset": "QQQ", "signal": "NORMAL", "probability": 0.62},
                    {"asset": "BTC/USDT", "signal": "CRASH", "probability": 0.71}
                ]
            })


def show_pricing():
    """Pricing tiers"""
    st.markdown("---")
    st.markdown("## 💳 API Pricing")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        ### Free Tier
        **$0/mo**

        - 1,000 requests/month
        - All 59 assets
        - Basic support
        - API documentation

        **For:** Testing & research
        """)

    with col2:
        st.markdown("""
        ### Developer
        **$99/mo**

        - 50,000 requests/month
        - All features
        - Priority support
        - Backtesting API
        - Low latency

        **For:** Individual traders
        """)

    with col3:
        st.markdown("""
        ### Professional
        **$499/mo** ⭐

        - 500,000 requests/month
        - All features
        - Dedicated support
        - WebSocket streaming
        - Custom models
        - SLA guarantee

        **For:** Hedge funds
        """)

    with col4:
        st.markdown("""
        ### Enterprise
        **Custom**

        - Unlimited requests
        - White-glove support
        - On-premise deployment
        - Custom integrations
        - Training workshops

        **For:** Institutions
        """)


def show_cta():
    """Call to action"""
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## Ready to Get Started?")
        st.markdown("Join hedge funds, quant developers, and institutions using NeuroVest API")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("[Start Free Trial](https://neurovestdemo.streamlit.app/)")
        with col_b:
            st.markdown("[Schedule Demo](https://neurovestdemo.streamlit.app/)")

        st.markdown("---")
        st.caption("No credit card required • 1,000 free requests/month • Cancel anytime")


def main():
    show_hero()
    show_quick_stats()
    show_value_proposition()
    show_features()
    show_api_playground()
    show_pricing()
    show_cta()


if __name__ == "__main__":
    main()
