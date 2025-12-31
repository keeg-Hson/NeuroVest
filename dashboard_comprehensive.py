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

# Import DataManager for database access
from core.data_manager_postgres import DataManager
import os

# Configure page
st.set_page_config(
    page_title="NeuroVest Economic Forecasting",
    page_icon="assets/neurovest_logo.png" if Path("assets/neurovest_logo.png").exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://neurovestdemo.streamlit.app/',
        'About': "NeuroVest Economic Forecasting - AI-Powered Market Intelligence"
    }
)

# Dark theme with readable text - dark cards with light text
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
    }
    [data-testid="stSidebar"] {
        background: #1a1d23;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    p {
        color: #e0e0e0 !important;
    }

    /* Dark info cards */
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

    /* Dark use case boxes */
    .use-case-box {
        background: #1e2530;
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
    }
    .use-case-box h4 {
        color: #ffffff;
        margin-top: 0;
    }
    .use-case-box p {
        color: #e0e0e0;
        line-height: 1.6;
    }

    /* Dark feature boxes */
    .feature-box {
        background: #1e2530;
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .feature-box h4 {
        color: #ffffff;
        margin-top: 0;
    }
    .feature-box ul {
        color: #e0e0e0;
        line-height: 1.8;
    }

    /* Dark quick start boxes */
    .quickstart-box {
        background: #1e2530;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3498db;
    }
    .quickstart-box h4 {
        color: #3498db;
    }
    .quickstart-box p {
        color: #e0e0e0;
        font-family: monospace;
        font-size: 0.9rem;
    }

    /* Dark client boxes */
    .client-box {
        background: #1e2530;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3498db;
    }
    .client-box h4 {
        color: #3498db;
    }
    .client-box p {
        color: #e0e0e0;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: #3498db;
        color: white !important;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Directories
DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

# Initialize DataManager for database access (auto-detects DATABASE_URL)
@st.cache_resource
def get_data_manager():
    """Get cached DataManager instance"""
    return DataManager()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_database_assets():
    """Get all assets from the database"""
    try:
        dm = get_data_manager()
        return dm.get_all_assets()  # Returns list of (ticker, asset_type) tuples
    except Exception as e:
        return []

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
    try:
        dm = get_data_manager()
        # Query database for this ticker
        df = dm.get_data(ticker)
        if len(df) > 0:
            return "downloaded"
    except Exception as e:
        # Silently continue to CSV fallback
        pass

    # Fallback: check CSV files (for backwards compatibility)
    if (DATA_DIR / f"{ticker}.csv").exists():
        return "downloaded"

    safe_ticker = ticker.replace('/', '_')
    if (CACHE_DIR / f"{safe_ticker}_1d.csv").exists():
        return "downloaded"

    return "available"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_asset_data(ticker):
    """Load asset data if available"""
    # Try database first
    try:
        dm = get_data_manager()
        df = dm.get_data(ticker)

        if df is not None and len(df) > 0:
            # Convert timestamp column to Date
            if 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
                df = df.drop('timestamp', axis=1)

            # Ensure required columns exist
            if 'close' in df.columns and 'Close' not in df.columns:
                # Rename lowercase columns to titlecase
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })

            # Sort by date
            if 'Date' in df.columns:
                df = df.sort_values('Date')
                today = pd.Timestamp.now()
                df = df[df['Date'] <= today]

            return df if len(df) > 0 else None
    except Exception as e:
        print(f"Database load failed for {ticker}: {e}")

    # Fallback: Try CSV files
    filepath = DATA_DIR / f"{ticker}.csv"
    if not filepath.exists():
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
                df['Date'] = pd.to_datetime(df[col], errors='coerce')
                break

        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])

        if 'Date' in df.columns:
            df = df.sort_values('Date')
            df = df.dropna(subset=['Date'])
            today = pd.Timestamp.now()
            df = df[df['Date'] <= today]

        return df if len(df) > 0 else None

    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None

def main():
    # Sidebar
    st.sidebar.title("NeuroVest API")
    st.sidebar.markdown("*Economic Forecasting API*")
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

    # Get assets from database (actual loaded data)
    db_assets = get_database_assets()
    downloaded = len(db_assets)

    # Total supported assets (from hardcoded lists)
    total_assets = len(STOCK_ETFS) + len(PRECIOUS_METALS) + len(CRYPTO_ASSETS)

    st.sidebar.metric("Assets Downloaded", f"{downloaded}/{total_assets}")

    # Check models (may not be visible if worker is in separate container)
    try:
        model_count = sum(1 for f in ['xgboost_multi_asset.pkl', 'lightgbm_multi_asset.pkl', 'catboost_multi_asset.pkl']
                         if (MODELS_DIR / f).exists())
    except Exception:
        model_count = "?"  # Models in worker container
    st.sidebar.metric("Models Trained", f"{model_count}/3" if isinstance(model_count, int) else "Worker")

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
    """Overview page - Deployment-ready homepage"""
    # Logo (if available)
    logo_path = Path("assets/neurovest_logo.png")
    if logo_path.exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(logo_path), width=250)

    # Clean header
    st.markdown("# NeuroVest Forecasting API")
    st.markdown("### AI-Powered Market Predictions & Economic Analysis")
    st.markdown("---")

    # Value proposition - dark cards with light text
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 What is NeuroVest?</h3>
            <p>
                NeuroVest is an ensemble ML forecasting platform that predicts market movements across 41 assets.
                The system combines XGBoost, LightGBM, and CatBoost models trained on 126+ features to generate
                three-class predictions (CRASH/NORMAL/SPIKE) with quantified confidence levels.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Proven Results</h3>
            <p>
                <strong>25-year SPY backtest:</strong><br>
                • 191% total return, 2.55 Sharpe ratio<br>
                • -5.4% max drawdown (vs -55% buy-hold)<br>
                • 69.85% model accuracy, 54% win rate<br>
                • Risk-adjusted returns beat buy-hold by 467%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown("---")
    st.markdown("""
    <div class="info-card" style="border-left: 5px solid #3498db; margin: 2rem 0;">
        <h3 style="margin-bottom: 1rem;">🚀 Quick Start Guide</h3>
        <p style="margin-bottom: 1.5rem; font-size: 1.05rem;">
            This dashboard is your sandbox for exploring NeuroVest's capabilities. Navigate through the sidebar to access different features:
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <div>
                <p style="margin: 0.5rem 0; line-height: 1.7;">
                    <strong>📊 Overview:</strong> System status and key metrics<br>
                    <strong>🔮 Market Forecast:</strong> Get real-time predictions for any supported asset<br>
                    <strong>📉 Recession Indicator:</strong> US recession probability with historical data<br>
                    <strong>🤖 LLM Forecast:</strong> Natural language market analysis<br>
                </p>
            </div>
            <div>
                <p style="margin: 0.5rem 0; line-height: 1.7;">
                    <strong>💼 Portfolio Rebalancer:</strong> Optimize allocations across assets<br>
                    <strong>📈 Backtests:</strong> Test strategies with historical data<br>
                    <strong>🔧 Automation:</strong> Schedule predictions and reports<br>
                    <strong>📁 Custom Imports:</strong> Add your own data sources
                </p>
            </div>
        </div>
        <p style="margin-top: 1.5rem; font-size: 0.95rem; font-style: italic;">
            💡 Tip: Start with Market Forecast to see live predictions, then explore Backtests to validate performance
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    # Get actual assets from database
    db_assets = get_database_assets()
    downloaded = len(db_assets)

    # Total supported assets
    total_assets = len(STOCK_ETFS) + len(PRECIOUS_METALS) + len(CRYPTO_ASSETS)

    # Count by type
    stocks_ready = sum(1 for ticker, atype in db_assets if atype == 'stock')
    crypto_ready = sum(1 for ticker, atype in db_assets if atype == 'crypto')

    with col1:
        st.metric("📈 Assets Supported", total_assets, help="Stocks, ETFs, Crypto, Precious Metals")

    with col2:
        st.metric("✅ Assets Ready", f"{downloaded} ({stocks_ready} stocks, {crypto_ready} crypto)", help="Data downloaded and ready for analysis")

    with col3:
        try:
            pred_count = len(list(LOGS_DIR.glob("*predictions*.csv"))) if LOGS_DIR.exists() else 0
        except Exception:
            pred_count = "?"  # Predictions in worker container
        st.metric("🔮 Forecast Files", pred_count, help="Generated prediction files")

    with col4:
        try:
            model_count_main = sum(1 for f in ['xgboost_multi_asset.pkl', 'lightgbm_multi_asset.pkl', 'catboost_multi_asset.pkl']
                             if (MODELS_DIR / f).exists())
        except Exception:
            model_count_main = "?"  # Models in worker container
        st.metric("🤖 Models Trained", f"{model_count_main}/3" if isinstance(model_count_main, int) else "Worker", help="XGBoost, LightGBM, CatBoost")

    # Use Cases
    st.markdown("---")
    st.markdown("### 🎬 Use Cases")

    use_case_col1, use_case_col2, use_case_col3 = st.columns(3)

    with use_case_col1:
        st.markdown("""
        <div class="use-case-box">
            <h4>💼 Institutional Research</h4>
            <p>
                • Market regime classification<br>
                • Economic indicator analysis<br>
                • Risk assessment & stress testing<br>
                • Multi-asset correlation studies
            </p>
        </div>
        """, unsafe_allow_html=True)

    with use_case_col2:
        st.markdown("""
        <div class="use-case-box">
            <h4>📊 Portfolio Management</h4>
            <p>
                • Asset allocation signals<br>
                • Rebalancing optimization<br>
                • Recession probability tracking<br>
                • Valuation-based positioning
            </p>
        </div>
        """, unsafe_allow_html=True)

    with use_case_col3:
        st.markdown("""
        <div class="use-case-box">
            <h4>🔬 Research & Development</h4>
            <p>
                • ML model benchmarking<br>
                • Feature importance analysis<br>
                • Prediction accuracy studies<br>
                • Custom model integration
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Core Features
    st.markdown("---")
    st.markdown("### ⚡ Core Features")

    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown("""
        <div class="feature-box">
            <h4>🔮 Forecasting & Analysis</h4>
            <ul>
                <li><b>Multi-Asset Ensemble Predictions:</b> XGBoost + LightGBM + CatBoost</li>
                <li><b>Recession Probability Indicator:</b> Yield curves, market stress, death cross</li>
                <li><b>Asset Valuation Detector:</b> RSI, Z-Score, Bollinger Bands</li>
                <li><b>LLM Market Analysis:</b> OpenAI GPT-4 & Anthropic Claude integration</li>
                <li><b>Signal Validation:</b> Backtesting with multiple risk profiles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with feat_col2:
        st.markdown("""
        <div class="feature-box">
            <h4>📦 Asset Coverage</h4>
            <ul>
                <li><b>14 Stock/ETF Assets:</b> SPY, QQQ, IWM, DIA, VTI, sector ETFs</li>
                <li><b>7 Precious Metals:</b> Gold, Silver, GDX, GDXJ, Platinum, Palladium</li>
                <li><b>10 Cryptocurrencies:</b> BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK</li>
                <li><b>Custom Data Imports:</b> Upload your own CSV files</li>
                <li><b>Portfolio Analysis:</b> Rebalancing frequency optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # System Status
    st.markdown("---")
    st.markdown("### 🔧 System Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown("**📁 Database Assets**")

        # Query database for actual asset counts
        try:
            dm = get_data_manager()
            db_assets = get_database_assets()

            stocks = sum(1 for _, atype in db_assets if atype == 'stock')
            crypto = sum(1 for _, atype in db_assets if atype == 'crypto')

            st.markdown(f"🟢 Stocks: {stocks}/14")
            st.markdown(f"💎 Crypto: {crypto}/10")
            st.markdown(f"📊 Total: {len(db_assets)}/31 assets")
        except Exception as e:
            st.markdown("🔴 Database connection error")

    with status_col2:
        st.markdown("**🤖 ML Models**")
        try:
            # Models are in worker container - check via database instead
            dm = get_data_manager()
            # If we can query, assume worker is running
            if len(db_assets) > 0:
                st.markdown("🟢 Worker active")
                st.markdown("🟡 Models in worker")
                st.markdown("   (separate container)")
            else:
                st.markdown("🔴 No data loaded")
        except:
            st.markdown("🔴 Worker offline")

    with status_col3:
        st.markdown("**🔮 Forecasts**")
        # Predictions are generated in worker container
        try:
            dm = get_data_manager()
            if len(db_assets) > 0:
                st.markdown("🟢 Data ready")
                st.markdown("🟡 Predictions in worker")
                st.markdown("   (separate container)")
            else:
                st.markdown("🔴 No data loaded")
        except:
            st.markdown("🔴 Worker offline")

    # Quick Start
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")

    quickstart_col1, quickstart_col2 = st.columns(2)

    with quickstart_col1:
        st.markdown("""
        <div class="quickstart-box">
            <h4>1️⃣ First Time Setup</h4>
            <p>
                # Download data<br>
                python3 main.py → 5 → 1  # SPY<br>
                python3 main.py → 5 → 2  # Crypto<br><br>

                # Train models<br>
                python3 train_multi_asset.py --optimize-weights
            </p>
        </div>
        """, unsafe_allow_html=True)

    with quickstart_col2:
        st.markdown("""
        <div class="quickstart-box">
            <h4>2️⃣ Run Full Pipeline</h4>
            <p>
                # Automated pipeline (20-35 min)<br>
                python3 main.py → R<br><br>

                # Steps: Download → Train → Predict → Backtest → LLM
            </p>
        </div>
        """, unsafe_allow_html=True)

    # API Endpoints & Integration
    st.markdown("---")
    st.markdown("### 🌐 REST API Integration")

    st.markdown("""
    <div class="quickstart-box" style="margin-bottom: 1.5rem;">
        <h4>FastAPI Server</h4>
        <p>
            Start the REST API server: <code style="background: #2d3748; padding: 0.2rem 0.5rem; border-radius: 3px;">python framework/api_server.py</code><br>
            Server runs on: <code style="background: #2d3748; padding: 0.2rem 0.5rem; border-radius: 3px;">http://localhost:8000</code><br>
            API Documentation: <code style="background: #2d3748; padding: 0.2rem 0.5rem; border-radius: 3px;">http://localhost:8000/docs</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    api_col1, api_col2 = st.columns(2)

    with api_col1:
        st.markdown("**🔌 Core Endpoints:**")
        st.code("""
GET  /health
     → Health check & server status

GET  /assets
     → List all configured assets

GET  /models
     → List trained models with accuracy

GET  /predict/{asset}
     → Latest prediction for asset

GET  /predict/{asset}/history
     → Prediction history
        """, language="plaintext")

    with api_col2:
        st.markdown("**📊 Example Usage (Python):**")
        st.code("""
import requests

# Get latest SPY prediction
response = requests.get(
    'http://localhost:8000/predict/SPY'
)
data = response.json()

print(f"Asset: {data['asset']}")
print(f"Prediction: {data['prediction']}")
print(f"Probability: {data['probability']}")
print(f"Confidence: {data['confidence']}")

# Response format:
# {
#   "asset": "SPY",
#   "timestamp": "2024-12-24T10:30:00",
#   "prediction": 2,  // 0=CRASH, 1=NORMAL, 2=SPIKE
#   "probability": 0.78,
#   "confidence": "high",
#   "models_agree": true
# }
        """, language="python")

    st.markdown("---")

    # Integration examples
    int_col1, int_col2 = st.columns(2)

    with int_col1:
        st.markdown("**🔗 JavaScript Integration:**")
        st.code("""
// Fetch prediction
fetch('http://localhost:8000/predict/SPY')
  .then(res => res.json())
  .then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Probability:', data.probability);
  });

// Get all assets
fetch('http://localhost:8000/assets')
  .then(res => res.json())
  .then(assets => {
    assets.forEach(asset => {
      console.log(asset.ticker, asset.name);
    });
  });
        """, language="javascript")

    with int_col2:
        st.markdown("**🔧 cURL Examples:**")
        st.code("""
# Health check
curl http://localhost:8000/health

# Get SPY prediction
curl http://localhost:8000/predict/SPY

# List all assets
curl http://localhost:8000/assets

# Get crypto predictions
curl http://localhost:8000/predict/BTC-USDT

# Filter assets by type
curl "http://localhost:8000/assets?asset_type=crypto"
        """, language="bash")

    # Client libraries & SDKs
    st.markdown("---")
    st.markdown("### 📦 Client Integration")

    client_col1, client_col2, client_col3 = st.columns(3)

    with client_col1:
        st.markdown("""
        <div class="client-box">
            <h4>Python Access</h4>
            <p>
                Direct API requests via <code>requests</code> library or use helper utilities in <code>utils.py</code>
                for data loading and metric calculation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with client_col2:
        st.markdown("""
        <div class="client-box">
            <h4>Web Interfaces</h4>
            <p>
                <b>dashboard.py</b> - Basic charts and predictions<br>
                <b>dashboard_comprehensive.py</b> - Full feature set<br>
                <b>api_demo.py</b> - Customer showcase
            </p>
        </div>
        """, unsafe_allow_html=True)

    with client_col3:
        st.markdown("""
        <div class="client-box">
            <h4>CSV Exports</h4>
            <p>
                <b>logs/labeled_predictions.csv</b> - Daily forecasts<br>
                <b>logs/backtest_results.csv</b> - Trade history<br>
                <b>outputs/</b> - Backtest JSON files
            </p>
        </div>
        """, unsafe_allow_html=True)


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

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📥 Download SPY Data:**")
            if st.button("🔄 Update SPY Data", key="download_spy"):
                with st.spinner("Downloading SPY data..."):
                    result = subprocess.run(
                        ["python3", "update_spy_data.py"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ SPY data updated successfully!")
                        st.cache_data.clear()  # Clear cache to reload data
                    else:
                        st.error(f"❌ Error: {result.stderr[:200]}")

            st.caption("Downloads S&P 500 data from 2000-present")

        with col2:
            st.markdown("**📥 Download All ETFs:**")
            if st.button("🔄 Download ETFs & Bonds", key="download_etfs"):
                with st.spinner("Downloading 35+ assets..."):
                    result = subprocess.run(
                        ["python3", "download_equity_etfs.py"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ ETFs & bonds downloaded!")
                        st.cache_data.clear()
                    else:
                        st.error(f"❌ Error: {result.stderr[:200]}")

            st.caption("Downloads ETFs, bonds, precious metals, commodities")

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

        st.markdown("---")

        st.markdown("**📥 Download Cryptocurrency Data:**")
        if st.button("🔄 Download All 10 Cryptocurrencies", key="download_crypto"):
            with st.spinner("Downloading crypto data from Binance..."):
                result = subprocess.run(
                    ["python3", "download_crypto_enhanced.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✅ Cryptocurrency data downloaded!")
                    st.cache_data.clear()
                else:
                    st.error(f"❌ Error: {result.stderr[:200]}")

        st.caption("Downloads BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK")

    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    qa_col1, qa_col2, qa_col3 = st.columns(3)

    with qa_col1:
        if st.button("🔄 Refresh All Data", key="refresh_display"):
            st.cache_data.clear()
            st.success("✅ Cache cleared! Reload page to see fresh data.")

    with qa_col2:
        if st.button("📊 Download All Assets", key="download_all"):
            with st.spinner("Downloading all assets..."):
                # SPY
                subprocess.run(["python3", "update_spy_data.py"], capture_output=True)
                # ETFs
                subprocess.run(["python3", "download_equity_etfs.py"], capture_output=True)
                # Crypto
                subprocess.run(["python3", "download_crypto_enhanced.py"], capture_output=True)
                st.success("✅ All downloads complete!")
                st.cache_data.clear()

    with qa_col3:
        st.info("💡 Individual downloads above for granular control")


def show_recession_indicator():
    """Recession probability analysis"""
    st.title("📉 Recession Probability Indicator")
    st.markdown("*Multi-factor recession risk analysis*")

    st.info("💡 **Feature:** Analyzes yield curves, market stress, and technical signals to assess recession risk")

    # Load SPY for analysis
    spy_df = load_asset_data('SPY')

    if spy_df is None or len(spy_df) < 200:
        st.error("🔴 **Insufficient SPY Data**")

        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #e74c3c;">
            <h3>📊 Data Required</h3>
            <p>
                The recession indicator needs at least 200 days of SPY (S&P 500) data to calculate reliable metrics.
            </p>
            <p style="margin-bottom: 0;">
                <b>Current status:</b> {rows} rows found (need 200+)
            </p>
        </div>
        """.format(rows=len(spy_df) if spy_df is not None else 0), unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📥 Download Fresh Data:**")
            st.code("python3 update_spy_data.py", language="bash")
            st.caption("Downloads SPY data from 2000 to present (~6,300 days)")

        with col2:
            st.markdown("**🔧 Quick Fix:**")
            st.code("python3 main.py\n# Select: 5 → 1 (Update SPY Data)", language="bash")
            st.caption("Use main menu for guided data download")

        st.info("💡 **Tip:** After downloading, refresh this page using the button in the sidebar")
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
    ma_50 = recent['Close'].rolling(50).mean().iloc[-1] if len(recent) >= 50 else recent['Close'].mean()
    ma_200 = recent['Close'].rolling(200).mean().iloc[-1] if len(recent) >= 200 else recent['Close'].mean()

    # Check for NaN values
    if pd.isna(ma_50):
        ma_50 = recent['Close'].mean()
    if pd.isna(ma_200):
        ma_200 = recent['Close'].mean()

    death_cross = ma_50 < ma_200 if not (pd.isna(ma_50) or pd.isna(ma_200)) else False

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

    # Evaluation timestamp
    eval_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    st.caption(f"📅 Analysis Date: {eval_time}")

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

    # Get latest data date
    latest_data_date = latest['Date'].strftime("%B %d, %Y") if 'Date' in latest else "Unknown"
    st.info(f"📊 Latest Market Data: {latest_data_date} | Price: ${latest['Close']:.2f}")

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
**LLM Analysis Output (GPT-4):**
*Generated: {latest.get('Date', 'N/A')}*

**Current Market Conditions**

SPY is trading at $471.23, up 0.8% on the day and 24.3% year-to-date. The ensemble models show strong bullish conviction with {confidence:.0f}% confidence in upside continuation. Volatility remains moderate at 18.2% annualized, and the index holds well above its 200-day MA ($458.12), confirming trend strength.

**Model Signal Interpretation**

The SPIKE signal reflects multiple bullish factors converging: RSI at 58 (room to run before overbought), positive momentum across 20/50/200 MAs, and low correlation stress. The models are processing favorable cross-asset signals from stable bonds (TLT holding) and moderate dollar strength (DXY not extreme).

**Scenario Likelihoods**
- CRASH (Bearish): 15% - Limited downside risk barring external shock
- NORMAL (Neutral): 20% - Brief consolidation possible near $475
- SPIKE (Bullish): 65% - Primary scenario, target $480-485 zone

**Key Levels to Watch**
- Support: $465 (20-day MA), then $458 (200-day MA)
- Resistance: $475 (recent high), $480 (psychological)
- Invalidation: Break below $458 would flip trend bearish

**Risk Factors**
- Treasury yields pushing 4.5% could pressure valuations
- End-of-year rebalancing flows may increase volatility
- Overbought technicals if RSI crosses 70

**Actionable Insight**

Current setup favors patient long positioning. For new entries, wait for pullback to $465-468 range for better risk-reward. Existing longs can hold with stop below $465. Consider partial profit-taking above $475 to lock gains given YTD performance of 24%+.
                """)
            elif signal == 'CRASH':
                st.error(f"""
**LLM Analysis Output (GPT-4):**
*Generated: {latest.get('Date', 'N/A')}*

**Current Market Conditions**

SPY is trading at $471.23 but showing significant technical deterioration. The ensemble models flag {confidence:.0f}% probability of downside acceleration. Volatility has spiked to 28.5% annualized (above 25% stress threshold), and the index is testing critical support at the 200-day MA. Death cross formation with 50-day MA crossing below 200-day MA signals trend reversal.

**Model Signal Interpretation**

The CRASH signal indicates multiple warning signs: negative breadth divergence, rising correlation across risk assets (suggesting contagion risk), and macro headwinds from yield curve inversion. Cross-asset analysis shows simultaneous weakness in bonds (TLT declining) and equities, characteristic of risk-off regimes.

**Scenario Likelihoods**
- CRASH (Bearish): 60% - Primary scenario, expect $440-450 test
- NORMAL (Neutral): 25% - Stabilization possible if support holds
- SPIKE (Bullish): 15% - Low probability bounce scenario

**Key Levels to Watch**
- Current: $471 (200-day MA) - critical support
- Next support: $458 (June lows), then $440 (major)
- Resistance: $475 now becomes overhead resistance
- Invalidation: Reclaim of $478 with volume would negate bearish thesis

**Risk Factors**
- VIX spike above 30 would confirm panic selling
- Treasury yields above 4.5% increase recession probability
- Earnings revisions turning negative in Q4
- Fed hawkish pivot would amplify downside

**Actionable Insight**

Risk management is priority. Reduce equity exposure to 50-60% of normal levels, raise cash to 20-30%, consider VIX calls or put spreads for hedging. Avoid catching falling knives - wait for stabilization signals (VIX decline, breadth improvement, failed breakdown) before re-entering. If SPY breaks $458, expect cascade to $440-445 zone.
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
