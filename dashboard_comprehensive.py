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

# Authentication and custom asset management
from auth_middleware import AuthManager, save_custom_asset_to_db

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

        # Try original ticker format (e.g., BTC/USDT)
        df = dm.get_data(ticker)
        if df is not None and len(df) > 0:
            return "downloaded"

        # Try underscore format for crypto (e.g., BTC_USDT)
        if '/' in ticker:
            ticker_underscore = ticker.replace('/', '_')
            df = dm.get_data(ticker_underscore)
            if df is not None and len(df) > 0:
                return "downloaded"
    except Exception as e:
        # If database fails, continue to CSV fallback
        # (Don't silently hide errors - but don't crash either)
        import sys
        print(f"Database check failed for {ticker}: {e}", file=sys.stderr)

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

        # Try original ticker format
        df = dm.get_data(ticker)

        # Try underscore format for crypto if original didn't work
        if (df is None or len(df) == 0) and '/' in ticker:
            ticker_underscore = ticker.replace('/', '_')
            df = dm.get_data(ticker_underscore)

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
            "Overview",
            "Asset Manager",
            "Recession Indicator",
            "Valuation Detector",
            "LLM Analysis",
            "Portfolio Rebalancing",
            "Forecast Results",
            "Automation",
            "Custom Imports"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")

    if st.sidebar.button("Refresh Display"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")

    # Get assets from database (actual loaded data)
    db_assets = get_database_assets()
    downloaded = len(db_assets)

    # Total supported assets (use actual count from database)
    total_assets = len(db_assets) if db_assets else 40

    st.sidebar.metric("Assets Downloaded", f"{downloaded}/{total_assets}")

    # Check models from PostgreSQL metadata
    try:
        dm = get_data_manager()
        models_df = dm.get_latest_models()
        model_count = len(models_df)
    except Exception:
        model_count = 0

    st.sidebar.metric("Models Trained", f"{model_count}/3")

    # Authentication (for custom asset uploads)
    user_id = AuthManager.get_session_user()

    # Route to pages
    if page == "Overview":
        show_overview()
    elif page == "Asset Manager":
        show_asset_manager()
    elif page == "Recession Indicator":
        show_recession_indicator()
    elif page == "Valuation Detector":
        show_valuation_detector()
    elif page == "LLM Analysis":
        show_llm_analysis()
    elif page == "Portfolio Rebalancing":
        show_portfolio_rebalancing()
    elif page == "Forecast Results":
        show_forecast_results()
    elif page == "Automation":
        show_automation()
    elif page == "Custom Imports":
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
            <h3>What is NeuroVest?</h3>
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
            <h3>Proven Results</h3>
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
        <h3 style="margin-bottom: 1rem;">Quick Start Guide</h3>
        <p style="margin-bottom: 1.5rem; font-size: 1.05rem;">
            This dashboard is your sandbox for exploring NeuroVest's capabilities. Navigate through the sidebar to access different features:
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <div>
                <p style="margin: 0.5rem 0; line-height: 1.7;">
                    <strong>Overview:</strong> System status and key metrics<br>
                    <strong>Market Forecast:</strong> Get real-time predictions for any supported asset<br>
                    <strong>Recession Indicator:</strong> US recession probability with historical data<br>
                    <strong>LLM Forecast:</strong> Natural language market analysis<br>
                </p>
            </div>
            <div>
                <p style="margin: 0.5rem 0; line-height: 1.7;">
                    <strong>Portfolio Rebalancer:</strong> Optimize allocations across assets<br>
                    <strong>Backtests:</strong> Test strategies with historical data<br>
                    <strong>Automation:</strong> Schedule predictions and reports<br>
                    <strong>Custom Imports:</strong> Add your own data sources
                </p>
            </div>
        </div>
        <p style="margin-top: 1.5rem; font-size: 0.95rem; font-style: italic;">
            Tip: Start with Market Forecast to see live predictions, then explore Backtests to validate performance
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    # Get actual assets from database
    db_assets = get_database_assets()
    downloaded = len(db_assets)

    # Total supported assets (actual count from database)
    total_assets = len(db_assets)

    # Count by type
    stocks_ready = sum(1 for ticker, atype in db_assets if atype == 'stock')
    crypto_ready = sum(1 for ticker, atype in db_assets if atype == 'crypto')

    with col1:
        st.metric("Assets Supported", total_assets, help="All assets in database (stocks, ETFs, crypto, metals)")

    with col2:
        st.metric("Assets Ready", f"{downloaded} ({stocks_ready} stocks, {crypto_ready} crypto)", help="Data downloaded and ready for analysis")

    with col3:
        try:
            dm = get_data_manager()
            predictions_df = dm.get_latest_predictions(limit=100)
            pred_count = len(predictions_df)
        except Exception:
            pred_count = 0
        st.metric("Predictions", pred_count, help="Latest forecast records in database")

    with col4:
        try:
            dm = get_data_manager()
            models_df = dm.get_latest_models()
            model_count_main = len(models_df)
        except Exception:
            model_count_main = 0
        st.metric("Models Trained", f"{model_count_main}/3", help="XGBoost, LightGBM, CatBoost")

    # Use Cases
    st.markdown("---")
    st.markdown("### Use Cases")

    use_case_col1, use_case_col2, use_case_col3 = st.columns(3)

    with use_case_col1:
        st.markdown("""
        <div class="use-case-box">
            <h4>Institutional Research</h4>
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
            <h4>Portfolio Management</h4>
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
            <h4>Research & Development</h4>
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
    st.markdown("### Core Features")

    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown("""
        <div class="feature-box">
            <h4>Forecasting & Analysis</h4>
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
        # Get actual counts from database
        try:
            db_assets = get_database_assets()
            db_stocks = sum(1 for _, atype in db_assets if atype == 'stock')
            db_crypto = sum(1 for _, atype in db_assets if atype == 'crypto')
            db_total = len(db_assets)
        except Exception:
            db_stocks, db_crypto, db_total = 29, 11, 40  # Fallback to known counts

        st.markdown(f"""
        <div class="feature-box">
            <h4>Asset Coverage</h4>
            <ul>
                <li><b>{db_stocks} Stock/ETF Assets:</b> SPY, QQQ, IWM, DIA, VTI, sector ETFs, macro indicators</li>
                <li><b>7 Precious Metals:</b> Gold, Silver, GDX, GDXJ, Platinum, Palladium</li>
                <li><b>{db_crypto} Cryptocurrencies:</b> BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK</li>
                <li><b>Custom Data Imports:</b> Upload your own CSV files (feature available)</li>
                <li><b>Portfolio Analysis:</b> Rebalancing frequency optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # System Status
    st.markdown("---")
    st.markdown("### System Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown("**Database Assets**")

        # Query database for actual asset counts
        try:
            dm = get_data_manager()
            db_assets = get_database_assets()

            stocks = sum(1 for _, atype in db_assets if atype == 'stock')
            crypto = sum(1 for _, atype in db_assets if atype == 'crypto')

            # Show actual counts from database (not hardcoded denominators)
            st.markdown(f"Stocks: {stocks} assets")
            st.markdown(f"Crypto: {crypto} assets")
            st.markdown(f"Total: {len(db_assets)} assets")
        except Exception as e:
            st.markdown("Database connection error")

    with status_col2:
        st.markdown("**ML Models**")
        try:
            dm = get_data_manager()
            models_df = dm.get_latest_models()

            if len(models_df) > 0:
                st.markdown(f"{len(models_df)}/3 models trained")
                for _, model in models_df.iterrows():
                    trained_at = pd.to_datetime(model['trained_at'])
                    age = (pd.Timestamp.now() - trained_at).days
                    st.markdown(f"   • {model['model_type']} ({age}d ago)")
            else:
                st.markdown("No models yet")
                st.markdown("   (training scheduled)")
        except Exception as e:
            st.markdown("Database error")

    with status_col3:
        st.markdown("**Forecasts**")
        try:
            dm = get_data_manager()
            predictions_df = dm.get_latest_predictions(limit=1)

            if len(predictions_df) > 0:
                latest_pred = predictions_df.iloc[0]
                pred_date = pd.to_datetime(latest_pred['prediction_date'])
                age_hours = (pd.Timestamp.now() - pd.to_datetime(latest_pred['prediction_timestamp'])).total_seconds() / 3600

                # Show status based on prediction age
                if age_hours < 48:
                    status_text = "Fresh"
                elif age_hours < 168:  # 1 week
                    status_text = "Stale"
                else:
                    status_text = "VERY OLD"

                st.markdown(f"{status_text}: {pred_date.strftime('%Y-%m-%d')}")
                st.markdown(f"   {latest_pred['prediction_label']}")
                st.markdown(f"   ({age_hours:.1f}h ago)")
            else:
                st.markdown("No predictions yet")
                st.markdown("   (daily @ 4:30pm)")
        except Exception as e:
            st.markdown("Database error")

    # Quick Start
    st.markdown("---")
    st.markdown("### Quick Start")

    quickstart_col1, quickstart_col2 = st.columns(2)

    with quickstart_col1:
        st.markdown("""
        <div class="quickstart-box">
            <h4>1. First Time Setup</h4>
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
            <h4>2. Run Full Pipeline</h4>
            <p>
                # Automated pipeline (20-35 min)<br>
                python3 main.py → R<br><br>

                # Steps: Download → Train → Predict → Backtest → LLM
            </p>
        </div>
        """, unsafe_allow_html=True)

    # API Endpoints & Integration
    st.markdown("---")
    st.markdown("### REST API Integration")

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
        st.markdown("**Core Endpoints:**")
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
        st.markdown("**Example Usage (Python):**")
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
        st.markdown("**JavaScript Integration:**")
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
        st.markdown("**cURL Examples:**")
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
    st.markdown("### Client Integration")

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
    st.title("Asset Manager")
    st.markdown("*Download and manage data for all 41 supported assets*")

    # Asset category tabs
    tab1, tab2, tab3 = st.tabs(["Stocks & ETFs", "Precious Metals", "Crypto"])

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
                'Status': 'Downloaded' if status == 'downloaded' else 'Available',
                'Rows': rows if rows > 0 else '-'
            })

        df_assets = pd.DataFrame(asset_data)
        st.dataframe(df_assets, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Download SPY Data:**")
            if st.button("Update SPY Data", key="download_spy"):
                with st.spinner("Downloading SPY data..."):
                    result = subprocess.run(
                        ["python3", "update_spy_data.py"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("SPY data updated successfully!")
                        st.cache_data.clear()  # Clear cache to reload data
                    else:
                        st.error(f"Error: {result.stderr[:200]}")

            st.caption("Downloads S&P 500 data from 2000-present")

        with col2:
            st.markdown("**Download All ETFs:**")
            if st.button("Download ETFs & Bonds", key="download_etfs"):
                with st.spinner("Downloading 35+ assets..."):
                    result = subprocess.run(
                        ["python3", "download_equity_etfs.py"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("ETFs & bonds downloaded!")
                        st.cache_data.clear()
                    else:
                        st.error(f"Error: {result.stderr[:200]}")

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
                'Status': 'Downloaded' if status == 'downloaded' else 'Available',
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
                'Status': 'Downloaded' if status == 'downloaded' else 'Available',
                'Rows': rows if rows > 0 else '-'
            })

        df_crypto = pd.DataFrame(crypto_data)
        st.dataframe(df_crypto, use_container_width=True)

        st.markdown("---")

        st.markdown("**Download Cryptocurrency Data:**")
        if st.button("Download All 10 Cryptocurrencies", key="download_crypto"):
            with st.spinner("Downloading crypto data from Binance..."):
                result = subprocess.run(
                    ["python3", "download_crypto_enhanced.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("Cryptocurrency data downloaded!")
                    st.cache_data.clear()
                else:
                    st.error(f"Error: {result.stderr[:200]}")

        st.caption("Downloads BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK")

    # Quick Actions
    st.markdown("---")
    st.subheader("Quick Actions")

    qa_col1, qa_col2, qa_col3 = st.columns(3)

    with qa_col1:
        if st.button("Refresh All Data", key="refresh_display"):
            st.cache_data.clear()
            st.success("Cache cleared! Reload page to see fresh data.")

    with qa_col2:
        if st.button("Download All Assets", key="download_all"):
            with st.spinner("Downloading all assets..."):
                # SPY
                subprocess.run(["python3", "update_spy_data.py"], capture_output=True)
                # ETFs
                subprocess.run(["python3", "download_equity_etfs.py"], capture_output=True)
                # Crypto
                subprocess.run(["python3", "download_crypto_enhanced.py"], capture_output=True)
                st.success("All downloads complete!")
                st.cache_data.clear()

    with qa_col3:
        st.info("Individual downloads above for granular control")


def show_recession_indicator():
    """Recession probability analysis"""
    st.title("Recession Probability Indicator")
    st.markdown("*Multi-factor recession risk analysis*")

    st.info("Analyzes yield curves, market stress, and technical signals to assess recession risk")

    # Load SPY for analysis
    spy_df = load_asset_data('SPY')

    if spy_df is None or len(spy_df) < 200:
        st.error("Insufficient SPY Data")

        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #e74c3c;">
            <h3>Data Required</h3>
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
            st.markdown("**Download Fresh Data:**")
            st.code("python3 update_spy_data.py", language="bash")
            st.caption("Downloads SPY data from 2000 to present (~6,300 days)")

        with col2:
            st.markdown("**Quick Fix:**")
            st.code("python3 main.py\n# Select: 5 → 1 (Update SPY Data)", language="bash")
            st.caption("Use main menu for guided data download")

        st.info("After downloading, refresh this page using the button in the sidebar")
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
    st.subheader("Recession Risk Assessment")

    if recession_score > 70:
        st.error(f"HIGH RECESSION RISK (Score: {recession_score}/100)")
        st.markdown("""
        **Recommended Actions:**
        - Reduce equity exposure
        - Increase cash/bond allocation
        - Consider defensive sectors
        - Hedge positions if appropriate
        """)
    elif recession_score > 40:
        st.warning(f"MODERATE RECESSION RISK (Score: {recession_score}/100)")
        st.markdown("""
        **Recommended Actions:**
        - Monitor indicators closely
        - Reduce position sizes
        - Maintain cash reserves
        - Avoid aggressive strategies
        """)
    else:
        st.success(f"LOW RECESSION RISK (Score: {recession_score}/100)")
        st.markdown("""
        **Market Conditions:**
        - Normal market conditions
        - Standard strategies appropriate
        - Maintain diversified portfolio
        """)

    # Signals
    st.markdown("---")
    st.subheader("Technical Signals")

    col1, col2 = st.columns(2)

    with col1:
        if death_cross:
            st.markdown("**Death Cross** - 50-MA below 200-MA (Bearish)")
        else:
            st.markdown("**Golden Cross** - 50-MA above 200-MA (Bullish)")

        if latest['Close'] < ma_200:
            st.markdown("**Below 200-MA** - Bearish trend")
        else:
            st.markdown("**Above 200-MA** - Bullish trend")

    with col2:
        st.markdown(f"**Stress Score:** {min(100, (volatility * 2 + abs(drawdown)) / 2):.1f}/100")

        if volatility > 25:
            st.markdown("**High Volatility** - Market stress")
        else:
            st.markdown("**Normal Volatility**")

    # Price chart
    st.markdown("---")
    st.subheader("Price vs Moving Averages")

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
    st.title("Valuation Detector")
    st.markdown("*Comprehensive asset valuation analysis using multiple technical indicators*")

    # Evaluation timestamp
    eval_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    st.caption(f"Analysis Date: {eval_time}")

    # Asset selector
    all_assets = list(STOCK_ETFS.keys()) + list(PRECIOUS_METALS.keys()) + list(CRYPTO_ASSETS.keys())
    downloaded_assets = [a for a in all_assets if check_asset_status(a) == "downloaded"]

    if not downloaded_assets:
        st.warning("No assets downloaded. Visit Asset Manager to download data.")
        return

    col_select, col_period = st.columns([2, 1])
    with col_select:
        selected_asset = st.selectbox("Select Asset", downloaded_assets)
    with col_period:
        analysis_period = st.selectbox("Analysis Period", ["1 Year", "6 Months", "3 Months"], index=0)

    df = load_asset_data(selected_asset)

    if df is None or len(df) < 100:
        st.warning(f"Insufficient data for {selected_asset}")
        return

    # Set period based on selection
    period_days = {"1 Year": 252, "6 Months": 126, "3 Months": 63}
    days = period_days.get(analysis_period, 252)
    recent = df.tail(max(days, 252))  # Need at least 252 for some calcs
    analysis_window = df.tail(days)
    latest = recent.iloc[-1]

    # Get latest data date
    latest_data_date = latest['Date'].strftime("%B %d, %Y") if 'Date' in latest else "Unknown"

    # ==================== CALCULATE ALL METRICS ====================

    # 1. RSI (14-day)
    delta = recent['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    latest_rsi = rsi_series.iloc[-1]

    # 2. Z-Score (252-day)
    mean_price = recent['Close'].mean()
    std_price = recent['Close'].std()
    z_score = (latest['Close'] - mean_price) / std_price if std_price > 0 else 0

    # 3. Bollinger Bands
    sma_20 = recent['Close'].rolling(20).mean()
    std_20 = recent['Close'].rolling(20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_position = ((latest['Close'] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])) * 100 if upper_band.iloc[-1] > lower_band.iloc[-1] else 50

    # 4. Moving Average Deviations
    ma_50 = recent['Close'].rolling(50).mean().iloc[-1] if len(recent) >= 50 else recent['Close'].mean()
    ma_200 = recent['Close'].rolling(200).mean().iloc[-1] if len(recent) >= 200 else recent['Close'].mean()
    ma_50_deviation = ((latest['Close'] - ma_50) / ma_50) * 100
    ma_200_deviation = ((latest['Close'] - ma_200) / ma_200) * 100

    # 5. MACD
    ema_12 = recent['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = recent['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_histogram = macd_histogram.iloc[-1]

    # 6. Stochastic Oscillator (14-day)
    low_14 = recent['Low'].rolling(14).min()
    high_14 = recent['High'].rolling(14).max()
    stoch_k = ((recent['Close'] - low_14) / (high_14 - low_14)) * 100
    stoch_d = stoch_k.rolling(3).mean()
    latest_stoch_k = stoch_k.iloc[-1]
    latest_stoch_d = stoch_d.iloc[-1]

    # 7. Williams %R (14-day)
    williams_r = ((high_14 - recent['Close']) / (high_14 - low_14)) * -100
    latest_williams = williams_r.iloc[-1]

    # 8. CCI (Commodity Channel Index, 20-day)
    typical_price = (recent['High'] + recent['Low'] + recent['Close']) / 3
    tp_sma = typical_price.rolling(20).mean()
    tp_mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - tp_sma) / (0.015 * tp_mad)
    latest_cci = cci.iloc[-1]

    # 9. Rate of Change (multiple periods)
    roc_10 = ((latest['Close'] - recent.iloc[-10]['Close']) / recent.iloc[-10]['Close']) * 100 if len(recent) >= 10 else 0
    roc_30 = ((latest['Close'] - recent.iloc[-30]['Close']) / recent.iloc[-30]['Close']) * 100 if len(recent) >= 30 else 0
    roc_90 = ((latest['Close'] - recent.iloc[-90]['Close']) / recent.iloc[-90]['Close']) * 100 if len(recent) >= 90 else 0

    # 10. ATR (Average True Range, 14-day)
    high_low = recent['High'] - recent['Low']
    high_close = np.abs(recent['High'] - recent['Close'].shift())
    low_close = np.abs(recent['Low'] - recent['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    latest_atr = atr.iloc[-1]
    atr_percent = (latest_atr / latest['Close']) * 100

    # 11. Volume indicators (if volume available)
    has_volume = 'Volume' in recent.columns and recent['Volume'].sum() > 0
    if has_volume:
        avg_volume_20 = recent['Volume'].rolling(20).mean().iloc[-1]
        latest_volume = recent['Volume'].iloc[-1]
        volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        # OBV trend
        obv = (np.sign(recent['Close'].diff()) * recent['Volume']).fillna(0).cumsum()
        obv_sma = obv.rolling(20).mean()
        obv_trend = "Bullish" if obv.iloc[-1] > obv_sma.iloc[-1] else "Bearish"
    else:
        volume_ratio = None
        obv_trend = None

    # 12. Price vs 52-week range
    high_52w = recent['High'].max()
    low_52w = recent['Low'].min()
    range_52w_position = ((latest['Close'] - low_52w) / (high_52w - low_52w)) * 100 if high_52w > low_52w else 50

    # ==================== VALUATION SCORING ====================

    # Build comprehensive valuation score with weighted signals
    signals = []

    # RSI Signal
    if latest_rsi > 70:
        signals.append(("RSI", "Overbought", 0.15, "bearish"))
    elif latest_rsi > 60:
        signals.append(("RSI", "Elevated", 0.05, "bearish"))
    elif latest_rsi < 30:
        signals.append(("RSI", "Oversold", -0.15, "bullish"))
    elif latest_rsi < 40:
        signals.append(("RSI", "Depressed", -0.05, "bullish"))
    else:
        signals.append(("RSI", "Neutral", 0, "neutral"))

    # Z-Score Signal
    if z_score > 2:
        signals.append(("Z-Score", "Significantly Elevated", 0.15, "bearish"))
    elif z_score > 1:
        signals.append(("Z-Score", "Above Average", 0.08, "bearish"))
    elif z_score < -2:
        signals.append(("Z-Score", "Significantly Depressed", -0.15, "bullish"))
    elif z_score < -1:
        signals.append(("Z-Score", "Below Average", -0.08, "bullish"))
    else:
        signals.append(("Z-Score", "Near Mean", 0, "neutral"))

    # Bollinger Band Signal
    if bb_position > 95:
        signals.append(("Bollinger %", "Extreme Upper", 0.12, "bearish"))
    elif bb_position > 80:
        signals.append(("Bollinger %", "Upper Band", 0.06, "bearish"))
    elif bb_position < 5:
        signals.append(("Bollinger %", "Extreme Lower", -0.12, "bullish"))
    elif bb_position < 20:
        signals.append(("Bollinger %", "Lower Band", -0.06, "bullish"))
    else:
        signals.append(("Bollinger %", "Mid Range", 0, "neutral"))

    # MA Deviation Signals
    if ma_200_deviation > 30:
        signals.append(("200-MA Dev", "Far Above", 0.10, "bearish"))
    elif ma_200_deviation > 15:
        signals.append(("200-MA Dev", "Above", 0.05, "bearish"))
    elif ma_200_deviation < -30:
        signals.append(("200-MA Dev", "Far Below", -0.10, "bullish"))
    elif ma_200_deviation < -15:
        signals.append(("200-MA Dev", "Below", -0.05, "bullish"))
    else:
        signals.append(("200-MA Dev", "Near MA", 0, "neutral"))

    # Stochastic Signal
    if latest_stoch_k > 80 and latest_stoch_d > 80:
        signals.append(("Stochastic", "Overbought", 0.08, "bearish"))
    elif latest_stoch_k < 20 and latest_stoch_d < 20:
        signals.append(("Stochastic", "Oversold", -0.08, "bullish"))
    else:
        signals.append(("Stochastic", "Neutral", 0, "neutral"))

    # Williams %R Signal
    if latest_williams > -20:
        signals.append(("Williams %R", "Overbought", 0.06, "bearish"))
    elif latest_williams < -80:
        signals.append(("Williams %R", "Oversold", -0.06, "bullish"))
    else:
        signals.append(("Williams %R", "Neutral", 0, "neutral"))

    # CCI Signal
    if latest_cci > 200:
        signals.append(("CCI", "Extreme Overbought", 0.10, "bearish"))
    elif latest_cci > 100:
        signals.append(("CCI", "Overbought", 0.05, "bearish"))
    elif latest_cci < -200:
        signals.append(("CCI", "Extreme Oversold", -0.10, "bullish"))
    elif latest_cci < -100:
        signals.append(("CCI", "Oversold", -0.05, "bullish"))
    else:
        signals.append(("CCI", "Neutral", 0, "neutral"))

    # MACD Signal
    if latest_macd > latest_signal and latest_histogram > 0:
        signals.append(("MACD", "Bullish Cross", -0.05, "bullish"))
    elif latest_macd < latest_signal and latest_histogram < 0:
        signals.append(("MACD", "Bearish Cross", 0.05, "bearish"))
    else:
        signals.append(("MACD", "Neutral", 0, "neutral"))

    # 52-Week Range Position
    if range_52w_position > 90:
        signals.append(("52W Range", "Near High", 0.08, "bearish"))
    elif range_52w_position < 10:
        signals.append(("52W Range", "Near Low", -0.08, "bullish"))
    else:
        signals.append(("52W Range", "Mid Range", 0, "neutral"))

    # Calculate total valuation score
    valuation_score = sum(s[2] for s in signals)
    valuation_score = max(-1.0, min(1.0, valuation_score))  # Clamp to [-1, 1]

    # Count signals
    bullish_count = sum(1 for s in signals if s[3] == "bullish")
    bearish_count = sum(1 for s in signals if s[3] == "bearish")
    neutral_count = sum(1 for s in signals if s[3] == "neutral")

    # Classification
    if valuation_score > 0.4:
        classification = "OVERVALUED"
        classification_color = "error"
    elif valuation_score > 0.2:
        classification = "SLIGHTLY OVERVALUED"
        classification_color = "warning"
    elif valuation_score < -0.4:
        classification = "UNDERVALUED"
        classification_color = "success"
    elif valuation_score < -0.2:
        classification = "SLIGHTLY UNDERVALUED"
        classification_color = "info"
    else:
        classification = "FAIRLY VALUED"
        classification_color = "info"

    # ==================== DISPLAY ====================

    # Header with price info
    st.markdown("---")
    header_cols = st.columns([2, 1, 1, 1])
    with header_cols[0]:
        st.markdown(f"### {selected_asset}")
        st.markdown(f"**${latest['Close']:.2f}** as of {latest_data_date}")
    with header_cols[1]:
        st.metric("52W High", f"${high_52w:.2f}")
    with header_cols[2]:
        st.metric("52W Low", f"${low_52w:.2f}")
    with header_cols[3]:
        st.metric("52W Position", f"{range_52w_position:.1f}%")

    # Main valuation verdict
    st.markdown("---")
    verdict_cols = st.columns([2, 1, 1])

    with verdict_cols[0]:
        if classification_color == "error":
            st.error(f"## {classification}")
        elif classification_color == "warning":
            st.warning(f"## {classification}")
        elif classification_color == "success":
            st.success(f"## {classification}")
        else:
            st.info(f"## {classification}")

    with verdict_cols[1]:
        st.metric("Valuation Score", f"{valuation_score:+.2f}", help="Range: -1.0 (undervalued) to +1.0 (overvalued)")

    with verdict_cols[2]:
        st.markdown("**Signal Summary**")
        st.markdown(f"Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {neutral_count}")

    # Recommendation
    if classification in ["OVERVALUED", "SLIGHTLY OVERVALUED"]:
        st.markdown("**Recommendation:** Consider taking profits or reducing position size. Wait for pullback before adding.")
    elif classification in ["UNDERVALUED", "SLIGHTLY UNDERVALUED"]:
        st.markdown("**Recommendation:** Potential accumulation opportunity. Consider building position on further weakness.")
    else:
        st.markdown("**Recommendation:** Asset trading near fair value. Hold existing positions, wait for clearer signals.")

    # ==================== DETAILED METRICS TABS ====================

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Momentum Indicators", "Trend Indicators", "Volatility Metrics", "Charts"])

    with tab1:
        st.subheader("Momentum Indicators")
        st.markdown("*Measure the speed and magnitude of price movements*")

        # RSI Card
        st.markdown("#### RSI (Relative Strength Index)")
        rsi_cols = st.columns([1, 2])
        with rsi_cols[0]:
            st.metric("RSI (14)", f"{latest_rsi:.1f}")
            rsi_signal = [s for s in signals if s[0] == "RSI"][0]
            if rsi_signal[3] == "bullish":
                st.success(f"Signal: {rsi_signal[1]}")
            elif rsi_signal[3] == "bearish":
                st.error(f"Signal: {rsi_signal[1]}")
            else:
                st.info(f"Signal: {rsi_signal[1]}")
        with rsi_cols[1]:
            st.markdown("""
            **Interpretation:**
            - RSI > 70: Overbought - momentum may be exhausted, potential reversal
            - RSI < 30: Oversold - selling may be exhausted, potential bounce
            - RSI 30-70: Neutral zone - trend following appropriate

            **Current Reading:** """ + (
                "Strong upward momentum suggests caution for new longs" if latest_rsi > 70 else
                "Depressed momentum may present buying opportunity" if latest_rsi < 30 else
                "Momentum is balanced, follow the prevailing trend"
            ))

        st.markdown("---")

        # Stochastic Card
        st.markdown("#### Stochastic Oscillator")
        stoch_cols = st.columns([1, 2])
        with stoch_cols[0]:
            st.metric("%K", f"{latest_stoch_k:.1f}")
            st.metric("%D", f"{latest_stoch_d:.1f}")
            stoch_signal = [s for s in signals if s[0] == "Stochastic"][0]
            if stoch_signal[3] == "bullish":
                st.success(f"Signal: {stoch_signal[1]}")
            elif stoch_signal[3] == "bearish":
                st.error(f"Signal: {stoch_signal[1]}")
            else:
                st.info(f"Signal: {stoch_signal[1]}")
        with stoch_cols[1]:
            st.markdown("""
            **Interpretation:**
            - Both %K and %D > 80: Overbought territory
            - Both %K and %D < 20: Oversold territory
            - %K crossing above %D: Bullish signal
            - %K crossing below %D: Bearish signal

            **Current Reading:** """ + (
                "Both lines in overbought zone, watch for bearish crossover" if latest_stoch_k > 80 and latest_stoch_d > 80 else
                "Both lines in oversold zone, watch for bullish crossover" if latest_stoch_k < 20 and latest_stoch_d < 20 else
                "Stochastic in neutral territory"
            ))

        st.markdown("---")

        # Williams %R Card
        st.markdown("#### Williams %R")
        wr_cols = st.columns([1, 2])
        with wr_cols[0]:
            st.metric("Williams %R", f"{latest_williams:.1f}")
            wr_signal = [s for s in signals if s[0] == "Williams %R"][0]
            if wr_signal[3] == "bullish":
                st.success(f"Signal: {wr_signal[1]}")
            elif wr_signal[3] == "bearish":
                st.error(f"Signal: {wr_signal[1]}")
            else:
                st.info(f"Signal: {wr_signal[1]}")
        with wr_cols[1]:
            st.markdown("""
            **Interpretation:**
            - Williams %R > -20: Overbought (near period high)
            - Williams %R < -80: Oversold (near period low)
            - Range: -100 to 0

            **Current Reading:** """ + (
                "Price near recent highs, overbought conditions" if latest_williams > -20 else
                "Price near recent lows, oversold conditions" if latest_williams < -80 else
                "Price in middle of recent range"
            ))

        st.markdown("---")

        # CCI Card
        st.markdown("#### CCI (Commodity Channel Index)")
        cci_cols = st.columns([1, 2])
        with cci_cols[0]:
            st.metric("CCI (20)", f"{latest_cci:.1f}")
            cci_signal = [s for s in signals if s[0] == "CCI"][0]
            if cci_signal[3] == "bullish":
                st.success(f"Signal: {cci_signal[1]}")
            elif cci_signal[3] == "bearish":
                st.error(f"Signal: {cci_signal[1]}")
            else:
                st.info(f"Signal: {cci_signal[1]}")
        with cci_cols[1]:
            st.markdown("""
            **Interpretation:**
            - CCI > +200: Extremely overbought
            - CCI > +100: Overbought, strong uptrend
            - CCI < -100: Oversold, strong downtrend
            - CCI < -200: Extremely oversold

            **Current Reading:** """ + (
                "Extreme overbought - high probability of pullback" if latest_cci > 200 else
                "Overbought - uptrend may be overextended" if latest_cci > 100 else
                "Extreme oversold - high probability of bounce" if latest_cci < -200 else
                "Oversold - downtrend may be overextended" if latest_cci < -100 else
                "CCI in normal range"
            ))

    with tab2:
        st.subheader("Trend Indicators")
        st.markdown("*Identify price direction and trend strength*")

        # Z-Score Card
        st.markdown("#### Z-Score (Statistical Deviation)")
        zscore_cols = st.columns([1, 2])
        with zscore_cols[0]:
            st.metric("Z-Score", f"{z_score:+.2f}")
            zscore_signal = [s for s in signals if s[0] == "Z-Score"][0]
            if zscore_signal[3] == "bullish":
                st.success(f"Signal: {zscore_signal[1]}")
            elif zscore_signal[3] == "bearish":
                st.error(f"Signal: {zscore_signal[1]}")
            else:
                st.info(f"Signal: {zscore_signal[1]}")
        with zscore_cols[1]:
            st.markdown(f"""
            **Interpretation:**
            - Z > +2: Price 2+ std devs above mean (top ~2.5%)
            - Z > +1: Price above average
            - Z < -1: Price below average
            - Z < -2: Price 2+ std devs below mean (bottom ~2.5%)

            **Statistics (252-day):**
            - Mean Price: ${mean_price:.2f}
            - Std Dev: ${std_price:.2f}
            - Current: ${latest['Close']:.2f}
            """)

        st.markdown("---")

        # Moving Average Card
        st.markdown("#### Moving Average Analysis")
        ma_cols = st.columns([1, 2])
        with ma_cols[0]:
            st.metric("50-MA", f"${ma_50:.2f}")
            st.metric("200-MA", f"${ma_200:.2f}")
            st.metric("50-MA Deviation", f"{ma_50_deviation:+.1f}%")
            st.metric("200-MA Deviation", f"{ma_200_deviation:+.1f}%")
        with ma_cols[1]:
            ma_cross = "Golden Cross (Bullish)" if ma_50 > ma_200 else "Death Cross (Bearish)"
            st.markdown(f"""
            **Interpretation:**
            - Price > 200-MA: Long-term uptrend
            - Price < 200-MA: Long-term downtrend
            - 50-MA > 200-MA: Golden Cross (bullish)
            - 50-MA < 200-MA: Death Cross (bearish)

            **Current Status:**
            - MA Cross: {ma_cross}
            - Price is {ma_200_deviation:+.1f}% from 200-MA
            - {"Above both MAs - bullish structure" if latest['Close'] > ma_50 and latest['Close'] > ma_200 else "Below both MAs - bearish structure" if latest['Close'] < ma_50 and latest['Close'] < ma_200 else "Between MAs - mixed signals"}
            """)

        st.markdown("---")

        # MACD Card
        st.markdown("#### MACD (Moving Average Convergence Divergence)")
        macd_cols = st.columns([1, 2])
        with macd_cols[0]:
            st.metric("MACD Line", f"{latest_macd:.4f}")
            st.metric("Signal Line", f"{latest_signal:.4f}")
            st.metric("Histogram", f"{latest_histogram:+.4f}")
            macd_signal = [s for s in signals if s[0] == "MACD"][0]
            if macd_signal[3] == "bullish":
                st.success(f"Signal: {macd_signal[1]}")
            elif macd_signal[3] == "bearish":
                st.error(f"Signal: {macd_signal[1]}")
            else:
                st.info(f"Signal: {macd_signal[1]}")
        with macd_cols[1]:
            st.markdown("""
            **Interpretation:**
            - MACD > Signal: Bullish momentum
            - MACD < Signal: Bearish momentum
            - Histogram expanding: Momentum increasing
            - Histogram contracting: Momentum decreasing

            **Current Reading:** """ + (
                "MACD above signal line with positive histogram - bullish momentum" if latest_macd > latest_signal and latest_histogram > 0 else
                "MACD below signal line with negative histogram - bearish momentum" if latest_macd < latest_signal and latest_histogram < 0 else
                "MACD near signal line - momentum neutral/transitioning"
            ))

        st.markdown("---")

        # Rate of Change Card
        st.markdown("#### Rate of Change (Price Momentum)")
        roc_cols = st.columns(3)
        with roc_cols[0]:
            st.metric("10-Day ROC", f"{roc_10:+.1f}%")
        with roc_cols[1]:
            st.metric("30-Day ROC", f"{roc_30:+.1f}%")
        with roc_cols[2]:
            st.metric("90-Day ROC", f"{roc_90:+.1f}%")

        st.markdown("""
        **Interpretation:** ROC measures percentage price change over specified periods.
        Positive values indicate upward momentum, negative values indicate downward momentum.
        Compare short vs long-term ROC to identify momentum divergences.
        """)

    with tab3:
        st.subheader("Volatility & Range Metrics")
        st.markdown("*Measure price variability and trading ranges*")

        # Bollinger Bands Card
        st.markdown("#### Bollinger Bands")
        bb_cols = st.columns([1, 2])
        with bb_cols[0]:
            st.metric("Upper Band", f"${upper_band.iloc[-1]:.2f}")
            st.metric("Middle (SMA20)", f"${sma_20.iloc[-1]:.2f}")
            st.metric("Lower Band", f"${lower_band.iloc[-1]:.2f}")
            st.metric("Band Position", f"{bb_position:.1f}%")
            bb_signal = [s for s in signals if s[0] == "Bollinger %"][0]
            if bb_signal[3] == "bullish":
                st.success(f"Signal: {bb_signal[1]}")
            elif bb_signal[3] == "bearish":
                st.error(f"Signal: {bb_signal[1]}")
            else:
                st.info(f"Signal: {bb_signal[1]}")
        with bb_cols[1]:
            band_width = ((upper_band.iloc[-1] - lower_band.iloc[-1]) / sma_20.iloc[-1]) * 100
            st.markdown(f"""
            **Interpretation:**
            - Price near upper band (>80%): Overbought/strong uptrend
            - Price near lower band (<20%): Oversold/strong downtrend
            - Band Width: {band_width:.1f}% (wider = higher volatility)

            **Current Reading:**
            - Price at {bb_position:.1f}% of band range
            - {"Near upper band - potential resistance" if bb_position > 80 else "Near lower band - potential support" if bb_position < 20 else "Mid-band - no extreme reading"}
            """)

        st.markdown("---")

        # ATR Card
        st.markdown("#### ATR (Average True Range)")
        atr_cols = st.columns([1, 2])
        with atr_cols[0]:
            st.metric("ATR (14)", f"${latest_atr:.2f}")
            st.metric("ATR %", f"{atr_percent:.2f}%")
        with atr_cols[1]:
            st.markdown(f"""
            **Interpretation:**
            - ATR measures average daily price movement
            - Higher ATR = Higher volatility
            - ATR % shows volatility relative to price

            **Current Reading:**
            - Average daily range: ${latest_atr:.2f}
            - Volatility: {atr_percent:.2f}% of price
            - {"High volatility - use wider stops" if atr_percent > 3 else "Moderate volatility" if atr_percent > 1.5 else "Low volatility - tighter ranges expected"}
            """)

        st.markdown("---")

        # 52-Week Range Card
        st.markdown("#### 52-Week Range Analysis")
        range_cols = st.columns([1, 2])
        with range_cols[0]:
            st.metric("52W High", f"${high_52w:.2f}")
            st.metric("52W Low", f"${low_52w:.2f}")
            st.metric("Range Position", f"{range_52w_position:.1f}%")
            range_signal = [s for s in signals if s[0] == "52W Range"][0]
            if range_signal[3] == "bullish":
                st.success(f"Signal: {range_signal[1]}")
            elif range_signal[3] == "bearish":
                st.error(f"Signal: {range_signal[1]}")
            else:
                st.info(f"Signal: {range_signal[1]}")
        with range_cols[1]:
            from_high = ((latest['Close'] - high_52w) / high_52w) * 100
            from_low = ((latest['Close'] - low_52w) / low_52w) * 100
            st.markdown(f"""
            **Current Position:**
            - {from_high:+.1f}% from 52-week high
            - {from_low:+.1f}% from 52-week low
            - Trading at {range_52w_position:.1f}% of annual range

            **Interpretation:**
            - Near 52W high (>90%): Strong momentum but extended
            - Near 52W low (<10%): Weak momentum but potentially oversold
            """)

        # Volume section (if available)
        if has_volume:
            st.markdown("---")
            st.markdown("#### Volume Analysis")
            vol_cols = st.columns([1, 2])
            with vol_cols[0]:
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                st.metric("OBV Trend", obv_trend)
            with vol_cols[1]:
                st.markdown(f"""
                **Interpretation:**
                - Volume Ratio: Current volume vs 20-day average
                - Ratio > 1.5: High volume (strong conviction)
                - Ratio < 0.5: Low volume (weak conviction)

                **OBV (On-Balance Volume):**
                - OBV rising: Accumulation (buying pressure)
                - OBV falling: Distribution (selling pressure)
                - Current trend: {obv_trend}
                """)

    with tab4:
        st.subheader("Technical Charts")

        chart_type = st.selectbox("Select Chart", [
            "Price with Bollinger Bands",
            "RSI History",
            "MACD",
            "Stochastic Oscillator",
            "Moving Averages"
        ])

        if chart_type == "Price with Bollinger Bands":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                     mode='lines', name='Price', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=upper_band.tail(days),
                                     mode='lines', name='Upper BB', line=dict(color='red', width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=sma_20.tail(days),
                                     mode='lines', name='SMA(20)', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=lower_band.tail(days),
                                     mode='lines', name='Lower BB', line=dict(color='green', width=1, dash='dash')))
            fig.update_layout(title=f"{selected_asset} - Bollinger Bands", height=500)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "RSI History":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=rsi_series.tail(days),
                                     mode='lines', name='RSI', line=dict(color='purple', width=2)))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig.add_hline(y=50, line_dash="dot", line_color="gray")
            fig.update_layout(title=f"{selected_asset} - RSI (14)", height=400, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "MACD":
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                               row_heights=[0.6, 0.4])
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                     mode='lines', name='Price', line=dict(color='black')), row=1, col=1)
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=macd_line.tail(days),
                                     mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=signal_line.tail(days),
                                     mode='lines', name='Signal', line=dict(color='orange')), row=2, col=1)
            colors = ['green' if val >= 0 else 'red' for val in macd_histogram.tail(days)]
            fig.add_trace(go.Bar(x=analysis_window['Date'], y=macd_histogram.tail(days),
                                 name='Histogram', marker_color=colors), row=2, col=1)
            fig.update_layout(title=f"{selected_asset} - MACD", height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Stochastic Oscillator":
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                               row_heights=[0.6, 0.4])
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                     mode='lines', name='Price', line=dict(color='black')), row=1, col=1)
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=stoch_k.tail(days),
                                     mode='lines', name='%K', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=stoch_d.tail(days),
                                     mode='lines', name='%D', line=dict(color='orange')), row=2, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(title=f"{selected_asset} - Stochastic Oscillator", height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Moving Averages":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                     mode='lines', name='Price', line=dict(color='black', width=2)))
            ma_50_series = recent['Close'].rolling(50).mean()
            ma_200_series = recent['Close'].rolling(200).mean()
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=ma_50_series.tail(days),
                                     mode='lines', name='50-MA', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=analysis_window['Date'], y=ma_200_series.tail(days),
                                     mode='lines', name='200-MA', line=dict(color='red', width=1)))
            fig.update_layout(title=f"{selected_asset} - Moving Averages", height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ==================== SIGNAL SUMMARY TABLE ====================

    st.markdown("---")
    st.subheader("Complete Signal Summary")

    signal_data = []
    for name, reading, score, direction in signals:
        signal_data.append({
            "Indicator": name,
            "Reading": reading,
            "Score": f"{score:+.2f}",
            "Signal": direction.upper()
        })

    signal_df = pd.DataFrame(signal_data)
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This analysis is for informational purposes only and does not constitute financial advice.
    Technical indicators are based on historical data and may not predict future performance.
    Always conduct your own research and consider consulting a financial advisor before making investment decisions.
    """)


def show_llm_analysis():
    """LLM-powered market analysis"""
    st.title("LLM Market Analysis")
    st.markdown("*AI-powered market commentary using GPT-4 or Claude*")

    st.info("Integrates with OpenAI/Anthropic to generate market analysis based on predictions, price data, and news")

    # Configuration
    st.subheader("Configuration")

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
    st.subheader("Sample Analysis Output")

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
    st.subheader("Run LLM Analysis")

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
    st.title("Portfolio Rebalancing Optimizer")
    st.markdown("*Find optimal rebalancing frequency for your portfolio*")

    st.info("Tests different rebalancing frequencies (daily, weekly, monthly, etc.) and identifies the optimal strategy based on returns, Sharpe ratio, and transaction costs")

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
    st.subheader("Example Results")

    example_data = {
        'Strategy': ['Buy & Hold', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annual', 'Annual'],
        'Gross Return': ['45.2%', '52.1%', '51.3%', '50.8%', '48.5%', '46.8%', '45.9%'],
        'Sharpe Ratio': [1.85, 1.92, 1.95, 2.10, 2.08, 1.98, 1.88],
        'Max Drawdown': ['-8.2%', '-7.1%', '-7.3%', '-7.0%', '-7.5%', '-8.0%', '-8.3%'],
        'Rebalances': [0, 1260, 260, 60, 20, 10, 5],
        'Net Return': ['45.2%', '27.6%', '41.9%', '49.6%', '48.1%', '46.5%', '45.7%'],
        'Recommended': ['', '', '', 'Yes', '', '', '']
    }

    df_example = pd.DataFrame(example_data)
    st.dataframe(df_example, use_container_width=True)

    st.success("**Optimal Strategy:** Monthly rebalancing provides best risk-adjusted returns")

    # Commands
    st.markdown("---")
    st.subheader("Run Rebalancing Analysis")

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
    """Display forecast results from PostgreSQL"""
    st.title("Forecast Results")
    st.markdown("*View predictions generated by the forecasting API*")

    # Load predictions from PostgreSQL
    try:
        dm = get_data_manager()
        predictions_df = dm.get_latest_predictions(limit=1000)
        dm.close()

        if len(predictions_df) == 0:
            st.warning("No forecast results available. Predictions generate daily at 4:30 PM EST.")
            return

        # Map database columns to display format
        df = predictions_df.copy()

        # Map prediction labels to numbers for signal distribution
        label_map = {'CRASH': 0, 'NORMAL': 1, 'SPIKE': 2}
        df['Prediction'] = df['prediction_label'].map(label_map)

        # Rename columns for display
        df['Date'] = pd.to_datetime(df['prediction_date'])
        df['Proba'] = df['prediction_proba']
        df['Confidence'] = df['confidence']

    except Exception as e:
        st.error(f"Error loading predictions from database: {e}")
        return

    # Signal distribution
    st.subheader("Signal Distribution")

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
    st.subheader("Recent Forecasts (Last 20)")

    recent = df.tail(20).copy()
    if 'Prediction' in recent.columns:
        recent['Signal'] = recent['Prediction'].map({0: 'CRASH', 1: 'NORMAL', 2: 'SPIKE'})

    display_cols = ['Date', 'Signal', 'Proba', 'Confidence']
    available_cols = [col for col in display_cols if col in recent.columns]

    if available_cols:
        st.dataframe(recent[available_cols].sort_values('Date', ascending=False), use_container_width=True)


def show_automation():
    """Automation and pipeline execution"""
    st.title("Automation & Pipeline")
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
    st.subheader("Run Complete Pipeline")

    st.code("""
python3 main.py
# Then select "Option R" - Run Full Pipeline
    """)

    st.markdown("---")

    # Individual commands
    st.subheader("Individual Commands")

    tab1, tab2, tab3, tab4 = st.tabs(["Download Data", "Train Models", "Generate Forecasts", "Validate"])

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
    st.title("Custom Data Imports")
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

            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {list(df.columns)}")

            # Import settings
            st.subheader("Import Settings")

            col1, col2 = st.columns(2)

            with col1:
                ticker = st.text_input("Ticker Symbol", value="CUSTOM")

            with col2:
                asset_type = st.selectbox("Asset Type", ["Stock", "ETF", "Crypto", "Other"])

            if st.button("Import Data", type="primary"):
                # Get current user (create demo user if not exists)
                current_user_id = AuthManager.get_session_user()

                # Save to PostgreSQL with user isolation
                success, result = save_custom_asset_to_db(
                    ticker=ticker,
                    asset_type=asset_type.lower(),
                    df=df,
                    user_id=current_user_id
                )

                if success:
                    st.success(f"Successfully imported {ticker} ({result} records)")
                    st.info("Saved to PostgreSQL - persists across restarts")
                    st.info(f"User-specific asset (only visible to you)")
                    st.markdown("**Next Steps:**")
                    st.markdown(f"""
- Navigate to **Market Forecast** to generate predictions for `{ticker}`
- Your custom asset is now available for analysis
- Data will persist even if the server restarts
                    """)
                else:
                    st.error(f"Import failed: {result}")
                    st.info("Check that your CSV has 'Date' and 'Close' columns")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # CLI import
    st.markdown("---")
    st.subheader("Command Line Import")

    st.code("""
# Import with validation
python3 import_custom_asset.py my_data.csv MYTICKER

# Create sample template
python3 import_custom_asset.py --sample
    """)


if __name__ == "__main__":
    main()
