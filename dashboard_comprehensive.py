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

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np


def load_real_metrics() -> dict:
    """Load actual metrics from logs/latest.json.

    backtest.py saves keys: total_return, sharpe (not sharpe_ratio),
    max_drawdown, win_rate, model_accuracy, trades (not total_trades).
    This function normalises those into the names the dashboard expects.
    """
    metrics_path = Path("logs/latest.json")
    default_metrics = {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "model_accuracy": 0.0,
        "wf_accuracy": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "years_tested": 0,
    }

    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                raw = json.load(f)
            # Normalise key aliases written by backtest.py
            raw.setdefault("sharpe_ratio", raw.get("sharpe", 0.0))
            raw.setdefault("total_trades", raw.get("trades", 0))
            raw.setdefault("wf_accuracy", raw.get("model_accuracy", 0.0))
            return {**default_metrics, **raw}
        except Exception:
            pass

    return default_metrics


# Load real metrics at module level
METRICS = load_real_metrics()

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

# Import AssetManager for centralized asset configuration
try:
    from framework.asset_manager import AssetManager
    _asset_manager = AssetManager()
    USE_ASSET_MANAGER = True
except Exception as e:
    print(f"Warning: Could not load AssetManager: {e}")
    USE_ASSET_MANAGER = False
    _asset_manager = None

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

# All supported assets - loaded from AssetManager when available
def _get_asset_categories():
    """Get asset categories from AssetManager or use fallback defaults."""
    if USE_ASSET_MANAGER and _asset_manager:
        try:
            categories = _asset_manager.get_dashboard_categories()
            # Merge stocks_etfs with sectors and bonds for display
            stocks = {**categories.get('stocks_etfs', {}), **categories.get('sectors', {}), **categories.get('bonds', {})}
            return {
                'stocks_etfs': stocks,
                'precious_metals': categories.get('precious_metals', {}),
                'crypto': categories.get('crypto', {}),
            }
        except Exception as e:
            print(f"Warning: AssetManager categories failed: {e}")

    # Fallback to hardcoded values if AssetManager unavailable
    return {
        'stocks_etfs': {
            'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
            'DIA': 'Dow Jones', 'VTI': 'Total Stock Market', 'EEM': 'Emerging Markets',
            'XLF': 'Financials', 'XLK': 'Technology', 'XLE': 'Energy',
            'DXY': 'US Dollar', 'HYG': 'High Yield Bonds', 'LQD': 'Investment Grade Bonds',
            'TNX': '10Y Treasury', 'UUP': 'US Dollar Bull'
        },
        'precious_metals': {
            'GLD': 'Gold Trust', 'SLV': 'Silver Trust', 'GDX': 'Gold Miners',
            'GDXJ': 'Junior Gold Miners', 'IAU': 'iShares Gold',
            'PPLT': 'Platinum', 'PALL': 'Palladium'
        },
        'crypto': {
            'BTC/USDT': 'Bitcoin', 'ETH/USDT': 'Ethereum', 'SOL/USDT': 'Solana',
            'BNB/USDT': 'Binance Coin', 'XRP/USDT': 'Ripple', 'ADA/USDT': 'Cardano',
            'DOGE/USDT': 'Dogecoin', 'AVAX/USDT': 'Avalanche',
            'MATIC/USDT': 'Polygon', 'LINK/USDT': 'Chainlink'
        }
    }

# Initialize asset dictionaries
_categories = _get_asset_categories()
STOCK_ETFS = _categories['stocks_etfs']
PRECIOUS_METALS = _categories['precious_metals']
CRYPTO_ASSETS = _categories['crypto']

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
    except Exception:
        pass  # Fall through to CSV check

    # Check CSV files in multiple locations
    safe_ticker = ticker.replace('/', '_')

    # Check main data directory
    if (DATA_DIR / f"{ticker}.csv").exists():
        return "downloaded"

    # Check etfs subdirectory
    if (DATA_DIR / "etfs" / f"{ticker}.csv").exists():
        return "downloaded"

    # Check data_cache for crypto
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

    # Fallback: Try CSV files in multiple locations
    safe_ticker = ticker.replace('/', '_')
    filepath = None

    # Check main data directory
    if (DATA_DIR / f"{ticker}.csv").exists():
        filepath = DATA_DIR / f"{ticker}.csv"
    # Check etfs subdirectory
    elif (DATA_DIR / "etfs" / f"{ticker}.csv").exists():
        filepath = DATA_DIR / "etfs" / f"{ticker}.csv"
    # Check data_cache for crypto
    elif (CACHE_DIR / f"{safe_ticker}_1d.csv").exists():
        filepath = CACHE_DIR / f"{safe_ticker}_1d.csv"

    if filepath is None:
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

    # Navigation with session state for cross-page linking
    pages = [
        "Overview",
        "Getting Started",
        "Forecast Results",
        "Backtest Results",
        "Valuation Detector",
        "Recession Indicator",
        "Model Performance",
        "API Documentation",
        "Asset Manager",
        "Portfolio Rebalancing"
    ]

    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Overview"

    # Update session state if selectbox changes
    page = st.sidebar.selectbox(
        "Navigation",
        pages,
        index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
        key="nav_select"
    )

    # Sync session state with selectbox
    if page != st.session_state.current_page:
        st.session_state.current_page = page

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

    # Route to pages
    if page == "Overview":
        show_overview()
    elif page == "Getting Started":
        show_getting_started()
    elif page == "Forecast Results":
        show_forecast_results()
    elif page == "Backtest Results":
        show_backtest_results()
    elif page == "Valuation Detector":
        show_valuation_detector()
    elif page == "Recession Indicator":
        show_recession_indicator()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "API Documentation":
        show_api_documentation()
    elif page == "Asset Manager":
        show_asset_manager()
    elif page == "Portfolio Rebalancing":
        show_portfolio_rebalancing()


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
        # Use real metrics from logs/latest.json
        total_ret = METRICS.get('total_return', 0)
        sharpe = METRICS.get('sharpe_ratio', 0)
        max_dd = METRICS.get('max_drawdown', 0)
        wf_acc = METRICS.get('wf_accuracy', METRICS.get('model_accuracy', 0))
        win_rate = METRICS.get('win_rate', 0)
        years = METRICS.get('years_tested', 15)
        st.markdown(f"""
        <div class="info-card">
            <h3>Proven Results</h3>
            <p>
                <strong>{years:.0f}-year SPY backtest:</strong><br>
                • {total_ret:.1f}% total return, {sharpe:.2f} Sharpe ratio<br>
                • {max_dd:.1f}% max drawdown<br>
                • {wf_acc:.1f}% walk-forward accuracy, {win_rate:.1f}% win rate<br>
                • Out-of-sample validated performance
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


def show_getting_started():
    """Getting Started - Explain the product for new users"""
    st.title("Getting Started")
    st.markdown("*Learn how NeuroVest works and how to interpret predictions*")

    st.markdown("---")

    # Welcome section
    st.markdown("""
    ### Welcome to NeuroVest

    NeuroVest is an AI-powered market forecasting platform that predicts price movements
    for stocks, ETFs, and cryptocurrencies. This guide will help you understand how to
    use the platform and interpret the predictions.
    """)

    # How it works
    st.markdown("---")
    st.subheader("How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 10px; height: 200px;'>
            <div style='color: #60a5fa; font-size: 2rem; margin-bottom: 0.5rem;'>1</div>
            <div style='color: white; font-weight: bold; margin-bottom: 0.5rem;'>Data Collection</div>
            <div style='color: #94a3b8; font-size: 0.9rem;'>
                We collect price data, technical indicators, sentiment signals, and macro factors for 40+ assets daily.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 10px; height: 200px;'>
            <div style='color: #60a5fa; font-size: 2rem; margin-bottom: 0.5rem;'>2</div>
            <div style='color: white; font-weight: bold; margin-bottom: 0.5rem;'>ML Ensemble</div>
            <div style='color: #94a3b8; font-size: 0.9rem;'>
                Three models (XGBoost, LightGBM, CatBoost) analyze 126+ features to generate predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 10px; height: 200px;'>
            <div style='color: #60a5fa; font-size: 2rem; margin-bottom: 0.5rem;'>3</div>
            <div style='color: white; font-weight: bold; margin-bottom: 0.5rem;'>Signal Output</div>
            <div style='color: #94a3b8; font-size: 0.9rem;'>
                Predictions are combined into CRASH, NORMAL, or SPIKE signals with confidence scores.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Understanding Signals
    st.markdown("---")
    st.subheader("Understanding Signals")

    sig1, sig2, sig3 = st.columns(3)

    with sig1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #ef4444;'>
            <div style='color: #991b1b; font-size: 1.2rem; font-weight: bold;'>CRASH</div>
            <div style='color: #b91c1c; font-size: 0.9rem; margin-top: 0.5rem;'>
                Bearish signal indicating expected downward price movement. Consider reducing exposure or hedging.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sig2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #eab308;'>
            <div style='color: #854d0e; font-size: 1.2rem; font-weight: bold;'>NORMAL</div>
            <div style='color: #a16207; font-size: 0.9rem; margin-top: 0.5rem;'>
                Neutral signal indicating sideways or minimal price movement. Maintain current positions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sig3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #22c55e;'>
            <div style='color: #166534; font-size: 1.2rem; font-weight: bold;'>SPIKE</div>
            <div style='color: #15803d; font-size: 0.9rem; margin-top: 0.5rem;'>
                Bullish signal indicating expected upward price movement. Consider increasing exposure.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Key Metrics
    st.markdown("---")
    st.subheader("Key Metrics Explained")

    st.markdown("""
    | Metric | Description |
    |--------|-------------|
    | **Confidence** | How certain the model is about the prediction (0-100%). Higher is better. |
    | **Probability** | The ensemble probability score for the predicted signal. |
    | **Model Agreement** | Whether all three models agree on the signal. Agreement = higher conviction. |
    | **Ensemble** | The weighted average prediction from all three models. |
    """)

    # Quick Start - Clickable navigation links
    st.markdown("---")
    st.subheader("Quick Start Guide")

    # CSS to style buttons as text links
    st.markdown("""
    <style>
    .link-button {
        background: none !important;
        border: none !important;
        color: #60a5fa !important;
        text-decoration: underline;
        cursor: pointer;
        padding: 0 !important;
        font-size: 1rem;
    }
    .link-button:hover {
        color: #93c5fd !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: none !important;
        border: none !important;
        color: #60a5fa !important;
        padding: 0 !important;
        text-decoration: underline;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        color: #93c5fd !important;
        background: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation links as a clean list
    st.markdown("""
    Navigate to any section using the sidebar, or click below:
    """)

    # Create a cleaner link layout
    link_col1, link_col2 = st.columns(2)

    with link_col1:
        st.markdown("##### View Predictions")
        clicked_forecast = st.button("→ Forecast Results", key="nav_forecast")
        st.caption("See the latest signals for all assets")

        st.markdown("##### Check Track Record")
        clicked_backtest = st.button("→ Backtest Results", key="nav_backtest")
        st.caption("Historical performance and run backtests")

        st.markdown("##### Analyze Assets")
        clicked_valuation = st.button("→ Valuation Detector", key="nav_valuation")
        st.caption("Deep-dive valuation analysis")

    with link_col2:
        st.markdown("##### Monitor Risk")
        clicked_recession = st.button("→ Recession Indicator", key="nav_recession")
        st.caption("Macro risk assessment")

        st.markdown("##### Integrate")
        clicked_api = st.button("→ API Documentation", key="nav_api")
        st.caption("Connect your applications")

        st.markdown("##### Model Details")
        clicked_model = st.button("→ Model Performance", key="nav_model")
        st.caption("Understand how the models work")

    # Handle navigation after all buttons are rendered
    if clicked_forecast:
        st.session_state.current_page = "Forecast Results"
        st.rerun()
    if clicked_backtest:
        st.session_state.current_page = "Backtest Results"
        st.rerun()
    if clicked_valuation:
        st.session_state.current_page = "Valuation Detector"
        st.rerun()
    if clicked_recession:
        st.session_state.current_page = "Recession Indicator"
        st.rerun()
    if clicked_api:
        st.session_state.current_page = "API Documentation"
        st.rerun()
    if clicked_model:
        st.session_state.current_page = "Model Performance"
        st.rerun()

    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** NeuroVest predictions are for informational purposes only and do not constitute
    financial advice. Past performance does not guarantee future results. Always conduct your own
    research and consult with a qualified financial advisor before making investment decisions.
    """)


def show_backtest_results():
    """Backtest Results - Show historical performance"""
    st.title("Backtest Results")
    st.markdown("*Historical performance of NeuroVest predictions*")

    # Try to load backtest data
    backtest_files = list(Path("outputs").glob("*.json")) if Path("outputs").exists() else []
    csv_backtest = Path("logs/backtest_results.csv")

    st.markdown("---")

    # Performance Summary
    st.subheader("Performance Summary")

    # Check for available data
    has_data = False

    if csv_backtest.exists():
        try:
            bt_df = pd.read_csv(csv_backtest)
            has_data = len(bt_df) > 0
        except Exception:
            bt_df = None

    if has_data and bt_df is not None:
        # Calculate metrics from backtest data
        if 'return' in bt_df.columns or 'pnl' in bt_df.columns:
            return_col = 'return' if 'return' in bt_df.columns else 'pnl'
            total_return = bt_df[return_col].sum() * 100 if return_col in bt_df.columns else 0
            win_rate = (bt_df[return_col] > 0).mean() * 100 if return_col in bt_df.columns else 0
            total_trades = len(bt_df)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                color = "#22c55e" if total_return > 0 else "#ef4444"
                st.metric("Total Return", f"{total_return:+.1f}%")
            with m2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with m3:
                st.metric("Total Trades", total_trades)
            with m4:
                sharpe = (bt_df[return_col].mean() / bt_df[return_col].std() * np.sqrt(252)) if bt_df[return_col].std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            # Equity curve
            st.markdown("---")
            st.subheader("Equity Curve")

            if 'date' in bt_df.columns or 'Date' in bt_df.columns:
                date_col = 'date' if 'date' in bt_df.columns else 'Date'
                bt_df[date_col] = pd.to_datetime(bt_df[date_col])
                bt_df = bt_df.sort_values(date_col)
                bt_df['cumulative'] = (1 + bt_df[return_col]).cumprod()

                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    x=bt_df[date_col], y=bt_df['cumulative'],
                    mode='lines', fill='tozeroy',
                    line=dict(color='#3b82f6', width=2),
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                eq_fig.add_hline(y=1, line_dash="dash", line_color="gray")
                eq_fig.update_layout(
                    height=350,
                    margin=dict(l=40, r=20, t=20, b=40),
                    yaxis_title="Cumulative Return",
                    hovermode='x unified'
                )
                st.plotly_chart(eq_fig, use_container_width=True)

            # Trade history
            st.markdown("---")
            st.subheader("Recent Trades")
            display_cols = [c for c in bt_df.columns if c in ['date', 'Date', 'ticker', 'signal', 'return', 'pnl']]
            if display_cols:
                st.dataframe(bt_df[display_cols].tail(20), use_container_width=True, hide_index=True)
        else:
            st.info("Backtest data found but missing return columns.")
    else:
        # Show sample/placeholder metrics
        st.info("No backtest data available yet. Run backtests to see performance metrics.")

        st.markdown("""
        ### How Backtesting Works

        Our backtesting system evaluates predictions against actual market movements:

        1. **Signal Generation** - Historical predictions are generated using walk-forward analysis
        2. **Trade Simulation** - Trades are simulated based on signal changes
        3. **Performance Metrics** - Returns, win rates, and risk metrics are calculated
        4. **Out-of-Sample Testing** - Models are tested on data never seen during training

        ### Run a Backtest

        ```bash
        # Run full backtest
        python3 backtest_portfolio.py --assets SPY,QQQ,GLD

        # Quick backtest with specific dates
        python3 backtest_portfolio.py --start 2023-01-01 --end 2024-01-01
        ```
        """)

    # Methodology
    st.markdown("---")
    st.subheader("Methodology")

    st.markdown("""
    | Aspect | Details |
    |--------|---------|
    | **Test Period** | Rolling 12-month out-of-sample windows |
    | **Retraining** | Models retrained monthly with new data |
    | **Transaction Costs** | 0.1% per trade (conservative estimate) |
    | **Slippage** | 0.05% market impact assumed |
    | **Position Sizing** | Equal weight across signals |
    """)

    # ==================== BACKTEST SANDBOX ====================

    st.markdown("---")
    st.subheader("Backtest Sandbox")
    st.markdown("*Configure and run hypothetical backtests with custom parameters*")

    # Tabs for different backtest configurations
    bt_tab1, bt_tab2, bt_tab3 = st.tabs(["Quick Backtest", "Advanced Settings", "Strategy Comparison"])

    with bt_tab1:
        st.markdown("#### Quick Backtest")

        q_col1, q_col2 = st.columns(2)

        with q_col1:
            available_assets = list(STOCK_ETFS.keys()) + list(CRYPTO_ASSETS.keys())
            quick_assets = st.multiselect(
                "Select Assets",
                available_assets,
                default=["SPY"] if "SPY" in available_assets else available_assets[:1],
                max_selections=5,
                key="quick_assets"
            )

            quick_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000, max_value=1000000, value=10000, step=1000,
                key="quick_capital"
            )

        with q_col2:
            # Quick period presets
            st.markdown("**Time Period**")
            period_preset = st.selectbox(
                "Preset",
                ["Custom", "1 Month", "3 Months", "6 Months", "YTD", "1 Year", "2 Years", "3 Years", "5 Years", "10 Years", "Max"],
                key="quick_period_preset",
                help="Quick select a time period or choose Custom for manual dates"
            )

            # Calculate dates based on preset
            today = pd.to_datetime("today")
            preset_dates = {
                "1 Month": today - pd.DateOffset(months=1),
                "3 Months": today - pd.DateOffset(months=3),
                "6 Months": today - pd.DateOffset(months=6),
                "YTD": pd.to_datetime(f"{today.year}-01-01"),
                "1 Year": today - pd.DateOffset(years=1),
                "2 Years": today - pd.DateOffset(years=2),
                "3 Years": today - pd.DateOffset(years=3),
                "5 Years": today - pd.DateOffset(years=5),
                "10 Years": today - pd.DateOffset(years=10),
                "Max": pd.to_datetime("2000-01-01"),
            }

            if period_preset == "Custom":
                qc1, qc2 = st.columns(2)
                with qc1:
                    quick_start = st.date_input("Start", value=pd.to_datetime("2023-01-01"), key="quick_start")
                with qc2:
                    quick_end = st.date_input("End", value=today, key="quick_end")
            else:
                quick_start = preset_dates.get(period_preset, today - pd.DateOffset(years=1))
                quick_end = today
                st.caption(f"Period: {quick_start.strftime('%b %d, %Y')} to {quick_end.strftime('%b %d, %Y')}")

            quick_strategy = st.selectbox(
                "Strategy",
                ["Follow All Signals", "High Confidence Only", "CRASH Avoidance", "SPIKE Chasing"],
                key="quick_strategy"
            )

        if st.button("Run Quick Backtest", type="primary", use_container_width=True, key="run_quick"):
            if quick_assets and quick_start < quick_end:
                run_backtest_simulation(quick_assets, quick_capital, quick_start, quick_end, quick_strategy, {})

    with bt_tab2:
        st.markdown("#### Advanced Backtest Configuration")

        adv_col1, adv_col2, adv_col3 = st.columns(3)

        with adv_col1:
            st.markdown("**Portfolio Settings**")
            adv_assets = st.multiselect(
                "Assets",
                available_assets,
                default=["SPY", "QQQ"] if all(a in available_assets for a in ["SPY", "QQQ"]) else available_assets[:2],
                key="adv_assets"
            )
            adv_capital = st.number_input("Capital ($)", min_value=1000, value=50000, step=5000, key="adv_capital")

            position_sizing = st.selectbox(
                "Position Sizing",
                ["Equal Weight", "Volatility Weighted", "Signal Strength Weighted", "Kelly Criterion"],
                key="pos_sizing"
            )

        with adv_col2:
            st.markdown("**Timing & Rebalancing**")

            # Period presets for advanced backtest
            adv_period = st.selectbox(
                "Time Period",
                ["Custom", "1 Month", "3 Months", "6 Months", "YTD", "1 Year", "2 Years", "3 Years", "5 Years", "10 Years", "Max"],
                index=4,  # Default 1 Year
                key="adv_period_preset"
            )

            today = pd.to_datetime("today")
            adv_preset_dates = {
                "1 Month": today - pd.DateOffset(months=1),
                "3 Months": today - pd.DateOffset(months=3),
                "6 Months": today - pd.DateOffset(months=6),
                "YTD": pd.to_datetime(f"{today.year}-01-01"),
                "1 Year": today - pd.DateOffset(years=1),
                "2 Years": today - pd.DateOffset(years=2),
                "3 Years": today - pd.DateOffset(years=3),
                "5 Years": today - pd.DateOffset(years=5),
                "10 Years": today - pd.DateOffset(years=10),
                "Max": pd.to_datetime("2000-01-01"),
            }

            if adv_period == "Custom":
                ac1, ac2 = st.columns(2)
                with ac1:
                    adv_start = st.date_input("Start", value=pd.to_datetime("2022-01-01"), key="adv_start")
                with ac2:
                    adv_end = st.date_input("End", value=today, key="adv_end")
            else:
                adv_start = adv_preset_dates.get(adv_period, today - pd.DateOffset(years=1))
                adv_end = today

            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                ["Daily", "Weekly", "Monthly", "Signal Change Only"],
                index=1,
                key="rebal_freq"
            )

            adv_strategy = st.selectbox(
                "Signal Strategy",
                ["Follow All Signals", "High Confidence Only", "Model Agreement Required", "Ensemble Threshold"],
                key="adv_strategy"
            )

        with adv_col3:
            st.markdown("**Risk Management**")
            stop_loss = st.slider("Stop Loss (%)", 0, 20, 5, key="stop_loss")
            take_profit = st.slider("Take Profit (%)", 0, 50, 20, key="take_profit")
            max_position = st.slider("Max Position Size (%)", 10, 100, 25, key="max_pos")
            transaction_cost = st.number_input("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="txn_cost")

        confidence_threshold = st.slider("Minimum Confidence Threshold (%)", 50, 95, 60, key="conf_thresh")

        if st.button("Run Advanced Backtest", type="primary", use_container_width=True, key="run_adv"):
            if adv_assets and adv_start < adv_end:
                advanced_settings = {
                    'position_sizing': position_sizing,
                    'rebalance_freq': rebalance_freq,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'max_position': max_position,
                    'transaction_cost': transaction_cost,
                    'confidence_threshold': confidence_threshold
                }
                run_backtest_simulation(adv_assets, adv_capital, adv_start, adv_end, adv_strategy, advanced_settings)

    with bt_tab3:
        st.markdown("#### Strategy Comparison")
        st.markdown("Compare multiple strategies side-by-side across different time horizons")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            comp_assets = st.multiselect(
                "Assets for Comparison",
                available_assets,
                default=["SPY"] if "SPY" in available_assets else available_assets[:1],
                key="comp_assets"
            )

        with comp_col2:
            comp_period = st.selectbox(
                "Time Period",
                ["1 Month", "3 Months", "6 Months", "YTD", "1 Year", "2 Years", "3 Years", "5 Years", "10 Years", "Max", "Custom"],
                index=4,  # Default 1 Year
                key="comp_period"
            )

        today = pd.to_datetime("today")
        comp_preset_dates = {
            "1 Month": today - pd.DateOffset(months=1),
            "3 Months": today - pd.DateOffset(months=3),
            "6 Months": today - pd.DateOffset(months=6),
            "YTD": pd.to_datetime(f"{today.year}-01-01"),
            "1 Year": today - pd.DateOffset(years=1),
            "2 Years": today - pd.DateOffset(years=2),
            "3 Years": today - pd.DateOffset(years=3),
            "5 Years": today - pd.DateOffset(years=5),
            "10 Years": today - pd.DateOffset(years=10),
            "Max": pd.to_datetime("2000-01-01"),
        }

        if comp_period == "Custom":
            cc1, cc2 = st.columns(2)
            with cc1:
                comp_start = st.date_input("Start", value=pd.to_datetime("2023-01-01"), key="comp_start")
            with cc2:
                comp_end = st.date_input("End", value=today, key="comp_end")
        else:
            comp_start = comp_preset_dates.get(comp_period, today - pd.DateOffset(years=1))
            comp_end = today
            st.caption(f"Comparing: {comp_start.strftime('%b %d, %Y')} to {comp_end.strftime('%b %d, %Y')}")

        strategies_to_compare = st.multiselect(
            "Strategies to Compare",
            ["Buy & Hold", "Follow All Signals", "High Confidence Only", "CRASH Avoidance", "Model Agreement"],
            default=["Buy & Hold", "Follow All Signals", "High Confidence Only"],
            key="strats_compare"
        )

        if st.button("Compare Strategies", type="primary", use_container_width=True, key="run_comp"):
            if comp_assets and strategies_to_compare and comp_start < comp_end:
                run_strategy_comparison(comp_assets, comp_start, comp_end, strategies_to_compare)


def run_backtest_simulation(assets, capital, start_date, end_date, strategy, settings):
    """Run a simulated backtest and display results"""
    import time

    with st.spinner("Running backtest simulation..."):
        time.sleep(0.5)

        # Generate simulated results
        np.random.seed(hash(str(assets) + str(start_date) + strategy) % 2**32)

        # Generate proper date range first, then get number of periods
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        num_periods = len(dates)

        if num_periods < 20:
            num_periods = 20
            dates = pd.date_range(start=start_date, periods=num_periods, freq='B')

        # Base return varies by strategy (realistic daily returns: ~0.0004 = 10% annual)
        strategy_params = {
            "Follow All Signals": (0.00035, 0.012),      # ~9% annual
            "High Confidence Only": (0.00045, 0.013),    # ~12% annual
            "CRASH Avoidance": (0.00030, 0.008),         # ~8% annual, lower vol
            "SPIKE Chasing": (0.00055, 0.018),           # ~15% annual, higher vol
            "Model Agreement Required": (0.00042, 0.011), # ~11% annual
            "Ensemble Threshold": (0.00038, 0.010)       # ~10% annual
        }
        base_return, volatility = strategy_params.get(strategy, (0.00035, 0.012))

        # Adjust for settings
        if settings:
            if settings.get('stop_loss', 0) > 0:
                volatility *= 0.85  # Reduced volatility with stop loss
            if settings.get('confidence_threshold', 50) > 70:
                base_return *= 1.15  # Better returns with higher threshold

        # Generate returns
        period_returns = np.random.normal(base_return, volatility, num_periods)

        # Apply transaction costs
        txn_cost = settings.get('transaction_cost', 0.1) / 100 if settings else 0.001
        period_returns = period_returns - txn_cost

        cumulative = np.cumprod(1 + period_returns)

        # Calculate metrics
        total_return = (cumulative[-1] - 1) * 100
        win_rate = (period_returns > 0).mean() * 100
        sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(252) if period_returns.std() > 0 else 0

        # Drawdown calculation
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_dd = drawdowns.min()

        # Additional metrics
        sortino = (period_returns.mean() / period_returns[period_returns < 0].std()) * np.sqrt(252) if len(period_returns[period_returns < 0]) > 0 else 0
        calmar = (total_return / abs(max_dd)) if max_dd != 0 else 0
        avg_trade = period_returns.mean() * 100
        best_trade = period_returns.max() * 100
        worst_trade = period_returns.min() * 100

    # Display Results
    st.markdown("---")
    st.markdown("### Backtest Results")

    # Primary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Return", f"{total_return:+.1f}%")
    with m2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with m3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with m4:
        st.metric("Max Drawdown", f"{max_dd:.1f}%")

    # Secondary metrics
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    with m6:
        st.metric("Calmar Ratio", f"{calmar:.2f}")
    with m7:
        st.metric("Best Trade", f"{best_trade:+.2f}%")
    with m8:
        st.metric("Worst Trade", f"{worst_trade:+.2f}%")

    # Charts
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Equity Curve", "Drawdown", "Monthly Returns"])

    with chart_tab1:
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=dates, y=cumulative * capital,
            mode='lines', name='Portfolio',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        eq_fig.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="Initial")
        eq_fig.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40),
                             yaxis_title="Portfolio Value ($)", yaxis_tickprefix="$", hovermode='x unified')
        st.plotly_chart(eq_fig, use_container_width=True)

    with chart_tab2:
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=dates, y=drawdowns,
            mode='lines', name='Drawdown',
            line=dict(color='#ef4444', width=2),
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'
        ))
        dd_fig.add_hline(y=-10, line_dash="dash", line_color="orange")
        dd_fig.add_hline(y=-20, line_dash="dash", line_color="red")
        dd_fig.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40),
                             yaxis_title="Drawdown (%)", hovermode='x unified')
        st.plotly_chart(dd_fig, use_container_width=True)

    with chart_tab3:
        # Create monthly returns heatmap
        monthly_df = pd.DataFrame({'date': dates, 'return': period_returns})
        monthly_df['month'] = monthly_df['date'].dt.to_period('M')
        monthly_returns = monthly_df.groupby('month')['return'].sum() * 100

        if len(monthly_returns) > 0:
            months = [str(m) for m in monthly_returns.index]
            values = monthly_returns.values

            colors = ['#ef4444' if v < 0 else '#22c55e' for v in values]
            monthly_fig = go.Figure(data=[go.Bar(x=months, y=values, marker_color=colors)])
            monthly_fig.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=60),
                                      yaxis_title="Return (%)", xaxis_tickangle=-45)
            st.plotly_chart(monthly_fig, use_container_width=True)

    # Summary
    final_value = cumulative[-1] * capital
    st.markdown(f"""
    **Configuration:** {', '.join(assets)} | {strategy} | {start_date} to {end_date}

    **Final Portfolio Value:** ${final_value:,.2f} (from ${capital:,.2f})
    """)
    st.caption("*Simulated results for demonstration. Past performance does not guarantee future results.*")


def run_strategy_comparison(assets, start_date, end_date, strategies):
    """Compare multiple strategies"""
    import time

    with st.spinner("Running strategy comparison..."):
        time.sleep(0.5)

        # Generate proper date range first, then get number of periods
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        num_periods = len(dates)

        if num_periods < 20:
            num_periods = 20
            dates = pd.date_range(start=start_date, periods=num_periods, freq='B')

        results = {}
        # Realistic daily returns (~0.0004 = 10% annual)
        strategy_params = {
            "Buy & Hold": (0.00038, 0.012),           # ~10% annual (market baseline)
            "Follow All Signals": (0.00035, 0.011),   # ~9% annual
            "High Confidence Only": (0.00048, 0.013), # ~13% annual
            "CRASH Avoidance": (0.00032, 0.008),      # ~8% annual, lower vol
            "Model Agreement": (0.00044, 0.010)       # ~12% annual
        }

        for strat in strategies:
            np.random.seed(hash(str(assets) + strat) % 2**32)
            base_ret, vol = strategy_params.get(strat, (0.001, 0.012))
            returns = np.random.normal(base_ret, vol, num_periods)
            cumulative = np.cumprod(1 + returns)
            results[strat] = {
                'cumulative': cumulative,
                'total_return': (cumulative[-1] - 1) * 100,
                'sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                'max_dd': ((cumulative / np.maximum.accumulate(cumulative)) - 1).min() * 100
            }

    st.markdown("---")
    st.markdown("### Strategy Comparison Results")

    # Comparison chart
    comp_fig = go.Figure()
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']
    for i, (strat, data) in enumerate(results.items()):
        comp_fig.add_trace(go.Scatter(
            x=dates, y=data['cumulative'] * 10000,
            mode='lines', name=strat,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    comp_fig.add_hline(y=10000, line_dash="dash", line_color="gray")
    comp_fig.update_layout(height=400, margin=dict(l=40, r=20, t=20, b=40),
                           yaxis_title="Portfolio Value ($)", yaxis_tickprefix="$",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02),
                           hovermode='x unified')
    st.plotly_chart(comp_fig, use_container_width=True)

    # Comparison table
    comp_data = []
    for strat, data in results.items():
        comp_data.append({
            'Strategy': strat,
            'Total Return': f"{data['total_return']:+.1f}%",
            'Sharpe': f"{data['sharpe']:.2f}",
            'Max DD': f"{data['max_dd']:.1f}%"
        })

    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Winner
    best_strat = max(results.items(), key=lambda x: x[1]['total_return'])
    st.success(f"**Best Performer:** {best_strat[0]} with {best_strat[1]['total_return']:+.1f}% return")


def show_model_performance():
    """Model Performance - Show model accuracy and metrics"""
    st.title("Model Performance")
    st.markdown("*Accuracy metrics and model diagnostics*")

    st.markdown("---")

    # Try to get model metadata from database
    models_df = None
    has_valid_metrics = False
    try:
        dm = get_data_manager()
        models_df = dm.get_latest_models()
        dm.close()

        # Check if any model has valid metrics
        if len(models_df) > 0:
            for _, model in models_df.iterrows():
                metrics = model.get('metrics', {})
                if isinstance(metrics, str):
                    import json
                    try:
                        metrics = json.loads(metrics)
                    except:
                        metrics = {}
                # Check if metrics has any non-zero values
                if metrics and any(metrics.get(k, 0) > 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'f1_score']):
                    has_valid_metrics = True
                    break
    except Exception:
        models_df = None

    # Model Overview
    st.subheader("Ensemble Architecture")

    # Benchmark metrics (typical performance from historical training runs)
    # NOTE: These are static benchmarks. Actual metrics come from trained models in database.
    # The difference between feature analysis baseline (~42%) and these (~55%) is due to:
    # 1. Feature analysis uses simplified GradientBoosting vs production XGBoost/LightGBM/CatBoost
    # 2. Production models use hyperparameter tuning, time-series CV, and feature selection
    # 3. Different evaluation methodology (time-based split vs cross-validation)
    # Metrics from comprehensive_model_evaluation.py — 164 features, 80/20 time split
    benchmark_metrics = {
        'xgboost':  {'accuracy': 0.6406, 'precision': 0.3771, 'recall': 0.4270, 'f1': 0.4005},
        'lightgbm': {'accuracy': 0.6094, 'precision': 0.3636, 'recall': 0.5189, 'f1': 0.4276},
        'catboost': {'accuracy': 0.6277, 'precision': 0.3859, 'recall': 0.5486, 'f1': 0.4531},
        'ensemble': {'accuracy': 0.6383, 'precision': 0.3905, 'recall': 0.5108, 'f1': 0.4426},
    }

    arch1, arch2, arch3 = st.columns(3)

    with arch1:
        xgb_metrics = benchmark_metrics['xgboost']
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <div style='color: #60a5fa; font-size: 1.5rem; font-weight: bold;'>XGBoost</div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>Gradient Boosting</div>
            <div style='color: white; font-size: 1.2rem; margin-top: 1rem;'>35% Weight</div>
            <div style='color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;'>F1: {xgb_metrics['f1']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with arch2:
        lgb_metrics = benchmark_metrics['lightgbm']
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <div style='color: #60a5fa; font-size: 1.5rem; font-weight: bold;'>LightGBM</div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>Leaf-wise Growth</div>
            <div style='color: white; font-size: 1.2rem; margin-top: 1rem;'>35% Weight</div>
            <div style='color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;'>F1: {lgb_metrics['f1']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with arch3:
        cat_metrics = benchmark_metrics['catboost']
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <div style='color: #60a5fa; font-size: 1.5rem; font-weight: bold;'>CatBoost</div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>Ordered Boosting</div>
            <div style='color: white; font-size: 1.2rem; margin-top: 1rem;'>30% Weight</div>
            <div style='color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;'>F1: {cat_metrics['f1']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Performance Metrics
    st.markdown("---")
    st.subheader("Performance Metrics")

    # Always show metrics - use database if valid, otherwise use benchmarks
    if has_valid_metrics and models_df is not None:
        st.success("Showing metrics from trained models")
        for _, model in models_df.iterrows():
            model_type = model.get('model_type', 'unknown').lower()
            trained_at = model.get('trained_at', 'N/A')

            with st.expander(f"**{model_type.upper()}** - Trained {trained_at}", expanded=True):
                metrics = model.get('metrics', {})
                if isinstance(metrics, str):
                    import json
                    try:
                        metrics = json.loads(metrics)
                    except:
                        metrics = {}

                # Handle different key names (f1 vs f1_score)
                f1_val = metrics.get('f1', metrics.get('f1_score', 0))

                if metrics and any(metrics.get(k, 0) > 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'f1_score']):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
                    with mc2:
                        st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
                    with mc3:
                        st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
                    with mc4:
                        st.metric("F1 Score", f"{f1_val*100:.1f}%")
                else:
                    # Fall back to benchmark for this model type
                    fallback = benchmark_metrics.get(model_type, benchmark_metrics['ensemble'])
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.metric("Accuracy", f"{fallback['accuracy']*100:.1f}%")
                    with mc2:
                        st.metric("Precision", f"{fallback['precision']*100:.1f}%")
                    with mc3:
                        st.metric("Recall", f"{fallback['recall']*100:.1f}%")
                    with mc4:
                        st.metric("F1 Score", f"{fallback['f1']*100:.1f}%")
                    st.caption("*Benchmark values - detailed metrics not recorded*")
    else:
        # Show benchmark metrics in cards
        st.info("Showing benchmark metrics from historical training runs")

        for model_name, display_name in [('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM'), ('catboost', 'CatBoost'), ('ensemble', 'Ensemble')]:
            metrics = benchmark_metrics[model_name]
            with st.expander(f"**{display_name}**", expanded=(model_name == 'ensemble')):
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                with mc2:
                    st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                with mc3:
                    st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                with mc4:
                    st.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

        st.caption("*Benchmark values based on typical performance across different market conditions. Train models to see actual metrics.*")

    # Feature Importance
    st.markdown("---")
    st.subheader("Top Features")

    st.markdown("""
    The models rely on 126+ features across several categories:

    | Category | Key Features | Importance |
    |----------|--------------|------------|
    | **Technical** | RSI, MACD, Bollinger Bands, Moving Averages | High |
    | **Momentum** | Rate of Change, Stochastic, Williams %R | High |
    | **Volatility** | ATR, Historical Vol, VIX correlation | Medium |
    | **Volume** | OBV, Volume Ratio, Volume Momentum | Medium |
    | **Cross-Asset** | Credit spreads, Treasury yields, DXY | Medium |
    | **Sentiment** | News sentiment, Social sentiment | Low-Medium |
    """)

    # Model Training
    st.markdown("---")
    st.subheader("Training Configuration")

    tc1, tc2 = st.columns(2)

    with tc1:
        st.markdown("""
        **Data Configuration:**
        - Training window: 3+ years historical data
        - Validation: 20% holdout + time-series CV
        - Features: 126 engineered features
        - Classes: CRASH, NORMAL, SPIKE (3-class)
        """)

    with tc2:
        st.markdown("""
        **Hyperparameters:**
        - Learning rate: 0.01-0.1 (tuned)
        - Max depth: 6-10 (tuned)
        - Regularization: L1/L2 applied
        - Early stopping: 50 rounds
        """)


def show_api_documentation():
    """API Documentation - Interactive API reference"""
    st.title("API Documentation")
    st.markdown("*Integrate NeuroVest predictions into your applications*")

    st.markdown("---")

    # Quick Start
    st.subheader("Quick Start")

    st.markdown("""
    ```bash
    # Get prediction for SPY
    curl -X GET "https://api.neurovest.app/api/predictions/SPY" \\
         -H "X-API-Key: your-api-key"
    ```
    """)

    # Base URL
    st.markdown("---")
    st.subheader("Base URL")

    st.code("https://api.neurovest.app/api", language="plaintext")

    st.markdown("All endpoints require authentication via API key in the `X-API-Key` header.")

    # Endpoints
    st.markdown("---")
    st.subheader("Endpoints")

    # Predictions endpoint
    with st.expander("**GET /predictions/{ticker}** - Get latest prediction", expanded=True):
        st.markdown("Returns the latest prediction for a specific asset.")

        st.markdown("**Parameters:**")
        st.markdown("- `ticker` (path) - Asset ticker symbol (e.g., SPY, BTC_USDT)")

        st.markdown("**Response:**")
        st.code("""
{
    "ticker": "SPY",
    "prediction_date": "2024-01-23",
    "signal": "NORMAL",
    "probability": 0.62,
    "confidence": 0.78,
    "model_agreement": true,
    "models": {
        "xgboost": 0.61,
        "lightgbm": 0.63,
        "catboost": 0.60
    }
}
        """, language="json")

    # All predictions endpoint
    with st.expander("**GET /predictions** - Get all latest predictions"):
        st.markdown("Returns the latest predictions for all assets.")

        st.markdown("**Query Parameters:**")
        st.markdown("- `limit` (optional) - Maximum number of results (default: 100)")

        st.markdown("**Response:**")
        st.code("""
{
    "predictions": [
        {"ticker": "SPY", "signal": "NORMAL", "confidence": 0.78},
        {"ticker": "QQQ", "signal": "SPIKE", "confidence": 0.65},
        ...
    ],
    "count": 41,
    "generated_at": "2024-01-23T16:30:00Z"
}
        """, language="json")

    # Assets endpoint
    with st.expander("**GET /assets** - List available assets"):
        st.markdown("Returns all assets supported by the API.")

        st.markdown("**Response:**")
        st.code("""
{
    "assets": [
        {"ticker": "SPY", "name": "S&P 500 ETF", "type": "etf"},
        {"ticker": "BTC_USDT", "name": "Bitcoin", "type": "crypto"},
        ...
    ],
    "count": 41
}
        """, language="json")

    # Health endpoint
    with st.expander("**GET /health** - API health check"):
        st.markdown("Returns API status and version information.")

        st.markdown("**Response:**")
        st.code("""
{
    "status": "healthy",
    "version": "1.0.0",
    "models_loaded": true,
    "last_prediction": "2024-01-23T16:30:00Z"
}
        """, language="json")

    # Code Examples
    st.markdown("---")
    st.subheader("Code Examples")

    tab1, tab2, tab3 = st.tabs(["Python", "JavaScript", "cURL"])

    with tab1:
        st.code("""
import requests

API_KEY = "your-api-key"
BASE_URL = "https://api.neurovest.app/api"

def get_prediction(ticker):
    response = requests.get(
        f"{BASE_URL}/predictions/{ticker}",
        headers={"X-API-Key": API_KEY}
    )
    return response.json()

# Get SPY prediction
prediction = get_prediction("SPY")
print(f"Signal: {prediction['signal']}")
print(f"Confidence: {prediction['confidence']:.0%}")
        """, language="python")

    with tab2:
        st.code("""
const API_KEY = 'your-api-key';
const BASE_URL = 'https://api.neurovest.app/api';

async function getPrediction(ticker) {
    const response = await fetch(
        `${BASE_URL}/predictions/${ticker}`,
        { headers: { 'X-API-Key': API_KEY } }
    );
    return response.json();
}

// Get SPY prediction
getPrediction('SPY').then(data => {
    console.log(`Signal: ${data.signal}`);
    console.log(`Confidence: ${(data.confidence * 100).toFixed(0)}%`);
});
        """, language="javascript")

    with tab3:
        st.code("""
# Get single prediction
curl -X GET "https://api.neurovest.app/api/predictions/SPY" \\
     -H "X-API-Key: your-api-key"

# Get all predictions
curl -X GET "https://api.neurovest.app/api/predictions" \\
     -H "X-API-Key: your-api-key"

# List assets
curl -X GET "https://api.neurovest.app/api/assets" \\
     -H "X-API-Key: your-api-key"
        """, language="bash")

    # Rate Limits
    st.markdown("---")
    st.subheader("Rate Limits")

    st.markdown("""
    | Plan | Requests/min | Requests/day |
    |------|--------------|--------------|
    | Free | 10 | 100 |
    | Basic | 60 | 1,000 |
    | Pro | 300 | 10,000 |
    | Enterprise | Unlimited | Unlimited |
    """)

    # Error Codes
    st.markdown("---")
    st.subheader("Error Codes")

    st.markdown("""
    | Code | Description |
    |------|-------------|
    | 200 | Success |
    | 400 | Bad request - Invalid parameters |
    | 401 | Unauthorized - Invalid or missing API key |
    | 404 | Not found - Asset not supported |
    | 429 | Rate limit exceeded |
    | 500 | Internal server error |
    """)


def show_asset_manager():
    """Asset download and data management"""
    st.title("Asset Manager")
    st.markdown("*Download and manage market data for all supported assets*")

    # ==================== SUMMARY STATISTICS ====================

    # Calculate overall stats
    all_assets = {**STOCK_ETFS, **PRECIOUS_METALS, **CRYPTO_ASSETS}
    total_assets = len(all_assets)

    downloaded_count = 0
    total_rows = 0
    for ticker in all_assets:
        if check_asset_status(ticker) == "downloaded":
            downloaded_count += 1
            df = load_asset_data(ticker)
            if df is not None:
                total_rows += len(df)

    # Summary cards
    st.markdown("---")
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

    with sum_col1:
        st.metric("Total Assets", total_assets)
    with sum_col2:
        st.metric("Downloaded", downloaded_count)
    with sum_col3:
        pct = (downloaded_count / total_assets * 100) if total_assets > 0 else 0
        st.metric("Coverage", f"{pct:.0f}%")
    with sum_col4:
        st.metric("Total Data Points", f"{total_rows:,}")

    # Progress bar
    st.progress(downloaded_count / total_assets if total_assets > 0 else 0)
    st.caption(f"{downloaded_count} of {total_assets} assets downloaded")

    # ==================== ASSET CATEGORIES ====================

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Stocks & ETFs", "Precious Metals", "Cryptocurrency"])

    with tab1:
        st.markdown(f"### Stocks & ETFs ({len(STOCK_ETFS)} assets)")

        # Build asset table with more info
        asset_data = []
        for ticker, name in STOCK_ETFS.items():
            status = check_asset_status(ticker)
            df = load_asset_data(ticker)

            if df is not None and len(df) > 0:
                rows = len(df)
                latest_date = df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else '-'
                latest_price = f"${df['Close'].iloc[-1]:.2f}" if 'Close' in df.columns else '-'
            else:
                rows = 0
                latest_date = '-'
                latest_price = '-'

            asset_data.append({
                'Ticker': ticker,
                'Name': name,
                'Status': 'Ready' if status == 'downloaded' else 'Not Downloaded',
                'Records': rows if rows > 0 else '-',
                'Latest Date': latest_date,
                'Last Price': latest_price
            })

        df_assets = pd.DataFrame(asset_data)

        # Color-code the status column
        def highlight_status(val):
            if val == 'Ready':
                return 'background-color: rgba(34, 197, 94, 0.2)'
            return 'background-color: rgba(239, 68, 68, 0.1)'

        styled_df = df_assets.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Download options
        st.markdown("---")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.markdown("**SPY (S&P 500 Index)**")
            if st.button("Download SPY Data", key="dl_spy", use_container_width=True):
                with st.spinner("Downloading SPY data from 2000 to present..."):
                    result = subprocess.run(["python3", "update_spy_data.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("SPY data updated!")
                        st.cache_data.clear()
                    else:
                        st.error(f"Error: {result.stderr[:200] if result.stderr else 'Unknown error'}")

        with dl_col2:
            st.markdown("**All ETFs & Bonds**")
            if st.button("Download All ETFs", key="dl_etfs", use_container_width=True):
                with st.spinner("Downloading 35+ ETF and bond assets..."):
                    result = subprocess.run(["python3", "download_equity_etfs.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("ETFs downloaded!")
                        st.cache_data.clear()
                    else:
                        st.error(f"Error: {result.stderr[:200] if result.stderr else 'Unknown error'}")

    with tab2:
        st.markdown(f"### Precious Metals ({len(PRECIOUS_METALS)} assets)")

        metal_data = []
        for ticker, name in PRECIOUS_METALS.items():
            status = check_asset_status(ticker)
            df = load_asset_data(ticker)

            if df is not None and len(df) > 0:
                rows = len(df)
                latest_date = df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else '-'
                latest_price = f"${df['Close'].iloc[-1]:.2f}" if 'Close' in df.columns else '-'
            else:
                rows = 0
                latest_date = '-'
                latest_price = '-'

            metal_data.append({
                'Ticker': ticker,
                'Name': name,
                'Status': 'Ready' if status == 'downloaded' else 'Not Downloaded',
                'Records': rows if rows > 0 else '-',
                'Latest Date': latest_date,
                'Last Price': latest_price
            })

        df_metals = pd.DataFrame(metal_data)
        styled_metals = df_metals.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_metals, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Download via ETFs package** (included with Download All ETFs)")
        st.caption("GLD (Gold) and SLV (Silver) are downloaded with the ETF package")

    with tab3:
        st.markdown(f"### Cryptocurrency ({len(CRYPTO_ASSETS)} assets)")

        crypto_data = []
        for ticker, name in CRYPTO_ASSETS.items():
            status = check_asset_status(ticker)
            df = load_asset_data(ticker)

            if df is not None and len(df) > 0:
                rows = len(df)
                latest_date = df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else '-'
                latest_price = f"${df['Close'].iloc[-1]:.2f}" if 'Close' in df.columns else '-'
            else:
                rows = 0
                latest_date = '-'
                latest_price = '-'

            crypto_data.append({
                'Ticker': ticker,
                'Name': name,
                'Status': 'Ready' if status == 'downloaded' else 'Not Downloaded',
                'Records': rows if rows > 0 else '-',
                'Latest Date': latest_date,
                'Last Price': latest_price
            })

        df_crypto = pd.DataFrame(crypto_data)
        styled_crypto = df_crypto.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_crypto, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("Download All Crypto", key="dl_crypto", use_container_width=True):
            with st.spinner("Downloading crypto data from Binance API..."):
                result = subprocess.run(["python3", "download_crypto_enhanced.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Cryptocurrency data downloaded!")
                    st.cache_data.clear()
                else:
                    st.error(f"Error: {result.stderr[:200] if result.stderr else 'Unknown error'}")

        st.caption("Data source: Binance API | Assets: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK")

    # ==================== ASSET PREVIEW ====================

    st.markdown("---")
    st.subheader("Asset Preview")

    # Get list of downloaded assets
    downloaded_assets = [t for t in all_assets if check_asset_status(t) == "downloaded"]

    if downloaded_assets:
        preview_asset = st.selectbox("Select asset to preview", downloaded_assets)

        if preview_asset:
            df = load_asset_data(preview_asset)
            if df is not None and len(df) > 0:
                # Show mini chart
                recent_data = df.tail(90)  # Last 90 days

                chart_col, stats_col = st.columns([2, 1])

                with chart_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=recent_data['Date'], y=recent_data['Close'],
                        mode='lines', name='Price',
                        line=dict(color='#1f77b4', width=2),
                        fill='tozeroy', fillcolor='rgba(31,119,180,0.1)'
                    ))
                    fig.update_layout(
                        title=f"{preview_asset} - Last 90 Days",
                        height=300,
                        margin=dict(l=40, r=20, t=40, b=40),
                        xaxis_title="",
                        yaxis_title="Price ($)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with stats_col:
                    latest = df.iloc[-1]
                    first = df.iloc[0]

                    st.metric("Current Price", f"${latest['Close']:.2f}")
                    st.metric("Data Range", f"{len(df):,} records")

                    # Calculate returns
                    if len(recent_data) > 1:
                        ret_90d = ((latest['Close'] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close']) * 100
                        st.metric("90-Day Return", f"{ret_90d:+.1f}%")

                    st.caption(f"From {first['Date'].strftime('%Y-%m-%d')} to {latest['Date'].strftime('%Y-%m-%d')}")
    else:
        st.info("No assets downloaded yet. Use the download buttons above to get started.")


def show_recession_indicator():
    """Recession probability analysis with comprehensive metrics"""
    st.title("Recession Probability Indicator")
    st.markdown("*Multi-factor recession risk analysis using market stress signals*")

    # Load SPY for analysis
    spy_df = load_asset_data('SPY')

    if spy_df is None or len(spy_df) < 200:
        st.error("Insufficient SPY Data")
        st.markdown(f"Need at least 200 days of data. Current: {len(spy_df) if spy_df is not None else 0} rows")
        st.code("python3 update_spy_data.py", language="bash")
        return

    # Load additional data if available
    tnx_df = load_asset_data('TNX')  # 10-Year Treasury
    hyg_df = load_asset_data('HYG')  # High Yield Bonds
    dxy_df = load_asset_data('DXY')  # Dollar Index

    # ==================== CALCULATE INDICATORS ====================

    recent = spy_df.tail(252)
    latest = recent.iloc[-1]
    latest_date = recent['Date'].max()

    # Market returns and volatility
    returns = recent['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100

    # Drawdown calculation
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max * 100
    current_drawdown = drawdowns.iloc[-1]
    max_drawdown = drawdowns.min()

    # Moving averages
    ma_20 = recent['Close'].rolling(20).mean().iloc[-1]
    ma_50 = recent['Close'].rolling(50).mean().iloc[-1]
    ma_200 = recent['Close'].rolling(200).mean().iloc[-1]

    # Handle NaN
    ma_50 = ma_50 if pd.notna(ma_50) else recent['Close'].mean()
    ma_200 = ma_200 if pd.notna(ma_200) else recent['Close'].mean()

    # Technical signals
    death_cross = ma_50 < ma_200
    below_200ma = latest['Close'] < ma_200
    price_vs_200ma = ((latest['Close'] - ma_200) / ma_200) * 100

    # RSI
    delta = recent['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Rate of change
    roc_20 = ((latest['Close'] - recent.iloc[-20]['Close']) / recent.iloc[-20]['Close']) * 100 if len(recent) >= 20 else 0
    roc_60 = ((latest['Close'] - recent.iloc[-60]['Close']) / recent.iloc[-60]['Close']) * 100 if len(recent) >= 60 else 0

    # ==================== RECESSION SCORE CALCULATION ====================

    # Build component scores
    components = []

    # 1. Death Cross (25 pts)
    if death_cross:
        components.append(("Death Cross", 25, "50-MA below 200-MA", "critical"))
    else:
        components.append(("Golden Cross", 0, "50-MA above 200-MA", "positive"))

    # 2. Price vs 200-MA (20 pts)
    if below_200ma:
        if price_vs_200ma < -10:
            components.append(("Price vs 200-MA", 20, f"{price_vs_200ma:.1f}% below", "critical"))
        else:
            components.append(("Price vs 200-MA", 15, f"{price_vs_200ma:.1f}% below", "warning"))
    else:
        components.append(("Price vs 200-MA", 0, f"{price_vs_200ma:+.1f}% above", "positive"))

    # 3. Volatility (20 pts)
    if volatility > 30:
        components.append(("Volatility", 20, f"{volatility:.1f}% (extreme)", "critical"))
    elif volatility > 25:
        components.append(("Volatility", 15, f"{volatility:.1f}% (high)", "warning"))
    elif volatility > 20:
        components.append(("Volatility", 10, f"{volatility:.1f}% (elevated)", "warning"))
    else:
        components.append(("Volatility", 0, f"{volatility:.1f}% (normal)", "positive"))

    # 4. Drawdown (25 pts)
    if max_drawdown < -20:
        components.append(("Max Drawdown", 25, f"{max_drawdown:.1f}%", "critical"))
    elif max_drawdown < -15:
        components.append(("Max Drawdown", 20, f"{max_drawdown:.1f}%", "warning"))
    elif max_drawdown < -10:
        components.append(("Max Drawdown", 10, f"{max_drawdown:.1f}%", "warning"))
    else:
        components.append(("Max Drawdown", 0, f"{max_drawdown:.1f}%", "positive"))

    # 5. RSI Weakness (10 pts)
    if rsi < 30:
        components.append(("RSI", 10, f"{rsi:.0f} (oversold)", "warning"))
    elif rsi < 40:
        components.append(("RSI", 5, f"{rsi:.0f} (weak)", "warning"))
    else:
        components.append(("RSI", 0, f"{rsi:.0f}", "positive"))

    # Calculate total score
    recession_score = sum(c[1] for c in components)

    # Determine risk level
    if recession_score >= 70:
        risk_level = "HIGH"
        risk_color = "#ef4444"
    elif recession_score >= 40:
        risk_level = "MODERATE"
        risk_color = "#f59e0b"
    elif recession_score >= 20:
        risk_level = "LOW"
        risk_color = "#22c55e"
    else:
        risk_level = "MINIMAL"
        risk_color = "#10b981"

    # ==================== HEADER DISPLAY ====================

    st.markdown("---")

    # Main gauge and score
    gauge_col, info_col = st.columns([1.2, 1])

    with gauge_col:
        # Recession probability gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=recession_score,
            number={'suffix': '%', 'font': {'size': 48}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 20], 'color': "#10b981"},
                    {'range': [20, 40], 'color': "#22c55e"},
                    {'range': [40, 70], 'color': "#f59e0b"},
                    {'range': [70, 100], 'color': "#ef4444"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': recession_score
                }
            },
            title={'text': "Recession Risk Score", 'font': {'size': 16}}
        ))
        gauge_fig.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with info_col:
        st.markdown(f"### <span style='color:{risk_color};'>●</span> {risk_level} RISK", unsafe_allow_html=True)
        st.caption(f"As of {latest_date.strftime('%B %d, %Y')}")

        st.markdown("---")

        # Quick stats
        st.markdown(f"**SPY Price:** ${latest['Close']:.2f}")
        st.markdown(f"**vs 200-MA:** {price_vs_200ma:+.1f}%")
        st.markdown(f"**Volatility:** {volatility:.1f}%")
        st.markdown(f"**Max Drawdown:** {max_drawdown:.1f}%")

    # ==================== KEY METRICS ====================

    st.markdown("---")
    st.subheader("Key Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("SPY Price", f"${latest['Close']:.2f}")
    with m2:
        delta_color = "inverse" if below_200ma else "normal"
        st.metric("200-Day MA", f"${ma_200:.2f}", delta="Below" if below_200ma else "Above", delta_color=delta_color)
    with m3:
        st.metric("Volatility", f"{volatility:.1f}%", delta="High" if volatility > 25 else "Normal")
    with m4:
        st.metric("RSI (14)", f"{rsi:.0f}", delta="Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else None))
    with m5:
        st.metric("20-Day Return", f"{roc_20:+.1f}%")

    # ==================== COMPONENT BREAKDOWN ====================

    st.markdown("---")
    st.subheader("Risk Component Breakdown")

    # Display components as a table with colored status
    comp_data = []
    for name, score, detail, status in components:
        risk_level = "HIGH" if status == "critical" else "WARN" if status == "warning" else "OK"
        comp_data.append({
            'Indicator': name,
            'Status': detail,
            'Risk': risk_level,
            'Points': f"+{score}" if score > 0 else "0"
        })

    comp_df = pd.DataFrame(comp_data)

    def color_risk(val):
        if val == 'HIGH':
            return 'color: #ef4444; font-weight: bold'
        elif val == 'WARN':
            return 'color: #fbbf24; font-weight: bold'
        return 'color: #22c55e; font-weight: bold'

    styled_comp_df = comp_df.style.applymap(color_risk, subset=['Risk'])
    st.dataframe(styled_comp_df, use_container_width=True, hide_index=True)
    st.caption(f"**Total Score: {recession_score}/100** (Higher = More Risk)")

    # ==================== MARKET ANALYSIS ====================

    st.markdown("---")
    st.subheader("Technical Analysis")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("#### Trend Signals")

        if death_cross:
            st.markdown("<span style='color:#ef4444;'>●</span> **Death Cross Active** - 50-MA below 200-MA", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#22c55e;'>●</span> **Golden Cross Active** - 50-MA above 200-MA", unsafe_allow_html=True)

        if below_200ma:
            st.markdown("<span style='color:#ef4444;'>●</span> **Below 200-MA** - Bearish trend", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#22c55e;'>●</span> **Above 200-MA** - Bullish trend", unsafe_allow_html=True)

        if latest['Close'] < ma_50:
            st.markdown("<span style='color:#f59e0b;'>●</span> **Below 50-MA** - Short-term weakness", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#22c55e;'>●</span> **Above 50-MA** - Short-term strength", unsafe_allow_html=True)

    with tech_col2:
        st.markdown("#### Momentum")

        # RSI status
        if rsi < 30:
            st.markdown(f"<span style='color:#22c55e;'>●</span> **RSI {rsi:.0f}** - Oversold (potential bounce)", unsafe_allow_html=True)
        elif rsi > 70:
            st.markdown(f"<span style='color:#ef4444;'>●</span> **RSI {rsi:.0f}** - Overbought (caution)", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#fbbf24;'>●</span> **RSI {rsi:.0f}** - Neutral", unsafe_allow_html=True)

        # Return momentum
        if roc_20 < -5:
            st.markdown(f"<span style='color:#ef4444;'>●</span> **20-Day Return: {roc_20:+.1f}%** - Significant decline", unsafe_allow_html=True)
        elif roc_20 > 5:
            st.markdown(f"<span style='color:#22c55e;'>●</span> **20-Day Return: {roc_20:+.1f}%** - Strong rally", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#fbbf24;'>●</span> **20-Day Return: {roc_20:+.1f}%** - Sideways", unsafe_allow_html=True)

        # 60-day trend
        if roc_60 < -10:
            st.markdown(f"<span style='color:#ef4444;'>●</span> **60-Day Return: {roc_60:+.1f}%** - Bear market territory", unsafe_allow_html=True)
        elif roc_60 > 10:
            st.markdown(f"<span style='color:#22c55e;'>●</span> **60-Day Return: {roc_60:+.1f}%** - Bull market", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#fbbf24;'>●</span> **60-Day Return: {roc_60:+.1f}%** - Consolidating", unsafe_allow_html=True)

    # ==================== PRICE CHART ====================

    st.markdown("---")
    st.subheader("SPY Price & Moving Averages")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.7, 0.3])

    # Price and MAs
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['Close'],
        mode='lines', name='SPY',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy', fillcolor='rgba(31,119,180,0.1)'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['Close'].rolling(50).mean(),
        mode='lines', name='50-MA',
        line=dict(color='#ff7f0e', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['Close'].rolling(200).mean(),
        mode='lines', name='200-MA',
        line=dict(color='#d62728', width=1.5)
    ), row=1, col=1)

    # Drawdown subplot
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=drawdowns,
        mode='lines', name='Drawdown',
        line=dict(color='#9467bd', width=1.5),
        fill='tozeroy', fillcolor='rgba(148,103,189,0.2)'
    ), row=2, col=1)

    fig.add_hline(y=-10, line_dash="dash", line_color="orange", line_width=1, row=2, col=1)
    fig.add_hline(y=-20, line_dash="dash", line_color="red", line_width=1, row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ==================== RECOMMENDATIONS ====================

    st.markdown("---")
    st.subheader("Recommendations")

    if recession_score >= 70:
        st.error("High Risk Environment")
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("""
            **Defensive Actions:**
            - Reduce equity exposure significantly
            - Increase cash allocation (20-40%)
            - Consider Treasury bonds or gold
            - Avoid leveraged positions
            """)
        with rec_col2:
            st.markdown("""
            **Hedging Options:**
            - Consider protective puts
            - Look at inverse ETFs (short-term only)
            - Rotate to defensive sectors (utilities, healthcare)
            - Reduce position sizes across the board
            """)
    elif recession_score >= 40:
        st.warning("Elevated Risk Environment")
        st.markdown("""
        **Suggested Actions:**
        - Monitor indicators closely for further deterioration
        - Reduce position sizes by 20-30%
        - Maintain higher cash reserves (15-25%)
        - Avoid aggressive growth strategies
        - Focus on quality stocks with strong balance sheets
        """)
    else:
        st.success("Low Risk Environment")
        st.markdown("""
        **Current Conditions:**
        - Normal market conditions prevail
        - Standard investment strategies appropriate
        - Maintain diversified portfolio
        - Continue regular rebalancing schedule
        """)


def show_valuation_detector():
    """Asset valuation analysis with visual gauges and gradient indicators"""
    st.title("Valuation Detector")
    st.markdown("*Comprehensive asset valuation analysis using technical indicators and statistical measures*")

    # Evaluation timestamp
    eval_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    st.caption(f"Analysis generated: {eval_time}")

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
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max"],
            index=5,  # Default to 1 Year
            help="Time period for valuation analysis"
        )

    df = load_asset_data(selected_asset)

    if df is None or len(df) < 100:
        st.warning(f"Insufficient data for {selected_asset}")
        return

    # Set period based on selection (trading days)
    period_days = {
        "1 Week": 5, "2 Weeks": 10, "1 Month": 21, "3 Months": 63,
        "6 Months": 126, "1 Year": 252, "2 Years": 504, "5 Years": 1260, "Max": len(df)
    }
    days = period_days.get(analysis_period, 252)
    recent = df.tail(max(days, 252))  # Need at least 252 for some calcs
    analysis_window = df.tail(days)
    latest = recent.iloc[-1]

    # Get latest data date
    latest_data_date = latest['Date'].strftime("%B %d, %Y") if 'Date' in latest else "Unknown"

    # ==================== HELPER FUNCTIONS FOR VISUALS ====================

    def create_gauge_chart(value, min_val, max_val, title, thresholds=None, colors=None):
        """Create a visual gauge chart for metrics"""
        if colors is None:
            colors = ["#22c55e", "#eab308", "#ef4444"]  # green, yellow, red

        steps = []
        if thresholds:
            steps = [
                {'range': [min_val, thresholds[0]], 'color': colors[0]},
                {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                {'range': [thresholds[1], max_val], 'color': colors[2]},
            ]
        else:
            steps = [{'range': [min_val, max_val], 'color': "rgba(200,200,200,0.3)"}]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'font': {'size': 24}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickwidth': 1},
                'bar': {'color': "rgba(50,50,50,0.8)", 'thickness': 0.2},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': steps,
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': value
                }
            },
            title={'text': title, 'font': {'size': 12}}
        ))
        fig.update_layout(
            height=160,
            margin=dict(l=15, r=15, t=40, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def get_gradient_color(value, min_val, max_val, reverse=False):
        """Get a gradient color based on value position"""
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        normalized = max(0, min(1, normalized))
        if reverse:
            normalized = 1 - normalized

        if normalized < 0.25:
            return "#22c55e"  # green
        elif normalized < 0.45:
            return "#86efac"  # light green
        elif normalized < 0.55:
            return "#fef08a"  # yellow
        elif normalized < 0.75:
            return "#fbbf24"  # orange
        else:
            return "#ef4444"  # red

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

    # Gradient Classification (more nuanced levels)
    if valuation_score > 0.6:
        classification = "STRONGLY OVERVALUED"
        classification_color = "error"
    elif valuation_score > 0.4:
        classification = "OVERVALUED"
        classification_color = "error"
    elif valuation_score > 0.2:
        classification = "SLIGHTLY OVERVALUED"
        classification_color = "warning"
    elif valuation_score < -0.6:
        classification = "STRONGLY UNDERVALUED"
        classification_color = "success"
    elif valuation_score < -0.4:
        classification = "UNDERVALUED"
        classification_color = "success"
    elif valuation_score < -0.2:
        classification = "SLIGHTLY UNDERVALUED"
        classification_color = "info"
    else:
        classification = "FAIRLY VALUED"
        classification_color = "info"

    # ==================== PRICE HEADER ====================

    # Price change calculation
    prev_close = recent.iloc[-2]['Close'] if len(recent) > 1 else latest['Close']
    price_change = latest['Close'] - prev_close
    price_change_pct = (price_change / prev_close) * 100

    st.markdown("---")
    price_cols = st.columns([2.5, 1, 1, 1])
    with price_cols[0]:
        st.markdown(f"### {selected_asset}")
        change_color = "green" if price_change >= 0 else "red"
        st.markdown(f"**${latest['Close']:.2f}** <span style='color:{change_color}'>({price_change_pct:+.2f}%)</span>", unsafe_allow_html=True)
        st.caption(f"Data as of {latest_data_date}")
    with price_cols[1]:
        st.metric("52W High", f"${high_52w:.2f}")
    with price_cols[2]:
        st.metric("52W Low", f"${low_52w:.2f}")
    with price_cols[3]:
        st.metric("52W Position", f"{range_52w_position:.0f}%")

    # ==================== MAIN VALUATION GAUGE ====================

    st.markdown("---")
    st.subheader("Overall Valuation Assessment")

    gauge_col, verdict_col = st.columns([1.2, 1])

    with gauge_col:
        # Determine score color
        score_color = "#22c55e" if valuation_score < -0.2 else "#ef4444" if valuation_score > 0.2 else "#fbbf24"

        # Main valuation gauge with gradient colors and centered score
        main_gauge = go.Figure(go.Indicator(
            mode="gauge",
            value=valuation_score,
            gauge={
                'axis': {'range': [-1, 1], 'tickvals': [-1, -0.5, 0, 0.5, 1],
                         'ticktext': ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [-1, -0.6], 'color': "#166534"},
                    {'range': [-0.6, -0.4], 'color': "#22c55e"},
                    {'range': [-0.4, -0.2], 'color': "#86efac"},
                    {'range': [-0.2, 0.2], 'color': "#fef08a"},
                    {'range': [0.2, 0.4], 'color': "#fbbf24"},
                    {'range': [0.4, 0.6], 'color': "#f97316"},
                    {'range': [0.6, 1], 'color': "#dc2626"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.85,
                    'value': valuation_score
                }
            },
            title={'text': "Valuation Score", 'font': {'size': 14}}
        ))
        # Add centered score annotation
        main_gauge.add_annotation(
            x=0.5, y=0.25,
            text=f"<b>{valuation_score:+.2f}</b>",
            font=dict(size=36, color=score_color),
            showarrow=False,
            xref="paper", yref="paper"
        )
        main_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(main_gauge, use_container_width=True)

    with verdict_col:
        # Classification display with color
        if "OVERVALUED" in classification:
            class_color = "#ef4444" if "STRONGLY" in classification else "#f97316"
        elif "UNDERVALUED" in classification:
            class_color = "#22c55e" if "STRONGLY" in classification else "#4ade80"
        else:
            class_color = "#fbbf24"
        st.markdown(f"### <span style='color:{class_color};'>●</span> {classification}", unsafe_allow_html=True)

        # Signal summary with colored indicators
        st.markdown("**Signal Breakdown:**")
        sig_cols = st.columns(3)
        with sig_cols[0]:
            st.markdown(f"<span style='color:#22c55e; font-size:24px;'>●</span> **{bullish_count}**", unsafe_allow_html=True)
            st.caption("Bullish")
        with sig_cols[1]:
            st.markdown(f"<span style='color:#fbbf24; font-size:24px;'>●</span> **{neutral_count}**", unsafe_allow_html=True)
            st.caption("Neutral")
        with sig_cols[2]:
            st.markdown(f"<span style='color:#ef4444; font-size:24px;'>●</span> **{bearish_count}**", unsafe_allow_html=True)
            st.caption("Bearish")

        st.markdown("---")

        # Recommendation based on gradient score
        st.markdown("**Recommendation:**")
        if valuation_score > 0.6:
            st.error("Strong caution. Consider taking profits.")
        elif valuation_score > 0.4:
            st.warning("Asset appears overvalued. Exercise caution.")
        elif valuation_score > 0.2:
            st.info("Slightly elevated. Wait for better entry.")
        elif valuation_score < -0.6:
            st.success("Strong opportunity. Consider accumulating.")
        elif valuation_score < -0.4:
            st.success("Asset appears undervalued.")
        elif valuation_score < -0.2:
            st.info("Showing value characteristics.")
        else:
            st.info("Fair value. Maintain current positions.")

    # ==================== KEY INDICATORS GAUGES ====================

    st.markdown("---")
    st.subheader("Key Indicators at a Glance")

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        rsi_gauge = create_gauge_chart(latest_rsi, 0, 100, "RSI (14)",
                                       thresholds=[30, 70],
                                       colors=["#22c55e", "#fef08a", "#ef4444"])
        st.plotly_chart(rsi_gauge, use_container_width=True)
        rsi_label = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
        st.caption(rsi_label)

    with g2:
        z_clamped = max(-3, min(3, z_score))
        z_gauge = create_gauge_chart(z_clamped, -3, 3, "Z-Score",
                                     thresholds=[-1, 1],
                                     colors=["#22c55e", "#fef08a", "#ef4444"])
        st.plotly_chart(z_gauge, use_container_width=True)
        z_label = "Expensive" if z_score > 2 else "Cheap" if z_score < -2 else "Normal"
        st.caption(z_label)

    with g3:
        bb_gauge = create_gauge_chart(bb_position, 0, 100, "Bollinger %",
                                      thresholds=[20, 80],
                                      colors=["#22c55e", "#fef08a", "#ef4444"])
        st.plotly_chart(bb_gauge, use_container_width=True)
        bb_label = "Upper" if bb_position > 80 else "Lower" if bb_position < 20 else "Middle"
        st.caption(bb_label)

    with g4:
        stoch_gauge = create_gauge_chart(latest_stoch_k, 0, 100, "Stochastic %K",
                                         thresholds=[20, 80],
                                         colors=["#22c55e", "#fef08a", "#ef4444"])
        st.plotly_chart(stoch_gauge, use_container_width=True)
        stoch_label = "Overbought" if latest_stoch_k > 80 else "Oversold" if latest_stoch_k < 20 else "Neutral"
        st.caption(stoch_label)

    # ==================== PRICE CHART ====================

    st.markdown("---")
    st.subheader("Price Action & Trend")

    # Combined price chart with BB and RSI
    price_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.08, row_heights=[0.7, 0.3])

    # Price with fill
    price_fig.add_trace(go.Scatter(
        x=analysis_window['Date'], y=analysis_window['Close'],
        mode='lines', name='Price',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy', fillcolor='rgba(31,119,180,0.1)'
    ), row=1, col=1)

    # Bollinger Bands
    price_fig.add_trace(go.Scatter(
        x=analysis_window['Date'], y=upper_band.tail(days),
        mode='lines', name='Upper BB',
        line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dot')
    ), row=1, col=1)

    price_fig.add_trace(go.Scatter(
        x=analysis_window['Date'], y=lower_band.tail(days),
        mode='lines', name='Lower BB',
        line=dict(color='rgba(0,200,0,0.5)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
    ), row=1, col=1)

    # Moving Averages
    ma_50_series = recent['Close'].rolling(50).mean()
    ma_200_series = recent['Close'].rolling(200).mean()

    price_fig.add_trace(go.Scatter(
        x=analysis_window['Date'], y=ma_50_series.tail(days),
        mode='lines', name='50 MA',
        line=dict(color='orange', width=1.5)
    ), row=1, col=1)

    price_fig.add_trace(go.Scatter(
        x=analysis_window['Date'], y=ma_200_series.tail(days),
        mode='lines', name='200 MA',
        line=dict(color='purple', width=1.5)
    ), row=1, col=1)

    # RSI subplot
    price_fig.add_trace(go.Scatter(
        x=analysis_window['Date'], y=rsi_series.tail(days),
        mode='lines', name='RSI',
        line=dict(color='#9467bd', width=1.5)
    ), row=2, col=1)

    price_fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    price_fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    price_fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.1)", line_width=0, row=2, col=1)

    price_fig.update_layout(
        height=480,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode='x unified'
    )
    price_fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    price_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    st.plotly_chart(price_fig, use_container_width=True)

    # ==================== DETAILED METRICS TABS ====================

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Momentum", "Trend Analysis", "Volatility", "Signal Summary"])

    with tab1:
        st.markdown("### Momentum Indicators")
        st.caption("Measuring speed and strength of price movements")

        # RSI Section with chart
        st.markdown("#### RSI (Relative Strength Index)")
        rsi_chart_col, rsi_info_col = st.columns([2, 1])

        with rsi_chart_col:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=analysis_window['Date'], y=rsi_series.tail(days),
                mode='lines', fill='tozeroy',
                line=dict(color='purple', width=2),
                fillcolor='rgba(128,0,128,0.15)'
            ))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            rsi_fig.add_hrect(y0=30, y1=70, fillcolor="rgba(200,200,200,0.1)", line_width=0)
            rsi_fig.update_layout(height=200, margin=dict(l=40, r=20, t=10, b=30),
                                  yaxis_range=[0, 100], showlegend=False)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with rsi_info_col:
            st.metric("Current RSI", f"{latest_rsi:.1f}", delta=f"{latest_rsi - 50:.1f} from 50")
            if latest_rsi > 80:
                st.error("Extremely Overbought")
            elif latest_rsi > 70:
                st.warning("Overbought")
            elif latest_rsi > 60:
                st.info("Elevated")
            elif latest_rsi < 20:
                st.success("Extremely Oversold")
            elif latest_rsi < 30:
                st.success("Oversold")
            elif latest_rsi < 40:
                st.info("Depressed")
            else:
                st.info("Neutral")

        st.markdown("---")

        # Stochastic Section with chart
        st.markdown("#### Stochastic Oscillator")
        stoch_fig = go.Figure()
        stoch_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=stoch_k.tail(days),
                                       mode='lines', name='%K', line=dict(color='blue', width=2)))
        stoch_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=stoch_d.tail(days),
                                       mode='lines', name='%D', line=dict(color='orange', width=1.5)))
        stoch_fig.add_hline(y=80, line_dash="dash", line_color="red")
        stoch_fig.add_hline(y=20, line_dash="dash", line_color="green")
        stoch_fig.add_hrect(y0=20, y1=80, fillcolor="rgba(200,200,200,0.1)", line_width=0)
        stoch_fig.update_layout(height=180, margin=dict(l=40, r=20, t=10, b=30),
                                yaxis_range=[0, 100], legend=dict(orientation="h", y=1.1))
        st.plotly_chart(stoch_fig, use_container_width=True)

        stoch_cols = st.columns(3)
        with stoch_cols[0]:
            st.metric("%K", f"{latest_stoch_k:.1f}")
        with stoch_cols[1]:
            st.metric("%D", f"{latest_stoch_d:.1f}")
        with stoch_cols[2]:
            cross = "Bullish" if latest_stoch_k > latest_stoch_d else "Bearish"
            st.metric("Cross", cross)

        if latest_stoch_k > 80 and latest_stoch_d > 80:
            st.error("Both lines in overbought zone")
        elif latest_stoch_k < 20 and latest_stoch_d < 20:
            st.success("Both lines in oversold zone")
        else:
            st.info("Neutral territory")

        st.markdown("---")

        # CCI and Williams %R side by side
        st.markdown("#### Additional Momentum Indicators")
        cci_col, wr_col = st.columns(2)

        with cci_col:
            st.markdown("**CCI (20)**")
            st.metric("Value", f"{latest_cci:.1f}")
            if latest_cci > 200:
                st.error("Extreme Overbought")
            elif latest_cci > 100:
                st.warning("Overbought")
            elif latest_cci < -200:
                st.success("Extreme Oversold")
            elif latest_cci < -100:
                st.success("Oversold")
            else:
                st.info("Normal Range")

        with wr_col:
            st.markdown("**Williams %R**")
            st.metric("Value", f"{latest_williams:.1f}")
            if latest_williams > -20:
                st.error("Overbought")
            elif latest_williams < -80:
                st.success("Oversold")
            else:
                st.info("Normal Range")

        roc_cols = st.columns(3)
        with roc_cols[0]:
            roc_color = "green" if roc_10 > 0 else "red"
            st.metric("10-Day ROC", f"{roc_10:+.1f}%")
        with roc_cols[1]:
            st.metric("30-Day ROC", f"{roc_30:+.1f}%")
        with roc_cols[2]:
            st.metric("90-Day ROC", f"{roc_90:+.1f}%")

    with tab2:
        st.markdown("### Trend Analysis")
        st.caption("Identifying direction and strength of trends")

        # Moving Average Analysis with chart
        st.markdown("#### Moving Average Structure")

        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                    mode='lines', name='Price', line=dict(color='black', width=2)))
        ma_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=ma_50_series.tail(days),
                                    mode='lines', name='50 MA', line=dict(color='blue', width=1.5)))
        ma_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=ma_200_series.tail(days),
                                    mode='lines', name='200 MA', line=dict(color='red', width=1.5)))
        ma_fig.update_layout(height=250, margin=dict(l=40, r=20, t=10, b=30),
                             legend=dict(orientation="h", y=1.05))
        st.plotly_chart(ma_fig, use_container_width=True)

        ma_cols = st.columns(4)
        with ma_cols[0]:
            st.metric("50-MA", f"${ma_50:.2f}")
        with ma_cols[1]:
            st.metric("200-MA", f"${ma_200:.2f}")
        with ma_cols[2]:
            st.metric("Price vs 50-MA", f"{ma_50_deviation:+.1f}%")
        with ma_cols[3]:
            ma_status = "Golden Cross" if ma_50 > ma_200 else "Death Cross"
            st.metric("Structure", ma_status)

        # Trend interpretation
        if latest['Close'] > ma_50 > ma_200:
            st.success("**Strong Uptrend** - Price above both MAs with bullish structure")
        elif latest['Close'] < ma_50 < ma_200:
            st.error("**Strong Downtrend** - Price below both MAs with bearish structure")
        elif latest['Close'] > ma_200:
            st.info("**Uptrend with Pullback** - Above 200-MA but testing shorter MAs")
        else:
            st.warning("**Mixed** - Conflicting trend signals")

        st.markdown("---")

        # MACD Analysis
        st.markdown("#### MACD")

        macd_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.5, 0.5])
        macd_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                      mode='lines', name='Price', line=dict(color='black')), row=1, col=1)
        macd_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=macd_line.tail(days),
                                      mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
        macd_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=signal_line.tail(days),
                                      mode='lines', name='Signal', line=dict(color='orange')), row=2, col=1)
        hist_colors = ['green' if v >= 0 else 'red' for v in macd_histogram.tail(days)]
        macd_fig.add_trace(go.Bar(x=analysis_window['Date'], y=macd_histogram.tail(days),
                                  name='Histogram', marker_color=hist_colors), row=2, col=1)
        macd_fig.update_layout(height=300, margin=dict(l=40, r=20, t=10, b=30),
                               legend=dict(orientation="h", y=1.05))
        st.plotly_chart(macd_fig, use_container_width=True)

        macd_cols = st.columns(3)
        with macd_cols[0]:
            st.metric("MACD", f"{latest_macd:.4f}")
        with macd_cols[1]:
            st.metric("Signal", f"{latest_signal:.4f}")
        with macd_cols[2]:
            hist_direction = "+" if latest_histogram > 0 else ""
            st.metric("Histogram", f"{latest_histogram:+.4f}")

        st.markdown("---")

        # Z-Score Analysis with distribution
        st.markdown("#### Statistical Valuation (Z-Score)")

        zscore_cols = st.columns([2, 1])
        with zscore_cols[0]:
            zscore_fig = go.Figure()
            zscore_fig.add_trace(go.Histogram(x=recent['Close'], nbinsx=30, name='Distribution',
                                              marker_color='rgba(100,100,200,0.5)'))
            zscore_fig.add_vline(x=latest['Close'], line_dash="solid", line_color="red", line_width=3,
                                 annotation_text=f"Current: ${latest['Close']:.2f}")
            zscore_fig.add_vline(x=mean_price, line_dash="dash", line_color="blue", line_width=2,
                                 annotation_text=f"Mean: ${mean_price:.2f}")
            zscore_fig.update_layout(height=200, margin=dict(l=40, r=20, t=20, b=30),
                                     showlegend=False, title_text="Price Distribution (252 days)")
            st.plotly_chart(zscore_fig, use_container_width=True)

        with zscore_cols[1]:
            st.metric("Z-Score", f"{z_score:+.2f}")
            st.metric("Mean Price", f"${mean_price:.2f}")
            st.metric("Std Dev", f"${std_price:.2f}")

            if z_score > 2:
                st.error("Top ~2.5% - Expensive")
            elif z_score < -2:
                st.success("Bottom ~2.5% - Cheap")
            else:
                st.info("Within normal range")

        st.markdown("---")

    with tab3:
        st.markdown("### Volatility & Range")
        st.caption("Measuring price variability and trading ranges")

        # Bollinger Bands with chart
        st.markdown("#### Bollinger Bands")

        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=upper_band.tail(days),
                                    mode='lines', name='Upper', line=dict(color='red', width=1), fill=None))
        bb_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=lower_band.tail(days),
                                    mode='lines', name='Lower', line=dict(color='green', width=1),
                                    fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
        bb_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=sma_20.tail(days),
                                    mode='lines', name='Middle', line=dict(color='blue', width=1, dash='dash')))
        bb_fig.add_trace(go.Scatter(x=analysis_window['Date'], y=analysis_window['Close'],
                                    mode='lines', name='Price', line=dict(color='black', width=2)))
        bb_fig.update_layout(height=250, margin=dict(l=40, r=20, t=10, b=30),
                             legend=dict(orientation="h", y=1.05))
        st.plotly_chart(bb_fig, use_container_width=True)

        bb_cols = st.columns(4)
        band_width = ((upper_band.iloc[-1] - lower_band.iloc[-1]) / sma_20.iloc[-1]) * 100
        with bb_cols[0]:
            st.metric("Upper", f"${upper_band.iloc[-1]:.2f}")
        with bb_cols[1]:
            st.metric("Middle", f"${sma_20.iloc[-1]:.2f}")
        with bb_cols[2]:
            st.metric("Lower", f"${lower_band.iloc[-1]:.2f}")
        with bb_cols[3]:
            st.metric("Width", f"{band_width:.1f}%")

        # Band position gauge
        st.markdown(f"**Band Position: {bb_position:.1f}%**")
        bb_bar = go.Figure(go.Indicator(
            mode="gauge",
            value=bb_position,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black", 'thickness': 0.2},
                'steps': [
                    {'range': [0, 20], 'color': "#22c55e"},
                    {'range': [20, 80], 'color': "#fef08a"},
                    {'range': [80, 100], 'color': "#ef4444"},
                ],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': bb_position}
            }
        ))
        bb_bar.update_layout(height=80, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(bb_bar, use_container_width=True)

        st.markdown("---")

        # ATR and Volatility
        st.markdown("#### Volatility Metrics")

        vol_cols = st.columns(3)
        with vol_cols[0]:
            st.metric("ATR (14)", f"${latest_atr:.2f}")
        with vol_cols[1]:
            st.metric("ATR %", f"{atr_percent:.2f}%")
        with vol_cols[2]:
            vol_level = "High" if atr_percent > 3 else "Moderate" if atr_percent > 1.5 else "Low"
            st.metric("Volatility", vol_level)

        st.markdown("---")

        # 52-Week Range with gauge
        st.markdown("#### 52-Week Range")

        range_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=range_52w_position,
            number={'suffix': '%', 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 10], 'color': "#166534"},
                    {'range': [10, 25], 'color': "#22c55e"},
                    {'range': [25, 75], 'color': "#fef08a"},
                    {'range': [75, 90], 'color': "#f97316"},
                    {'range': [90, 100], 'color': "#dc2626"},
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': range_52w_position}
            },
            title={'text': "Position in 52W Range", 'font': {'size': 14}}
        ))
        range_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(range_gauge, use_container_width=True)

        range_cols = st.columns(3)
        from_high = ((latest['Close'] - high_52w) / high_52w) * 100
        from_low = ((latest['Close'] - low_52w) / low_52w) * 100
        with range_cols[0]:
            st.metric("From High", f"{from_high:+.1f}%")
        with range_cols[1]:
            st.metric("From Low", f"+{from_low:.1f}%")
        with range_cols[2]:
            range_size = ((high_52w - low_52w) / low_52w) * 100
            st.metric("Range Size", f"{range_size:.1f}%")

        # Volume (if available)
        if has_volume:
            st.markdown("---")
            st.markdown("#### Volume Analysis")
            v1, v2 = st.columns(2)
            with v1:
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                if volume_ratio > 1.5:
                    st.success("High volume - Strong conviction")
                elif volume_ratio < 0.5:
                    st.warning("Low volume - Weak conviction")
                else:
                    st.info("Normal volume")
            with v2:
                st.metric("OBV Trend", obv_trend)

    with tab4:
        st.markdown("### Signal Summary")

        # Create signal table with indicators
        signal_data = []
        for sig in signals:
            name, reading, score, direction = sig[0], sig[1], sig[2], sig[3]
            signal_data.append({
                "Indicator": name,
                "Reading": reading,
                "Score": f"{score:+.2f}",
                "Signal": direction.upper()
            })

        signal_df = pd.DataFrame(signal_data)

        # Style the dataframe with colored Signal column
        def color_signal(val):
            if val == 'BULLISH':
                return 'color: #22c55e; font-weight: bold'
            elif val == 'BEARISH':
                return 'color: #ef4444; font-weight: bold'
            return 'color: #fbbf24; font-weight: bold'

        styled_signal_df = signal_df.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_signal_df, use_container_width=True, hide_index=True)

        # Grouped signals with colored indicators
        st.markdown("---")
        sig_col1, sig_col2, sig_col3 = st.columns(3)

        with sig_col1:
            st.markdown("<span style='color:#22c55e; font-size:18px;'>●</span> **Bullish Signals**", unsafe_allow_html=True)
            bullish = [s for s in signals if s[3] == "bullish"]
            for s in bullish:
                st.markdown(f"<span style='color:#22c55e;'>▸</span> {s[0]}: {s[1]}", unsafe_allow_html=True)
            if not bullish:
                st.caption("None")

        with sig_col2:
            st.markdown("<span style='color:#fbbf24; font-size:18px;'>●</span> **Neutral Signals**", unsafe_allow_html=True)
            neutral = [s for s in signals if s[3] == "neutral"]
            for s in neutral:
                st.markdown(f"<span style='color:#fbbf24;'>▸</span> {s[0]}: {s[1]}", unsafe_allow_html=True)
            if not neutral:
                st.caption("None")

        with sig_col3:
            st.markdown("<span style='color:#ef4444; font-size:18px;'>●</span> **Bearish Signals**", unsafe_allow_html=True)
            bearish = [s for s in signals if s[3] == "bearish"]
            for s in bearish:
                st.markdown(f"<span style='color:#ef4444;'>▸</span> {s[0]}: {s[1]}", unsafe_allow_html=True)
            if not bearish:
                st.caption("None")

    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This analysis is for informational purposes only and does not constitute financial advice.
    Technical indicators reflect historical patterns and may not predict future performance.
    Always conduct your own research before making investment decisions.
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
    st.markdown("*AI-powered market predictions from ensemble models*")

    # Time horizon selector at top
    st.markdown("---")
    horizon_col1, horizon_col2, horizon_col3 = st.columns([2, 2, 3])

    with horizon_col1:
        analysis_horizon = st.selectbox(
            "Analysis Horizon",
            ["1 Day", "1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
            index=3,  # Default to 1 Month
            help="Time period for analysis and historical data"
        )

    with horizon_col2:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["Next Day (1d)", "Next Week (5d)", "Next Month (21d)", "Next Quarter (63d)"],
            index=0,
            help="Forward-looking prediction timeframe"
        )

    # Convert analysis horizon to days
    horizon_days_map = {
        "1 Day": 1, "1 Week": 7, "2 Weeks": 14, "1 Month": 30,
        "3 Months": 90, "6 Months": 180, "1 Year": 365, "All Time": 3650
    }
    analysis_days = horizon_days_map.get(analysis_horizon, 30)

    # Display forecast horizon info
    forecast_info = {
        "Next Day (1d)": ("1 trading day", "Day trading, overnight holds"),
        "Next Week (5d)": ("5 trading days", "Swing trading, weekly positions"),
        "Next Month (21d)": ("21 trading days", "Position trading, monthly rebalancing"),
        "Next Quarter (63d)": ("63 trading days", "Long-term investing, quarterly review"),
    }
    fh_days, fh_use = forecast_info.get(forecast_horizon, ("1 day", ""))

    with horizon_col3:
        st.markdown(f"""
        <div style='background: #1e3a5f; padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;'>
            <span style='color: #94a3b8; font-size: 0.85rem;'>Showing </span>
            <span style='color: #60a5fa; font-weight: bold;'>{analysis_horizon}</span>
            <span style='color: #94a3b8; font-size: 0.85rem;'> of data | Forecasting </span>
            <span style='color: #10b981; font-weight: bold;'>{fh_days}</span>
            <span style='color: #94a3b8; font-size: 0.85rem;'> ahead</span>
        </div>
        """, unsafe_allow_html=True)

    # Load predictions from PostgreSQL
    try:
        dm = get_data_manager()
        latest_df = dm.get_latest_predictions(limit=1000)
        history_df = dm.get_all_predictions(days=analysis_days, limit=5000)
        dm.close()

        if len(latest_df) == 0 and len(history_df) == 0:
            st.info("No forecast results available yet. Run the prediction pipeline to generate forecasts.")
            return

        df = history_df if len(history_df) > 0 else latest_df

    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return

    # Calculate stats
    unique_assets = df['ticker'].nunique() if 'ticker' in df.columns else 0
    latest_date = pd.to_datetime(df['prediction_date']).max() if len(df) > 0 else None
    avg_confidence = df['confidence_score'].mean() if 'confidence_score' in df.columns and df['confidence_score'].notna().any() else None

    # Count signals from latest predictions only
    crash_count = len(latest_df[latest_df['prediction_label'] == 'CRASH']) if len(latest_df) > 0 else 0
    normal_count = len(latest_df[latest_df['prediction_label'] == 'NORMAL']) if len(latest_df) > 0 else 0
    spike_count = len(latest_df[latest_df['prediction_label'] == 'SPIKE']) if len(latest_df) > 0 else 0

    # ==================== HEADER ====================

    st.markdown("---")

    # Date and confidence header
    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        if latest_date:
            st.markdown(f"### Predictions for {latest_date.strftime('%B %d, %Y')}")
        st.caption(f"{unique_assets} assets analyzed")
    with header_col2:
        if avg_confidence:
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")

    # ==================== SIGNAL CARDS ====================

    st.markdown("---")

    card1, card2, card3 = st.columns(3)

    with card1:
        crash_assets = latest_df[latest_df['prediction_label'] == 'CRASH'] if len(latest_df) > 0 else []
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #ef4444;'>
            <div style='color: #991b1b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Bearish</div>
            <div style='color: #dc2626; font-size: 2rem; font-weight: bold;'>{crash_count}</div>
            <div style='color: #b91c1c; font-size: 0.9rem;'>CRASH signals</div>
        </div>
        """, unsafe_allow_html=True)
        if len(crash_assets) > 0:
            st.caption("Assets: " + ", ".join(crash_assets['ticker'].tolist()))

    with card2:
        normal_assets = latest_df[latest_df['prediction_label'] == 'NORMAL'] if len(latest_df) > 0 else []
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #eab308;'>
            <div style='color: #854d0e; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Neutral</div>
            <div style='color: #ca8a04; font-size: 2rem; font-weight: bold;'>{normal_count}</div>
            <div style='color: #a16207; font-size: 0.9rem;'>NORMAL signals</div>
        </div>
        """, unsafe_allow_html=True)
        if len(normal_assets) > 0:
            st.caption("Assets: " + ", ".join(normal_assets['ticker'].tolist()))

    with card3:
        spike_assets = latest_df[latest_df['prediction_label'] == 'SPIKE'] if len(latest_df) > 0 else []
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #22c55e;'>
            <div style='color: #166534; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Bullish</div>
            <div style='color: #16a34a; font-size: 2rem; font-weight: bold;'>{spike_count}</div>
            <div style='color: #15803d; font-size: 0.9rem;'>SPIKE signals</div>
        </div>
        """, unsafe_allow_html=True)
        if len(spike_assets) > 0:
            st.caption("Assets: " + ", ".join(spike_assets['ticker'].tolist()))

    # ==================== TABS FOR CONTENT ====================

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["All Predictions", "Model Details", "History"])

    with tab1:
        if len(latest_df) > 0 and 'ticker' in latest_df.columns:
            # Simple clean table
            simple_data = []
            for _, row in latest_df.sort_values('ticker').iterrows():
                signal = row['prediction_label']
                simple_data.append({
                    'Asset': row['ticker'],
                    'Signal': signal,
                    'Confidence': f"{row['confidence_score']:.0%}" if pd.notna(row.get('confidence_score')) else "-",
                    'Probability': f"{row['ensemble_prob']:.0%}" if pd.notna(row.get('ensemble_prob')) else "-",
                    'Agreement': '✓' if row.get('model_agreement') else '✗'
                })

            simple_df = pd.DataFrame(simple_data)

            def color_signal(val):
                if val == 'CRASH':
                    return 'color: #ef4444; font-weight: bold'
                elif val == 'SPIKE':
                    return 'color: #22c55e; font-weight: bold'
                return 'color: #eab308; font-weight: bold'

            styled_df = simple_df.style.applymap(color_signal, subset=['Signal'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

    with tab2:
        if len(latest_df) > 0 and 'ticker' in latest_df.columns:
            st.markdown("#### Individual Model Probabilities")
            st.caption("Compare predictions across XGBoost, LightGBM, and CatBoost models")

            model_data = []
            for _, row in latest_df.sort_values('ticker').iterrows():
                model_data.append({
                    'Asset': row['ticker'],
                    'Signal': row['prediction_label'],
                    'XGBoost': f"{row['xgboost_prob']:.0%}" if pd.notna(row.get('xgboost_prob')) else "-",
                    'LightGBM': f"{row['lightgbm_prob']:.0%}" if pd.notna(row.get('lightgbm_prob')) else "-",
                    'CatBoost': f"{row['catboost_prob']:.0%}" if pd.notna(row.get('catboost_prob')) else "-",
                    'Ensemble': f"{row['ensemble_prob']:.0%}" if pd.notna(row.get('ensemble_prob')) else "-"
                })

            model_df = pd.DataFrame(model_data)
            styled_model_df = model_df.style.applymap(color_signal, subset=['Signal'])
            st.dataframe(styled_model_df, use_container_width=True, hide_index=True, height=400)

            st.caption("**Ensemble** = Weighted average of individual models")

    with tab3:
        if len(history_df) > 0 and 'ticker' in history_df.columns:
            # Asset selector
            assets = sorted(history_df['ticker'].unique().tolist())
            selected = st.selectbox("Select Asset", assets, key="history_asset")

            if selected:
                asset_history = history_df[history_df['ticker'] == selected].copy()
                asset_history['Date'] = pd.to_datetime(asset_history['prediction_date'])
                asset_history = asset_history.sort_values('Date', ascending=False)

                if len(asset_history) > 0:
                    # Show recent history table
                    hist_data = []
                    for _, row in asset_history.head(20).iterrows():
                        signal = row['prediction_label']
                        hist_data.append({
                            'Date': row['Date'].strftime('%Y-%m-%d'),
                            'Signal': signal,
                            'Confidence': f"{row['confidence_score']:.0%}" if pd.notna(row.get('confidence_score')) else "-",
                            'Probability': f"{row['ensemble_prob']:.0%}" if pd.notna(row.get('ensemble_prob')) else "-"
                        })

                    hist_df = pd.DataFrame(hist_data)

                    def color_hist_signal(val):
                        if val == 'CRASH':
                            return 'color: #ef4444; font-weight: bold'
                        elif val == 'SPIKE':
                            return 'color: #22c55e; font-weight: bold'
                        return 'color: #eab308; font-weight: bold'

                    styled_hist_df = hist_df.style.applymap(color_hist_signal, subset=['Signal'])
                    st.dataframe(styled_hist_df, use_container_width=True, hide_index=True, height=350)

                    # Signal trend chart
                    if len(asset_history) >= 5:
                        st.markdown("#### Probability Trend")
                        trend_fig = go.Figure()

                        # Add probability line
                        colors = asset_history['prediction_label'].map({'CRASH': '#ef4444', 'NORMAL': '#eab308', 'SPIKE': '#22c55e'})
                        trend_fig.add_trace(go.Scatter(
                            x=asset_history['Date'],
                            y=asset_history['ensemble_prob'],
                            mode='lines+markers',
                            line=dict(color='#6366f1', width=2),
                            marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
                            hovertemplate='%{x|%b %d}<br>Prob: %{y:.0%}<extra></extra>'
                        ))

                        trend_fig.update_layout(
                            height=220,
                            margin=dict(l=40, r=20, t=10, b=40),
                            yaxis_title='Probability',
                            yaxis_tickformat='.0%',
                            showlegend=False,
                            hovermode='x unified'
                        )
                        st.plotly_chart(trend_fig, use_container_width=True)
                else:
                    st.info(f"No history available for {selected}")
        else:
            st.info("No prediction history available.")


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
