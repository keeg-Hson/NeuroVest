#!/usr/bin/env python3
"""
NeuroVest API Demo - Customer-Facing Showcase

Professional demonstration of the NeuroVest Forecasting API for prospective customers.
Complete feature showcase with API examples, integrations, and pricing.
"""

import json
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Install required packages: pip install streamlit")
    sys.exit(1)


def load_real_metrics() -> dict:
    """Load actual metrics from logs/latest.json"""
    metrics_path = Path("logs/latest.json")
    default_metrics = {
        "total_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "model_accuracy": 0,
        "wf_accuracy": 0,
        "win_rate": 0,
        "total_trades": 0,
        "years_tested": 0,
    }

    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load metrics: {e}")

    return default_metrics


# Load real metrics at startup
METRICS = load_real_metrics()

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

# Dark theme with READABLE text - locked in aesthetics
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
    }

    h1, h2, h3 { color: #ffffff; }
    p { color: #e0e0e0; }

    /* Metric boxes */
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
        height: 100%;
    }
    .info-card h3 {
        color: #3498db;
        margin-top: 0;
    }
    .info-card p {
        color: #e0e0e0;
        line-height: 1.7;
    }

    /* Feature boxes */
    .feature-box {
        background: #1e2530;
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        border-radius: 8px;
        height: 100%;
    }
    .feature-box h4 {
        color: #ffffff;
        margin-top: 0;
    }
    .feature-box p {
        color: #e0e0e0;
        line-height: 1.6;
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
        transform: translateY(-2px);
        transition: all 0.3s ease;
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

    /* Code example boxes */
    .code-box {
        background: #1e2530;
        border: 1px solid #3498db;
        border-radius: 8px;
        padding: 1.5rem;
    }
    .code-box h4 {
        color: #3498db;
        margin-top: 0;
    }

    /* Use case boxes */
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
    .use-case-box ul {
        color: #e0e0e0;
        line-height: 1.8;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Hero Section
logo_path = Path("assets/neurovest_logo.png")
if logo_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(logo_path), width=250)

st.markdown("# NeuroVest Forecasting API")
st.markdown("### AI-Powered Market Intelligence for Quantitative Developers")
years_tested = METRICS.get('years_tested', 15)
st.markdown(f"Real-time predictions for 59+ assets • Sub-500ms latency • {years_tested:.0f}-year track record")
st.markdown("---")

# Performance Metrics - REAL DATA from logs/latest.json
col1, col2, col3, col4 = st.columns(4)

total_return = METRICS.get('total_return', 0)
sharpe_ratio = METRICS.get('sharpe_ratio', 0)
max_drawdown = METRICS.get('max_drawdown', 0)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{total_return:.1f}%</h2>
        <p>Total Return<br>{years_tested:.0f}-year SPY backtest</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{sharpe_ratio:.2f}</h2>
        <p>Sharpe Ratio<br>Risk-adjusted returns</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{max_drawdown:.1f}%</h2>
        <p>Max Drawdown<br>Capital preservation</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box">
        <h2>59+</h2>
        <p>Assets Covered<br>Stocks, ETFs, Crypto</p>
    </div>
    """, unsafe_allow_html=True)

# Value Proposition
st.markdown("---")
st.markdown("## What is NeuroVest?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🔬 How It Works</h3>
        <p>
            Ensemble ML forecasting platform that predicts market movements across 59 assets.
            Combines XGBoost, LightGBM, and CatBoost models trained on 126+ features to generate
            three-class predictions (CRASH/NORMAL/SPIKE) with quantified confidence levels.
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
    model_acc = METRICS.get('model_accuracy', 0)
    wf_acc = METRICS.get('wf_accuracy', model_acc)
    win_rate = METRICS.get('win_rate', 0)
    total_trades = METRICS.get('total_trades', 0)
    st.markdown(f"""
    <div class="info-card">
        <h3>📊 Track Record</h3>
        <p>
            <strong>{years_tested:.0f}-year SPY backtest:</strong><br>
            • {wf_acc:.1f}% walk-forward accuracy<br>
            • {sharpe_ratio:.2f} Sharpe ratio<br>
            • {win_rate:.1f}% win rate ({total_trades:,} trades)<br>
            • {max_drawdown:.1f}% max drawdown<br>
            Performance validated with rolling out-of-sample testing.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Core Features
st.markdown("---")
st.markdown("## 🚀 Core API Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🎯 Real-Time Predictions</h4>
        <p>
            Single-asset forecasts with three-class output (CRASH/NORMAL/SPIKE) and probability distributions.
            Response times under 500ms with 99.9% uptime SLA.
        </p>
        <p style="margin-top: 1rem; color: #3498db;">
            <strong>Typical use:</strong> Algorithmic entry/exit signals, risk alerts, portfolio rebalancing triggers
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>📊 Batch Analysis</h4>
        <p>
            Multi-asset requests (up to 50 tickers) processed in parallel with structured JSON output.
            Perfect for portfolio-level decision making.
        </p>
        <p style="margin-top: 1rem; color: #3498db;">
            <strong>Typical use:</strong> Cross-sectional screening, sector rotation strategies, daily portfolio reports
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>📈 Custom Backtests</h4>
        <p>
            Test signal performance with configurable parameters: stop losses, profit targets,
            position sizing, transaction costs, and risk profiles.
        </p>
        <p style="margin-top: 1rem; color: #3498db;">
            <strong>Typical use:</strong> Strategy validation, parameter optimization, compliance reporting
        </p>
    </div>
    """, unsafe_allow_html=True)

# Asset Coverage
st.markdown("---")
st.markdown("## 📦 Asset Coverage (59+ Assets)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Stocks & ETFs (14)</h3>
        <p>
            <strong>Major Indices:</strong><br>
            SPY, QQQ, IWM, DIA, VTI<br><br>

            <strong>Sector ETFs:</strong><br>
            XLF (Financials), XLK (Tech), XLE (Energy)<br><br>

            <strong>International:</strong><br>
            EEM (Emerging Markets)<br><br>

            <strong>Bonds & Dollar:</strong><br>
            HYG, LQD, TNX, UUP, DXY
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🥇 Precious Metals (7)</h3>
        <p>
            <strong>Physical ETFs:</strong><br>
            GLD (Gold Trust)<br>
            SLV (Silver Trust)<br>
            IAU (iShares Gold)<br><br>

            <strong>Mining Stocks:</strong><br>
            GDX (Gold Miners)<br>
            GDXJ (Junior Miners)<br><br>

            <strong>Other Metals:</strong><br>
            PPLT (Platinum)<br>
            PALL (Palladium)
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>💎 Cryptocurrencies (10)</h3>
        <p>
            <strong>Large Cap:</strong><br>
            BTC/USDT, ETH/USDT<br>
            BNB/USDT, XRP/USDT<br><br>

            <strong>Alt Coins:</strong><br>
            SOL/USDT (Solana)<br>
            ADA/USDT (Cardano)<br>
            AVAX/USDT (Avalanche)<br>
            MATIC/USDT (Polygon)<br>
            LINK/USDT (Chainlink)<br>
            DOGE/USDT (Dogecoin)
        </p>
    </div>
    """, unsafe_allow_html=True)

# API Playground
st.markdown("---")
st.markdown("## 🔌 API Integration Examples")

tab1, tab2, tab3 = st.tabs(["Python", "JavaScript", "cURL"])

with tab1:
    st.markdown("### Python Integration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="code-box">
            <h4>Single Asset Prediction</h4>
        </div>
        """, unsafe_allow_html=True)

        st.code("""
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.neurovest.io"

# Get prediction for SPY
response = requests.get(
    f"{BASE_URL}/predict/SPY",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

data = response.json()
print(f"Asset: {data['asset']}")
print(f"Signal: {data['signal']}")
print(f"Probability: {data['probability']:.2%}")
print(f"Confidence: {data['confidence']}")

# Example Response:
# {
#   "asset": "SPY",
#   "prediction": 2,
#   "signal": "SPIKE",
#   "probability": 0.78,
#   "confidence": "high",
#   "models_agree": true,
#   "timestamp": "2024-12-24T10:30:00Z"
# }
        """, language="python")

    with col2:
        st.markdown("""
        <div class="code-box">
            <h4>Batch Predictions</h4>
        </div>
        """, unsafe_allow_html=True)

        st.code("""
import requests

# Analyze multiple assets at once
assets = ["SPY", "QQQ", "BTC/USDT", "GLD"]

response = requests.post(
    f"{BASE_URL}/batch-predict",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"assets": assets}
)

results = response.json()

for item in results['results']:
    print(f"{item['asset']}: {item['signal']} "
          f"({item['probability']:.1%})")

# Output:
# SPY: SPIKE (78.0%)
# QQQ: NORMAL (62.0%)
# BTC/USDT: CRASH (71.0%)
# GLD: SPIKE (68.0%)
        """, language="python")

with tab2:
    st.markdown("### JavaScript Integration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="code-box">
            <h4>Fetch API Example</h4>
        </div>
        """, unsafe_allow_html=True)

        st.code("""
// Single asset prediction
const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://api.neurovest.io';

async function getPrediction(asset) {
  const response = await fetch(
    `${BASE_URL}/predict/${asset}`,
    {
      headers: {
        'Authorization': `Bearer ${API_KEY}`
      }
    }
  );

  const data = await response.json();
  console.log(`${data.asset}: ${data.signal}`);
  console.log(`Probability: ${data.probability}`);

  return data;
}

// Usage
getPrediction('SPY').then(data => {
  // Update UI with prediction data
  updateDashboard(data);
});
        """, language="javascript")

    with col2:
        st.markdown("""
        <div class="code-box">
            <h4>Batch Request with Axios</h4>
        </div>
        """, unsafe_allow_html=True)

        st.code("""
const axios = require('axios');

async function batchPredict(assets) {
  const response = await axios.post(
    `${BASE_URL}/batch-predict`,
    { assets: assets },
    {
      headers: {
        'Authorization': `Bearer ${API_KEY}`
      }
    }
  );

  return response.data.results;
}

// Analyze portfolio
const portfolio = ['SPY', 'QQQ', 'GLD'];

batchPredict(portfolio).then(results => {
  results.forEach(item => {
    console.log(
      `${item.asset}: ${item.signal} ` +
      `(${(item.probability * 100).toFixed(1)}%)`
    );
  });
});
        """, language="javascript")

with tab3:
    st.markdown("### cURL Examples")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="code-box">
            <h4>Single Prediction</h4>
        </div>
        """, unsafe_allow_html=True)

        st.code("""
# Get prediction for Bitcoin
curl -X GET "https://api.neurovest.io/predict/BTC-USDT" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"

# Response:
{
  "asset": "BTC/USDT",
  "prediction": 0,
  "signal": "CRASH",
  "probability": 0.71,
  "confidence": "high",
  "models_agree": true
}
        """, language="bash")

    with col2:
        st.markdown("""
        <div class="code-box">
            <h4>Batch Analysis</h4>
        </div>
        """, unsafe_allow_html=True)

        st.code("""
# Analyze multiple assets
curl -X POST "https://api.neurovest.io/batch-predict" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "assets": ["SPY", "QQQ", "BTC/USDT"]
  }'

# List all available assets
curl "https://api.neurovest.io/assets" \\
  -H "Authorization: Bearer YOUR_API_KEY"

# Health check
curl "https://api.neurovest.io/health"
        """, language="bash")

# Use Cases
st.markdown("---")
st.markdown("## 🎯 Use Cases")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="use-case-box">
        <h4>💼 Institutional Research</h4>
        <ul>
            <li>Market regime classification</li>
            <li>Economic indicator analysis</li>
            <li>Risk assessment & stress testing</li>
            <li>Multi-asset correlation studies</li>
            <li>Client reporting & newsletters</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="use-case-box">
        <h4>📊 Portfolio Management</h4>
        <ul>
            <li>Asset allocation signals</li>
            <li>Rebalancing optimization</li>
            <li>Recession probability tracking</li>
            <li>Valuation-based positioning</li>
            <li>Risk-adjusted entries/exits</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="use-case-box">
        <h4>🤖 Algorithmic Trading</h4>
        <ul>
            <li>Automated signal generation</li>
            <li>Confidence-based position sizing</li>
            <li>Multi-timeframe confirmation</li>
            <li>Volatility regime detection</li>
            <li>Real-time risk management</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Pricing
st.markdown("---")
st.markdown("## 💳 API Pricing Plans")

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
            <li>✓ Community access</li>
        </ul>
        <p style="margin-top: 1.5rem; color: #3498db;"><strong>Perfect for:</strong> Testing & research</p>
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
            <li>✓ Historical data access</li>
        </ul>
        <p style="margin-top: 1.5rem; color: #3498db;"><strong>Perfect for:</strong> Individual traders</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pricing-box" style="border: 2px solid #3498db;">
        <h3>Professional ⭐</h3>
        <h2>$499/mo</h2>
        <ul>
            <li>✓ 500,000 requests/month</li>
            <li>✓ Dedicated support</li>
            <li>✓ WebSocket streaming</li>
            <li>✓ Custom models</li>
            <li>✓ SLA guarantee (99.9%)</li>
        </ul>
        <p style="margin-top: 1.5rem; color: #3498db;"><strong>Perfect for:</strong> Hedge funds & firms</p>
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
            <li>✓ Training & workshops</li>
        </ul>
        <p style="margin-top: 1.5rem; color: #3498db;"><strong>Perfect for:</strong> Institutions</p>
    </div>
    """, unsafe_allow_html=True)

# Technical Details
st.markdown("---")
st.markdown("## 🔧 Technical Specifications")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🚀 Performance & Reliability</h4>
        <p>
            <strong>Latency:</strong> < 500ms average response time<br>
            <strong>Uptime:</strong> 99.9% SLA on Professional+ plans<br>
            <strong>Rate Limits:</strong> Tier-based (1K-unlimited/month)<br>
            <strong>Data Freshness:</strong> Updated every trading day<br>
            <strong>Historical Access:</strong> 25+ years of backtest data
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>🔐 Security & Compliance</h4>
        <p>
            <strong>Authentication:</strong> OAuth 2.0 + API keys<br>
            <strong>Encryption:</strong> TLS 1.3 for all endpoints<br>
            <strong>Data Privacy:</strong> GDPR compliant<br>
            <strong>Audit Logs:</strong> Full request history<br>
            <strong>IP Whitelisting:</strong> Available on Enterprise
        </p>
    </div>
    """, unsafe_allow_html=True)

# Call to Action
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("## Ready to Get Started?")
    st.markdown("Join hedge funds, quant developers, and institutions using NeuroVest API")

    st.markdown("<br>", unsafe_allow_html=True)

    cta_col1, cta_col2 = st.columns(2)
    with cta_col1:
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://neurovestdemo.streamlit.app/" style="
                background: #3498db;
                color: white;
                padding: 1rem 2rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                display: inline-block;
            ">🚀 Start Free Trial</a>
        </div>
        """, unsafe_allow_html=True)

    with cta_col2:
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://neurovestdemo.streamlit.app/" style="
                background: #1e2530;
                color: #3498db;
                border: 2px solid #3498db;
                padding: 1rem 2rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                display: inline-block;
            ">📅 Schedule Demo</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>No credit card required • 1,000 free requests/month • Cancel anytime</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0;">
    <p><strong>NeuroVest Forecasting API</strong></p>
    <p>AI-Powered Market Intelligence | 25-Year Proven Track Record | 59+ Assets</p>
    <p style="font-size: 0.9rem;">
        <a href="https://neurovestdemo.streamlit.app/" style="color: #3498db; text-decoration: none;">Documentation</a> •
        <a href="https://neurovestdemo.streamlit.app/" style="color: #3498db; text-decoration: none;">Support</a> •
        <a href="https://neurovestdemo.streamlit.app/" style="color: #3498db; text-decoration: none;">Terms of Service</a>
    </p>
</div>
""", unsafe_allow_html=True)
