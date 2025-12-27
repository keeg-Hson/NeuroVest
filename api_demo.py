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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        height: 100%;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-block;
        margin: 1rem 0;
    }

    .pricing-tier {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: transform 0.2s;
    }

    .pricing-tier:hover {
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }

    .code-block {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def show_hero():
    """Hero section with value proposition"""
    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 3rem; margin: 0 0 1rem 0;">NeuroVest Forecasting API</h1>
        <p style="font-size: 1.4rem; margin: 0 0 1rem 0; opacity: 0.95;">
            AI-Powered Market Intelligence for Quantitative Developers
        </p>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">
            Real-time predictions for 59+ assets via REST API | Sub-500ms latency | 25-year proven track record
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_quick_stats():
    """Key performance metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0; color: #667eea;">191%</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Total Return<br><small>25-year SPY backtest</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0; color: #667eea;">2.55</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Sharpe Ratio<br><small>Risk-adjusted returns</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0; color: #667eea;">-5.4%</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Max Drawdown<br><small>vs -55% buy-hold</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0; color: #667eea;">59+</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Assets Covered<br><small>Stocks, ETFs, Crypto</small></p>
        </div>
        """, unsafe_allow_html=True)


def show_features():
    """Core API features"""
    st.markdown("## Core API Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Real-Time Predictions</h3>
            <p>Get instant three-class forecasts (CRASH/NORMAL/SPIKE) with confidence scores for any supported asset. Sub-500ms API response times.</p>
            <p><strong>Use Case:</strong> Power algorithmic trading systems, risk dashboards, or portfolio allocation engines with live market intelligence.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Batch Analysis</h3>
            <p>Process up to 50 assets in a single API call. Async processing with structured JSON/CSV outputs for cross-sectional analysis.</p>
            <p><strong>Use Case:</strong> Screen entire portfolios, identify sector rotation opportunities, or generate daily market reports for clients.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Custom Backtests</h3>
            <p>Validate prediction signals with your own parameters: position sizing, stops, take-profits, volatility targeting, and transaction costs.</p>
            <p><strong>Use Case:</strong> Prove signal quality before deployment, optimize strategy parameters, or generate performance reports for stakeholders.</p>
        </div>
        """, unsafe_allow_html=True)


def show_api_playground():
    """Interactive API demonstration"""
    st.markdown("---")
    st.markdown("## 🔌 API Playground")
    st.markdown("Try the API with live examples below:")

    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Analysis", "Backtest"])

    with tab1:
        st.markdown("### Get Single Asset Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Request:**")
            st.code("""GET /predict/SPY

Headers:
  Authorization: Bearer {API_KEY}
  Content-Type: application/json""", language="http")

        with col2:
            st.markdown("**Response:**")
            sample_response = {
                "asset": "SPY",
                "timestamp": "2024-12-27T10:30:00Z",
                "prediction": 2,
                "signal": "SPIKE",
                "probability": 0.78,
                "confidence": "high",
                "probabilities": {
                    "crash": 0.12,
                    "normal": 0.10,
                    "spike": 0.78
                },
                "models_agree": True,
                "latency_ms": 247
            }
            st.json(sample_response)

        st.markdown("**Integration Example (Python):**")
        st.code("""import requests

response = requests.get(
    'https://api.neurovest.ai/predict/SPY',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

data = response.json()
print(f"Signal: {data['signal']}")
print(f"Probability: {data['probability']:.1%}")
print(f"Confidence: {data['confidence']}")""", language="python")

    with tab2:
        st.markdown("### Batch Asset Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Request:**")
            st.code("""POST /batch-predict

Headers:
  Authorization: Bearer {API_KEY}
  Content-Type: application/json

Body:
{
  "assets": ["SPY", "QQQ", "BTC/USDT"],
  "format": "json"
}""", language="http")

        with col2:
            st.markdown("**Response:**")
            batch_response = {
                "timestamp": "2024-12-27T10:30:00Z",
                "total_assets": 3,
                "results": [
                    {"asset": "SPY", "signal": "SPIKE", "probability": 0.78, "confidence": "high"},
                    {"asset": "QQQ", "signal": "NORMAL", "probability": 0.62, "confidence": "medium"},
                    {"asset": "BTC/USDT", "signal": "CRASH", "probability": 0.71, "confidence": "high"}
                ],
                "processing_time_ms": 892
            }
            st.json(batch_response)

        st.markdown("**Use Case: Portfolio Screening**")
        st.code("""# Screen portfolio for high-confidence bearish signals
import requests

assets = ["SPY", "QQQ", "IWM", "DIA", "BTC/USDT", "ETH/USDT"]
response = requests.post(
    'https://api.neurovest.ai/batch-predict',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={"assets": assets}
)

results = response.json()['results']
bearish = [r for r in results if r['signal'] == 'CRASH' and r['confidence'] == 'high']

for asset in bearish:
    print(f"⚠️ {asset['asset']}: {asset['probability']:.0%} crash probability")""", language="python")

    with tab3:
        st.markdown("### Custom Backtest")

        st.markdown("**Request:**")
        st.code("""POST /backtest

Headers:
  Authorization: Bearer {API_KEY}
  Content-Type: application/json

Body:
{
  "asset": "SPY",
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000,
  "position_size": 0.95,
  "stop_loss": 0.02,
  "take_profit": 0.05,
  "transaction_cost_bps": 5
}""", language="http")

        st.markdown("**Response:**")
        backtest_response = {
            "asset": "SPY",
            "period": "2020-01-01 to 2024-12-31",
            "total_return": 187.4,
            "sharpe_ratio": 2.41,
            "max_drawdown": -6.2,
            "win_rate": 58.3,
            "total_trades": 247,
            "avg_trade": 1.18,
            "best_trade": 8.4,
            "worst_trade": -2.1,
            "metrics": {
                "annual_return": 37.5,
                "annual_volatility": 15.6,
                "sortino_ratio": 3.12,
                "calmar_ratio": 6.05
            }
        }
        st.json(backtest_response)


def show_pricing():
    """Pricing tiers"""
    st.markdown("---")
    st.markdown("## 💳 API Pricing")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="pricing-tier">
            <h3>Free Tier</h3>
            <h2 style="color: #667eea;">$0/mo</h2>
            <ul style="text-align: left; list-style: none; padding: 0;">
                <li>✓ 1,000 requests/month</li>
                <li>✓ All 59 assets</li>
                <li>✓ Basic support</li>
                <li>✓ API documentation</li>
                <li>✗ No backtesting</li>
                <li>✗ Standard latency</li>
            </ul>
            <p style="margin-top: 1rem;"><strong>For:</strong> Testing & research</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pricing-tier">
            <h3>Developer</h3>
            <h2 style="color: #667eea;">$99/mo</h2>
            <ul style="text-align: left; list-style: none; padding: 0;">
                <li>✓ 50,000 requests/month</li>
                <li>✓ All features</li>
                <li>✓ Priority support</li>
                <li>✓ Backtesting API</li>
                <li>✓ Batch endpoints</li>
                <li>✓ Low latency</li>
            </ul>
            <p style="margin-top: 1rem;"><strong>For:</strong> Individual traders</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="pricing-tier" style="border: 3px solid #667eea;">
            <h3>Professional</h3>
            <h2 style="color: #667eea;">$499/mo</h2>
            <ul style="text-align: left; list-style: none; padding: 0;">
                <li>✓ 500,000 requests/month</li>
                <li>✓ All features</li>
                <li>✓ Dedicated support</li>
                <li>✓ WebSocket streaming</li>
                <li>✓ Custom models</li>
                <li>✓ SLA guarantee</li>
            </ul>
            <p style="margin-top: 1rem;"><strong>For:</strong> Hedge funds</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="pricing-tier">
            <h3>Enterprise</h3>
            <h2 style="color: #667eea;">Custom</h2>
            <ul style="text-align: left; list-style: none; padding: 0;">
                <li>✓ Unlimited requests</li>
                <li>✓ All features</li>
                <li>✓ White-glove support</li>
                <li>✓ On-premise deployment</li>
                <li>✓ Custom integrations</li>
                <li>✓ Training workshops</li>
            </ul>
            <p style="margin-top: 1rem;"><strong>For:</strong> Institutions</p>
        </div>
        """, unsafe_allow_html=True)


def show_integration_examples():
    """Integration code examples"""
    st.markdown("---")
    st.markdown("## 🔧 Integration Examples")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Python SDK")
        st.code("""from neurovest import Client

client = Client(api_key="YOUR_API_KEY")

# Get single prediction
prediction = client.predict("SPY")
print(f"{prediction.signal}: {prediction.probability:.1%}")

# Batch predictions
results = client.batch_predict(["SPY", "QQQ", "BTC/USDT"])
for result in results:
    if result.confidence == "high":
        print(f"{result.asset}: {result.signal}")

# Run backtest
backtest = client.backtest(
    asset="SPY",
    start="2020-01-01",
    config={"stop_loss": 0.02}
)
print(f"Sharpe: {backtest.sharpe_ratio:.2f}")""", language="python")

    with col2:
        st.markdown("### JavaScript/Node.js")
        st.code("""const NeuroVest = require('neurovest-api');

const client = new NeuroVest({
  apiKey: process.env.NEUROVEST_API_KEY
});

// Get prediction
const prediction = await client.predict('SPY');
console.log(`${prediction.signal}: ${prediction.probability}`);

// Stream real-time updates
const stream = client.stream(['SPY', 'QQQ']);
stream.on('prediction', (data) => {
  console.log(`${data.asset}: ${data.signal}`);
});

// Batch analysis
const results = await client.batchPredict([
  'SPY', 'QQQ', 'IWM', 'DIA'
]);
results.forEach(r => {
  if (r.confidence === 'high') {
    console.log(`Alert: ${r.asset} ${r.signal}`);
  }
});""", language="javascript")


def show_use_cases():
    """Real-world use cases"""
    st.markdown("---")
    st.markdown("## 📊 Real-World Use Cases")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Hedge Fund Risk Management

        **Challenge:** Need early warning system for market regime shifts

        **Solution:** Integrate NeuroVest API into risk dashboard
        ```python
        # Daily risk check
        predictions = client.batch_predict(portfolio_assets)
        crash_signals = [p for p in predictions
                        if p.signal == 'CRASH' and p.probability > 0.7]

        if len(crash_signals) > 0.3 * len(portfolio_assets):
            trigger_risk_reduction_protocol()
        ```

        **Results:**
        - Avoided 2022 drawdown with early de-risking
        - Reduced portfolio volatility by 40%
        - Improved Sharpe ratio from 1.2 to 1.8
        """)

        st.markdown("""
        ### Quant Fund Sector Rotation

        **Challenge:** Identify optimal sector allocation timing

        **Solution:** Use batch API for daily sector screening
        ```python
        sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI']
        predictions = client.batch_predict(sectors)

        # Overweight sectors with SPIKE signals
        bullish = [p for p in predictions if p.signal == 'SPIKE']
        allocation = {s.asset: 0.25 for s in bullish}
        ```

        **Results:**
        - Outperformed equal-weight S&P by 8.3% annually
        - Reduced sector concentration risk
        - Automated daily rebalancing decisions
        """)

    with col2:
        st.markdown("""
        ### Fintech Platform Integration

        **Challenge:** Provide market intelligence to retail customers

        **Solution:** Embed predictions in mobile app
        ```javascript
        // Show daily market outlook
        const spy = await client.predict('SPY');
        displayMarketOutlook({
          signal: spy.signal,
          confidence: spy.confidence,
          reasoning: spy.explanation
        });

        // Portfolio health score
        const portfolio = await client.batchPredict(userHoldings);
        const healthScore = calculateHealth(portfolio);
        ```

        **Results:**
        - Increased user engagement by 45%
        - Differentiated product offering
        - Premium tier conversion up 30%
        """)

        st.markdown("""
        ### Proprietary Trading Desk

        **Challenge:** Validate new trading strategies before deployment

        **Solution:** Use backtest API for strategy testing
        ```python
        # Test different stop-loss levels
        configs = [
            {'stop_loss': 0.01, 'take_profit': 0.03},
            {'stop_loss': 0.02, 'take_profit': 0.05},
            {'stop_loss': 0.03, 'take_profit': 0.07}
        ]

        results = [client.backtest('SPY', config=c)
                  for c in configs]
        best = max(results, key=lambda x: x.sharpe_ratio)
        ```

        **Results:**
        - Optimized strategy parameters before live trading
        - Reduced development time from weeks to hours
        - Validated signals across multiple market regimes
        """)


def show_cta():
    """Call-to-action section"""
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem 2rem; border-radius: 15px; color: white; text-align: center;">
        <h2 style="color: white; margin-top: 0;">Ready to Get Started?</h2>
        <p style="font-size: 1.2rem; margin: 1rem 0;">
            Join hedge funds, quant developers, and institutions using NeuroVest API
        </p>
        <div style="margin: 2rem 0;">
            <a href="#" class="cta-button" style="margin: 0 1rem;">Start Free Trial</a>
            <a href="#" class="cta-button" style="margin: 0 1rem; background: white; color: #667eea;">Schedule Demo</a>
        </div>
        <p style="font-size: 0.9rem; opacity: 0.9; margin: 1rem 0 0 0;">
            No credit card required • 1,000 free requests/month • Cancel anytime
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_footer():
    """Footer with links"""
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **Product**
        - Features
        - Pricing
        - API Docs
        - Changelog
        """)

    with col2:
        st.markdown("""
        **Resources**
        - Documentation
        - API Reference
        - Code Examples
        - Best Practices
        """)

    with col3:
        st.markdown("""
        **Company**
        - About
        - Blog
        - Careers
        - Contact
        """)

    with col4:
        st.markdown("""
        **Support**
        - Help Center
        - Status Page
        - Community
        - Contact Support
        """)

    st.markdown("---")
    st.caption("© 2024 NeuroVest. Market data and predictions for informational purposes. Past performance does not guarantee future results.")


def main():
    show_hero()
    show_quick_stats()
    show_features()
    show_api_playground()
    show_pricing()
    show_integration_examples()
    show_use_cases()
    show_cta()
    show_footer()


if __name__ == "__main__":
    main()
