#!/usr/bin/env python3
"""
Dashboard Improvements Module

This module contains improved implementations of dashboard functions
that call the actual backend scripts instead of showing mock data.

Usage:
    Import these functions in dashboard_comprehensive.py to replace
    the mock implementations.
"""

import subprocess
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import time


def show_recession_indicator_improved():
    """
    Improved recession indicator that calls the actual script.
    """
    st.title("📉 Recession Probability Indicator")
    st.markdown("*Multi-factor recession risk analysis using actual backend script*")

    st.info("💡 **Feature:** Calls `recession_indicator.py` for real-time recession probability analysis")

    # Check SPY data availability
    spy_path = Path("data/SPY.csv")
    if not spy_path.exists():
        st.error("🔴 **SPY Data Required**")
        st.markdown("""
        The recession indicator requires SPY (S&P 500) data.

        **Download SPY data:**
        ```bash
        python3 update_spy_data.py
        ```
        """)

        if st.button("🔄 Download SPY Data Now"):
            with st.spinner("Downloading SPY data..."):
                result = subprocess.run(
                    ["python3", "update_spy_data.py"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    st.success("✅ SPY data downloaded successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Download failed: {result.stderr[:500]}")
        return

    # Run analysis button
    if st.button("🚀 Run Recession Analysis", type="primary"):
        with st.spinner("Running recession analysis..."):
            try:
                result = subprocess.run(
                    ["python3", "recession_indicator.py"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    st.success("✅ Analysis Complete")

                    # Display output
                    with st.expander("📊 Full Analysis Report", expanded=True):
                        st.text(result.stdout)

                    # Try to extract key metrics
                    output = result.stdout
                    if "recession" in output.lower() or "risk" in output.lower():
                        st.markdown("---")
                        st.markdown("### 📈 Analysis Summary")
                        st.info("Review the full report above for detailed recession indicators")

                else:
                    st.error("❌ Analysis failed")
                    with st.expander("Error Details"):
                        st.code(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Analysis timed out (>60s)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.info("👆 Click the button above to run the recession analysis")

    st.markdown("---")
    st.markdown("**Manual Execution:**")
    st.code("python3 recession_indicator.py --save", language="bash")


def show_valuation_detector_improved():
    """
    Improved valuation detector that calls the actual script.
    """
    st.title("💰 Valuation Detector")
    st.markdown("*Asset over/undervaluation analysis using actual backend script*")

    st.info("💡 **Feature:** Calls `valuation_detector.py` for real valuation analysis")

    # Get available assets
    assets = []

    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.glob("*.csv"):
            if f.stem not in ['cross_asset_features', 'macro_features', 'sentiment_features']:
                assets.append(f.stem)

    # Check cache directory
    cache_dir = Path("data_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*_1d.csv"):
            ticker = f.stem.replace("_1d", "").replace("_", "/")
            if ticker not in assets:
                assets.append(ticker)

    if not assets:
        st.warning("⚠️ No assets found. Download data first:")
        st.code("python3 update_spy_data.py\npython3 download_equity_etfs.py")
        return

    # Asset selector
    selected_asset = st.selectbox("Select Asset for Valuation Analysis", sorted(assets))

    # Analysis button
    if st.button(f"🚀 Analyze {selected_asset} Valuation", type="primary"):
        with st.spinner(f"Analyzing {selected_asset} valuation..."):
            try:
                result = subprocess.run(
                    ["python3", "valuation_detector.py", "--asset", selected_asset],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    st.success(f"✅ Valuation Analysis Complete for {selected_asset}")

                    # Display output
                    with st.expander("📊 Full Valuation Report", expanded=True):
                        st.text(result.stdout)

                    # Extract valuation classification if available
                    output = result.stdout
                    if "OVERVALUED" in output:
                        st.error("🔴 **OVERVALUED** - Consider taking profits")
                    elif "UNDERVALUED" in output:
                        st.success("🟢 **UNDERVALUED** - Potential buying opportunity")
                    elif "FAIRLY VALUED" in output or "NEUTRAL" in output:
                        st.info("🟡 **FAIRLY VALUED** - Hold current positions")

                else:
                    st.error(f"❌ Analysis failed for {selected_asset}")
                    with st.expander("Error Details"):
                        st.code(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Analysis timed out (>60s)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.info(f"👆 Click the button above to analyze {selected_asset} valuation")

    st.markdown("---")
    st.markdown("**Available Commands:**")
    st.code(f"""
# Analyze specific asset
python3 valuation_detector.py --asset {selected_asset}

# Analyze all assets
python3 valuation_detector.py --all --save
    """, language="bash")


def show_llm_analysis_improved():
    """
    Improved LLM analysis that calls the actual script.
    """
    st.title("🤖 LLM Market Analysis")
    st.markdown("*AI-powered market commentary using GPT-4 or Claude*")

    st.info("💡 **Feature:** Calls `llm_forecast.py` for real AI-generated market analysis")

    # Check API key configuration
    st.markdown("### ⚙️ Configuration")

    env_file = Path(".env")
    has_env = env_file.exists()

    if not has_env:
        st.warning("⚠️ No .env file found. API keys required for LLM analysis.")
        st.markdown("""
        **Setup Instructions:**
        1. Copy .env.example to .env:
           ```bash
           cp .env.example .env
           ```

        2. Add your API keys:
           ```
           OPENAI_API_KEY=sk-your-key-here
           # OR
           ANTHROPIC_API_KEY=sk-ant-your-key-here
           ```

        3. Optionally add NewsAPI key:
           ```
           NEWS_API_KEY=your-newsapi-key
           ```
        """)
        return

    # Provider selection
    provider = st.radio("Select LLM Provider:", ["openai", "anthropic"])

    # Asset selection
    assets = []
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.glob("*.csv"):
            if f.stem not in ['cross_asset_features', 'macro_features', 'sentiment_features']:
                assets.append(f.stem)

    cache_dir = Path("data_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*_1d.csv"):
            ticker = f.stem.replace("_1d", "").replace("_", "/")
            if ticker not in assets:
                assets.append(ticker)

    if not assets:
        st.warning("⚠️ No assets found.")
        return

    selected_asset = st.selectbox("Select Asset", sorted(assets))

    # Generate analysis button
    if st.button(f"🚀 Generate AI Analysis for {selected_asset}", type="primary"):
        with st.spinner(f"Generating AI analysis for {selected_asset}..."):
            try:
                result = subprocess.run(
                    ["python3", "llm_forecast.py", "--asset", selected_asset, "--provider", provider],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    st.success(f"✅ AI Analysis Generated for {selected_asset}")

                    # Display the analysis
                    with st.expander("🤖 AI Market Analysis", expanded=True):
                        st.markdown(result.stdout)

                else:
                    st.error("❌ Analysis failed")
                    error_msg = result.stderr

                    if "API key" in error_msg or "OPENAI_API_KEY" in error_msg or "ANTHROPIC_API_KEY" in error_msg:
                        st.error("🔑 API Key Error: Please check your .env file contains valid API keys")

                    with st.expander("Error Details"):
                        st.code(error_msg)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Analysis timed out (>120s)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.info(f"👆 Click the button above to generate AI analysis for {selected_asset}")

    st.markdown("---")
    st.markdown("**Available Commands:**")
    st.code(f"""
# Single asset analysis
python3 llm_forecast.py --asset {selected_asset} --provider {provider}

# Multi-asset summary
python3 llm_forecast.py --all --provider {provider}

# Generate newsletter
python3 newsletter_generator.py --preview --assets {selected_asset}
    """, language="bash")


def show_portfolio_rebalancing_improved():
    """
    Improved portfolio rebalancing that calls the actual script.
    """
    st.title("🔄 Portfolio Rebalancing Optimizer")
    st.markdown("*Find optimal rebalancing frequency using actual backend script*")

    st.info("💡 **Feature:** Calls `portfolio_rebalancer.py` for real portfolio optimization")

    # Get available assets
    assets = []
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.glob("*.csv"):
            if f.stem not in ['cross_asset_features', 'macro_features', 'sentiment_features']:
                assets.append(f.stem)

    cache_dir = Path("data_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*_1d.csv"):
            ticker = f.stem.replace("_1d", "").replace("_", "/")
            if ticker not in assets:
                assets.append(ticker)

    if len(assets) < 2:
        st.warning("⚠️ Need at least 2 assets for portfolio analysis. Download more data:")
        st.code("""
python3 update_spy_data.py
python3 download_equity_etfs.py
        """)
        return

    # Portfolio configuration
    st.markdown("### 📊 Portfolio Configuration")

    selected_assets = st.multiselect(
        "Select Assets for Portfolio",
        sorted(assets),
        default=sorted(assets)[:min(3, len(assets))]
    )

    if len(selected_assets) < 2:
        st.warning("⚠️ Select at least 2 assets")
        return

    # Weights input
    st.markdown("**Asset Weights** (must sum to 1.0):")

    weights = []
    cols = st.columns(len(selected_assets))
    for i, asset in enumerate(selected_assets):
        with cols[i]:
            weight = st.number_input(
                f"{asset}",
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(selected_assets),
                step=0.05,
                key=f"weight_{asset}"
            )
            weights.append(weight)

    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        st.error(f"⚠️ Weights must sum to 1.0 (currently: {total_weight:.2f})")
        return

    st.success(f"✅ Weights sum to {total_weight:.2f}")

    # Run optimization
    if st.button("🚀 Find Optimal Rebalancing Frequency", type="primary"):
        assets_str = ",".join(selected_assets)
        weights_str = ",".join([f"{w:.2f}" for w in weights])

        with st.spinner("Running portfolio optimization..."):
            try:
                result = subprocess.run(
                    [
                        "python3", "portfolio_rebalancer.py",
                        "--find-optimal",
                        "--assets", assets_str,
                        "--weights", weights_str
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180
                )

                if result.returncode == 0:
                    st.success("✅ Optimization Complete")

                    # Display results
                    with st.expander("📊 Optimization Results", expanded=True):
                        st.text(result.stdout)

                    # Try to parse and display summary
                    output = result.stdout
                    if "Optimal" in output or "Best" in output:
                        st.markdown("---")
                        st.markdown("### 🎯 Recommendation")
                        st.info("Review the detailed results above for the optimal rebalancing strategy")

                else:
                    st.error("❌ Optimization failed")
                    with st.expander("Error Details"):
                        st.code(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Optimization timed out (>180s). Portfolio may be too large.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.info("👆 Click the button above to find the optimal rebalancing frequency")

    st.markdown("---")
    st.markdown("**Manual Execution:**")
    assets_str = ",".join(selected_assets) if selected_assets else "SPY,GLD,TLT"
    weights_str = ",".join([f"{w:.2f}" for w in weights]) if weights else "0.60,0.30,0.10"

    st.code(f"""
python3 portfolio_rebalancer.py --find-optimal \\
    --assets {assets_str} \\
    --weights {weights_str}
    """, language="bash")


# Export functions for easy import
__all__ = [
    'show_recession_indicator_improved',
    'show_valuation_detector_improved',
    'show_llm_analysis_improved',
    'show_portfolio_rebalancing_improved'
]
