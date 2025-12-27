# Demo Page Strategy: GitHub vs Customer-Facing

## Overview

NeuroVest needs two distinct demo experiences to serve different audiences effectively. This document outlines the strategy for separating GitHub repository demos from customer-facing API demos.

## Recommendation: Two-Demo Approach

### 1. GitHub Demo (`demo.py`) - For Developers

**Purpose:** Showcase technical capabilities to developers who want to self-host or understand the codebase.

**Target Audience:**
- Open-source developers exploring the repository
- Quants evaluating the ML architecture
- Researchers interested in the methodology
- Contributors looking to extend functionality

**Key Features:**
- Full system demonstration with all assets
- Technical details about model architecture
- Adjustable prediction thresholds (sensitivity sliders)
- Backtest performance metrics with configuration details
- Links to documentation and GitHub repository
- Setup instructions and requirements

**Tone:** Technical, educational, transparent about methodology and performance metrics.

**Current Implementation:** `demo.py` (already optimized for this purpose)

**Deployment:**
- Streamlit Community Cloud (free tier)
- Linked from GitHub README
- URL: https://neurovest-demo.streamlit.app

---

### 2. Customer-Facing API Demo (`api_demo.py`) - For API Customers

**Purpose:** Demonstrate API value proposition to potential paying customers.

**Target Audience:**
- Institutional traders evaluating the API
- Quantitative developers at hedge funds
- Fintech platforms looking for prediction APIs
- Portfolio managers seeking market intelligence

**Key Features:**
- API endpoint examples with live responses
- Interactive API playground (try prediction calls)
- Pricing tiers and subscription options
- Performance metrics (Sharpe, returns, accuracy)
- Integration code snippets (Python, JavaScript, cURL)
- Beta waitlist signup
- Customer testimonials/case studies

**Tone:** Professional, business-focused, emphasizes value and ROI.

**Recommended Implementation:** Create new `api_demo.py` focused on API capabilities

**Deployment:**
- Professional hosting (Render/Railway paid tier)
- Custom domain: demo.neurovest.ai or api.neurovest.ai
- Integrated with landing page at neurovest.netlify.app

---

## Implementation Plan

### Phase 1: Optimize Existing GitHub Demo (Complete ✅)
- ✅ Updated `demo.py` with professional descriptions
- ✅ Removed defensive "not a trading system" language
- ✅ Enhanced feature explanations
- ✅ Improved backtest metrics presentation

### Phase 2: Create API-Focused Demo (Recommended Next Step)

Create `api_demo.py` with these sections:

1. **Hero Section**
   - "Market Intelligence API for Quantitative Developers"
   - Key value prop: "59+ assets, real-time predictions, 25-year backtest"
   - CTA: "Start Free Trial" or "Get API Key"

2. **Live API Playground**
   ```python
   # Interactive widget to test API calls
   import streamlit as st
   import requests

   asset = st.selectbox("Asset", ["SPY", "BTC/USDT", "QQQ"])
   if st.button("Get Prediction"):
       response = requests.post(
           "https://api.neurovest.ai/predict",
           json={"asset": asset},
           headers={"Authorization": f"Bearer {api_key}"}
       )
       st.json(response.json())
   ```

3. **Integration Examples**
   - Python SDK usage
   - REST API examples
   - WebSocket streaming
   - Batch prediction workflows

4. **Pricing Table**
   | Tier | Requests/Month | Price | Use Case |
   |------|----------------|-------|----------|
   | Free | 1,000 | $0 | Testing & research |
   | Developer | 50,000 | $99/mo | Individual traders |
   | Professional | 500,000 | $499/mo | Hedge funds |
   | Enterprise | Unlimited | Custom | Institutions |

5. **Performance Metrics Dashboard**
   - Real-time accuracy tracker
   - Live Sharpe ratio
   - Asset coverage matrix
   - Prediction latency stats

6. **Customer Logos / Testimonials**
   - "How XYZ Fund uses NeuroVest for sector rotation"
   - "Integrating NeuroVest into our risk management system"

7. **Signup Flow**
   - Email capture for beta waitlist
   - Direct signup for free tier
   - Contact sales for enterprise

---

## Key Differences Summary

| Aspect | GitHub Demo (`demo.py`) | API Demo (`api_demo.py`) |
|--------|-------------------------|-------------------------|
| **Audience** | Developers, researchers | Paying customers |
| **Focus** | Technical depth | Business value |
| **Tone** | Educational | Persuasive |
| **Primary CTA** | "Star on GitHub" | "Start Free Trial" |
| **Content** | Full system capabilities | API-specific features |
| **Deployment** | Free (Streamlit Cloud) | Paid (custom domain) |
| **Metrics Shown** | All technical details | Highlights + ROI |
| **Integration** | Links to docs | Signup flow |

---

## Why Separate Demos Work Better

### For GitHub Demo:
- Developers want to see the full system, not just the API
- Need to showcase local installation capabilities
- Educational content builds trust in methodology
- Code transparency is the value proposition

### For API Demo:
- Business buyers don't care about local setup
- Need clear pricing and value justification
- Professional presentation builds commercial credibility
- Focus on integration speed and reliability

### Combined Demo Risks:
- Confusing value proposition (is this open-source or paid?)
- Mixed messaging (technical vs business language)
- Different CTAs compete (star repo vs buy API)
- One audience gets neglected

---

## Example Content Differences

### GitHub Demo Heading:
> "NeuroVest - Economic Forecasting Platform
> Ensemble ML system for multi-asset market predictions. Open-source, fully documented, ready to deploy."

### API Demo Heading:
> "Market Intelligence API
> Get real-time predictions for 59+ assets via REST API. Used by hedge funds and quant developers worldwide."

---

## Deployment Strategy

### Current State:
- ✅ `demo.py` deployed to Streamlit Community Cloud
- ✅ Linked from GitHub README
- ⏳ `api_demo.py` to be created

### Recommended Next Steps:

1. **Create `api_demo.py`** (2-3 hours)
   - Focus on API value proposition
   - Add pricing table
   - Include signup flow
   - Showcase integration examples

2. **Deploy to Production** (30 minutes)
   - Use Render or Railway paid tier
   - Setup custom domain (demo.neurovest.ai)
   - Enable HTTPS
   - Add analytics tracking

3. **Update Landing Page** (1 hour)
   - Link API demo from neurovest.netlify.app
   - Separate CTA buttons:
     - "View GitHub Demo" → neurovest-demo.streamlit.app
     - "Try API Demo" → demo.neurovest.ai

4. **Add Beta Waitlist Backend** (optional, 2 hours)
   - Google Forms integration
   - Mailchimp signup
   - Custom database

---

## Conclusion

**Yes, GitHub and customer-facing demos should be different.** They serve distinct audiences with different goals:

- **`demo.py`**: Educate developers about the open-source system
- **`api_demo.py`**: Convert prospects into paying API customers

Both demos can coexist and cross-promote:
- GitHub demo can link to "Enterprise API available"
- API demo can link to "View open-source code on GitHub"

This dual-demo strategy maximizes reach while keeping messaging clear for each audience.
