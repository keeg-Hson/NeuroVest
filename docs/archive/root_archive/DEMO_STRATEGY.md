# Demo Strategy

## Two-Demo Approach

NeuroVest uses separate demos for technical users (GitHub) vs business users (API customers).

### GitHub Demo (Technical Users)

**Files:** demo.py, dashboard_comprehensive.py

**Audience:** Developers, quants, researchers exploring the codebase

**Focus:**
- Full system capabilities (all 41 assets)
- Model architecture details
- Configurable thresholds and parameters
- Complete backtest metrics
- Setup and installation guides

**Tone:** Technical, educational, transparent

**Deployment:**
- demo.py: Included in GitHub repo for CLI testing
- dashboard_comprehensive.py: Optional Streamlit Cloud deployment
- Linked from README

### API Demo (Business Users)

**File:** api_demo.py

**Audience:** Hedge funds, fintech platforms, institutional traders

**Focus:**
- API playground with live examples
- Pricing tiers ($0 to Enterprise)
- Integration code (Python, JavaScript, cURL)
- Real-world use cases (hedge funds, quant desks)
- Performance metrics (191% return, 2.55 Sharpe)
- Signup flow

**Tone:** Professional, business-focused, ROI-driven

**Deployment:**
- Render/Railway paid tier
- Custom domain (demo.neurovest.ai)
- Linked from landing page

## Current Status

### Complete
- demo.py: CLI demo for terminal testing
- dashboard_comprehensive.py: Full internal dashboard
- api_demo.py: Customer-facing API showcase

### api_demo.py Sections

1. Hero: "Market Intelligence API for Quant Developers"
2. Quick Stats: 191% return, 2.55 Sharpe, -5.4% drawdown, 59+ assets
3. Features: Real-time predictions, batch analysis, custom backtests
4. API Playground: Live examples (single prediction, batch, backtest)
5. Pricing: Free, Developer ($99), Professional ($499), Enterprise (custom)
6. Integration: Code samples (Python, JavaScript, cURL)
7. Use Cases: Hedge fund risk, sector rotation, fintech integration, prop trading
8. CTA: "Start Free Trial" + "Schedule Demo"

## Why Separate Demos

GitHub demo targets technical users who care about:
- Full codebase access
- Model architecture
- Local installation
- Contributing to open source

API demo targets business users who care about:
- Pricing
- Integration time
- ROI and performance
- Support and SLAs

Combining them creates mixed messaging and confuses both audiences.

## Deployment

### GitHub Demo
- demo.py in repository (CLI)
- Optional: dashboard_comprehensive.py on Streamlit Community Cloud
- Free tier, linked from README

### API Demo
- Deploy api_demo.py to Render/Railway
- Custom domain: demo.neurovest.ai
- Professional hosting with analytics

### Cross-Promotion
- GitHub demo links to "API access available"
- API demo links to "View source on GitHub"

Both demos serve their audiences without mixing messages.
