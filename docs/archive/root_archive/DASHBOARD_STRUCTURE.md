# Dashboard Structure

## File Hierarchy

```
NeuroVest/
├── demo.py                          # CLI demo (terminal-based)
├── dashboard.py                     # Basic Streamlit interface
├── dashboard_comprehensive.py       # Full-featured Streamlit dashboard (internal/dev)
└── api_demo.py                      # Customer-facing API showcase (business)
```

## Purpose of Each File

### demo.py
Terminal script for quick testing. Shows predictions, model status, and backtest results via command line.

**Use when:** You want to verify system status without launching a web server.

**Run:** `python3 demo.py` or `python3 demo.py --full`

---

### dashboard.py
Basic Streamlit interface with standard features:
- Asset data viewing
- Price charts
- Prediction display
- Backtest results

**Use when:** You need a simple web UI for local development.

**Run:** `streamlit run dashboard.py`

---

### dashboard_comprehensive.py
Full internal dashboard with all features:
- All 41 assets (stocks, ETFs, crypto, metals)
- Recession probability analysis
- Valuation detection
- LLM market analysis
- Portfolio rebalancing
- Custom data imports
- Full automation pipeline

**Use when:** You need access to all platform capabilities. This is the main working dashboard for development and testing.

**Run:** `streamlit run dashboard_comprehensive.py`

**Target users:** Internal development, research, testing all features

---

### api_demo.py
Customer-facing API demonstration focused on business value:
- API playground with live examples
- Pricing tiers
- Integration code samples (Python, JavaScript)
- Real-world use cases
- Call-to-action for signups

**Use when:** Showcasing the platform to potential customers or investors.

**Run:** `streamlit run api_demo.py`

**Target users:** Prospective customers, business development, sales demos

**Deployment:** Should be hosted at custom domain (demo.neurovest.ai or api.neurovest.ai)

---

## Recommended Workflow

**For Development:**
1. Use `demo.py` for quick checks
2. Use `dashboard_comprehensive.py` for full feature testing
3. Use `dashboard.py` if you only need basic charts

**For Demonstrations:**
- **Technical audience (developers, researchers):** `dashboard_comprehensive.py`
- **Business audience (customers, investors):** `api_demo.py`

**For Deployment:**
- Deploy `api_demo.py` to production with custom domain for customer demos
- Keep `dashboard_comprehensive.py` on internal staging for team access
- Share `demo.py` in GitHub for CLI users

## Feature Matrix

| Feature | demo.py | dashboard.py | dashboard_comprehensive.py | api_demo.py |
|---------|---------|--------------|---------------------------|-------------|
| Predictions | ✓ (CLI) | ✓ | ✓ | Sample only |
| Price charts | ✗ | ✓ | ✓ | ✗ |
| Asset manager | ✗ | Limited | ✓ (41 assets) | ✗ |
| Recession indicator | ✗ | ✗ | ✓ | ✗ |
| Valuation detector | ✗ | ✗ | ✓ | ✗ |
| LLM analysis | ✗ | ✗ | ✓ | Sample only |
| Portfolio rebalancing | ✗ | ✗ | ✓ | ✗ |
| API playground | ✗ | ✗ | ✗ | ✓ |
| Pricing info | ✗ | ✗ | ✗ | ✓ |
| Integration examples | ✗ | ✗ | Limited | ✓ (Multiple languages) |
| Business use cases | ✗ | ✗ | ✗ | ✓ |

## Deployment Strategy

### Local Development
All dashboards can run locally on different ports:
```bash
streamlit run dashboard.py --server.port 8501
streamlit run dashboard_comprehensive.py --server.port 8502
streamlit run api_demo.py --server.port 8503
```

### Production Deployment

**Internal Dashboard (dashboard_comprehensive.py):**
- Deploy to private URL or password-protected Streamlit Cloud
- Use for team access, testing, and research
- Not public-facing

**Customer Demo (api_demo.py):**
- Deploy to professional hosting (Render, Railway, or AWS)
- Custom domain: demo.neurovest.ai or api.neurovest.ai
- Integrated with main landing page
- Analytics tracking enabled
- Performance optimized

**GitHub Demo:**
- `demo.py` showcased in README as CLI option
- Optionally deploy basic `dashboard.py` to Streamlit Community Cloud
- Link from GitHub repository

## Customization

**To modify the internal dashboard:**
Edit `dashboard_comprehensive.py` and add new sections to the navigation sidebar.

**To modify the customer demo:**
Edit `api_demo.py` focusing on business value, pricing, and integration examples.

**To add new features:**
1. Implement in `dashboard_comprehensive.py` first
2. If customer-relevant, add simplified version to `api_demo.py`
3. Update this documentation

## Navigation

Each Streamlit dashboard uses sidebar navigation:

**dashboard_comprehensive.py:**
- Overview
- Asset Manager
- Recession Indicator
- Valuation Detector
- LLM Analysis
- Portfolio Rebalancing
- Forecast Results
- Automation
- Custom Imports

**api_demo.py:**
- Single-page scrollable layout
- Sections: Hero, Features, API Playground, Pricing, Integration, Use Cases, CTA

## Future Considerations

**Potential additions:**
- Mobile-responsive version of api_demo.py
- Video demos embedded in api_demo.py
- Live chat support widget in api_demo.py
- A/B testing different CTAs in api_demo.py
- White-label version of dashboard_comprehensive.py for enterprise customers

**Performance:**
- dashboard_comprehensive.py uses @st.cache_data for asset loading
- api_demo.py has no data dependencies, loads instantly
- Consider Redis caching if API playground becomes live

## Support

For dashboard issues or feature requests, check:
1. This documentation
2. README.md for setup instructions
3. GitHub issues for known problems
