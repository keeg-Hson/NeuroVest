# NeuroVest Assets

## Logo

Place the NeuroVest logo file in this directory as:
- `neurovest_logo.png`

The logo will automatically appear in:
- `api_demo.py` - Customer-facing API demo
- `dashboard_comprehensive.py` - Full-featured dashboard
- Main `README.md` - GitHub repository

### Logo Specifications

- Format: PNG with transparent background
- Recommended size: 800x800px or larger
- Current design: Blue hexagon with neural network pattern and "NeuroVest" text
- Color scheme: Blue (#3498db) to match dashboard theme

### Usage

The streamlit dashboards check for logo existence and display it centered at the top if available. If the logo file is not present, the dashboards will display without it (graceful degradation).
