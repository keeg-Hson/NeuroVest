# Logo Setup Instructions

## Adding the NeuroVest Logo

1. **Save the logo file:**
   - Name it exactly: `neurovest_logo.png`
   - Place it in this directory (`assets/`)
   - The file should be PNG format with transparent background
   - Recommended size: 800x800px or larger

2. **Where the logo appears:**
   - **Favicon** (browser tab icon) - Both dashboards
   - **Header** - Centered at top of both api_demo.py and dashboard_comprehensive.py
   - **README.md** - Centered before title on GitHub

3. **Current setup:**
   - Logo references are already in place
   - Dashboards check if file exists before displaying
   - If logo is missing, dashboards use emoji fallback (📊)
   - Everything works with or without the logo

## Quick Start

```bash
# From the image you have, save it as:
cp /path/to/your-logo.png assets/neurovest_logo.png

# Verify it's in place:
ls -la assets/neurovest_logo.png

# Logo will now appear in:
# - streamlit run api_demo.py
# - streamlit run dashboard_comprehensive.py
# - GitHub README (after git push)
```

## Logo Specifications

Based on the design shown:
- **Format:** PNG with transparency
- **Design:** Blue hexagon with neural network node pattern
- **Text:** "NeuroVest" in blue below the icon
- **Color:** #3498db (matches UI theme)
- **Dimensions:** Square aspect ratio (e.g., 800x800px, 1000x1000px)

The logo will automatically:
- Resize to 200px width in api_demo.py
- Resize to 250px width in dashboard_comprehensive.py
- Display at 200px width in README.md
- Appear as favicon in browser tabs (16x16px and 32x32px)
