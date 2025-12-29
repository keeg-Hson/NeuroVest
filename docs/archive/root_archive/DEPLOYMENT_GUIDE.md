# NeuroVest Render Deployment Guide

## 🚀 Quick Deploy to Render (2 Options)

You have **TWO dashboards** ready to deploy:
1. **api_demo.py** - Customer-facing API showcase (recommended for public)
2. **dashboard_comprehensive.py** - Full-featured sandbox dashboard

---

## Option A: Deploy Using Blueprint (EASIEST - Both Apps)

### Step 1: Push to GitHub
```bash
git add .streamlit/config.toml render.yaml DEPLOYMENT_GUIDE.md
git commit -m "Add Render deployment configuration"
git push origin claude/assess-codebase-AqOfb
```

### Step 2: Deploy on Render
1. Go to [https://render.com](https://render.com)
2. Sign up or log in (free account works)
3. Click **"New +"** → **"Blueprint"**
4. Connect your GitHub account
5. Select your **NeuroVest** repository
6. Select branch: **claude/assess-codebase-AqOfb** (or main if merged)
7. Click **"Apply"**

✅ **Result:** Both dashboards will deploy automatically!
- `neurovest-api-demo` → Customer showcase
- `neurovest-dashboard` → Full sandbox

---

## Option B: Deploy Single Dashboard Manually

### For API Demo (Customer-Facing):

1. Go to [https://render.com](https://render.com) → Sign in
2. Click **"New +"** → **"Web Service"**
3. Connect GitHub repo: **NeuroVest**
4. Configure:
   - **Name:** `neurovest-api-demo`
   - **Region:** Oregon (or closest)
   - **Branch:** `claude/assess-codebase-AqOfb`
   - **Root Directory:** (leave blank)
   - **Runtime:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```
     streamlit run api_demo.py --server.port=$PORT --server.address=0.0.0.0
     ```
   - **Plan:** Free

5. Click **"Create Web Service"**
6. Wait 3-5 minutes for deployment

✅ **Your app will be live at:** `https://neurovest-api-demo.onrender.com`

---

### For Comprehensive Dashboard:

Same steps but use:
- **Name:** `neurovest-dashboard`
- **Start Command:**
  ```
  streamlit run dashboard_comprehensive.py --server.port=$PORT --server.address=0.0.0.0
  ```

---

## 🎨 What's Already Configured

✅ Dark theme locked in (`.streamlit/config.toml`)
✅ All dependencies in `requirements.txt`
✅ Port configuration for Render
✅ Streamlit server settings
✅ CORS and security settings

---

## ⚙️ Environment Variables (Optional)

If you want LLM features, add these in Render dashboard:

1. Go to your service → **"Environment"**
2. Add:
   - `OPENAI_API_KEY` = `sk-your-key`
   - `ANTHROPIC_API_KEY` = `sk-ant-your-key`
   - `NEWS_API_KEY` = `your-newsapi-key`

---

## 📝 Custom Domain (Optional)

1. In Render dashboard → Your service
2. Go to **"Settings"**
3. Scroll to **"Custom Domain"**
4. Add your domain (e.g., `api.neurovest.io`)
5. Update DNS CNAME record to Render's URL

---

## 🔥 Free Tier Limits

- **Spin down after 15 min** of inactivity
- **750 hours/month** free (enough for 1 service 24/7)
- **First request slow** after spin-down (~30s)
- Upgrade to **$7/mo** for always-on

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Check Python version
echo "3.11.0" > runtime.txt
git add runtime.txt
git commit -m "Specify Python version"
git push
```

### Port Issues
✅ Already fixed - using `$PORT` environment variable

### Missing Logo
The logo won't appear until you add `assets/neurovest_logo.png` to the repo.
Currently shows fallback emoji 📈

### Dashboard Won't Load
1. Check Render logs: Dashboard → "Logs" tab
2. Common fix: Clear build cache
   - Dashboard → "Settings" → "Clear build cache & deploy"

---

## 📊 Monitoring

View logs in real-time:
1. Render Dashboard → Your service
2. Click **"Logs"** tab
3. See all Streamlit output

---

## 🚀 Recommended Next Steps

1. **Deploy api_demo.py first** (customer-facing)
2. Test at your Render URL
3. **Deploy dashboard_comprehensive.py** as private internal tool
4. Add logo PNG to repo for full branding
5. Consider custom domain for production

---

## 💡 Tips

- **Free tier spins down:** First load takes ~30s
- **Keep alive:** Use a ping service (UptimeRobot, cron-job.org)
- **Updates:** Push to GitHub → Auto-deploys on Render
- **Multiple environments:** Deploy from different branches

---

**Need help?** Check Render logs or Streamlit Cloud docs.
