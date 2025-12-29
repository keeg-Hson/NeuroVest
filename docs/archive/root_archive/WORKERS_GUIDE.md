# NeuroVest Workers & Cron Jobs Guide

Complete guide to setting up and managing background workers and scheduled tasks on Render.

---

## 🎯 Overview

NeuroVest uses **three types of Render services**:

1. **Web Services** - Streamlit dashboards (already deployed)
2. **Background Workers** - Continuous processes that run 24/7
3. **Cron Jobs** - Scheduled tasks that run at specific times

This is similar to **Railway workers** but uses Render's infrastructure.

---

## 📋 Current Worker Setup

### 🔄 Background Worker

**`neurovest-data-worker`** - Continuous data scheduler
- **Type**: Background Worker
- **Runtime**: 24/7
- **Purpose**: Keep market data fresh
- **Update Interval**: Every 60 minutes
- **Assets**: 17 stocks/commodities + 10 cryptocurrencies
- **Script**: `worker_data_scheduler.py`

**Features:**
- Automatic data updates from yfinance (stocks) and CCXT (crypto)
- Market hours awareness (stocks: 9 AM - 5 PM weekdays)
- Crypto updates 24/7
- Error handling with fallback to synthetic data
- Real-time status monitoring
- Graceful shutdown on SIGTERM/SIGINT

**Environment Variables:**
```bash
PYTHON_VERSION=3.11.14
UPDATE_INTERVAL=60
DATABASE_PATH=data/market_data.db
```

---

### ⏰ Cron Jobs

#### 1. Daily Predictions (`neurovest-daily-predictions`)
- **Schedule**: Mon-Fri at 4:30 PM EST (after market close)
- **Cron Expression**: `30 16 * * 1-5`
- **Script**: `cron_daily_predictions.py`

**What it does:**
1. Updates SPY data (market indicator)
2. Updates all asset data
3. Generates ensemble predictions for all 59 assets

**Output**: Fresh predictions ready for next trading day

---

#### 2. Weekly Model Retraining (`neurovest-weekly-retrain`)
- **Schedule**: Sundays at 2:00 AM EST
- **Cron Expression**: `0 2 * * 0`
- **Script**: `cron_weekly_retrain.py`

**What it does:**
1. Updates all market data
2. Retrains ML models (XGBoost, LightGBM, CatBoost) with latest data
3. Generates fresh predictions with retrained models

**Duration**: ~30-60 minutes depending on data size

---

## 🚀 Deployment Instructions

### Option 1: Blueprint Deployment (Recommended)

The `render.yaml` already includes all workers and cron jobs!

1. **Push to your branch:**
   ```bash
   git add .
   git commit -m "Add workers and cron jobs"
   git push -u origin claude/assess-codebase-AqOfb
   ```

2. **Render will automatically deploy:**
   - ✅ neurovest-api-demo (Web)
   - ✅ neurovest-dashboard (Web)
   - ✅ neurovest-data-worker (Worker)
   - ✅ neurovest-daily-predictions (Cron)
   - ✅ neurovest-weekly-retrain (Cron)

3. **Verify deployment:**
   - Go to Render Dashboard
   - Check all 5 services show "Deploy live"
   - Worker should show "Running"
   - Cron jobs should show next run time

---

### Option 2: Manual Deployment

If you need to deploy workers separately:

#### Deploy Background Worker

1. Go to Render Dashboard → **New** → **Background Worker**
2. Configure:
   - **Name**: neurovest-data-worker
   - **Runtime**: Python 3
   - **Branch**: claude/assess-codebase-AqOfb
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python3 worker_data_scheduler.py`
   - **Plan**: Free
3. Add Environment Variables:
   - `PYTHON_VERSION`: 3.11.14
   - `UPDATE_INTERVAL`: 60
   - `DATABASE_PATH`: data/market_data.db
4. Click **Create Background Worker**

#### Deploy Cron Jobs

1. Go to Render Dashboard → **New** → **Cron Job**
2. Configure for Daily Predictions:
   - **Name**: neurovest-daily-predictions
   - **Runtime**: Python 3
   - **Branch**: claude/assess-codebase-AqOfb
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Command**: `python3 cron_daily_predictions.py`
   - **Schedule**: `30 16 * * 1-5` (4:30 PM EST, Mon-Fri)
   - **Plan**: Free
3. Click **Create Cron Job**
4. Repeat for Weekly Retraining with schedule `0 2 * * 0`

---

## 🧪 Local Testing

Test workers locally before deploying:

### Test Data Worker
```bash
# Run data worker (Ctrl+C to stop)
python3 worker_data_scheduler.py
```

**Expected output:**
```
======================================================================
🚀 NEUROVEST DATA WORKER - STARTING
======================================================================
  Platform: Render
  Started: 2024-12-28 10:30:00
  Python: 3.11.14
======================================================================

📊 Registering stock/commodity assets...
  ✓ SPY (yfinance, 60min)
  ✓ QQQ (yfinance, 60min)
  ...

₿ Registering crypto assets...
  ✓ BTC_USDT (CCXT, 15min)
  ✓ ETH_USDT (CCXT, 15min)
  ...

✅ Worker setup complete!

🔄 Running initial data update...
======================================================================
✅ WORKER RUNNING - Updates every 60 minutes
======================================================================
  Press Ctrl+C to stop

[10:30:15] Assets:  27 | Records:   45,678 | Cache:  92% | DB: 12.5MB
```

---

### Test Daily Predictions Cron
```bash
# Run daily predictions once
python3 cron_daily_predictions.py
```

**Expected output:**
```
======================================================================
🔮 NEUROVEST DAILY PREDICTIONS - CRON JOB
======================================================================
[2024-12-28 16:30:00] Starting daily prediction pipeline
======================================================================

[2024-12-28 16:30:00] STEP 1/3: Updating SPY market data
[2024-12-28 16:30:00] ▶️  Download latest SPY data
[2024-12-28 16:30:03] ✅ Download latest SPY data - SUCCESS

[2024-12-28 16:30:03] STEP 2/3: Updating all asset data
[2024-12-28 16:30:03] ▶️  Update all market data
[2024-12-28 16:30:45] ✅ Update all market data - SUCCESS

[2024-12-28 16:30:45] STEP 3/3: Generating ensemble predictions
[2024-12-28 16:30:45] ▶️  Generate ensemble predictions
[2024-12-28 16:32:10] ✅ Generate ensemble predictions - SUCCESS

======================================================================
[2024-12-28 16:32:10] ✅ DAILY PREDICTION PIPELINE COMPLETE
======================================================================
[2024-12-28 16:32:10] Next run: Tomorrow at 16:30 EST
```

---

### Test Weekly Retraining
```bash
# Run weekly retraining (takes 30-60 min)
python3 cron_weekly_retrain.py
```

---

## 📊 Monitoring

### Check Worker Status

**Via Render Dashboard:**
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click on `neurovest-data-worker`
3. View logs in real-time
4. Check resource usage (CPU, memory)

**Via Logs:**
```bash
# Worker logs show real-time status
[10:30:15] Assets:  27 | Records:   45,678 | Cache:  92% | DB: 12.5MB
```

### Check Cron Job Status

**Via Render Dashboard:**
1. Click on cron job name
2. View **Last Run** and **Next Run** times
3. Check **Run History** for success/failure
4. View logs for each execution

---

## 🔧 Configuration

### Adjust Update Frequency

Change worker update interval in `render.yaml`:
```yaml
envVars:
  - key: UPDATE_INTERVAL
    value: "30"  # Update every 30 minutes instead of 60
```

### Adjust Cron Schedules

**Cron expression format:** `minute hour day month day-of-week`

Examples:
```yaml
# Every 6 hours
schedule: "0 */6 * * *"

# Weekdays at noon
schedule: "0 12 * * 1-5"

# First day of month at midnight
schedule: "0 0 1 * *"

# Every Sunday at 3 AM
schedule: "0 3 * * 0"
```

**Helpful tool:** [Crontab Guru](https://crontab.guru/)

---

## 🆚 Render vs Railway Workers

Both platforms support similar worker types:

| Feature | Render | Railway |
|---------|--------|---------|
| **Background Workers** | ✅ Worker type | ✅ Service type |
| **Cron Jobs** | ✅ Native support | ✅ Native support |
| **Free Tier** | ✅ 750 hours/month | ✅ 500 hours/month |
| **Auto Deploy** | ✅ Blueprint (render.yaml) | ✅ Config (railway.json) |
| **Logs** | ✅ Real-time | ✅ Real-time |
| **Environment Variables** | ✅ Per service | ✅ Per service |
| **Persistent Storage** | ⚠️ Disk (paid) | ⚠️ Volume (paid) |

**Key Differences:**
- Render has more free hours per month (750 vs 500)
- Railway has easier persistent storage setup
- Render Blueprint = Railway railway.json
- Both support Python workers equally well

---

## 📁 Worker File Structure

```
NeuroVest/
├── render.yaml                    # Blueprint with all services
├── worker_data_scheduler.py       # Background worker (24/7)
├── cron_daily_predictions.py      # Daily cron job
├── cron_weekly_retrain.py         # Weekly cron job
├── run_daily_pipeline.py          # Original pipeline (reference)
├── core/
│   └── scheduler.py               # Scheduler utilities
├── update_data.py                 # Data update utilities
├── requirements.txt               # Dependencies
└── data/
    └── market_data.db            # SQLite database
```

---

## ⚠️ Important Notes

### Free Tier Limits

**Render Free Plan:**
- 750 hours/month total across all services
- With 5 services (2 web + 1 worker + 2 cron):
  - Web services: ~24 hrs/day = 720 hrs/month each
  - Worker: ~24 hrs/day = 720 hrs/month
  - Cron jobs: Minimal hours (only run time)
- **You may exceed free tier!** Monitor usage.

**Recommendations:**
- Start with worker only, add cron jobs later
- Use cron jobs for heavy tasks (they only run when scheduled)
- Consider upgrading to paid plan if running 24/7 workers

### Database Persistence

**⚠️ CRITICAL:** Render's free tier has **ephemeral storage**!
- Data is lost when service restarts
- Solution 1: Use Render Disk (paid, $1/GB/month)
- Solution 2: Use external database (PostgreSQL, MongoDB)
- Solution 3: Use cloud storage (S3, Google Cloud Storage)

**For production:**
```yaml
# Add disk to worker (paid)
- type: worker
  name: neurovest-data-worker
  disk:
    name: neurovest-data
    mountPath: /home/user/NeuroVest/data
    sizeGB: 10  # $10/month
```

### Error Handling

All workers include:
- ✅ Graceful shutdown on SIGTERM
- ✅ Error logging with timestamps
- ✅ Fallback to synthetic data if APIs fail
- ✅ Retry logic for transient failures
- ✅ Status monitoring

---

## 🚨 Troubleshooting

### Worker Won't Start

1. Check logs in Render Dashboard
2. Verify all dependencies in `requirements.txt`
3. Test locally: `python3 worker_data_scheduler.py`
4. Check environment variables are set

### Cron Job Fails

1. Check cron expression is valid
2. Verify script has execute permissions
3. Test locally: `python3 cron_daily_predictions.py`
4. Check timeout settings (default 5 min, increase if needed)

### Data Not Updating

1. Check worker logs for errors
2. Verify API keys if needed (yfinance, CCXT)
3. Check network connectivity
4. Verify database permissions

### Out of Memory

1. Reduce UPDATE_INTERVAL (longer intervals)
2. Process fewer assets at once
3. Upgrade to paid plan for more memory

---

## 📚 Additional Resources

- [Render Workers Documentation](https://render.com/docs/background-workers)
- [Render Cron Jobs Documentation](https://render.com/docs/cronjobs)
- [Render Blueprint Spec](https://render.com/docs/blueprint-spec)
- [Railway Workers (Alternative)](https://docs.railway.app/develop/services)

---

## 🎉 Quick Start Summary

**To deploy everything:**
```bash
# 1. Commit worker files
git add worker_data_scheduler.py cron_daily_predictions.py cron_weekly_retrain.py render.yaml
git commit -m "Add Render workers and cron jobs"
git push -u origin claude/assess-codebase-AqOfb

# 2. Render automatically deploys all services from render.yaml

# 3. Monitor in Render Dashboard
# - Check all 5 services are "Deploy live"
# - View worker logs
# - Check cron next run times
```

**To test locally first:**
```bash
# Test data worker (Ctrl+C to stop)
python3 worker_data_scheduler.py

# Test daily predictions
python3 cron_daily_predictions.py

# Test weekly retraining (long)
python3 cron_weekly_retrain.py
```

---

**Last Updated:** December 28, 2024
