# Railway DataWorker2 Configuration Guide

## The Problem (Why Predictions Were Stuck at 11/05)

The system has been loading data correctly, but **predictions haven't been generated since November 5th**. This is because:

1. ✅ **Data loading works** - Bootstrap Steps 0-1.5 complete successfully
2. ❌ **Predictions not running** - No daily cron job or worker generating fresh predictions
3. ❌ **Worker misconfigured** - DataWorker2 service not running the prediction scheduler

## The Solution

DataWorker2 needs to run one of these scripts:

### Option 1: Comprehensive Worker (RECOMMENDED)

Configure DataWorker2 to run:
```bash
bash start_worker.sh
```

This script:
- ✅ Checks if bootstrap is needed (first run)
- ✅ Runs bootstrap if database is empty
- ✅ Starts continuous worker with automated predictions
- ✅ Generates predictions daily at 4:30 PM EST
- ✅ Retrains models weekly on Sundays at 2 AM EST

**How to Configure in Railway:**
1. Go to DataWorker2 service settings
2. Under "Deploy" → "Start Command", set:
   ```
   bash start_worker.sh
   ```
3. Redeploy the service

---

### Option 2: One-Time Bootstrap Only

If you want DataWorker2 to only run bootstrap and exit:
```bash
bash bootstrap_all.sh
```

**Note:** This will NOT generate ongoing predictions. You'll need Option 1 or Option 3 for continuous updates.

---

### Option 3: Manual Prediction Generation

To immediately generate fresh predictions (for testing or one-time fix):
```bash
bash generate_fresh_predictions.sh
```

This:
- ✅ Checks if models exist
- ✅ Trains models if needed (first time only)
- ✅ Generates fresh predictions immediately
- ❌ Does NOT set up automated daily predictions

---

## How to Verify It's Working

### Check 1: Worker Logs
In Railway DataWorker2 logs, you should see:
```
🔄 STARTING CONTINUOUS WORKER
Worker starting...
```

### Check 2: Dashboard Status
The dashboard should show:
- 🟢 **Fresh**: if predictions <48 hours old
- 🟡 **Stale**: if predictions 2-7 days old
- 🔴 **VERY OLD**: if predictions >7 days old (like your 11/05 ones)

### Check 3: Database Check
Run this Python script to check prediction age:
```python
from core.data_manager_postgres import DataManager
dm = DataManager()
preds = dm.get_latest_predictions(limit=1)
print(f"Latest prediction: {preds.iloc[0]['prediction_date']}")
print(f"Generated at: {preds.iloc[0]['prediction_timestamp']}")
dm.close()
```

---

## Architecture Overview

### Current Setup (PostgreSQL)

```
┌─────────────────┐
│   Dashboard2    │  ← Streamlit UI (shows predictions)
│  (web service)  │
└────────┬────────┘
         │
         ├─── Reads from ───→ ┌──────────────────┐
         │                     │   PostgreSQL     │
         │                     │   (predictions)  │
┌────────┴────────┐           └──────────────────┘
│   DataWorker2   │  ← MUST run start_worker.sh
│  (worker service)│
└─────────────────┘
    │
    ├─ Hourly: Update price data
    ├─ Daily 4:30 PM: Generate predictions ← THIS WAS MISSING!
    └─ Weekly Sun 2 AM: Retrain models
```

### What Was Happening Before

```
DataWorker2 was running: ???
  ├─ Maybe just bootstrap_all.sh once?
  ├─ Maybe nothing at all?
  └─ NOT running worker_data_scheduler.py

Result: Predictions stuck at 11/05 (last manual run)
```

---

## Quick Fix for Production RIGHT NOW

### Step 1: Configure DataWorker2
1. Open Railway dashboard
2. Go to DataWorker2 service
3. Settings → Deploy → Start Command
4. Set to: `bash start_worker.sh`
5. Click "Deploy Latest"

### Step 2: Verify Deployment
Watch the logs - you should see:
```
🚀 NEUROVEST DATAWORKER2 STARTING
✅ Database: PostgreSQL configured
🔍 Checking if bootstrap is needed...
   Found XX assets - Bootstrap already done
🔄 STARTING CONTINUOUS WORKER
```

### Step 3: Wait for Next Prediction Run
- Predictions generate daily at 4:30 PM EST
- Or manually trigger with `generate_fresh_predictions.sh`

### Step 4: Check Dashboard
- Refresh dashboard after 4:30 PM
- Status should change from 🔴 VERY OLD to 🟢 Fresh

---

## Troubleshooting

### Issue: Worker crashes with "DATABASE_URL not set"
**Fix:** Ensure DATABASE_URL environment variable is set in Railway for DataWorker2

### Issue: Bootstrap runs every time worker starts
**Fix:** Check if database connection is working. Worker checks asset count to determine if bootstrap is needed.

### Issue: Predictions still showing old date
**Fix:**
1. Check worker logs for errors in `generate_predictions()`
2. Manually run `bash generate_fresh_predictions.sh`
3. Check if `predict_multi_asset_ensemble.py` is working

### Issue: "No models found" error
**Fix:** Run training first:
```bash
python3 train_multi_asset.py
```

---

## Files Reference

- `start_worker.sh` - Main worker start script (RECOMMENDED)
- `bootstrap_all.sh` - One-time setup (Steps 0-3)
- `generate_fresh_predictions.sh` - Manual prediction generation
- `worker_data_scheduler.py` - Continuous worker (called by start_worker.sh)
- `train_multi_asset.py` - Model training (Step 2)
- `predict_multi_asset_ensemble.py` - Prediction generation (Step 3)

---

## Summary

**Root Cause:** DataWorker2 wasn't running the continuous worker that generates daily predictions.

**Fix:** Configure DataWorker2 to run `start_worker.sh` which handles both bootstrap and ongoing prediction generation.

**Expected Outcome:** Fresh predictions every day at 4:30 PM EST, with 🟢 green status in dashboard.
