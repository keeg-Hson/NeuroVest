# Railway Deployment Guide

## Overview

NeuroVest API requires two Railway services:
1. **PostgreSQL Database** - Stores predictions, user data, and asset metadata
2. **DataWorker2** - Trains models, generates predictions, and runs scheduler

## Critical Issue: Missing Models

The `models/` directory is in `.gitignore`, so trained model files (.pkl) are NOT deployed to Railway.

This causes the error:
```
❌ No models found. Run train_multi_asset.py first.
```

## Solution: Bootstrap Process

### First-Time Deployment

**Railway DataWorker2 Start Command:**
```bash
bash bootstrap_railway.sh
```

This script will:
1. ✅ Train all 3 models (xgboost, lightgbm, catboost) - ~15-30 min
2. ✅ Generate predictions for all 40 assets - ~5 min
3. ✅ Start the worker scheduler for ongoing updates

**Total time:** ~30-45 minutes on first run

### After Initial Bootstrap

Once models are trained, you can switch to faster ongoing operations:

**Railway DataWorker2 Start Command:**
```bash
python3 predict_all_assets.py && python3 worker_data_scheduler.py
```

This will:
- Generate fresh predictions (~5 min)
- Start scheduler for daily updates

## Required Environment Variables

### DataWorker2 Service

```bash
DATABASE_URL=postgresql://user:password@host:5432/railway
PYTHONUNBUFFERED=1
```

### PostgreSQL Service

Railway automatically provides:
```bash
PGHOST
PGPORT
PGUSER
PGPASSWORD
PGDATABASE
```

## API Service (Separate Container)

If running API in a separate Railway service:

**Start Command:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
```bash
DATABASE_URL=postgresql://user:password@host:5432/railway
PYTHONUNBUFFERED=1
```

## Troubleshooting

### Error: "No models found"

**Cause:** Models weren't trained yet or Railway container restarted

**Fix:** Run `bash bootstrap_railway.sh` to retrain models

### Error: "Database connection failed"

**Cause:** DATABASE_URL not set or incorrect

**Fix:**
1. Check PostgreSQL service is running
2. Verify DATABASE_URL environment variable
3. Ensure it starts with `postgresql://` (not `postgres://`)

### Error: "No data available for asset XXX"

**Cause:** Historical data wasn't loaded yet

**Fix:** Run `bash bootstrap_all.sh` for full data load

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Railway Services                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   PostgreSQL     │◄────────│   DataWorker2    │          │
│  │   Database       │         │                  │          │
│  │                  │         │  - Train models  │          │
│  │  - predictions   │         │  - Predict all   │          │
│  │  - users         │         │  - Scheduler     │          │
│  │  - asset_metadata│         │                  │          │
│  └────────▲─────────┘         └──────────────────┘          │
│           │                                                  │
│           │                                                  │
│  ┌────────┴─────────┐                                       │
│  │   API Server     │                                       │
│  │                  │                                       │
│  │  - FastAPI       │                                       │
│  │  - 8 endpoints   │                                       │
│  │  - Auth layer    │                                       │
│  └──────────────────┘                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Model Files

After training, these files are created in `models/`:

```
models/
├── xgboost_multi_asset.pkl      (~50 MB)
├── lightgbm_multi_asset.pkl     (~40 MB)
├── catboost_multi_asset.pkl     (~60 MB)
├── multi_asset_features.txt     (list of 108 features)
└── multi_asset_results.csv      (performance metrics)
```

**Note:** These are stored in Railway's ephemeral filesystem. They persist across restarts but may be lost if container is recreated. In that case, re-run bootstrap.

## Deployment Checklist

- [ ] PostgreSQL service running on Railway
- [ ] DATABASE_URL environment variable set on DataWorker2
- [ ] DataWorker2 start command set to `bash bootstrap_railway.sh`
- [ ] First deployment runs successfully (30-45 min)
- [ ] Check logs for "✅ RAILWAY BOOTSTRAP COMPLETE!"
- [ ] Verify predictions table has data: `SELECT COUNT(*) FROM predictions;`
- [ ] Test API endpoint: `curl https://your-api.railway.app/health`
- [ ] After initial bootstrap, optionally change start command to faster version

## Monitoring

Check Railway logs for:

✅ **Success indicators:**
```
✅ RAILWAY BOOTSTRAP COMPLETE!
Training:    Exit code 0
Predictions: Exit code 0
```

❌ **Error indicators:**
```
❌ No models found. Run train_multi_asset.py first.
❌ Model training failed (exit code 1)
❌ Database connection failed
```

## Cost Optimization

- **First deployment:** Use bootstrap (one-time cost)
- **Ongoing:** Use `predict_all_assets.py && worker_data_scheduler.py` (faster, cheaper)
- **Models:** Stored in ephemeral filesystem (no extra storage cost)
- **Database:** Consider upgrading if you hit connection limits

## Support

If you encounter issues:

1. Check Railway logs for error messages
2. Verify DATABASE_URL is correct
3. Ensure PostgreSQL service is running
4. Try manual commands: `python3 train_multi_asset.py`
5. Check this repo's issues: https://github.com/user/NeuroVest/issues
