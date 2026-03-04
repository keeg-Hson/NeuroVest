# Recovery Notes — Branch: claude/recover-overwritten-changes-ifMUS

**Date:** 2026-03-04
**Incident:** ~40 days of local work overwritten (Jan 23 – Mar 4, 2026)

## What This Branch Contains

This branch is pinned to the **last known-good state of the repository** — commit `87436f79` from Jan 23, 2026, which includes:

- Full multi-asset prediction pipeline (40 assets — stocks + crypto)
- PostgreSQL schema with users, predictions, analytics tables
- FastAPI server (`api_server.py`, `api/`) with analytics, auth, websocket
- All ML training scripts (XGBoost, LightGBM, LSTM, CNN-LSTM, Transformer, Ensemble)
- Crypto data download pipeline
- Cron jobs for daily predictions and weekly retraining
- Bootstrap scripts for production deployment

## What Was Lost

Local uncommitted work from **Jan 23 – Mar 4, 2026** (~40 days) was overwritten when `git checkout --theirs .` ran during merge conflict resolution.

**This work was never pushed to GitHub, so it cannot be recovered from git.**

## How to Recover Local Changes (Mac-Side)

### 1. VS Code Local History (best option)
VS Code auto-saves file history. Check in the app:
- Open any `.py` file → right-click in editor → **Open Timeline**
- Or via the Source Control panel → Timeline tab

Raw files are stored at:
```
~/Library/Application Support/Code/User/History/
```

To find recently modified Python files:
```bash
find ~/Library/Application\ Support/Code/User/History -name "*.py" | xargs ls -lt | head -30
```

### 2. Time Machine (if enabled)
```bash
# Browse backup of your NeuroVest folder
open /Volumes/Time\ Machine/Backups.backupdb/
```

### 3. Compare with this branch
After cloning/pulling this recovery branch, compare any recovered files against the Jan 23 baseline to understand what changed.

## Next Steps

1. Check VS Code Local History on your Mac for any `.py` files modified after Jan 23
2. Manually re-apply any recovered work on top of this branch
3. **Commit and push early and often** going forward
