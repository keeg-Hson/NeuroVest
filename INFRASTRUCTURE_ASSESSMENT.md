# NeuroVest Infrastructure Assessment & Fixes
**Date:** 2026-01-16
**Issue:** Crypto data disappearing repeatedly despite successful reloads

---

## 🔴 CRITICAL ISSUES FOUND

### 1. **INFRASTRUCTURE MISMATCH - Crypto Asset Lists Not Synchronized**

**Problem:** The crypto downloader is NOT wired to the same infrastructure!

**Evidence:**
- `reload_crypto_max_history.py` tracks **10 cryptos**:
  - BTC, ETH, SOL, **BNB**, XRP, ADA, DOGE, AVAX, **MATIC**, **LINK**

- `worker_data_scheduler.py` tracks **8 cryptos**:
  - BTC, ETH, SOL, XRP, ADA, DOGE, **DOT**, AVAX

**Missing from worker:** BNB_USDT, MATIC_USDT, LINK_USDT
**Missing from reload:** DOT_USDT

**Impact:** When reload script loads all 10 cryptos but worker only maintains 8, the 3 missing cryptos (BNB, MATIC, LINK) have no automated updates and can become stale or disappear.

---

### 2. **DESTRUCTIVE DATA CLEARING - Race Condition Window**

**Problem:** `reload_crypto_max_history.py` DELETES ALL crypto data before reload

**Evidence:** Lines 116-125:
```python
# Delete old crypto data
with dm.engine.begin() as conn:
    for config in crypto_configs:
        ticker = config[1]
        conn.execute(text("DELETE FROM price_data WHERE ticker = :ticker"), {"ticker": ticker})
        conn.execute(text("""
            UPDATE asset_metadata
            SET last_update = NULL, last_timestamp = NULL, total_records = 0
            WHERE ticker = :ticker
        """), {"ticker": ticker})
```

**Impact:**
- Creates a window where ALL crypto data is deleted
- If reload fails or is interrupted, crypto data is LOST
- If background worker runs during this window, race condition occurs
- If any crypto fails to reload, it stays deleted forever

---

### 3. **BACKGROUND WORKER INTERFERENCE**

**Problem:** DataWorker2 service runs every 60 minutes, causing conflicts with manual operations

**Evidence:**
- Background worker calls `update_from_source()` every 60 minutes
- Manual reload script calls DELETE then reload
- If worker runs during manual reload → race condition
- If manual reload fails, worker can't recover (missing assets!)

---

## ✅ WHAT'S WORKING CORRECTLY

### 1. **Incremental Updates Are Safe**
- `update_from_source()` in `data_manager_postgres.py` (lines 594-624)
- Only adds new data after last timestamp
- Does NOT delete anything
- This is the correct approach!

### 2. **Automation Pipeline Timing**
- Daily predictions: 4:30 PM EST ✅
- Weekly retraining: Sunday 2:00 AM ✅
- Hourly data updates: Every 60 minutes ✅
- All scripts fixed to use `predict_all_assets.py` ✅

### 3. **Database Integrity**
- PostgreSQL ON CONFLICT DO NOTHING prevents duplicates ✅
- Transactions properly handled ✅
- No SQL injection vulnerabilities ✅

---

## 🔧 ANSWERS TO YOUR QUESTIONS

### Q1: "Is the crypto downloader wired to the same infrastructure?"
**A: NO!** Asset lists are out of sync:
- Reload script: 10 cryptos (includes BNB, MATIC, LINK)
- Worker: 8 cryptos (includes DOT instead)
- **3 cryptos have no automated maintenance!**

### Q2: "EVERYTHING UPDATES TOGETHER, CORRECT?"
**A: NO!** Currently there's desynchronization:
- Worker updates every 60 minutes (incremental)
- Manual reload deletes everything first (destructive)
- Different asset lists mean some assets never update
- Race conditions between manual and automated updates

### Q3: "Are all timed parts of this pipeline working appropriately?"
**A: Partially:**
- ✅ Timing is correct (daily predictions, weekly training, hourly updates)
- ✅ All scripts use correct prediction script (predict_all_assets.py)
- ❌ Asset lists not synchronized between worker and reload
- ❌ Reload script uses destructive DELETE instead of incremental update
- ❌ No coordination between manual operations and background worker

---

## 🛠️ FIXES APPLIED

### Fix 1: Synchronize Crypto Asset Lists
**Action:** Update worker to track same 10 cryptos as reload script

### Fix 2: Make Reload Script Incremental (Optional Safe Mode)
**Action:** Add `--safe` mode that doesn't delete data first, uses incremental updates

### Fix 3: Add Worker Status Check
**Action:** Create diagnostic to check if worker is running before manual operations

---

## 📊 INFRASTRUCTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                    Railway PostgreSQL                        │
│                  (Shared by all services)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
    ┌───────────▼──────────┐    ┌───────────▼──────────┐
    │   DataWorker2        │    │  Manual Operations   │
    │   (Background)       │    │  (Your Commands)     │
    ├──────────────────────┤    ├──────────────────────┤
    │ • Runs 24/7          │    │ • bootstrap_all.sh   │
    │ • Every 60 minutes   │    │ • reload_crypto.py   │
    │ • Incremental only   │    │ • update_data.py     │
    │ • 8 cryptos (WRONG!) │◄──►│ • 10 cryptos         │
    └──────────────────────┘    └──────────────────────┘
           │                              │
           └──────────┬───────────────────┘
                      │ RACE CONDITION!
                      │ Conflicting asset lists
                      │ DELETE vs UPDATE
                      ▼
           ⚠️  CRYPTO DATA DISAPPEARS
```

---

## 📝 RECOMMENDATIONS

### Immediate (Applied Now):
1. ✅ Sync worker crypto list to match reload script (10 cryptos)
2. ✅ Document the infrastructure mismatch
3. ✅ Create this assessment document

### Short-term (Recommended):
1. Remove DELETE logic from reload script, use incremental only
2. Add mutex/lock to prevent worker conflicts during manual ops
3. Add health check script to verify data consistency

### Long-term (Best Practice):
1. Single source of truth for asset list (config file)
2. Pause worker service during bootstrap/manual reloads
3. Add alerting for missing predictions
4. Implement automatic recovery for failed crypto loads

---

## 🎯 ROOT CAUSE SUMMARY

**Why crypto keeps disappearing:**

1. **Infrastructure Mismatch:** Reload script manages 10 cryptos, worker only manages 8
2. **Destructive Reload:** DELETE all crypto data creates unrecoverable failure window
3. **No Recovery Mechanism:** If reload fails, worker can't restore missing cryptos (BNB, MATIC, LINK)
4. **Race Conditions:** Background worker interferes with manual operations

**The fix:** Synchronize asset lists and use incremental updates everywhere.
