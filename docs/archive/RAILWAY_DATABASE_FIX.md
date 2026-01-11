# Railway Database Connection Fix

## Problem

Your API is **falling back to SQLite** instead of using PostgreSQL, causing:
- Health check shows `assets_count: 0`
- Analytics endpoints fail with "Connection refused"
- Database appears "connected" but is actually empty SQLite

## Root Cause

The `DATABASE_URL` environment variable is **not configured** on your Railway API service, so it can't connect to PostgreSQL.

## Fix (5 minutes)

### Step 1: Check PostgreSQL Service

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Open your project
3. Verify **Postgres service** shows status: **Active** (green)
   - If not active, start it first

### Step 2: Add DATABASE_URL to API Service

1. Click on your **neurovest-api-production** service
2. Go to **Variables** tab
3. Click **+ New Variable**
4. Add:
   ```
   Name:  DATABASE_URL
   Value: ${{Postgres.DATABASE_URL}}
   ```
5. Click **Add**

### Step 3: Redeploy

Railway will automatically redeploy when you add the variable.

**OR** manually trigger:
```bash
railway up
```

### Step 4: Verify Fix

After deployment completes (~3 minutes):

```bash
# Test health endpoint
curl https://neurovest-api-production-f8dc.up.railway.app/health

# Should now show:
# {
#   "status": "healthy",
#   "database": "connected",
#   "assets_count": <number>,  # NOT 0!
#   "last_prediction": "2026-01-XX"
# }

# Test analytics
curl https://neurovest-api-production-f8dc.up.railway.app/api/analytics/dashboard?days=1

# Should return analytics data (not connection error)
```

## Alternative: Run Diagnostic Tool

If you have Railway CLI installed:

```bash
# Upload and run diagnostic
railway run python diagnose_database.py
```

This will:
- Check if DATABASE_URL is set
- Test PostgreSQL connection
- Verify tables exist
- Show detailed error messages

## How Railway References Work

Railway uses **service references** to auto-inject variables:

- `${{Postgres.DATABASE_URL}}` - Automatically resolves to the Postgres service's connection string
- Railway handles the internal networking (`.railway.internal` domains)
- Both services must be in the **same project**

## Expected Log Output After Fix

When DATABASE_URL is correctly configured, you'll see in logs:

```
======================================================================
🔍 DATABASE CONNECTION DIAGNOSTICS
======================================================================
[DB] Backend: PostgreSQL
[DB] Scheme: postgresql
[DB] Host: postgres.railway.internal
[DB] Port: 5432
[DB] Database: railway
[DB] User: postgres
======================================================================
[DB] Current price_data rows: 1,234
```

## Still Having Issues?

### Check Railway Logs

```bash
railway logs --service neurovest-api-production | grep -i "database\|postgres"
```

Look for:
- "PostgreSQL initialization failed" - Connection problem
- "Falling back to SQLite" - DATABASE_URL not set or wrong

### Verify Service Networking

1. Dashboard → Project Settings
2. Both services should show same **Project ID**
3. If different projects, move them to same project

### Check PostgreSQL Credentials

```bash
# Connect to Postgres directly
railway run bash -c 'psql $DATABASE_URL -c "\\dt"'
```

Should list tables:
- `users`
- `predictions`
- `price_data`
- `request_logs` (if migrations ran)

## Next Steps After Fix

Once DATABASE_URL is working:

1. **Run migrations**:
   ```bash
   railway run bash -c 'psql $DATABASE_URL -f migrations/add_user_tiers.sql'
   railway run bash -c 'psql $DATABASE_URL -f migrations/add_request_logging.sql'
   ```

2. **Verify analytics works**:
   ```bash
   curl https://your-api.up.railway.app/api/analytics/usage?days=7
   ```

3. **Test WebSocket**:
   ```bash
   wscat -c "wss://your-api.up.railway.app/ws/predictions?api_key=YOUR_KEY"
   ```

## Summary

The issue is **not a code problem** - it's a Railway configuration issue. Once you add the DATABASE_URL environment variable to your API service, everything should work correctly.

**Quick Fix:**
1. Railway Dashboard → neurovest-api-production → Variables
2. Add `DATABASE_URL = ${{Postgres.DATABASE_URL}}`
3. Wait for redeploy
4. Test endpoints

That's it! 🚀
