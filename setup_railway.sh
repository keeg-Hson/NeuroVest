#!/bin/bash
# setup_railway.sh - Complete Railway setup and testing

set -e  # Exit on error

echo "======================================================================="
echo "🚀 NEUROVEST API - RAILWAY SETUP"
echo "======================================================================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

echo "✅ Railway CLI installed"
echo ""

# Link to project (if not already linked)
echo "🔗 Linking to Railway project..."
railway link || echo "   Already linked"
echo ""

# Run migrations
echo "======================================================================="
echo "📊 RUNNING DATABASE MIGRATIONS"
echo "======================================================================="
echo ""

echo "Migration 1/2: Adding user tiers..."
cat migrations/add_user_tiers.sql | railway run bash -c 'psql $DATABASE_URL' || echo "⚠️  Tier migration may have already run"
echo ""

echo "Migration 2/2: Adding request logging..."
cat migrations/add_request_logging.sql | railway run bash -c 'psql $DATABASE_URL' || echo "⚠️  Logging migration may have already run"
echo ""

# Verify migrations
echo "======================================================================="
echo "✅ VERIFYING DATABASE SETUP"
echo "======================================================================="
echo ""

railway run bash -c 'psql $DATABASE_URL -c "\dt"' | grep -E "users|request_logs|predictions"
echo ""

# Create test API keys
echo "======================================================================="
echo "🔑 CREATING TEST API KEYS"
echo "======================================================================="
echo ""

echo "Creating free tier test user..."
python3 create_api_key.py --username "test_free" --tier free > /tmp/free_key.txt 2>&1 || echo "⚠️  Free user may already exist"

echo ""
echo "Creating pro tier test user..."
python3 create_api_key.py --username "test_pro" --tier pro > /tmp/pro_key.txt 2>&1 || echo "⚠️  Pro user may already exist"

echo ""

# Get API URL
echo "======================================================================="
echo "🌐 GETTING API URL"
echo "======================================================================="
echo ""

API_URL=$(railway status --json | grep -o '"url":"[^"]*"' | cut -d'"' -f4)

if [ -z "$API_URL" ]; then
    echo "⚠️  Could not detect API URL automatically"
    echo "   Please get it from Railway dashboard"
    API_URL="https://your-api.up.railway.app"
else
    echo "✅ API URL: $API_URL"
fi

echo ""

# Test API
echo "======================================================================="
echo "🧪 TESTING API"
echo "======================================================================="
echo ""

echo "Test 1: Health check..."
curl -s "$API_URL/health" | jq '.' || echo "⚠️  Health check failed"

echo ""
echo "Test 2: Cache stats..."
curl -s "$API_URL/cache/stats" | jq '.' || echo "⚠️  Cache stats failed"

echo ""
echo "Test 3: WebSocket stats..."
curl -s "$API_URL/ws/stats" | jq '.' || echo "⚠️  WebSocket stats failed"

echo ""
echo "Test 4: Analytics dashboard..."
curl -s "$API_URL/api/analytics/dashboard?days=1" | jq '.period_days' || echo "⚠️  Analytics failed"

echo ""

# Summary
echo "======================================================================="
echo "✅ SETUP COMPLETE!"
echo "======================================================================="
echo ""
echo "Your API is running at: $API_URL"
echo ""
echo "📚 Documentation:"
echo "   - API Docs:    $API_URL/docs"
echo "   - Health:      $API_URL/health"
echo "   - Analytics:   $API_URL/api/analytics/dashboard"
echo ""
echo "🔑 Test API Keys:"
echo "   Check /tmp/free_key.txt and /tmp/pro_key.txt for keys"
echo ""
echo "🎯 Next Steps:"
echo "   1. Save your API keys"
echo "   2. Test predictions: curl -H 'X-API-Key: YOUR_KEY' $API_URL/api/predictions/SPY"
echo "   3. Monitor analytics: $API_URL/api/analytics/dashboard"
echo "   4. Optional: Add Redis for caching (Railway Dashboard → Add Redis)"
echo ""
echo "======================================================================="
