#!/bin/bash
# run_migrations.sh - Simple migration runner for Railway

echo "🔄 Running NeuroVest database migrations..."
echo ""

# Migration 1: User tiers
echo "Migration 1/2: Adding user tiers column..."
cat migrations/add_user_tiers.sql | railway run bash -c 'psql $DATABASE_URL'

if [ $? -eq 0 ]; then
    echo "✅ User tiers migration complete"
else
    echo "⚠️  User tiers migration failed (may already exist)"
fi

echo ""

# Migration 2: Request logging
echo "Migration 2/2: Adding request logging table..."
cat migrations/add_request_logging.sql | railway run bash -c 'psql $DATABASE_URL'

if [ $? -eq 0 ]; then
    echo "✅ Request logging migration complete"
else
    echo "⚠️  Request logging migration failed (may already exist)"
fi

echo ""
echo "🎉 Migrations complete!"
echo ""
echo "Verify with:"
echo "  railway run bash -c 'psql \$DATABASE_URL -c \"\\dt\"'"
