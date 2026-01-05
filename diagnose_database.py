#!/usr/bin/env python3
"""
Database Connection Diagnostic Tool

Run this on Railway to diagnose the PostgreSQL connection issue:
    railway run python diagnose_database.py
"""

import os
import sys
from urllib.parse import urlparse

print("="*70)
print("DATABASE CONNECTION DIAGNOSTICS")
print("="*70)

# Check DATABASE_URL environment variable
database_url = os.environ.get("DATABASE_URL", "")

if not database_url:
    print("\n❌ PROBLEM FOUND: DATABASE_URL environment variable is NOT SET")
    print("\nFIX:")
    print("  1. Go to Railway Dashboard → neurovest-api-production")
    print("  2. Click 'Variables' tab")
    print("  3. Add new variable:")
    print("     Name:  DATABASE_URL")
    print("     Value: ${{Postgres.DATABASE_URL}}")
    print("  4. Redeploy the service")
    sys.exit(1)

print(f"\n✓ DATABASE_URL is set")
print(f"  Value: {database_url[:50]}...")

# Parse DATABASE_URL
if database_url.startswith("postgres"):
    u = urlparse(database_url)
    print(f"\n✓ PostgreSQL connection string detected")
    print(f"  Scheme:   {u.scheme}")
    print(f"  Host:     {u.hostname}")
    print(f"  Port:     {u.port or 5432}")
    print(f"  Database: {u.path.lstrip('/')}")
    print(f"  User:     {u.username}")
else:
    print(f"\n⚠️  WARNING: DATABASE_URL doesn't start with 'postgres'")
    print(f"  Actual: {database_url[:30]}...")

# Test connection
print("\n" + "="*70)
print("TESTING CONNECTION")
print("="*70)

try:
    from sqlalchemy import create_engine, text

    print("\nCreating engine...")
    engine = create_engine(database_url, pool_pre_ping=True, connect_args={'connect_timeout': 5})

    print("Attempting connection...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"\n✅ SUCCESS! Connected to PostgreSQL")
        print(f"   Version: {version}")

        # Check tables
        print("\nChecking tables...")
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]

        if tables:
            print(f"✓ Found {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
        else:
            print("⚠️  No tables found - migrations may not have run")

        # Check request_logs table specifically
        if 'request_logs' not in tables:
            print("\n⚠️  'request_logs' table missing!")
            print("   Run migration: railway run psql $DATABASE_URL -f migrations/add_request_logging.sql")
        else:
            # Count rows in request_logs
            result = conn.execute(text("SELECT COUNT(*) FROM request_logs"))
            count = result.scalar()
            print(f"\n✓ request_logs table exists with {count} rows")

        # Check price_data
        if 'price_data' in tables:
            result = conn.execute(text("SELECT COUNT(*) FROM price_data"))
            count = result.scalar()
            print(f"✓ price_data table exists with {count:,} rows")

    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED - Database is working correctly")
    print("="*70)

except Exception as e:
    print(f"\n❌ CONNECTION FAILED!")
    print(f"   Error: {e}")
    print("\n" + "="*70)
    print("TROUBLESHOOTING STEPS:")
    print("="*70)
    print("\n1. Check PostgreSQL service is running:")
    print("   - Go to Railway Dashboard")
    print("   - Verify Postgres service shows 'Active' (green)")
    print("\n2. Verify DATABASE_URL variable:")
    print("   - neurovest-api-production → Variables")
    print("   - Should be: ${{Postgres.DATABASE_URL}}")
    print("\n3. Check service networking:")
    print("   - Both services should be in same project")
    print("   - Railway auto-connects services in same project")
    print("\n4. Check logs:")
    print("   - railway logs --service neurovest-api-production")
    print("\n" + "="*70)
    sys.exit(1)
