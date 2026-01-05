#!/usr/bin/env python3
"""
CLI Tool for Creating NeuroVest API Keys

Usage:
    python3 create_api_key.py --username "customer_name"
    python3 create_api_key.py --username "premium_user" --tier pro
    python3 create_api_key.py --list  # List all users

Tiers:
    - free: 10 requests/minute (default)
    - individual: 60 requests/minute
    - pro: 300 requests/minute
    - enterprise: 10,000 requests/minute
"""

import argparse
import sys
from core.data_manager_postgres import DataManager
from auth_middleware import AuthManager
from sqlalchemy import text
from tabulate import tabulate

def create_user(username: str, tier: str = 'free'):
    """Create a new API user"""
    valid_tiers = ['free', 'individual', 'pro', 'enterprise']

    if tier not in valid_tiers:
        print(f"❌ Invalid tier: {tier}")
        print(f"   Valid tiers: {', '.join(valid_tiers)}")
        return False

    try:
        print(f"\n{'=' * 70}")
        print(f"Creating API Key for: {username}")
        print(f"Tier: {tier}")
        print(f"{'=' * 70}\n")

        # Create user with AuthManager
        user_data = AuthManager.create_user(username)
        user_id = user_data['user_id']
        api_key = user_data['api_key']

        # Update tier if not 'free'
        if tier != 'free':
            dm = DataManager()
            with dm.engine.begin() as conn:
                conn.execute(
                    text("UPDATE users SET tier = :tier WHERE id = :user_id"),
                    {"tier": tier, "user_id": user_id}
                )
            dm.close()

        # Display results
        print("✅ API Key Created Successfully!\n")
        print(f"User ID:  {user_id}")
        print(f"Username: {username}")
        print(f"Tier:     {tier}")
        print(f"\n{'─' * 70}")
        print("API Key (save this - it won't be shown again!):")
        print(f"{'─' * 70}")
        print(f"\n{api_key}\n")
        print(f"{'─' * 70}\n")

        # Rate limit info
        rate_limits = {
            'free': '10 requests/minute',
            'individual': '60 requests/minute',
            'pro': '300 requests/minute',
            'enterprise': '10,000 requests/minute'
        }
        print(f"Rate Limit: {rate_limits[tier]}")

        print("\n📝 Next Steps:")
        print("   1. Save the API key in a secure location")
        print("   2. Test the key:")
        print(f"      curl -H 'X-API-Key: {api_key}' https://your-api.railway.app/health")
        print("   3. Share with customer (send securely!)")
        print(f"\n{'=' * 70}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error creating user: {e}\n")
        return False


def list_users():
    """List all API users"""
    try:
        dm = DataManager()

        with dm.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        id,
                        username,
                        tier,
                        LEFT(api_key, 16) || '...' as api_key_preview,
                        created_at
                    FROM users
                    ORDER BY id DESC
                    LIMIT 50
                """)
            )
            users = result.fetchall()

        dm.close()

        if not users:
            print("\n📋 No users found.\n")
            return

        print(f"\n{'=' * 100}")
        print("API USERS")
        print(f"{'=' * 100}\n")

        # Format as table
        headers = ['ID', 'Username', 'Tier', 'API Key Preview', 'Created']
        table_data = [
            [
                user[0],  # id
                user[1],  # username
                user[2] or 'free',  # tier
                user[3],  # api_key_preview
                str(user[4])[:19] if user[4] else 'N/A'  # created_at
            ]
            for user in users
        ]

        print(tabulate(table_data, headers=headers, tablefmt='grid'))

        # Summary stats
        with dm.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        tier,
                        COUNT(*) as count
                    FROM users
                    GROUP BY tier
                    ORDER BY count DESC
                """)
            )
            stats = result.fetchall()

        dm.close()

        print(f"\n{'─' * 100}")
        print("TIER DISTRIBUTION:")
        for tier, count in stats:
            tier_name = tier or 'free'
            print(f"   {tier_name:12s}: {count:3d} users")
        print(f"{'─' * 100}\n")

    except Exception as e:
        print(f"\n❌ Error listing users: {e}\n")


def get_user_details(username: str):
    """Get details for a specific user"""
    try:
        dm = DataManager()

        with dm.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        id,
                        username,
                        tier,
                        api_key,
                        created_at
                    FROM users
                    WHERE username = :username
                """),
                {"username": username}
            )
            user = result.fetchone()

        dm.close()

        if not user:
            print(f"\n❌ User not found: {username}\n")
            return False

        print(f"\n{'=' * 70}")
        print(f"USER DETAILS: {username}")
        print(f"{'=' * 70}\n")
        print(f"User ID:    {user[0]}")
        print(f"Username:   {user[1]}")
        print(f"Tier:       {user[2] or 'free'}")
        print(f"Created:    {user[4]}")
        print(f"\nAPI Key:")
        print(f"{'─' * 70}")
        print(f"{user[3]}")
        print(f"{'─' * 70}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error getting user details: {e}\n")
        return False


def update_user_tier(username: str, tier: str):
    """Update user's tier"""
    valid_tiers = ['free', 'individual', 'pro', 'enterprise']

    if tier not in valid_tiers:
        print(f"❌ Invalid tier: {tier}")
        print(f"   Valid tiers: {', '.join(valid_tiers)}")
        return False

    try:
        dm = DataManager()

        with dm.engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE users
                    SET tier = :tier
                    WHERE username = :username
                    RETURNING id
                """),
                {"tier": tier, "username": username}
            )
            updated = result.fetchone()

        dm.close()

        if not updated:
            print(f"\n❌ User not found: {username}\n")
            return False

        print(f"\n✅ Updated {username} to tier: {tier}\n")
        return True

    except Exception as e:
        print(f"\n❌ Error updating tier: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='NeuroVest API Key Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create free tier user
  python3 create_api_key.py --username "john_doe"

  # Create pro tier user
  python3 create_api_key.py --username "premium_customer" --tier pro

  # List all users
  python3 create_api_key.py --list

  # Get user details
  python3 create_api_key.py --get "john_doe"

  # Update user tier
  python3 create_api_key.py --update "john_doe" --tier pro

Tiers and Rate Limits:
  free:       10 requests/minute
  individual: 60 requests/minute
  pro:        300 requests/minute
  enterprise: 10,000 requests/minute
        """
    )

    parser.add_argument('--username', type=str, help='Username for new API key')
    parser.add_argument('--tier', type=str, default='free',
                       choices=['free', 'individual', 'pro', 'enterprise'],
                       help='User tier (default: free)')
    parser.add_argument('--list', action='store_true', help='List all users')
    parser.add_argument('--get', type=str, help='Get details for specific user')
    parser.add_argument('--update', type=str, help='Update tier for existing user')

    args = parser.parse_args()

    # List mode
    if args.list:
        list_users()
        return

    # Get mode
    if args.get:
        get_user_details(args.get)
        return

    # Update mode
    if args.update:
        if not args.tier:
            print("❌ --tier required for update")
            parser.print_help()
            sys.exit(1)
        update_user_tier(args.update, args.tier)
        return

    # Create mode
    if args.username:
        success = create_user(args.username, args.tier)
        sys.exit(0 if success else 1)
    else:
        print("❌ --username required to create new user\n")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
