-- Add tier column to users table for premium subscription levels
-- Migration: add_user_tiers.sql

-- Add tier column with default 'free'
ALTER TABLE users
ADD COLUMN IF NOT EXISTS tier VARCHAR(20) DEFAULT 'free';

-- Add check constraint for valid tiers
ALTER TABLE users
DROP CONSTRAINT IF EXISTS users_tier_check;

ALTER TABLE users
ADD CONSTRAINT users_tier_check
CHECK (tier IN ('free', 'individual', 'pro', 'enterprise'));

-- Update existing users to 'free' tier if NULL
UPDATE users
SET tier = 'free'
WHERE tier IS NULL;

-- Create index for faster tier lookups
CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier);

-- Display current tier distribution
SELECT
    tier,
    COUNT(*) as user_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM users
GROUP BY tier
ORDER BY user_count DESC;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'User tiers migration completed successfully';
    RAISE NOTICE 'Available tiers: free, individual, pro, enterprise';
END $$;
