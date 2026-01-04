-- Add user table for authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    username VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add user_id to asset_metadata for ownership
ALTER TABLE asset_metadata
ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id),
ADD COLUMN IF NOT EXISTS is_custom BOOLEAN DEFAULT FALSE;

-- Add user_id to price_data for isolation
ALTER TABLE price_data
ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id);

-- Index for faster user-specific queries
CREATE INDEX IF NOT EXISTS idx_price_data_user_ticker ON price_data(user_id, ticker);
CREATE INDEX IF NOT EXISTS idx_asset_metadata_user ON asset_metadata(user_id, ticker);

-- Create view for user-specific assets
CREATE OR REPLACE VIEW user_assets AS
SELECT
    am.ticker,
    am.asset_type,
    am.frequency,
    am.user_id,
    am.is_custom,
    COUNT(pd.id) as record_count,
    MAX(pd.timestamp) as last_update
FROM asset_metadata am
LEFT JOIN price_data pd ON am.ticker = pd.ticker AND (am.user_id = pd.user_id OR am.user_id IS NULL)
GROUP BY am.ticker, am.asset_type, am.frequency, am.user_id, am.is_custom;

COMMENT ON TABLE users IS 'User accounts for custom asset uploads';
COMMENT ON COLUMN asset_metadata.user_id IS 'NULL = public asset, NOT NULL = user-specific custom asset';
COMMENT ON COLUMN asset_metadata.is_custom IS 'TRUE = custom uploaded asset, FALSE = system asset';
