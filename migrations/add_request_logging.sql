-- Add request logging table for API analytics
-- Migration: add_request_logging.sql

-- Create request_logs table
CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms FLOAT NOT NULL,
    ip_address VARCHAR(50),
    user_agent TEXT,
    tier VARCHAR(20),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast analytics queries
CREATE INDEX IF NOT EXISTS idx_request_logs_user_id ON request_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_status_code ON request_logs(status_code);
CREATE INDEX IF NOT EXISTS idx_request_logs_tier ON request_logs(tier);

-- Composite index for common analytics queries
CREATE INDEX IF NOT EXISTS idx_request_logs_user_date ON request_logs(user_id, created_at);

-- View for quick analytics
CREATE OR REPLACE VIEW request_stats_daily AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as total_requests,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(response_time_ms) as avg_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
    COUNT(CASE WHEN status_code = 200 THEN 1 END) as success_count,
    tier,
    endpoint
FROM request_logs
GROUP BY DATE(created_at), tier, endpoint
ORDER BY date DESC, total_requests DESC;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Request logging migration completed successfully';
    RAISE NOTICE 'Table: request_logs created';
    RAISE NOTICE 'View: request_stats_daily created';
    RAISE NOTICE '6 indexes created for optimal performance';
END $$;
