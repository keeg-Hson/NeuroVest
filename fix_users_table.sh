#!/bin/bash
# Fix users table to allow NULL email

python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')

from core.data_manager_postgres import DataManager
from sqlalchemy import text

dm = DataManager()

if dm.backend != 'postgresql':
    print("Error: PostgreSQL required")
    sys.exit(1)

with dm.engine.begin() as conn:
    # Drop the table and recreate with correct schema
    conn.execute(text("DROP TABLE IF EXISTS request_logs CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))

    # Create users table with email nullable
    conn.execute(text("""
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE,
            api_key VARCHAR(255) UNIQUE NOT NULL,
            tier VARCHAR(20) DEFAULT 'free',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    # Recreate request_logs table
    conn.execute(text("""
        CREATE TABLE request_logs (
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
        )
    """))

    # Create indexes
    conn.execute(text("CREATE INDEX idx_request_logs_created_at ON request_logs(created_at)"))
    conn.execute(text("CREATE INDEX idx_request_logs_endpoint ON request_logs(endpoint)"))
    conn.execute(text("CREATE INDEX idx_request_logs_user_id ON request_logs(user_id)"))

print("✅ Tables fixed! Email is now nullable.")
PYTHON_SCRIPT
