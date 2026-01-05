"""
Request Logging Middleware for NeuroVest API

Logs all API requests to database for analytics:
- User ID and tier
- Endpoint and method
- Response time and status code
- IP address and user agent
- Error messages (if any)

Usage:
    from request_logger import RequestLoggerMiddleware
    app.add_middleware(RequestLoggerMiddleware)
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import text
from core.data_manager_postgres import DataManager

logger = logging.getLogger(__name__)

class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests to database"""

    async def dispatch(self, request: Request, call_next):
        # Start timer
        start_time = time.time()

        # Get request info
        method = request.method
        endpoint = str(request.url.path)
        ip_address = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Get user info from request state (set by verify_api_key)
        user_id = None
        tier = None
        error_message = None

        # Try to extract user from request state
        if hasattr(request.state, "user"):
            user_info = request.state.user
            user_id = user_info.get("user_id")
            tier = user_info.get("tier", "free")

        # Log to database (async, non-blocking)
        try:
            self._log_request(
                user_id=user_id,
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                ip_address=ip_address,
                user_agent=user_agent,
                tier=tier,
                error_message=error_message
            )
        except Exception as e:
            # Don't fail request if logging fails
            logger.error(f"Failed to log request: {e}")

        return response

    def _log_request(
        self,
        user_id,
        endpoint,
        method,
        status_code,
        response_time_ms,
        ip_address,
        user_agent,
        tier,
        error_message
    ):
        """Log request to database"""
        try:
            dm = DataManager()

            with dm.engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO request_logs
                        (user_id, endpoint, method, status_code, response_time_ms,
                         ip_address, user_agent, tier, error_message)
                        VALUES
                        (:user_id, :endpoint, :method, :status_code, :response_time_ms,
                         :ip_address, :user_agent, :tier, :error_message)
                    """),
                    {
                        "user_id": user_id,
                        "endpoint": endpoint,
                        "method": method,
                        "status_code": status_code,
                        "response_time_ms": response_time_ms,
                        "ip_address": ip_address[:50] if ip_address else None,  # Truncate if too long
                        "user_agent": user_agent[:500] if user_agent else None,  # Truncate
                        "tier": tier,
                        "error_message": error_message[:1000] if error_message else None
                    }
                )

            dm.close()

        except Exception as e:
            logger.error(f"Database logging error: {e}")


# Utility function to get request stats
def get_request_stats(user_id=None, days=7):
    """
    Get request statistics

    Args:
        user_id: Filter by user ID (None = all users)
        days: Number of days to analyze

    Returns:
        dict with statistics
    """
    dm = DataManager()

    try:
        # Build query based on filters
        where_clause = "WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL :days DAY"
        params = {"days": days}

        if user_id:
            where_clause += " AND user_id = :user_id"
            params["user_id"] = user_id

        query = f"""
            SELECT
                COUNT(*) as total_requests,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(response_time_ms) as avg_response_time,
                MAX(response_time_ms) as max_response_time,
                COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
                COUNT(CASE WHEN status_code = 200 THEN 1 END) as success_count
            FROM request_logs
            {where_clause}
        """

        with dm.engine.connect() as conn:
            result = conn.execute(text(query), params)
            row = result.fetchone()

        dm.close()

        if row:
            return {
                "total_requests": row[0] or 0,
                "unique_users": row[1] or 0,
                "avg_response_time_ms": round(row[2], 2) if row[2] else 0,
                "max_response_time_ms": round(row[3], 2) if row[3] else 0,
                "error_count": row[4] or 0,
                "success_count": row[5] or 0,
                "error_rate": round((row[4] or 0) / (row[0] or 1) * 100, 2)
            }
        else:
            return {
                "total_requests": 0,
                "unique_users": 0,
                "avg_response_time_ms": 0,
                "max_response_time_ms": 0,
                "error_count": 0,
                "success_count": 0,
                "error_rate": 0
            }

    except Exception as e:
        logger.error(f"Error getting request stats: {e}")
        dm.close()
        return None
