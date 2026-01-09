"""
Analytics API Endpoints for NeuroVest

Provides insights into API usage, popular assets, and performance metrics.

Endpoints:
- GET /api/analytics/usage - Request statistics by user/tier
- GET /api/analytics/popular - Most requested assets
- GET /api/analytics/errors - Error analysis
- GET /api/analytics/performance - Response time metrics
- GET /api/analytics/dashboard - Complete analytics dashboard
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import text
import logging

from core.data_manager_postgres import DataManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


def require_admin_or_self(user_id: Optional[int] = None):
    """
    Simple auth check - in production, verify user has permission
    For now, returns True (open access)
    """
    # TODO: Add proper admin role checking
    return True


@router.get("/usage")
def get_usage_stats(
    days: int = Query(7, ge=1, le=90, description="Days to analyze"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    tier: Optional[str] = Query(None, description="Filter by tier")
):
    """
    Get API usage statistics

    Returns:
        - Total requests
        - Unique users
        - Requests by tier
        - Daily breakdown
    """
    try:
        dm = DataManager()

        # Build WHERE clause
        where_parts = [f"created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'"]
        params = {}

        if user_id is not None:
            where_parts.append("user_id = :user_id")
            params["user_id"] = user_id

        if tier is not None:
            where_parts.append("tier = :tier")
            params["tier"] = tier

        where_clause = " AND ".join(where_parts)

        # Overall stats
        query = f"""
            SELECT
                COUNT(*) as total_requests,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(response_time_ms) as avg_response_time,
                COUNT(CASE WHEN status_code >= 400 THEN 1 END) as errors
            FROM request_logs
            WHERE {where_clause}
        """

        with dm.engine.connect() as conn:
            result = conn.execute(text(query), params)
            overall = result.fetchone()

            # Requests by tier
            tier_query = f"""
                SELECT tier, COUNT(*) as count
                FROM request_logs
                WHERE {where_clause}
                GROUP BY tier
                ORDER BY count DESC
            """
            tier_result = conn.execute(text(tier_query), params)
            by_tier = [{"tier": row[0] or "free", "count": row[1]} for row in tier_result]

            # Daily breakdown
            daily_query = f"""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as requests,
                    COUNT(DISTINCT user_id) as users
                FROM request_logs
                WHERE {where_clause}
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 30
            """
            daily_result = conn.execute(text(daily_query), params)
            daily = [
                {
                    "date": str(row[0]),
                    "requests": row[1],
                    "users": row[2]
                }
                for row in daily_result
            ]

        dm.close()

        return {
            "period_days": days,
            "overall": {
                "total_requests": overall[0] or 0,
                "unique_users": overall[1] or 0,
                "avg_response_time_ms": round(overall[2], 2) if overall[2] else 0,
                "error_count": overall[3] or 0
            },
            "by_tier": by_tier,
            "daily": daily
        }

    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/popular")
def get_popular_assets(
    days: int = Query(7, ge=1, le=90, description="Days to analyze"),
    limit: int = Query(10, ge=1, le=100, description="Number of assets to return")
):
    """
    Get most requested assets

    Returns top N most requested ticker symbols
    """
    try:
        dm = DataManager()

        query = f"""
            SELECT
                SUBSTRING(endpoint FROM '/api/predictions/([^/]+)') as ticker,
                COUNT(*) as request_count,
                COUNT(DISTINCT user_id) as unique_users
            FROM request_logs
            WHERE endpoint LIKE '/api/predictions/%'
              AND endpoint NOT LIKE '%/history'
              AND endpoint NOT LIKE '%/batch'
              AND created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
            GROUP BY ticker
            HAVING ticker IS NOT NULL
            ORDER BY request_count DESC
            LIMIT :limit
        """

        with dm.engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit})
            assets = [
                {
                    "ticker": row[0],
                    "requests": row[1],
                    "unique_users": row[2]
                }
                for row in result
            ]

        dm.close()

        return {
            "period_days": days,
            "total_assets": len(assets),
            "assets": assets
        }

    except Exception as e:
        logger.error(f"Error getting popular assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors")
def get_error_analysis(
    days: int = Query(7, ge=1, le=90, description="Days to analyze")
):
    """
    Get error analysis

    Returns:
        - Error count by status code
        - Error rate trends
        - Most error-prone endpoints
    """
    try:
        dm = DataManager()

        with dm.engine.connect() as conn:
            # Errors by status code
            status_query = f"""
                SELECT status_code, COUNT(*) as count
                FROM request_logs
                WHERE status_code >= 400
                  AND created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
                GROUP BY status_code
                ORDER BY count DESC
            """
            status_result = conn.execute(text(status_query))
            by_status = [
                {"status_code": row[0], "count": row[1]}
                for row in status_result
            ]

            # Error-prone endpoints
            endpoint_query = f"""
                SELECT
                    endpoint,
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as errors,
                    ROUND(COUNT(CASE WHEN status_code >= 400 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as error_rate
                FROM request_logs
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
                GROUP BY endpoint
                HAVING COUNT(CASE WHEN status_code >= 400 THEN 1 END) > 0
                ORDER BY error_rate DESC
                LIMIT 10
            """
            endpoint_result = conn.execute(text(endpoint_query))
            by_endpoint = [
                {
                    "endpoint": row[0],
                    "total_requests": row[1],
                    "errors": row[2],
                    "error_rate": float(row[3])
                }
                for row in endpoint_result
            ]

            # Daily error trend
            trend_query = f"""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as total,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as errors
                FROM request_logs
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """
            trend_result = conn.execute(text(trend_query))
            trend = [
                {
                    "date": str(row[0]),
                    "total_requests": row[1],
                    "errors": row[2],
                    "error_rate": round(row[2] / row[1] * 100, 2) if row[1] > 0 else 0
                }
                for row in trend_result
            ]

        dm.close()

        return {
            "period_days": days,
            "by_status_code": by_status,
            "by_endpoint": by_endpoint,
            "daily_trend": trend
        }

    except Exception as e:
        logger.error(f"Error getting error analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
def get_performance_metrics(
    days: int = Query(7, ge=1, le=90, description="Days to analyze")
):
    """
    Get API performance metrics

    Returns response time statistics and trends
    """
    try:
        dm = DataManager()

        with dm.engine.connect() as conn:
            # Overall performance
            overall_query = f"""
                SELECT
                    AVG(response_time_ms) as avg,
                    MIN(response_time_ms) as min,
                    MAX(response_time_ms) as max,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) as p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99
                FROM request_logs
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
                  AND status_code = 200
            """
            overall_result = conn.execute(text(overall_query))
            overall = overall_result.fetchone()

            # Performance by endpoint
            endpoint_query = f"""
                SELECT
                    endpoint,
                    COUNT(*) as requests,
                    AVG(response_time_ms) as avg_time,
                    MAX(response_time_ms) as max_time
                FROM request_logs
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
                  AND status_code = 200
                GROUP BY endpoint
                ORDER BY avg_time DESC
                LIMIT 10
            """
            endpoint_result = conn.execute(text(endpoint_query))
            by_endpoint = [
                {
                    "endpoint": row[0],
                    "requests": row[1],
                    "avg_ms": round(row[2], 2),
                    "max_ms": round(row[3], 2)
                }
                for row in endpoint_result
            ]

        dm.close()

        return {
            "period_days": days,
            "overall": {
                "avg_ms": round(overall[0], 2) if overall[0] else 0,
                "min_ms": round(overall[1], 2) if overall[1] else 0,
                "max_ms": round(overall[2], 2) if overall[2] else 0,
                "p50_ms": round(overall[3], 2) if overall[3] else 0,
                "p95_ms": round(overall[4], 2) if overall[4] else 0,
                "p99_ms": round(overall[5], 2) if overall[5] else 0
            },
            "slowest_endpoints": by_endpoint
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
def get_analytics_dashboard(
    days: int = Query(7, ge=1, le=90, description="Days to analyze")
):
    """
    Complete analytics dashboard

    Returns all key metrics in one endpoint
    """
    try:
        usage = get_usage_stats(days=days)
        popular = get_popular_assets(days=days, limit=5)
        errors = get_error_analysis(days=days)
        performance = get_performance_metrics(days=days)

        return {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "usage": usage,
            "popular_assets": popular,
            "errors": errors,
            "performance": performance
        }

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
