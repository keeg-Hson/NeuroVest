"""
WebSocket Real-Time Prediction Streaming for NeuroVest API

Provides real-time updates when new predictions are generated.

Features:
- Subscribe to all predictions
- Subscribe to specific assets
- Automatic reconnection handling
- JSON message format

Usage (Client):
    // JavaScript example
    const ws = new WebSocket('ws://api.neurovest.com/ws/predictions?api_key=YOUR_KEY');

    ws.onmessage = (event) => {
        const prediction = JSON.parse(event.data);
        console.log('New prediction:', prediction);
    };

Usage (Python):
    import asyncio
    import websockets

    async def stream_predictions():
        uri = "ws://api.neurovest.com/ws/predictions?api_key=YOUR_KEY"
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                prediction = json.loads(message)
                print(f"New prediction: {prediction}")
"""

from fastapi import WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import List, Set, Optional
import asyncio
import json
import logging
from datetime import datetime

from auth_middleware import AuthManager
from core.data_manager_postgres import DataManager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        # All active connections
        self.active_connections: List[WebSocket] = []

        # Connections subscribed to specific tickers
        self.ticker_subscriptions: dict[str, Set[WebSocket]] = {}

        # Connection metadata (user_id, tier, etc.)
        self.connection_info: dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, user: dict, tickers: Optional[List[str]] = None):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)

        # Store user info
        self.connection_info[websocket] = {
            "user_id": user.get("user_id"),
            "tier": user.get("tier", "free"),
            "connected_at": datetime.now().isoformat(),
            "tickers": tickers or []
        }

        # Subscribe to specific tickers if provided
        if tickers:
            for ticker in tickers:
                ticker_upper = ticker.upper()
                if ticker_upper not in self.ticker_subscriptions:
                    self.ticker_subscriptions[ticker_upper] = set()
                self.ticker_subscriptions[ticker_upper].add(websocket)

        logger.info(f"WebSocket connected: user={user.get('user_id')} tickers={tickers}")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to NeuroVest prediction stream",
            "subscriptions": tickers or ["all"],
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Remove from ticker subscriptions
        for ticker_set in self.ticker_subscriptions.values():
            ticker_set.discard(websocket)

        # Remove metadata
        user_id = self.connection_info.get(websocket, {}).get("user_id")
        if websocket in self.connection_info:
            del self.connection_info[websocket]

        logger.info(f"WebSocket disconnected: user={user_id}")

    async def broadcast_prediction(self, prediction: dict):
        """
        Broadcast prediction to all relevant connections

        Args:
            prediction: Prediction dict with 'ticker' field
        """
        ticker = prediction.get("ticker", "").upper()
        message = {
            "type": "prediction",
            "data": prediction,
            "timestamp": datetime.now().isoformat()
        }

        # Send to ticker-specific subscribers
        if ticker in self.ticker_subscriptions:
            disconnected = []
            for websocket in self.ticker_subscriptions[ticker]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")
                    disconnected.append(websocket)

            # Clean up disconnected clients
            for ws in disconnected:
                self.disconnect(ws)

        # Send to "all" subscribers (no specific tickers)
        disconnected = []
        for websocket in self.active_connections:
            # Skip if already sent to ticker-specific subscriber
            if ticker and websocket in self.ticker_subscriptions.get(ticker, set()):
                continue

            # Skip if subscribed to specific tickers and this isn't one
            info = self.connection_info.get(websocket, {})
            if info.get("tickers") and ticker not in [t.upper() for t in info["tickers"]]:
                continue

            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def send_heartbeat(self):
        """Send periodic heartbeat to keep connections alive"""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds

            message = {
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "active_connections": len(self.active_connections)
            }

            disconnected = []
            for websocket in self.active_connections:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
                    disconnected.append(websocket)

            # Clean up
            for ws in disconnected:
                self.disconnect(ws)

    def get_stats(self) -> dict:
        """Get connection statistics"""
        tier_counts = {}
        for info in self.connection_info.values():
            tier = info.get("tier", "free")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "total_connections": len(self.active_connections),
            "by_tier": tier_counts,
            "ticker_subscriptions": {
                ticker: len(subs)
                for ticker, subs in self.ticker_subscriptions.items()
            }
        }


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    api_key: str = Query(..., description="Your API key"),
    tickers: Optional[str] = Query(None, description="Comma-separated tickers (e.g., SPY,QQQ)")
):
    """
    WebSocket endpoint for real-time prediction streaming

    Args:
        api_key: User's API key for authentication
        tickers: Optional comma-separated list of tickers to subscribe to

    Message Types:
        - connected: Initial connection confirmation
        - prediction: New prediction available
        - heartbeat: Keep-alive ping every 30s
        - error: Error message
    """
    # Validate API key
    user = AuthManager.validate_api_key(api_key)
    if not user:
        await websocket.close(code=1008, reason="Invalid API key")
        return

    # Parse tickers
    ticker_list = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]

    # Connect
    await manager.connect(websocket, user, ticker_list)

    try:
        # Keep connection alive and listen for messages
        while True:
            # Receive messages from client (e.g., subscribe/unsubscribe)
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle subscribe command
                if message.get("action") == "subscribe":
                    new_tickers = message.get("tickers", [])
                    for ticker in new_tickers:
                        ticker_upper = ticker.upper()
                        if ticker_upper not in manager.ticker_subscriptions:
                            manager.ticker_subscriptions[ticker_upper] = set()
                        manager.ticker_subscriptions[ticker_upper].add(websocket)

                    await websocket.send_json({
                        "type": "subscribed",
                        "tickers": new_tickers,
                        "timestamp": datetime.now().isoformat()
                    })

                # Handle unsubscribe command
                elif message.get("action") == "unsubscribe":
                    remove_tickers = message.get("tickers", [])
                    for ticker in remove_tickers:
                        ticker_upper = ticker.upper()
                        if ticker_upper in manager.ticker_subscriptions:
                            manager.ticker_subscriptions[ticker_upper].discard(websocket)

                    await websocket.send_json({
                        "type": "unsubscribed",
                        "tickers": remove_tickers,
                        "timestamp": datetime.now().isoformat()
                    })

                # Handle ping (respond with pong)
                elif message.get("action") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_new_prediction(prediction: dict):
    """
    Broadcast new prediction to all connected clients

    Call this function when a new prediction is generated

    Args:
        prediction: Prediction dictionary with ticker, label, probabilities, etc.
    """
    await manager.broadcast_prediction(prediction)


def get_websocket_stats() -> dict:
    """Get WebSocket connection statistics"""
    return manager.get_stats()


# Start heartbeat task when module loads
import threading

def start_heartbeat():
    """Start heartbeat in background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(manager.send_heartbeat())

# Start heartbeat thread
heartbeat_thread = threading.Thread(target=start_heartbeat, daemon=True)
heartbeat_thread.start()
