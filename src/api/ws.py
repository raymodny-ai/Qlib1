"""
WebSocket Real-Time Monitor

Provides a WebSocket endpoint for real-time system monitoring:
- PM Gate status changes
- Circuit breaker state transitions
- Backtest task progress
- System alerts

Authentication: JWT token passed as query parameter `?token=...`
Channels: gate, backtest, system, alerts
"""

import asyncio
import json
from typing import Any, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect, Query

from src.security.auth import decode_token_optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket connections with channel-based pub/sub.

    Channels:
    - gate: PM gate state changes
    - backtest: Backtest task progress updates
    - system: System health, cache stats, metrics
    - alerts: High-priority alerts
    """

    def __init__(self):
        # channel → set of (websocket, user_id)
        self._subscriptions: Dict[str, Set[tuple]] = {
            "gate": set(),
            "backtest": set(),
            "system": set(),
            "alerts": set(),
        }
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str = "anonymous",
        channels: List[str] = None,
    ):
        """Accept a WebSocket connection and subscribe to channels."""
        await websocket.accept()
        if channels is None:
            channels = ["system", "alerts"]

        async with self._lock:
            for channel in channels:
                if channel in self._subscriptions:
                    self._subscriptions[channel].add((websocket, user_id))

        logger.info(
            f"WebSocket 已连接: user={user_id}, channels={channels}"
        )
        # Send welcome message
        await self._send(websocket, {
            "type": "connected",
            "channels": channels,
            "message": "WebSocket 已连接",
        })

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket from all subscriptions."""
        async with self._lock:
            for channel in self._subscriptions:
                self._subscriptions[channel] = {
                    (ws, uid) for ws, uid in self._subscriptions[channel]
                    if ws != websocket
                }

    async def broadcast(
        self,
        channel: str,
        data: Dict[str, Any],
        exclude: WebSocket = None,
    ):
        """
        Broadcast a message to all subscribers of a channel.

        Args:
            channel: Target channel name
            data: JSON-serializable payload
            exclude: WebSocket to skip (optional)
        """
        if channel not in self._subscriptions:
            return

        message = json.dumps(data, default=str)

        async with self._lock:
            dead: List[WebSocket] = []
            for ws, _ in self._subscriptions[channel]:
                if ws == exclude:
                    continue
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)

            # Clean up dead connections
            for ws in dead:
                await self.disconnect(ws)

    async def _send(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send a JSON message to a specific WebSocket."""
        try:
            await websocket.send_text(json.dumps(data, default=str))
        except Exception:
            pass


# Global singleton
_manager = ConnectionManager()


def get_ws_manager() -> ConnectionManager:
    """Get the global WebSocket connection manager."""
    return _manager


# ---------------------------------------------------------------------------
#  FastAPI WebSocket Endpoint
# ---------------------------------------------------------------------------

async def ws_monitor_endpoint(
    websocket: WebSocket,
    token: str = Query(""),
    channels: str = Query("system,alerts"),
):
    """
    WebSocket endpoint: /ws/monitor

    Connect with: ws://host:8000/ws/monitor?token=<JWT>&channels=gate,backtest,system,alerts

    Receives: JSON messages with {type, ...} based on subscribed channels.
    """
    manager = get_ws_manager()

    # Authenticate
    user_id = "anonymous"
    if token:
        payload = decode_token_optional(token)
        if payload:
            user_id = payload.get("sub", "anonymous")

    channel_list = [c.strip() for c in channels.split(",") if c.strip()]

    await manager.connect(websocket, user_id=user_id, channels=channel_list)

    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            # Client can send heartbeats or channel changes
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket 错误: {e}")
        await manager.disconnect(websocket)
