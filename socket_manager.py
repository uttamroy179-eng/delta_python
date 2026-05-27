# engine/socket_manager.py

from fastapi import WebSocket
from typing import List
import json


class ConnectionManager:

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

        print(f"[Socket] Client connected ({len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        print(f"[Socket] Client disconnected ({len(self.active_connections)})")

    async def broadcast(self, data: dict):

        disconnected = []

        for connection in self.active_connections:

            try:
                await connection.send_text(json.dumps(data))

            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()