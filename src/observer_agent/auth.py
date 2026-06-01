"""x-api-key header authentication for the observer agent."""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


async def require_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> None:
    """FastAPI dependency — raises 401 if the key doesn't match AGENT_SECRET."""
    secret = os.getenv("AGENT_SECRET", "")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AGENT_SECRET not configured on this server",
        )
    if x_api_key != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
