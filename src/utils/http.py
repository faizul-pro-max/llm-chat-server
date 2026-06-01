"""Shared httpx client with retry logic."""
import httpx


def get_client(timeout: float = 10.0, retries: int = 3) -> httpx.Client:
    """Return a synchronous httpx.Client with retry transport."""
    transport = httpx.HTTPTransport(retries=retries)
    return httpx.Client(timeout=timeout, transport=transport)


def get_async_client(timeout: float = 10.0, retries: int = 3) -> httpx.AsyncClient:
    """Return an async httpx.AsyncClient with retry transport."""
    transport = httpx.AsyncHTTPTransport(retries=retries)
    return httpx.AsyncClient(timeout=timeout, transport=transport)
