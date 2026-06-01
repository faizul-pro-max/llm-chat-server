"""Load .env and validate required environment variables."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def load(env_file: str = ".env") -> None:
    """Load .env file. Silently skips if the file does not exist."""
    load_dotenv(dotenv_path=Path(env_file), override=False)


def require(key: str) -> str:
    """Return env var value or exit with a clear error."""
    value = os.getenv(key)
    if not value:
        print(f"[env] Missing required environment variable: {key}", file=sys.stderr)
        print(f"[env] Copy .env.example to .env and fill in the value.", file=sys.stderr)
        sys.exit(3)
    return value


def get(key: str, default: str = "") -> str:
    return os.getenv(key, default)
