"""Download a model from HuggingFace Hub with a progress display."""
from __future__ import annotations

import os
import sys

from src.utils import logging as log


def download_model(model_id: str, show_progress: bool = True) -> None:
    """Download model_id via huggingface_hub. Idempotent — skips if cached."""
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    token = os.getenv("HF_TOKEN") or None
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    log.info(f"Model:     {model_id}")
    log.info(f"Cache dir: {cache_dir}")

    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            token=token,
            local_files_only=False,
        )
        log.success(f"Model ready at: {local_dir}")
    except Exception as exc:
        log.error(f"Download failed: {exc}")
        if "401" in str(exc) or "gated" in str(exc).lower():
            log.warning("This model may be gated. Set HF_TOKEN in .env with a valid token.")
        sys.exit(1)
