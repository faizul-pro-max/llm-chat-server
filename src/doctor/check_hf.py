"""Check HuggingFace Hub reachability and token validity."""
from __future__ import annotations

import os

from src.doctor.runner import CheckResult


def check_hf_access(_scenario) -> CheckResult:
    """Verify HF Hub is reachable and the token (if set) is valid."""
    try:
        from huggingface_hub import HfApi
        import httpx

        token = os.getenv("HF_TOKEN")
        api = HfApi(token=token)

        # Quick connectivity check
        try:
            info = api.whoami()
            username = info.get("name", "unknown")
            return CheckResult(
                name="HuggingFace Hub access",
                passed=True,
                message=f"Authenticated as {username}",
            )
        except Exception:
            pass

        # No token or invalid token — check if hub is reachable at all
        r = httpx.get("https://huggingface.co/api/models?limit=1", timeout=10)
        if r.status_code == 200:
            msg = "Reachable (unauthenticated — gated models will fail)"
            return CheckResult(
                name="HuggingFace Hub access",
                passed=False,
                message=msg,
                severity="warning",
                detail="Set HF_TOKEN in .env for gated model access.",
            )
        return CheckResult(
            name="HuggingFace Hub access",
            passed=False,
            message=f"Hub returned HTTP {r.status_code}",
        )

    except Exception as exc:
        return CheckResult(
            name="HuggingFace Hub access",
            passed=False,
            message=f"Cannot reach HF Hub: {exc}",
            detail="Check network connectivity.",
        )
