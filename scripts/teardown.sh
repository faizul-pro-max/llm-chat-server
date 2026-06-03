#!/usr/bin/env bash
# Undo everything created by make install + make start.
#
# Usage:
#   bash scripts/teardown.sh               # safe teardown (keeps models)
#   bash scripts/teardown.sh --purge-models # also deletes HF model cache
#   bash scripts/teardown.sh --full        # + removes apt packages (tmux, curl)

set -euo pipefail

PURGE_MODELS=false
FULL=false

for arg in "$@"; do
  case "$arg" in
    --purge-models) PURGE_MODELS=true ;;
    --full)         FULL=true ;;
    *)
      echo "Unknown flag: $arg"
      echo "Usage: bash scripts/teardown.sh [--purge-models] [--full]"
      exit 1
      ;;
  esac
done

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'; RESET='\033[0m'
step()  { echo -e "\n${YELLOW}==> $1${RESET}"; }
ok()    { echo -e "  ${GREEN}✓ $1${RESET}"; }
skip()  { echo -e "  (skip) $1"; }
warn()  { echo -e "  ${RED}⚠ $1${RESET}"; }

echo ""
echo "GPU Server Orchestrator — Teardown"
echo "==================================="
if $PURGE_MODELS; then echo "  Mode: full teardown + model cache purge"; fi
if $FULL;         then echo "  Mode: full teardown + apt package removal"; fi
echo ""

# ── 1. Kill tmux sessions ────────────────────────────────────────────────────
step "Stopping tmux sessions (vllm, agent)"
for session in vllm agent; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
    ok "Killed session: $session"
  else
    skip "Session not running: $session"
  fi
done

# ── 2. Remove log files ──────────────────────────────────────────────────────
step "Removing log files"
if [ -d logs ]; then
  rm -f logs/*.log
  rmdir --ignore-fail-on-non-empty logs
  ok "Cleared logs/"
else
  skip "logs/ directory not found"
fi

# ── 3. Remove .env ───────────────────────────────────────────────────────────
step "Removing .env"
if [ -f .env ]; then
  rm .env
  ok "Deleted .env"
else
  skip ".env not found"
fi

# ── 4. Remove virtual environment ────────────────────────────────────────────
step "Removing .venv"
if [ -d .venv ]; then
  rm -rf .venv
  ok "Deleted .venv/"
else
  skip ".venv/ not found"
fi

# ── 5. Remove __pycache__ dirs ───────────────────────────────────────────────
step "Removing Python cache"
find . -path ./.venv -prune -o -type d -name __pycache__ -print -exec rm -rf {} + 2>/dev/null || true
find . -path ./.venv -prune -o -name "*.pyc" -print -delete 2>/dev/null || true
ok "Cleaned __pycache__ and .pyc files"

# ── 6. HuggingFace model cache (opt-in) ─────────────────────────────────────
step "HuggingFace model cache"
HF_CACHE="${HF_HOME:-/root/.cache/huggingface}"
if $PURGE_MODELS; then
  if [ -d "$HF_CACHE" ]; then
    CACHE_SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | cut -f1 || echo "unknown")
    rm -rf "$HF_CACHE"
    ok "Deleted $HF_CACHE ($CACHE_SIZE freed)"
  else
    skip "$HF_CACHE not found"
  fi
else
  if [ -d "$HF_CACHE" ]; then
    CACHE_SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | cut -f1 || echo "unknown")
    warn "Keeping model cache at $HF_CACHE ($CACHE_SIZE)"
    echo "       Re-run with --purge-models to delete it."
  else
    skip "No model cache found at $HF_CACHE"
  fi
fi

# ── 7. Remove apt packages (opt-in) ─────────────────────────────────────────
step "System packages (tmux, curl)"
if $FULL; then
  if command -v apt-get &>/dev/null; then
    apt-get remove -y tmux curl 2>/dev/null && ok "Removed tmux and curl via apt" || warn "apt-get remove failed — skipping"
  else
    skip "apt-get not available on this system"
  fi
else
  skip "Keeping tmux and curl — re-run with --full to remove them"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Teardown complete.${RESET}"
echo ""
echo "To reinstall from scratch:"
echo "  cp .env.example .env   # fill in secrets"
echo "  python -m venv .venv && source .venv/bin/activate"
echo "  make install"
echo "  make start SCENARIO=baseline"
echo ""
