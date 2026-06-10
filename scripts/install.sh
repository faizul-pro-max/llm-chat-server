#!/usr/bin/env bash
set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[install] $*"; }
warn() { echo "[install][WARN] $*" >&2; }

log "Cleaning up stale third-party apt sources (if any)"
rm -f /etc/apt/sources.list.d/ookla_speedtest-cli.list \
      /etc/apt/keyrings/ookla_speedtest-cli-archive-keyring.gpg 2>/dev/null || true

log "Running apt-get update"
if ! apt-get update -qq; then
    warn "apt-get update reported errors (likely a stale repo); continuing anyway"
    apt-get update -qq --fix-missing 2>/dev/null || true
fi

log "Installing system dependencies: tmux curl"
apt-get install -y --no-install-recommends tmux curl
log "System dependencies installed"

log "Installing speedtest CLI (best-effort — network check has HTTP fallback)"
(
    set +e
    curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash 2>&1 \
        && apt-get install -y speedtest 2>&1
    if command -v speedtest &>/dev/null; then
        log "Ookla speedtest installed: $(speedtest --version 2>&1 | head -1)"
    else
        warn "Ookla speedtest not available for this distro; network check will use HTTP fallback"
    fi
) || warn "speedtest install block failed — skipping"

# Remove Ookla repo so it doesn't poison subsequent apt-get update calls
rm -f /etc/apt/sources.list.d/ookla_speedtest-cli.list \
      /etc/apt/keyrings/ookla_speedtest-cli-archive-keyring.gpg 2>/dev/null || true
log "Ookla apt source removed"

log "Installing ngrok"
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | tee /etc/apt/trusted.gpg.d/ngrok.asc > /dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | tee /etc/apt/sources.list.d/ngrok.list > /dev/null
apt-get update -qq
apt-get install -y ngrok
log "ngrok installed: $(ngrok version 2>&1 | head -1)"

echo "==> Installing Python dependencies"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Creating logs directory"
mkdir -p logs

echo ""
echo "==> Done. Next steps:"
echo "    1. cp .env.example .env  — fill in secrets"
echo "    2. make doctor           — verify the box is ready"
echo "    3. make start SCENARIO=baseline"
echo "    4. bash scripts/tunnel.sh  — expose ports via ngrok (optional)"
