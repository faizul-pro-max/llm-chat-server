#!/usr/bin/env bash
# Create ngrok tunnels for vLLM (:8000) and the observer agent (:9100).
# Run AFTER `make start` — both services must already be up.
#
# Usage: bash scripts/tunnel.sh
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓ $1${RESET}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${RESET}"; }
die()  { echo -e "  ${RED}✗ $1${RESET}"; exit 1; }

# ── Load .env ─────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── Pre-checks ────────────────────────────────────────────────────────────────
command -v ngrok &>/dev/null || die "ngrok not found — run: make install"

if [ -z "${NGROK_AUTHTOKEN:-}" ]; then
    die "NGROK_AUTHTOKEN not set. Add it to .env:\n    NGROK_AUTHTOKEN=your_token_here\n    Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken"
fi

# ── Configure ngrok auth token ────────────────────────────────────────────────
echo "==> Configuring ngrok"
ngrok config add-authtoken "$NGROK_AUTHTOKEN" --config /root/.config/ngrok/ngrok.yml 2>/dev/null \
    || ngrok config add-authtoken "$NGROK_AUTHTOKEN"
ok "Auth token set"

# ── Write ngrok config for both tunnels ───────────────────────────────────────
NGROK_CFG="/tmp/ngrok-orchestrator.yml"
cat > "$NGROK_CFG" <<EOF
version: "2"
tunnels:
  vllm:
    addr: 8000
    proto: http
  agent:
    addr: 9100
    proto: http
EOF

# ── Kill any existing ngrok sessions ─────────────────────────────────────────
pkill -f "ngrok" 2>/dev/null && warn "Killed existing ngrok process" || true

# ── Start ngrok ───────────────────────────────────────────────────────────────
echo "==> Starting ngrok tunnels (vLLM :8000, Agent :9100)"
nohup ngrok start --all --config "$NGROK_CFG" > logs/ngrok.log 2>&1 &
NGROK_PID=$!
echo "  PID: $NGROK_PID"

# ── Wait for ngrok API to be ready ────────────────────────────────────────────
echo "==> Waiting for tunnels to establish…"
for i in $(seq 1 10); do
    sleep 1
    if curl -s http://localhost:4040/api/tunnels &>/dev/null; then
        break
    fi
    if [ "$i" -eq 10 ]; then
        die "ngrok did not start — check logs/ngrok.log"
    fi
done

# ── Fetch tunnel URLs from ngrok local API ────────────────────────────────────
TUNNELS_JSON=$(curl -s http://localhost:4040/api/tunnels)

VLLM_URL=$(echo "$TUNNELS_JSON" | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if t['config']['addr'].endswith('8000'):
        print(t['public_url']); break
" 2>/dev/null || echo "")

AGENT_URL=$(echo "$TUNNELS_JSON" | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if t['config']['addr'].endswith('9100'):
        print(t['public_url']); break
" 2>/dev/null || echo "")

# ── Print results ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ngrok tunnels active"
echo "════════════════════════════════════════════════════════════"
echo ""
if [ -n "$VLLM_URL" ]; then
    ok "vLLM:   $VLLM_URL"
else
    warn "vLLM tunnel not found — free ngrok tier only supports 1 tunnel at a time"
    warn "Upgrade to a paid ngrok plan for multiple tunnels, or tunnel only vLLM."
fi
if [ -n "$AGENT_URL" ]; then
    ok "Agent:  $AGENT_URL"
else
    warn "Agent tunnel not found (expected on free tier)"
fi

echo ""
echo "  Add to your Node.js app server .env:"
[ -n "$VLLM_URL"  ] && echo "  VLLM_BASE_URL=$VLLM_URL"
[ -n "$AGENT_URL" ] && echo "  GPU_AGENT_URL=$AGENT_URL"
echo ""
echo "  Tunnel log:  logs/ngrok.log"
echo "  ngrok UI:    http://localhost:4040"
echo "  To stop:     pkill -f ngrok"
echo ""
