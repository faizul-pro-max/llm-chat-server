#!/usr/bin/env bash
# Optional: set up Tailscale so the app server can reach vLLM over a private network.
# Run once per GPU instance after `make install`.
set -euo pipefail

if ! command -v tailscale &>/dev/null; then
    echo "==> Installing Tailscale"
    curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "==> Starting Tailscale (you will need to authenticate)"
tailscale up --advertise-exit-node=false
tailscale ip -4
echo "==> Add the IP above to your app server .env as GPU_SERVER_IP"
