#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing system dependencies"
apt-get update -qq
apt-get install -y --no-install-recommends tmux curl

echo "==> Installing speedtest CLI"
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash
if ! apt-get install -y speedtest 2>/dev/null; then
    echo "    Ookla package unavailable for this distro; falling back to speedtest-cli (Python)"
    pip install speedtest-cli
fi

echo "==> Installing ngrok"
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | tee /etc/apt/trusted.gpg.d/ngrok.asc > /dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | tee /etc/apt/sources.list.d/ngrok.list > /dev/null
apt-get update -qq
apt-get install -y ngrok

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
