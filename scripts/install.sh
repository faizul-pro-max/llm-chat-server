#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing system dependencies"
apt-get update -qq
apt-get install -y --no-install-recommends tmux curl

echo "==> Installing Ookla speedtest CLI"
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash
apt-get install -y speedtest

echo "==> Installing ngrok"
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | tee /etc/apt/trusted.gpg.d/ngrok.asc > /dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | tee /etc/apt/sources.list.d/ngrok.list > /dev/null
apt-get update -qq
apt-get install -y ngrok

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Creating logs directory"
mkdir -p logs

echo ""
echo "==> Done. Next steps:"
echo "    1. cp .env.example .env  — fill in secrets"
echo "    2. make doctor           — verify the box is ready"
echo "    3. make start SCENARIO=baseline"
echo "    4. bash scripts/tunnel.sh  — expose ports via ngrok (optional)"
