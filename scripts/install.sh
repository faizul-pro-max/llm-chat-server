#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing system dependencies"
apt-get update -qq
apt-get install -y --no-install-recommends tmux curl

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Creating logs directory"
mkdir -p logs

echo "==> Done. Copy .env.example to .env and fill in your secrets."
echo "    cp .env.example .env"
