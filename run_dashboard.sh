#!/usr/bin/env bash
# OneVoice AI — Start Dashboard and Public Tunnel
set -e

PORT=${1:-8080}
NGROK_TOKEN=${2:-""}

echo "====================================================================="
echo " 🚀 STARTING ONEVOICE AI QUALCOMM AI HUB DASHBOARD (PORT: $PORT)"
echo "====================================================================="

# Check python environment
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# 1. Start Server in background if not already running
if ! lsof -i :$PORT >/dev/null 2>&1; then
    echo "[+] Starting Dashboard Server on port $PORT..."
    $PYTHON dashboard/server.py $PORT &
    SERVER_PID=$!
    sleep 2
else
    echo "[+] Port $PORT is already active."
fi

# 2. Start Tunnel
if [ -n "$NGROK_TOKEN" ]; then
    echo "[+] Launching ngrok with provided authtoken..."
    $PYTHON dashboard/tunnel.py $PORT "$NGROK_TOKEN"
elif [ -n "$NGROK_AUTHTOKEN" ]; then
    echo "[+] Launching ngrok with environment NGROK_AUTHTOKEN..."
    $PYTHON dashboard/tunnel.py $PORT "$NGROK_AUTHTOKEN"
else
    echo "[*] Launching free public HTTPS tunnel (localtunnel)..."
    echo "[*] (To use ngrok: ./run_dashboard.sh 8080 <your_ngrok_token>)"
    $PYTHON dashboard/tunnel.py $PORT
fi
