#!/usr/bin/env bash
# ==============================================================================
# PS26052 ANC — Raspberry Pi 3 Automated Deployment Script
# Configures system ALSA packages, virtual environment, and ReSpeaker 2-Mic HAT
# ==============================================================================

set -e

echo "=== [1/4] Installing System Audio & Build Dependencies ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv portaudio19-dev libasound2-dev git

echo "=== [2/4] Setting Up Virtual Environment ==="
python3 -m venv ~/anc_env
source ~/anc_env/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== [3/4] Installing Python Requirements ==="
pip install -r requirements.txt

echo "=== [4/4] Verifying ReSpeaker 2-Mic ALSA Detection ==="
arecord -l || true

echo "=============================================================================="
echo "Deployment Complete! To launch capture and streaming to laptop:"
echo "  source ~/anc_env/bin/activate"
echo "  python capture_stream.py --host <LAPTOP_IP> --port 5005"
echo "=============================================================================="
