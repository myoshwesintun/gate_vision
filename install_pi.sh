#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/4] Installing Raspberry Pi OS system packages..."
sudo apt update
sudo apt install -y \
  python3-venv python3-tk python3-gpiozero \
  tesseract-ocr libgl1 libglib2.0-0

# Newer Raspberry Pi OS versions may use lgpio as gpiozero's backend.
sudo apt install -y python3-lgpio 2>/dev/null || true

echo "[2/4] Creating Python virtual environment..."
if [ ! -d venv ]; then
  python3 -m venv --system-site-packages venv
fi

source venv/bin/activate

echo "[3/4] Installing Python packages..."
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
# Ensure Pillow/ImageTk is installed inside this venv (not only inherited from system).
python -m pip install --force-reinstall pillow

echo "[4/4] Running basic diagnostics..."
python check_system.py || true

echo
echo "Installation finished."
echo "Run the project with: ./run_gate_vision.sh"
