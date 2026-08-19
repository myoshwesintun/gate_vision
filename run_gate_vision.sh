#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
  echo "Virtual environment not found. Run ./install_pi.sh first."
  exit 1
fi

source venv/bin/activate
exec python gate_vision.py
