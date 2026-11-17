#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================="
echo "   Camera Calibrator - Linux Standalone Build"
echo "==================================================="

# 1. Virtual environment setup. Python 3.12 (setup.py): Ubuntu 22.04's python3 is 3.10, so install
# python3.12 next to it first (deadsnakes PPA, see the message below). Build on Ubuntu 22.04.
if [ ! -f ".venv/bin/activate" ]; then
    PY=""
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
            PY="$candidate"
            break
        fi
    done
    if [ -z "$PY" ]; then
        echo "[ERROR] Python 3.12 is needed (python3 here is $(python3 --version 2>&1))."
        echo "        Ubuntu 22.04: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
    echo "[1/4] Creating virtual environment (.venv) with $($PY --version)..."
    "$PY" -m venv .venv
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# 2. Dependencies
echo "[3/4] Installing dependencies..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
pip install --no-deps -e .

# 3. Build with PyInstaller
echo "[4/4] Building with PyInstaller..."
pyinstaller --clean --noconfirm camera_calibrator.spec

echo ""
echo "==================================================="
echo "   Build Successful! (Linux)"
echo "   Output binary: dist/camera_calibrator_$(uname -m)"
echo "==================================================="
