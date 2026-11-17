#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================="
echo "   Camera Calibrator - Linux Python Launcher"
echo "==================================================="

if [ -f ".venv/bin/activate" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "[!] Warning: .venv not found. Using system Python3..."
fi

# Python 3.12 (setup.py); Ubuntu 22.04's system python3 is 3.10.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "[ERROR] Python 3.12 is needed, found $(python3 --version 2>&1)."
    echo "        Create .venv with python3.12 (Ubuntu 22.04: deadsnakes PPA) or run scripts/linux/build_linux.sh."
    exit 1
fi

if [ -z "$1" ]; then
    echo "[*] Starting main_ui.py..."
    echo "[*] Tip: Pass '--ui' for simulation / UI-only mode."
    python3 main_ui.py
else
    echo "[*] Starting main_ui.py with args: $@"
    python3 main_ui.py "$@"
fi
