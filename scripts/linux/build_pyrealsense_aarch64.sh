#!/bin/bash
# Builds pyrealsense2 from the librealsense source for this project's .venv on aarch64 Linux (Jetson Orin, robot UPC).
# PyPI's aarch64 wheels are tied to a JetPack line (cp310 = JetPack 6, cp312 = JetPack 7), so the Python 3.12 wheel
# does not fit a JetPack 6 / Ubuntu 22.04 board. Usage: bash scripts/linux/build_pyrealsense_aarch64.sh [tag]
set -euo pipefail

TAG="${1:-v2.59.1}"         # same release as the tested Windows set (pyrealsense2 2.59.1)
JOBS="${JOBS:-$(nproc)}"    # JOBS=2 if the compiler runs out of memory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$PROJECT_ROOT/.venv/bin/python"
SRC="${LIBREALSENSE_SRC:-$HOME/librealsense-${TAG#v}}"
BUILD="$SRC/build-py312"

fail() { echo "[ERROR] $*"; exit 1; }

echo "==================================================="
echo "   pyrealsense2 $TAG for .venv (aarch64)"
echo "==================================================="

[ "$(uname -m)" = "aarch64" ] || fail "This is for aarch64 (Jetson / UPC). Windows and x86_64 Linux get pyrealsense2 from requirements.txt."
[ -x "$PY" ] || fail ".venv not found. Create it first (README, Linux step 5: python3.12 -m venv .venv)."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' \
    || fail ".venv is $("$PY" --version 2>&1), Python 3.12 is needed."

echo "[1/5] Installing build packages (sudo)..."
sudo apt-get update
sudo apt-get install -y git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libudev-dev python3.12-dev
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[ -f "$PY_INC/Python.h" ] || fail "Python.h not found in $PY_INC (python3.12-dev from the deadsnakes PPA)."

echo "[2/5] librealsense $TAG source in $SRC..."
if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/IntelRealSense/librealsense.git "$SRC"
fi

echo "[3/5] Building with $JOBS jobs (this takes a while)..."
# RSUSB backend: no kernel patch on Jetson. CUDA off: the app only reads frames.
# CHECK_FOR_UPDATES off: the library must not contact the internet at run time.
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY" \
    -DPYTHON_INSTALL_DIR="$BUILD/python-install" \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_WITH_CUDA=OFF \
    -DCHECK_FOR_UPDATES=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build "$BUILD" -j "$JOBS"

echo "[4/5] Installing librealsense2 (sudo) and pyrealsense2 into .venv..."
sudo cmake --install "$BUILD"
sudo ldconfig
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Find the module before touching .venv: an empty pyrealsense2 folder imports as a namespace package
# and the app then fails later with "module 'pyrealsense2' has no attribute 'context'".
# The same file sits in the build output and in python-install; cp stops on a repeated name, so keep one of each.
mapfile -t MODS < <(find "$BUILD" \( -name 'pyrealsense2*.so*' -o -name 'pybackend2*.so*' \) -not -path '*/CMakeFiles/*' \
    | awk -F/ '!seen[$NF]++')
[ "${#MODS[@]}" -gt 0 ] || fail "No pyrealsense2 module was built under $BUILD."
"$PY" -m pip uninstall -y pyrealsense2 >/dev/null 2>&1 || true
rm -rf "$SITE/pyrealsense2"
mkdir -p "$SITE/pyrealsense2"
cp -P "${MODS[@]}" "$SITE/pyrealsense2/"
if [ -f "$SRC/wrappers/python/pyrealsense2/__init__.py" ]; then
    cp "$SRC/wrappers/python/pyrealsense2/__init__.py" "$SITE/pyrealsense2/"
else
    echo "from .pyrealsense2 import *" > "$SITE/pyrealsense2/__init__.py"
fi
[ -f "$SITE/pyrealsense2/__init__.py" ] || fail "__init__.py was not written to $SITE/pyrealsense2."

echo "[5/5] RealSense udev rule (sudo)..."
sudo cp "$SRC/config/99-realsense-libusb.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

"$PY" - <<'EOF'
from pyrealsense2 import pyrealsense2 as _ext   # raises the real loader error (missing .so, glibc) if any
import pyrealsense2 as rs
assert hasattr(rs, "context"), f"pyrealsense2 at {list(rs.__path__)} has no __init__.py importing the module"
devices = [f"{d.get_info(rs.camera_info.name)} ({d.get_info(rs.camera_info.serial_number)})" for d in rs.context().devices]
print("pyrealsense2 loaded from", rs.__file__)
print("Cameras:", ", ".join(devices) if devices else "none found (connect the camera to a USB 3 port, or re-plug it after the udev rule)")
EOF
echo "Done. Start the app with: bash scripts/linux/run_linux.sh"
