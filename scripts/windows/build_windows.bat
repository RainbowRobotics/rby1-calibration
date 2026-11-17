@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Camera Calibrator - Windows Standalone Build
echo ===================================================

:: Move to project root directory
cd /d "%~dp0..\.."

:: 0. Close running instance if any
echo [*] Checking and closing running instances...
taskkill /F /IM camera_calibrator*.exe 2>nul

:: 1. Find Python 3.12 (setup.py requires 3.12)
set "PY_CMD="

for %%v in (3.12) do (
    if not defined PY_CMD (
        py -%%v --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=py -%%v"
            echo [*] Found supported Python via py launcher: Python %%v
        )
    )
)

if not defined PY_CMD (
    python --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
        echo [*] Using system default python
    )
)

if not defined PY_CMD (
    echo [ERROR] Python 3.12 is not installed or not in PATH!
    echo [ERROR] Please install Python 3.12 from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Check / Create virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment .venv using !PY_CMD!...
    !PY_CMD! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

:: 3. Activate virtual environment
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Install / Update dependencies
echo [3/4] Installing required dependencies and rby1_sdk...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
pip install --no-deps -e .

:: 5. PyInstaller Standalone Executable Build
echo [4/4] Building Windows Standalone Executable (.exe)...
pyinstaller --clean --noconfirm camera_calibrator.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   Build Complete! (Windows)
echo   Output files located in: .\dist\
echo ===================================================
echo.
pause
