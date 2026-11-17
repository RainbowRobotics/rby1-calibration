@echo off
setlocal enabledelayedexpansion

:: Move to project root directory
cd /d "%~dp0..\.."

echo ===================================================
echo   Camera Calibrator - Windows Python Launcher
echo ===================================================

:: 1. Check virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [!] Warning: .venv not found. Using system Python...
)

:: 2. Run main_ui.py with any passed arguments (e.g. --ui)
if "%~1"=="" (
    echo [*] Starting main_ui.py...
    echo [*] Tip: Pass '--ui' for simulation / UI-only mode.
    python main_ui.py
) else (
    echo [*] Starting main_ui.py with args: %*
    python main_ui.py %*
)

if errorlevel 1 (
    echo.
    echo [ERROR] Application terminated with error code !errorlevel!
    echo Check if all dependencies are installed: pip install -r requirements.txt
    pause
)
