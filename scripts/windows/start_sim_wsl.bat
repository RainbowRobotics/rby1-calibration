@echo off
chcp 65001 >nul
setlocal
set DISTRO=Ubuntu-22.04
set IMAGE=rainbowroboticsofficial/rby1-sim:0.10.6-m_v

set TRIES=0
:choose
set /a TRIES+=1
:: 입력이 끝난 창(EOF)에서 끝없이 다시 묻지 않도록 세 번까지만 묻는다.
if %TRIES% gtr 3 (
    echo [ERROR] 번호를 받지 못해 종료합니다.
    pause
    exit /b 1
)
echo RBY1 시뮬레이터(모델 m) 버전을 고르세요.
echo   1. v1.2
echo   2. v1.3
set CHOICE=
set VER=
set /p CHOICE=번호 입력:
if "%CHOICE%"=="1" set VER=1.2
if "%CHOICE%"=="2" set VER=1.3
if not defined VER (
    echo [ERROR] 1 또는 2를 입력하세요.
    echo.
    goto choose
)

echo [INFO] WSL2에서 RBY1 시뮬레이터 v%VER% 컨테이너를 구동합니다...

:: X11 접속 권한 허용
wsl -d %DISTRO% -e bash -c "DISPLAY=:0 xhost +" >nul 2>&1

:: 기존 실행 중인 컨테이너가 있으면 삭제
wsl -d %DISTRO% -e docker rm -f rby1_sim >nul 2>&1

:: 시뮬레이터 컨테이너 실행 (WSLg GUI 및 50051 포트 연동). 자동 재시작은 걸지 않는다: WSL이 뜰 때마다 시뮬이 켜졌다.
wsl -d %DISTRO% -e docker run -d --name rby1_sim --ipc=host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /mnt/wslg:/mnt/wslg:rw -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -p 50051:50051 %IMAGE%%VER%
if errorlevel 1 (
    echo [ERROR] 시뮬레이터 실행 실패. Docker 상태를 확인해주세요.
    pause
    exit /b 1
)

echo [SUCCESS] RBY1 시뮬레이터 v%VER%가 시작되었습니다. 포트: 127.0.0.1:50051
echo [INFO] 화면에 MuJoCo 3D 뷰어 창이 표시됩니다.
echo [INFO] 시뮬레이터를 쓰는 동안 이 창을 열어 두세요. 창을 닫으면 약 1분 뒤 WSL이 꺼지면서 시뮬레이터도 꺼집니다.
echo [INFO] 끝낼 때는 stop_sim_wsl.bat을 실행하거나 이 창에서 Ctrl+C를 누르세요.
echo.
:: WSL은 붙어 있는 프로세스가 없으면 약 1분 뒤 꺼지고 컨테이너도 함께 죽는다. 로그를 따라가며 창을 붙잡아 둔다.
wsl -d %DISTRO% -e docker logs -f rby1_sim
echo [INFO] 시뮬레이터가 종료되었습니다.
pause
