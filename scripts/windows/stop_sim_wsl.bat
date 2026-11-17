@echo off
chcp 65001 >nul
echo [INFO] RBY1 시뮬레이터 컨테이너를 중지합니다...
wsl -d Ubuntu-22.04 -e docker rm -f rby1_sim
echo [INFO] 시뮬레이터가 안전하게 종료되었습니다.
pause
