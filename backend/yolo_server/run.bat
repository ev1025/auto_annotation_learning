@echo off
chcp 65001 >nul
cd /d %~dp0
echo [YOLO 연동 테스트 서버] 기동 중... (창을 닫으면 서버가 꺼집니다)
python server.py
pause
