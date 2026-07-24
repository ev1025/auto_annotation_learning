@echo off
rem 검증 대시보드 실행 (더블클릭) - http://127.0.0.1:7862
cd /d "%~dp0"
start "" http://127.0.0.1:7862
venv\Scripts\python.exe scripts\verify\dashboard_api.py
pause
