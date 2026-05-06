@echo off
cd /d "%~dp0"
start "Safety Monitor API" cmd /k "python -m uvicorn api_server:app --host 0.0.0.0 --port 8000"
start "Safety Monitor Mobile" cmd /k "cd /d mobile_app && run_mobile_app.bat"
