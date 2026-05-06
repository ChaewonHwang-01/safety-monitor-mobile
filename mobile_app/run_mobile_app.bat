@echo off
cd /d "%~dp0"
set "NODE24=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
if not exist node_modules (
  npm install
)
if exist "%NODE24%" (
  "%NODE24%" node_modules\expo\bin\cli start --lan --clear
) else (
  npm run start -- --lan --clear
)
pause
