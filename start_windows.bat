@echo off
cd /d "%~dp0"
if not exist venv (
  echo Run setup_windows.bat first.
  pause & exit /b 1
)
call venv\Scripts\activate.bat
echo Starting CrimeVision AI at http://localhost:5000  (close this window to stop)
start "" http://localhost:5000
python -m backend.app
pause
