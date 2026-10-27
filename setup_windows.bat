@echo off
setlocal
cd /d "%~dp0"
echo === CrimeVision AI setup ===
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10, 3.11 or 3.12 from python.org and tick "Add Python to PATH".
  pause & exit /b 1
)
if not exist venv python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo [1/2] Installing core packages...
pip install -r requirements-core.txt
if errorlevel 1 (
  echo Core install failed. Copy the red error text and ask for help.
  pause & exit /b 1
)
if not exist .env copy .env.example .env >nul
echo.
set /p AI=[2/2] Install the AI models now? Needed for knife/person detection and forensic tags (about 3 GB) [Y/N]: 
if /i "%AI%"=="Y" (
  pip install -r requirements-ai.txt
  if errorlevel 1 echo AI install failed. The app still works in basic mode.
)
echo.
echo Done. Next: run start_windows.bat
pause
