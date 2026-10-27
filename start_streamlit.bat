@echo off
cd /d "%~dp0"
if not exist venv (
  echo Run setup_windows.bat first.
  pause & exit /b 1
)
call venv\Scripts\activate.bat
pip install streamlit
echo Starting CrimeVision AI (Streamlit) ... a browser tab will open.
streamlit run streamlit_app.py
pause
