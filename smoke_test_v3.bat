@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
"%PYTHON_EXE%" deluge_v3.py --config config_v3_rtx5070.json --output outputs\smoke --smoke --no-video
if errorlevel 1 pause
endlocal
