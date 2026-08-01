@echo off
setlocal
cd /d "%~dp0"
if not defined PYTHON_EXE set "PYTHON_EXE=.venv\Scripts\python.exe"
"%PYTHON_EXE%" deluge_v3.py --config config_v3_rtx5070.json --output outputs\smoke --smoke --no-video
if errorlevel 1 pause
endlocal
