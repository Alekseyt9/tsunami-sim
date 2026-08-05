@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
echo DELUGE V3 HYBRID / RTX 5070
echo Do not run concurrently with another GPU simulation.
echo Production preset: sustained surge reaching all three building rows.
"%PYTHON_EXE%" deluge_v3.py --config config_v3_sustained_surge_30s.json
if errorlevel 1 pause
endlocal
