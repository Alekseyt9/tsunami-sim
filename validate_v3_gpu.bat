@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python environment not found: %PYTHON_EXE%
  exit /b 1
)
"%PYTHON_EXE%" validate_refinement.py || exit /b 1
"%PYTHON_EXE%" validate_rigid_clusters.py || exit /b 1
"%PYTHON_EXE%" validate_rigid_transition.py || exit /b 1
"%PYTHON_EXE%" validate_rigid_contacts.py || exit /b 1
"%PYTHON_EXE%" validate_multirate.py || exit /b 1
"%PYTHON_EXE%" validate_multirate_city.py || exit /b 1
"%PYTHON_EXE%" validate_shallow_water.py || exit /b 1
"%PYTHON_EXE%" validate_surface_reconstruction.py || exit /b 1
"%PYTHON_EXE%" validate_water_mesh.py || exit /b 1
"%PYTHON_EXE%" validate_water_mesh_bounds.py || exit /b 1
"%PYTHON_EXE%" validate_mesh_hysteresis.py || exit /b 1
"%PYTHON_EXE%" validate_city_styles.py || exit /b 1
"%PYTHON_EXE%" validate_progressive_video.py || exit /b 1
echo All DELUGE V3 CUDA validations passed.
endlocal
