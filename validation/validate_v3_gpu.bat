@echo off
setlocal
set "VALIDATION_DIR=%~dp0"
for %%I in ("%VALIDATION_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  python -m venv "%REPO_ROOT%\.venv"
  "%PYTHON_EXE%" -m pip install -r "%REPO_ROOT%\requirements.txt"
)
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_refinement.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_adaptive_water.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_rigid_clusters.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_rigid_transition.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_rigid_contacts.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_multirate.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_multirate_city.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_impact_gates.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_fragment_scale.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_structural_hierarchy.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_support_graph.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_fracture_checkpoint.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_shallow_water.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_shallow_return.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_surface_reconstruction.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_water_mesh.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_water_mesh_bounds.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_mesh_hysteresis.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_city_styles.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_crack_rendering.py" || exit /b 1
"%PYTHON_EXE%" "%VALIDATION_DIR%validate_progressive_video.py" || exit /b 1
echo All DELUGE V3 CUDA validations passed.
endlocal
