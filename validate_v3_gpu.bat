@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
"%PYTHON_EXE%" validate_refinement.py || exit /b 1
"%PYTHON_EXE%" validate_rigid_clusters.py || exit /b 1
"%PYTHON_EXE%" validate_rigid_transition.py || exit /b 1
"%PYTHON_EXE%" validate_rigid_contacts.py || exit /b 1
"%PYTHON_EXE%" validate_multirate.py || exit /b 1
"%PYTHON_EXE%" validate_multirate_city.py || exit /b 1
"%PYTHON_EXE%" validate_impact_gates.py || exit /b 1
"%PYTHON_EXE%" validate_fragment_scale.py || exit /b 1
"%PYTHON_EXE%" validate_structural_hierarchy.py || exit /b 1
"%PYTHON_EXE%" validate_shallow_water.py || exit /b 1
"%PYTHON_EXE%" validate_shallow_return.py || exit /b 1
"%PYTHON_EXE%" validate_surface_reconstruction.py || exit /b 1
"%PYTHON_EXE%" validate_water_mesh.py || exit /b 1
"%PYTHON_EXE%" validate_water_mesh_bounds.py || exit /b 1
"%PYTHON_EXE%" validate_mesh_hysteresis.py || exit /b 1
"%PYTHON_EXE%" validate_city_styles.py || exit /b 1
"%PYTHON_EXE%" validate_progressive_video.py || exit /b 1
echo All DELUGE V3 CUDA validations passed.
endlocal
