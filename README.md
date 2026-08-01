# DELUGE V3 — GPU Tsunami and City Destruction Simulator

DELUGE V3 is an offline CUDA simulation of a large tsunami hitting a destructible modern city. It combines adaptive 3D SPH water, a GPU shallow-water far field, structural fracture, rigid debris, a reconstructed water surface, and four synchronized camera views.

![DELUGE V3 four-camera simulation](assets/deluge_v3_quad.jpg)

The project targets an NVIDIA RTX 5070 with 12 GB of VRAM. It prioritizes physical state, reproducibility, and high-quality offline output over real-time playback.

## Main features

- CUDA simulation through NVIDIA Warp.
- Adaptive water particles at 1.0, 0.5, 0.325, and 0.1625 m scales.
- A conservative 2D shallow-water far field coupled to the local 3D SPH region.
- GPU free-surface classification, anisotropic surface samples, foam, and Marching Cubes reconstruction.
- Temporally stable mesh bounds and voxel LOD with local high-resolution splash bricks.
- Fifteen physically different building silhouettes: rectangular, podium, setback, tapered, and offset towers.
- Six coordinated facade palettes for concrete, stone, brick, glass, and roofs.
- Explicit slabs, walls, beams, columns, cores, glazing, floor spans, and roofs.
- Dormant structural LOD: untouched buildings remain inexpensive fixed water boundaries.
- Local structural refinement near predicted impact and growing cracks.
- Cohesive architectural fragments that prevent buildings from dissolving into particle dust.
- Rigid-cluster conversion for detached, settled debris.
- Frictional rigid-debris contacts with equal-and-opposite forces and torques.
- Rigid-to-deformable reactivation after a new strong collision.
- Original, front, side, and top cameras combined into one 1920×1080 video.
- Direct NVENC output without intermediate PNG sequences.
- A playable MP4 is atomically updated after every completed simulated second.
- Checkpoint and resume support for both particle and V3 hybrid state.

## Requirements

- Windows 10 or 11
- NVIDIA GPU with CUDA support
- A current NVIDIA driver
- Python 3.11 or newer
- FFmpeg/ffprobe available on `PATH` for validation and video inspection

The launcher creates `.venv` automatically and installs the Python packages from `requirements.txt`.

## Run

Double-click:

```bat
start_v3_rtx5070.bat
```

Or run from a terminal:

```bat
.venv\Scripts\python.exe deluge_v3.py --config config_v3_rtx5070.json
```

Useful short runs:

```bat
.venv\Scripts\python.exe deluge_v3.py --config config_v3_rtx5070.json --frames 100
.venv\Scripts\python.exe deluge_v3.py --config config_v3_rtx5070.json --duration 0.25
```

Do not run production V2 and V3 simulations simultaneously. They will compete for the same GPU and invalidate timing measurements.

## Progressive video output

The default configuration writes four views into one `deluge_v3.mp4`. No PNG frame sequence is created.

Frames are encoded in completed one-second segments. After each simulated second, all finished segments are stream-copied into a new preview and atomically replace the public MP4. This has three useful properties:

- the MP4 becomes non-zero and playable after the first 24 frames at 24 FPS;
- readers never observe a half-written replacement file;
- if the solver is interrupted, the last public MP4 and completed files in the `.segments` directory remain recoverable.

Reopen the file in VLC or mpv after another simulated second is completed to see the latest version. On normal completion, the temporary segment directory is removed.

## Current RTX 5070 results

The clean 100-frame V3.6 QA run used four cameras in one H.264 video:

- 100 output frames, 4.1667 simulated seconds;
- 459.4 seconds total wall time;
- 4.57 seconds average per output frame;
- 170,131 initial particles and 566,895 peak particles after adaptive refinement;
- 255,829 peak fluid particles;
- 1,659.6 MiB peak reported VRAM use;
- 788 released cohesive fragments and one active rigid cluster;
- 37,652 late water-mesh vertices and 74,870 triangles;
- a constant 0.65 m water voxel from frame 0 through frame 99;
- zero water-mesh LOD changes and no late collapse to spherical spray.

The V3.4 wide-scene validation used a 420 m domain and 45 buildings:

- 508,747 initial particles;
- 19,320 shallow-water cells;
- about 1.64 GiB reported VRAM use.

The shallow-water regression measured 0.249% volume drift after one simulated second and a zero float32 residual for the SPH↔2D exchange impulse.

## Water representation

The local impact region is simulated with 3D SPH particles. Only classified free-surface particles contribute to rendering. A compact GPU scalar field is smoothed and reconstructed with Warp Marching Cubes.

Sparse distant droplets do not expand the global reconstruction box. Bounds expand immediately but shrink gradually; improved voxel LOD requires eight stable frames. Dense splash sheets outside the main water body receive up to six local 12 m mesh bricks with a 0.4 m voxel, while isolated droplets stay inexpensive anisotropic spray samples.

The shallow-water far field remains part of the physics and momentum coupling, but is not drawn as a second independent surface. This avoids two overlapping water planes at different heights in side views.

## Structural model

Buildings begin as inexpensive fixed SPH boundaries. A building activates only after a coherent hydrodynamic load reaches enough facade samples. Foundations remain anchored.

Each building contains physically sampled floor slabs, roofs, walls, beams, columns, cores, and glass. Structural particles are grouped into cohesive architectural fragments approximately 3×3×3 m. Joints between fragments can fail, but the fragment interior remains connected, so failure produces slabs and wall sections instead of independent particle dust.

Detached calm fragments can become rigid clusters. Their mass, center of mass, inertia tensor, linear momentum, and angular momentum are fitted in double precision. Rigid motion and contacts then run on CUDA. Concrete, glass, and reinforcement use different contact stiffness and friction. A sufficiently strong later collision reactivates the cohesive deformable model.

## Validation

Run the complete CUDA regression batch:

```bat
validate_v3_gpu.bat
```

It covers:

- adaptive refinement and conservation of mass, volume, momentum, and center of mass;
- rigid fitting, motion, contact force/torque balance, friction, and reactivation;
- multirate momentum and city-state agreement;
- shallow-water volume and conservative overlap impulse;
- free-surface classification and reconstructed top-layer closure;
- robust late mesh bounds, temporal hysteresis, and splash-brick selection;
- physical building-style diversity and facade palettes;
- early opening and finalization of progressive NVENC MP4 output.

## Important files

- `deluge_v3.py` — V3 orchestration, checkpoints, surface reconstruction, and simulation loop integration.
- `solver_base.py` — shared particle solver and output loop in the standalone repository.
- `shallow_water.py` — GPU shallow-water solver and conservative SPH interface coupling.
- `hybrid_kernels.py` — structural LOD, fracture, multirate, rigid-body, contact, and facade CUDA kernels.
- `hybrid_model.py` — cohesive fragments, refinement axes, and facade generation.
- `hybrid_renderer.py` — facade and reconstructed-water rendering.
- `surface_kernels.py` — free-surface classification, sparse fields, temporal blending, and water rasterization.
- `config_v3_rtx5070.json` — production RTX 5070 configuration.
- `ROADMAP.md` — development stages and acceptance criteria.

## Status

V3.3d mesh stability, V3.4 shallow-water far field, V3.5 debris contacts, and the 100-frame portion of V3.6 are implemented and validated. The previous 8-second continuation was intentionally stopped after detecting the duplicated far-water rendering plane; the renderer configuration now keeps only the unified local surface visible. A new full 8-second production run should be started from a checkpoint produced with the corrected configuration.
