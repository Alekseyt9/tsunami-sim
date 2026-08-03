# DELUGE V3 — GPU Tsunami and City Destruction Simulator

DELUGE V3 is an offline CUDA simulation of a large tsunami hitting a destructible modern city. It combines adaptive 3D SPH water, a GPU shallow-water far field, structural fracture, rigid debris, a reconstructed water surface, and four synchronized camera views.

![DELUGE V3 four-camera simulation](assets/deluge_v3_quad.jpg)

The project targets an NVIDIA RTX 5070 with 12 GB of VRAM. It prioritizes physical state, reproducibility, and high-quality offline output over real-time playback.

## Main features

- CUDA simulation through NVIDIA Warp.
- Adaptive water particles at 1.0, 0.5, 0.325, and 0.1625 m scales.
- A conservative 2D shallow-water far field with bidirectional transfer to the local 3D SPH region.
- GPU free-surface classification, anisotropic surface samples, foam, and Marching Cubes reconstruction.
- Temporally stable mesh bounds and voxel LOD with local high-resolution splash bricks.
- Fifteen physically different building silhouettes: rectangular, podium, setback, tapered, and offset towers.
- Six coordinated facade palettes for concrete, stone, brick, glass, and roofs.
- Explicit slabs, walls, beams, columns, cores, glazing, floor spans, and roofs.
- Dormant structural LOD: untouched buildings remain inexpensive fixed water boundaries.
- Local structural refinement near predicted impact and growing cracks.
- Cohesive architectural fragments that prevent buildings from dissolving into particle dust.
- Structural-role fracture hierarchy: glazing yields first, while slabs, beams, columns, and cores progressively resist more strain and accumulate damage more slowly.
- Rigid-cluster conversion for detached, settled debris.
- Frictional rigid-debris contacts with equal-and-opposite forces and torques.
- Rigid-to-deformable reactivation after a new strong collision.
- Target-side rubble impacts: dormant buildings receive local glazing/wall damage and wake only after a spatially coherent high-energy hit.
- A finite conservative second long-wave pulse, with injected volume and momentum recorded in checkpoints and metrics.
- Water-coupled cars, breakable trees, destructible low-rise shops, roads, sidewalks, and distinct environment materials.
- Foreground low-rise streetscape, directional building/object shadows, and
  deterministic sun glints on facade glass and vehicle paint.
- A short 10 m secondary bore with a validated open-field arrival at all three
  building-row depths; it replaces the former long pulse that read as a slow
  water-level rise.
- Cinematic lighting, water Fresnel/specular response, foam, screen-space contact shading, wetness, atmospheric haze, and vignette.
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

The V3.10 100-frame production run used four cameras in one H.264 video:

- 100 output frames, 4.1667 simulated seconds;
- 591.1 seconds total wall time before the V3.11 query optimization;
- 5.84 seconds average and 18.11 seconds maximum per output frame;
- 170,131 initial particles and 598,436 peak particles after adaptive refinement;
- 276,342 peak fluid particles and 322,094 peak structural particles;
- 1,932 MiB peak reported VRAM use;
- 343 released architectural fragments;
- 40,582-44,435 late water-mesh vertices after frame 85;
- a constant 0.65 m water voxel from frame 0 through frame 99;
- zero water-mesh LOD changes and no late collapse to spherical spray;
- 21,952 m3 emitted from shallow water and 7,914 m3 returned from SPH;
- -0.070% combined 2D+3D water-volume drift over 4.1667 simulated seconds;
- 27.69 m maximum water height and zero water particles above 30 m.

V3.11 reduces the physically complete neighbour radius from 1.8 to 1.55 structural spacings. The longest possible bond is 3.2 times a 0.48-spacing particle radius, or 1.536 spacings, so no valid bond is clipped. On checkpoint 96 this reduced a substep from 49.30 to 40.65 ms (17.5%). Real four-view frames 98-100 fell from 17.70 to 14.67 seconds on average (17.1%) while retaining 343 released fragments and changing final damaged-particle count by only 0.06%.

The V3.4 wide-scene validation used a 420 m domain and 45 buildings:

- 508,747 initial particles;
- 19,320 shallow-water cells;
- about 1.64 GiB reported VRAM use.

The shallow-water regression measured 0.249% volume drift after one simulated second. Both the SPH↔2D exchange impulse and the shallow-to-SPH emission volume/momentum tests have zero float32 residual.

## GPU neighbour optimizations

The production solver caches the SPH equation-of-state pressure, inverse
density, `mass / density`, and `pressure / density^2` once per active particle.
The force pass therefore avoids repeating divisions for every neighbour.

Fluid neighbours are stored in a GPU CSR Verlet list. A 0.30 m halo makes the
list safe for twelve substeps at the configured velocity limits. Rebuilding is
performed with GPU count, prefix-scan, and fill passes. Periodic rebuilds keep
their final entry count on the GPU, so only a particle-topology change requires
a CPU readback. A 35% allocation reserve absorbs ordinary neighbour-count
variation; overflow is bounds-safe and schedules a larger rebuild at the next
frame boundary. The density and force passes launch over a spatially sorted
fluid-only list rather than all city particles.

Radius-dependent SPH support, squared support, Poly6, Spiky, and viscosity
coefficient precomputation is implemented under `v3.sph_kernel_coefficients`.
It is disabled on the RTX 5070: the extra irregular VRAM reads cost more than
recomputing the small kernels. The 763,784-particle late checkpoint's Verlet
list contains about 22.5 million live indices for 92,905 fluid particles; the
current reserved capacity is about 30.4 million indices (116 MiB).

An experimental deformable-fragment BVH is available under
`v3.deformable_fragment_bvh`, but is disabled in the production configuration.
An initial fast gate incorrectly culled isolated released fragments, which still
require non-bonded self-contact when folding; the first-substep trajectory A/B
caught a 0.72 m/s velocity error. The corrected conservative gate agrees within
9.6e-7 m/s, but 2,057 of 2,058 active fragments remain candidates and the BVH
raises the late-checkpoint substep from 16.48 to 17.01 ms. The correct slower
variant is retained for future pair-list work, while production uses HashGrid.

With the two positive changes enabled (asynchronous repeated rebuilds and the
spatial fluid-only launch), a 100-substep run from the 763,784-particle late
checkpoint averages 15.78 ms/substep versus 16.03 ms before this pass (1.6%
faster). All tracked state remains finite, and the 22.49-million-entry list does
not overflow its reserve.

## Prepared high-impact solver paths

Three larger changes are now isolated behind disabled production flags so they
can be audited from identical checkpoints before changing the established
physics.

- `v3.implicit_fluid` now contains an executable constant-density projection in
  addition to its CFL diagnostic. The constraint denominator and symmetric
  pressure correction use each particle's real mass and density reference, so
  the adaptive 1:8 mass ratio is not treated as equal-volume SPH. Execution is
  selected explicitly with `mode: density_projection`; the checked-in
  `mode: diagnostic` never changes the WCSPH integrator.
  The experimental path also contains a divergence projection and a fully
  device-side high-compression work list. The GPU expands selected particles
  by one neighbour ring before compacting their fluid slots, so a pressure
  correction cannot stop abruptly at the threshold boundary. Both stages stay
  subordinate to the disabled top-level flag.
- `v3.rigid_clusters.early_rigidification` is a complete switchable transition
  path. It scans detached fragments more frequently, requires independent
  detached and quiet histories, keeps foundation-supported fragments
  deformable, preserves the fitted centre-of-mass/angular motion and existing
  impact reactivation, and checkpoints both histories and conversion totals.
- `v3.rigid_clusters.sleeping` adds a second rigid state for grounded quiet
  proxies. Sleeping bodies remain in collision broad phases but skip rigid
  integration; impacts or strong external loading wake them as rigid bodies,
  while the higher existing threshold still expands an active proxy back to
  deformable particles. Sleep counters and transition totals are checkpointed.
- `v3.narrow_band_volume` prepares a conservative 3D coarse-volume grid. Its
  GPU audit keeps every particle close to a free surface, solid boundary, or
  local velocity shear in detailed SPH and deposits only coherently moving
  calm connected interior samples. Deposited
  mass, volume, three-axis momentum, active cells, and removable-particle ratio
  are measured without deleting SPH. This diagnostic transfer must balance
  exactly before a grid pressure/advection solve replaces the interior.

The preparation code lives in `experimental_optimizations.py`; all prepared flags
remain false in `config_v3_rtx5070.json`, so current production runs are
bit-for-bit unaffected by merely adding these paths.

### Optimization checkpoint results (2026-08-03)

`benchmark_implicit_projection.py` compares equal physical horizons from the
same early and late production checkpoints. With four pressure iterations and
`dt=0.0006` (5x the WCSPH step), a 0.06 s late-checkpoint test is 1.57x faster
per simulated second. The final density p99 is 1.0017 and all tracked state is
finite. The early checkpoint is approximately performance-neutral (1.02x).
The solver remains disabled because one late local projection residual reached
30.9%, trajectories differ materially from WCSPH, and solid-reaction/damage
equivalence has not passed a production-length run.

`benchmark_divergence_one_second.py` is the promotion gate for the combined
density/divergence solve. It recompiles on a disposable solver, reloads the
checkpoint for measurement, advances both methods across identical output-frame
boundaries without rendering, and records full-frame time, structural response,
and combined SPH plus shallow-water volume. This avoids the unequal warm-up
offset present in older microbenchmarks.

The 2026-08-04 one-second late-checkpoint gate at `dt=0.000595238` remains a
rejection, not a production result. The core/halo selective solve is 1.87x
faster in physics and 1.77x faster per complete no-render frame, stays finite,
and preserves combined water volume to `1.17e-8`. It nevertheless raises final
fluid height p99 from 7.32 m to 12.40 m, changes longitudinal fluid momentum,
and releases 58 fewer cohesive fragments. The checked-in top-level implicit
flag therefore remains disabled while 2x/3x timestep gates are evaluated.
The subsequent 0.06012 s `3x dt` checkpoint sweep is also a rejection: the
late-stage speedup is only 0.99x and structural position RMS still differs by
0.25 m. A `2x` step cannot recover performance with the same six projection
iterations, so the next fluid optimization target is the conservative
narrow-band SPH/grid split rather than enabling this DFSPH path.

`benchmark_early_rigidification.py` converts 343 additional detached clusters
(159,684 particles) in the late checkpoint, but does not reduce total substep
time. Most of those fragments were already rejected by the contact-candidate
gate. A direct equal-and-opposite atomic reaction prototype was also rejected:
contention on a small number of rigid bodies increased the deformable-contact
kernel from about 4.8 ms to about 41 ms. The next viable implementation is a
particle-to-OBB BVH narrowphase or a block-local force reduction.

`benchmark_narrow_band_sweep.py` also caught and fixed an undersized diagnostic
HashGrid. With the corrected grid, a 1.5 m detail band and 3 m/s local RMS limit
classify 21.6% of early water but only 0.8% of late turbulent water as safe
interior. A 1.0 m band classifies 38.6% early and 4.2% late. Deposited mass and
volume match exactly; aggregate momentum differs only by float32 atomic
rounding. Consequently the volume-grid path can help calm approach water, but
cannot yet deliver a 1.3--1.8x late-stage speedup on its own.

Intact city buildings expose only their exterior facades, perimeter frame,
roofs, and terraces to the fluid solver. Interior apartment walls and floor
plates become hydraulic boundaries after local damage or rigid-fragment
release. This avoids paying for hidden solid neighbours without making broken
buildings watertight. Checkpoint density references are renormalized so enabling
the exterior layer cannot create a pressure impulse.

An optional spatially compact dynamic-solid contact list is implemented under
`v3.dynamic_solid_contact_list`. It remains disabled in the RTX 5070 production
configuration: A/B profiling found that prefix-scan overhead canceled its small
contact-kernel saving at the current active-solid fraction.

## Water representation

The local impact region is simulated with 3D SPH particles. Only classified free-surface particles contribute to rendering. A compact GPU scalar field is smoothed and reconstructed with Warp Marching Cubes.

Sparse distant droplets do not expand the global reconstruction box. Bounds expand immediately but shrink gradually; improved voxel LOD requires eight stable frames. Dense splash sheets outside the main water body receive up to six local 12 m mesh bricks with a 0.4 m voxel, while isolated droplets stay inexpensive anisotropic spray samples.

The shallow-water far field and the local SPH surface are now reconstructed as one mesh. Graphical samples from the 2D field are smoothly blended to the robust SPH free-surface height in the overlap, then splatted into the same scalar field before Marching Cubes. There is no second water plane at another height.

Empty interface sites can emit SPH particles from the shallow field. Every emitted particle removes the same volume and horizontal momentum from its source 2D cell in the same rendered frame, including checkpoint boundaries.

Return flow is symmetric. SPH particles crossing back through the overlap give their exact volume and horizontal momentum to the shallow field. A GPU prefix scan then compacts every particle-aligned array, including fracture, rigid-body, multirate, and facade-anchor indices. The return-flow regression has zero volume and momentum residual.

## Structural model

Buildings begin as inexpensive fixed SPH boundaries. A building activates only after a coherent hydrodynamic load reaches enough facade samples. Foundations remain anchored.

Each building contains physically sampled floor slabs, roofs, walls, beams, columns, cores, and glass. Structural particles are grouped into cohesive architectural fragments approximately 4.5×3×4.5 m. Joints between fragments can fail, but the fragment interior remains connected, so failure produces apartment-scale slabs and wall sections instead of independent particle dust.

Failure strain and damage rate depend on structural role. Glass is deliberately fragile, ordinary walls are the baseline, and resistance increases through slabs, beams, columns, and reinforced cores. Metrics are volume-weighted so adaptive 1-to-4 refinement cannot fake an increase in damaged structure merely by creating more samples.

Detached calm fragments can become rigid clusters. Their mass, center of mass, inertia tensor, linear momentum, and angular momentum are fitted in double precision. Rigid motion and contacts then run on CUDA. Concrete, glass, and reinforcement use different contact stiffness and friction. A sufficiently strong later collision reactivates the cohesive deformable model.

## Validation

Run the complete CUDA regression batch:

```bat
validation\validate_v3_gpu.bat
```

Validate a completed 100-frame production directory:

```bat
.venv\Scripts\python.exe validation\validate_production_output.py outputs\your_run --expected-frames 100
```

It covers:

- adaptive refinement and conservation of mass, volume, momentum, and center of mass;
- rigid fitting, motion, contact force/torque balance, friction, and reactivation;
- multirate momentum and city-state agreement;
- architectural fragment scale and the glass-to-core fracture hierarchy;
- deterministic damage-driven facade cracks, including earlier brittle-glass cracking;
- shallow-water volume and conservative overlap impulse;
- conservative shallow-to-SPH emission and SPH-to-shallow return-flow compaction;
- free-surface classification and reconstructed top-layer closure;
- robust late mesh bounds, temporal hysteresis, and splash-brick selection;
- physical building-style diversity and facade palettes;
- early opening and finalization of progressive NVENC MP4 output.

## Important files

- `deluge_v3.py` — V3 orchestration, checkpoints, surface reconstruction, and simulation loop integration.
- `solver_base.py` — shared particle solver and output loop in the standalone repository.
- `shallow_water.py` — GPU shallow-water solver and conservative SPH interface coupling.
- `validation\validate_production_output.py` — MP4, water-balance, late-mesh, and metric validation.
- `assemble_resumed_run.py` — lossless MP4 and metric assembly for checkpoint-resumed runs.
- `validation\validate_shallow_return.py` — end-to-end return-flow, compaction, and facade-anchor validation.
- `validation\validate_fragment_scale.py` — apartment-scale anti-dust fragment validation.
- `validation\validate_structural_hierarchy.py` — CUDA fracture-resistance hierarchy validation.
- `profile_v3_kernels.py` — per-kernel CUDA timing at a fresh scene or checkpoint.
- `hybrid_kernels.py` — structural LOD, fracture, multirate, rigid-body, contact, and facade CUDA kernels.
- `hybrid_model.py` — cohesive fragments, refinement axes, and facade generation.
- `hybrid_renderer.py` — facade and reconstructed-water rendering.
- `surface_kernels.py` — free-surface classification, sparse fields, temporal blending, and water rasterization.
- `config_v3_rtx5070.json` — production RTX 5070 configuration.
- `ROADMAP.md` — development stages and acceptance criteria.

## Status

V3.3d mesh stability, V3.4 shallow-water far field, V3.5 debris contacts, V3.6 progressive output, the 100-frame V3.7 stitched-water run, and V3.8 bidirectional water transfer are implemented and validated. The duplicated far-water plane is removed, shallow and SPH samples feed one reconstructed surface, and a playable MP4 is exposed every completed simulated second.

At checkpoint 96, real return flow merged 618 particles / 593.5 m3 over three frames while preserving facade bindings. The contour audit measured only 32.9 ms for the complete water-mesh build versus seconds for a late output frame, so tiled Marching Cubes remains lower priority. A separate solid-only grid was tested and rejected because its extra indirection made the substep slower. The physically complete 1.55-spacing query radius was retained after CUDA regressions and a resumed production comparison showed a 17.1% frame-time improvement.

The complete eight-second validation now passes. The assembled four-view H.264 video contains 192 frames at 1920x1080 and 24 fps. The water reconstruction remained at a 0.65 m voxel for every frame, used at most 5,121,225 field nodes, and retained 34,956-42,873 late mesh vertices. Peak load was 928,147 particles and 1,995.6 MiB VRAM; combined shallow/SPH water-volume drift was -0.750%. The main field limit is 6 million nodes, while detached water above 42 m is routed to local splash bricks so high spray cannot coarsen the complete connected surface.

The next milestone is structural fracture calibration: preserve larger slabs, wall panels, floor spans, and reinforced cores instead of producing excessive fine debris during the first impact.

## V3.9 impact-energy and activation guards

The first V3.8 full-run review found a non-physical WCSPH tail: checkpoint 144 contained water moving vertically at up to 129.6 m/s, with particles reaching the 90 m domain ceiling. The configured 14 m depth, 6 m crest, and 19 m/s incoming flow provide an ideal total energy head equivalent to only about 27.4 m/s. V3.9 therefore limits water to 30 m/s total and 18 m/s vertically; the ordinary incoming flow remains untouched.

Building activation now requires at least 12 lower-facade samples (rest elevation at most 8 m) to carry a +Z load above 5 m/s2 for 0.02 continuous seconds. Exposure decays four times faster when the load disappears. High, lateral, reverse, or momentary splash impacts no longer unlock the complete deformable building graph.

In the clean 60-frame / 2.5-second impact validation, the first row activated at 0.875 s, with zero active buildings and zero damage beforehand. The second and third rows remained dormant. Peak water height was 22.01 m; the final 99th and 99.9th percentiles were 15.36 m and 17.03 m, and no water particle exceeded 30 m. `validation\validate_impact_gates.py` covers the CUDA speed and sustained-load gates.

## V3.10 architectural fracture and volume diagnostics

The coarse city now contains 2,991 apartment-scale cohesive fragments with a median of 20 coarse samples. Minimum fragment and release thresholds were raised, intra-fragment stiffness was increased, and fracture propagation was slowed. In the clean 60-frame impact test, released fragments fell from 357 in V3.9 to 115, a 67.8% reduction.

Diagnostics report damaged physical volume and volume-integrated damage per structural role, in addition to legacy particle counts. At 2.5 seconds, damage began one frame after causal activation; columns had 2.08% damaged volume and cores 2.57%, while more exposed walls, beams, and glass reached 3.17%, 4.50%, and 4.44%. Adaptive refinement therefore no longer corrupts the comparison by multiplying sample counts.

## V3.14-V3.17 causal collapse, adaptive water, and debris skins

V3.14 propagates gravity release through a sparse fragment support graph. Upper floors and walls fall when their load path to a foundation is actually broken; the solver no longer relies on a delayed whole-building collapse patch. V3.15 adds material-specific local impact impulse. A sufficiently massive splash can break glass locally, while complete building activation still requires a sustained coherent lower-facade water load.

V3.16 refines only free-surface or strongly vertical/turbulent SPH samples. Every 1-to-8 split records a sibling ID. A complete octet may merge back only below the surface band when its vertical speed, internal velocity RMS, and spatial span are all small. The replacement preserves total mass, volume, center of mass, and linear momentum. In the 100-frame RTX 5070 validation, 1,855 calm octets merged back, removing 12,985 fine samples without changing the fixed 0.65 m water-mesh voxel.

V3.17 adds six-face render hulls for all 3,060 apartment-scale cohesive fragments. The 18,360 extra faces are culled before rasterization while their fragment has a live foundation path. When support is lost, the closed hull becomes visible with the original building palette, so detached concrete, slabs, and frame pieces remain large volumetric debris instead of reverting to particle circles.

The V3.17 100-frame four-view run completed in 269.75 seconds on an RTX 5070. It peaked at 409,831 particles, 1,963.6 MiB VRAM, 251 released fragments, 36 unsupported fragments, and 19 rigid clusters. The connected water mesh stayed at 0.65 m and reached 56,868 vertices; combined shallow/SPH water-volume drift was -0.813%.

## V3.18 convex debris and fast checkpoint resume

The axis-aligned debris boxes are replaced by cached convex render hulls built from each fragment's actual structural support points. SciPy/Qhull generated 61,936 watertight triangles for all 3,060 coarse fragments; a minimal installation without SciPy keeps the previous safe box fallback. Hidden hull vertices are no longer deformed until their fragment loses foundation support, so intact buildings do not pay the full extra render cost.

Facade/debris geometry is compressed into an NPZ cache keyed by the physical city layout, fragmentation topology, and source particle state. Repeated fresh-scene startup fell from 31.6 to 9.1 seconds in the validation environment. Anchor binding uses compiled KD-tree queries and fragment-grouped particle indices instead of repeated Python-wide scans.

New V3 checkpoints also store the sparse support graph: 21,508 fragment edges, 133,377 representative boundary samples, rest lengths, anchored fragments, and current support/intact state. In the round-trip test, initialization of the same scene fell from 19.88 seconds to 0.735 seconds. Older checkpoints remain readable and rebuild the graph once through the legacy fallback.

## V3.19 visible progressive cracking

Facade panels now expose deterministic procedural cracks before their cohesive fragment separates. Glass starts showing hairline damage at a lower threshold than concrete; floor and roof plates remain visually intact longer. The renderer uses the maximum damage at the four panel anchors for crack initiation and the average damage for broad material darkening, so a local impact can create a local crack without recolouring the entire building.

Cracks begin as two thin rays. Further rays, branches, and a small high-damage chip appear progressively and vary by panel, avoiding a synchronized four-point pattern. The effect is generated during GPU rasterization from rest-space panel coordinates and does not add texture files, facade geometry, physics particles, or checkpoint data. Convex debris skins retain their building palette and do not receive the architectural decal.

`validation\validate_crack_rendering.py` checks the material thresholds, deterministic output, and bounded coverage. At 0.28 concrete damage, the test pattern covers 354 of 9,216 samples (3.8%); intact concrete has zero crack pixels while brittle glass already shows hairlines at 0.04 damage.

## V3.20 fracture-energy-driven crack opening

Visible cracks are now coupled to the sparse physical boundaries between cohesive architectural fragments. For every representative inter-fragment bond, the solver compares current and rest length and evaluates the tensile spring-energy fraction against the same material and structural-role failure envelope used by the force kernel. Hairlines begin at 35% of the failure energy, before complete separation. Boundary energy is blended from its peak and mean samples, so a real local crack front remains visible without letting one noisy bond paint the complete interface.

This state is irreversible: removing the load cannot visually heal a crack. A failed support edge is forced to full crack energy. Each facade panel receives the maximum energy of boundaries incident to its owning fragment, while anchor damage still permits more local brittle-glass impact cracks. The evaluation is vectorized over the 133,377 representative samples and does not traverse the full particle neighbour graph a second time.

Both edge and fragment crack states are stored in compressed V3 checkpoints. The end-to-end regression preserved all 21,508 production edge values exactly across save and resume while the V3 checkpoint remained about 0.6 MiB. New metrics report visible-energy edge count and maximum normalized fracture energy. A calm smoke scene reports zero visible edges, zero maximum energy, and zero damaged particles.

## V3.21 rigid-debris collision proxies

A released fragment that has passed the existing quiet-motion audit and becomes a rigid cluster now receives an eight-vertex convex oriented-box collision proxy fitted in body-local coordinates. The proxy includes particle radius padding and keeps the mass-dominant concrete, glass, or reinforcement contact material. A 15-axis separating-axis test resolves proxy-to-proxy overlap on CUDA; equal/opposite normal and Coulomb-friction forces are applied at one shared contact point, preserving net force and world-space torque.

The proxy replaces only rigid-to-rigid and rigid-to-domain particle contacts. The underlying structural particles remain in the hash grid and continue to collect pressure, drag, and buoyant loading from water, as well as contacts with deformable fragments. A strong later proxy collision uses the existing acceleration threshold to restore the complete deformable particle model. This prevents a coarse box from permanently replacing fracture physics.

Ground and domain contacts use projected OBB radius instead of accumulating one penalty force per rigid particle. Pair tables contain only proxy bodies, so 19 settled fragments require 171 SAT tests rather than a city-wide all-body quadratic launch. Proxy geometry and material are checkpointed; old checkpoints reconstruct proxies from saved rigid local particles. `validation\validate_rigid_collision_proxy.py` covers enclosure, SAT force/torque conservation, friction, ground support, and deformable reactivation.

The durable late production checkpoint-96 A/B summary is stored in
`validation\collision_proxy_checkpoint96_summary.json` (the complete local
kernel profile remains in `outputs\v3_21_proxy_ab_checkpoint96_20260802\comparison.json`). With only one
rigid body and therefore zero rigid-proxy pairs, it is an intentionally
unfavourable but honest case: contact work fell from 0.2972 to 0.2850 ms per
substep (4.3%), while the complete four-view output frame changed from 8.701
to 8.760 seconds (-0.67%). Solid-position RMS was 0.017 mm and the maximum
individual difference was 8.9 mm. This does not demonstrate a production-frame
speedup; the proxy remains useful only when many debris bodies contact each
other or the domain.

## V3.22 separated water representations

Water surface samples now carry an explicit GPU phase: connected body, thin
sheet, or ballistic droplet. The classifier measures local support thickness
along the reconstructed surface normal, rather than treating every sparse
surface particle as spray. Entering and leaving ballistic mode requires four
and two consecutive output classifications respectively, preventing force-mode
chatter. A rejoining droplet requests a fresh SPH density normalization.

Connected particles alone define the global Marching Cubes field. Coherent
thin sheets and connected outliers use bounded local reconstruction bricks;
ballistic drops never enlarge the global active bounding box. Thin sheets are
rendered with a broad tangential and narrow normal footprint. Droplets are
velocity-aligned streaks and use gravity plus solid-contact reaction instead of
bulk SPH pressure. No particles are created or deleted by a phase transition,
so mass, position, velocity, and momentum are preserved exactly.

Foam is a fourth, render-only field generated by vorticity, overturning, and
energetic detached spray. It decays over successive classifications and never
adds fluid mass or collision force. Phase, transition history, and foam lifetime
are stored in compressed V3 checkpoints and survive adaptive compaction.
Per-frame and cumulative entry/rejoin counters expose phase chatter in long
runs instead of hiding it behind the instantaneous particle counts.
`validation\validate_water_phase_separation.py` covers phase hysteresis,
ballistic gravity, exact mass/momentum preservation, SPH re-entry, and foam.
On the late checkpoint-96 audit, the phase-aware renderer classified 60,641
connected surface particles, 15 sheet particles, and 825 ballistic droplets.

## Next stages

1. Calibrate sheet thickness, droplet neighbour count, and foam decay on a focused impact sequence, then compare the same late views against V3.21.
2. Move conservative sibling-group selection fully onto CUDA. The current CPU audit runs only every eight output frames and is already small, but a sorted GPU group table will scale better beyond one million active SPH samples.
3. Run a many-body debris checkpoint A/B; the one-body checkpoint-96 test cannot exercise proxy-pair acceleration.
4. Calibrate the 35% crack-energy onset and proxy padding against visible panel opening and settled rubble contacts.
5. Run the complete eight-second V3.22 sequence and audit phase counts, water balance, bounding-box stability, crack timing, debris contacts, checkpoint resume, and progressive-video recovery.
