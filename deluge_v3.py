"""DELUGE V3 hybrid solver.

The shared CUDA water/fracture kernels are packaged with V3. Structural LOD
keeps dormant buildings as cheap fixed boundaries until a coherent
hydrodynamic load activates their deformable bond graph.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent

from solver_base import DelugeSolver  # noqa: E402
from kernels import (  # noqa: E402
    clear_int,
    clear_vec3,
    compute_density,
    compute_fluid_forces,
    integrate,
)
from hybrid_kernels import (  # noqa: E402
    accumulate_rigid_body_loads,
    activate_buildings_from_hits,
    apply_building_activity,
    clear_body_accumulators,
    classify_time_levels,
    compute_clustered_solid_forces,
    compute_density_multirate,
    compute_fluid_forces_multirate,
    consume_deferred_fluid_impulse,
    count_loaded_building_particles,
    integrate_rigid_bodies,
    integrate_multirate,
    mask_rigid_particles_as_fixed,
    refine_impacted_solids,
    scatter_rigid_particles,
    select_active_time_level,
)
from hybrid_model import (  # noqa: E402
    SolidRefinementPolicy,
    build_fragment_ids,
    build_refinement_axes,
    write_facade_skin,
)
from hybrid_renderer import HybridRenderer  # noqa: E402
from rigid_clusters import fit_rigid_cluster  # noqa: E402
from surface_kernels import (  # noqa: E402
    classify_water_surface,
    smooth_sparse_field_axis,
    splat_sparse_surface_field,
)


def robust_axis_bounds(positions: np.ndarray, lower_quantile, upper_quantile):
    """Axis-wise bounds that ignore sparse spray without cropping the core body."""
    lower_quantile = np.broadcast_to(np.asarray(lower_quantile, dtype=np.float64), (3,))
    upper_quantile = np.broadcast_to(np.asarray(upper_quantile, dtype=np.float64), (3,))
    if np.any(lower_quantile < 0.0) or np.any(upper_quantile > 1.0) or np.any(lower_quantile >= upper_quantile):
        raise ValueError("water_mesh bbox quantiles must satisfy 0 <= lower < upper <= 1 per axis")
    lower = np.asarray(
        [np.quantile(positions[:, axis], lower_quantile[axis]) for axis in range(3)], dtype=np.float32
    )
    upper = np.asarray(
        [np.quantile(positions[:, axis], upper_quantile[axis]) for axis in range(3)], dtype=np.float32
    )
    return lower, upper


class HybridDelugeSolver(DelugeSolver):
    def __init__(self, cfg: dict, output: Path, resume: Path | None = None):
        self.resume_path = Path(resume) if resume else None
        runtime_cfg = copy.deepcopy(cfg)
        refinement_cfg = runtime_cfg["v3"]["solid_refinement"]
        if bool(refinement_cfg.get("start_coarse", False)):
            runtime_cfg["solid_spacing"] = float(refinement_cfg["coarse_spacing"])
        super().__init__(runtime_cfg, output, resume)
        cfg = runtime_cfg
        self.v3_cfg = cfg["v3"]
        self.building_count = len(cfg["buildings"])
        self.refinement_policy = SolidRefinementPolicy.from_config(cfg)

        kind_host = self.arrays["kind"][:self.count].numpy()
        building_host = self.arrays["building_id"][:self.count].numpy()
        rest_host = self.arrays["rest_x"][:self.count].numpy()
        structural_class_host = self.arrays["structural_class"][:self.count].numpy()
        fragment_host, fragment_counts = build_fragment_ids(rest_host, kind_host, building_host, cfg)
        self.fragment_host = fragment_host
        self.fragment_counts_host = fragment_counts
        fragment_capacity = np.full(self.capacity, -1, dtype=np.int32)
        fragment_capacity[:self.count] = fragment_host
        self.fragment_id = wp.array(fragment_capacity, dtype=wp.int32, device=self.device)
        self.fragment_counts = wp.array(fragment_counts, dtype=wp.int32, device=self.device)
        self.fragment_count = len(fragment_counts)
        normal_host = build_refinement_axes(
            rest_host, kind_host, building_host, float(cfg["solid_spacing"]), structural_class_host
        )
        normal_capacity = np.full(self.capacity, -1, dtype=np.int32)
        normal_capacity[:self.count] = normal_host
        self.normal_axis = wp.array(normal_capacity, dtype=wp.int32, device=self.device)

        base_host = np.zeros(self.capacity, dtype=np.int32)
        base_host[:self.count] = self.arrays["fixed"][:self.count].numpy()
        active_host = np.zeros(max(1, self.building_count), dtype=np.int32)
        v3_resume = self._v3_checkpoint_path(self.resume_path) if self.resume_path else None
        if v3_resume is not None and v3_resume.exists():
            with np.load(v3_resume, allow_pickle=False) as state:
                saved_base = state["base_fixed"]
                base_host[:len(saved_base)] = saved_base
                saved_active = state["building_active"]
                active_host[:min(len(saved_active), len(active_host))] = saved_active[:len(active_host)]
                if "fragment_id" in state:
                    fragment_host = state["fragment_id"].astype(np.int32, copy=True)
                    fragment_capacity.fill(-1)
                    fragment_capacity[:len(fragment_host)] = fragment_host
                    valid_fragments = fragment_host[fragment_host >= 0]
                    self.fragment_count = int(valid_fragments.max()) + 1 if len(valid_fragments) else 0
                    fragment_counts = np.bincount(
                        valid_fragments, minlength=self.fragment_count
                    ).astype(np.int32)
                    self.fragment_host = fragment_host
                    self.fragment_counts_host = fragment_counts
                    self.fragment_id = wp.array(fragment_capacity, dtype=wp.int32, device=self.device)
                    self.fragment_counts = wp.array(fragment_counts, dtype=wp.int32, device=self.device)
                if "normal_axis" in state:
                    saved_normal = state["normal_axis"].astype(np.int32, copy=False)
                    normal_capacity.fill(-1)
                    normal_capacity[:len(saved_normal)] = saved_normal
                    self.normal_axis = wp.array(normal_capacity, dtype=wp.int32, device=self.device)
            print(f"V3 state restored from {v3_resume.name}")
        else:
            for bid in self.v3_cfg.get("initially_active_buildings", []):
                if 0 <= int(bid) < self.building_count:
                    active_host[int(bid)] = 1
            if self.resume_path is not None:
                print("WARNING: matching V3 checkpoint is absent; using configured building activity")

        self.base_fixed = wp.array(base_host, dtype=wp.int32, device=self.device)
        self.building_active = wp.array(active_host, dtype=wp.int32, device=self.device)
        self.activation_hits = wp.zeros(max(1, self.building_count), dtype=wp.int32, device=self.device)
        self.last_active_count = int(np.count_nonzero(active_host))
        self.last_released_fragment_count = 0
        self.last_preimpact_building_count = 0
        self.preimpact_building = wp.zeros(max(1, self.building_count), dtype=wp.int32, device=self.device)
        self.refinement_counters = wp.zeros(7, dtype=wp.int32, device=self.device)
        self._initialize_rigid_clusters(v3_resume)
        self._initialize_multirate()
        self._initialize_water_surface()

        wp.launch(
            apply_building_activity, dim=self.count,
            inputs=[self.arrays["kind"][:self.count], self.arrays["building_id"][:self.count],
                    self.base_fixed[:self.count], self.building_active, self.arrays["fixed"][:self.count]],
            device=self.device,
        )
        skin_path = output / "facade_skin.npz"
        panel_count = write_facade_skin(
            skin_path, cfg, rest_host, kind_host, building_host, fragment_host
        )
        render = cfg["render"]
        configured_views = render.get("views", {"original": render["camera"]})
        view_width = int(render.get("view_width", render["width"]))
        view_height = int(render.get("view_height", render["height"]))
        self.renderers = {
            str(name): HybridRenderer(
                view_width, view_height, camera, self.device, skin_path, str(name),
                float(render.get("maximum_panel_stretch", 1.8)),
                float(self.v3_cfg.get("water_surface", {}).get("tangent_scale", 2.8)),
                float(self.v3_cfg.get("water_surface", {}).get("normal_scale", 2.45)),
            )
            for name, camera in configured_views.items()
        }
        self.renderer = next(iter(self.renderers.values()))
        dormant_count = self.building_count - int(np.count_nonzero(active_host))
        smallest = int(fragment_counts.min()) if len(fragment_counts) else 0
        largest = int(fragment_counts.max()) if len(fragment_counts) else 0
        print(
            f"V3 structural LOD: {dormant_count}/{self.building_count} dormant buildings; "
            f"cohesive fragments: {self.fragment_count:,} ({smallest}-{largest} particles); "
            f"facade skin: {panel_count:,} panels; views: {', '.join(self.renderers)}"
        )

    def _initialize_multirate(self):
        self.multirate_cfg = self.v3_cfg.get("multirate", {})
        self.multirate_enabled = bool(self.multirate_cfg.get("enabled", False))
        self.time_level = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.time_active = wp.ones(self.capacity, dtype=wp.int32, device=self.device)
        self.deferred_fluid_impulse = wp.zeros((self.capacity, 3), dtype=float, device=self.device)
        self.multirate_tick = 0
        if self.multirate_enabled:
            # Every resolution level must have a calibrated density before a
            # fast neighbour can read it.  Leaving sleeping particles at rho=0
            # makes the first pressure denominator singular and collapses the
            # hash grid into catastrophically dense clumps.
            initial_view = self.arrays["x"][:self.count]
            self.grid.build(initial_view, self.max_support)
            wp.launch(
                compute_density, dim=self.count,
                inputs=[self.grid.id, initial_view, self.arrays["radius"][:self.count],
                        self.arrays["mass"][:self.count], self.arrays["volume"][:self.count],
                        self.arrays["kind"][:self.count], self.arrays["rho"][:self.count],
                        self.arrays["rho_reference"][:self.count], float(self.cfg["rest_density"]),
                        float(self.cfg["sound_speed"]), float(self.cfg["water_depth"]),
                        float(self.cfg["wave_height"]), float(self.cfg["reservoir_z_max"]),
                        self.max_support], device=self.device,
            )
            wp.launch(
                classify_time_levels, dim=self.count,
                inputs=[self.arrays["radius"][:self.count], self.arrays["v"][:self.count],
                        self.arrays["kind"][:self.count], self.arrays["damage"][:self.count],
                        self.time_level[:self.count],
                        float(self.multirate_cfg.get("fine_radius", 0.30)),
                        float(self.multirate_cfg.get("active_speed", 8.0)),
                        float(self.multirate_cfg.get("active_damage", 0.02))],
                device=self.device,
            )

    def _initialize_water_surface(self):
        self.surface_cfg = self.v3_cfg.get("water_surface", {})
        self.surface_enabled = bool(self.surface_cfg.get("enabled", True))
        self.arrays["water_surface_mask"] = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.arrays["water_surface_normal"] = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.arrays["water_foam_strength"] = wp.zeros(self.capacity, dtype=float, device=self.device)
        self.water_mesh_cfg = self.v3_cfg.get("water_mesh", {})
        self.water_mesh_enabled = bool(self.water_mesh_cfg.get("enabled", True))
        self.water_mesh_frame = 0
        self.water_mesh_vertices = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self.water_mesh_indices = wp.zeros(0, dtype=wp.int32, device=self.device)
        self.water_mesh_triangle_count = 0
        self.water_mesh_voxel_size = 0.0
        self.water_field_shape = (0, 0, 0)
        self.water_mesh_excluded_surface_count = 0

    def update_water_surface(self):
        if not self.surface_enabled:
            return
        view = self.arrays["x"][:self.count]
        query_radius = max(
            self.max_support,
            float(self.cfg.get("coarse_spacing", 1.0)) * float(self.surface_cfg.get("query_scale", 2.5)),
        )
        self.grid.build(view, query_radius)
        wp.launch(
            classify_water_surface, dim=self.count,
            inputs=[self.grid.id, view, self.arrays["v"][:self.count], self.arrays["radius"][:self.count],
                    self.arrays["kind"][:self.count], self.arrays["water_surface_mask"][:self.count],
                    self.arrays["water_surface_normal"][:self.count],
                    self.arrays["water_foam_strength"][:self.count], query_radius,
                    int(self.surface_cfg.get("minimum_neighbours", 18))], device=self.device,
        )
        if self.water_mesh_enabled:
            self._build_water_mesh()

    def _build_water_mesh(self):
        rebuild_every = max(1, int(self.water_mesh_cfg.get("rebuild_every_frames", 1)))
        if self.water_mesh_frame % rebuild_every != 0:
            self.water_mesh_frame += 1
            return
        self.water_mesh_frame += 1
        mask = self.arrays["water_surface_mask"][:self.count].numpy() != 0
        kind = self.arrays["kind"][:self.count].numpy()
        surface_indices = np.flatnonzero(mask & (kind == 0))
        if len(surface_indices) == 0:
            self.water_mesh_vertices = wp.zeros(0, dtype=wp.vec3, device=self.device)
            self.water_mesh_indices = wp.zeros(0, dtype=wp.int32, device=self.device)
            self.water_mesh_triangle_count = 0
            return
        positions = self.arrays["x"][:self.count].numpy()[surface_indices]
        voxel = float(self.water_mesh_cfg.get("voxel_size", 0.65))
        max_nodes = int(self.water_mesh_cfg.get("maximum_field_nodes", 2500000))
        margin_cells = float(self.water_mesh_cfg.get("margin_cells", 3.0))

        # Detached spray must not determine the resolution of the connected
        # water body.  Late in an impact a handful of particles can be tens of
        # metres above/ahead of the wave; a raw min/max box then forces the
        # entire field to a very coarse voxel size.  Axis-specific robust
        # quantiles retain the reservoir width and rear boundary while
        # excluding only the extreme upper/forward spray.  Those particles
        # are still rendered by the anisotropic droplet layer.
        robust_minimum, robust_maximum = robust_axis_bounds(
            positions,
            self.water_mesh_cfg.get("bbox_lower_quantile", [0.0, 0.0, 0.0]),
            self.water_mesh_cfg.get("bbox_upper_quantile", [1.0, 0.995, 0.9975]),
        )
        excluded = np.any((positions < robust_minimum) | (positions > robust_maximum), axis=1)
        self.water_mesh_excluded_surface_count = int(np.count_nonzero(excluded))

        def quantized_domain(size: float):
            brick = size * 8.0
            lower = np.floor((robust_minimum - size * margin_cells) / brick) * brick
            upper = np.ceil((robust_maximum + size * margin_cells) / brick) * brick
            dims = np.maximum(3, np.rint((upper - lower) / size).astype(np.int32) + 1)
            upper = lower + (dims - 1) * size
            return lower.astype(np.float32), upper.astype(np.float32), tuple(int(v) for v in dims)

        lower, upper, shape = quantized_domain(voxel)
        while int(np.prod(shape, dtype=np.int64)) > max_nodes:
            voxel *= 1.25
            lower, upper, shape = quantized_domain(voxel)
        nx, ny, nz = shape
        field = wp.zeros(shape, dtype=float, device=self.device)
        wp.launch(
            splat_sparse_surface_field, dim=self.count,
            inputs=[self.arrays["x"][:self.count], self.arrays["radius"][:self.count],
                    self.arrays["kind"][:self.count], self.arrays["water_surface_mask"][:self.count],
                    field, wp.vec3(*lower), voxel, nx, ny, nz], device=self.device,
        )
        smoothing_iterations = max(0, int(self.water_mesh_cfg.get("field_smoothing_iterations", 1)))
        if smoothing_iterations:
            temporary = wp.zeros(shape, dtype=float, device=self.device)
            source, target = field, temporary
            for _ in range(smoothing_iterations):
                for axis in range(3):
                    wp.launch(
                        smooth_sparse_field_axis, dim=shape,
                        inputs=[source, target, nx, ny, nz, axis], device=self.device,
                    )
                    source, target = target, source
            field = source
        vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
            field,
            threshold=float(self.water_mesh_cfg.get("iso_threshold", 0.72)),
            domain_bounds_lower_corner=tuple(float(v) for v in lower),
            domain_bounds_upper_corner=tuple(float(v) for v in upper),
        )
        self.water_sparse_field = field
        self.water_mesh_vertices = vertices
        self.water_mesh_indices = indices
        self.water_mesh_triangle_count = len(indices) // 3
        self.water_mesh_voxel_size = voxel
        self.water_field_shape = shape
        self.arrays["water_mesh_vertices"] = vertices
        self.arrays["water_mesh_indices"] = indices

    @staticmethod
    def _v3_checkpoint_path(base_path: Path) -> Path:
        if base_path.name.startswith("state_"):
            return base_path.with_name("v3_" + base_path.name)
        return base_path.with_name("v3_state_" + base_path.name)

    def save_checkpoint(self, frame: int):
        super().save_checkpoint(frame)
        base_path = self.checkpoint_dir / f"state_{frame:05d}.npz"
        v3_path = self._v3_checkpoint_path(base_path)
        np.savez_compressed(
            v3_path,
            building_active=self.building_active.numpy(),
            base_fixed=self.base_fixed[:self.count].numpy(),
            fragment_id=self.fragment_id[:self.count].numpy(),
            normal_axis=self.normal_axis[:self.count].numpy(),
            rigid_state=self.rigid_state.numpy(),
            rigid_quiet_scans=self.rigid_quiet_scans_host,
            rigid_local_position=self.rigid_local_position[:self.count].numpy(),
            body_center=self.body_center.numpy(),
            body_orientation=self.body_orientation.numpy(),
            body_linear_velocity=self.body_linear_velocity.numpy(),
            body_angular_velocity=self.body_angular_velocity.numpy(),
            body_mass=self.body_mass.numpy(),
            body_inverse_inertia=self.body_inverse_inertia.numpy(),
        )
        print(f"  V3 checkpoint: {v3_path.name} ({v3_path.stat().st_size / 1024**2:.1f} MiB)")

    def _initialize_rigid_clusters(self, checkpoint: Path | None):
        body_capacity = max(1, self.fragment_count)
        state = np.zeros(body_capacity, dtype=np.int32)
        quiet = np.zeros(body_capacity, dtype=np.int32)
        local = np.zeros((self.capacity, 3), dtype=np.float32)
        center = np.zeros((body_capacity, 3), dtype=np.float32)
        orientation = np.zeros((body_capacity, 4), dtype=np.float32)
        orientation[:, 3] = 1.0
        linear_velocity = np.zeros((body_capacity, 3), dtype=np.float32)
        angular_velocity = np.zeros((body_capacity, 3), dtype=np.float32)
        body_mass = np.zeros(body_capacity, dtype=np.float32)
        inverse_inertia = np.zeros((body_capacity, 3, 3), dtype=np.float32)
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                def restore(name: str, target: np.ndarray):
                    if name not in saved:
                        return
                    source = saved[name]
                    length = min(len(source), len(target))
                    target[:length] = source[:length]
                restore("rigid_state", state)
                restore("rigid_quiet_scans", quiet)
                restore("rigid_local_position", local)
                restore("body_center", center)
                restore("body_orientation", orientation)
                restore("body_linear_velocity", linear_velocity)
                restore("body_angular_velocity", angular_velocity)
                restore("body_mass", body_mass)
                restore("body_inverse_inertia", inverse_inertia)

        self.rigid_state_host = state
        self.rigid_quiet_scans_host = quiet
        self.rigid_local_host = local
        self.rigid_state = wp.array(state, dtype=wp.int32, device=self.device)
        self.rigid_local_position = wp.array(local, dtype=wp.vec3, device=self.device)
        self.body_center = wp.array(center, dtype=wp.vec3, device=self.device)
        self.body_orientation = wp.array(orientation, dtype=wp.quat, device=self.device)
        self.body_linear_velocity = wp.array(linear_velocity, dtype=wp.vec3, device=self.device)
        self.body_angular_velocity = wp.array(angular_velocity, dtype=wp.vec3, device=self.device)
        self.body_mass = wp.array(body_mass, dtype=float, device=self.device)
        self.body_inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33, device=self.device)
        self.body_force = wp.zeros((body_capacity, 3), dtype=float, device=self.device)
        self.body_torque = wp.zeros((body_capacity, 3), dtype=float, device=self.device)
        self.rigid_stats_calls = 0
        self.last_rigid_count = int(np.count_nonzero(state))
        self.rigid_active_count = self.last_rigid_count

    def update_rigid_clusters(self):
        policy = self.v3_cfg.get("rigid_clusters", {})
        if not bool(policy.get("enabled", True)) or self.fragment_count == 0:
            return
        self.rigid_stats_calls += 1
        if self.rigid_stats_calls % max(1, int(policy.get("scan_every_frames", 8))) != 0:
            return

        fragment = self.fragment_id[:self.count].numpy()
        kind = self.arrays["kind"][:self.count].numpy()
        damage = self.arrays["damage"][:self.count].numpy()
        base_fixed = self.base_fixed[:self.count].numpy()
        position = self.arrays["x"][:self.count].numpy()
        velocity = self.arrays["v"][:self.count].numpy()
        mass = self.arrays["mass"][:self.count].numpy()
        fully_damaged_threshold = float(policy.get("fully_damaged_threshold", 0.95))
        release_fraction = float(policy.get("release_damage_fraction", 0.12))
        minimum_particles = int(policy.get("minimum_particles", 6))
        maximum_residual = float(policy.get("maximum_internal_velocity_rms", 2.5))
        required_quiet = int(policy.get("required_quiet_scans", 2))
        converted: list[tuple[int, np.ndarray, object]] = []

        for fid in range(self.fragment_count):
            if self.rigid_state_host[fid] != 0:
                continue
            indices = np.flatnonzero((fragment == fid) & (kind != 0))
            if len(indices) < minimum_particles or np.any(base_fixed[indices] != 0):
                self.rigid_quiet_scans_host[fid] = 0
                continue
            released_fraction = float(np.count_nonzero(damage[indices] >= fully_damaged_threshold)) / len(indices)
            if released_fraction < release_fraction:
                self.rigid_quiet_scans_host[fid] = 0
                continue
            fit = fit_rigid_cluster(position[indices], velocity[indices], mass[indices])
            if fit.internal_velocity_rms > maximum_residual:
                self.rigid_quiet_scans_host[fid] = 0
                continue
            self.rigid_quiet_scans_host[fid] += 1
            if self.rigid_quiet_scans_host[fid] >= required_quiet:
                converted.append((fid, indices, fit))

        if not converted:
            return
        # Preserve bodies that have already moved on the GPU before uploading
        # the newly fitted entries.
        center = self.body_center.numpy()
        orientation = self.body_orientation.numpy()
        linear_velocity = self.body_linear_velocity.numpy()
        angular_velocity = self.body_angular_velocity.numpy()
        body_mass = self.body_mass.numpy()
        inverse_inertia = self.body_inverse_inertia.numpy()
        for fid, indices, fit in converted:
            self.rigid_state_host[fid] = 1
            self.rigid_local_host[indices] = fit.local_positions
            center[fid] = fit.center
            orientation[fid] = (0.0, 0.0, 0.0, 1.0)
            linear_velocity[fid] = fit.linear_velocity
            angular_velocity[fid] = fit.angular_velocity
            body_mass[fid] = fit.mass
            inverse_inertia[fid] = fit.inverse_inertia

        self.rigid_state = wp.array(self.rigid_state_host, dtype=wp.int32, device=self.device)
        self.rigid_local_position = wp.array(self.rigid_local_host, dtype=wp.vec3, device=self.device)
        self.body_center = wp.array(center, dtype=wp.vec3, device=self.device)
        self.body_orientation = wp.array(orientation, dtype=wp.quat, device=self.device)
        self.body_linear_velocity = wp.array(linear_velocity, dtype=wp.vec3, device=self.device)
        self.body_angular_velocity = wp.array(angular_velocity, dtype=wp.vec3, device=self.device)
        self.body_mass = wp.array(body_mass, dtype=float, device=self.device)
        self.body_inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33, device=self.device)
        self.rigid_active_count = int(np.count_nonzero(self.rigid_state_host))
        print(
            f"  V3 rigid conversion: +{len(converted)} clusters / "
            f"{sum(len(indices) for _, indices, _ in converted):,} surface particles"
        )

    def substep(self, dt: float):
        # The base fluid pass computes pressure against sleeping buildings as fixed
        # boundaries. The resulting reaction force is then used to wake a whole
        # structural graph for the following substep.
        a = self.arrays
        view = a["x"][:self.count]
        self.grid.build(view, self.max_support)
        wp.launch(clear_vec3, dim=self.count, inputs=[a["solid_force"][:self.count]], device=self.device)
        if self.multirate_enabled:
            classify_every = max(4, int(self.multirate_cfg.get("classify_every_substeps", 256)))
            if self.multirate_tick % classify_every == 0 and self.multirate_tick % 4 == 0:
                wp.launch(
                    classify_time_levels, dim=self.count,
                    inputs=[a["radius"][:self.count], a["v"][:self.count], a["kind"][:self.count],
                            a["damage"][:self.count], self.time_level[:self.count],
                            float(self.multirate_cfg.get("fine_radius", 0.30)),
                            float(self.multirate_cfg.get("active_speed", 8.0)),
                            float(self.multirate_cfg.get("active_damage", 0.02))], device=self.device,
                )
            wp.launch(
                select_active_time_level, dim=self.count,
                inputs=[self.time_level[:self.count], a["kind"][:self.count], self.multirate_tick,
                        self.time_active[:self.count]], device=self.device,
            )
            wp.launch(
                compute_density_multirate, dim=self.count,
                inputs=[self.grid.id, view, a["radius"][:self.count], a["mass"][:self.count],
                        a["volume"][:self.count], a["kind"][:self.count], self.time_active[:self.count],
                        a["rho"][:self.count], a["rho_reference"][:self.count],
                        float(self.cfg["rest_density"]), float(self.cfg["sound_speed"]),
                        float(self.cfg["water_depth"]), float(self.cfg["wave_height"]),
                        float(self.cfg["reservoir_z_max"]), self.max_support], device=self.device,
            )
            wp.launch(
                compute_fluid_forces_multirate, dim=self.count,
                inputs=[self.grid.id, view, a["v"][:self.count], a["radius"][:self.count],
                        a["mass"][:self.count], a["volume"][:self.count], a["kind"][:self.count],
                        a["rho"][:self.count], self.time_level[:self.count], self.time_active[:self.count],
                        self.deferred_fluid_impulse, a["acceleration"][:self.count],
                        a["solid_force"][:self.count], float(self.cfg["rest_density"]),
                        float(self.cfg["sound_speed"]), float(self.cfg.get("max_density_ratio", 1.08)),
                        float(self.cfg["viscosity"]), float(self.cfg.get("xsph_strength", 0.0)),
                        self.max_support, dt], device=self.device,
            )
            wp.launch(
                consume_deferred_fluid_impulse, dim=self.count,
                inputs=[a["mass"][:self.count], a["kind"][:self.count], self.time_level[:self.count],
                        self.time_active[:self.count], self.deferred_fluid_impulse,
                        a["acceleration"][:self.count], dt], device=self.device,
            )
        else:
            wp.launch(
                compute_density, dim=self.count,
                inputs=[self.grid.id, view, a["radius"][:self.count], a["mass"][:self.count], a["volume"][:self.count],
                        a["kind"][:self.count], a["rho"][:self.count], a["rho_reference"][:self.count],
                        float(self.cfg["rest_density"]), float(self.cfg["sound_speed"]),
                        float(self.cfg["water_depth"]), float(self.cfg["wave_height"]),
                        float(self.cfg["reservoir_z_max"]), self.max_support], device=self.device,
            )
            wp.launch(
                compute_fluid_forces, dim=self.count,
                inputs=[self.grid.id, view, a["v"][:self.count], a["radius"][:self.count], a["mass"][:self.count],
                        a["volume"][:self.count], a["kind"][:self.count], a["rho"][:self.count],
                        a["acceleration"][:self.count], a["solid_force"][:self.count],
                        float(self.cfg["rest_density"]), float(self.cfg["sound_speed"]),
                        float(self.cfg.get("max_density_ratio", 1.08)), float(self.cfg["viscosity"]),
                        float(self.cfg.get("xsph_strength", 0.0)), self.max_support, dt], device=self.device,
            )
        clustering = self.v3_cfg["fragment_clustering"]
        wp.launch(
            compute_clustered_solid_forces, dim=self.count,
            inputs=[self.grid.id, view, a["rest_x"][:self.count], a["v"][:self.count], a["radius"][:self.count],
                    a["mass"][:self.count], a["kind"][:self.count], a["material"][:self.count],
                    a["building_id"][:self.count], self.fragment_id[:self.count], self.rigid_state,
                    a["fixed"][:self.count],
                    a["damage"][:self.count], a["solid_force"][:self.count], a["acceleration"][:self.count],
                    self.max_support, dt, float(clustering.get("internal_stiffness_multiplier", 2.0)),
                    float(clustering.get("damage_rate", 1.5)),
                    float(clustering.get("propagation_threshold", 0.65)),
                    float(clustering.get("max_damage_per_substep", 0.0004)),
                    float(clustering.get("fracture_reference_spacing", 0.65)) * 0.48 * 1.25],
            device=self.device,
        )
        rigid_policy = self.v3_cfg.get("rigid_clusters", {})
        if bool(rigid_policy.get("enabled", True)) and self.rigid_active_count > 0:
            wp.launch(
                clear_body_accumulators, dim=max(1, self.fragment_count),
                inputs=[self.body_force, self.body_torque], device=self.device,
            )
            wp.launch(
                accumulate_rigid_body_loads, dim=self.count,
                inputs=[view, a["v"][:self.count], a["radius"][:self.count], a["mass"][:self.count],
                        a["kind"][:self.count], self.fragment_id[:self.count], self.rigid_state,
                        a["acceleration"][:self.count], self.body_center, self.body_force, self.body_torque,
                        float(self.cfg["domain_width"]) * 0.5, float(self.cfg["reservoir_z_min"]),
                        float(self.cfg["domain_z_max"]), float(self.cfg["domain_y_max"]),
                        float(rigid_policy.get("boundary_stiffness", 4.0e6)),
                        float(rigid_policy.get("boundary_damping", 1.8e4))],
                device=self.device,
            )
            wp.launch(
                integrate_rigid_bodies, dim=max(1, self.fragment_count),
                inputs=[self.rigid_state, self.body_center, self.body_orientation,
                        self.body_linear_velocity, self.body_angular_velocity, self.body_mass,
                        self.body_inverse_inertia, self.body_force, self.body_torque, dt,
                        float(rigid_policy.get("linear_damping", 0.015)),
                        float(rigid_policy.get("angular_damping", 0.03)),
                        float(rigid_policy.get("maximum_angular_speed", 18.0))],
                device=self.device,
            )
            wp.launch(
                scatter_rigid_particles, dim=self.count,
                inputs=[view, a["v"][:self.count], a["kind"][:self.count], self.fragment_id[:self.count],
                        self.rigid_state, self.rigid_local_position, self.body_center, self.body_orientation,
                        self.body_linear_velocity, self.body_angular_velocity],
                device=self.device,
            )
            wp.launch(
                mask_rigid_particles_as_fixed, dim=self.count,
                inputs=[a["kind"][:self.count], self.fragment_id[:self.count], self.rigid_state,
                        a["fixed"][:self.count]], device=self.device,
            )
        if self.multirate_enabled:
            wp.launch(
                integrate_multirate, dim=self.count,
                inputs=[view, a["v"][:self.count], a["acceleration"][:self.count], a["kind"][:self.count],
                        a["fixed"][:self.count], self.time_level[:self.count], self.time_active[:self.count],
                        dt, float(self.cfg["domain_width"]) * 0.5, float(self.cfg["reservoir_z_min"]),
                        float(self.cfg["domain_z_max"]), float(self.cfg["domain_y_max"]),
                        float(self.cfg.get("fluid_bed_drag", 0.12))], device=self.device,
            )
            self.multirate_tick += 1
        else:
            wp.launch(
                integrate, dim=self.count,
                inputs=[view, a["v"][:self.count], a["acceleration"][:self.count], a["kind"][:self.count],
                        a["fixed"][:self.count], dt, float(self.cfg["domain_width"]) * 0.5,
                        float(self.cfg["reservoir_z_min"]), float(self.cfg["domain_z_max"]),
                        float(self.cfg["domain_y_max"]), float(self.cfg.get("fluid_bed_drag", 0.12))],
                device=self.device,
            )
        self.time += dt
        if self.building_count == 0:
            return
        wp.launch(clear_int, dim=self.building_count, inputs=[self.activation_hits], device=self.device)
        wp.launch(
            count_loaded_building_particles, dim=self.count,
            inputs=[self.arrays["kind"][:self.count], self.arrays["building_id"][:self.count],
                    self.arrays["mass"][:self.count], self.arrays["solid_force"][:self.count],
                    float(self.v3_cfg.get("activation_force_per_mass", 0.8)), self.activation_hits],
            device=self.device,
        )
        wp.launch(
            activate_buildings_from_hits, dim=self.building_count,
            inputs=[self.activation_hits, self.building_active,
                    int(self.v3_cfg.get("minimum_contact_particles", 8))], device=self.device,
        )
        wp.launch(
            apply_building_activity, dim=self.count,
            inputs=[self.arrays["kind"][:self.count], self.arrays["building_id"][:self.count],
                    self.base_fixed[:self.count], self.building_active, self.arrays["fixed"][:self.count]],
            device=self.device,
        )

    def refine(self):
        # Water keeps conservative 1->8 volume refinement. Structural
        # surfaces use planar 1->4 refinement so a thin wall does not become a
        # volumetric cloud when resolution increases near an impact.
        super().refine()
        if not bool(self.v3_cfg["solid_refinement"].get("enabled", True)):
            return
        old_count = self.count
        count_device = wp.array(np.asarray([old_count], dtype=np.int32), dtype=wp.int32, device=self.device)
        policy = self.v3_cfg["solid_refinement"]
        predicted_front = (
            float(self.cfg["reservoir_z_max"])
            + (float(self.cfg.get("background_current", 0.0)) + float(self.cfg.get("wave_speed", 0.0))) * self.time
        )
        preimpact_limit = predicted_front + float(policy.get("preimpact_margin", 6.0))
        preimpact_host = np.zeros(max(1, self.building_count), dtype=np.int32)
        for bid, spec in enumerate(self.cfg["buildings"]):
            _cx, cz, _width, depth, _height = map(float, spec)
            if cz - depth * 0.5 <= preimpact_limit:
                preimpact_host[bid] = 1
        self.preimpact_building = wp.array(preimpact_host, dtype=wp.int32, device=self.device)
        preimpact_count = int(np.count_nonzero(preimpact_host))
        if preimpact_count != self.last_preimpact_building_count:
            print(
                f"  V3 pre-impact structural LOD: {self.last_preimpact_building_count} -> "
                f"{preimpact_count} buildings (predicted front z={predicted_front:.1f} m)"
            )
            self.last_preimpact_building_count = preimpact_count
        wp.launch(clear_int, dim=7, inputs=[self.refinement_counters], device=self.device)
        wp.launch(
            refine_impacted_solids, dim=old_count,
            inputs=[self.arrays["x"], self.arrays["rest_x"], self.arrays["v"], self.arrays["radius"],
                    self.arrays["mass"], self.arrays["volume"], self.arrays["kind"], self.arrays["material"],
                    self.arrays["structural_class"],
                    self.arrays["building_id"], self.arrays["fixed"], self.base_fixed, self.arrays["damage"],
                    self.arrays["rho_reference"], self.arrays["solid_force"], self.fragment_id, self.normal_axis,
                    self.rigid_state, self.preimpact_building, self.refinement_counters,
                    count_device, old_count, self.capacity,
                    float(policy["crack_spacing"]) * 0.48 * 1.25,
                    float(policy.get("glass_min_spacing", 0.1625)) * 0.48 * 1.25,
                    float(policy["impact_spacing"]) * 0.48 * 1.25,
                    float(policy["crack_damage_trigger"]),
                    float(policy.get("impact_acceleration_trigger", 25.0))],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        self.count = min(int(count_device.numpy()[0]), self.capacity)
        added = self.count - old_count
        if added > 0:
            self.solid_count += added
            self.fragment_host = self.fragment_id[:self.count].numpy()
            valid_fragments = self.fragment_host[self.fragment_host >= 0]
            self.fragment_counts_host = np.bincount(
                valid_fragments, minlength=self.fragment_count
            ).astype(np.int32)
            refined_by_role = self.refinement_counters.numpy()
            print(
                f"  V3 structural refinement: {old_count:,} -> {self.count:,} particles "
                f"(+{added:,} solid children; slab={refined_by_role[1]}, wall={refined_by_role[2]}, "
                f"beam={refined_by_role[3]}, column={refined_by_role[4]}, core={refined_by_role[5]}, "
                f"glass={refined_by_role[6]})"
            )

    def stats(self):
        self.update_rigid_clusters()
        result = super().stats()
        active_count = int(np.count_nonzero(self.building_active.numpy())) if self.building_count else 0
        if active_count != self.last_active_count:
            print(f"  V3 building activation: {self.last_active_count} -> {active_count}")
            self.last_active_count = active_count
        result["active_buildings"] = active_count
        result["cohesive_fragments"] = self.fragment_count
        damage_host = self.arrays["damage"][:len(self.fragment_host)].numpy()
        solid_mask = self.fragment_host >= 0
        fully_damaged = solid_mask & (damage_host >= 0.95)
        released_hits = np.bincount(
            self.fragment_host[fully_damaged], minlength=self.fragment_count
        ) if self.fragment_count else np.empty(0, dtype=np.int64)
        release_fraction = float(self.v3_cfg["fragment_clustering"].get("release_damage_fraction", 0.12))
        released = int(np.count_nonzero(
            released_hits >= np.maximum(2, np.ceil(self.fragment_counts_host * release_fraction))
        )) if self.fragment_count else 0
        if released != self.last_released_fragment_count:
            print(f"  V3 released cohesive fragments: {self.last_released_fragment_count} -> {released}")
            self.last_released_fragment_count = released
        result["released_fragments"] = released
        rigid_count = int(np.count_nonzero(self.rigid_state_host))
        rigid_particles = int(np.count_nonzero(
            (self.fragment_host >= 0) & self.rigid_state_host[np.maximum(self.fragment_host, 0)]
        )) if self.fragment_count else 0
        if rigid_count != self.last_rigid_count:
            print(f"  V3 active rigid clusters: {self.last_rigid_count} -> {rigid_count}")
            self.last_rigid_count = rigid_count
        result["rigid_clusters"] = rigid_count
        result["rigid_particles"] = rigid_particles
        if self.multirate_enabled:
            levels = self.time_level[:self.count].numpy()
            fluid = self.arrays["kind"][:self.count].numpy() == 0
            for level in range(3):
                result[f"time_level_{level}_particles"] = int(np.count_nonzero(fluid & (levels == level)))
        self.update_water_surface()
        if self.surface_enabled:
            surface_mask = self.arrays["water_surface_mask"][:self.count].numpy()
            result["surface_water_particles"] = int(np.count_nonzero(surface_mask))
            result["water_mesh_vertices"] = len(self.water_mesh_vertices)
            result["water_mesh_triangles"] = self.water_mesh_triangle_count
            result["water_field_nodes"] = int(np.prod(self.water_field_shape, dtype=np.int64))
            result["water_mesh_excluded_surface_particles"] = self.water_mesh_excluded_surface_count
            result["water_mesh_voxel_millimeters"] = int(round(self.water_mesh_voxel_size * 1000.0))
        return result


def main():
    parser = argparse.ArgumentParser(description="DELUGE V3 hybrid CUDA simulator")
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--duration", type=float, help="Override simulated duration for validation runs")
    parser.add_argument("--frames", type=int, help="Override output frame count")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.duration is not None:
        if args.duration <= 0.0:
            raise ValueError("--duration must be positive")
        cfg["duration_seconds"] = float(args.duration)
    if args.frames is not None:
        if args.frames <= 0:
            raise ValueError("--frames must be positive")
        cfg["duration_seconds"] = float(args.frames) / float(cfg["output_fps"])
    output = args.output or HERE / "outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    wp.init()
    solver = HybridDelugeSolver(cfg, output, args.resume)
    solver.run(smoke=args.smoke, no_video=args.no_video)


if __name__ == "__main__":
    main()
