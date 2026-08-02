"""DELUGE V3 hybrid solver prototype.

The proven V2 water/fracture kernels remain untouched. V3 layers structural
LOD on top: dormant buildings are cheap fixed boundaries until a coherent
hydrodynamic load activates their deformable bond graph.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
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
    accumulate_rigid_contacts,
    accumulate_rigid_proxy_boundaries,
    accumulate_rigid_proxy_contacts,
    accumulate_building_damage,
    accumulate_material_impact,
    apply_conservative_fluid_merges,
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
    reactivate_rigid_after_impact,
    scatter_rigid_particles,
    select_active_time_level,
)
from hybrid_model import (  # noqa: E402
    FragmentSupportGraph,
    SolidRefinementPolicy,
    build_fragment_ids,
    build_fragment_support_graph,
    build_refinement_axes,
    evaluate_fragment_fracture_energy,
    evaluate_fragment_support,
    select_conservative_fluid_merges,
    write_facade_skin,
)
from hybrid_renderer import HybridRenderer  # noqa: E402
from rigid_clusters import fit_rigid_cluster, fit_rigid_collision_proxy  # noqa: E402
from shallow_water import (  # noqa: E402
    ShallowWaterFarField,
    compact_float_particles,
    compact_int_particles,
    compact_vec3_components,
    compact_vec3_particles,
    emit_sph_interface_particles,
    mark_sph_return_particles,
    remap_particle_indices,
)
from surface_kernels import (  # noqa: E402
    blend_sparse_fields,
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


def limit_water_core_height(lower: np.ndarray, upper: np.ndarray, maximum_height):
    """Cap only the connected reconstruction; callers can mesh excluded spray locally."""
    lower = np.asarray(lower, dtype=np.float32).copy()
    upper = np.asarray(upper, dtype=np.float32).copy()
    if maximum_height is not None:
        upper[1] = min(upper[1], float(maximum_height))
        if upper[1] <= lower[1]:
            raise ValueError("water_mesh maximum_core_height must exceed the lower water bound")
    return lower, upper


def hysteretic_bounds(previous_lower, previous_upper, target_lower, target_upper, shrink_alpha):
    """Expand a reconstruction domain immediately and shrink it gradually."""
    target_lower = np.asarray(target_lower, dtype=np.float32)
    target_upper = np.asarray(target_upper, dtype=np.float32)
    if previous_lower is None or previous_upper is None:
        return target_lower.copy(), target_upper.copy()
    previous_lower = np.asarray(previous_lower, dtype=np.float32)
    previous_upper = np.asarray(previous_upper, dtype=np.float32)
    alpha = float(np.clip(shrink_alpha, 0.0, 1.0))
    lower = np.where(
        target_lower < previous_lower,
        target_lower,
        previous_lower + (target_lower - previous_lower) * alpha,
    )
    upper = np.where(
        target_upper > previous_upper,
        target_upper,
        previous_upper + (target_upper - previous_upper) * alpha,
    )
    return lower.astype(np.float32), upper.astype(np.float32)


def select_splash_bricks(
    positions: np.ndarray,
    excluded: np.ndarray,
    brick_size: float,
    enter_particles: int,
    keep_particles: int,
    previous_keys,
    maximum_bricks: int,
):
    """Select dense outlier groups while rejecting isolated spray droplets."""
    candidates = np.asarray(positions, dtype=np.float32)[np.asarray(excluded, dtype=bool)]
    if len(candidates) == 0 or maximum_bricks <= 0:
        return [], {}
    keys = np.floor(candidates / float(brick_size)).astype(np.int32)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    previous = {tuple(int(v) for v in key) for key in previous_keys}
    eligible = []
    count_by_key = {}
    for key_array, count in zip(unique, counts):
        key = tuple(int(v) for v in key_array)
        count = int(count)
        count_by_key[key] = count
        threshold = keep_particles if key in previous else enter_particles
        if count >= threshold:
            eligible.append((key, count))
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [key for key, _ in eligible[:maximum_bricks]], count_by_key


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
        fragment_host, fragment_counts = build_fragment_ids(
            rest_host, kind_host, building_host, cfg, structural_class_host
        )
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
        impact_host = np.zeros(self.capacity, dtype=np.float32)
        local_impact_host = np.zeros(self.capacity, dtype=np.int32)
        active_host = np.zeros(max(1, self.building_count), dtype=np.int32)
        activation_exposure_host = np.zeros(max(1, self.building_count), dtype=np.float32)
        self.adaptive_merged_groups_total = 0
        self.adaptive_merged_particles_total = 0
        saved_support_graph = None
        saved_fragment_support = None
        saved_edge_intact = None
        saved_edge_fracture_energy = None
        v3_resume = self._v3_checkpoint_path(self.resume_path) if self.resume_path else None
        if v3_resume is not None and v3_resume.exists():
            with np.load(v3_resume, allow_pickle=False) as state:
                expected_fragment_schema = int(
                    self.v3_cfg["fragment_clustering"].get("schema_version", 1)
                )
                saved_fragment_schema = int(
                    state["fragment_schema_version"]
                ) if "fragment_schema_version" in state else 1
                if saved_fragment_schema != expected_fragment_schema:
                    raise RuntimeError(
                        "Incompatible V3 fragment topology in checkpoint "
                        f"{v3_resume.name}: schema {saved_fragment_schema}, expected "
                        f"{expected_fragment_schema}. Start a fresh simulation so facade, "
                        "floor and frame particles are clustered independently."
                    )
                saved_base = state["base_fixed"]
                base_host[:len(saved_base)] = saved_base
                saved_active = state["building_active"]
                active_host[:min(len(saved_active), len(active_host))] = saved_active[:len(active_host)]
                if "building_activation_exposure_seconds" in state:
                    saved_exposure = state["building_activation_exposure_seconds"]
                    activation_exposure_host[:min(len(saved_exposure), len(activation_exposure_host))] = (
                        saved_exposure[:len(activation_exposure_host)]
                    )
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
                if "material_impact_impulse" in state:
                    saved_impact = state["material_impact_impulse"]
                    impact_host[:len(saved_impact)] = saved_impact
                if "local_impact_active" in state:
                    saved_local_impact = state["local_impact_active"]
                    local_impact_host[:len(saved_local_impact)] = saved_local_impact
                if "adaptive_merged_groups_total" in state:
                    self.adaptive_merged_groups_total = int(state["adaptive_merged_groups_total"])
                if "adaptive_merged_particles_total" in state:
                    self.adaptive_merged_particles_total = int(state["adaptive_merged_particles_total"])
                support_keys = (
                    "support_edge_fragments", "support_sample_offsets", "support_sample_pairs",
                    "support_sample_rest_length", "support_anchored_fragments",
                )
                if all(name in state for name in support_keys):
                    saved_support_graph = FragmentSupportGraph(
                        state["support_edge_fragments"].astype(np.int32, copy=True),
                        state["support_sample_offsets"].astype(np.int32, copy=True),
                        state["support_sample_pairs"].astype(np.int32, copy=True),
                        state["support_sample_rest_length"].astype(np.float32, copy=True),
                        state["support_anchored_fragments"].astype(bool, copy=True),
                    )
                    if "fragment_support_state" in state:
                        saved_fragment_support = state["fragment_support_state"].astype(
                            np.float32, copy=True
                        )
                    if "support_edge_intact_state" in state:
                        saved_edge_intact = state["support_edge_intact_state"].astype(
                            bool, copy=True
                        )
                    if "support_edge_fracture_energy" in state:
                        saved_edge_fracture_energy = state[
                            "support_edge_fracture_energy"
                        ].astype(np.float32, copy=True)
            print(f"V3 state restored from {v3_resume.name}")
        else:
            for bid in self.v3_cfg.get("initially_active_buildings", []):
                if 0 <= int(bid) < self.building_count:
                    active_host[int(bid)] = 1
            if self.resume_path is not None:
                print("WARNING: matching V3 checkpoint is absent; using configured building activity")

        self.base_fixed = wp.array(base_host, dtype=wp.int32, device=self.device)
        self.arrays["material_impact_impulse"] = wp.array(
            impact_host, dtype=float, device=self.device
        )
        self.arrays["local_impact_active"] = wp.array(
            local_impact_host, dtype=wp.int32, device=self.device
        )
        support_policy = self.v3_cfg.get("support_graph", {})
        radius_host = self.arrays["radius"][:self.count].numpy()
        if saved_support_graph is not None:
            self.fragment_support_graph = saved_support_graph
            self.fragment_support_host = (
                saved_fragment_support
                if saved_fragment_support is not None
                else np.ones(self.fragment_count, dtype=np.float32)
            )
            self.fragment_edge_intact_host = (
                saved_edge_intact
                if saved_edge_intact is not None
                else np.ones(len(saved_support_graph.edge_fragments), dtype=bool)
            )
            self.fragment_edge_fracture_energy_host = (
                saved_edge_fracture_energy
                if saved_edge_fracture_energy is not None
                and len(saved_edge_fracture_energy) == len(saved_support_graph.edge_fragments)
                else np.zeros(len(saved_support_graph.edge_fragments), dtype=np.float32)
            )
            print(
                f"V3 support graph restored: {len(saved_support_graph.edge_fragments):,} edges / "
                f"{len(saved_support_graph.sample_pairs):,} boundary samples"
            )
        else:
            self.fragment_support_graph = build_fragment_support_graph(
                rest_host, radius_host, kind_host, building_host, base_host[:self.count],
                self.fragment_host,
                int(support_policy.get("maximum_samples_per_edge", 12)),
            )
            self.fragment_support_host = np.ones(self.fragment_count, dtype=np.float32)
            self.fragment_edge_intact_host = np.ones(
                len(self.fragment_support_graph.edge_fragments), dtype=bool
            )
            self.fragment_edge_fracture_energy_host = np.zeros(
                len(self.fragment_support_graph.edge_fragments), dtype=np.float32
            )
        self.fragment_fracture_energy_host = np.zeros(self.fragment_count, dtype=np.float32)
        if len(self.fragment_support_graph.edge_fragments):
            np.maximum.at(
                self.fragment_fracture_energy_host,
                self.fragment_support_graph.edge_fragments[:, 0],
                self.fragment_edge_fracture_energy_host,
            )
            np.maximum.at(
                self.fragment_fracture_energy_host,
                self.fragment_support_graph.edge_fragments[:, 1],
                self.fragment_edge_fracture_energy_host,
            )
        self.fragment_support = wp.array(
            self.fragment_support_host, dtype=float, device=self.device
        )
        self.fragment_fracture_energy = wp.array(
            self.fragment_fracture_energy_host, dtype=float, device=self.device
        )
        self.fragment_building_host = np.full(self.fragment_count, -1, dtype=np.int32)
        valid_fragment_particles = np.flatnonzero(self.fragment_host >= 0)
        if len(valid_fragment_particles):
            unique_fragment, first = np.unique(
                self.fragment_host[valid_fragment_particles], return_index=True
            )
            self.fragment_building_host[unique_fragment] = building_host[
                valid_fragment_particles[first]
            ]
        self.last_unsupported_fragment_count = 0
        volume_host = self.arrays["volume"][:self.count].numpy()
        solid_building_mask = (kind_host != 0) & (building_host >= 0)
        building_volume_host = np.bincount(
            building_host[solid_building_mask],
            weights=volume_host[solid_building_mask].astype(np.float64, copy=False),
            minlength=max(1, self.building_count),
        ).astype(np.float32)
        self.building_structural_volume = wp.array(
            building_volume_host, dtype=float, device=self.device
        )
        self.building_damage_integral = wp.zeros(
            max(1, self.building_count), dtype=float, device=self.device
        )
        self.fragment_role_host = self._build_fragment_roles()
        self.building_active = wp.array(active_host, dtype=wp.int32, device=self.device)
        self.building_activation_exposure = wp.array(
            activation_exposure_host, dtype=float, device=self.device
        )
        self.activation_hits = wp.zeros(max(1, self.building_count), dtype=wp.int32, device=self.device)
        self.last_active_count = int(np.count_nonzero(active_host))
        self.last_released_fragment_count = 0
        self.last_preimpact_building_count = 0
        self.preimpact_building = wp.zeros(max(1, self.building_count), dtype=wp.int32, device=self.device)
        self.refinement_counters = wp.zeros(7, dtype=wp.int32, device=self.device)
        self._initialize_rigid_clusters(v3_resume)
        self._initialize_multirate()
        self.shallow_water = ShallowWaterFarField(cfg, self.device, v3_resume)
        self.return_keep = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.return_offsets = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.particle_compaction_scratch = None
        self.last_adaptive_merge_frame = -1
        self._initialize_water_surface(v3_resume)

        self._update_fragment_support_graph(
            self.arrays["x"][:self.count].numpy(),
            self.arrays["damage"][:self.count].numpy(),
            self.arrays["material"][:self.count].numpy(),
            structural_class_host,
            force=True,
        )

        wp.launch(
            apply_building_activity, dim=self.count,
            inputs=[self.arrays["kind"][:self.count], self.arrays["building_id"][:self.count],
                    self.arrays["structural_class"][:self.count], self.base_fixed[:self.count],
                    self.building_active, self.arrays["local_impact_active"][:self.count],
                    self.arrays["fixed"][:self.count]],
            device=self.device,
        )
        skin_path = output / "facade_skin.npz"
        panel_count = write_facade_skin(
            skin_path, cfg, rest_host, kind_host, building_host, fragment_host,
            self.arrays["radius"][:self.count].numpy(), structural_class_host,
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
                float(self.v3_cfg.get("crack_rendering", {}).get("strength", 1.0))
                if bool(self.v3_cfg.get("crack_rendering", {}).get("enabled", True)) else 0.0,
            )
            for name, camera in configured_views.items()
        }
        self.renderer = next(iter(self.renderers.values()))
        for renderer in self.renderers.values():
            renderer.fragment_support = self.fragment_support
            renderer.fragment_fracture_energy = self.fragment_fracture_energy
        dormant_count = self.building_count - int(np.count_nonzero(active_host))
        smallest = int(fragment_counts.min()) if len(fragment_counts) else 0
        largest = int(fragment_counts.max()) if len(fragment_counts) else 0
        print(
            f"V3 structural LOD: {dormant_count}/{self.building_count} dormant buildings; "
            f"cohesive fragments: {self.fragment_count:,} ({smallest}-{largest} particles); "
            f"facade skin: {panel_count:,} panels; views: {', '.join(self.renderers)}"
        )

    def _update_fragment_support_graph(
        self,
        position_host: np.ndarray,
        damage_host: np.ndarray,
        material_host: np.ndarray | None = None,
        structural_class_host: np.ndarray | None = None,
        force: bool = False,
    ) -> None:
        policy = self.v3_cfg.get("support_graph", {})
        if not bool(policy.get("enabled", True)) or self.fragment_count == 0:
            return
        update_every = max(1, int(policy.get("update_every_frames", 1)))
        frame_index = int(round(self.time * float(self.cfg["output_fps"])))
        if not force and frame_index % update_every != 0:
            return
        support, intact_edges = evaluate_fragment_support(
            self.fragment_support_graph,
            position_host,
            damage_host,
            float(policy.get("intact_damage_threshold", 0.95)),
            float(policy.get("maximum_bond_stretch", 1.60)),
            float(policy.get("minimum_intact_sample_fraction", 0.25)),
        )
        self.fragment_support_host = support.astype(np.float32, copy=False)
        self.fragment_edge_intact_host = intact_edges
        if material_host is None:
            material_host = self.arrays["material"][:self.count].numpy()
        if structural_class_host is None:
            structural_class_host = self.arrays["structural_class"][:self.count].numpy()
        edge_energy, fragment_energy = evaluate_fragment_fracture_energy(
            self.fragment_support_graph,
            position_host,
            damage_host,
            material_host,
            structural_class_host,
            self.fragment_edge_fracture_energy_host,
            float(self.v3_cfg.get("crack_rendering", {}).get("fracture_energy_onset", 0.35)),
        )
        edge_energy[~intact_edges] = 1.0
        if len(edge_energy):
            fragment_energy.fill(0.0)
            np.maximum.at(
                fragment_energy, self.fragment_support_graph.edge_fragments[:, 0], edge_energy
            )
            np.maximum.at(
                fragment_energy, self.fragment_support_graph.edge_fragments[:, 1], edge_energy
            )
        self.fragment_edge_fracture_energy_host = edge_energy
        self.fragment_fracture_energy_host = fragment_energy
        self.fragment_support = wp.array(
            self.fragment_support_host, dtype=float, device=self.device
        )
        self.fragment_fracture_energy = wp.array(
            self.fragment_fracture_energy_host, dtype=float, device=self.device
        )
        unsupported = int(np.count_nonzero(~support))
        if unsupported != self.last_unsupported_fragment_count:
            print(
                f"  V3 support graph: {unsupported:,}/{self.fragment_count:,} fragments "
                "disconnected from foundations"
            )
            self.last_unsupported_fragment_count = unsupported

    def _build_fragment_roles(self) -> np.ndarray:
        """Return the dominant structural role of each cohesive fragment by volume.

        Particle counts are not stable under adaptive 1->4 refinement.  Volume is
        conserved, so a volume-weighted role remains meaningful before and after
        local refinement and when a run is restored from a checkpoint.
        """
        roles = np.zeros(self.fragment_count, dtype=np.int32)
        if self.fragment_count == 0:
            return roles
        fragment = np.asarray(self.fragment_host, dtype=np.int32)
        structural_role = self.arrays["structural_class"][:len(fragment)].numpy()
        volume = self.arrays["volume"][:len(fragment)].numpy().astype(np.float64, copy=False)
        role_volume = np.zeros((6, self.fragment_count), dtype=np.float64)
        for role in range(1, 7):
            mask = (fragment >= 0) & (structural_role == role)
            if np.any(mask):
                role_volume[role - 1] = np.bincount(
                    fragment[mask], weights=volume[mask], minlength=self.fragment_count
                )
        roles[:] = np.argmax(role_volume, axis=0).astype(np.int32) + 1
        roles[np.max(role_volume, axis=0) <= 0.0] = 0
        return roles

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

    def _initialize_water_surface(self, checkpoint: Path | None = None):
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
        self.water_mesh_domain_lower = None
        self.water_mesh_domain_upper = None
        self.water_mesh_lod_recovery_count = 0
        self.water_mesh_lod_change_count = 0
        self.water_mesh_field_lower = None
        self.water_sparse_field = None
        self.water_splash_active_keys = set()
        self.water_splash_brick_count = 0
        self.water_splash_mesh_vertices = 0
        self.water_stitch_sample_count = 0
        self.water_surface_classify_ms = 0.0
        self.water_mesh_preprocess_ms = 0.0
        self.water_mesh_field_ms = 0.0
        self.water_mesh_marching_cubes_ms = 0.0
        self.water_mesh_splash_ms = 0.0
        self.water_mesh_total_ms = 0.0
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if "water_mesh_domain_lower" in saved:
                    self.water_mesh_domain_lower = saved["water_mesh_domain_lower"].astype(np.float32)
                    self.water_mesh_domain_upper = saved["water_mesh_domain_upper"].astype(np.float32)
                if "water_mesh_voxel_size" in saved:
                    self.water_mesh_voxel_size = float(saved["water_mesh_voxel_size"])
                if "water_mesh_lod_recovery_count" in saved:
                    self.water_mesh_lod_recovery_count = int(saved["water_mesh_lod_recovery_count"])
                if "water_mesh_lod_change_count" in saved:
                    self.water_mesh_lod_change_count = int(saved["water_mesh_lod_change_count"])

    def update_water_surface(self):
        if not self.surface_enabled:
            return
        classify_started = time.perf_counter()
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
        wp.synchronize_device(self.device)
        self.water_surface_classify_ms = (time.perf_counter() - classify_started) * 1000.0
        if self.water_mesh_enabled:
            self._build_water_mesh()

    def _build_water_mesh(self):
        build_started = time.perf_counter()
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
        surface_radii = self.arrays["radius"][:self.count].numpy()[surface_indices]
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
        # Keep the connected-body reconstruction bounded vertically.  Water
        # above this height is detached impact spray: it is reconstructed in
        # local splash bricks below instead of enlarging the reservoir-to-city
        # field and forcing a global LOD drop.
        robust_minimum, robust_maximum = limit_water_core_height(
            robust_minimum,
            robust_maximum,
            self.water_mesh_cfg.get("maximum_core_height"),
        )
        excluded = np.any((positions < robust_minimum) | (positions > robust_maximum), axis=1)
        self.water_mesh_excluded_surface_count = int(np.count_nonzero(excluded))
        stitch_positions, stitch_radii = self.shallow_water.stitched_surface_samples(positions)
        self.water_stitch_sample_count = len(stitch_positions)
        if len(stitch_positions):
            robust_minimum = np.minimum(robust_minimum, stitch_positions.min(axis=0))
            robust_maximum = np.maximum(robust_maximum, stitch_positions.max(axis=0))

        robust_minimum, robust_maximum = hysteretic_bounds(
            self.water_mesh_domain_lower,
            self.water_mesh_domain_upper,
            robust_minimum,
            robust_maximum,
            float(self.water_mesh_cfg.get("domain_shrink_alpha", 0.12)),
        )
        self.water_mesh_domain_lower = robust_minimum.copy()
        self.water_mesh_domain_upper = robust_maximum.copy()

        def quantized_domain(size: float):
            brick = size * 8.0
            lower = np.floor((robust_minimum - size * margin_cells) / brick) * brick
            upper = np.ceil((robust_maximum + size * margin_cells) / brick) * brick
            dims = np.maximum(3, np.rint((upper - lower) / size).astype(np.int32) + 1)
            upper = lower + (dims - 1) * size
            return lower.astype(np.float32), upper.astype(np.float32), tuple(int(v) for v in dims)

        target_voxel = voxel
        lower, upper, shape = quantized_domain(target_voxel)
        while int(np.prod(shape, dtype=np.int64)) > max_nodes:
            target_voxel *= 1.25
            lower, upper, shape = quantized_domain(target_voxel)

        previous_voxel = float(self.water_mesh_voxel_size)
        recovery_frames = max(1, int(self.water_mesh_cfg.get("lod_recovery_frames", 8)))
        if previous_voxel > 0.0 and target_voxel < previous_voxel * 0.999:
            self.water_mesh_lod_recovery_count += 1
            if self.water_mesh_lod_recovery_count >= recovery_frames:
                voxel = max(target_voxel, previous_voxel / 1.25)
                self.water_mesh_lod_recovery_count = 0
            else:
                voxel = previous_voxel
        else:
            voxel = target_voxel
            self.water_mesh_lod_recovery_count = 0
        lower, upper, shape = quantized_domain(voxel)
        # Memory safety takes precedence over temporal recovery: expansion may
        # require an immediate coarsening, whereas refinement is rate-limited.
        while int(np.prod(shape, dtype=np.int64)) > max_nodes:
            voxel *= 1.25
            lower, upper, shape = quantized_domain(voxel)
        if previous_voxel > 0.0 and abs(voxel - previous_voxel) > 1.0e-6:
            self.water_mesh_lod_change_count += 1
        self.water_mesh_preprocess_ms = (time.perf_counter() - build_started) * 1000.0
        nx, ny, nz = shape
        field_started = time.perf_counter()
        field = wp.zeros(shape, dtype=float, device=self.device)
        wp.launch(
            splat_sparse_surface_field, dim=self.count,
            inputs=[self.arrays["x"][:self.count], self.arrays["radius"][:self.count],
                    self.arrays["kind"][:self.count], self.arrays["water_surface_mask"][:self.count],
                    field, wp.vec3(*lower), voxel, nx, ny, nz], device=self.device,
        )
        if len(stitch_positions):
            stitch_x = wp.array(stitch_positions, dtype=wp.vec3, device=self.device)
            stitch_radius = wp.array(stitch_radii, dtype=float, device=self.device)
            stitch_kind = wp.zeros(len(stitch_positions), dtype=wp.int32, device=self.device)
            stitch_mask = wp.ones(len(stitch_positions), dtype=wp.int32, device=self.device)
            wp.launch(
                splat_sparse_surface_field, dim=len(stitch_positions),
                inputs=[stitch_x, stitch_radius, stitch_kind, stitch_mask, field,
                        wp.vec3(*lower), voxel, nx, ny, nz], device=self.device,
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
        temporal_weight = float(self.water_mesh_cfg.get("temporal_field_previous_weight", 0.12))
        same_field_domain = (
            self.water_sparse_field is not None
            and self.water_field_shape == shape
            and self.water_mesh_field_lower is not None
            and np.allclose(self.water_mesh_field_lower, lower, atol=1.0e-5)
            and abs(previous_voxel - voxel) <= 1.0e-6
        )
        if same_field_domain and temporal_weight > 0.0:
            wp.launch(
                blend_sparse_fields, dim=shape,
                inputs=[field, self.water_sparse_field, min(0.45, temporal_weight)],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        self.water_mesh_field_ms = (time.perf_counter() - field_started) * 1000.0
        marching_started = time.perf_counter()
        vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
            field,
            threshold=float(self.water_mesh_cfg.get("iso_threshold", 0.72)),
            domain_bounds_lower_corner=tuple(float(v) for v in lower),
            domain_bounds_upper_corner=tuple(float(v) for v in upper),
        )
        wp.synchronize_device(self.device)
        self.water_mesh_marching_cubes_ms = (time.perf_counter() - marching_started) * 1000.0
        splash_started = time.perf_counter()
        vertices, indices = self._append_splash_brick_meshes(
            vertices, indices, positions, surface_radii, excluded
        )
        wp.synchronize_device(self.device)
        self.water_mesh_splash_ms = (time.perf_counter() - splash_started) * 1000.0
        self.water_sparse_field = field
        self.water_mesh_field_lower = lower.copy()
        self.water_mesh_vertices = vertices
        self.water_mesh_indices = indices
        self.water_mesh_triangle_count = len(indices) // 3
        self.water_mesh_voxel_size = voxel
        self.water_field_shape = shape
        self.arrays["water_mesh_vertices"] = vertices
        self.arrays["water_mesh_indices"] = indices
        self.water_mesh_total_ms = (time.perf_counter() - build_started) * 1000.0

    def _append_shallow_surface_mesh(self, vertices, indices):
        shallow_vertices, shallow_indices = self.shallow_water.surface_mesh()
        if shallow_vertices is None or len(shallow_indices) < 3:
            return vertices, indices
        base = len(vertices)
        combined_vertices = np.concatenate([vertices.numpy(), shallow_vertices], axis=0)
        combined_indices = np.concatenate(
            [indices.numpy(), shallow_indices + base], axis=0
        ).astype(np.int32)
        return (
            wp.array(combined_vertices, dtype=wp.vec3, device=self.device),
            wp.array(combined_indices, dtype=wp.int32, device=self.device),
        )

    def _append_splash_brick_meshes(self, vertices, indices, positions, surface_radii, excluded):
        """Append local high-resolution meshes for dense spray sheets."""
        if not bool(self.water_mesh_cfg.get("splash_bricks_enabled", True)):
            self.water_splash_active_keys.clear()
            self.water_splash_brick_count = 0
            self.water_splash_mesh_vertices = 0
            return vertices, indices
        brick_size = float(self.water_mesh_cfg.get("splash_brick_size", 12.0))
        enter = max(2, int(self.water_mesh_cfg.get("splash_enter_particles", 48)))
        keep = max(2, int(self.water_mesh_cfg.get("splash_keep_particles", enter // 2)))
        keys, _ = select_splash_bricks(
            positions, excluded, brick_size, enter, keep,
            self.water_splash_active_keys,
            max(0, int(self.water_mesh_cfg.get("maximum_splash_bricks", 6))),
        )
        self.water_splash_active_keys = set(keys)
        self.water_splash_brick_count = len(keys)
        self.water_splash_mesh_vertices = 0
        if not keys:
            return vertices, indices

        outlier_positions = positions[excluded]
        outlier_radii = surface_radii[excluded]
        candidate_keys = np.floor(outlier_positions / brick_size).astype(np.int32)
        local_voxel = float(self.water_mesh_cfg.get("splash_voxel_size", 0.4))
        margin_cells = max(1, int(self.water_mesh_cfg.get("splash_margin_cells", 2)))
        local_vertices = []
        local_indices = []
        vertex_offset = len(vertices)
        for key in keys:
            key_array = np.asarray(key, dtype=np.int32)
            selected = np.all(candidate_keys == key_array, axis=1)
            sample_positions = outlier_positions[selected]
            sample_radii = outlier_radii[selected]
            if len(sample_positions) < keep:
                continue
            lower = key_array.astype(np.float32) * brick_size - local_voxel * margin_cells
            upper = (key_array.astype(np.float32) + 1.0) * brick_size + local_voxel * margin_cells
            shape = tuple(
                int(v) for v in np.maximum(
                    3, np.rint((upper - lower) / local_voxel).astype(np.int32) + 1
                )
            )
            upper = lower + (np.asarray(shape, dtype=np.float32) - 1.0) * local_voxel
            local_x = wp.array(sample_positions, dtype=wp.vec3, device=self.device)
            local_radius = wp.array(sample_radii, dtype=float, device=self.device)
            local_kind = wp.zeros(len(sample_positions), dtype=wp.int32, device=self.device)
            local_mask = wp.ones(len(sample_positions), dtype=wp.int32, device=self.device)
            local_field = wp.zeros(shape, dtype=float, device=self.device)
            wp.launch(
                splat_sparse_surface_field, dim=len(sample_positions),
                inputs=[local_x, local_radius, local_kind, local_mask, local_field,
                        wp.vec3(*lower), local_voxel, *shape], device=self.device,
            )
            local_v, local_i = wp.MarchingCubes.extract_surface_marching_cubes(
                local_field,
                threshold=float(self.water_mesh_cfg.get("splash_iso_threshold", 1.35)),
                domain_bounds_lower_corner=tuple(float(v) for v in lower),
                domain_bounds_upper_corner=tuple(float(v) for v in upper),
            )
            if len(local_i) < 3:
                continue
            local_vertices.append(local_v.numpy())
            local_indices.append(local_i.numpy().astype(np.int32) + vertex_offset)
            vertex_offset += len(local_v)
            self.water_splash_mesh_vertices += len(local_v)
        if not local_vertices:
            return vertices, indices
        combined_vertices = np.concatenate([vertices.numpy(), *local_vertices], axis=0)
        combined_indices = np.concatenate([indices.numpy(), *local_indices], axis=0).astype(np.int32)
        return (
            wp.array(combined_vertices, dtype=wp.vec3, device=self.device),
            wp.array(combined_indices, dtype=wp.int32, device=self.device),
        )

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
            fragment_schema_version=np.int32(
                self.v3_cfg["fragment_clustering"].get("schema_version", 1)
            ),
            building_active=self.building_active.numpy(),
            building_activation_exposure_seconds=self.building_activation_exposure.numpy(),
            base_fixed=self.base_fixed[:self.count].numpy(),
            material_impact_impulse=self.arrays["material_impact_impulse"][:self.count].numpy(),
            local_impact_active=self.arrays["local_impact_active"][:self.count].numpy(),
            adaptive_merged_groups_total=np.int64(self.adaptive_merged_groups_total),
            adaptive_merged_particles_total=np.int64(self.adaptive_merged_particles_total),
            support_edge_fragments=self.fragment_support_graph.edge_fragments,
            support_sample_offsets=self.fragment_support_graph.sample_offsets,
            support_sample_pairs=self.fragment_support_graph.sample_pairs,
            support_sample_rest_length=self.fragment_support_graph.sample_rest_length,
            support_anchored_fragments=self.fragment_support_graph.anchored_fragments,
            fragment_support_state=self.fragment_support_host,
            support_edge_intact_state=self.fragment_edge_intact_host,
            support_edge_fracture_energy=self.fragment_edge_fracture_energy_host,
            fragment_fracture_energy_state=self.fragment_fracture_energy_host,
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
            rigid_proxy_enabled=self.rigid_proxy_enabled_host,
            rigid_proxy_local_center=self.rigid_proxy_local_center_host,
            rigid_proxy_half_extent=self.rigid_proxy_half_extent_host,
            rigid_proxy_material=self.rigid_proxy_material_host,
            rigid_reactivated_total=self.rigid_reactivated_counter.numpy(),
            water_mesh_domain_lower=(
                self.water_mesh_domain_lower if self.water_mesh_domain_lower is not None
                else np.zeros(3, dtype=np.float32)
            ),
            water_mesh_domain_upper=(
                self.water_mesh_domain_upper if self.water_mesh_domain_upper is not None
                else np.zeros(3, dtype=np.float32)
            ),
            water_mesh_voxel_size=np.float32(self.water_mesh_voxel_size),
            water_mesh_lod_recovery_count=np.int32(self.water_mesh_lod_recovery_count),
            water_mesh_lod_change_count=np.int32(self.water_mesh_lod_change_count),
            shallow_water_state=self.shallow_water.state.numpy(),
            shallow_water_accumulated_dt=np.float32(self.shallow_water.accumulated_dt),
            shallow_emitted_particles_total=np.int64(self.shallow_water.emitted_particles_total),
            shallow_emitted_volume_total=np.float64(self.shallow_water.emitted_volume_total),
            shallow_merged_particles_total=np.int64(self.shallow_water.merged_particles_total),
            shallow_merged_volume_total=np.float64(self.shallow_water.merged_volume_total),
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
        proxy_enabled = np.zeros(body_capacity, dtype=np.int32)
        proxy_local_center = np.zeros((body_capacity, 3), dtype=np.float32)
        proxy_half_extent = np.zeros((body_capacity, 3), dtype=np.float32)
        proxy_material = np.ones(body_capacity, dtype=np.int32)
        restored_proxy_state = False
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
                restore("rigid_proxy_enabled", proxy_enabled)
                restore("rigid_proxy_local_center", proxy_local_center)
                restore("rigid_proxy_half_extent", proxy_half_extent)
                restore("rigid_proxy_material", proxy_material)
                restored_proxy_state = "rigid_proxy_enabled" in saved

        proxy_policy = self.v3_cfg.get("rigid_clusters", {}).get("collision_proxy", {})
        if bool(proxy_policy.get("enabled", True)) and not restored_proxy_state:
            fragment = self.fragment_host
            radius = self.arrays["radius"][:self.count].numpy()
            material = self.arrays["material"][:self.count].numpy()
            particle_mass = self.arrays["mass"][:self.count].numpy()
            minimum_particles = int(proxy_policy.get("minimum_particles", 12))
            for fid in np.flatnonzero(state != 0):
                indices = np.flatnonzero(fragment == fid)
                if len(indices) < minimum_particles:
                    continue
                proxy = fit_rigid_collision_proxy(
                    local[indices], radius[indices], material[indices], particle_mass[indices],
                    float(proxy_policy.get("padding_scale", 0.70)),
                )
                proxy_enabled[fid] = 1
                proxy_local_center[fid] = proxy.local_center
                proxy_half_extent[fid] = proxy.half_extent
                proxy_material[fid] = proxy.material

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
        self.rigid_proxy_enabled_host = proxy_enabled
        self.rigid_proxy_local_center_host = proxy_local_center
        self.rigid_proxy_half_extent_host = proxy_half_extent
        self.rigid_proxy_material_host = proxy_material
        self.rigid_proxy_enabled = wp.array(proxy_enabled, dtype=wp.int32, device=self.device)
        self.rigid_proxy_local_center = wp.array(
            proxy_local_center, dtype=wp.vec3, device=self.device
        )
        self.rigid_proxy_half_extent = wp.array(
            proxy_half_extent, dtype=wp.vec3, device=self.device
        )
        self.rigid_proxy_material = wp.array(proxy_material, dtype=wp.int32, device=self.device)
        self._refresh_collision_proxy_pairs()
        self.body_force = wp.zeros((body_capacity, 3), dtype=float, device=self.device)
        self.body_torque = wp.zeros((body_capacity, 3), dtype=float, device=self.device)
        self.rigid_stats_calls = 0
        self.last_rigid_count = int(np.count_nonzero(state))
        self.rigid_active_count = self.last_rigid_count
        self.rigid_contact_acceleration_peak = wp.zeros(body_capacity, dtype=float, device=self.device)
        self.rigid_reactivated_counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.last_rigid_reactivated_count = 0
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if "rigid_reactivated_total" in saved:
                    restored = saved["rigid_reactivated_total"].astype(np.int32, copy=True)
                    self.rigid_reactivated_counter = wp.array(restored, dtype=wp.int32, device=self.device)
                    self.last_rigid_reactivated_count = int(restored[0])

    def _refresh_collision_proxy_pairs(self):
        proxy_ids = np.flatnonzero(self.rigid_proxy_enabled_host != 0).astype(np.int32)
        if len(proxy_ids) >= 2:
            left_index, right_index = np.triu_indices(len(proxy_ids), 1)
            pair_left = proxy_ids[left_index]
            pair_right = proxy_ids[right_index]
        else:
            pair_left = np.zeros(1, dtype=np.int32)
            pair_right = np.zeros(1, dtype=np.int32)
        self.rigid_proxy_pair_count = len(proxy_ids) * (len(proxy_ids) - 1) // 2
        self.rigid_proxy_pair_left = wp.array(pair_left, dtype=wp.int32, device=self.device)
        self.rigid_proxy_pair_right = wp.array(pair_right, dtype=wp.int32, device=self.device)
        self.rigid_proxy_active_count = int(np.count_nonzero(
            (self.rigid_proxy_enabled_host != 0) & (self.rigid_state_host != 0)
        ))

    def update_rigid_clusters(self):
        policy = self.v3_cfg.get("rigid_clusters", {})
        if not bool(policy.get("enabled", True)) or self.fragment_count == 0:
            return
        # A strong rubble collision may switch a body back to its cohesive
        # deformable fragment on the GPU. Preserve that transition before the
        # host-side quiet-fragment scanner uploads new rigid conversions.
        self.rigid_state_host[:] = self.rigid_state.numpy()
        self.rigid_active_count = int(np.count_nonzero(self.rigid_state_host))
        self.rigid_proxy_active_count = int(np.count_nonzero(
            (self.rigid_proxy_enabled_host != 0) & (self.rigid_state_host != 0)
        ))
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
        radius = self.arrays["radius"][:self.count].numpy()
        material = self.arrays["material"][:self.count].numpy()
        fully_damaged_threshold = float(policy.get("fully_damaged_threshold", 0.95))
        release_fraction = float(policy.get("release_damage_fraction", 0.12))
        minimum_particles = int(policy.get("minimum_particles", 6))
        maximum_residual = float(policy.get("maximum_internal_velocity_rms", 2.5))
        required_quiet = int(policy.get("required_quiet_scans", 2))
        proxy_policy = policy.get("collision_proxy", {})
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
            if (
                bool(proxy_policy.get("enabled", True))
                and len(indices) >= int(proxy_policy.get("minimum_particles", 12))
            ):
                proxy = fit_rigid_collision_proxy(
                    fit.local_positions,
                    radius[indices],
                    material[indices],
                    mass[indices],
                    float(proxy_policy.get("padding_scale", 0.70)),
                )
                self.rigid_proxy_enabled_host[fid] = 1
                self.rigid_proxy_local_center_host[fid] = proxy.local_center
                self.rigid_proxy_half_extent_host[fid] = proxy.half_extent
                self.rigid_proxy_material_host[fid] = proxy.material

        self.rigid_state = wp.array(self.rigid_state_host, dtype=wp.int32, device=self.device)
        self.rigid_local_position = wp.array(self.rigid_local_host, dtype=wp.vec3, device=self.device)
        self.body_center = wp.array(center, dtype=wp.vec3, device=self.device)
        self.body_orientation = wp.array(orientation, dtype=wp.quat, device=self.device)
        self.body_linear_velocity = wp.array(linear_velocity, dtype=wp.vec3, device=self.device)
        self.body_angular_velocity = wp.array(angular_velocity, dtype=wp.vec3, device=self.device)
        self.body_mass = wp.array(body_mass, dtype=float, device=self.device)
        self.body_inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33, device=self.device)
        self.rigid_proxy_enabled = wp.array(
            self.rigid_proxy_enabled_host, dtype=wp.int32, device=self.device
        )
        self.rigid_proxy_local_center = wp.array(
            self.rigid_proxy_local_center_host, dtype=wp.vec3, device=self.device
        )
        self.rigid_proxy_half_extent = wp.array(
            self.rigid_proxy_half_extent_host, dtype=wp.vec3, device=self.device
        )
        self.rigid_proxy_material = wp.array(
            self.rigid_proxy_material_host, dtype=wp.int32, device=self.device
        )
        self._refresh_collision_proxy_pairs()
        self.rigid_active_count = int(np.count_nonzero(self.rigid_state_host))
        print(
            f"  V3 rigid conversion: +{len(converted)} clusters / "
            f"{sum(len(indices) for _, indices, _ in converted):,} surface particles"
        )

    def substep(self, dt: float):
        # V2 computes fluid pressure against sleeping buildings as fixed
        # boundaries. The resulting reaction force is then used to wake a whole
        # structural graph for the following substep.
        a = self.arrays
        view = a["x"][:self.count]
        self.grid.build(view, self.max_support)
        wp.launch(clear_vec3, dim=self.count, inputs=[a["solid_force"][:self.count]], device=self.device)
        shallow_policy = self.v3_cfg.get("shallow_water", {})
        merge_enabled = bool(shallow_policy.get("merge_sph", False))
        particle_z_min = (
            float(shallow_policy.get("sph_z_min", self.cfg["reservoir_z_min"]))
            - (float(shallow_policy.get("merge_buffer_depth", 3.0)) if merge_enabled else 0.0)
            if bool(shallow_policy.get("enabled", False))
            and bool(shallow_policy.get("replace_far_sph", False))
            else float(self.cfg["reservoir_z_min"])
        )
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
        self.shallow_water.couple(a, self.count, dt)
        wp.launch(
            accumulate_material_impact, dim=self.count,
            inputs=[a["kind"][:self.count], a["structural_class"][:self.count],
                    a["mass"][:self.count], a["solid_force"][:self.count],
                    a["material_impact_impulse"][:self.count],
                    a["local_impact_active"][:self.count], dt],
            device=self.device,
        )
        self.building_damage_integral.zero_()
        wp.launch(
            accumulate_building_damage, dim=self.count,
            inputs=[a["kind"][:self.count], a["building_id"][:self.count],
                    a["volume"][:self.count], a["damage"][:self.count],
                    self.building_damage_integral], device=self.device,
        )
        wp.launch(
            compute_clustered_solid_forces, dim=self.count,
            inputs=[self.grid.id, view, a["rest_x"][:self.count], a["v"][:self.count], a["radius"][:self.count],
                    a["mass"][:self.count], a["kind"][:self.count], a["material"][:self.count],
                    a["structural_class"][:self.count], a["building_id"][:self.count],
                    self.building_damage_integral, self.building_structural_volume,
                    self.fragment_id[:self.count], self.rigid_state, self.fragment_support,
                    a["material_impact_impulse"][:self.count],
                    a["fixed"][:self.count],
                    a["damage"][:self.count], a["solid_force"][:self.count], a["acceleration"][:self.count],
                    self.max_support, dt, float(clustering.get("internal_stiffness_multiplier", 2.0)),
                    float(clustering.get("damage_rate", 1.5)),
                    float(clustering.get("propagation_threshold", 0.65)),
                    float(clustering.get("max_damage_per_substep", 0.0004)),
                    float(clustering.get("fracture_reference_spacing", 0.65)) * 0.48 * 1.25,
                    float(clustering.get("collapse_gravity_damage_onset", 0.015)),
                    float(clustering.get("collapse_gravity_damage_full", 0.10)),
                    float(clustering.get("facade_support_loss_minimum_elevation", 4.0)),
                    float(clustering.get("facade_support_loss_collapse_threshold", 0.75)),
                    float(clustering.get("facade_support_loss_damage_rate", 1.0)),
                    float(clustering.get("facade_unsupported_damage_rate", 0.75)),
                    float(clustering.get("elastic_force_cap_multiplier", 1.25)),
                    float(clustering.get("compression_force_cap_multiplier", 2.0))],
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
                        self.rigid_proxy_enabled,
                        a["acceleration"][:self.count], self.body_center, self.body_force, self.body_torque,
                        float(self.cfg["domain_width"]) * 0.5, float(self.cfg["reservoir_z_min"]),
                        float(self.cfg["domain_z_max"]), float(self.cfg["domain_y_max"]),
                        float(rigid_policy.get("boundary_stiffness", 4.0e6)),
                        float(rigid_policy.get("boundary_damping", 1.8e4))],
                device=self.device,
            )
            if self.rigid_proxy_active_count > 0:
                wp.launch(
                    accumulate_rigid_proxy_boundaries, dim=max(1, self.fragment_count),
                    inputs=[self.rigid_state, self.rigid_proxy_enabled,
                            self.rigid_proxy_local_center, self.rigid_proxy_half_extent,
                            self.rigid_proxy_material, self.body_center, self.body_orientation,
                            self.body_linear_velocity, self.body_angular_velocity,
                            self.body_force, self.body_torque,
                            float(self.cfg["domain_width"]) * 0.5,
                            float(self.cfg["reservoir_z_min"]),
                            float(self.cfg["domain_z_max"]), float(self.cfg["domain_y_max"]),
                            float(rigid_policy.get("boundary_stiffness", 4.0e6)),
                            float(rigid_policy.get("boundary_damping", 1.8e4)),
                            float(rigid_policy.get("contact_tangential_damping", 1800.0)),
                            float(rigid_policy.get("collision_proxy", {}).get(
                                "maximum_penetration", 0.35
                            ))],
                    device=self.device,
                )
            self.rigid_contact_acceleration_peak.zero_()
            wp.launch(
                accumulate_rigid_contacts, dim=self.count,
                inputs=[self.grid.id, view, a["v"][:self.count], a["radius"][:self.count],
                        a["kind"][:self.count], a["material"][:self.count], self.fragment_id[:self.count],
                        self.rigid_state, self.rigid_proxy_enabled,
                        self.body_center, self.body_mass, self.body_force,
                        self.body_torque, self.rigid_contact_acceleration_peak, self.max_support,
                        float(rigid_policy.get("contact_normal_damping", 3200.0)),
                        float(rigid_policy.get("contact_tangential_damping", 1800.0))],
                device=self.device,
            )
            if self.rigid_proxy_pair_count > 0:
                wp.launch(
                    accumulate_rigid_proxy_contacts, dim=self.rigid_proxy_pair_count,
                    inputs=[self.rigid_proxy_pair_left, self.rigid_proxy_pair_right,
                            self.rigid_state, self.rigid_proxy_enabled,
                            self.rigid_proxy_local_center, self.rigid_proxy_half_extent,
                            self.rigid_proxy_material, self.body_center, self.body_orientation,
                            self.body_linear_velocity, self.body_angular_velocity, self.body_mass,
                            self.body_force, self.body_torque,
                            self.rigid_contact_acceleration_peak,
                            float(rigid_policy.get("contact_normal_damping", 3200.0)),
                            float(rigid_policy.get("contact_tangential_damping", 1800.0)),
                            float(rigid_policy.get("collision_proxy", {}).get(
                                "maximum_penetration", 0.35
                            ))],
                    device=self.device,
                )
            wp.launch(
                reactivate_rigid_after_impact, dim=max(1, self.fragment_count),
                inputs=[self.rigid_state, self.rigid_contact_acceleration_peak,
                        self.rigid_reactivated_counter,
                        float(rigid_policy.get("reactivate_acceleration", 120.0))],
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
                        dt, float(self.cfg["domain_width"]) * 0.5, particle_z_min,
                        float(self.cfg["domain_z_max"]), float(self.cfg["domain_y_max"]),
                        float(self.cfg.get("fluid_bed_drag", 0.12)),
                        float(self.cfg.get("maximum_fluid_speed", 0.0)),
                        float(self.cfg.get("maximum_fluid_vertical_speed", 0.0)),
                        float(self.cfg.get("maximum_solid_speed", 0.0)),
                        float(self.cfg.get("maximum_solid_upward_speed", 0.0))], device=self.device,
            )
            self.multirate_tick += 1
        else:
            wp.launch(
                integrate, dim=self.count,
                inputs=[view, a["v"][:self.count], a["acceleration"][:self.count], a["kind"][:self.count],
                        a["fixed"][:self.count], dt, float(self.cfg["domain_width"]) * 0.5,
                        particle_z_min, float(self.cfg["domain_z_max"]),
                        float(self.cfg["domain_y_max"]), float(self.cfg.get("fluid_bed_drag", 0.12)),
                        float(self.cfg.get("maximum_fluid_speed", 0.0)),
                        float(self.cfg.get("maximum_fluid_vertical_speed", 0.0)),
                        float(self.cfg.get("maximum_solid_speed", 0.0)),
                        float(self.cfg.get("maximum_solid_upward_speed", 0.0))],
                device=self.device,
            )
        self.time += dt
        self.shallow_water.advance(dt, float(self.cfg["rest_density"]))
        if self.building_count == 0:
            return
        wp.launch(clear_int, dim=self.building_count, inputs=[self.activation_hits], device=self.device)
        wp.launch(
            count_loaded_building_particles, dim=self.count,
            inputs=[self.arrays["rest_x"][:self.count], self.arrays["kind"][:self.count],
                    self.arrays["building_id"][:self.count],
                    self.arrays["mass"][:self.count], self.arrays["solid_force"][:self.count],
                    float(self.v3_cfg.get("activation_force_per_mass", 5.0)),
                    float(self.v3_cfg.get("maximum_activation_elevation", 8.0)),
                    self.activation_hits],
            device=self.device,
        )
        wp.launch(
            activate_buildings_from_hits, dim=self.building_count,
            inputs=[self.activation_hits, self.building_active, self.building_activation_exposure,
                    int(self.v3_cfg.get("minimum_contact_particles", 12)), dt,
                    float(self.v3_cfg.get("activation_required_seconds", 0.02)),
                    float(self.v3_cfg.get("activation_exposure_decay_multiplier", 4.0))],
            device=self.device,
        )
        wp.launch(
            apply_building_activity, dim=self.count,
            inputs=[self.arrays["kind"][:self.count], self.arrays["building_id"][:self.count],
                    self.arrays["structural_class"][:self.count], self.base_fixed[:self.count],
                    self.building_active, self.arrays["local_impact_active"][:self.count],
                    self.arrays["fixed"][:self.count]],
            device=self.device,
        )

    def refine(self):
        # Water keeps V2's conservative 1->8 volume refinement. Structural
        # surfaces use planar 1->4 refinement so a thin wall does not become a
        # volumetric cloud when resolution increases near an impact.
        super().refine()
        # Water refinement appends particles with fragment_id=-1.  Keep the
        # host mirror aligned even on frames where no structural child is
        # created; support diagnostics and later compaction use full-length
        # particle masks.
        self.fragment_host = self.fragment_id[:self.count].numpy()
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
                    self.arrays["rho_reference"], self.arrays["solid_force"],
                    self.arrays["material_impact_impulse"], self.arrays["local_impact_active"],
                    self.fragment_id, self.normal_axis,
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
        self._merge_adaptive_fluid_groups()
        self._merge_sph_interface_particles()
        self._emit_shallow_interface_particles()
        self.update_rigid_clusters()
        result = super().stats()
        kind_host = self.arrays["kind"][:self.count].numpy()
        building_host = self.arrays["building_id"][:self.count].numpy()
        fluid_mask = kind_host == 0
        volume_host = self.arrays["volume"][:self.count].numpy()
        result["fluid_volume_m3"] = float(np.sum(volume_host[fluid_mask], dtype=np.float64))
        mass_host = self.arrays["mass"][:self.count].numpy()
        radius_host = self.arrays["radius"][:self.count].numpy()
        result["invalid_zero_volume_particles"] = int(np.count_nonzero(
            (mass_host <= 0.0) | (volume_host <= 0.0)
        ))
        fine_fluid_mask = fluid_mask & (
            radius_host <= float(self.cfg["fine_spacing"]) * 0.5 * 1.25
        )
        result["fine_fluid_particles"] = int(np.count_nonzero(fine_fluid_mask))
        result["coarse_fluid_particles"] = int(np.count_nonzero(fluid_mask & ~fine_fluid_mask))
        result["fine_fluid_volume_percent"] = float(
            100.0 * np.sum(volume_host[fine_fluid_mask], dtype=np.float64)
            / max(np.sum(volume_host[fluid_mask], dtype=np.float64), 1.0e-9)
        )
        result["adaptive_merged_groups"] = self.adaptive_merged_groups_total
        result["adaptive_merged_particles"] = self.adaptive_merged_particles_total
        velocity_host = self.arrays["v"][:self.count].numpy()
        position_host = self.arrays["x"][:self.count].numpy()
        fixed_host = self.arrays["fixed"][:self.count].numpy()
        result["fluid_momentum_z_kg_m_s"] = float(
            np.sum(mass_host[fluid_mask] * velocity_host[fluid_mask, 2], dtype=np.float64)
        )
        fluid_height = position_host[fluid_mask, 1]
        fluid_vertical_velocity = velocity_host[fluid_mask, 1]
        if len(fluid_height):
            result["fluid_height_p99_m"] = float(np.quantile(fluid_height, 0.99))
            result["fluid_height_p999_m"] = float(np.quantile(fluid_height, 0.999))
            result["fluid_height_max_m"] = float(np.max(fluid_height))
            result["fluid_vertical_speed_max_m_s"] = float(
                np.max(np.abs(fluid_vertical_velocity))
            )
            result["fluid_particles_above_30m"] = int(np.count_nonzero(fluid_height > 30.0))
            result["fluid_particles_above_42m"] = int(np.count_nonzero(fluid_height > 42.0))
            result["fluid_particles_above_60m"] = int(np.count_nonzero(fluid_height > 60.0))
        movable_solid_mask = (kind_host != 0) & (fixed_host == 0)
        if np.any(movable_solid_mask):
            solid_velocity = velocity_host[movable_solid_mask]
            solid_speed = np.linalg.norm(solid_velocity, axis=1)
            solid_mass = mass_host[movable_solid_mask]
            result["solid_speed_p99_m_s"] = float(np.quantile(solid_speed, 0.99))
            result["solid_speed_max_m_s"] = float(np.max(solid_speed))
            result["solid_upward_speed_max_m_s"] = float(np.max(solid_velocity[:, 1]))
            result["solid_mass_upward_above_10m_s_percent"] = float(
                100.0 * np.sum(solid_mass[solid_velocity[:, 1] > 10.0], dtype=np.float64)
                / max(np.sum(solid_mass, dtype=np.float64), 1.0)
            )
        structural_role = self.arrays["structural_class"][:self.count].numpy()
        damage_values = self.arrays["damage"][:self.count].numpy()
        impact_values = self.arrays["material_impact_impulse"][:self.count].numpy()
        local_impact_values = self.arrays["local_impact_active"][:self.count].numpy()
        result["local_impact_glass_particles"] = int(np.count_nonzero(
            (structural_role == 6) & (local_impact_values != 0)
        ))
        result["material_impact_impulse_max_m_s"] = float(
            np.max(impact_values[kind_host != 0]) if np.any(kind_host != 0) else 0.0
        )
        material_host = self.arrays["material"][:self.count].numpy()
        self._update_fragment_support_graph(
            position_host, damage_values, material_host, structural_role
        )
        for renderer in self.renderers.values():
            renderer.fragment_support = self.fragment_support
            renderer.fragment_fracture_energy = self.fragment_fracture_energy
        damaged_mask = damage_values > 0.05
        for role, role_name in (
            (1, "slab"), (2, "wall"), (3, "beam"), (4, "column"), (5, "core"), (6, "glass")
        ):
            role_mask = structural_role == role
            role_damaged_mask = damaged_mask & role_mask
            # Keep the legacy count for debugging, but use the conserved
            # volume metrics for physical comparisons across adaptive LODs.
            result[f"damaged_{role_name}_particles"] = int(np.count_nonzero(role_damaged_mask))
            result[f"structural_{role_name}_volume_m3"] = float(
                np.sum(volume_host[role_mask], dtype=np.float64)
            )
            result[f"damaged_{role_name}_volume_m3"] = float(
                np.sum(volume_host[role_damaged_mask], dtype=np.float64)
            )
            result[f"damage_integral_{role_name}_m3"] = float(
                np.sum(volume_host[role_mask] * damage_values[role_mask], dtype=np.float64)
            )
        active_count = int(np.count_nonzero(self.building_active.numpy())) if self.building_count else 0
        if active_count != self.last_active_count:
            print(f"  V3 building activation: {self.last_active_count} -> {active_count}")
            self.last_active_count = active_count
        result["active_buildings"] = active_count
        building_volume = self.building_structural_volume.numpy()
        particle_fragment = self.fragment_id[:self.count].numpy()
        valid_support_particle = (kind_host != 0) & (particle_fragment >= 0)
        unsupported_particle = valid_support_particle.copy()
        unsupported_particle[valid_support_particle] = ~self.fragment_support_host[
            particle_fragment[valid_support_particle]
        ].astype(bool)
        unsupported_volume = np.bincount(
            building_host[unsupported_particle],
            weights=volume_host[unsupported_particle].astype(np.float64, copy=False),
            minlength=max(1, self.building_count),
        ) if np.any(unsupported_particle) else np.zeros(max(1, self.building_count))
        building_gravity_fraction = np.divide(
            unsupported_volume, np.maximum(building_volume, 1.0e-6)
        )
        result["collapse_gravity_buildings"] = int(np.count_nonzero(building_gravity_fraction > 0.0))
        result["structural_collapse_gravity_max"] = float(
            np.max(building_gravity_fraction) if len(building_gravity_fraction) else 0.0
        )
        result["unsupported_fragments"] = int(
            np.count_nonzero(self.fragment_support_host < 0.5)
        )
        result["support_graph_edges"] = int(len(self.fragment_edge_intact_host))
        result["support_graph_intact_edges"] = int(
            np.count_nonzero(self.fragment_edge_intact_host)
        )
        result["fracture_energy_edges_visible"] = int(
            np.count_nonzero(self.fragment_edge_fracture_energy_host > 0.01)
        )
        result["fracture_energy_edge_max"] = float(
            np.max(self.fragment_edge_fracture_energy_host)
            if len(self.fragment_edge_fracture_energy_host) else 0.0
        )
        result["cohesive_fragments"] = self.fragment_count
        damage_host = self.arrays["damage"][:len(self.fragment_host)].numpy()
        solid_mask = self.fragment_host >= 0
        fully_damaged = solid_mask & (damage_host >= 0.95)
        released_hits = np.bincount(
            self.fragment_host[fully_damaged], minlength=self.fragment_count
        ) if self.fragment_count else np.empty(0, dtype=np.int64)
        release_fraction = float(self.v3_cfg["fragment_clustering"].get("release_damage_fraction", 0.12))
        released_mask = (
            released_hits >= np.maximum(2, np.ceil(self.fragment_counts_host * release_fraction))
        ) if self.fragment_count else np.empty(0, dtype=bool)
        released = int(np.count_nonzero(released_mask))
        if released != self.last_released_fragment_count:
            print(f"  V3 released cohesive fragments: {self.last_released_fragment_count} -> {released}")
            self.last_released_fragment_count = released
        result["released_fragments"] = released
        for role, role_name in (
            (1, "slab"), (2, "wall"), (3, "beam"), (4, "column"), (5, "core"), (6, "glass")
        ):
            result[f"released_{role_name}_fragments"] = int(
                np.count_nonzero(released_mask & (self.fragment_role_host == role))
            )
        rigid_count = int(np.count_nonzero(self.rigid_state_host))
        rigid_particles = int(np.count_nonzero(
            (self.fragment_host >= 0) & self.rigid_state_host[np.maximum(self.fragment_host, 0)]
        )) if self.fragment_count else 0
        if rigid_count != self.last_rigid_count:
            print(f"  V3 active rigid clusters: {self.last_rigid_count} -> {rigid_count}")
            self.last_rigid_count = rigid_count
        result["rigid_clusters"] = rigid_count
        result["rigid_particles"] = rigid_particles
        result["rigid_collision_proxies"] = int(np.count_nonzero(
            (self.rigid_proxy_enabled_host != 0) & (self.rigid_state_host != 0)
        ))
        result["rigid_proxy_pairs"] = self.rigid_proxy_pair_count
        reactivated_total = int(self.rigid_reactivated_counter.numpy()[0])
        if reactivated_total != self.last_rigid_reactivated_count:
            print(
                f"  V3 rigid -> deformable after impact: "
                f"+{reactivated_total - self.last_rigid_reactivated_count} fragments"
            )
            self.last_rigid_reactivated_count = reactivated_total
        result["rigid_reactivated_fragments"] = reactivated_total
        if self.multirate_enabled:
            levels = self.time_level[:self.count].numpy()
            for level in range(3):
                result[f"time_level_{level}_particles"] = int(
                    np.count_nonzero(fluid_mask & (levels == level))
                )
        self.update_water_surface()
        if self.surface_enabled:
            surface_mask = self.arrays["water_surface_mask"][:self.count].numpy()
            result["surface_water_particles"] = int(np.count_nonzero(surface_mask))
            result["water_mesh_vertices"] = len(self.water_mesh_vertices)
            result["water_mesh_triangles"] = self.water_mesh_triangle_count
            result["water_field_nodes"] = int(np.prod(self.water_field_shape, dtype=np.int64))
            result["water_field_nx"] = int(self.water_field_shape[0])
            result["water_field_ny"] = int(self.water_field_shape[1])
            result["water_field_nz"] = int(self.water_field_shape[2])
            result["water_mesh_excluded_surface_particles"] = self.water_mesh_excluded_surface_count
            result["water_mesh_voxel_millimeters"] = int(round(self.water_mesh_voxel_size * 1000.0))
            if self.water_mesh_domain_lower is not None and self.water_mesh_domain_upper is not None:
                for axis, axis_name in enumerate(("x", "y", "z")):
                    result[f"water_mesh_core_lower_{axis_name}_m"] = float(
                        self.water_mesh_domain_lower[axis]
                    )
                    result[f"water_mesh_core_upper_{axis_name}_m"] = float(
                        self.water_mesh_domain_upper[axis]
                    )
            if self.water_mesh_field_lower is not None and self.water_mesh_voxel_size > 0.0:
                field_span = (
                    (np.asarray(self.water_field_shape, dtype=np.float64) - 1.0)
                    * self.water_mesh_voxel_size
                )
                result["water_mesh_span_x_m"] = float(field_span[0])
                result["water_mesh_span_y_m"] = float(field_span[1])
                result["water_mesh_span_z_m"] = float(field_span[2])
                for axis, axis_name in enumerate(("x", "y", "z")):
                    result[f"water_mesh_lower_{axis_name}_m"] = float(
                        self.water_mesh_field_lower[axis]
                    )
                    result[f"water_mesh_upper_{axis_name}_m"] = float(
                        self.water_mesh_field_lower[axis] + field_span[axis]
                    )
            result["water_mesh_lod_changes"] = self.water_mesh_lod_change_count
            result["water_splash_bricks"] = self.water_splash_brick_count
            result["water_splash_mesh_vertices"] = self.water_splash_mesh_vertices
            result["water_stitch_surface_samples"] = self.water_stitch_sample_count
            result["water_surface_classify_ms"] = self.water_surface_classify_ms
            result["water_mesh_preprocess_ms"] = self.water_mesh_preprocess_ms
            result["water_mesh_field_ms"] = self.water_mesh_field_ms
            result["water_mesh_marching_cubes_ms"] = self.water_mesh_marching_cubes_ms
            result["water_mesh_splash_ms"] = self.water_mesh_splash_ms
            result["water_mesh_total_ms"] = self.water_mesh_total_ms
            result.update(self.shallow_water.diagnostics())
        return result

    def _emit_shallow_interface_particles(self):
        policy = self.v3_cfg.get("shallow_water", {})
        if not (
            bool(policy.get("enabled", False))
            and bool(policy.get("replace_far_sph", False))
            and bool(policy.get("emit_sph", True))
        ):
            return
        spacing = float(policy.get("emitter_spacing", self.cfg.get("coarse_spacing", 1.0)))
        emitter_nx = max(1, int(np.floor(float(self.cfg["domain_width"]) / spacing)))
        emitter_ny = max(
            1,
            int(np.ceil((float(self.cfg["water_depth"]) + float(self.cfg["wave_height"])) / spacing)),
        )
        old_count = self.count
        self.grid.build(self.arrays["x"][:old_count], self.max_support)
        counter = wp.array(np.asarray([old_count], dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            emit_sph_interface_particles, dim=(emitter_nx, emitter_ny),
            inputs=[self.grid.id, self.arrays["x"], self.arrays["rest_x"], self.arrays["v"],
                    self.arrays["radius"], self.arrays["mass"], self.arrays["volume"],
                    self.arrays["kind"], self.arrays["material"], self.arrays["building_id"],
                    self.arrays["structural_class"], self.arrays["fixed"], self.arrays["damage"],
                    self.arrays["material_impact_impulse"], self.arrays["local_impact_active"],
                    self.arrays["rho_reference"], self.arrays["rho"], self.arrays["acceleration"],
                    self.arrays["solid_force"], self.base_fixed, self.fragment_id, self.normal_axis,
                    self.time_level, self.time_active, self.arrays["water_surface_mask"],
                    self.arrays["water_surface_normal"], self.arrays["water_foam_strength"],
                    self.arrays["fluid_group_id"],
                    self.shallow_water.state, self.shallow_water.exchange_volume,
                    self.shallow_water.exchange_x, self.shallow_water.exchange_z,
                    counter, old_count, self.capacity, emitter_nx, emitter_ny,
                    self.shallow_water.lower_x, self.shallow_water.lower_z,
                    self.shallow_water.interface_z, self.shallow_water.cell_size,
                    self.shallow_water.nx, self.shallow_water.nz, spacing,
                    float(self.cfg["rest_density"]),
                    float(policy.get("minimum_emission_velocity", 0.25))], device=self.device,
        )
        wp.synchronize_device(self.device)
        self.count = min(int(counter.numpy()[0]), self.capacity)
        emitted = self.count - old_count
        if emitted > 0:
            self.shallow_water.emitted_particles_total += emitted
            self.shallow_water.emitted_volume_total += emitted * spacing ** 3
            self.shallow_water.commit_exchange(float(self.cfg["rest_density"]))
            print(
                f"  V3 shallow -> SPH emission: +{emitted:,} particles "
                f"({self.shallow_water.emitted_particles_total:,} total)"
            )

    def _ensure_particle_compaction_scratch(self):
        if self.particle_compaction_scratch is not None:
            return
        array_names = (
            "x", "rest_x", "v", "radius", "mass", "volume", "kind", "material",
            "building_id", "structural_class", "fixed", "damage", "rho_reference", "rho",
            "material_impact_impulse", "local_impact_active",
            "fluid_group_id",
            "acceleration", "solid_force", "water_surface_mask", "water_surface_normal",
            "water_foam_strength",
        )
        self.particle_compaction_scratch = {
            "arrays": {
                name: wp.zeros(self.arrays[name].shape, dtype=self.arrays[name].dtype, device=self.device)
                for name in array_names
            },
            "extra": {
                "base_fixed": wp.zeros(self.capacity, dtype=wp.int32, device=self.device),
                "fragment_id": wp.zeros(self.capacity, dtype=wp.int32, device=self.device),
                "normal_axis": wp.zeros(self.capacity, dtype=wp.int32, device=self.device),
                "time_level": wp.zeros(self.capacity, dtype=wp.int32, device=self.device),
                "time_active": wp.zeros(self.capacity, dtype=wp.int32, device=self.device),
                "rigid_local_position": wp.zeros(self.capacity, dtype=wp.vec3, device=self.device),
                "deferred_fluid_impulse": wp.zeros((self.capacity, 3), dtype=float, device=self.device),
            },
        }

    def _compact_particle_arrays(self, old_count: int):
        self._ensure_particle_compaction_scratch()
        scratch = self.particle_compaction_scratch
        float_names = (
            "radius", "mass", "volume", "damage", "rho_reference", "rho",
            "water_foam_strength", "material_impact_impulse",
        )
        int_names = (
            "kind", "material", "building_id", "structural_class", "fixed", "water_surface_mask",
            "local_impact_active", "fluid_group_id",
        )
        vec3_names = ("x", "rest_x", "v", "acceleration", "solid_force", "water_surface_normal")
        for name in float_names:
            wp.launch(compact_float_particles, dim=old_count,
                      inputs=[self.arrays[name], scratch["arrays"][name], self.return_keep,
                              self.return_offsets], device=self.device)
        for name in int_names:
            if name == "fluid_group_id":
                scratch["arrays"][name].fill_(-1)
            wp.launch(compact_int_particles, dim=old_count,
                      inputs=[self.arrays[name], scratch["arrays"][name], self.return_keep,
                              self.return_offsets], device=self.device)
        for name in vec3_names:
            wp.launch(compact_vec3_particles, dim=old_count,
                      inputs=[self.arrays[name], scratch["arrays"][name], self.return_keep,
                              self.return_offsets], device=self.device)
        for name in float_names + int_names + vec3_names:
            self.arrays[name], scratch["arrays"][name] = scratch["arrays"][name], self.arrays[name]

        extra_int = ("base_fixed", "fragment_id", "normal_axis", "time_level", "time_active")
        for name in extra_int:
            source = getattr(self, name)
            target = scratch["extra"][name]
            if name in ("fragment_id", "normal_axis"):
                target.fill_(-1)
            elif name == "time_active":
                target.fill_(1)
            else:
                target.zero_()
            wp.launch(compact_int_particles, dim=old_count,
                      inputs=[source, target, self.return_keep, self.return_offsets], device=self.device)
            setattr(self, name, target)
            scratch["extra"][name] = source
        source = self.rigid_local_position
        target = scratch["extra"]["rigid_local_position"]
        target.zero_()
        wp.launch(compact_vec3_particles, dim=old_count,
                  inputs=[source, target, self.return_keep, self.return_offsets], device=self.device)
        self.rigid_local_position = target
        scratch["extra"]["rigid_local_position"] = source
        source = self.deferred_fluid_impulse
        target = scratch["extra"]["deferred_fluid_impulse"]
        target.zero_()
        wp.launch(compact_vec3_components, dim=old_count,
                  inputs=[source, target, self.return_keep, self.return_offsets], device=self.device)
        self.deferred_fluid_impulse = target
        scratch["extra"]["deferred_fluid_impulse"] = source
        for renderer in self.renderers.values():
            wp.launch(
                remap_particle_indices, dim=len(renderer.anchor),
                inputs=[renderer.anchor, self.return_keep, self.return_offsets], device=self.device,
            )
        # Shallow-water return compacts the shared particle arrays. Structural
        # particles are retained, but their indices shift when earlier fluid
        # entries are removed; keep the sparse load-path boundary samples in
        # the same index space as the solver and facade anchors.
        graph = self.fragment_support_graph
        if len(graph.sample_pairs):
            old_to_new = self.return_offsets[:old_count].numpy().astype(np.int32, copy=False)
            remapped_pairs = old_to_new[graph.sample_pairs]
            self.fragment_support_graph = FragmentSupportGraph(
                graph.edge_fragments,
                graph.sample_offsets,
                remapped_pairs,
                graph.sample_rest_length,
                graph.anchored_fragments,
            )

    def _merge_adaptive_fluid_groups(self):
        policy = self.v3_cfg.get("adaptive_water_merge", {})
        if not bool(policy.get("enabled", False)) or self.count <= 0:
            return
        completed_frame = max(0, int(round(self.time * float(self.cfg["output_fps"]))) - 1)
        every_frames = max(1, int(policy.get("every_frames", 8)))
        if completed_frame == self.last_adaptive_merge_frame or completed_frame % every_frames != 0:
            return
        self.last_adaptive_merge_frame = completed_frame
        old_count = self.count
        group_host = self.arrays["fluid_group_id"][:old_count].numpy()
        if not np.any(group_host >= 0):
            return
        merge = select_conservative_fluid_merges(
            group_host,
            self.arrays["kind"][:old_count].numpy(),
            self.arrays["x"][:old_count].numpy(),
            self.arrays["v"][:old_count].numpy(),
            self.arrays["mass"][:old_count].numpy(),
            self.arrays["volume"][:old_count].numpy(),
            self.arrays["radius"][:old_count].numpy(),
            maximum_y=(
                float(self.cfg["water_depth"])
                - float(self.cfg.get("fine_surface_band", 0.0))
                - float(policy.get("surface_margin", 0.75))
            ),
            maximum_vertical_speed=float(policy.get("maximum_vertical_speed", 0.8)),
            maximum_velocity_rms=float(policy.get("maximum_velocity_rms", 0.6)),
            maximum_span=float(policy.get("maximum_span", 0.9)),
            maximum_fine_radius=float(self.cfg["fine_spacing"]) * 0.5 * 1.25,
        )
        group_count = len(merge["representatives"])
        if group_count == 0:
            return
        keep_host = np.ones(old_count, dtype=np.int32)
        keep_host[merge["removed"]] = 0
        wp.copy(
            self.return_keep,
            wp.array(keep_host, dtype=wp.int32, device=self.device),
            count=old_count,
        )
        wp.launch(
            apply_conservative_fluid_merges, dim=group_count,
            inputs=[
                wp.array(merge["representatives"], dtype=wp.int32, device=self.device),
                wp.array(merge["position"], dtype=wp.vec3, device=self.device),
                wp.array(merge["velocity"], dtype=wp.vec3, device=self.device),
                wp.array(merge["mass"], dtype=float, device=self.device),
                wp.array(merge["volume"], dtype=float, device=self.device),
                wp.array(merge["radius"], dtype=float, device=self.device),
                self.arrays["x"], self.arrays["rest_x"], self.arrays["v"],
                self.arrays["mass"], self.arrays["volume"], self.arrays["radius"],
                self.arrays["rho_reference"], self.arrays["rho"],
                self.arrays["acceleration"], self.arrays["solid_force"],
                self.arrays["fluid_group_id"], self.arrays["water_surface_mask"],
                self.arrays["water_surface_normal"], self.arrays["water_foam_strength"],
            ],
            device=self.device,
        )
        wp.utils.array_scan(
            self.return_keep[:old_count], self.return_offsets[:old_count], inclusive=False
        )
        self._compact_particle_arrays(old_count)
        wp.synchronize_device(self.device)
        removed_count = group_count * 7
        self.count = old_count - removed_count
        self.fragment_host = self.fragment_id[:self.count].numpy()
        self.rigid_local_host.fill(0.0)
        self.rigid_local_host[:self.count] = self.rigid_local_position[:self.count].numpy()
        self.adaptive_merged_groups_total += group_count
        self.adaptive_merged_particles_total += removed_count
        print(
            f"  V3 adaptive SPH merge: {group_count:,} sibling octets -> coarse "
            f"(-{removed_count:,} particles; {self.adaptive_merged_groups_total:,} groups total)"
        )

    def _merge_sph_interface_particles(self):
        policy = self.v3_cfg.get("shallow_water", {})
        if not (
            bool(policy.get("enabled", False))
            and bool(policy.get("replace_far_sph", False))
            and bool(policy.get("merge_sph", False))
        ):
            return
        old_count = self.count
        # Flush any pending coupling impulse first so the following exchange
        # counters contain only the representation transfer being measured.
        self.shallow_water.commit_exchange(float(self.cfg["rest_density"]))
        merged_volume = wp.zeros(1, dtype=float, device=self.device)
        wp.launch(
            mark_sph_return_particles, dim=old_count,
            inputs=[self.arrays["x"][:old_count], self.arrays["v"][:old_count],
                    self.arrays["mass"][:old_count], self.arrays["volume"][:old_count],
                    self.arrays["kind"][:old_count], self.return_keep[:old_count],
                    self.shallow_water.exchange_volume, self.shallow_water.exchange_x,
                    self.shallow_water.exchange_z, merged_volume, self.shallow_water.lower_x,
                    self.shallow_water.lower_z, self.shallow_water.interface_z,
                    self.shallow_water.cell_size, self.shallow_water.nx, self.shallow_water.nz,
                    float(policy.get("minimum_return_speed", 0.25)),
                    float(policy.get("forced_capture_depth", 0.75))], device=self.device,
        )
        wp.utils.array_scan(
            self.return_keep[:old_count], self.return_offsets[:old_count], inclusive=False
        )
        wp.synchronize_device(self.device)
        tail_keep = int(self.return_keep[old_count - 1:old_count].numpy()[0])
        new_count = int(self.return_offsets[old_count - 1:old_count].numpy()[0]) + tail_keep
        merged = old_count - new_count
        if merged <= 0:
            return
        volume = float(merged_volume.numpy()[0])
        self._compact_particle_arrays(old_count)
        wp.synchronize_device(self.device)
        self.count = new_count
        self.fragment_host = self.fragment_id[:self.count].numpy()
        self.rigid_local_host.fill(0.0)
        self.rigid_local_host[:self.count] = self.rigid_local_position[:self.count].numpy()
        self.shallow_water.commit_exchange(float(self.cfg["rest_density"]))
        self.shallow_water.merged_particles_total += merged
        self.shallow_water.merged_volume_total += volume
        print(
            f"  V3 SPH -> shallow merge: -{merged:,} particles / {volume:,.2f} m3 "
            f"({self.shallow_water.merged_particles_total:,} total)"
        )


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
