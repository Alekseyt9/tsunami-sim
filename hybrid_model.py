"""V3 structural LOD policy and render-skin metadata.

The facade skin is deliberately independent from simulation particles.  The
first V3 renderer will deform these panels from the structural particle graph
instead of drawing every particle as a circle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError, cKDTree
except ImportError:  # The renderer retains its box fallback in minimal environments.
    ConvexHull = None
    QhullError = RuntimeError
    cKDTree = None

from scene import (
    STRUCT_BEAM,
    STRUCT_COLUMN,
    STRUCT_CORE,
    STRUCT_GLASS,
    STRUCT_SLAB,
    STRUCT_WALL,
    building_profile,
    environment_layout,
)


@dataclass(frozen=True)
class SolidRefinementPolicy:
    coarse_spacing: float
    impact_spacing: float
    crack_spacing: float
    pressure_trigger: float
    strain_fraction_trigger: float
    crack_damage_trigger: float

    @classmethod
    def from_config(cls, cfg: dict) -> "SolidRefinementPolicy":
        data = cfg["v3"]["solid_refinement"]
        return cls(**{name: float(data[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class FragmentSupportGraph:
    """Sparse breakable load paths between architectural solid fragments."""

    edge_fragments: np.ndarray
    sample_offsets: np.ndarray
    sample_pairs: np.ndarray
    sample_rest_length: np.ndarray
    anchored_fragments: np.ndarray


def rebaseline_fragment_support_graph(
    graph: FragmentSupportGraph,
    rest_x: np.ndarray,
) -> FragmentSupportGraph:
    """Return ``graph`` with sample lengths measured in the current rest LOD.

    Refinement keeps one child at the parent's particle index, so sparse sample
    indices remain valid while their rest positions change.  Recomputing only
    the undeformed lengths preserves edge topology and irreversible fracture
    state without mistaking the refinement offset for elastic strain.
    """
    if not len(graph.sample_pairs):
        return graph
    pair = graph.sample_pairs
    rest_length = np.linalg.norm(
        rest_x[pair[:, 1]] - rest_x[pair[:, 0]], axis=1
    ).astype(np.float32)
    if not np.all(np.isfinite(rest_length)) or np.any(rest_length <= 1.0e-6):
        raise ValueError("invalid support-graph rest length after structural refinement")
    return FragmentSupportGraph(
        graph.edge_fragments,
        graph.sample_offsets,
        graph.sample_pairs,
        rest_length,
        graph.anchored_fragments,
    )


def build_fragment_support_graph(
    rest_x: np.ndarray,
    radius: np.ndarray,
    kind: np.ndarray,
    building_id: np.ndarray,
    base_fixed: np.ndarray,
    fragment_id: np.ndarray,
    maximum_samples_per_edge: int = 12,
) -> FragmentSupportGraph:
    """Build fragment adjacency and retain representative boundary bonds.

    The particle solver still resolves every local spring.  This much smaller
    graph only answers whether a fragment has an intact load path to a fixed
    foundation fragment, avoiding a nonlinear global FEM solve every substep.
    """
    fragment_count = int(fragment_id[fragment_id >= 0].max()) + 1 if np.any(fragment_id >= 0) else 0
    anchored = np.zeros(fragment_count, dtype=bool)
    anchored_ids = fragment_id[(kind != 0) & (base_fixed != 0) & (fragment_id >= 0)]
    anchored[anchored_ids] = True
    solid = np.flatnonzero((kind != 0) & (building_id >= 0) & (fragment_id >= 0))
    if len(solid) == 0:
        empty2 = np.empty((0, 2), dtype=np.int32)
        return FragmentSupportGraph(
            empty2, np.zeros(1, dtype=np.int32), empty2,
            np.empty(0, dtype=np.float32), anchored,
        )

    maximum_samples_per_edge = max(1, int(maximum_samples_per_edge))
    maximum_bond = 3.2 * float(np.max(radius[solid]))
    bucket_size = max(maximum_bond, 1.0e-4)
    cell = np.floor(rest_x[solid] / bucket_size).astype(np.int32)
    buckets: dict[tuple[int, int, int, int], list[int]] = {}
    for particle, key in zip(solid, cell):
        bucket = (int(building_id[particle]), int(key[0]), int(key[1]), int(key[2]))
        buckets.setdefault(bucket, []).append(int(particle))

    edge_samples: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for particle, key in zip(solid, cell):
        bid = int(building_id[particle])
        fi = int(fragment_id[particle])
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbours = buckets.get(
                        (bid, int(key[0] + dx), int(key[1] + dy), int(key[2] + dz)), ()
                    )
                    for other in neighbours:
                        if other <= particle:
                            continue
                        fj = int(fragment_id[other])
                        if fj == fi:
                            continue
                        delta = rest_x[other] - rest_x[particle]
                        distance = float(np.linalg.norm(delta))
                        bond_range = 3.2 * max(float(radius[particle]), float(radius[other]))
                        if distance <= 1.0e-5 or distance >= bond_range:
                            continue
                        edge = (fi, fj) if fi < fj else (fj, fi)
                        edge_samples.setdefault(edge, []).append((int(particle), int(other), distance))

    edges = np.asarray(sorted(edge_samples), dtype=np.int32).reshape(-1, 2)
    offsets = [0]
    pairs: list[tuple[int, int]] = []
    lengths: list[float] = []
    for edge in map(tuple, edges):
        candidates = edge_samples[edge]
        if len(candidates) > maximum_samples_per_edge:
            selection = np.linspace(0, len(candidates) - 1, maximum_samples_per_edge, dtype=np.int32)
            candidates = [candidates[int(index)] for index in selection]
        for left, right, distance in candidates:
            pairs.append((left, right)); lengths.append(distance)
        offsets.append(len(pairs))
    return FragmentSupportGraph(
        edges,
        np.asarray(offsets, dtype=np.int32),
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        np.asarray(lengths, dtype=np.float32),
        anchored,
    )


def evaluate_fragment_support(
    graph: FragmentSupportGraph,
    position: np.ndarray,
    damage: np.ndarray,
    damage_threshold: float = 0.95,
    maximum_stretch: float = 1.60,
    minimum_intact_sample_fraction: float = 0.25,
    fragment_rest_center: np.ndarray | None = None,
    fragment_role: np.ndarray | None = None,
    maximum_downward_step: float = 0.75,
    same_level_tolerance: float = 0.75,
    maximum_lateral_transfer: float = 0.0,
    minimum_load_capacity_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fragments with a physically directed path to the foundation.

    Without rest centers this retains the legacy undirected reachability used
    by small unit fixtures.  Production passes fragment centers and roles: a
    load path may then propagate upward or approximately level, but cannot
    travel up a facade, back down through another fragment and thereby suspend
    an upper storey above a destroyed base.  Same-level transfer requires a
    diaphragm or frame member, so wall/glass chains cannot support each other
    cyclically in mid-air.
    """
    edge_count = len(graph.edge_fragments)
    if edge_count == 0:
        return graph.anchored_fragments.copy(), np.empty(0, dtype=bool)
    pair = graph.sample_pairs
    current_length = np.linalg.norm(position[pair[:, 1]] - position[pair[:, 0]], axis=1)
    intact_sample = (
        (damage[pair[:, 0]] < damage_threshold)
        & (damage[pair[:, 1]] < damage_threshold)
        & (current_length < graph.sample_rest_length * maximum_stretch)
    )
    sample_edge = np.repeat(np.arange(edge_count, dtype=np.int32), np.diff(graph.sample_offsets))
    intact_count = np.bincount(sample_edge, weights=intact_sample.astype(np.int32), minlength=edge_count)
    sample_count = np.diff(graph.sample_offsets)
    required = np.maximum(1, np.ceil(sample_count * minimum_intact_sample_fraction)).astype(np.int32)
    edge_intact = intact_count >= required

    supported = graph.anchored_fragments.copy()
    live = graph.edge_fragments[edge_intact]
    if fragment_rest_center is None or fragment_role is None:
        for _ in range(len(supported)):
            before = int(np.count_nonzero(supported))
            if len(live):
                from_left = supported[live[:, 0]]
                from_right = supported[live[:, 1]]
                supported[live[from_left, 1]] = True
                supported[live[from_right, 0]] = True
            if int(np.count_nonzero(supported)) == before:
                break
        return supported, edge_intact

    center = np.asarray(fragment_rest_center, dtype=np.float32)
    role = np.asarray(fragment_role, dtype=np.int32)
    all_left, all_right = graph.edge_fragments[:, 0], graph.edge_fragments[:, 1]
    all_dy = center[all_right, 1] - center[all_left, 1]
    frame_roles = np.asarray((1, 3, 4, 5), dtype=np.int32)  # slab, beam, column, core
    all_left_frame = np.isin(role[all_left], frame_roles)
    all_right_frame = np.isin(role[all_right], frame_roles)
    all_level = np.abs(all_dy) <= max(float(same_level_tolerance), 0.0)
    all_frame_transfer = all_left_frame | all_right_frame
    all_allow_left_to_right = (
        (all_dy >= -float(maximum_downward_step))
        & ((~all_level) | all_frame_transfer)
        & (role[all_left] != 6)  # glass never carries a storey above it
    )
    all_allow_right_to_left = (
        (-all_dy >= -float(maximum_downward_step))
        & ((~all_level) | all_frame_transfer)
        & (role[all_right] != 6)
    )
    left, right = live[:, 0], live[:, 1]
    allow_left_to_right = all_allow_left_to_right[edge_intact]
    allow_right_to_left = all_allow_right_to_left[edge_intact]
    level = all_level[edge_intact]

    minimum_capacity = max(float(minimum_load_capacity_fraction), 0.0)
    local_capacity = np.ones(len(supported), dtype=np.float32)
    live_edge_capacity = np.ones(len(live), dtype=np.float32)
    if minimum_capacity > 0.0:
        # Boolean reachability lets one surviving joint carry an arbitrarily
        # large upper section. Approximate load capacity from the fraction of
        # each fragment's originally available incoming boundary samples.
        # This preserves redundancy but makes loss of most columns/diaphragms
        # trigger a real progressive collapse.
        baseline_incoming = np.zeros(len(supported), dtype=np.float32)
        intact_incoming = np.zeros(len(supported), dtype=np.float32)
        np.add.at(
            baseline_incoming,
            all_right[all_allow_left_to_right],
            sample_count[all_allow_left_to_right],
        )
        np.add.at(
            intact_incoming,
            all_right[all_allow_left_to_right],
            intact_count[all_allow_left_to_right],
        )
        np.add.at(
            baseline_incoming,
            all_left[all_allow_right_to_left],
            sample_count[all_allow_right_to_left],
        )
        np.add.at(
            intact_incoming,
            all_left[all_allow_right_to_left],
            intact_count[all_allow_right_to_left],
        )
        local_capacity = np.divide(
            intact_incoming,
            np.maximum(baseline_incoming, 1.0),
            out=np.zeros_like(intact_incoming),
        )
        local_capacity[graph.anchored_fragments] = 1.0
        live_edge_capacity = np.divide(
            intact_count[edge_intact],
            np.maximum(sample_count[edge_intact], 1),
        ).astype(np.float32)
    if maximum_lateral_transfer > 0.0:
        # Track the shortest accumulated horizontal transfer from any local
        # foundation anchor. Vertical columns cost almost nothing, while a
        # chain of diaphragms crossing the entire building exhausts the load-
        # redistribution budget instead of suspending a remote upper tower.
        cost = np.full(len(supported), np.inf, dtype=np.float32)
        cost[graph.anchored_fragments] = 0.0
        horizontal_cost = np.linalg.norm(
            center[right][:, (0, 2)] - center[left][:, (0, 2)], axis=1
        ).astype(np.float32)
        # Small plan offsets between a column and the member above it must not
        # accumulate over every storey. Only diaphragm/same-level travel is a
        # lateral redistribution; vertical propagation carries the current
        # support column upward at zero additional budget.
        horizontal_cost[~level] = 0.0
        limit = float(maximum_lateral_transfer)
        for _ in range(len(supported)):
            previous = cost.copy()
            left_candidate = cost[left] + horizontal_cost
            right_candidate = cost[right] + horizontal_cost
            valid_left = allow_left_to_right & (left_candidate <= limit)
            valid_right = allow_right_to_left & (right_candidate <= limit)
            np.minimum.at(cost, right[valid_left], left_candidate[valid_left])
            np.minimum.at(cost, left[valid_right], right_candidate[valid_right])
            if np.array_equal(cost, previous):
                break
        if minimum_capacity <= 0.0:
            return cost <= limit, edge_intact
        capacity = np.zeros(len(supported), dtype=np.float32)
        capacity[graph.anchored_fragments] = 1.0
        for _ in range(len(supported)):
            previous_capacity = capacity.copy()
            left_candidate = np.minimum(
                np.minimum(capacity[left], live_edge_capacity), local_capacity[right]
            )
            right_candidate = np.minimum(
                np.minimum(capacity[right], live_edge_capacity), local_capacity[left]
            )
            valid_left = allow_left_to_right & (cost[left] + horizontal_cost <= limit)
            valid_right = allow_right_to_left & (cost[right] + horizontal_cost <= limit)
            np.maximum.at(capacity, right[valid_left], left_candidate[valid_left])
            np.maximum.at(capacity, left[valid_right], right_candidate[valid_right])
            if np.array_equal(capacity, previous_capacity):
                break
        return (cost <= limit) & (capacity >= minimum_capacity), edge_intact

    if minimum_capacity > 0.0:
        capacity = np.zeros(len(supported), dtype=np.float32)
        capacity[graph.anchored_fragments] = 1.0
        for _ in range(len(supported)):
            previous_capacity = capacity.copy()
            left_candidate = np.minimum(
                np.minimum(capacity[left], live_edge_capacity), local_capacity[right]
            )
            right_candidate = np.minimum(
                np.minimum(capacity[right], live_edge_capacity), local_capacity[left]
            )
            np.maximum.at(
                capacity, right[allow_left_to_right], left_candidate[allow_left_to_right]
            )
            np.maximum.at(
                capacity, left[allow_right_to_left], right_candidate[allow_right_to_left]
            )
            if np.array_equal(capacity, previous_capacity):
                break
        return capacity >= minimum_capacity, edge_intact

    for _ in range(len(supported)):
        before = int(np.count_nonzero(supported))
        if len(live):
            from_left = supported[left] & allow_left_to_right
            from_right = supported[right] & allow_right_to_left
            supported[right[from_left]] = True
            supported[left[from_right]] = True
        if int(np.count_nonzero(supported)) == before:
            break
    return supported, edge_intact


def evaluate_fragment_fracture_energy(
    graph: FragmentSupportGraph,
    position: np.ndarray,
    damage: np.ndarray,
    material: np.ndarray,
    structural_class: np.ndarray,
    previous_edge_energy: np.ndarray | None = None,
    hairline_energy_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Return irreversible normalized crack energy for edges and fragments.

    The particle solver already applies the physical joint law.  This reduced
    graph samples that same boundary and converts tensile spring energy plus
    accumulated material damage into a stable 0..1 visualization state.  A
    value of one means the representative boundary samples have reached their
    role/material failure envelope; unloading cannot visually heal the crack.
    """
    edge_count = len(graph.edge_fragments)
    fragment_count = len(graph.anchored_fragments)
    if edge_count == 0:
        return np.empty(0, dtype=np.float32), np.zeros(fragment_count, dtype=np.float32)

    pair = graph.sample_pairs
    left, right = pair[:, 0], pair[:, 1]
    current_length = np.linalg.norm(position[right] - position[left], axis=1)
    tensile_strain = np.maximum(
        current_length / np.maximum(graph.sample_rest_length, 1.0e-6) - 1.0,
        0.0,
    )

    # Match the failure envelope used by compute_clustered_solid_forces.
    material_failure_table = np.asarray((0.032, 0.032, 0.012, 0.11), dtype=np.float32)
    left_material = np.clip(material[left], 0, len(material_failure_table) - 1)
    right_material = np.clip(material[right], 0, len(material_failure_table) - 1)
    material_limit = np.minimum(
        material_failure_table[left_material], material_failure_table[right_material]
    )
    role_multiplier_table = np.asarray((1.0, 1.25, 1.0, 1.60, 1.90, 2.20, 0.65), dtype=np.float32)
    left_role = np.clip(structural_class[left], 0, len(role_multiplier_table) - 1)
    right_role = np.clip(structural_class[right], 0, len(role_multiplier_table) - 1)
    failure_strain = material_limit * np.minimum(
        role_multiplier_table[left_role], role_multiplier_table[right_role]
    )

    # Elastic energy is quadratic in strain.  Start exposing a hairline at
    # 35% of the failure energy, before the joint actually separates.
    energy_ratio = np.square(tensile_strain / np.maximum(failure_strain, 1.0e-5))
    onset = float(np.clip(hairline_energy_fraction, 0.0, 0.95))
    elastic_crack = np.clip((energy_ratio - onset) / (1.0 - onset), 0.0, 1.0)
    material_crack = np.maximum(damage[left], damage[right])
    sample_energy = np.maximum(elastic_crack, material_crack)

    # A crack front is local. Blend the strongest representative sample with
    # the boundary mean so one real opening is visible without letting a lone
    # noisy sample immediately paint the whole interface. reduceat keeps this
    # O(samples) for the 20k+ edge production graph.
    starts = graph.sample_offsets[:-1]
    sample_count = np.maximum(np.diff(graph.sample_offsets), 1)
    edge_peak = np.maximum.reduceat(sample_energy, starts)
    edge_mean = np.add.reduceat(sample_energy, starts) / sample_count
    edge_energy = np.asarray(0.72 * edge_peak + 0.28 * edge_mean, dtype=np.float32)
    if previous_edge_energy is not None and len(previous_edge_energy) == edge_count:
        edge_energy = np.maximum(edge_energy, previous_edge_energy).astype(np.float32, copy=False)

    fragment_energy = np.zeros(fragment_count, dtype=np.float32)
    np.maximum.at(fragment_energy, graph.edge_fragments[:, 0], edge_energy)
    np.maximum.at(fragment_energy, graph.edge_fragments[:, 1], edge_energy)
    return edge_energy, fragment_energy


def build_facade_skin(cfg: dict) -> dict[str, np.ndarray]:
    """Create continuous facade panels at architectural, not particle, scale."""
    centers: list[tuple[float, float, float]] = []
    sizes: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    materials: list[int] = []
    building_ids: list[int] = []
    vertices: list[tuple[tuple[float, float, float], ...]] = []
    floor_height = 3.0
    bay_width = 3.0

    def add_panel(bid, center, size, normal, material):
        centers.append(center); sizes.append(size); normals.append(normal)
        materials.append(material); building_ids.append(bid)
        cx, cy, cz = center
        if abs(normal[0]) > 0.5:
            hy = size[1] * 0.5; hz = size[2] * 0.5
            vertices.append(((cx, cy - hy, cz - hz), (cx, cy + hy, cz - hz),
                             (cx, cy + hy, cz + hz), (cx, cy - hy, cz + hz)))
        elif abs(normal[1]) > 0.5:
            hx = size[0] * 0.5; hz = size[2] * 0.5
            vertices.append(((cx - hx, cy, cz - hz), (cx - hx, cy, cz + hz),
                             (cx + hx, cy, cz + hz), (cx + hx, cy, cz - hz)))
        else:
            hx = size[0] * 0.5; hy = size[1] * 0.5
            vertices.append(((cx - hx, cy - hy, cz), (cx - hx, cy + hy, cz),
                             (cx + hx, cy + hy, cz), (cx + hx, cy - hy, cz)))

    def add_wall_bay(bid, center, size, normal, wall_material, glass_material):
        # Continuous concrete backing prevents checkerboard holes. A smaller
        # glass quad sits slightly outward, leaving a visible concrete frame.
        add_panel(bid, center, size, normal, wall_material)
        if size[1] < 1.2:
            return
        outward = 0.09
        window_center = (
            center[0] + normal[0] * outward,
            center[1] + size[1] * 0.04,
            center[2] + normal[2] * outward,
        )
        if abs(normal[0]) > 0.5:
            window_size = (0.12, size[1] * 0.54, size[2] * 0.72)
        else:
            window_size = (size[0] * 0.72, size[1] * 0.54, 0.12)
        add_panel(bid, window_center, window_size, normal, glass_material)

    styles = cfg.get("building_styles", [])
    palettes = cfg.get("building_palettes", [])
    for bid, spec in enumerate(cfg["buildings"]):
        cx, cz, width, depth, height = map(float, spec)
        style = styles[bid] if bid < len(styles) else None
        palette = int(palettes[bid]) % 6 if bid < len(palettes) else bid % 6
        wall_material = 10 + palette
        glass_material = 20 + palette
        roof_material = 30 + palette
        floors = max(1, int(np.ceil(height / floor_height)))
        for floor in range(floors):
            y0 = floor * floor_height
            panel_h = min(floor_height, height - y0)
            cy = y0 + panel_h * 0.5
            offset_x, offset_z, local_width, local_depth = building_profile(style, width, depth, height, cy)
            profile_cx = cx + offset_x
            profile_cz = cz + offset_z
            x_bays = max(1, int(np.ceil(local_width / bay_width)))
            z_bays = max(1, int(np.ceil(local_depth / bay_width)))
            for bay in range(z_bays):
                z0 = -local_depth * 0.5 + bay * local_depth / z_bays
                z1 = -local_depth * 0.5 + (bay + 1) * local_depth / z_bays
                add_wall_bay(bid, (profile_cx - local_width * 0.5, cy, profile_cz + (z0 + z1) * 0.5),
                             (0.16, panel_h, z1 - z0), (-1.0, 0.0, 0.0), wall_material, glass_material)
                add_wall_bay(bid, (profile_cx + local_width * 0.5, cy, profile_cz + (z0 + z1) * 0.5),
                             (0.16, panel_h, z1 - z0), (1.0, 0.0, 0.0), wall_material, glass_material)
            for bay in range(x_bays):
                x0 = -local_width * 0.5 + bay * local_width / x_bays
                x1 = -local_width * 0.5 + (bay + 1) * local_width / x_bays
                add_wall_bay(bid, (profile_cx + (x0 + x1) * 0.5, cy, profile_cz - local_depth * 0.5),
                             (x1 - x0, panel_h, 0.16), (0.0, 0.0, -1.0), wall_material, glass_material)
                add_wall_bay(bid, (profile_cx + (x0 + x1) * 0.5, cy, profile_cz + local_depth * 0.5),
                             (x1 - x0, panel_h, 0.16), (0.0, 0.0, 1.0), wall_material, glass_material)

        # Render every physical floor plate, including the roof. Tiling keeps
        # panels aligned with architectural fragments, so exposed interiors
        # remain visible after facade sections detach.
        for floor_y in np.arange(0.0, height + 0.1, floor_height):
            offset_x, offset_z, local_width, local_depth = building_profile(style, width, depth, height, floor_y)
            profile_cx = cx + offset_x
            profile_cz = cz + offset_z
            x_bays = max(1, int(np.ceil(local_width / bay_width)))
            z_bays = max(1, int(np.ceil(local_depth / bay_width)))
            for x_bay in range(x_bays):
                x0 = -local_width * 0.5 + x_bay * local_width / x_bays
                x1 = -local_width * 0.5 + (x_bay + 1) * local_width / x_bays
                for z_bay in range(z_bays):
                    z0 = -local_depth * 0.5 + z_bay * local_depth / z_bays
                    z1 = -local_depth * 0.5 + (z_bay + 1) * local_depth / z_bays
                    add_panel(
                        bid,
                        (profile_cx + (x0 + x1) * 0.5, floor_y + 0.04, profile_cz + (z0 + z1) * 0.5),
                        (x1 - x0, 0.16, z1 - z0),
                        (0.0, 1.0, 0.0),
                        roof_material,
                    )

    return {
        "center": np.asarray(centers, dtype=np.float32),
        "size": np.asarray(sizes, dtype=np.float32),
        "normal": np.asarray(normals, dtype=np.float32),
        "material": np.asarray(materials, dtype=np.int32),
        "building_id": np.asarray(building_ids, dtype=np.int32),
        "vertex": np.asarray(vertices, dtype=np.float32),
        "panel_mode": np.zeros(len(materials), dtype=np.int32),
        "owner_fragment": np.full(len(materials), -1, dtype=np.int32),
    }


def build_convex_fragment_triangles(
    position: np.ndarray,
    radius: np.ndarray,
    plane_tolerance: float = 0.025,
) -> np.ndarray:
    """Return a small deterministic convex boundary around one particle group."""
    position = np.asarray(position, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    directions = np.asarray([
        (x, y, z)
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
        for z in (-1.0, 0.0, 1.0)
        if x != 0.0 or y != 0.0 or z != 0.0
    ], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    support: list[np.ndarray] = []
    for direction in directions:
        cube_extension = radius * np.sum(np.abs(direction))
        particle = int(np.argmax(position @ direction + cube_extension))
        point = position[particle] + np.sign(direction) * radius[particle]
        if not any(np.linalg.norm(point - existing) <= 1.0e-5 for existing in support):
            support.append(point)
    points = np.asarray(support, dtype=np.float64)
    if len(points) < 4:
        return np.empty((0, 3, 3), dtype=np.float32)

    if ConvexHull is None:
        return np.empty((0, 3, 3), dtype=np.float32)
    try:
        hull = ConvexHull(points)
    except QhullError:
        return np.empty((0, 3, 3), dtype=np.float32)
    triangles = points[np.asarray(hull.simplices, dtype=np.int32)].copy()
    for index, outward_plane in enumerate(hull.equations):
        triangle_normal = np.cross(
            triangles[index, 1] - triangles[index, 0],
            triangles[index, 2] - triangles[index, 0],
        )
        if float(np.dot(triangle_normal, outward_plane[:3])) < 0.0:
            triangles[index] = triangles[index, (0, 2, 1)]
    return np.asarray(triangles, dtype=np.float32)


def build_fragment_cell_faces(
    position: np.ndarray,
    radius: np.ndarray,
    structural_class: np.ndarray,
    palette: int,
    nominal_spacing: float,
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float], int]]:
    """Return a watertight, hole-preserving union of local particle cells.

    A global convex hull fills courtyards, apartment voids and L-shaped frame
    sections with giant triangles.  Here each occupied structural cell exposes
    only faces that have no neighbour in the same rigid fragment. Coplanar
    adjacent faces are greedily merged, so the mesh remains compact without
    bridging an empty cell.
    """
    points = np.asarray(position, dtype=np.float64)
    radii = np.asarray(radius, dtype=np.float64)
    roles = np.asarray(structural_class, dtype=np.int32)
    if len(points) == 0:
        return []
    inferred_spacing = float(np.median(radii)) / 0.48
    spacing = max(min(float(nominal_spacing), inferred_spacing * 1.05), 1.0e-3)
    origin = np.min(points, axis=0)
    keys = np.rint((points - origin) / spacing).astype(np.int32)

    # Refined checkpoints may contain coincident parent/child quantizations.
    # Keep the largest physical sample as the surface representative.
    occupied: dict[tuple[int, int, int], int] = {}
    for index, key_array in enumerate(keys):
        key = tuple(int(value) for value in key_array)
        previous = occupied.get(key)
        if previous is None or radii[index] > radii[previous]:
            occupied[key] = index

    def role_material(role: int) -> int:
        if role == STRUCT_WALL:
            return 10 + palette
        if role == STRUCT_GLASS:
            return 20 + palette
        if role == STRUCT_SLAB:
            return 30 + palette
        return 40 + palette

    # (axis, sign, plane, material) -> {(u, v): particle index}
    groups: dict[tuple[int, int, int, int], dict[tuple[int, int], int]] = {}
    transverse = ((1, 2), (0, 2), (0, 1))
    for key, particle in occupied.items():
        for axis in range(3):
            for sign in (-1, 1):
                neighbour = list(key)
                neighbour[axis] += sign
                if tuple(neighbour) in occupied:
                    continue
                u_axis, v_axis = transverse[axis]
                # Twice-cell coordinates distinguish the two sides of one
                # occupied voxel while keeping a stable integer plane key.
                plane = key[axis] * 2 + sign
                group_key = (
                    axis, sign, plane, role_material(int(roles[particle]))
                )
                groups.setdefault(group_key, {})[(key[u_axis], key[v_axis])] = particle

    faces: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float], int]] = []
    for (axis, sign, _plane, material), cells in groups.items():
        remaining = set(cells)
        u_axis, v_axis = transverse[axis]
        while remaining:
            u0, v0 = min(remaining, key=lambda value: (value[1], value[0]))
            u1 = u0
            while (u1 + 1, v0) in remaining:
                u1 += 1
            v1 = v0
            while all((u, v1 + 1) in remaining for u in range(u0, u1 + 1)):
                v1 += 1
            rectangle = [
                (u, v) for v in range(v0, v1 + 1) for u in range(u0, u1 + 1)
            ]
            remaining.difference_update(rectangle)
            particle_indices = np.asarray([cells[cell] for cell in rectangle], dtype=np.int32)
            lower = np.min(points[particle_indices] - radii[particle_indices, None], axis=0)
            upper = np.max(points[particle_indices] + radii[particle_indices, None], axis=0)
            face_coordinate = np.mean(
                points[particle_indices, axis] + sign * radii[particle_indices]
            )
            center = 0.5 * (lower + upper)
            center[axis] = face_coordinate
            size = np.maximum(upper - lower, 0.08)
            size[axis] = 0.08
            normal = [0.0, 0.0, 0.0]
            normal[axis] = float(sign)
            faces.append(
                (center.astype(np.float32), size.astype(np.float32), tuple(normal), material)
            )
    return faces


def build_fragment_debris_skin(
    cfg: dict,
    rest_x: np.ndarray,
    kind: np.ndarray,
    building_id: np.ndarray,
    fragment_id: np.ndarray,
    radius: np.ndarray,
    structural_class: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build hidden box hulls that become visible after a fragment detaches."""
    centers: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    normals: list[tuple[float, float, float]] = []
    materials: list[int] = []
    building_ids: list[int] = []
    vertices: list[tuple[tuple[float, float, float], ...]] = []
    owners: list[int] = []
    palettes = cfg.get("building_palettes", [])

    def add_face(fid: int, bid: int, center: np.ndarray, size: np.ndarray,
                 normal: tuple[float, float, float], material: int) -> None:
        cx, cy, cz = map(float, center)
        sx, sy, sz = map(float, size)
        centers.append(np.asarray((cx, cy, cz), dtype=np.float32))
        sizes.append(np.asarray((sx, sy, sz), dtype=np.float32))
        normals.append(normal); materials.append(material); building_ids.append(bid); owners.append(fid)
        if abs(normal[0]) > 0.5:
            vertices.append(((cx, cy - sy * 0.5, cz - sz * 0.5), (cx, cy + sy * 0.5, cz - sz * 0.5),
                             (cx, cy + sy * 0.5, cz + sz * 0.5), (cx, cy - sy * 0.5, cz + sz * 0.5)))
        elif abs(normal[1]) > 0.5:
            vertices.append(((cx - sx * 0.5, cy, cz - sz * 0.5), (cx - sx * 0.5, cy, cz + sz * 0.5),
                             (cx + sx * 0.5, cy, cz + sz * 0.5), (cx + sx * 0.5, cy, cz - sz * 0.5)))
        else:
            vertices.append(((cx - sx * 0.5, cy - sy * 0.5, cz), (cx - sx * 0.5, cy + sy * 0.5, cz),
                             (cx + sx * 0.5, cy + sy * 0.5, cz), (cx + sx * 0.5, cy - sy * 0.5, cz)))

    def add_triangle(fid: int, bid: int, triangle: np.ndarray, material: int) -> None:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        lower = np.min(triangle, axis=0); upper = np.max(triangle, axis=0)
        centers.append(np.mean(triangle, axis=0).astype(np.float32))
        sizes.append(np.maximum(upper - lower, 0.08).astype(np.float32))
        normals.append(tuple(float(value) for value in normal))
        materials.append(material); building_ids.append(bid); owners.append(fid)
        a, b, c = (tuple(float(value) for value in point) for point in triangle)
        vertices.append((a, b, c, c))

    valid_particles = np.flatnonzero((fragment_id >= 0) & (kind != 0))
    order = np.argsort(fragment_id[valid_particles], kind="stable")
    sorted_particles = valid_particles[order]
    sorted_fragments = fragment_id[sorted_particles]
    valid_fragments, starts = np.unique(sorted_fragments, return_index=True)
    grouped_particles = np.split(sorted_particles, starts[1:])
    for fid_value, indices in zip(valid_fragments, grouped_particles):
        fid = int(fid_value)
        bid = int(building_id[indices[0]])
        palette = int(palettes[bid]) % 6 if 0 <= bid < len(palettes) else max(bid, 0) % 6
        faces = build_fragment_cell_faces(
            rest_x[indices], radius[indices], structural_class[indices], palette,
            float(cfg.get("solid_spacing", 1.3)),
        )
        if faces:
            for face_center, face_size, face_normal, face_material in faces:
                add_face(fid, bid, face_center, face_size, face_normal, face_material)
        else:
            roles, role_counts = np.unique(structural_class[indices], return_counts=True)
            role = int(roles[int(np.argmax(role_counts))])
            material = 40 + palette
            if role == STRUCT_WALL:
                material = 10 + palette
            elif role == STRUCT_GLASS:
                material = 20 + palette
            elif role == STRUCT_SLAB:
                material = 30 + palette
            padding = max(0.18, float(np.median(radius[indices])) * 0.72)
            lower = np.min(rest_x[indices], axis=0).astype(np.float64) - padding
            upper = np.max(rest_x[indices], axis=0).astype(np.float64) + padding
            center = ((lower + upper) * 0.5).astype(np.float32)
            size = np.maximum(upper - lower, 2.0 * padding).astype(np.float32)
            for axis, sign in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)):
                face_center = center.copy()
                face_center[axis] += sign * size[axis] * 0.5
                normal = [0.0, 0.0, 0.0]; normal[axis] = sign
                add_face(fid, bid, face_center, size, tuple(normal), material)
    return {
        "center": np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        "size": np.asarray(sizes, dtype=np.float32).reshape(-1, 3),
        "normal": np.asarray(normals, dtype=np.float32).reshape(-1, 3),
        "material": np.asarray(materials, dtype=np.int32),
        "building_id": np.asarray(building_ids, dtype=np.int32),
        "vertex": np.asarray(vertices, dtype=np.float32).reshape(-1, 4, 3),
        "panel_mode": np.ones(len(materials), dtype=np.int32),
        "owner_fragment": np.asarray(owners, dtype=np.int32),
    }


def bind_facade_anchors(
    skin: dict[str, np.ndarray],
    rest_x: np.ndarray,
    kind: np.ndarray,
    building_id: np.ndarray,
    spacing: float,
    fragment_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bind each panel to particles belonging to one cohesive fragment."""
    anchors = np.full((len(skin["vertex"]), 4), -1, dtype=np.int32)
    preferred_owner = np.asarray(
        skin.get("owner_fragment", np.full(len(skin["vertex"]), -1, dtype=np.int32)),
        dtype=np.int32,
    )
    owner_fragment = np.full(len(skin["vertex"]), -1, dtype=np.int32)
    if fragment_id is not None and cKDTree is not None:
        owner_fragment[:] = preferred_owner
        # Assign ordinary facade/floor panels to the fragment nearest their
        # center using one compiled KD-tree query per building.
        for bid in np.unique(skin["building_id"]):
            panel_indices = np.flatnonzero(
                (skin["building_id"] == bid) & (owner_fragment < 0)
            )
            particle_indices = np.flatnonzero((kind != 0) & (building_id == bid))
            if len(panel_indices) == 0 or len(particle_indices) == 0:
                continue
            nearest_local = cKDTree(rest_x[particle_indices]).query(
                skin["center"][panel_indices], workers=-1
            )[1]
            owner_fragment[panel_indices] = fragment_id[particle_indices[nearest_local]]

        valid_particles = np.flatnonzero((kind != 0) & (fragment_id >= 0))
        order = np.argsort(fragment_id[valid_particles], kind="stable")
        sorted_particles = valid_particles[order]
        sorted_fragments = fragment_id[sorted_particles]
        unique_fragments, starts = np.unique(sorted_fragments, return_index=True)
        fragment_particles = {
            int(owner): indices
            for owner, indices in zip(unique_fragments, np.split(sorted_particles, starts[1:]))
        }
        # Bind every panel corner to a physical sample of the same cohesive
        # fragment. Qhull triangles and architectural quads share this path.
        for owner in np.unique(owner_fragment[owner_fragment >= 0]):
            panel_indices = np.flatnonzero(owner_fragment == owner)
            particle_indices = fragment_particles.get(int(owner), np.empty(0, dtype=np.int32))
            if len(panel_indices) == 0 or len(particle_indices) == 0:
                continue
            panel_vertices = skin["vertex"][panel_indices].reshape(-1, 3)
            nearest_local = cKDTree(rest_x[particle_indices]).query(
                panel_vertices
            )[1]
            anchors[panel_indices] = particle_indices[nearest_local].reshape(-1, 4)
        # Environment panels use negative object IDs and intentionally have no
        # structural fragment. Bind their vertices directly to the matching
        # low-cost prop particles so cars and trees translate with physics.
        for bid in np.unique(skin["building_id"][skin["building_id"] < 0]):
            panel_indices = np.flatnonzero(skin["building_id"] == bid)
            particle_indices = np.flatnonzero((kind != 0) & (building_id == bid))
            if len(panel_indices) == 0 or len(particle_indices) == 0:
                continue
            panel_vertices = skin["vertex"][panel_indices].reshape(-1, 3)
            nearest_local = cKDTree(rest_x[particle_indices]).query(panel_vertices)[1]
            anchors[panel_indices] = particle_indices[nearest_local].reshape(-1, 4)
        if np.any(anchors < 0):
            raise RuntimeError("Some facade vertices could not be bound to structural particles")
        return anchors, owner_fragment

    # Debris hulls already know their cohesive owner. Bind all of one hull's
    # vertices in vectorized chunks instead of repeating a bucket query for
    # every triangle corner. This is especially important when a resumed,
    # refined fragment contains hundreds of physical samples.
    if fragment_id is not None:
        valid_particles = np.flatnonzero((kind != 0) & (fragment_id >= 0))
        order = np.argsort(fragment_id[valid_particles], kind="stable")
        sorted_particles = valid_particles[order]
        sorted_fragments = fragment_id[sorted_particles]
        unique_fragments, starts = np.unique(sorted_fragments, return_index=True)
        fragment_particles = {
            int(owner): indices
            for owner, indices in zip(unique_fragments, np.split(sorted_particles, starts[1:]))
        }
        for owner in np.unique(preferred_owner[preferred_owner >= 0]):
            panel_indices = np.flatnonzero(preferred_owner == owner)
            particle_indices = fragment_particles.get(int(owner), np.empty(0, dtype=np.int32))
            if len(panel_indices) == 0 or len(particle_indices) == 0:
                continue
            points = rest_x[particle_indices]
            panel_vertices = skin["vertex"][panel_indices].reshape(-1, 3)
            nearest_global = np.empty(len(panel_vertices), dtype=np.int32)
            for start in range(0, len(panel_vertices), 512):
                stop = min(start + 512, len(panel_vertices))
                delta = panel_vertices[start:stop, None, :] - points[None, :, :]
                distance2 = np.sum(delta * delta, axis=2)
                nearest_global[start:stop] = particle_indices[np.argmin(distance2, axis=1)]
            anchors[panel_indices] = nearest_global.reshape(-1, 4)
            owner_fragment[panel_indices] = int(owner)
    cache: dict[tuple[int, ...], int] = {}
    for bid in np.unique(skin["building_id"]):
        particle_indices = np.flatnonzero((kind != 0) & (building_id == bid))
        if len(particle_indices) == 0:
            continue
        points = rest_x[particle_indices]
        particle_fragments = fragment_id[particle_indices] if fragment_id is not None else np.full(len(points), -1, dtype=np.int32)
        cell_keys = np.rint(points / spacing).astype(np.int32)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for local_index, key in enumerate(cell_keys):
            buckets.setdefault(tuple(map(int, key)), []).append(local_index)

        def nearest_particle(point: np.ndarray, required_fragment: int = -1) -> int:
            quantized = tuple(map(int, np.rint(point / spacing)))
            candidates: list[int] = []
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        candidates.extend(buckets.get(
                            (quantized[0] + dx, quantized[1] + dy, quantized[2] + dz), ()
                        ))
            if candidates:
                local = np.asarray(candidates, dtype=np.int32)
                if required_fragment >= 0:
                    local = local[particle_fragments[local] == required_fragment]
            else:
                local = np.empty(0, dtype=np.int32)
            if len(local) == 0:
                local = np.flatnonzero(particle_fragments == required_fragment) if required_fragment >= 0 else np.arange(len(points))
            return int(local[int(np.argmin(np.sum((points[local] - point) ** 2, axis=1)))])

        panel_indices = np.flatnonzero((skin["building_id"] == bid) & (preferred_owner < 0))
        for panel_index in panel_indices:
            owner = int(preferred_owner[panel_index])
            if owner < 0:
                center_local = nearest_particle(skin["center"][panel_index])
                owner = int(particle_fragments[center_local])
            owner_fragment[panel_index] = owner
            for corner in range(4):
                vertex = skin["vertex"][panel_index, corner]
                quantized = tuple(map(int, np.rint(vertex / spacing)))
                cache_key = (int(bid), owner, *quantized)
                if cache_key not in cache:
                    nearest = nearest_particle(vertex, owner)
                    cache[cache_key] = int(particle_indices[nearest])
                anchors[panel_index, corner] = cache[cache_key]
    if np.any(anchors < 0):
        raise RuntimeError("Some facade vertices could not be bound to structural particles")
    return anchors, owner_fragment


def build_environment_skin(cfg: dict) -> dict[str, np.ndarray]:
    """Create panel geometry for roads, moving cars, trees and small shops."""
    layout = environment_layout(cfg)
    centers: list[tuple[float, float, float]] = []
    sizes: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    materials: list[int] = []
    building_ids: list[int] = []
    vertices: list[tuple[tuple[float, float, float], ...]] = []

    def add_panel(bid: int, center, size, normal, material: int):
        centers.append(tuple(map(float, center)))
        sizes.append(tuple(map(float, size)))
        normals.append(tuple(map(float, normal)))
        materials.append(int(material)); building_ids.append(int(bid))
        cx, cy, cz = map(float, center)
        sx, sy, sz = map(float, size)
        nx, ny, nz = map(float, normal)
        if abs(nx) > 0.5:
            vertices.append(((cx, cy - sy * 0.5, cz - sz * 0.5),
                             (cx, cy + sy * 0.5, cz - sz * 0.5),
                             (cx, cy + sy * 0.5, cz + sz * 0.5),
                             (cx, cy - sy * 0.5, cz + sz * 0.5)))
        elif abs(ny) > 0.5:
            vertices.append(((cx - sx * 0.5, cy, cz - sz * 0.5),
                             (cx - sx * 0.5, cy, cz + sz * 0.5),
                             (cx + sx * 0.5, cy, cz + sz * 0.5),
                             (cx + sx * 0.5, cy, cz - sz * 0.5)))
        else:
            vertices.append(((cx - sx * 0.5, cy - sy * 0.5, cz),
                             (cx - sx * 0.5, cy + sy * 0.5, cz),
                             (cx + sx * 0.5, cy + sy * 0.5, cz),
                             (cx + sx * 0.5, cy - sy * 0.5, cz)))

    def add_box(bid: int, center, size, material: int):
        cx, cy, cz = map(float, center); sx, sy, sz = map(float, size)
        add_panel(bid, (cx - sx * 0.5, cy, cz), (0.10, sy, sz), (-1, 0, 0), material)
        add_panel(bid, (cx + sx * 0.5, cy, cz), (0.10, sy, sz), (1, 0, 0), material)
        add_panel(bid, (cx, cy, cz - sz * 0.5), (sx, sy, 0.10), (0, 0, -1), material)
        add_panel(bid, (cx, cy, cz + sz * 0.5), (sx, sy, 0.10), (0, 0, 1), material)
        add_panel(bid, (cx, cy + sy * 0.5, cz), (sx, 0.10, sz), (0, 1, 0), material)

    if any(layout.values()):
        # Terrain and roads are render-only panels bound to four fixed anchors.
        add_panel(-9000, (0.0, -0.04, 55.0), (140.0, 0.08, 150.0), (0, 1, 0), 90)
        for z in (34.0, 72.0):
            add_panel(-9000, (0.0, 0.015, z), (140.0, 0.08, 13.0), (0, 1, 0), 91)
            for edge in (-7.4, 7.4):
                add_panel(-9000, (0.0, 0.035, z + edge), (140.0, 0.08, 1.4), (0, 1, 0), 92)

    for car in layout["cars"]:
        cx, cz = car["center"]; bid = int(car["id"]); palette = int(car["palette"])
        add_box(bid, (cx, 0.72, cz), (4.3, 0.85, 1.85), 50 + palette)
        add_box(bid, (cx + 0.15, 1.32, cz), (2.25, 0.58, 1.58), 20 + palette)
        for dx in (-1.45, 1.45):
            for dz in (-0.92, 0.92):
                add_box(bid, (cx + dx, 0.43, cz + dz), (0.62, 0.62, 0.22), 58)

    for tree in layout["trees"]:
        cx, cz = tree["center"]; height = float(tree["height"]); bid = int(tree["id"])
        add_box(bid, (cx, height * 0.43, cz), (0.62, height * 0.86, 0.62), 60)
        add_box(bid, (cx, height - 0.15, cz), (3.1, 2.4, 2.5), 70 + (abs(bid) % 3))

    for shop in layout["small_buildings"]:
        cx, cz = shop["center"]; width, depth, height = map(float, shop["size"])
        bid = int(shop["id"]); palette = int(shop["palette"])
        add_box(bid, (cx, height * 0.5, cz), (width, height, depth), 10 + palette)
        add_panel(bid, (cx, height + 0.08, cz), (width + 0.4, 0.16, depth + 0.4), (0, 1, 0), 30 + palette)
        add_panel(bid, (cx, 2.0, cz - depth * 0.5 - 0.06),
                  (width * 0.62, 2.2, 0.10), (0, 0, -1), 20 + palette)

    panel_count = len(materials)
    return {
        "center": np.asarray(centers, dtype=np.float32).reshape((-1, 3)),
        "size": np.asarray(sizes, dtype=np.float32).reshape((-1, 3)),
        "normal": np.asarray(normals, dtype=np.float32).reshape((-1, 3)),
        "material": np.asarray(materials, dtype=np.int32),
        "building_id": np.asarray(building_ids, dtype=np.int32),
        "vertex": np.asarray(vertices, dtype=np.float32).reshape((-1, 4, 3)),
        "panel_mode": np.zeros(panel_count, dtype=np.int32),
        "owner_fragment": np.full(panel_count, -1, dtype=np.int32),
    }


def write_facade_skin(
    path: Path,
    cfg: dict,
    rest_x: np.ndarray | None = None,
    kind: np.ndarray | None = None,
    building_id: np.ndarray | None = None,
    fragment_id: np.ndarray | None = None,
    radius: np.ndarray | None = None,
    structural_class: np.ndarray | None = None,
) -> int:
    debris_policy = cfg.get("v3", {}).get("debris_skin", {})
    complete_geometry = all(
        value is not None
        for value in (rest_x, kind, building_id, fragment_id, radius, structural_class)
    )
    cache_path: Path | None = None
    if complete_geometry and bool(debris_policy.get("cache", True)):
        geometry_cfg = {
            "schema": "cell-union-fragment-skin-v6-environment",
            "solid_spacing": cfg.get("solid_spacing"),
            "buildings": cfg.get("buildings"),
            "building_styles": cfg.get("building_styles"),
            "building_palettes": cfg.get("building_palettes"),
            "fragment_clustering": cfg.get("v3", {}).get("fragment_clustering"),
            "debris_skin": debris_policy,
            "environment": cfg.get("environment", {}),
        }
        digest = hashlib.sha256(
            json.dumps(geometry_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for values in (rest_x, kind, building_id, fragment_id, radius, structural_class):
            digest.update(np.ascontiguousarray(values).view(np.uint8))
        cache_dir = path.parent.parent / "_geometry_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"facade_skin_{digest.hexdigest()[:20]}.npz"
        if cache_path.exists():
            shutil.copyfile(cache_path, path)
            with np.load(cache_path, allow_pickle=False) as cached:
                panel_count = len(cached["building_id"])
            print(f"Facade/debris skin cache hit: {cache_path.name}")
            return panel_count

    skin = build_facade_skin(cfg)
    environment_skin = build_environment_skin(cfg)
    environment_particles_present = (
        building_id is None or np.any(np.asarray(building_id) <= -1000)
    )
    if len(environment_skin["building_id"]) and environment_particles_present:
        skin = {
            name: np.concatenate((skin[name], environment_skin[name]), axis=0)
            for name in skin
        }
    if (
        rest_x is not None and kind is not None and building_id is not None
        and fragment_id is not None and radius is not None and structural_class is not None
        and bool(debris_policy.get("enabled", True))
    ):
        debris = build_fragment_debris_skin(
            cfg, rest_x, kind, building_id, fragment_id, radius, structural_class
        )
        skin = {name: np.concatenate((skin[name], debris[name]), axis=0) for name in skin}
    if rest_x is not None and kind is not None and building_id is not None:
        skin["anchor"], skin["owner_fragment"] = bind_facade_anchors(
            skin, rest_x, kind, building_id, float(cfg["solid_spacing"]), fragment_id
        )
    np.savez_compressed(path, **skin)
    if cache_path is not None:
        temporary_cache = cache_path.with_suffix(".tmp.npz")
        shutil.copyfile(path, temporary_cache)
        temporary_cache.replace(cache_path)
        print(f"Facade/debris skin cached: {cache_path.name}")
    return len(skin["building_id"])


def build_fragment_ids(
    rest_x: np.ndarray,
    kind: np.ndarray,
    building_id: np.ndarray,
    cfg: dict,
    structural_class: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition each building into architectural-scale cohesive fragments.

    These IDs do not pre-crack the structure.  Bonds between IDs are ordinary
    breakable joints, while bonds inside an ID remain cohesive after a crack.
    Sparse edge cells are merged into their nearest populated neighbour so a
    single lattice particle can never become a dust fragment.
    """
    clustering = cfg["v3"]["fragment_clustering"]
    cell = np.asarray(clustering.get("cell_size", [3.0, 3.0, 3.0]), dtype=np.float32)
    reference_spacing = float(cfg["v3"]["solid_refinement"].get("impact_spacing", cfg["solid_spacing"]))
    current_spacing = float(cfg["solid_spacing"])
    # Keep the minimum fragment *area* stable when the starting lattice is
    # coarse: a 1.3 m wall has roughly four times fewer surface particles than
    # the same 0.65 m wall.
    area_scale = (reference_spacing / current_spacing) ** 2
    minimum = max(4, int(round(int(clustering.get("minimum_cluster_particles", 24)) * area_scale)))
    merge_sparse_cells = bool(clustering.get("merge_sparse_cells", True))
    maximum_merge_distance = float(
        clustering.get("maximum_sparse_merge_cell_distance", np.inf)
    )
    maximum_span_cells = int(clustering.get("maximum_fragment_span_cells", 2**30))
    fragment_id = np.full(len(rest_x), -1, dtype=np.int32)
    next_fragment = 0

    # Never place facade, floors and the vertical frame in one unbreakable
    # chunk.  The previous spatial-only partition could bind a rear facade
    # panel to a core-dominated fragment, leaving a visually intact wall after
    # all of its real wall connections had failed.
    structural_family = np.zeros(len(rest_x), dtype=np.int32)
    if structural_class is not None:
        structural_family[np.isin(structural_class, (STRUCT_SLAB, STRUCT_BEAM))] = 1
        structural_family[np.isin(structural_class, (STRUCT_COLUMN, STRUCT_CORE))] = 2
        structural_family[np.isin(structural_class, (STRUCT_WALL, STRUCT_GLASS))] = 0

    for bid, spec in enumerate(cfg["buildings"]):
        building_indices = np.flatnonzero((kind != 0) & (building_id == bid))
        if len(building_indices) == 0:
            continue
        cx, cz, width, depth, _height = map(float, spec)
        origin = np.asarray((cx - width * 0.5, 0.0, cz - depth * 0.5), dtype=np.float32)
        for family in np.unique(structural_family[building_indices]):
            indices = building_indices[structural_family[building_indices] == family]
            coordinates = np.floor((rest_x[indices] - origin) / cell).astype(np.int32)
            unique_cells, inverse, counts = np.unique(
                coordinates, axis=0, return_inverse=True, return_counts=True
            )

            remap = np.arange(len(unique_cells), dtype=np.int64)
            if merge_sparse_cells:
                stable = np.flatnonzero(counts >= minimum)
                group_lower = unique_cells.copy()
                group_upper = unique_cells.copy()
                for source in np.flatnonzero(counts < minimum):
                    if len(stable) == 0:
                        continue
                    delta = (unique_cells[stable] - unique_cells[source]).astype(np.float32)
                    distance2 = np.sum(delta * delta, axis=1)
                    score = distance2 - np.minimum(counts[stable], minimum * 4) * 1.0e-4
                    for candidate_index in np.argsort(score):
                        target = int(stable[int(candidate_index)])
                        if distance2[int(candidate_index)] > maximum_merge_distance * maximum_merge_distance:
                            break
                        lower = np.minimum(group_lower[target], unique_cells[source])
                        upper = np.maximum(group_upper[target], unique_cells[source])
                        # A stable cell may absorb an adjacent sparse edge, but
                        # never sparse cells from both far sides.  This hard
                        # physical-span bound prevents a nearest-cell merge
                        # from creating a multi-storey rigid shard.
                        if np.any(upper - lower > maximum_span_cells):
                            continue
                        remap[source] = target
                        group_lower[target] = lower
                        group_upper[target] = upper
                        break

            merged = remap[inverse]
            representatives = np.unique(merged)
            local_ids = np.searchsorted(representatives, merged).astype(np.int32)
            fragment_id[indices] = local_ids + next_fragment
            next_fragment += len(representatives)

    counts = np.bincount(fragment_id[fragment_id >= 0], minlength=next_fragment).astype(np.int32)
    return fragment_id, counts


def build_refinement_axes(
    rest_x: np.ndarray,
    kind: np.ndarray,
    building_id: np.ndarray,
    spacing: float,
    structural_class: np.ndarray | None = None,
) -> np.ndarray:
    """Infer plane normal or longitudinal beam/column axis from connectivity."""
    axes = np.full(len(rest_x), -1, dtype=np.int32)
    neighbours = (
        ((-1, 0, 0), (1, 0, 0)),
        ((0, -1, 0), (0, 1, 0)),
        ((0, 0, -1), (0, 0, 1)),
    )
    for bid in np.unique(building_id[building_id >= 0]):
        indices = np.flatnonzero((kind != 0) & (building_id == bid))
        keys = np.rint(rest_x[indices] / spacing).astype(np.int32)
        occupied = {tuple(map(int, key)) for key in keys}
        for particle_index, key in zip(indices, keys):
            connectivity = []
            for pair in neighbours:
                count = 0
                for offset in pair:
                    neighbour = (int(key[0] + offset[0]), int(key[1] + offset[1]), int(key[2] + offset[2]))
                    count += int(neighbour in occupied)
                connectivity.append(count)
            role = int(structural_class[particle_index]) if structural_class is not None else 0
            if role == STRUCT_BEAM or role == STRUCT_COLUMN:
                axes[particle_index] = int(np.argmax(connectivity))
            else:
                axes[particle_index] = int(np.argmin(connectivity))
    return axes


def select_conservative_fluid_merges(
    group_id: np.ndarray,
    kind: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    mass: np.ndarray,
    volume: np.ndarray,
    radius: np.ndarray,
    *,
    maximum_y: float,
    maximum_vertical_speed: float,
    maximum_velocity_rms: float,
    maximum_span: float,
    maximum_fine_radius: float,
) -> dict[str, np.ndarray]:
    """Select intact calm sibling octets that can safely return to coarse SPH.

    A group is mergeable only while all eight original 1->8 children still
    exist.  The replacement state uses mass-weighted center and velocity, so
    total mass, volume and linear momentum are conserved exactly apart from
    float32 upload rounding.
    """
    group_id = np.asarray(group_id, dtype=np.int32)
    kind = np.asarray(kind, dtype=np.int32)
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    candidate_ids = np.unique(group_id[(group_id >= 0) & (kind == 0)])
    representatives: list[int] = []
    removed: list[int] = []
    merged_position: list[np.ndarray] = []
    merged_velocity: list[np.ndarray] = []
    merged_mass: list[float] = []
    merged_volume: list[float] = []
    merged_radius: list[float] = []
    for sibling_id in candidate_ids:
        indices = np.flatnonzero(group_id == sibling_id)
        if len(indices) != 8 or np.any(kind[indices] != 0):
            continue
        if np.any(radius[indices] > maximum_fine_radius):
            continue
        if np.any(position[indices, 1] > maximum_y):
            continue
        if np.any(np.abs(velocity[indices, 1]) > maximum_vertical_speed):
            continue
        total_mass = float(np.sum(mass[indices], dtype=np.float64))
        total_volume = float(np.sum(volume[indices], dtype=np.float64))
        if total_mass <= 0.0 or total_volume <= 0.0:
            continue
        center = np.sum(position[indices] * mass[indices, None], axis=0) / total_mass
        mean_velocity = np.sum(velocity[indices] * mass[indices, None], axis=0) / total_mass
        velocity_delta = velocity[indices] - mean_velocity
        velocity_rms = float(np.sqrt(
            np.sum(mass[indices] * np.sum(velocity_delta * velocity_delta, axis=1)) / total_mass
        ))
        span = float(np.max(np.linalg.norm(position[indices] - center, axis=1)))
        if velocity_rms > maximum_velocity_rms or span > maximum_span:
            continue
        representative = int(indices[0])
        representatives.append(representative)
        removed.extend(int(index) for index in indices[1:])
        merged_position.append(center)
        merged_velocity.append(mean_velocity)
        merged_mass.append(total_mass)
        merged_volume.append(total_volume)
        merged_radius.append(float(np.max(radius[indices]) * 2.0))
    count = len(representatives)
    return {
        "representatives": np.asarray(representatives, dtype=np.int32),
        "removed": np.asarray(removed, dtype=np.int32),
        "position": np.asarray(merged_position, dtype=np.float32).reshape(count, 3),
        "velocity": np.asarray(merged_velocity, dtype=np.float32).reshape(count, 3),
        "mass": np.asarray(merged_mass, dtype=np.float32),
        "volume": np.asarray(merged_volume, dtype=np.float32),
        "radius": np.asarray(merged_radius, dtype=np.float32),
    }
