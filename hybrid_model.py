"""V3 structural LOD policy and render-skin metadata.

The facade skin is deliberately independent from simulation particles.  The
first V3 renderer will deform these panels from the structural particle graph
instead of drawing every particle as a circle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from scene import (
    STRUCT_BEAM,
    STRUCT_COLUMN,
    STRUCT_CORE,
    STRUCT_GLASS,
    STRUCT_SLAB,
    STRUCT_WALL,
    building_profile,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return foundation-connected fragments and currently intact graph edges."""
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
    owner_fragment = np.full(len(skin["vertex"]), -1, dtype=np.int32)
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

        panel_indices = np.flatnonzero(skin["building_id"] == bid)
        for panel_index in panel_indices:
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


def write_facade_skin(
    path: Path,
    cfg: dict,
    rest_x: np.ndarray | None = None,
    kind: np.ndarray | None = None,
    building_id: np.ndarray | None = None,
    fragment_id: np.ndarray | None = None,
) -> int:
    skin = build_facade_skin(cfg)
    if rest_x is not None and kind is not None and building_id is not None:
        skin["anchor"], skin["owner_fragment"] = bind_facade_anchors(
            skin, rest_x, kind, building_id, float(cfg["solid_spacing"]), fragment_id
        )
    np.savez_compressed(path, **skin)
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

            stable = np.flatnonzero(counts >= minimum)
            if len(stable) == 0:
                stable = np.asarray([int(np.argmax(counts))], dtype=np.int64)
            remap = np.arange(len(unique_cells), dtype=np.int64)
            for source in np.flatnonzero(counts < minimum):
                delta = (unique_cells[stable] - unique_cells[source]).astype(np.float32)
                distance2 = np.sum(delta * delta, axis=1)
                # Prefer a nearby massive cell when two candidates are equidistant.
                score = distance2 - np.minimum(counts[stable], minimum * 4) * 1.0e-4
                remap[source] = stable[int(np.argmin(score))]

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
