"""Procedural particle city and adaptive-resolution tsunami reservoir."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


STRUCT_NONE = 0
STRUCT_SLAB = 1
STRUCT_WALL = 2
STRUCT_BEAM = 3
STRUCT_COLUMN = 4
STRUCT_CORE = 5
STRUCT_GLASS = 6


def environment_layout(cfg: dict) -> dict[str, list[dict]]:
    """Deterministic street furniture shared by physics and render skins."""
    policy = cfg.get("environment", {})
    if not bool(policy.get("enabled", False)):
        return {"cars": [], "trees": [], "small_buildings": []}
    seed = int(policy.get("seed", 7319))
    rng = np.random.default_rng(seed)
    cars = []
    car_count = int(policy.get("cars", {}).get("count", 30))
    car_palette = (0, 1, 2, 3, 4, 5)
    for index in range(car_count):
        road = index % 2
        lane = (index // 2) % 2
        z = (31.0 if road == 0 else 69.0) + lane * 6.0 + rng.uniform(-0.7, 0.7)
        x = -62.0 + 124.0 * ((index // 4 + 0.5) / max(1, int(np.ceil(car_count / 4))))
        x += rng.uniform(-2.0, 2.0)
        cars.append({"id": -1000 - index, "center": (x, z), "palette": car_palette[index % 6]})

    trees = []
    tree_count = int(policy.get("trees", {}).get("count", 24))
    street_x = np.asarray((-68.0, -42.0, -14.0, 14.0, 42.0, 68.0), dtype=np.float32)
    street_z = np.asarray((27.0, 41.0, 65.0, 79.0), dtype=np.float32)
    sites = [(float(x), float(z)) for z in street_z for x in street_x]
    for index, (x, z) in enumerate(sites[:tree_count]):
        trees.append({"id": -2000 - index, "center": (x, z), "height": 6.0 + 1.5 * (index % 3)})

    shops = []
    shop_policy = policy.get("small_buildings", {})
    requested = int(shop_policy.get("count", 6))
    # Keep the low-rise foreground legible instead of hiding it behind the
    # third tower row.  The first high-rise facades begin near z=8.5 m; these
    # centres leave a water corridor of roughly 2--3 m before that row.
    front_z = float(shop_policy.get("front_z", 2.0))
    stagger_z = float(shop_policy.get("stagger_z", 0.8))
    for index in range(requested):
        x = -55.0 + index * (110.0 / max(requested - 1, 1))
        shops.append({
            "id": -3000 - index,
            "center": (x, front_z + (index % 2) * stagger_z),
            "size": (8.0 + (index % 2) * 2.0, 7.0, 5.5 + (index % 3)),
            "palette": index % 6,
        })
    return {"cars": cars, "trees": trees, "small_buildings": shops}


def building_profile(style: dict | None, width: float, depth: float, height: float, y: float):
    """Return local centre offset and footprint for a styled height slice."""
    style = style or {}
    shape = str(style.get("shape", "rect"))
    ratio = float(np.clip(y / max(height, 1.0e-6), 0.0, 1.0))
    scale_x = 1.0
    scale_z = 1.0
    offset_x = 0.0
    offset_z = 0.0
    if shape == "podium":
        if ratio >= float(style.get("podium_ratio", 0.24)):
            tower_scale = style.get("tower_scale", [0.68, 0.72])
            tower_offset = style.get("tower_offset", [0.0, 0.0])
            scale_x, scale_z = map(float, tower_scale)
            offset_x = float(tower_offset[0]) * width
            offset_z = float(tower_offset[1]) * depth
    elif shape == "setback":
        for tier in sorted(style.get("tiers", [[0.55, 0.82, 0.86, 0.0, 0.0],
                                                  [0.78, 0.64, 0.70, 0.0, 0.0]])):
            if ratio >= float(tier[0]):
                scale_x = float(tier[1]); scale_z = float(tier[2])
                offset_x = float(tier[3]) * width; offset_z = float(tier[4]) * depth
    elif shape == "tapered":
        steps = max(1, int(style.get("steps", 6)))
        stepped_ratio = math.floor(ratio * steps + 1.0e-6) / steps
        taper = style.get("taper", [0.25, 0.18])
        scale_x = 1.0 - float(taper[0]) * stepped_ratio
        scale_z = 1.0 - float(taper[1]) * stepped_ratio
        lean = style.get("lean", [0.0, 0.0])
        offset_x = float(lean[0]) * width * stepped_ratio
        offset_z = float(lean[1]) * depth * stepped_ratio
    elif shape == "offset":
        if ratio >= float(style.get("split_ratio", 0.46)):
            upper_scale = style.get("upper_scale", [0.78, 0.82])
            upper_offset = style.get("upper_offset", [0.10, -0.06])
            scale_x, scale_z = map(float, upper_scale)
            offset_x = float(upper_offset[0]) * width
            offset_z = float(upper_offset[1]) * depth
    return offset_x, offset_z, max(width * scale_x, 3.0), max(depth * scale_z, 3.0)


@dataclass
class ParticleScene:
    rest_density: float = 1000.0
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    velocities: list[tuple[float, float, float]] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)
    masses: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    kinds: list[int] = field(default_factory=list)
    materials: list[int] = field(default_factory=list)
    building_ids: list[int] = field(default_factory=list)
    fixed: list[int] = field(default_factory=list)
    rest_positions: list[tuple[float, float, float]] = field(default_factory=list)
    damage: list[float] = field(default_factory=list)
    structural_classes: list[int] = field(default_factory=list)

    def append(self, pos, vel, radius, mass, volume, kind, material=0, building_id=-1, fixed=0,
               structural_class=STRUCT_NONE):
        p = tuple(float(v) for v in pos)
        self.positions.append(p)
        self.velocities.append(tuple(float(v) for v in vel))
        self.radii.append(float(radius))
        self.masses.append(float(mass))
        self.volumes.append(float(volume))
        self.kinds.append(int(kind))
        self.materials.append(int(material))
        self.building_ids.append(int(building_id))
        self.fixed.append(int(fixed))
        self.rest_positions.append(p)
        self.damage.append(0.0)
        self.structural_classes.append(int(structural_class))

    def add_water(self, cfg: dict):
        width = float(cfg["domain_width"])
        z_min = float(cfg["reservoir_z_min"])
        z_max = float(cfg["reservoir_z_max"])
        depth = float(cfg["water_depth"])
        crest = float(cfg["wave_height"])
        speed = float(cfg["wave_speed"])
        background_current = float(cfg.get("background_current", 0.0))
        coarse = float(cfg["coarse_spacing"])
        fine = float(cfg["fine_spacing"])
        surface_band = float(cfg.get("fine_surface_band", 2.0))
        rng = np.random.default_rng(4192)
        shallow_cfg = cfg.get("v3", {}).get("shallow_water", {})
        if bool(shallow_cfg.get("enabled", False)) and bool(shallow_cfg.get("replace_far_sph", False)):
            # V3 keeps expensive particles only in the near-field window. The
            # removed smooth rear reservoir is evolved by its 2D conservative
            # shallow-water field and overlaps this boundary for coupling.
            z_min = max(z_min, float(shallow_cfg.get("sph_z_min", z_min)))

        # Production-stable path: a uniform discretization avoids the density
        # inconsistency of a naive coarse/fine interface. The adaptive solver is
        # retained below as an experimental mode.
        if "uniform_water_spacing" in cfg:
            spacing = float(cfg["uniform_water_spacing"])
            for x in np.arange(-width / 2 + spacing / 2, width / 2, spacing):
                for z in np.arange(z_min + spacing / 2, z_max, spacing):
                    elevation = crest * math.exp(-((z - z_max + 5.0) / 7.5) ** 2)
                    local_depth = depth + elevation
                    # `wave_speed` is the phase speed of the long wave, not
                    # the material velocity of the whole reservoir. Shallow-
                    # water continuity gives u ~= c*eta/(h+eta), localized to
                    # the crest and nearly zero in the trailing water.
                    flow_speed = speed * elevation / max(local_depth, 1.0e-6)
                    for y in np.arange(spacing / 2, local_depth, spacing):
                        jitter = (rng.random(3) - 0.5) * spacing * 0.025
                        velocity = (0.0, 0.0, background_current + flow_speed * (0.82 + 0.18 * y / local_depth))
                        volume = spacing ** 3
                        self.append(np.array((x, y, z)) + jitter, velocity, spacing * 0.5,
                                    self.rest_density * volume, volume, 0)
            return

        # Conservative two-level layout. Build a complete coarse lattice first
        # and replace selected cells by their eight octree children.  The old
        # implementation generated coarse and fine layers independently; their
        # control volumes overlapped at the interface and raised rho/rho0 by
        # about 45 percent. Exact 1:2 splitting preserves occupied volume, mass
        # and momentum by construction.
        if abs(coarse - 2.0 * fine) > 1.0e-6:
            raise ValueError("adaptive water requires coarse_spacing == 2 * fine_spacing")

        coarse_volume = coarse ** 3
        fine_volume = fine ** 3
        child_offset = fine * 0.5
        for x in np.arange(-width / 2 + coarse / 2, width / 2, coarse):
            for z in np.arange(z_min + coarse / 2, z_max, coarse):
                elevation = crest * math.exp(-((z - z_max + 5.0) / 7.5) ** 2)
                local_depth = depth + elevation
                flow_speed = speed * elevation / max(local_depth, 1.0e-6)
                refine_from = max(coarse * 0.5, local_depth - surface_band)
                for y in np.arange(coarse / 2, local_depth, coarse):
                    parent = np.array((x, y, z), dtype=np.float64)
                    # Refine every coarse control volume intersecting the
                    # requested surface band, not an independently sampled
                    # slab. This leaves neither gaps nor double-counted volume.
                    if y + coarse * 0.5 > refine_from:
                        for sx in (-1.0, 1.0):
                            for sy in (-1.0, 1.0):
                                for sz in (-1.0, 1.0):
                                    child = parent + np.array((sx, sy, sz)) * child_offset
                                    velocity = (0.0, 0.0, background_current + flow_speed * (0.82 + 0.18 * child[1] / local_depth))
                                    self.append(child, velocity, fine * 0.5,
                                                self.rest_density * fine_volume, fine_volume, 0)
                    else:
                        velocity = (0.0, 0.0, background_current + flow_speed * (0.82 + 0.18 * y / local_depth))
                        self.append(parent, velocity, coarse * 0.5,
                                    self.rest_density * coarse_volume, coarse_volume, 0)

    def add_building(self, building_id: int, center_x: float, center_z: float, width: float, depth: float,
                     height: float, spacing: float, style: dict | None = None):
        # Coordinate-keyed lattice prevents duplicate particles where slabs,
        # columns and walls meet. Structural role priority decides which
        # refinement rule owns an intersection.
        lattice: dict[tuple[int, int, int], tuple[np.ndarray, int, int, int, int]] = {}
        floor_h = 3.0
        role_priority = {
            STRUCT_GLASS: 1,
            STRUCT_SLAB: 2,
            STRUCT_WALL: 3,
            STRUCT_BEAM: 4,
            STRUCT_CORE: 5,
            STRUCT_COLUMN: 6,
        }

        def add_local(x, y, z, material=1, is_fixed=0, structural_class=STRUCT_WALL):
            world = np.array((center_x + x, y, center_z + z), dtype=np.float32)
            key = tuple(np.rint(world / spacing).astype(np.int32))
            previous = lattice.get(key)
            priority = role_priority[structural_class]
            if previous is None:
                lattice[key] = (world, material, is_fixed, structural_class, priority)
            elif priority > previous[4] or (priority == previous[4] and material > previous[1]):
                lattice[key] = (world, material, max(is_fixed, previous[2]), structural_class, priority)
            elif is_fixed and not previous[2]:
                lattice[key] = (previous[0], previous[1], 1, previous[3], previous[4])

        ys = np.arange(0.0, height + spacing * 0.25, spacing)

        def profile(y):
            return building_profile(style, width, depth, height, y)

        def samples(length):
            return np.arange(-length / 2, length / 2 + spacing * 0.25, spacing)

        def add_plate(floor_y, ox, oz, local_width, local_depth):
            y = round(floor_y / spacing) * spacing
            for x in samples(local_width):
                for z in samples(local_depth):
                    add_local(ox + x, y, oz + z, 1, int(y < spacing * 0.6), STRUCT_SLAB)

        # Floor plates define actual apartment boxes rather than a solid cuboid.
        for floor_y in np.arange(0.0, height + 0.1, floor_h):
            ox, oz, local_width, local_depth = profile(floor_y)
            add_plate(floor_y, ox, oz, local_width, local_depth)
            # At a setback, keep the previous full plate as a real terrace and
            # load-transfer diaphragm instead of leaving a decorative overhang.
            if floor_y > 0.0:
                previous = profile(max(0.0, floor_y - 0.05))
                if any(abs(a - b) > spacing * 0.25 for a, b in zip(previous, (ox, oz, local_width, local_depth))):
                    add_plate(floor_y, *previous)

        # Four external walls with window particles and concrete mullions.
        for y in ys:
            ox, oz, local_width, local_depth = profile(y)
            xs = samples(local_width)
            zs = samples(local_depth)
            floor_phase = y % floor_h
            for z in zs:
                window = 0.8 < floor_phase < 2.5 and abs((z + local_depth / 2) % 3.0 - 1.5) < 1.05
                role = STRUCT_GLASS if window else STRUCT_WALL
                add_local(ox - local_width / 2, y, oz + z, 2 if window else 1, int(y < spacing * 0.6), role)
                add_local(ox + local_width / 2, y, oz + z, 2 if window else 1, int(y < spacing * 0.6), role)
            for x in xs:
                window = 0.8 < floor_phase < 2.5 and abs((x + local_width / 2) % 3.0 - 1.5) < 1.05
                role = STRUCT_GLASS if window else STRUCT_WALL
                add_local(ox + x, y, oz - local_depth / 2, 2 if window else 1, int(y < spacing * 0.6), role)
                add_local(ox + x, y, oz + local_depth / 2, 2 if window else 1, int(y < spacing * 0.6), role)

        # Central corridor/shear wall plus transverse apartment partitions.
        for y in ys:
            ox, oz, local_width, local_depth = profile(y)
            xs = samples(local_width)
            zs = samples(local_depth)
            for z in zs:
                add_local(ox, y, oz + z, 1, int(y < spacing * 0.6), STRUCT_CORE)
            for wall_z in np.arange(-local_depth / 2 + 3.5, local_depth / 2, 4.0):
                z = round(wall_z / spacing) * spacing
                for x in xs:
                    # Door opening into corridor on each floor.
                    if abs(x) < 1.2 and 0.2 < (y % floor_h) < 2.3:
                        continue
                    add_local(ox + x, y, oz + z, 1, int(y < spacing * 0.6), STRUCT_WALL)

        # Steel perimeter beams tie every floor plate into the column frame.
        for floor_y in np.arange(0.0, height + 0.1, floor_h):
            y = round(floor_y / spacing) * spacing
            ox, oz, local_width, local_depth = profile(floor_y)
            xs = samples(local_width)
            zs = samples(local_depth)
            for x in xs:
                add_local(ox + x, y, oz - local_depth / 2, 3, int(y < spacing * 0.75), STRUCT_BEAM)
                add_local(ox + x, y, oz + local_depth / 2, 3, int(y < spacing * 0.75), STRUCT_BEAM)
            for z in zs:
                add_local(ox - local_width / 2, y, oz + z, 3, int(y < spacing * 0.75), STRUCT_BEAM)
                add_local(ox + local_width / 2, y, oz + z, 3, int(y < spacing * 0.75), STRUCT_BEAM)

        # Reinforcement columns and service core use ductile steel particles.
        for y in ys:
            ox, oz, local_width, local_depth = profile(y)
            column_x = (-local_width / 2, -local_width / 4, local_width / 4, local_width / 2)
            column_z = (-local_depth / 2, 0.0, local_depth / 2)
            for x in column_x:
                for z in column_z:
                    add_local(ox + x, y, oz + z, 3, int(y < spacing * 0.75), STRUCT_COLUMN)
            for x in (-spacing, spacing):
                for z in (-spacing, spacing):
                    add_local(ox + x, y, oz + z, 3, int(y < spacing * 0.75), STRUCT_CORE)

        density_by_material = {1: 2350.0, 2: 2500.0, 3: 7850.0}
        volume = spacing ** 3 * 0.72
        for world, material, is_fixed, structural_class, _priority in lattice.values():
            self.append(world, (0.0, 0.0, 0.0), spacing * 0.48,
                        density_by_material[material] * volume, volume, 1, material,
                        building_id, is_fixed, structural_class)
        return len(lattice)

    def add_city(self, cfg: dict):
        spacing = float(cfg["solid_spacing"])
        buildings = cfg["buildings"]
        styles = cfg.get("building_styles", [])
        counts = []
        for i, spec in enumerate(buildings):
            style = styles[i] if i < len(styles) else None
            counts.append(self.add_building(i, *map(float, spec), spacing, style))
        return counts

    def add_environment(self, cfg: dict) -> dict[str, int]:
        """Add low-cost, water-coupled cars, breakable trees and small shops."""
        layout = environment_layout(cfg)
        counts = {"cars": 0, "trees": 0, "small_buildings": 0, "ground_anchors": 0}
        if not any(layout.values()):
            return counts

        # Four tiny fixed particles anchor render-only roads and terrain. They
        # sit inside the existing y=0 collision plane and add no obstacle.
        for x, z in ((-70.0, -20.0), (70.0, -20.0), (70.0, 130.0), (-70.0, 130.0)):
            self.append((x, 0.0, z), (0.0, 0.0, 0.0), 0.03, 1.0, 1.0e-5,
                        1, 1, -9000, 1, STRUCT_SLAB)
            counts["ground_anchors"] += 1

        for car in layout["cars"]:
            cx, cz = car["center"]
            for dx in (-1.5, 0.0, 1.5):
                for dy in (0.45, 1.15):
                    for dz in (-0.58, 0.58):
                        self.append(
                            (cx + dx, dy, cz + dz), (0.0, 0.0, 0.0), 0.48,
                            125.0, 0.60, 1, 3, int(car["id"]), 0, STRUCT_WALL,
                        )
                        counts["cars"] += 1

        for tree in layout["trees"]:
            cx, cz = tree["center"]
            height = float(tree["height"])
            trunk_levels = np.arange(0.4, height - 0.6, 0.8)
            for level, y in enumerate(trunk_levels):
                self.append(
                    (cx, float(y), cz), (0.0, 0.0, 0.0), 0.36,
                    72.0, 0.14, 1, 1, int(tree["id"]), int(level == 0), STRUCT_COLUMN,
                )
                counts["trees"] += 1
            crown_y = height - 0.2
            for dx, dy, dz in ((0, 0, 0), (-0.8, 0, 0), (0.8, 0, 0),
                               (0, 0.55, -0.65), (0, 0.55, 0.65)):
                self.append(
                    (cx + dx, crown_y + dy, cz + dz), (0.0, 0.0, 0.0), 0.82,
                    42.0, 0.75, 1, 1, int(tree["id"]), 0, STRUCT_WALL,
                )
                counts["trees"] += 1

        spacing = 1.2
        for shop in layout["small_buildings"]:
            cx, cz = shop["center"]
            width, depth, height = map(float, shop["size"])
            xs = np.arange(-width * 0.5, width * 0.5 + 0.1, spacing)
            zs = np.arange(-depth * 0.5, depth * 0.5 + 0.1, spacing)
            ys = np.arange(0.0, height + 0.1, spacing)
            occupied: set[tuple[int, int, int]] = set()
            for y in ys:
                for x in xs:
                    for z in (-depth * 0.5, depth * 0.5):
                        occupied.add((round(x / spacing), round(y / spacing), round(z / spacing)))
                for z in zs:
                    for x in (-width * 0.5, width * 0.5):
                        occupied.add((round(x / spacing), round(y / spacing), round(z / spacing)))
            roof_y = round(height / spacing)
            for x in xs:
                for z in zs:
                    occupied.add((round(x / spacing), roof_y, round(z / spacing)))
            volume = spacing ** 3 * 0.28
            for gx, gy, gz in occupied:
                y = gy * spacing
                self.append(
                    (cx + gx * spacing, y, cz + gz * spacing), (0.0, 0.0, 0.0),
                    spacing * 0.48, 850.0 * volume, volume, 1, 1,
                    int(shop["id"]), int(gy == 0), STRUCT_WALL if gy != roof_y else STRUCT_SLAB,
                )
                counts["small_buildings"] += 1
        return counts

    def as_numpy(self):
        return {
            "x": np.asarray(self.positions, dtype=np.float32),
            "v": np.asarray(self.velocities, dtype=np.float32),
            "radius": np.asarray(self.radii, dtype=np.float32),
            "mass": np.asarray(self.masses, dtype=np.float32),
            "volume": np.asarray(self.volumes, dtype=np.float32),
            "kind": np.asarray(self.kinds, dtype=np.int32),
            "material": np.asarray(self.materials, dtype=np.int32),
            "building_id": np.asarray(self.building_ids, dtype=np.int32),
            "fixed": np.asarray(self.fixed, dtype=np.int32),
            "rest_x": np.asarray(self.rest_positions, dtype=np.float32),
            "damage": np.asarray(self.damage, dtype=np.float32),
            "structural_class": np.asarray(self.structural_classes, dtype=np.int32),
        }
