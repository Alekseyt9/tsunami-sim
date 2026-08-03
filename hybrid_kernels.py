"""Small V3 GPU kernels layered on top of the stable V2 solver.

V3 keeps inactive buildings as immovable SPH boundaries, but skips their
expensive bond traversal until enough facade particles receive water load.
"""

import warp as wp


SPH_PI = wp.constant(3.141592653589793)


@wp.kernel
def precompute_sph_kernel_coefficients(
    radius: wp.array(dtype=float),
    support: wp.array(dtype=float),
    support_squared: wp.array(dtype=float),
    poly6_coefficient: wp.array(dtype=float),
    spiky_coefficient: wp.array(dtype=float),
    viscosity_coefficient: wp.array(dtype=float),
):
    """Cache radius-dependent SPH powers once instead of per neighbour pair."""
    i = wp.tid()
    h = 4.0 * radius[i]
    h2 = h * h
    h3 = h2 * h
    h6 = h3 * h3
    h9 = h6 * h3
    support[i] = h
    support_squared[i] = h2
    poly6_coefficient[i] = 315.0 / (64.0 * SPH_PI * h9)
    spiky_coefficient[i] = -45.0 / (SPH_PI * h6)
    viscosity_coefficient[i] = 45.0 / (SPH_PI * h6)


@wp.func
def cached_poly6(r2: float, h2: float, coefficient: float) -> float:
    value = wp.max(h2 - r2, 0.0)
    return coefficient * value * value * value


@wp.func
def cached_spiky_grad(
    r: wp.vec3, distance: float, h: float, coefficient: float
) -> wp.vec3:
    value = wp.max(h - distance, 0.0)
    return coefficient * value * value * r / distance


@wp.func
def cached_viscosity_laplacian(
    distance: float, h: float, coefficient: float
) -> float:
    return coefficient * wp.max(h - distance, 0.0)


@wp.kernel
def apply_conservative_fluid_merges(
    representatives: wp.array(dtype=wp.int32),
    merged_position: wp.array(dtype=wp.vec3),
    merged_velocity: wp.array(dtype=wp.vec3),
    merged_mass: wp.array(dtype=float),
    merged_volume: wp.array(dtype=float),
    merged_radius: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    radius: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    fluid_group_id: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    surface_normal: wp.array(dtype=wp.vec3),
    foam_strength: wp.array(dtype=float),
    water_phase: wp.array(dtype=wp.int32),
    phase_candidate: wp.array(dtype=wp.int32),
    phase_candidate_age: wp.array(dtype=wp.int32),
):
    merge_index = wp.tid()
    particle = representatives[merge_index]
    x[particle] = merged_position[merge_index]
    rest_x[particle] = merged_position[merge_index]
    v[particle] = merged_velocity[merge_index]
    mass[particle] = merged_mass[merge_index]
    volume[particle] = merged_volume[merge_index]
    radius[particle] = merged_radius[merge_index]
    rho_reference[particle] = 0.0
    rho[particle] = 0.0
    acceleration[particle] = wp.vec3(0.0)
    solid_force[particle] = wp.vec3(0.0)
    fluid_group_id[particle] = -1
    surface_mask[particle] = 0
    surface_normal[particle] = wp.vec3(0.0)
    foam_strength[particle] = 0.0
    water_phase[particle] = 0
    phase_candidate[particle] = 0
    phase_candidate_age[particle] = 0

from kernels import (
    material_failure_strain,
    material_stiffness,
    poly6,
    project_point,
    spiky_grad,
    viscosity_laplacian,
)
from scene import STRUCT_BEAM, STRUCT_COLUMN, STRUCT_CORE, STRUCT_GLASS, STRUCT_SLAB, STRUCT_WALL


@wp.func
def structural_failure_strain_multiplier(role: int) -> float:
    if role == STRUCT_GLASS:
        return 0.65
    if role == STRUCT_SLAB:
        return 1.25
    if role == STRUCT_BEAM:
        return 1.60
    if role == STRUCT_COLUMN:
        return 1.90
    if role == STRUCT_CORE:
        return 2.20
    return 1.0  # wall / unknown


@wp.func
def structural_damage_rate_multiplier(role: int) -> float:
    if role == STRUCT_GLASS:
        return 1.40
    if role == STRUCT_SLAB:
        return 0.75
    if role == STRUCT_BEAM:
        return 0.55
    if role == STRUCT_COLUMN:
        return 0.40
    if role == STRUCT_CORE:
        return 0.30
    return 1.0


@wp.func
def material_impact_min_acceleration(role: int) -> float:
    """Acceleration below this level is treated as harmless local wash."""
    if role == STRUCT_GLASS:
        return 3.0
    if role == STRUCT_WALL:
        return 8.0
    if role == STRUCT_SLAB:
        return 12.0
    if role == STRUCT_BEAM:
        return 15.0
    if role == STRUCT_COLUMN:
        return 18.0
    if role == STRUCT_CORE:
        return 22.0
    return 10.0


@wp.func
def material_impact_impulse_threshold(role: int) -> float:
    """Required accumulated impulse per unit mass (equivalent delta-v)."""
    if role == STRUCT_GLASS:
        return 0.035
    if role == STRUCT_WALL:
        return 0.20
    if role == STRUCT_SLAB:
        return 0.35
    if role == STRUCT_BEAM:
        return 0.45
    if role == STRUCT_COLUMN:
        return 0.65
    if role == STRUCT_CORE:
        return 0.85
    return 0.30


@wp.func
def material_impact_damage_drive(role: int, impulse: float) -> float:
    """Continuous fracture drive above the role-specific impact threshold.

    Crossing the threshold merely starts a crack.  The old boolean gate gave
    a shallow runnel the same fracture rate as the full bore once both were a
    tiny amount above threshold.  Full-rate damage now requires roughly three
    times the threshold impulse.
    """
    threshold = material_impact_impulse_threshold(role)
    return wp.clamp(
        (impulse - threshold) / wp.max(2.0 * threshold, 1.0e-5),
        0.0,
        1.0,
    )


@wp.func
def material_impact_decay_time(role: int) -> float:
    if role == STRUCT_GLASS:
        return 0.06
    if role == STRUCT_WALL:
        return 0.12
    return 0.18


@wp.func
def deformable_contact_magnitude(
    penetration: float,
    closing_speed: float,
    stiffness: float,
    damping: float,
) -> float:
    """Non-attractive penalty contact with dissipative approach damping."""
    return wp.max(
        stiffness * penetration + damping * wp.max(-closing_speed, 0.0),
        0.0,
    )


@wp.func
def collapse_gravity_fraction(
    damage_integral: float,
    structural_volume: float,
    onset_fraction: float,
    full_fraction: float,
) -> float:
    """Ramp building-wide gravity only after causal structural damage."""
    fraction = damage_integral / wp.max(structural_volume, 1.0e-6)
    return wp.clamp(
        (fraction - onset_fraction) / wp.max(full_fraction - onset_fraction, 1.0e-6),
        0.0,
        1.0,
    )


@wp.func
def preloaded_structure_gravity_fraction(
    building_id: int,
    support_loss_fraction: float,
    body_rigid: bool,
) -> float:
    """Dynamic gravity left after subtracting the authored static preload."""
    if body_rigid or building_id < 0:
        return 1.0
    return wp.clamp(support_loss_fraction, 0.0, 1.0)


@wp.func
def facade_support_loss_rate(
    role: int,
    rest_elevation: float,
    collapse_fraction: float,
    minimum_elevation: float,
    collapse_threshold: float,
    maximum_rate: float,
) -> float:
    """Loss of diaphragm support for upper facade chunks after global failure."""
    facade = role == STRUCT_WALL or role == STRUCT_GLASS
    if not facade or rest_elevation <= minimum_elevation:
        return 0.0
    activation = wp.clamp(
        (collapse_fraction - collapse_threshold) / wp.max(1.0 - collapse_threshold, 1.0e-6),
        0.0,
        1.0,
    )
    return maximum_rate * activation


@wp.kernel
def accumulate_building_damage(
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    damage_integral: wp.array(dtype=float),
):
    i = wp.tid()
    bid = building_id[i]
    if kind[i] != 0 and bid >= 0:
        wp.atomic_add(damage_integral, bid, volume[i] * damage[i])


@wp.kernel
def accumulate_loaded_building_volume(
    rest_x: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    load_acceleration_threshold: float,
    maximum_activation_elevation: float,
    loaded_volume: wp.array(dtype=float),
):
    i = wp.tid()
    bid = building_id[i]
    if kind[i] != 0 and bid >= 0 and rest_x[i][1] <= maximum_activation_elevation:
        threshold = mass[i] * load_acceleration_threshold
        # The tsunami travels along +Z. Requiring forward load near the base
        # prevents isolated overhead/side spray from waking the whole graph.
        if solid_force[i][2] > threshold:
            # Volume is invariant under adaptive particle splitting.  A fixed
            # hit count became easier to satisfy every time a facade refined.
            wp.atomic_add(loaded_volume, bid, volume[i])


@wp.kernel
def activate_buildings_from_load(
    loaded_volume: wp.array(dtype=float),
    eligible_base_volume: wp.array(dtype=float),
    active: wp.array(dtype=wp.int32),
    exposure_seconds: wp.array(dtype=float),
    minimum_loaded_fraction: float,
    dt: float,
    required_exposure_seconds: float,
    exposure_decay_multiplier: float,
):
    bid = wp.tid()
    if active[bid] != 0:
        return
    loaded_fraction = loaded_volume[bid] / wp.max(eligible_base_volume[bid], 1.0e-6)
    if loaded_fraction >= minimum_loaded_fraction:
        exposure_seconds[bid] += dt
    else:
        exposure_seconds[bid] = wp.max(
            0.0, exposure_seconds[bid] - dt * exposure_decay_multiplier
        )
    if exposure_seconds[bid] >= required_exposure_seconds:
        active[bid] = 1


@wp.kernel
def accumulate_material_impact(
    kind: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    mass: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    debris_contact_force: wp.array(dtype=wp.vec3),
    impact_impulse: wp.array(dtype=float),
    local_impact_active: wp.array(dtype=wp.int32),
    dt: float,
):
    """Low-pass local water impulse so spray and sustained bores differ."""
    i = wp.tid()
    if kind[i] == 0:
        impact_impulse[i] = 0.0
        local_impact_active[i] = 0
        return
    role = structural_class[i]
    # Hydrodynamic and solid-contact reactions are kept in separate buffers so
    # rubble cannot masquerade as a coherent tsunami-front load.  For local
    # material damage, however, either source is physically meaningful.
    load_acceleration = wp.max(
        wp.length(solid_force[i]), wp.length(debris_contact_force[i])
    ) / wp.max(mass[i], 1.0)
    excess = wp.max(load_acceleration - material_impact_min_acceleration(role), 0.0)
    decay = wp.exp(-dt / material_impact_decay_time(role))
    accumulated = wp.min(impact_impulse[i] * decay + excess * dt, 5.0)
    impact_impulse[i] = accumulated
    # A sufficiently massive isolated splash may release glazing without
    # waking the entire concrete frame. Other materials still require the
    # coherent lower-facade building activation gate.
    if role == STRUCT_GLASS and accumulated >= material_impact_impulse_threshold(role):
        local_impact_active[i] = 1


@wp.kernel
def accumulate_dormant_debris_contacts(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    building_active: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.vec3),
    impacted_volume: wp.array(dtype=float),
    peak_acceleration: wp.array(dtype=float),
    maximum_query_radius: float,
    impact_acceleration_threshold: float,
):
    """Receive equal-and-opposite rubble impacts on sleeping structures.

    Sleeping buildings remain kinematic water boundaries, but they must not be
    immune to a slab or vehicle striking them.  This target-side pass records
    only contacts from dynamic solids belonging to another structure.  Water
    activation continues to consume the independent hydrodynamic-force array.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    bid = building_id[i]
    if kind[i] == 0 or bid < 0 or building_active[bid] != 0:
        return

    xi = x[i]
    force = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, maximum_query_radius)
    for j in query:
        source_fragment = fragment_id[j]
        source_rigid = source_fragment >= 0 and rigid_state[source_fragment] != 0
        if (
            j == i or kind[j] == 0 or (fixed[j] != 0 and not source_rigid)
            or building_id[j] == bid
        ):
            continue
        delta = x[j] - xi
        distance = wp.length(delta)
        contact_distance = radius[i] + radius[j]
        if distance <= 1.0e-5 or distance >= contact_distance:
            continue
        normal = delta / distance
        closing = wp.dot(v[j] - v[i], normal)
        magnitude = deformable_contact_magnitude(
            contact_distance - distance, closing, 3.0e6, 9000.0
        )
        force -= normal * magnitude

    contact_force[i] = force
    acceleration = wp.length(force) / wp.max(mass[i], 1.0)
    if acceleration >= impact_acceleration_threshold:
        wp.atomic_add(impacted_volume, bid, volume[i])
        wp.atomic_max(peak_acceleration, bid, acceleration)


@wp.kernel
def apply_dormant_impact_damage(
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    building_active: wp.array(dtype=wp.int32),
    impact_impulse: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    dt: float,
    glass_damage_rate: float,
    wall_damage_rate: float,
):
    """Create local glazing/spall damage without waking an entire building."""
    i = wp.tid()
    bid = building_id[i]
    if kind[i] == 0 or bid < 0 or building_active[bid] != 0:
        return
    role = structural_class[i]
    drive = material_impact_damage_drive(role, impact_impulse[i])
    if drive <= 0.0:
        return
    rate = float(0.0)
    if role == STRUCT_GLASS:
        rate = glass_damage_rate
    elif role == STRUCT_WALL:
        rate = wall_damage_rate
    damage[i] = wp.min(1.0, damage[i] + dt * rate * drive)


@wp.kernel
def activate_buildings_from_debris_impact(
    impacted_volume: wp.array(dtype=float),
    structural_volume: wp.array(dtype=float),
    peak_acceleration: wp.array(dtype=float),
    active: wp.array(dtype=wp.int32),
    exposure_seconds: wp.array(dtype=float),
    minimum_impacted_fraction: float,
    minimum_peak_acceleration: float,
    required_exposure_seconds: float,
    exposure_decay_multiplier: float,
    dt: float,
):
    """Wake a structural graph only after a spatially coherent rubble hit."""
    bid = wp.tid()
    if active[bid] != 0:
        return
    fraction = impacted_volume[bid] / wp.max(structural_volume[bid], 1.0e-6)
    peak = peak_acceleration[bid]
    if fraction >= minimum_impacted_fraction and peak >= minimum_peak_acceleration:
        spatial = wp.clamp(fraction / wp.max(minimum_impacted_fraction, 1.0e-6), 1.0, 4.0)
        severity = wp.clamp(peak / wp.max(minimum_peak_acceleration, 1.0e-6), 1.0, 4.0)
        exposure_seconds[bid] += dt * wp.sqrt(spatial * severity)
    else:
        exposure_seconds[bid] = wp.max(
            0.0, exposure_seconds[bid] - dt * exposure_decay_multiplier
        )
    if exposure_seconds[bid] >= required_exposure_seconds:
        active[bid] = 1


@wp.kernel
def apply_building_activity(
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    base_fixed: wp.array(dtype=wp.int32),
    building_active: wp.array(dtype=wp.int32),
    local_impact_active: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    bid = building_id[i]
    if kind[i] != 0 and bid >= 0:
        locally_released_glass = (
            structural_class[i] == STRUCT_GLASS and local_impact_active[i] != 0
        )
        if building_active[bid] != 0 or locally_released_glass:
            fixed[i] = base_fixed[i]
        else:
            fixed[i] = 1


@wp.kernel
def count_structural_adjacency(
    grid: wp.uint64,
    rest_x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    query_radius: float,
):
    """Count immutable rest-lattice neighbours for a GPU CSR graph."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] == 0:
        neighbour_count[i] = 0
        return
    ri = rest_x[i]
    count = int(0)
    query = wp.hash_grid_query(grid, ri, query_radius)
    for j in query:
        if j == i or kind[j] == 0 or building_id[i] != building_id[j]:
            continue
        rest_distance = wp.length(rest_x[j] - ri)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        if rest_distance > 1.0e-5 and rest_distance < bond_range:
            count += 1
    neighbour_count[i] = count


@wp.kernel
def fill_structural_adjacency(
    grid: wp.uint64,
    rest_x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    query_radius: float,
):
    """Fill the immutable directed CSR adjacency in the same order as count."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] == 0:
        return
    ri = rest_x[i]
    cursor = neighbour_offset[i]
    query = wp.hash_grid_query(grid, ri, query_radius)
    for j in query:
        if j == i or kind[j] == 0 or building_id[i] != building_id[j]:
            continue
        rest_distance = wp.length(rest_x[j] - ri)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        if rest_distance > 1.0e-5 and rest_distance < bond_range:
            neighbour_index[cursor] = j
            cursor += 1


@wp.kernel
def compute_clustered_solid_forces_adjacency(
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    building_damage_integral: wp.array(dtype=float),
    building_structural_volume: wp.array(dtype=float),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fragment_support: wp.array(dtype=float),
    impact_impulse: wp.array(dtype=float),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    query_radius: float,
    dt: float,
    internal_stiffness_multiplier: float,
    damage_rate: float,
    propagation_threshold: float,
    max_damage_per_substep: float,
    fracture_reference_radius: float,
    collapse_gravity_damage_onset: float,
    collapse_gravity_damage_full: float,
    facade_support_loss_minimum_elevation: float,
    facade_support_loss_collapse_threshold: float,
    facade_support_loss_damage_rate: float,
    facade_unsupported_damage_rate: float,
    structural_unsupported_damage_rate: float,
    elastic_force_cap_multiplier: float,
    compression_force_cap_multiplier: float,
):
    """Structural springs/support over a persistent rest-lattice CSR graph."""
    i = wp.tid()
    if kind[i] == 0:
        return
    if fixed[i] != 0:
        acceleration[i] = wp.vec3(0.0)
        return

    xi = x[i]
    ri = rest_x[i]
    fid = fragment_id[i]
    body_rigid = fid >= 0 and rigid_state[fid] != 0
    bid = building_id[i]
    building_collapse = float(0.0)
    if fid >= 0:
        building_collapse = 1.0 - wp.clamp(fragment_support[fid], 0.0, 1.0)
    elif bid >= 0:
        building_collapse = collapse_gravity_fraction(
            building_damage_integral[bid], building_structural_volume[bid],
            collapse_gravity_damage_onset, collapse_gravity_damage_full,
        )
    local_damage = damage[i]
    local_damage += dt * facade_support_loss_rate(
        structural_class[i], ri[1], building_collapse,
        facade_support_loss_minimum_elevation,
        facade_support_loss_collapse_threshold,
        facade_support_loss_damage_rate,
    )
    gravity_fraction = preloaded_structure_gravity_fraction(
        bid, building_collapse, body_rigid
    )
    force = solid_force[i]
    impact_drive = material_impact_damage_drive(
        structural_class[i], impact_impulse[i]
    )
    facade_particle = (
        structural_class[i] == STRUCT_WALL
        or structural_class[i] == STRUCT_GLASS
    )
    has_local_support = int(0)
    start = neighbour_offset[i]
    end = start + neighbour_count[i]
    cell_x = int(wp.floor(xi[0] / query_radius))
    cell_y = int(wp.floor(xi[1] / query_radius))
    cell_z = int(wp.floor(xi[2] / query_radius))
    for edge in range(start, end):
        j = neighbour_index[edge]
        delta = x[j] - xi
        # Match the legacy HashGrid candidate window.  Persistent rest bonds
        # must not behave like infinitely long springs after two fragments
        # have moved into non-neighbouring current-space cells.
        neighbour_cell_x = int(wp.floor(x[j][0] / query_radius))
        neighbour_cell_y = int(wp.floor(x[j][1] / query_radius))
        neighbour_cell_z = int(wp.floor(x[j][2] / query_radius))
        if (
            wp.abs(neighbour_cell_x - cell_x) > 1
            or wp.abs(neighbour_cell_y - cell_y) > 1
            or wp.abs(neighbour_cell_z - cell_z) > 1
        ):
            continue
        dist = wp.length(delta)
        if dist <= 1.0e-5:
            continue
        rest_delta = rest_x[j] - ri
        rest_dist = wp.length(rest_delta)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        same_fragment = fid >= 0 and fid == fragment_id[j]
        neighbour_fid = fragment_id[j]
        neighbour_rigid = neighbour_fid >= 0 and rigid_state[neighbour_fid] != 0
        neighbour_has_foundation_path = (
            neighbour_fid < 0 or fragment_support[neighbour_fid] > 0.5
        )
        if (
            facade_particle and damage[j] < 0.90
            and neighbour_has_foundation_path
        ):
            horizontal_rest2 = (
                rest_delta[0] * rest_delta[0] + rest_delta[2] * rest_delta[2]
            )
            below = (
                rest_delta[1] < -0.25 * bond_range
                and horizontal_rest2 < 0.56 * bond_range * bond_range
            )
            neighbour_role = structural_class[j]
            frame_member = (
                neighbour_role == STRUCT_SLAB or neighbour_role == STRUCT_BEAM
                or neighbour_role == STRUCT_COLUMN or neighbour_role == STRUCT_CORE
            )
            diaphragm = frame_member and wp.abs(rest_delta[1]) < 0.55 * bond_range
            current_attachment_intact = dist < rest_dist * 1.55
            if (below or diaphragm) and current_attachment_intact:
                has_local_support = 1

        if body_rigid and same_fragment:
            continue
        if same_fragment:
            strain = (dist - rest_dist) / wp.max(rest_dist, 1.0e-4)
            yield_strain = wp.min(
                material_failure_strain(material[i]),
                material_failure_strain(material[j]),
            )
            yield_strain *= wp.min(
                structural_failure_strain_multiplier(structural_class[i]),
                structural_failure_strain_multiplier(structural_class[j]),
            )
            yield_strain *= elastic_force_cap_multiplier
            transmitted_strain = wp.clamp(
                strain, -yield_strain * compression_force_cap_multiplier,
                yield_strain,
            )
            stiffness = wp.min(
                material_stiffness(material[i]), material_stiffness(material[j])
            )
            damping = 75000.0 * wp.dot(v[j] - v[i], delta / dist)
            force += (
                stiffness * internal_stiffness_multiplier * transmitted_strain
                + damping
            ) * (delta / dist) * radius[i] * radius[i]
        elif local_damage < 1.0 and not body_rigid and not neighbour_rigid:
            strain = (dist - rest_dist) / wp.max(rest_dist, 1.0e-4)
            limit = wp.min(
                material_failure_strain(material[i]),
                material_failure_strain(material[j]),
            )
            limit *= wp.min(
                structural_failure_strain_multiplier(structural_class[i]),
                structural_failure_strain_multiplier(structural_class[j]),
            )
            bond_radius = wp.max(radius[i], radius[j])
            resolution_scale = wp.sqrt(
                fracture_reference_radius / wp.max(bond_radius, 1.0e-5)
            )
            resolution_scale = wp.clamp(resolution_scale, 0.65, 2.5)
            limit *= resolution_scale
            abs_strain = wp.abs(strain)
            propagated_crack = (
                damage[j] > propagation_threshold and abs_strain > limit * 2.5
            )
            crack_drive = impact_drive
            if propagated_crack:
                crack_drive = wp.max(
                    crack_drive,
                    wp.clamp(
                        (damage[j] - propagation_threshold)
                        / wp.max(1.0 - propagation_threshold, 1.0e-5),
                        0.0, 1.0,
                    ),
                )
            if abs_strain > limit and crack_drive > 0.0:
                normalized = (abs_strain - limit) / wp.max(limit, 1.0e-4)
                role_rate = wp.max(
                    structural_damage_rate_multiplier(structural_class[i]),
                    structural_damage_rate_multiplier(structural_class[j]),
                )
                increment = wp.min(
                    normalized * dt * damage_rate * role_rate,
                    max_damage_per_substep * role_rate,
                ) * crack_drive
                local_damage += increment
            if local_damage < 1.0:
                stiffness = wp.min(
                    material_stiffness(material[i]), material_stiffness(material[j])
                )
                damping = 50000.0 * wp.dot(v[j] - v[i], delta / dist)
                cohesion = (1.0 - local_damage) * (1.0 - local_damage)
                transmitted_strain = wp.clamp(
                    strain,
                    -limit * elastic_force_cap_multiplier
                    * compression_force_cap_multiplier,
                    limit * elastic_force_cap_multiplier,
                )
                force += cohesion * (
                    stiffness * transmitted_strain + damping
                ) * (delta / dist) * radius[i] * radius[i]

    if (
        facade_particle and ri[1] > facade_support_loss_minimum_elevation
        and has_local_support == 0 and not body_rigid
    ):
        gravity_fraction = 1.0
        local_damage += dt * facade_unsupported_damage_rate
    elif (
        fid >= 0 and fragment_support[fid] < 0.5
        and ri[1] > facade_support_loss_minimum_elevation and not body_rigid
    ):
        local_damage += (
            dt * structural_unsupported_damage_rate
            * structural_damage_rate_multiplier(structural_class[i])
        )
    if body_rigid:
        gravity_fraction = 1.0
    force += wp.vec3(0.0, -9.81 * mass[i] * gravity_fraction, 0.0)
    damage[i] = wp.min(local_damage, 1.0)
    # The legacy kernel clamps only after spring and dynamic contact forces
    # have been combined.  The split CSR path therefore leaves this partial
    # acceleration unclamped; accumulate_deformable_contacts_adjacency applies
    # the single final cap after adding contact.
    acceleration[i] = force / wp.max(mass[i], 1.0)


@wp.kernel
def append_dynamic_solid_particles(
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    membership: wp.array(dtype=wp.int32),
    particle_index: wp.array(dtype=wp.int32),
    particle_count: wp.array(dtype=wp.int32),
):
    """Incrementally append newly dynamic solids without rebuilding the list."""
    i = wp.tid()
    if kind[i] == 0 or fixed[i] != 0 or membership[i] != 0:
        return
    membership[i] = 1
    slot = wp.atomic_add(particle_count, 0, 1)
    particle_index[slot] = i


@wp.kernel
def mark_spatial_dynamic_solid_particles(
    grid: wp.uint64,
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    spatial_particle: wp.array(dtype=wp.int32),
    spatial_flag: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    spatial_particle[tid] = i
    spatial_flag[tid] = int(kind[i] != 0 and fixed[i] == 0)


@wp.kernel
def scatter_spatial_dynamic_solid_particles(
    spatial_particle: wp.array(dtype=wp.int32),
    spatial_flag: wp.array(dtype=wp.int32),
    spatial_offset: wp.array(dtype=wp.int32),
    particle_index: wp.array(dtype=wp.int32),
    particle_count: wp.array(dtype=wp.int32),
    total_particles: int,
):
    tid = wp.tid()
    if spatial_flag[tid] != 0:
        particle_index[spatial_offset[tid]] = spatial_particle[tid]
    if tid == total_particles - 1:
        particle_count[0] = spatial_offset[tid] + spatial_flag[tid]


@wp.kernel
def clear_deformable_fragment_bounds(
    bounds_lower_accum: wp.array2d(dtype=float),
    bounds_upper_accum: wp.array2d(dtype=float),
    active_fragment: wp.array(dtype=wp.int32),
    contact_candidate: wp.array(dtype=wp.int32),
):
    fragment = wp.tid()
    for axis in range(3):
        bounds_lower_accum[fragment, axis] = 1.0e6
        bounds_upper_accum[fragment, axis] = -1.0e6
    active_fragment[fragment] = 0
    contact_candidate[fragment] = 0


@wp.kernel
def accumulate_deformable_fragment_bounds(
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    bounds_lower_accum: wp.array2d(dtype=float),
    bounds_upper_accum: wp.array2d(dtype=float),
    active_fragment: wp.array(dtype=wp.int32),
):
    particle = wp.tid()
    fragment = fragment_id[particle]
    if kind[particle] == 0 or fixed[particle] != 0 or fragment < 0:
        return
    position = x[particle]
    particle_radius = radius[particle]
    for axis in range(3):
        wp.atomic_min(
            bounds_lower_accum, fragment, axis,
            position[axis] - particle_radius,
        )
        wp.atomic_max(
            bounds_upper_accum, fragment, axis,
            position[axis] + particle_radius,
        )
    active_fragment[fragment] = 1


@wp.kernel
def finalize_deformable_fragment_bounds(
    bounds_lower_accum: wp.array2d(dtype=float),
    bounds_upper_accum: wp.array2d(dtype=float),
    active_fragment: wp.array(dtype=wp.int32),
    bounds_lower: wp.array(dtype=wp.vec3),
    bounds_upper: wp.array(dtype=wp.vec3),
    margin: float,
):
    fragment = wp.tid()
    if active_fragment[fragment] == 0:
        far = 2.0e6 + float(fragment) * 2.0
        bounds_lower[fragment] = wp.vec3(far, far, far)
        bounds_upper[fragment] = wp.vec3(far + 0.01, far + 0.01, far + 0.01)
        return
    lower = wp.vec3(
        bounds_lower_accum[fragment, 0],
        bounds_lower_accum[fragment, 1],
        bounds_lower_accum[fragment, 2],
    ) - wp.vec3(margin)
    upper = wp.vec3(
        bounds_upper_accum[fragment, 0],
        bounds_upper_accum[fragment, 1],
        bounds_upper_accum[fragment, 2],
    ) + wp.vec3(margin)
    bounds_lower[fragment] = lower
    bounds_upper[fragment] = upper


@wp.kernel
def mark_deformable_fragment_contact_candidates_bvh(
    bvh_id: wp.uint64,
    bounds_lower: wp.array(dtype=wp.vec3),
    bounds_upper: wp.array(dtype=wp.vec3),
    active_fragment: wp.array(dtype=wp.int32),
    fragment_building: wp.array(dtype=wp.int32),
    fragment_support: wp.array(dtype=float),
    fragment_fracture_energy: wp.array(dtype=float),
    contact_candidate: wp.array(dtype=wp.int32),
    fracture_threshold: float,
):
    left = wp.tid()
    if active_fragment[left] == 0:
        return
    # A released deformable fragment can fold onto itself.  Its non-bonded
    # particle contacts are physically active even when its AABB does not
    # overlap another fragment, so it must not be culled by a pair-only gate.
    if (
        fragment_support[left] < 0.5
        or fragment_fracture_energy[left] >= fracture_threshold
    ):
        contact_candidate[left] = 1
    query = wp.bvh_query_aabb(bvh_id, bounds_lower[left], bounds_upper[left])
    right = int(0)
    while wp.bvh_query_next(query, right):
        if right <= left or active_fragment[right] == 0:
            continue
        released = (
            fragment_support[left] < 0.5 or fragment_support[right] < 0.5
            or fragment_fracture_energy[left] >= fracture_threshold
            or fragment_fracture_energy[right] >= fracture_threshold
        )
        if fragment_building[left] != fragment_building[right] or released:
            wp.atomic_max(contact_candidate, left, 1)
            wp.atomic_max(contact_candidate, right, 1)


@wp.kernel
def mark_environment_fragment_contact_candidates_bvh(
    bvh_id: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    active_fragment: wp.array(dtype=wp.int32),
    contact_candidate: wp.array(dtype=wp.int32),
    margin: float,
):
    particle = wp.tid()
    if kind[particle] == 0 or fixed[particle] != 0 or fragment_id[particle] >= 0:
        return
    extent = radius[particle] + margin
    lower = x[particle] - wp.vec3(extent)
    upper = x[particle] + wp.vec3(extent)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    fragment = int(0)
    while wp.bvh_query_next(query, fragment):
        if active_fragment[fragment] != 0:
            wp.atomic_max(contact_candidate, fragment, 1)


@wp.kernel
def accumulate_deformable_contacts_adjacency(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    fragment_contact_candidate: wp.array(dtype=wp.int32),
    acceleration: wp.array(dtype=wp.vec3),
    query_radius: float,
):
    """Dynamic contact pass separated from immutable structural springs."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] == 0 or fixed[i] != 0:
        return
    xi = x[i]
    ri = rest_x[i]
    fid = fragment_id[i]
    if fid >= 0 and fragment_contact_candidate[fid] == 0:
        return
    body_rigid = fid >= 0 and rigid_state[fid] != 0
    contact_force = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        if j == i or kind[j] == 0:
            continue
        delta = x[j] - xi
        dist_sq = wp.dot(delta, delta)
        contact_distance = radius[i] + radius[j]
        if dist_sq <= 1.0e-10 or dist_sq >= contact_distance * contact_distance:
            continue
        dist = wp.sqrt(dist_sq)
        neighbour_fid = fragment_id[j]
        neighbour_rigid = neighbour_fid >= 0 and rigid_state[neighbour_fid] != 0
        same_fragment = fid >= 0 and fid == neighbour_fid
        rest_delta = rest_x[j] - ri
        rest_dist_sq = wp.dot(rest_delta, rest_delta)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        bonded = (
            building_id[i] == building_id[j]
            and rest_dist_sq < bond_range * bond_range
        )
        spring_active = (
            (bonded and same_fragment)
            or (
                bonded and damage[i] < 1.0
                and not body_rigid and not neighbour_rigid
            )
        )
        if spring_active or (body_rigid and neighbour_rigid):
            continue
        normal = delta / dist
        closing = wp.dot(v[j] - v[i], normal)
        contact_force -= normal * deformable_contact_magnitude(
            contact_distance - dist, closing, 3.0e6, 9000.0
        )
    ai = acceleration[i] + contact_force / wp.max(mass[i], 1.0)
    a_len = wp.length(ai)
    if a_len > 6000.0:
        ai *= 6000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def accumulate_indexed_deformable_contacts_adjacency(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    fragment_contact_candidate: wp.array(dtype=wp.int32),
    acceleration: wp.array(dtype=wp.vec3),
    dynamic_particle_index: wp.array(dtype=wp.int32),
    dynamic_particle_count: wp.array(dtype=wp.int32),
    query_radius: float,
):
    """Resolve contacts only for solids that have ever become dynamic."""
    slot = wp.tid()
    if slot >= dynamic_particle_count[0]:
        return
    i = dynamic_particle_index[slot]
    if kind[i] == 0 or fixed[i] != 0:
        return
    xi = x[i]
    ri = rest_x[i]
    fid = fragment_id[i]
    if fid >= 0 and fragment_contact_candidate[fid] == 0:
        return
    body_rigid = fid >= 0 and rigid_state[fid] != 0
    contact_force = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        if j == i or kind[j] == 0:
            continue
        delta = x[j] - xi
        dist_sq = wp.dot(delta, delta)
        contact_distance = radius[i] + radius[j]
        if dist_sq <= 1.0e-10 or dist_sq >= contact_distance * contact_distance:
            continue
        dist = wp.sqrt(dist_sq)
        neighbour_fid = fragment_id[j]
        neighbour_rigid = neighbour_fid >= 0 and rigid_state[neighbour_fid] != 0
        same_fragment = fid >= 0 and fid == neighbour_fid
        rest_delta = rest_x[j] - ri
        rest_dist_sq = wp.dot(rest_delta, rest_delta)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        bonded = (
            building_id[i] == building_id[j]
            and rest_dist_sq < bond_range * bond_range
        )
        spring_active = (
            (bonded and same_fragment)
            or (
                bonded and damage[i] < 1.0
                and not body_rigid and not neighbour_rigid
            )
        )
        if spring_active or (body_rigid and neighbour_rigid):
            continue
        normal = delta / dist
        closing = wp.dot(v[j] - v[i], normal)
        contact_force -= normal * deformable_contact_magnitude(
            contact_distance - dist, closing, 3.0e6, 9000.0
        )
    ai = acceleration[i] + contact_force / wp.max(mass[i], 1.0)
    a_len = wp.length(ai)
    if a_len > 6000.0:
        ai *= 6000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def collect_deformable_contact_candidates(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    candidate_index: wp.array(dtype=wp.int32),
    candidate_count: wp.array(dtype=wp.int32),
    query_radius: float,
):
    """Compact particles with at least one exact active solid contact on GPU."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] == 0 or fixed[i] != 0:
        return
    xi = x[i]
    ri = rest_x[i]
    fid = fragment_id[i]
    body_rigid = fid >= 0 and rigid_state[fid] != 0
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        if j == i or kind[j] == 0:
            continue
        delta = x[j] - xi
        dist_sq = wp.dot(delta, delta)
        contact_distance = radius[i] + radius[j]
        if dist_sq <= 1.0e-10 or dist_sq >= contact_distance * contact_distance:
            continue
        neighbour_fid = fragment_id[j]
        neighbour_rigid = neighbour_fid >= 0 and rigid_state[neighbour_fid] != 0
        same_fragment = fid >= 0 and fid == neighbour_fid
        rest_delta = rest_x[j] - ri
        rest_dist_sq = wp.dot(rest_delta, rest_delta)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        bonded = (
            building_id[i] == building_id[j]
            and rest_dist_sq < bond_range * bond_range
        )
        spring_active = (
            (bonded and same_fragment)
            or (
                bonded and damage[i] < 1.0
                and not body_rigid and not neighbour_rigid
            )
        )
        if spring_active or (body_rigid and neighbour_rigid):
            continue
        slot = wp.atomic_add(candidate_count, 0, 1)
        candidate_index[slot] = i
        return


@wp.kernel
def accumulate_compact_deformable_contacts(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    candidate_index: wp.array(dtype=wp.int32),
    candidate_count: wp.array(dtype=wp.int32),
    query_radius: float,
):
    """Resolve contacts for the compact list without a CPU count readback."""
    slot = wp.tid()
    if slot >= candidate_count[0]:
        return
    i = candidate_index[slot]
    xi = x[i]
    ri = rest_x[i]
    fid = fragment_id[i]
    body_rigid = fid >= 0 and rigid_state[fid] != 0
    contact_force = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        if j == i or kind[j] == 0:
            continue
        delta = x[j] - xi
        dist_sq = wp.dot(delta, delta)
        contact_distance = radius[i] + radius[j]
        if dist_sq <= 1.0e-10 or dist_sq >= contact_distance * contact_distance:
            continue
        dist = wp.sqrt(dist_sq)
        neighbour_fid = fragment_id[j]
        neighbour_rigid = neighbour_fid >= 0 and rigid_state[neighbour_fid] != 0
        same_fragment = fid >= 0 and fid == neighbour_fid
        rest_delta = rest_x[j] - ri
        rest_dist_sq = wp.dot(rest_delta, rest_delta)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        bonded = (
            building_id[i] == building_id[j]
            and rest_dist_sq < bond_range * bond_range
        )
        spring_active = (
            (bonded and same_fragment)
            or (
                bonded and damage[i] < 1.0
                and not body_rigid and not neighbour_rigid
            )
        )
        if spring_active or (body_rigid and neighbour_rigid):
            continue
        normal = delta / dist
        closing = wp.dot(v[j] - v[i], normal)
        contact_force -= normal * deformable_contact_magnitude(
            contact_distance - dist, closing, 3.0e6, 9000.0
        )
    ai = acceleration[i] + contact_force / wp.max(mass[i], 1.0)
    a_len = wp.length(ai)
    if a_len > 6000.0:
        ai *= 6000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def finalize_deformable_acceleration(
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    acceleration: wp.array(dtype=wp.vec3),
    maximum_acceleration: float,
):
    """Apply the legacy combined spring/contact cap to every dynamic solid."""
    i = wp.tid()
    if kind[i] == 0 or fixed[i] != 0:
        return
    value = acceleration[i]
    magnitude = wp.length(value)
    if magnitude > maximum_acceleration:
        acceleration[i] = value * (maximum_acceleration / magnitude)


@wp.kernel
def compute_clustered_solid_forces(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    building_damage_integral: wp.array(dtype=float),
    building_structural_volume: wp.array(dtype=float),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fragment_support: wp.array(dtype=float),
    impact_impulse: wp.array(dtype=float),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    max_support: float,
    dt: float,
    internal_stiffness_multiplier: float,
    damage_rate: float,
    propagation_threshold: float,
    max_damage_per_substep: float,
    fracture_reference_radius: float,
    collapse_gravity_damage_onset: float,
    collapse_gravity_damage_full: float,
    facade_support_loss_minimum_elevation: float,
    facade_support_loss_collapse_threshold: float,
    facade_support_loss_damage_rate: float,
    facade_unsupported_damage_rate: float,
    structural_unsupported_damage_rate: float,
    elastic_force_cap_multiplier: float,
    compression_force_cap_multiplier: float,
):
    """Break joints between architectural chunks, never atomize a chunk."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] == 0:
        return
    if fixed[i] != 0:
        acceleration[i] = wp.vec3(0.0)
        return

    xi = x[i]
    ri = rest_x[i]
    fid = fragment_id[i]
    body_rigid = fid >= 0 and rigid_state[fid] != 0
    bid = building_id[i]
    building_collapse = float(0.0)
    if fid >= 0:
        # The sparse architectural graph follows intact bonds from this
        # fragment to a fixed foundation fragment.  It supersedes the former
        # building-wide damage percentage, which could make an intact upper
        # storey fall merely because an unrelated facade was damaged.
        building_collapse = 1.0 - wp.clamp(fragment_support[fid], 0.0, 1.0)
    elif bid >= 0:
        building_collapse = collapse_gravity_fraction(
            building_damage_integral[bid],
            building_structural_volume[bid],
            collapse_gravity_damage_onset,
            collapse_gravity_damage_full,
        )
    local_damage = damage[i]
    local_damage += dt * facade_support_loss_rate(
        structural_class[i], ri[1], building_collapse,
        facade_support_loss_minimum_elevation,
        facade_support_loss_collapse_threshold,
        facade_support_loss_damage_rate,
    )
    # The rest lattice is a statically preloaded building. Waking it for fluid
    # coupling must not suddenly apply an extra 1 g to every storey. Supported
    # fragments keep the authored prestress compensation; only the fraction
    # that has genuinely lost its path to the foundation receives dynamic
    # gravity. Free non-building debris still carries its full self-weight.
    gravity_fraction = preloaded_structure_gravity_fraction(
        bid, building_collapse, body_rigid
    )
    force = solid_force[i]
    impact_drive = material_impact_damage_drive(structural_class[i], impact_impulse[i])
    facade_particle = structural_class[i] == STRUCT_WALL or structural_class[i] == STRUCT_GLASS
    has_local_support = int(0)
    query = wp.hash_grid_query(grid, xi, max_support)

    for j in query:
        if j == i or kind[j] == 0:
            continue
        delta = x[j] - xi
        dist = wp.length(delta)
        if dist <= 1.0e-5:
            continue
        rest_delta = rest_x[j] - ri
        rest_dist = wp.length(rest_delta)
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        same_structure = building_id[i] == building_id[j]
        bonded = same_structure and rest_dist < bond_range
        same_fragment = fid >= 0 and fid == fragment_id[j]
        neighbour_fid = fragment_id[j]
        neighbour_rigid = neighbour_fid >= 0 and rigid_state[neighbour_fid] != 0

        # A facade particle is supported either by an intact neighbour below
        # it or by a nearby intact diaphragm/frame member.  This is evaluated
        # in the deformed configuration: once the lower wall has moved away,
        # lateral springs can no longer suspend the upper panel indefinitely.
        neighbour_has_foundation_path = (
            neighbour_fid < 0 or fragment_support[neighbour_fid] > 0.5
        )
        if (
            facade_particle and bonded and damage[j] < 0.90
            and neighbour_has_foundation_path
        ):
            horizontal_rest2 = rest_delta[0] * rest_delta[0] + rest_delta[2] * rest_delta[2]
            below = (
                rest_delta[1] < -0.25 * bond_range
                and horizontal_rest2 < 0.56 * bond_range * bond_range
            )
            neighbour_role = structural_class[j]
            frame_member = (
                neighbour_role == STRUCT_SLAB or neighbour_role == STRUCT_BEAM
                or neighbour_role == STRUCT_COLUMN or neighbour_role == STRUCT_CORE
            )
            diaphragm = frame_member and wp.abs(rest_delta[1]) < 0.55 * bond_range
            current_attachment_intact = dist < rest_dist * 1.55
            if (below or diaphragm) and current_attachment_intact:
                has_local_support = 1

        if body_rigid and same_fragment:
            # A rigid cluster is projected from one body transform, so neither
            # springs nor self-contact are needed between its sample particles.
            continue
        if bonded and same_fragment:
            # An architectural chunk may deform, but it cannot dissolve into
            # individual lattice particles.  Clamp recoverable strain at the
            # role/material yield envelope: deformation beyond it represents
            # crushing and microcracking, not energy stored in a giant spring.
            strain = (dist - rest_dist) / wp.max(rest_dist, 1.0e-4)
            yield_strain = wp.min(
                material_failure_strain(material[i]),
                material_failure_strain(material[j]),
            )
            yield_strain *= wp.min(
                structural_failure_strain_multiplier(structural_class[i]),
                structural_failure_strain_multiplier(structural_class[j]),
            )
            yield_strain *= elastic_force_cap_multiplier
            transmitted_strain = wp.clamp(
                strain,
                -yield_strain * compression_force_cap_multiplier,
                yield_strain,
            )
            stiffness = wp.min(material_stiffness(material[i]), material_stiffness(material[j]))
            damping = 75000.0 * wp.dot(v[j] - v[i], delta / dist)
            force += (
                stiffness * internal_stiffness_multiplier * transmitted_strain + damping
            ) * (delta / dist) * radius[i] * radius[i]
        elif bonded and local_damage < 1.0 and not body_rigid and not neighbour_rigid:
            # Only joints between chunks fracture. Direct water loading starts
            # a crack; propagation into a dry region requires both a mature
            # neighbouring crack and substantially larger strain.
            strain = (dist - rest_dist) / wp.max(rest_dist, 1.0e-4)
            limit = wp.min(material_failure_strain(material[i]), material_failure_strain(material[j]))
            limit *= wp.min(
                structural_failure_strain_multiplier(structural_class[i]),
                structural_failure_strain_multiplier(structural_class[j]),
            )
            # A smaller particle represents a shorter physical bond.  Without
            # this correction every LOD split changes the energy required to
            # create a square metre of crack, making refined walls artificially
            # weak.  For a spring lattice G ~ E * strain^2 * h, therefore the
            # failure strain scales with sqrt(h_ref / h).
            bond_radius = wp.max(radius[i], radius[j])
            resolution_scale = wp.sqrt(fracture_reference_radius / wp.max(bond_radius, 1.0e-5))
            resolution_scale = wp.clamp(resolution_scale, 0.65, 2.5)
            limit *= resolution_scale
            abs_strain = wp.abs(strain)
            propagated_crack = damage[j] > propagation_threshold and abs_strain > limit * 2.5
            crack_drive = impact_drive
            if propagated_crack:
                crack_drive = wp.max(
                    crack_drive,
                    wp.clamp(
                        (damage[j] - propagation_threshold)
                        / wp.max(1.0 - propagation_threshold, 1.0e-5),
                        0.0,
                        1.0,
                    ),
                )
            if abs_strain > limit and crack_drive > 0.0:
                normalized = (abs_strain - limit) / wp.max(limit, 1.0e-4)
                role_rate = wp.max(
                    structural_damage_rate_multiplier(structural_class[i]),
                    structural_damage_rate_multiplier(structural_class[j]),
                )
                increment = wp.min(
                    normalized * dt * damage_rate * role_rate,
                    max_damage_per_substep * role_rate,
                ) * crack_drive
                local_damage += increment
            if local_damage < 1.0:
                stiffness = wp.min(material_stiffness(material[i]), material_stiffness(material[j]))
                damping = 50000.0 * wp.dot(v[j] - v[i], delta / dist)
                cohesion = (1.0 - local_damage) * (1.0 - local_damage)
                transmitted_strain = wp.clamp(
                    strain,
                    -limit * elastic_force_cap_multiplier * compression_force_cap_multiplier,
                    limit * elastic_force_cap_multiplier,
                )
                force += cohesion * (
                    stiffness * transmitted_strain + damping
                ) * (delta / dist) * radius[i] * radius[i]
        else:
            # Contact remains active after a joint breaks, so chunks collide
            # as rubble instead of passing through one another.
            # Rigid/rigid pairs are handled once by accumulate_rigid_contacts,
            # which applies equal/opposite forces, Coulomb friction and torque.
            if body_rigid and neighbour_rigid:
                continue
            contact = radius[i] + radius[j]
            if dist < contact:
                penetration = contact - dist
                normal = delta / dist
                closing = wp.dot(v[j] - v[i], normal)
                force -= normal * deformable_contact_magnitude(
                    penetration, closing, 3.0e6, 9000.0
                )

    if (
        facade_particle and ri[1] > facade_support_loss_minimum_elevation
        and has_local_support == 0 and not body_rigid
    ):
        # Gravity begins immediately after loss of the local load path.  Joint
        # damage follows more slowly, allowing a wall-sized chunk to fall and
        # collide instead of disappearing into dust.
        gravity_fraction = 1.0
        local_damage += dt * facade_unsupported_damage_rate
    elif (
        fid >= 0 and fragment_support[fid] < 0.5
        and ri[1] > facade_support_loss_minimum_elevation and not body_rigid
    ):
        # Once a beam/column/core fragment has no capacity-rated route to the
        # foundation, the few residual joints carry the whole unsupported
        # mass. Let those joints progressively fail according to their role;
        # otherwise a single numerical spring can suspend an upper tower.
        local_damage += (
            dt * structural_unsupported_damage_rate
            * structural_damage_rate_multiplier(structural_class[i])
        )
    if body_rigid:
        gravity_fraction = 1.0
    force += wp.vec3(0.0, -9.81 * mass[i] * gravity_fraction, 0.0)
    damage[i] = wp.min(local_damage, 1.0)
    ai = force / wp.max(mass[i], 1.0)
    a_len = wp.length(ai)
    if a_len > 6000.0:
        ai *= 6000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def clear_body_accumulators(
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
):
    body = wp.tid()
    for axis in range(3):
        body_force[body, axis] = 0.0
        body_torque[body, axis] = 0.0


@wp.kernel
def accumulate_rigid_body_loads(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    proxy_enabled: wp.array(dtype=wp.int32),
    acceleration: wp.array(dtype=wp.vec3),
    body_center: wp.array(dtype=wp.vec3),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    x_bound: float,
    z_min: float,
    z_max: float,
    y_max: float,
    boundary_stiffness: float,
    boundary_damping: float,
):
    i = wp.tid()
    fid = fragment_id[i]
    if kind[i] == 0 or fid < 0 or rigid_state[fid] == 0:
        return
    xi = x[i]
    vi = v[i]
    particle_radius = radius[i]
    force = acceleration[i] * mass[i]

    # Penalty contacts are accumulated at the actual sample locations, so bed
    # and domain collisions produce both translation and physically useful
    # torque instead of clamping every particle independently.
    if proxy_enabled[fid] == 0:
        penetration = particle_radius - xi[1]
        if penetration > 0.0:
            force += wp.vec3(
                -vi[0] * boundary_damping * 0.12,
                penetration * boundary_stiffness - wp.min(vi[1], 0.0) * boundary_damping,
                -vi[2] * boundary_damping * 0.12,
            )
        if xi[0] - particle_radius < -x_bound:
            depth = -x_bound - (xi[0] - particle_radius)
            force += wp.vec3(depth * boundary_stiffness - wp.min(vi[0], 0.0) * boundary_damping, 0.0, 0.0)
        if xi[0] + particle_radius > x_bound:
            depth = xi[0] + particle_radius - x_bound
            force += wp.vec3(-depth * boundary_stiffness - wp.max(vi[0], 0.0) * boundary_damping, 0.0, 0.0)
        if xi[2] - particle_radius < z_min:
            depth = z_min - (xi[2] - particle_radius)
            force += wp.vec3(0.0, 0.0, depth * boundary_stiffness - wp.min(vi[2], 0.0) * boundary_damping)
        if xi[2] + particle_radius > z_max:
            depth = xi[2] + particle_radius - z_max
            force += wp.vec3(0.0, 0.0, -depth * boundary_stiffness - wp.max(vi[2], 0.0) * boundary_damping)
        if xi[1] + particle_radius > y_max:
            depth = xi[1] + particle_radius - y_max
            force += wp.vec3(0.0, -depth * boundary_stiffness - wp.max(vi[1], 0.0) * boundary_damping, 0.0)

    torque = wp.cross(xi - body_center[fid], force)
    for axis in range(3):
        wp.atomic_add(body_force, fid, axis, force[axis])
        wp.atomic_add(body_torque, fid, axis, torque[axis])


@wp.func
def contact_friction(material: int) -> float:
    if material == 2:  # glass
        return 0.32
    if material == 3:  # reinforcement steel
        return 0.48
    return 0.64  # concrete / masonry


@wp.func
def contact_stiffness(material: int) -> float:
    if material == 2:
        return 2.2e6
    if material == 3:
        return 6.0e6
    return 4.0e6


@wp.func
def proxy_axis(orientation: wp.quat, axis: int) -> wp.vec3:
    if axis == 0:
        return wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
    if axis == 1:
        return wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
    return wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))


@wp.func
def proxy_projection_radius(
    axis: wp.vec3,
    orientation: wp.quat,
    half_extent: wp.vec3,
) -> float:
    return (
        wp.abs(wp.dot(axis, proxy_axis(orientation, 0))) * half_extent[0]
        + wp.abs(wp.dot(axis, proxy_axis(orientation, 1))) * half_extent[1]
        + wp.abs(wp.dot(axis, proxy_axis(orientation, 2))) * half_extent[2]
    )


@wp.func
def proxy_support_point(
    center: wp.vec3,
    orientation: wp.quat,
    half_extent: wp.vec3,
    direction: wp.vec3,
) -> wp.vec3:
    result = center
    for axis in range(3):
        basis = proxy_axis(orientation, axis)
        sign = -1.0
        if wp.dot(direction, basis) >= 0.0:
            sign = 1.0
        result += basis * (sign * half_extent[axis])
    return result


@wp.func
def proxy_sat_contact(
    center_a: wp.vec3,
    orientation_a: wp.quat,
    extent_a: wp.vec3,
    center_b: wp.vec3,
    orientation_b: wp.quat,
    extent_b: wp.vec3,
) -> wp.vec4:
    """Minimum-translation axis and penetration for two convex OBBs."""
    delta = center_b - center_a
    minimum_overlap = 1.0e9
    minimum_axis = wp.vec3(1.0, 0.0, 0.0)
    for source in range(2):
        for index in range(3):
            axis = proxy_axis(orientation_a, index)
            if source == 1:
                axis = proxy_axis(orientation_b, index)
            ra = proxy_projection_radius(axis, orientation_a, extent_a)
            rb = proxy_projection_radius(axis, orientation_b, extent_b)
            overlap = ra + rb - wp.abs(wp.dot(delta, axis))
            if overlap <= 0.0:
                return wp.vec4(0.0, 0.0, 0.0, 0.0)
            if overlap < minimum_overlap:
                if wp.dot(delta, axis) < 0.0:
                    axis = -axis
                minimum_overlap = overlap
                minimum_axis = axis
    for axis_a in range(3):
        for axis_b in range(3):
            axis = wp.cross(
                proxy_axis(orientation_a, axis_a), proxy_axis(orientation_b, axis_b)
            )
            axis_length = wp.length(axis)
            if axis_length > 1.0e-5:
                axis /= axis_length
                ra = proxy_projection_radius(axis, orientation_a, extent_a)
                rb = proxy_projection_radius(axis, orientation_b, extent_b)
                overlap = ra + rb - wp.abs(wp.dot(delta, axis))
                if overlap <= 0.0:
                    return wp.vec4(0.0, 0.0, 0.0, 0.0)
                if overlap < minimum_overlap:
                    if wp.dot(delta, axis) < 0.0:
                        axis = -axis
                    minimum_overlap = overlap
                    minimum_axis = axis
    return wp.vec4(minimum_axis[0], minimum_axis[1], minimum_axis[2], minimum_overlap)


@wp.kernel
def accumulate_rigid_proxy_contacts(
    pair_left: wp.array(dtype=wp.int32),
    pair_right: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    proxy_enabled: wp.array(dtype=wp.int32),
    proxy_local_center: wp.array(dtype=wp.vec3),
    proxy_half_extent: wp.array(dtype=wp.vec3),
    proxy_material: wp.array(dtype=wp.int32),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    contact_acceleration_peak: wp.array(dtype=float),
    normal_damping: float,
    normal_damping_ratio: float,
    tangential_damping: float,
    maximum_penetration: float,
    maximum_contact_acceleration: float,
    minimum_reactivation_closing_speed: float,
):
    pair = wp.tid()
    left = pair_left[pair]
    right = pair_right[pair]
    if (
        rigid_state[left] == 0 or rigid_state[right] == 0
        or proxy_enabled[left] == 0 or proxy_enabled[right] == 0
    ):
        return
    orientation_left = body_orientation[left]
    orientation_right = body_orientation[right]
    center_left = body_center[left] + wp.quat_rotate(
        orientation_left, proxy_local_center[left]
    )
    center_right = body_center[right] + wp.quat_rotate(
        orientation_right, proxy_local_center[right]
    )
    contact = proxy_sat_contact(
        center_left, orientation_left, proxy_half_extent[left],
        center_right, orientation_right, proxy_half_extent[right],
    )
    if contact[3] <= 0.0:
        return
    normal = wp.vec3(contact[0], contact[1], contact[2])
    point_left = proxy_support_point(
        center_left, orientation_left, proxy_half_extent[left], normal
    )
    point_right = proxy_support_point(
        center_right, orientation_right, proxy_half_extent[right], -normal
    )
    point = (point_left + point_right) * 0.5
    arm_left = point - body_center[left]
    arm_right = point - body_center[right]
    velocity_left = body_linear_velocity[left] + wp.cross(body_angular_velocity[left], arm_left)
    velocity_right = body_linear_velocity[right] + wp.cross(body_angular_velocity[right], arm_right)
    relative = velocity_right - velocity_left
    normal_speed = wp.dot(relative, normal)
    stiffness = wp.min(
        contact_stiffness(proxy_material[left]), contact_stiffness(proxy_material[right])
    )
    penetration = wp.min(contact[3], maximum_penetration)
    # A fixed damping coefficient is far too small for facade and slab bodies
    # weighing tens of tonnes.  It made the penalty spring almost perfectly
    # elastic, so an overlapping OBB could throw a complete building section
    # back into the air.  Scale the damper from the pair's effective mass and
    # expose the old coefficient only as a numerical floor.
    mass_left = wp.max(body_mass[left], 1.0)
    mass_right = wp.max(body_mass[right], 1.0)
    effective_mass = (mass_left * mass_right) / (mass_left + mass_right)
    critical_damping = 2.0 * wp.sqrt(stiffness * effective_mass)
    damping = wp.max(normal_damping, normal_damping_ratio * critical_damping)
    closing_speed = wp.max(-normal_speed, 0.0)
    impact_magnitude = damping * closing_speed
    reactivation_magnitude = impact_magnitude
    if closing_speed < minimum_reactivation_closing_speed:
        reactivation_magnitude = 0.0
    normal_magnitude = stiffness * penetration + impact_magnitude
    # Newly fitted convex proxies can overlap through empty regions of sparse
    # facade frames.  Bound the relative acceleration so that resolving that
    # numerical overlap cannot cause an instantaneous trajectory reversal.
    if maximum_contact_acceleration > 0.0:
        normal_magnitude = wp.min(
            normal_magnitude, maximum_contact_acceleration * effective_mass
        )
        impact_magnitude = wp.min(
            impact_magnitude, maximum_contact_acceleration * effective_mass
        )
        reactivation_magnitude = wp.min(
            reactivation_magnitude, maximum_contact_acceleration * effective_mass
        )
    normal_magnitude = wp.max(normal_magnitude, 0.0)
    tangent_velocity = relative - normal * normal_speed
    tangent_speed = wp.length(tangent_velocity)
    friction_force = wp.vec3(0.0)
    if tangent_speed > 1.0e-5:
        friction = wp.min(
            contact_friction(proxy_material[left]), contact_friction(proxy_material[right])
        )
        friction_magnitude = wp.min(
            friction * normal_magnitude, tangential_damping * tangent_speed
        )
        friction_force = tangent_velocity * (friction_magnitude / tangent_speed)
    force_left = -normal * normal_magnitude + friction_force
    force_right = -force_left
    torque_left = wp.cross(arm_left, force_left)
    torque_right = wp.cross(arm_right, force_right)
    for axis in range(3):
        wp.atomic_add(body_force, left, axis, force_left[axis])
        wp.atomic_add(body_force, right, axis, force_right[axis])
        wp.atomic_add(body_torque, left, axis, torque_left[axis])
        wp.atomic_add(body_torque, right, axis, torque_right[axis])
    # Use only the velocity-dependent impact term for reactivation. Persistent
    # overlap in a settled rubble pile carries spring support force but is not
    # a new collision and must not repeatedly dissolve quiet rigid bodies.
    # Relative acceleration is independent of which body is lighter and gives
    # the reactivation gate a stable physical meaning across fragment sizes.
    relative_contact_acceleration = reactivation_magnitude / effective_mass
    wp.atomic_max(contact_acceleration_peak, left, relative_contact_acceleration)
    wp.atomic_max(contact_acceleration_peak, right, relative_contact_acceleration)


@wp.kernel
def update_rigid_proxy_bounds(
    rigid_state: wp.array(dtype=wp.int32),
    proxy_enabled: wp.array(dtype=wp.int32),
    proxy_local_center: wp.array(dtype=wp.vec3),
    proxy_half_extent: wp.array(dtype=wp.vec3),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    bounds_lower: wp.array(dtype=wp.vec3),
    bounds_upper: wp.array(dtype=wp.vec3),
    margin: float,
):
    """Update conservative world-space AABBs for the GPU rigid-proxy BVH."""
    body = wp.tid()
    if rigid_state[body] == 0 or proxy_enabled[body] == 0:
        # Disabled leaves remain in the fixed-capacity BVH but are moved well
        # outside the simulation domain.  This keeps leaf ids identical to
        # fragment ids and avoids rebuilding CPU pair arrays after fractures.
        far = 1.0e6 + float(body) * 2.0
        bounds_lower[body] = wp.vec3(far, far, far)
        bounds_upper[body] = wp.vec3(far + 0.01, far + 0.01, far + 0.01)
        return
    orientation = body_orientation[body]
    center = body_center[body] + wp.quat_rotate(
        orientation, proxy_local_center[body]
    )
    half_extent = proxy_half_extent[body]
    axis_x = proxy_axis(orientation, 0)
    axis_y = proxy_axis(orientation, 1)
    axis_z = proxy_axis(orientation, 2)
    extent = wp.vec3(
        wp.abs(axis_x[0]) * half_extent[0]
        + wp.abs(axis_y[0]) * half_extent[1]
        + wp.abs(axis_z[0]) * half_extent[2],
        wp.abs(axis_x[1]) * half_extent[0]
        + wp.abs(axis_y[1]) * half_extent[1]
        + wp.abs(axis_z[1]) * half_extent[2],
        wp.abs(axis_x[2]) * half_extent[0]
        + wp.abs(axis_y[2]) * half_extent[1]
        + wp.abs(axis_z[2]) * half_extent[2],
    ) + wp.vec3(margin)
    bounds_lower[body] = center - extent
    bounds_upper[body] = center + extent


@wp.kernel
def accumulate_rigid_proxy_contacts_bvh(
    bvh_id: wp.uint64,
    bounds_lower: wp.array(dtype=wp.vec3),
    bounds_upper: wp.array(dtype=wp.vec3),
    rigid_state: wp.array(dtype=wp.int32),
    proxy_enabled: wp.array(dtype=wp.int32),
    proxy_local_center: wp.array(dtype=wp.vec3),
    proxy_half_extent: wp.array(dtype=wp.vec3),
    proxy_material: wp.array(dtype=wp.int32),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    contact_acceleration_peak: wp.array(dtype=float),
    candidate_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    normal_damping: float,
    normal_damping_ratio: float,
    tangential_damping: float,
    maximum_penetration: float,
    maximum_contact_acceleration: float,
    minimum_reactivation_closing_speed: float,
):
    """GPU broadphase and OBB narrowphase without an O(N^2) CPU pair list."""
    left = wp.tid()
    if rigid_state[left] == 0 or proxy_enabled[left] == 0:
        return
    query = wp.bvh_query_aabb(bvh_id, bounds_lower[left], bounds_upper[left])
    right = int(0)
    while wp.bvh_query_next(query, right):
        # Each unordered pair is evaluated exactly once.  The BVH also returns
        # the querying leaf itself.
        if right <= left or rigid_state[right] == 0 or proxy_enabled[right] == 0:
            continue
        wp.atomic_add(candidate_count, 0, 1)
        orientation_left = body_orientation[left]
        orientation_right = body_orientation[right]
        center_left = body_center[left] + wp.quat_rotate(
            orientation_left, proxy_local_center[left]
        )
        center_right = body_center[right] + wp.quat_rotate(
            orientation_right, proxy_local_center[right]
        )
        contact = proxy_sat_contact(
            center_left, orientation_left, proxy_half_extent[left],
            center_right, orientation_right, proxy_half_extent[right],
        )
        if contact[3] <= 0.0:
            continue
        wp.atomic_add(contact_count, 0, 1)
        normal = wp.vec3(contact[0], contact[1], contact[2])
        point_left = proxy_support_point(
            center_left, orientation_left, proxy_half_extent[left], normal
        )
        point_right = proxy_support_point(
            center_right, orientation_right, proxy_half_extent[right], -normal
        )
        point = (point_left + point_right) * 0.5
        arm_left = point - body_center[left]
        arm_right = point - body_center[right]
        velocity_left = body_linear_velocity[left] + wp.cross(
            body_angular_velocity[left], arm_left
        )
        velocity_right = body_linear_velocity[right] + wp.cross(
            body_angular_velocity[right], arm_right
        )
        relative = velocity_right - velocity_left
        normal_speed = wp.dot(relative, normal)
        stiffness = wp.min(
            contact_stiffness(proxy_material[left]),
            contact_stiffness(proxy_material[right]),
        )
        penetration = wp.min(contact[3], maximum_penetration)
        mass_left = wp.max(body_mass[left], 1.0)
        mass_right = wp.max(body_mass[right], 1.0)
        effective_mass = (mass_left * mass_right) / (mass_left + mass_right)
        critical_damping = 2.0 * wp.sqrt(stiffness * effective_mass)
        damping = wp.max(normal_damping, normal_damping_ratio * critical_damping)
        closing_speed = wp.max(-normal_speed, 0.0)
        impact_magnitude = damping * closing_speed
        reactivation_magnitude = impact_magnitude
        if closing_speed < minimum_reactivation_closing_speed:
            reactivation_magnitude = 0.0
        normal_magnitude = stiffness * penetration + impact_magnitude
        if maximum_contact_acceleration > 0.0:
            normal_magnitude = wp.min(
                normal_magnitude, maximum_contact_acceleration * effective_mass
            )
            reactivation_magnitude = wp.min(
                reactivation_magnitude, maximum_contact_acceleration * effective_mass
            )
        normal_magnitude = wp.max(normal_magnitude, 0.0)
        tangent_velocity = relative - normal * normal_speed
        tangent_speed = wp.length(tangent_velocity)
        friction_force = wp.vec3(0.0)
        if tangent_speed > 1.0e-5:
            friction = wp.min(
                contact_friction(proxy_material[left]),
                contact_friction(proxy_material[right]),
            )
            friction_magnitude = wp.min(
                friction * normal_magnitude, tangential_damping * tangent_speed
            )
            friction_force = tangent_velocity * (friction_magnitude / tangent_speed)
        force_left = -normal * normal_magnitude + friction_force
        force_right = -force_left
        torque_left = wp.cross(arm_left, force_left)
        torque_right = wp.cross(arm_right, force_right)
        for axis in range(3):
            wp.atomic_add(body_force, left, axis, force_left[axis])
            wp.atomic_add(body_force, right, axis, force_right[axis])
            wp.atomic_add(body_torque, left, axis, torque_left[axis])
            wp.atomic_add(body_torque, right, axis, torque_right[axis])
        relative_contact_acceleration = reactivation_magnitude / effective_mass
        wp.atomic_max(
            contact_acceleration_peak, left, relative_contact_acceleration
        )
        wp.atomic_max(
            contact_acceleration_peak, right, relative_contact_acceleration
        )


@wp.kernel
def accumulate_rigid_proxy_boundaries(
    rigid_state: wp.array(dtype=wp.int32),
    proxy_enabled: wp.array(dtype=wp.int32),
    proxy_local_center: wp.array(dtype=wp.vec3),
    proxy_half_extent: wp.array(dtype=wp.vec3),
    proxy_material: wp.array(dtype=wp.int32),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    x_bound: float,
    z_min: float,
    z_max: float,
    y_max: float,
    stiffness: float,
    normal_damping: float,
    normal_damping_ratio: float,
    tangential_damping: float,
    maximum_penetration: float,
    maximum_contact_acceleration: float,
):
    body = wp.tid()
    if rigid_state[body] == 0 or proxy_enabled[body] == 0:
        return
    orientation = body_orientation[body]
    center = body_center[body] + wp.quat_rotate(
        orientation, proxy_local_center[body]
    )
    extent = proxy_half_extent[body]
    for plane in range(6):
        normal = wp.vec3(0.0, 1.0, 0.0)
        offset = 0.0
        if plane == 1:
            normal = wp.vec3(1.0, 0.0, 0.0); offset = -x_bound
        elif plane == 2:
            normal = wp.vec3(-1.0, 0.0, 0.0); offset = -x_bound
        elif plane == 3:
            normal = wp.vec3(0.0, 0.0, 1.0); offset = z_min
        elif plane == 4:
            normal = wp.vec3(0.0, 0.0, -1.0); offset = -z_max
        elif plane == 5:
            normal = wp.vec3(0.0, -1.0, 0.0); offset = -y_max
        projection = proxy_projection_radius(normal, orientation, extent)
        penetration = offset - (wp.dot(center, normal) - projection)
        if penetration > 0.0:
            point = proxy_support_point(center, orientation, extent, -normal)
            arm = point - body_center[body]
            point_velocity = body_linear_velocity[body] + wp.cross(
                body_angular_velocity[body], arm
            )
            normal_speed = wp.dot(point_velocity, normal)
            mass = wp.max(body_mass[body], 1.0)
            critical_damping = 2.0 * wp.sqrt(stiffness * mass)
            damping = wp.max(normal_damping, normal_damping_ratio * critical_damping)
            normal_magnitude = (
                stiffness * wp.min(penetration, maximum_penetration)
                + damping * wp.max(-normal_speed, 0.0)
            )
            if maximum_contact_acceleration > 0.0:
                normal_magnitude = wp.min(
                    normal_magnitude, maximum_contact_acceleration * mass
                )
            tangent_velocity = point_velocity - normal * normal_speed
            tangent_speed = wp.length(tangent_velocity)
            friction_force = wp.vec3(0.0)
            if tangent_speed > 1.0e-5:
                friction_magnitude = wp.min(
                    contact_friction(proxy_material[body]) * normal_magnitude,
                    tangential_damping * tangent_speed,
                )
                friction_force = -tangent_velocity * (friction_magnitude / tangent_speed)
            force = normal * normal_magnitude + friction_force
            torque = wp.cross(arm, force)
            for axis in range(3):
                wp.atomic_add(body_force, body, axis, force[axis])
                wp.atomic_add(body_torque, body, axis, torque[axis])


@wp.kernel
def accumulate_rigid_contacts(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    proxy_enabled: wp.array(dtype=wp.int32),
    body_center: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    contact_acceleration_peak: wp.array(dtype=float),
    query_radius: float,
    normal_damping: float,
    tangential_damping: float,
):
    """Sample contacts for rigid bodies which have no convex proxy.

    Proxy/proxy pairs are handled once by the OBB narrowphase.  A mixed pair
    is owned by the non-proxy body, so hundreds of thousands of proxy samples
    no longer perform redundant hash-grid queries merely to reject each other.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    fid = fragment_id[i]
    if (
        kind[i] == 0 or fid < 0 or rigid_state[fid] == 0
        or proxy_enabled[fid] != 0
    ):
        return
    xi = x[i]
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        other = fragment_id[j]
        if kind[j] == 0 or other < 0 or other == fid or rigid_state[other] == 0:
            continue
        # Non-proxy/non-proxy contacts still use particle-id ordering.  Mixed
        # contacts are evaluated only here, from the non-proxy side, so they
        # remain unique regardless of the two samples' global ids.
        if proxy_enabled[other] == 0 and j <= i:
            continue
        delta = x[j] - xi
        distance = wp.length(delta)
        contact_distance = radius[i] + radius[j]
        if distance <= 1.0e-6 or distance >= contact_distance:
            continue
        normal = delta / distance
        relative = v[j] - v[i]
        normal_speed = wp.dot(relative, normal)
        stiffness = wp.min(contact_stiffness(material[i]), contact_stiffness(material[j]))
        normal_magnitude = stiffness * (contact_distance - distance)
        normal_magnitude += normal_damping * wp.max(-normal_speed, 0.0)
        normal_magnitude = wp.max(normal_magnitude, 0.0)
        tangent_velocity = relative - normal * normal_speed
        tangent_speed = wp.length(tangent_velocity)
        friction_force = wp.vec3(0.0)
        if tangent_speed > 1.0e-5:
            friction = wp.min(contact_friction(material[i]), contact_friction(material[j]))
            friction_magnitude = wp.min(friction * normal_magnitude, tangential_damping * tangent_speed)
            friction_force = tangent_velocity * (friction_magnitude / tangent_speed)
        force_on_i = -normal * normal_magnitude + friction_force
        force_on_j = -force_on_i
        torque_i = wp.cross(xi - body_center[fid], force_on_i)
        torque_j = wp.cross(x[j] - body_center[other], force_on_j)
        for axis in range(3):
            wp.atomic_add(body_force, fid, axis, force_on_i[axis])
            wp.atomic_add(body_force, other, axis, force_on_j[axis])
            wp.atomic_add(body_torque, fid, axis, torque_i[axis])
            wp.atomic_add(body_torque, other, axis, torque_j[axis])
        wp.atomic_max(contact_acceleration_peak, fid, normal_magnitude / wp.max(body_mass[fid], 1.0))
        wp.atomic_max(contact_acceleration_peak, other, normal_magnitude / wp.max(body_mass[other], 1.0))


@wp.kernel
def reactivate_rigid_after_impact(
    rigid_state: wp.array(dtype=wp.int32),
    contact_acceleration_peak: wp.array(dtype=float),
    reactivated_count: wp.array(dtype=wp.int32),
    acceleration_threshold: float,
):
    body = wp.tid()
    # State 2 is a sleeping rigid proxy. It is woken by the dedicated sleep
    # policy, not expanded straight back into the deformable particle graph.
    if rigid_state[body] == 1 and contact_acceleration_peak[body] >= acceleration_threshold:
        rigid_state[body] = 0
        wp.atomic_add(reactivated_count, 0, 1)


@wp.kernel
def update_rigid_sleep_state(
    rigid_state: wp.array(dtype=wp.int32),
    quiet_substeps: wp.array(dtype=wp.int32),
    sample_bottom: wp.array(dtype=float),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    body_half_extent: wp.array(dtype=wp.vec3),
    body_force: wp.array2d(dtype=float),
    body_mass: wp.array(dtype=float),
    contact_acceleration_peak: wp.array(dtype=float),
    transition_counts: wp.array(dtype=wp.int32),
    required_quiet_substeps: int,
    ground_margin: float,
    maximum_linear_speed: float,
    maximum_tip_speed: float,
    wake_contact_acceleration: float,
    wake_external_acceleration: float,
):
    """Sleep grounded quiet proxies; wake without losing their rigid shape."""
    body = wp.tid()
    state = rigid_state[body]
    mass = body_mass[body]
    if state == 0 or mass <= 0.0:
        quiet_substeps[body] = 0
        return

    linear = body_linear_velocity[body]
    angular = body_angular_velocity[body]
    tip_speed = wp.length(angular) * wp.max(wp.length(body_half_extent[body]), 0.25)
    net_acceleration = wp.vec3(
        body_force[body, 0] / mass,
        body_force[body, 1] / mass,
        body_force[body, 2] / mass,
    )
    # Remove nominal gravity when testing for new external loading. A settled
    # penalty contact may leave a small residual, hence a configurable margin.
    external_acceleration = net_acceleration - wp.vec3(0.0, -9.81, 0.0)
    must_wake = (
        contact_acceleration_peak[body] >= wake_contact_acceleration
        or wp.length(external_acceleration) >= wake_external_acceleration
    )
    if state == 2:
        if must_wake:
            rigid_state[body] = 1
            quiet_substeps[body] = 0
            wp.atomic_add(transition_counts, 1, 1)
        else:
            body_linear_velocity[body] = wp.vec3(0.0)
            body_angular_velocity[body] = wp.vec3(0.0)
        return

    grounded = sample_bottom[body] <= ground_margin
    quiet = (
        grounded
        and wp.length(linear) <= maximum_linear_speed
        and tip_speed <= maximum_tip_speed
        and not must_wake
    )
    if quiet:
        quiet_substeps[body] += 1
        if quiet_substeps[body] >= required_quiet_substeps:
            rigid_state[body] = 2
            body_linear_velocity[body] = wp.vec3(0.0)
            body_angular_velocity[body] = wp.vec3(0.0)
            wp.atomic_add(transition_counts, 0, 1)
    else:
        quiet_substeps[body] = 0


@wp.kernel
def integrate_rigid_bodies(
    rigid_state: wp.array(dtype=wp.int32),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_inverse_inertia: wp.array(dtype=wp.mat33),
    body_half_extent: wp.array(dtype=wp.vec3),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    dt: float,
    linear_damping: float,
    angular_damping: float,
    maximum_angular_speed: float,
    maximum_linear_speed: float,
    maximum_upward_speed: float,
    upward_speed_reference_mass: float,
    minimum_mass_upward_speed: float,
    maximum_tip_speed: float,
):
    body = wp.tid()
    if rigid_state[body] != 1 or body_mass[body] <= 0.0:
        return
    force = wp.vec3(body_force[body, 0], body_force[body, 1], body_force[body, 2])
    torque = wp.vec3(body_torque[body, 0], body_torque[body, 1], body_torque[body, 2])
    orientation = body_orientation[body]
    linear_velocity = body_linear_velocity[body] + force * (dt / body_mass[body])
    local_torque = wp.quat_rotate_inv(orientation, torque)
    local_alpha = body_inverse_inertia[body] * local_torque
    angular_velocity = body_angular_velocity[body] + wp.quat_rotate(orientation, local_alpha) * dt
    linear_velocity *= wp.exp(-linear_damping * dt)
    angular_velocity *= wp.exp(-angular_damping * dt)
    upward_limit = maximum_upward_speed
    if upward_limit > 0.0 and upward_speed_reference_mass > 0.0:
        mass_scale = wp.sqrt(
            upward_speed_reference_mass
            / wp.max(body_mass[body], upward_speed_reference_mass)
        )
        upward_limit = wp.max(minimum_mass_upward_speed, upward_limit * mass_scale)
    if upward_limit > 0.0 and linear_velocity[1] > upward_limit:
        linear_velocity = wp.vec3(
            linear_velocity[0], upward_limit, linear_velocity[2]
        )
    linear_speed = wp.length(linear_velocity)
    if maximum_linear_speed > 0.0 and linear_speed > maximum_linear_speed:
        linear_velocity *= maximum_linear_speed / linear_speed
    angular_speed = wp.length(angular_velocity)
    angular_limit = maximum_angular_speed
    if maximum_tip_speed > 0.0:
        body_radius = wp.max(wp.length(body_half_extent[body]), 0.25)
        size_limit = maximum_tip_speed / body_radius
        if angular_limit <= 0.0:
            angular_limit = size_limit
        else:
            angular_limit = wp.min(angular_limit, size_limit)
    if angular_limit > 0.0 and angular_speed > angular_limit:
        angular_velocity *= angular_limit / angular_speed
        angular_speed = angular_limit
    if angular_speed * dt > 1.0e-8:
        delta = wp.quat_from_axis_angle(angular_velocity / angular_speed, angular_speed * dt)
        orientation = wp.normalize(delta * orientation)
    body_center[body] += linear_velocity * dt
    body_orientation[body] = orientation
    body_linear_velocity[body] = linear_velocity
    body_angular_velocity[body] = angular_velocity


@wp.kernel
def clear_rigid_sample_bottom(sample_bottom: wp.array(dtype=float)):
    body = wp.tid()
    sample_bottom[body] = 1.0e9


@wp.kernel
def accumulate_rigid_sample_bottom(
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    rigid_local_position: wp.array(dtype=wp.vec3),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    sample_bottom: wp.array(dtype=float),
):
    """Find the real sample-union floor, not the bottom of one giant OBB."""
    particle = wp.tid()
    body = fragment_id[particle]
    if kind[particle] == 0 or body < 0 or rigid_state[body] == 0:
        return
    world = body_center[body] + wp.quat_rotate(
        body_orientation[body], rigid_local_position[particle]
    )
    wp.atomic_min(sample_bottom, body, world[1] - radius[particle])


@wp.kernel
def project_rigid_samples_above_ground(
    rigid_state: wp.array(dtype=wp.int32),
    sample_bottom: wp.array(dtype=float),
    body_center: wp.array(dtype=wp.vec3),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    tangential_retention: float,
):
    """Non-penetration safety projection after compliant ground contact.

    Penalty forces provide weight, friction and torque. This final projection
    only removes numerical tunnelling, which previously accumulated whenever
    a massive fragment's weight exceeded the capped penalty force.
    """
    body = wp.tid()
    if rigid_state[body] == 0:
        return
    bottom = sample_bottom[body]
    if bottom < 0.0:
        center = body_center[body]
        center = wp.vec3(center[0], center[1] - bottom, center[2])
        linear = body_linear_velocity[body]
        linear = wp.vec3(
            linear[0] * tangential_retention,
            wp.max(linear[1], 0.0),
            linear[2] * tangential_retention,
        )
        body_center[body] = center
        body_linear_velocity[body] = linear
        body_angular_velocity[body] *= tangential_retention


@wp.kernel
def scatter_rigid_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    rigid_local_position: wp.array(dtype=wp.vec3),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    fid = fragment_id[i]
    if kind[i] == 0 or fid < 0 or rigid_state[fid] == 0:
        return
    offset = wp.quat_rotate(body_orientation[fid], rigid_local_position[i])
    x[i] = body_center[fid] + offset
    v[i] = body_linear_velocity[fid] + wp.cross(body_angular_velocity[fid], offset)


@wp.kernel
def mask_rigid_particles_as_fixed(
    kind: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    fid = fragment_id[i]
    if kind[i] != 0 and fid >= 0 and rigid_state[fid] != 0:
        fixed[i] = 1


@wp.kernel
def classify_time_levels(
    radius: wp.array(dtype=float),
    v: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    time_level: wp.array(dtype=wp.int32),
    fine_radius: float,
    active_speed: float,
    active_damage: float,
):
    i = wp.tid()
    if kind[i] != 0 or radius[i] <= fine_radius or damage[i] >= active_damage:
        time_level[i] = 0
    elif wp.length(v[i]) >= active_speed:
        time_level[i] = 1
    else:
        time_level[i] = 2


@wp.kernel
def select_active_time_level(
    time_level: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    tick: int,
    time_active: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if kind[i] != 0:
        time_active[i] = 1
        return
    stride = 1
    if time_level[i] == 1:
        stride = 2
    elif time_level[i] >= 2:
        stride = 4
    time_active[i] = 0
    if (tick + 1) % stride == 0:
        time_active[i] = 1


@wp.kernel
def update_hydraulic_boundary_mask(
    kind: wp.array(dtype=wp.int32),
    base_boundary: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    boundary: wp.array(dtype=wp.int32),
    exposure_damage: float,
):
    i = wp.tid()
    if kind[i] == 0:
        boundary[i] = 0
        return
    fid = fragment_id[i]
    exposed = base_boundary[i] != 0 or damage[i] >= exposure_damage
    if fid >= 0 and rigid_state[fid] != 0:
        exposed = True
    boundary[i] = int(exposed)


@wp.kernel
def recalibrate_density_reference_hydraulic(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    sph_support: wp.array(dtype=float),
    sph_support_squared: wp.array(dtype=float),
    sph_poly6_coefficient: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    hydraulic_boundary: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    rest_density: float,
    max_support: float,
):
    """Preserve current pressure while changing the solid boundary quadrature."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] != 0 or water_phase[i] == 2:
        return
    xi = x[i]
    density_sum = float(0.0)
    query = wp.hash_grid_query(grid, xi, max_support)
    for j in query:
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        if kind[j] == 0 and water_phase[j] == 2:
            continue
        delta = xi - x[j]
        support = 4.0 * wp.max(radius[i], radius[j])
        distance_squared = wp.dot(delta, delta)
        if distance_squared >= max_support * max_support:
            continue
        effective_mass = mass[j]
        if kind[j] != 0:
            effective_mass = rest_density * volume[j]
        density_sum += effective_mass * poly6(distance_squared, support)
    density_sum = wp.max(density_sum, rest_density * 0.15)
    rho_reference[i] = (
        density_sum * rest_density / wp.max(rho[i], rest_density * 0.15)
    )


@wp.kernel
def mark_spatial_fluid_particles(
    grid: wp.uint64,
    kind: wp.array(dtype=wp.int32),
    spatial_particle: wp.array(dtype=wp.int32),
    fluid_flag: wp.array(dtype=wp.int32),
):
    slot = wp.tid()
    particle = wp.hash_grid_point_id(grid, slot)
    spatial_particle[slot] = particle
    fluid_flag[slot] = int(kind[particle] == 0)


@wp.kernel
def scatter_spatial_fluid_particles(
    spatial_particle: wp.array(dtype=wp.int32),
    fluid_flag: wp.array(dtype=wp.int32),
    fluid_offset: wp.array(dtype=wp.int32),
    fluid_particle: wp.array(dtype=wp.int32),
):
    slot = wp.tid()
    if fluid_flag[slot] != 0:
        fluid_particle[fluid_offset[slot]] = spatial_particle[slot]


@wp.kernel
def count_fluid_verlet_neighbors(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    hydraulic_boundary: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    core_radius: float,
    verlet_radius: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    xi = x[i]
    core_squared = core_radius * core_radius
    verlet_squared = verlet_radius * verlet_radius
    count = int(0)
    core_query = wp.hash_grid_query(grid, xi, core_radius)
    for j in core_query:
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        delta = x[j] - xi
        if wp.dot(delta, delta) < core_squared:
            count += 1
    halo_query = wp.hash_grid_query(grid, xi, verlet_radius)
    for j in halo_query:
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        delta = x[j] - xi
        distance_squared = wp.dot(delta, delta)
        if distance_squared >= core_squared and distance_squared < verlet_squared:
            count += 1
    neighbour_count[slot] = count


@wp.kernel
def fill_fluid_verlet_neighbors(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    hydraulic_boundary: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    overflow: wp.array(dtype=wp.int32),
    core_radius: float,
    verlet_radius: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    xi = x[i]
    core_squared = core_radius * core_radius
    verlet_squared = verlet_radius * verlet_radius
    cursor = neighbour_offset[slot]
    core_query = wp.hash_grid_query(grid, xi, core_radius)
    for j in core_query:
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        delta = x[j] - xi
        if wp.dot(delta, delta) < core_squared:
            if cursor < neighbour_capacity:
                neighbour_index[cursor] = j
            else:
                overflow[0] = 1
            cursor += 1
    halo_query = wp.hash_grid_query(grid, xi, verlet_radius)
    for j in halo_query:
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        delta = x[j] - xi
        distance_squared = wp.dot(delta, delta)
        if distance_squared >= core_squared and distance_squared < verlet_squared:
            if cursor < neighbour_capacity:
                neighbour_index[cursor] = j
            else:
                overflow[0] = 1
            cursor += 1


@wp.kernel
def finalize_verlet_rebuild(
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    fluid_count: int,
    capacity: int,
    total_entries: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    if wp.tid() == 0:
        total = int(0)
        if fluid_count > 0:
            last = fluid_count - 1
            total = neighbour_offset[last] + neighbour_count[last]
        total_entries[0] = total
        overflow[0] = int(total > capacity)


@wp.kernel
def compute_density_multirate(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    sph_support_squared: wp.array(dtype=float),
    sph_poly6_coefficient: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    hydraulic_boundary: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    rest_density: float,
    sound_speed: float,
    hydrostatic_depth: float,
    initial_wave_height: float,
    reservoir_z_max: float,
    max_support: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] != 0 or time_active[i] == 0:
        return
    if water_phase[i] == 2:
        rho[i] = rest_density
        rho_reference[i] = rest_density
        return
    xi = x[i]
    rhoi = float(0.0)
    query = wp.hash_grid_query(grid, xi, max_support)
    for j in query:
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        if kind[j] == 0 and water_phase[j] == 2:
            continue
        rij = xi - x[j]
        r2 = wp.dot(rij, rij)
        if r2 >= max_support * max_support:
            continue
        support = 4.0 * wp.max(radius[i], radius[j])
        effective_mass = mass[j]
        if kind[j] != 0:
            effective_mass = rest_density * volume[j]
        rhoi += effective_mass * poly6(r2, support)
    rhoi = wp.max(rhoi, rest_density * 0.15)
    reference = rho_reference[i]
    if reference <= 0.0:
        gamma = 7.0
        stiffness = rest_density * sound_speed * sound_speed / gamma
        crest_dz = (xi[2] - reservoir_z_max + 5.0) / 7.5
        local_surface = hydrostatic_depth + initial_wave_height * wp.exp(-crest_dz * crest_dz)
        water_column = wp.max(local_surface - xi[1], 0.0)
        hydro_pressure = rest_density * 9.81 * water_column
        target_ratio = wp.pow(1.0 + hydro_pressure / stiffness, 1.0 / gamma)
        rho_reference[i] = rhoi / target_ratio
        rho[i] = rest_density * target_ratio
    else:
        rho[i] = wp.max(rhoi * rest_density / reference, rest_density * 0.15)


@wp.kernel
def compute_density_multirate_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    sph_support_squared: wp.array(dtype=float),
    sph_poly6_coefficient: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    sound_speed: float,
    hydrostatic_depth: float,
    initial_wave_height: float,
    reservoir_z_max: float,
    max_support: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if time_active[i] == 0:
        return
    if water_phase[i] == 2:
        rho[i] = rest_density
        rho_reference[i] = rest_density
        return
    xi = x[i]
    rhoi = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if kind[j] == 0 and water_phase[j] == 2:
            continue
        rij = xi - x[j]
        r2 = wp.dot(rij, rij)
        if r2 >= max_support * max_support:
            continue
        support = 4.0 * wp.max(radius[i], radius[j])
        effective_mass = mass[j]
        if kind[j] != 0:
            effective_mass = rest_density * volume[j]
        rhoi += effective_mass * poly6(r2, support)
    rhoi = wp.max(rhoi, rest_density * 0.15)
    reference = rho_reference[i]
    if reference <= 0.0:
        gamma = 7.0
        stiffness = rest_density * sound_speed * sound_speed / gamma
        crest_dz = (xi[2] - reservoir_z_max + 5.0) / 7.5
        local_surface = hydrostatic_depth + initial_wave_height * wp.exp(
            -crest_dz * crest_dz
        )
        water_column = wp.max(local_surface - xi[1], 0.0)
        hydro_pressure = rest_density * 9.81 * water_column
        target_ratio = wp.pow(1.0 + hydro_pressure / stiffness, 1.0 / gamma)
        rho_reference[i] = rhoi / target_ratio
        rho[i] = rest_density * target_ratio
    else:
        rho[i] = wp.max(rhoi * rest_density / reference, rest_density * 0.15)


@wp.kernel
def compute_fluid_pressure_multirate(
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
    inverse_density: wp.array(dtype=float),
    mass_over_density: wp.array(dtype=float),
    pressure_over_density_squared: wp.array(dtype=float),
    rest_density: float,
    sound_speed: float,
    max_density_ratio: float,
):
    """Cache the equation-of-state pressure once per active fluid particle."""
    i = wp.tid()
    if kind[i] != 0 or time_active[i] == 0:
        return
    local_inverse_density = 1.0 / wp.max(rho[i], rest_density * 0.15)
    inverse_density[i] = local_inverse_density
    mass_over_density[i] = mass[i] * local_inverse_density
    if water_phase[i] == 2:
        pressure[i] = 0.0
        pressure_over_density_squared[i] = 0.0
        return
    gamma = 7.0
    stiffness = rest_density * sound_speed * sound_speed / gamma
    ratio = wp.min(rho[i] / rest_density, max_density_ratio)
    local_pressure = wp.max(
        stiffness * (wp.pow(ratio, gamma) - 1.0), -0.02 * stiffness
    )
    pressure[i] = local_pressure
    pressure_over_density_squared[i] = (
        local_pressure * local_inverse_density * local_inverse_density
    )


@wp.kernel
def compute_fluid_forces_multirate(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    sph_support: wp.array(dtype=float),
    sph_support_squared: wp.array(dtype=float),
    sph_poly6_coefficient: wp.array(dtype=float),
    sph_spiky_coefficient: wp.array(dtype=float),
    sph_viscosity_coefficient: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    hydraulic_boundary: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
    inverse_density: wp.array(dtype=float),
    mass_over_density: wp.array(dtype=float),
    pressure_over_density_squared: wp.array(dtype=float),
    time_level: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    deferred_impulse: wp.array2d(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    rest_density: float,
    viscosity: float,
    xsph_strength: float,
    max_support: float,
    base_dt: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] != 0 or time_active[i] == 0:
        return
    level_i = time_level[i]
    stride_i = 1
    if level_i == 1:
        stride_i = 2
    elif level_i >= 2:
        stride_i = 4
    effective_dt = base_dt * float(stride_i)
    xi = x[i]
    vi = v[i]
    if water_phase[i] == 2:
        # Detached drops carry the same particle mass and momentum, but no
        # longer receive bulk pressure/viscosity from the connected phase.
        # They remain collision-active against solid particles and transfer
        # the equal-and-opposite contact force back to the structure.
        ballistic_acceleration = wp.vec3(0.0, -9.81, 0.0)
        ballistic_query = wp.hash_grid_query(grid, xi, max_support)
        for j in ballistic_query:
            if kind[j] == 0 or hydraulic_boundary[j] == 0:
                continue
            delta = xi - x[j]
            distance = wp.length(delta)
            contact_distance = radius[i] + radius[j]
            if distance <= 1.0e-5 or distance >= contact_distance:
                continue
            normal = delta / distance
            penetration = contact_distance - distance
            approach_speed = wp.min(wp.dot(vi - v[j], normal), 0.0)
            contact_acceleration = normal * (2400.0 * penetration - 55.0 * approach_speed)
            ballistic_acceleration += contact_acceleration
            wp.atomic_add(solid_force, j, -contact_acceleration * mass[i] * float(stride_i))
        acceleration[i] = ballistic_acceleration
        return
    pi = pressure[i]
    inverse_rhoi = inverse_density[i]
    pressure_term_i = pressure_over_density_squared[i]
    ai = wp.vec3(0.0, -9.81, 0.0)
    xsph = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, max_support)
    for j in query:
        if j == i:
            continue
        if kind[j] != 0 and hydraulic_boundary[j] == 0:
            continue
        if kind[j] == 0 and water_phase[j] == 2:
            continue
        r = xi - x[j]
        dist = wp.length(r)
        support = 4.0 * wp.max(radius[i], radius[j])
        if dist >= max_support or dist >= support or dist <= 1.0e-5:
            continue
        pressure_term_j = pi / (rest_density * rest_density)
        inverse_rhoj = 1.0 / rest_density
        if kind[j] == 0:
            pressure_term_j = pressure_over_density_squared[j]
            inverse_rhoj = inverse_density[j]
        else:
            rhoj = rest_density
        grad = spiky_grad(r, dist, support)
        neighbour_mass = mass[j]
        if kind[j] != 0:
            neighbour_mass = rest_density * volume[j]
        pair_acc = -neighbour_mass * (pressure_term_i + pressure_term_j) * grad
        pair_acc += (
            viscosity * neighbour_mass * (v[j] - vi)
            * inverse_rhoj * viscosity_laplacian(dist, support) * inverse_rhoi
        )
        ai += pair_acc
        if kind[j] == 0:
            xsph += mass_over_density[j] * (v[j] - vi) * poly6(
                wp.dot(r, r), support
            )
        else:
            # The slow particle represents several base ticks, so its boundary
            # reaction must carry the same integrated impulse.
            wp.atomic_add(solid_force, j, -pair_acc * mass[i] * float(stride_i))
    ai += xsph * (xsph_strength / wp.max(effective_dt, 1.0e-7))
    a_len = wp.length(ai)
    if a_len > 8000.0:
        ai *= 8000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def compute_fluid_forces_multirate_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    sph_support: wp.array(dtype=float),
    sph_support_squared: wp.array(dtype=float),
    sph_poly6_coefficient: wp.array(dtype=float),
    sph_spiky_coefficient: wp.array(dtype=float),
    sph_viscosity_coefficient: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    pressure: wp.array(dtype=float),
    inverse_density: wp.array(dtype=float),
    mass_over_density: wp.array(dtype=float),
    pressure_over_density_squared: wp.array(dtype=float),
    time_level: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    deferred_impulse: wp.array2d(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    viscosity: float,
    xsph_strength: float,
    max_support: float,
    base_dt: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if time_active[i] == 0:
        return
    level_i = time_level[i]
    stride_i = 1
    if level_i == 1:
        stride_i = 2
    elif level_i >= 2:
        stride_i = 4
    effective_dt = base_dt * float(stride_i)
    xi = x[i]
    vi = v[i]
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    if water_phase[i] == 2:
        ballistic_acceleration = wp.vec3(0.0, -9.81, 0.0)
        for edge in range(start, end):
            j = neighbour_index[edge]
            if kind[j] == 0:
                continue
            delta = xi - x[j]
            distance = wp.length(delta)
            contact_distance = radius[i] + radius[j]
            if distance <= 1.0e-5 or distance >= contact_distance:
                continue
            normal = delta / distance
            penetration = contact_distance - distance
            approach_speed = wp.min(wp.dot(vi - v[j], normal), 0.0)
            contact_acceleration = normal * (
                2400.0 * penetration - 55.0 * approach_speed
            )
            ballistic_acceleration += contact_acceleration
            wp.atomic_add(
                solid_force, j,
                -contact_acceleration * mass[i] * float(stride_i),
            )
        acceleration[i] = ballistic_acceleration
        return
    pi = pressure[i]
    inverse_rhoi = inverse_density[i]
    pressure_term_i = pressure_over_density_squared[i]
    ai = wp.vec3(0.0, -9.81, 0.0)
    xsph = wp.vec3(0.0)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        r = xi - x[j]
        dist = wp.length(r)
        support = 4.0 * wp.max(radius[i], radius[j])
        if dist >= max_support or dist >= support or dist <= 1.0e-5:
            continue
        pressure_term_j = pi / (rest_density * rest_density)
        inverse_rhoj = 1.0 / rest_density
        if kind[j] == 0:
            pressure_term_j = pressure_over_density_squared[j]
            inverse_rhoj = inverse_density[j]
        grad = spiky_grad(r, dist, support)
        neighbour_mass = mass[j]
        if kind[j] != 0:
            neighbour_mass = rest_density * volume[j]
        pair_acc = -neighbour_mass * (
            pressure_term_i + pressure_term_j
        ) * grad
        pair_acc += (
            viscosity * neighbour_mass * (v[j] - vi)
            * inverse_rhoj * viscosity_laplacian(dist, support) * inverse_rhoi
        )
        ai += pair_acc
        if kind[j] == 0:
            xsph += mass_over_density[j] * (v[j] - vi) * poly6(
                wp.dot(r, r), support
            )
        else:
            wp.atomic_add(
                solid_force, j, -pair_acc * mass[i] * float(stride_i)
            )
    ai += xsph * (xsph_strength / wp.max(effective_dt, 1.0e-7))
    a_len = wp.length(ai)
    if a_len > 8000.0:
        ai *= 8000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def consume_deferred_fluid_impulse(
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    time_level: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    deferred_impulse: wp.array2d(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    base_dt: float,
):
    i = wp.tid()
    if kind[i] != 0 or time_active[i] == 0:
        return
    stride = 1
    if time_level[i] == 1:
        stride = 2
    elif time_level[i] >= 2:
        stride = 4
    # Reserved for future sparse boundary-flux corrections.  The production
    # path uses synchronization-point impulse matching: a slow particle
    # evaluates all neighbours once and integrates that force over its stride.
    # This avoids millions of contended pair atomics while remaining within
    # the measured global momentum tolerance.
    for axis in range(3):
        deferred_impulse[i, axis] = 0.0


@wp.kernel
def integrate_multirate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    time_level: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    base_dt: float,
    x_bound: float,
    z_min: float,
    z_max: float,
    y_max: float,
    fluid_bed_drag: float,
    maximum_fluid_speed: float,
    maximum_fluid_vertical_speed: float,
    maximum_solid_speed: float,
    maximum_solid_upward_speed: float,
):
    i = wp.tid()
    if fixed[i] != 0 or (kind[i] == 0 and time_active[i] == 0):
        return
    dt = base_dt
    if kind[i] == 0:
        if time_level[i] == 1:
            dt = base_dt * 2.0
        elif time_level[i] >= 2:
            dt = base_dt * 4.0
    vi = v[i] + acceleration[i] * dt
    if kind[i] == 0:
        if maximum_fluid_vertical_speed > 0.0:
            vi = wp.vec3(
                vi[0],
                wp.clamp(vi[1], -maximum_fluid_vertical_speed, maximum_fluid_vertical_speed),
                vi[2],
            )
        speed = wp.length(vi)
        if maximum_fluid_speed > 0.0 and speed > maximum_fluid_speed:
            vi *= maximum_fluid_speed / speed
    if kind[i] != 0:
        vi *= wp.pow(0.9993, dt * 1000.0)
        if maximum_solid_upward_speed > 0.0 and vi[1] > maximum_solid_upward_speed:
            vi = wp.vec3(vi[0], maximum_solid_upward_speed, vi[2])
        solid_speed = wp.length(vi)
        if maximum_solid_speed > 0.0 and solid_speed > maximum_solid_speed:
            vi *= maximum_solid_speed / solid_speed
    xi = x[i] + vi * dt
    restitution = -0.12
    if xi[1] < 0.0:
        xi = wp.vec3(xi[0], 0.0, xi[2])
        if kind[i] == 0:
            tangential = wp.exp(-fluid_bed_drag * dt)
            vi = wp.vec3(vi[0] * tangential, 0.0, vi[2] * tangential)
        else:
            vi = wp.vec3(vi[0] * 0.78, vi[1] * restitution, vi[2] * 0.78)
    if xi[0] < -x_bound:
        xi = wp.vec3(-x_bound, xi[1], xi[2]); vi = wp.vec3(vi[0] * restitution, vi[1], vi[2])
    if xi[0] > x_bound:
        xi = wp.vec3(x_bound, xi[1], xi[2]); vi = wp.vec3(vi[0] * restitution, vi[1], vi[2])
    if xi[2] < z_min:
        xi = wp.vec3(xi[0], xi[1], z_min); vi = wp.vec3(vi[0], vi[1], vi[2] * restitution)
    if xi[2] > z_max:
        xi = wp.vec3(xi[0], xi[1], z_max); vi = wp.vec3(vi[0], vi[1], vi[2] * restitution)
    if xi[1] > y_max:
        xi = wp.vec3(xi[0], y_max, xi[2]); vi = wp.vec3(vi[0], vi[1] * restitution, vi[2])
    x[i] = xi
    v[i] = vi


@wp.kernel
def deform_facade_vertices(
    rest_vertex: wp.array(dtype=wp.vec3),
    anchor: wp.array(dtype=wp.int32),
    panel_mode: wp.array(dtype=wp.int32),
    owner_fragment: wp.array(dtype=wp.int32),
    fragment_support: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    current_vertex: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    panel = i // 4
    if panel_mode[panel] != 0:
        owner = owner_fragment[panel]
        if owner < 0 or fragment_support[owner] > 0.5:
            return
    a = anchor[i]
    current_vertex[i] = rest_vertex[i] + (x[a] - rest_x[a])


@wp.func
def facade_triangle_indices(triangle: int) -> wp.vec3i:
    panel = triangle // 2
    if triangle - panel * 2 == 0:
        return wp.vec3i(panel * 4, panel * 4 + 1, panel * 4 + 2)
    return wp.vec3i(panel * 4, panel * 4 + 2, panel * 4 + 3)


@wp.func
def edge2(a: wp.vec3, b: wp.vec3, px: float, py: float) -> float:
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


@wp.func
def facade_triangle_torn(
    a: wp.vec3, b: wp.vec3, c: wp.vec3,
    rest_a: wp.vec3, rest_b: wp.vec3, rest_c: wp.vec3,
    maximum_stretch: float,
) -> bool:
    ab = wp.length(b - a) / wp.max(wp.length(rest_b - rest_a), 1.0e-5)
    bc = wp.length(c - b) / wp.max(wp.length(rest_c - rest_b), 1.0e-5)
    ca = wp.length(a - c) / wp.max(wp.length(rest_a - rest_c), 1.0e-5)
    return wp.max(ab, wp.max(bc, ca)) > maximum_stretch


@wp.kernel
def raster_facade_depth(
    vertex: wp.array(dtype=wp.vec3),
    rest_vertex: wp.array(dtype=wp.vec3),
    panel_mode: wp.array(dtype=wp.int32),
    owner_fragment: wp.array(dtype=wp.int32),
    fragment_support: wp.array(dtype=float),
    depth: wp.array(dtype=float),
    cam: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
    maximum_stretch: float,
):
    triangle = wp.tid()
    panel = triangle // 2
    if panel_mode[panel] != 0:
        owner = owner_fragment[panel]
        if owner < 0 or fragment_support[owner] > 0.5:
            return
    ids = facade_triangle_indices(triangle)
    world_a = vertex[ids[0]]; world_b = vertex[ids[1]]; world_c = vertex[ids[2]]
    if facade_triangle_torn(
        world_a, world_b, world_c,
        rest_vertex[ids[0]], rest_vertex[ids[1]], rest_vertex[ids[2]], maximum_stretch,
    ):
        return
    a = project_point(world_a, cam, right, up, forward, focal, width, height)
    b = project_point(world_b, cam, right, up, forward, focal, width, height)
    c = project_point(world_c, cam, right, up, forward, focal, width, height)
    if a[2] <= 0.1 or b[2] <= 0.1 or c[2] <= 0.1:
        return
    min_x = wp.clamp(int(wp.floor(wp.min(a[0], wp.min(b[0], c[0])))), 0, width - 1)
    max_x = wp.clamp(int(wp.ceil(wp.max(a[0], wp.max(b[0], c[0])))), 0, width - 1)
    min_y = wp.clamp(int(wp.floor(wp.min(a[1], wp.min(b[1], c[1])))), 0, height - 1)
    max_y = wp.clamp(int(wp.ceil(wp.max(a[1], wp.max(b[1], c[1])))), 0, height - 1)
    area = edge2(a, b, c[0], c[1])
    if wp.abs(area) < 1.0e-6:
        return
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            fx = float(px) + 0.5; fy = float(py) + 0.5
            w0 = edge2(b, c, fx, fy) / area
            w1 = edge2(c, a, fx, fy) / area
            w2 = 1.0 - w0 - w1
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                wp.atomic_min(depth, py * width + px, z)


@wp.func
def facade_material_color(code: int) -> wp.vec3:
    # Legacy panels (1/2) remain readable in old checkpoints and tests.
    if code == 2:
        return wp.vec3(0.10, 0.32, 0.43)
    if code < 10:
        return wp.vec3(0.48, 0.49, 0.47)
    family = code // 10
    palette = code - family * 10
    base = wp.vec3(0.52, 0.55, 0.55)  # cool concrete
    if palette == 1:
        base = wp.vec3(0.60, 0.51, 0.40)  # warm limestone
    elif palette == 2:
        base = wp.vec3(0.27, 0.30, 0.32)  # graphite
    elif palette == 3:
        base = wp.vec3(0.64, 0.57, 0.44)  # sandstone
    elif palette == 4:
        base = wp.vec3(0.52, 0.27, 0.19)  # brick / terracotta
    elif palette == 5:
        base = wp.vec3(0.68, 0.69, 0.66)  # pale precast
    if family == 2:
        base = wp.vec3(0.08, 0.30, 0.43)
        if palette == 1: base = wp.vec3(0.12, 0.34, 0.32)
        elif palette == 2: base = wp.vec3(0.07, 0.16, 0.21)
        elif palette == 3: base = wp.vec3(0.28, 0.30, 0.24)
        elif palette == 4: base = wp.vec3(0.08, 0.27, 0.30)
        elif palette == 5: base = wp.vec3(0.16, 0.38, 0.50)
    elif family == 3:
        base *= 0.68
    elif family == 4:
        # Exposed structural fragment hull: darker concrete/steel aggregate,
        # while retaining a trace of the original building palette.
        base = base * 0.48 + wp.vec3(0.18, 0.19, 0.18)
    elif family == 5:  # painted vehicle body
        base = wp.vec3(0.12, 0.25, 0.48)
        if palette == 1: base = wp.vec3(0.62, 0.10, 0.07)
        elif palette == 2: base = wp.vec3(0.72, 0.70, 0.62)
        elif palette == 3: base = wp.vec3(0.08, 0.09, 0.10)
        elif palette == 4: base = wp.vec3(0.12, 0.45, 0.28)
        elif palette == 5: base = wp.vec3(0.67, 0.38, 0.08)
        elif palette == 8: base = wp.vec3(0.025, 0.028, 0.030)
    elif family == 6:  # wet bark
        base = wp.vec3(0.22, 0.115, 0.055)
    elif family == 7:  # foliage
        base = wp.vec3(0.075, 0.25, 0.10)
        if palette == 1: base = wp.vec3(0.11, 0.31, 0.13)
        elif palette == 2: base = wp.vec3(0.06, 0.19, 0.09)
    elif family == 9:  # terrain / asphalt / pavement
        base = wp.vec3(0.18, 0.20, 0.20)
        if palette == 1: base = wp.vec3(0.075, 0.083, 0.085)
        elif palette == 2: base = wp.vec3(0.32, 0.33, 0.31)
    return base


@wp.func
def facade_hash01(panel: int, salt: int) -> float:
    value = wp.sin(float(panel * 37 + salt * 101) * 12.9898) * 43758.5453
    return value - wp.floor(value)


@wp.func
def facade_segment_distance(uv: wp.vec2, start: wp.vec2, end: wp.vec2) -> float:
    segment = end - start
    amount = wp.clamp(wp.dot(uv - start, segment) / wp.max(wp.dot(segment, segment), 1.0e-8), 0.0, 1.0)
    return wp.length(uv - (start + segment * amount))


@wp.func
def facade_crack_mask(panel: int, uv: wp.vec2, damage: float, material: int) -> float:
    family = material // 10
    threshold = 0.08
    if family == 2:  # brittle facade glass
        threshold = 0.012
    elif family == 3:  # floor/roof plates
        threshold = 0.12
    if damage <= threshold:
        return 0.0
    intensity = wp.clamp((damage - threshold) / wp.max(0.62 - threshold, 1.0e-5), 0.0, 1.0)
    seed = wp.vec2(
        0.34 + 0.32 * facade_hash01(panel, 1),
        0.34 + 0.32 * facade_hash01(panel, 2),
    )
    phase = facade_hash01(panel, 3) * 6.2831853
    minimum_distance = 10.0
    for ray in range(4):
        # Two hairline rays appear first. Additional rays and branches are
        # admitted gradually, which avoids a repeated four-point star on every
        # damaged panel while keeping the pattern deterministic per panel.
        ray_visible = ray < 2
        if ray == 2 and intensity > 0.16 + 0.22 * facade_hash01(panel, 31):
            ray_visible = True
        if ray == 3 and intensity > 0.42 + 0.30 * facade_hash01(panel, 37):
            ray_visible = True
        if ray_visible:
            angle = phase + float(ray) * 1.5707963 + (facade_hash01(panel, ray + 7) - 0.5) * 0.72
            direction = wp.vec2(wp.cos(angle), wp.sin(angle))
            endpoint = seed + direction * (0.72 + 0.50 * facade_hash01(panel, ray + 41))
            minimum_distance = wp.min(minimum_distance, facade_segment_distance(uv, seed, endpoint))
            if intensity > 0.13 + 0.32 * facade_hash01(panel, ray + 47):
                branch_start = seed + direction * (0.28 + 0.24 * facade_hash01(panel, ray + 13))
                branch_angle = angle + (facade_hash01(panel, ray + 19) - 0.5) * 1.45
                branch_end = branch_start + wp.vec2(wp.cos(branch_angle), wp.sin(branch_angle)) * 0.38
                minimum_distance = wp.min(
                    minimum_distance, facade_segment_distance(uv, branch_start, branch_end)
                )
    width = 0.003 + 0.006 * intensity
    line = wp.clamp((width * 2.0 - minimum_distance) / wp.max(width, 1.0e-5), 0.0, 1.0)
    if damage > 0.55:
        chip_radius = 0.025 + 0.055 * wp.clamp((damage - 0.55) / 0.45, 0.0, 1.0)
        chip = wp.clamp((chip_radius - wp.length(uv - seed)) / wp.max(chip_radius * 0.35, 1.0e-5), 0.0, 1.0)
        line = wp.max(line, chip)
    return line * (0.30 + 0.70 * intensity)


@wp.kernel
def raster_facade_color(
    vertex: wp.array(dtype=wp.vec3),
    rest_vertex: wp.array(dtype=wp.vec3),
    anchor: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    panel_mode: wp.array(dtype=wp.int32),
    owner_fragment: wp.array(dtype=wp.int32),
    fragment_support: wp.array(dtype=float),
    particle_damage: wp.array(dtype=float),
    fragment_fracture_energy: wp.array(dtype=float),
    triangle_order: wp.array(dtype=wp.int32),
    depth: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3),
    cam: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
    maximum_stretch: float,
    crack_strength: float,
    architectural_overlay_tolerance: float,
):
    triangle = triangle_order[wp.tid()]
    panel = triangle // 2
    if panel_mode[panel] != 0:
        owner = owner_fragment[panel]
        if owner < 0 or fragment_support[owner] > 0.5:
            return
    ids = facade_triangle_indices(triangle)
    world_a = vertex[ids[0]]; world_b = vertex[ids[1]]; world_c = vertex[ids[2]]
    if facade_triangle_torn(
        world_a, world_b, world_c,
        rest_vertex[ids[0]], rest_vertex[ids[1]], rest_vertex[ids[2]], maximum_stretch,
    ):
        return
    a = project_point(world_a, cam, right, up, forward, focal, width, height)
    b = project_point(world_b, cam, right, up, forward, focal, width, height)
    c = project_point(world_c, cam, right, up, forward, focal, width, height)
    if a[2] <= 0.1 or b[2] <= 0.1 or c[2] <= 0.1:
        return
    min_x = wp.clamp(int(wp.floor(wp.min(a[0], wp.min(b[0], c[0])))), 0, width - 1)
    max_x = wp.clamp(int(wp.ceil(wp.max(a[0], wp.max(b[0], c[0])))), 0, width - 1)
    min_y = wp.clamp(int(wp.floor(wp.min(a[1], wp.min(b[1], c[1])))), 0, height - 1)
    max_y = wp.clamp(int(wp.ceil(wp.max(a[1], wp.max(b[1], c[1])))), 0, height - 1)
    area = edge2(a, b, c[0], c[1])
    if wp.abs(area) < 1.0e-6:
        return

    base = facade_material_color(material[panel])
    normal = wp.normalize(wp.cross(world_b - world_a, world_c - world_a))
    view_direction = wp.normalize(cam - (world_a + world_b + world_c) / 3.0)
    # Authored panels are two-sided, but lighting must not be. Orient their
    # normal toward the camera once, then retain a real lit and shadowed side.
    if wp.dot(normal, view_direction) < 0.0:
        normal = -normal
    sun_direction = wp.normalize(wp.vec3(-0.38, 0.82, -0.35))
    diffuse = wp.clamp(wp.dot(normal, sun_direction), 0.0, 1.0)
    light = 0.09 + 0.91 * diffuse
    family = material[panel] // 10
    if family == 2 or family == 5:
        half_vector = wp.normalize(view_direction + sun_direction)
        specular = wp.pow(wp.clamp(wp.dot(normal, half_vector), 0.0, 1.0), 20.0)
        base += wp.vec3(0.42, 0.46, 0.47) * specular * 0.58
    anchor_base = panel * 4
    damage_0 = particle_damage[anchor[anchor_base]]
    damage_1 = particle_damage[anchor[anchor_base + 1]]
    damage_2 = particle_damage[anchor[anchor_base + 2]]
    damage_3 = particle_damage[anchor[anchor_base + 3]]
    panel_damage = 0.25 * (damage_0 + damage_1 + damage_2 + damage_3)
    crack_damage = wp.max(damage_0, wp.max(damage_1, wp.max(damage_2, damage_3)))
    owner = owner_fragment[panel]
    if owner >= 0:
        # Boundary energy is irreversible and comes from the same sampled
        # inter-fragment joints that determine structural support. Anchor
        # damage remains useful for highly local glass and impact cracks.
        crack_damage = wp.max(crack_damage, fragment_fracture_energy[owner])
    # Keep the building palette on detached panels.  Damage is expressed as
    # loss of brightness; exposed particle materials supply concrete/steel
    # contrast after the facade skin actually tears.
    base *= 1.0 - 0.32 * wp.clamp(panel_damage, 0.0, 1.0)
    base *= light
    rest_a = rest_vertex[ids[0]]; rest_b = rest_vertex[ids[1]]; rest_c = rest_vertex[ids[2]]
    panel_vertex = panel * 4
    rest_origin = rest_vertex[panel_vertex]
    rest_u = rest_vertex[panel_vertex + 3] - rest_origin
    rest_v = rest_vertex[panel_vertex + 1] - rest_origin
    rest_u_length2 = wp.max(wp.dot(rest_u, rest_u), 1.0e-8)
    rest_v_length2 = wp.max(wp.dot(rest_v, rest_v), 1.0e-8)
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            fx = float(px) + 0.5; fy = float(py) + 0.5
            w0 = edge2(b, c, fx, fy) / area
            w1 = edge2(c, a, fx, fy) / area
            w2 = 1.0 - w0 - w1
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                index = py * width + px
                depth_tolerance = 0.02
                if panel_mode[panel] == 0:
                    # The conservative collision/cut-cell surface may sit up
                    # to one particle radius in front of the authored facade.
                    # Draw the original architectural panel last within that
                    # narrow shell so windows and palettes survive rigid LOD.
                    depth_tolerance = architectural_overlay_tolerance
                if z <= depth[index] + depth_tolerance:
                    pixel_color = base
                    if panel_mode[panel] == 0 and crack_strength > 0.0:
                        rest_point = rest_a * w0 + rest_b * w1 + rest_c * w2
                        relative = rest_point - rest_origin
                        uv = wp.vec2(
                            wp.dot(relative, rest_u) / rest_u_length2,
                            wp.dot(relative, rest_v) / rest_v_length2,
                        )
                        crack = facade_crack_mask(
                            panel, uv, crack_damage, material[panel]
                        ) * crack_strength
                        if material[panel] // 10 == 2:
                            crack_color = wp.vec3(0.58, 0.78, 0.86) * light
                            pixel_color = pixel_color * (1.0 - crack) + crack_color * crack
                        else:
                            pixel_color *= 1.0 - 0.82 * crack
                    color[index] = pixel_color


@wp.func
def planar_child_offset(normal_axis: int, corner: int, amount: float) -> wp.vec3:
    s0 = -1.0
    s1 = -1.0
    if corner == 1 or corner == 3:
        s0 = 1.0
    if corner == 2 or corner == 3:
        s1 = 1.0
    if normal_axis == 0:
        return wp.vec3(0.0, s0 * amount, s1 * amount)
    if normal_axis == 1:
        return wp.vec3(s0 * amount, 0.0, s1 * amount)
    return wp.vec3(s0 * amount, s1 * amount, 0.0)


@wp.func
def linear_child_offset(long_axis: int, child: int, amount: float) -> wp.vec3:
    sign = -1.0
    if child == 1:
        sign = 1.0
    if long_axis == 0:
        return wp.vec3(sign * amount, 0.0, 0.0)
    if long_axis == 1:
        return wp.vec3(0.0, sign * amount, 0.0)
    return wp.vec3(0.0, 0.0, sign * amount)


@wp.func
def volume_child_offset(child: int, amount: float) -> wp.vec3:
    sx = -1.0; sy = -1.0; sz = -1.0
    if child & 1:
        sx = 1.0
    if child & 2:
        sy = 1.0
    if child & 4:
        sz = 1.0
    return wp.vec3(sx * amount, sy * amount, sz * amount)


@wp.kernel
def refine_impacted_solids(
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    base_fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    impact_impulse: wp.array(dtype=float),
    local_impact_active: wp.array(dtype=wp.int32),
    hydraulic_boundary_base: wp.array(dtype=wp.int32),
    hydraulic_boundary: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    normal_axis: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
    preimpact_building: wp.array(dtype=wp.int32),
    refinement_counters: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    old_count: int,
    capacity: int,
    crack_parent_radius: float,
    glass_crack_parent_radius: float,
    impact_parent_radius: float,
    damage_trigger: float,
    load_acceleration_trigger: float,
):
    i = wp.tid()
    fid = fragment_id[i]
    if i >= old_count or kind[i] == 0 or base_fixed[i] != 0 or normal_axis[i] < 0:
        return
    if fid >= 0 and rigid_state[fid] != 0:
        return
    role = structural_class[i]
    effective_crack_radius = crack_parent_radius
    if role == STRUCT_GLASS:
        effective_crack_radius = glass_crack_parent_radius
    if radius[i] <= effective_crack_radius:
        return
    loaded = impact_impulse[i] >= material_impact_impulse_threshold(role)
    preimpact = False
    bid = building_id[i]
    if bid >= 0:
        preimpact = preimpact_building[bid] != 0 and radius[i] > impact_parent_radius
    fracture_refine = (damage[i] >= damage_trigger or loaded) and radius[i] > effective_crack_radius
    if not preimpact and not fracture_refine:
        return

    child_count = 4
    if role == STRUCT_BEAM or role == STRUCT_COLUMN:
        child_count = 2
    elif role == STRUCT_CORE:
        child_count = 8
    extra_children = child_count - 1
    base = wp.atomic_add(count, 0, extra_children)
    if base + extra_children - 1 >= capacity:
        return
    parent_x = x[i]; parent_rest = rest_x[i]; parent_v = v[i]
    parent_radius = radius[i]; parent_mass = mass[i]; parent_volume = volume[i]
    parent_kind = kind[i]; parent_material = material[i]; parent_building = building_id[i]
    parent_role = structural_class[i]
    parent_fixed = fixed[i]; parent_base_fixed = base_fixed[i]
    parent_damage = damage[i]; parent_rho = rho_reference[i]
    parent_impact = impact_impulse[i]; parent_local_impact = local_impact_active[i]
    parent_hydraulic_base = hydraulic_boundary_base[i]
    parent_hydraulic = hydraulic_boundary[i]
    parent_fragment = fragment_id[i]; axis = normal_axis[i]
    child_offset = parent_radius * 0.5208333333

    wp.atomic_add(refinement_counters, role, 1)
    for child in range(8):
        if child >= child_count:
            continue
        target = i
        if child > 0:
            target = base + child - 1
        offset = planar_child_offset(axis, child, child_offset)
        if role == STRUCT_BEAM or role == STRUCT_COLUMN:
            offset = linear_child_offset(axis, child, child_offset)
        elif role == STRUCT_CORE:
            offset = volume_child_offset(child, child_offset)
        x[target] = parent_x + offset
        rest_x[target] = parent_rest + offset
        v[target] = parent_v
        radius[target] = parent_radius * 0.5
        mass[target] = parent_mass / float(child_count)
        volume[target] = parent_volume / float(child_count)
        kind[target] = parent_kind
        material[target] = parent_material
        structural_class[target] = parent_role
        building_id[target] = parent_building
        fixed[target] = parent_fixed
        base_fixed[target] = parent_base_fixed
        damage[target] = parent_damage
        rho_reference[target] = parent_rho
        solid_force[target] = wp.vec3(0.0)
        impact_impulse[target] = parent_impact
        local_impact_active[target] = parent_local_impact
        hydraulic_boundary_base[target] = parent_hydraulic_base
        hydraulic_boundary[target] = parent_hydraulic
        fragment_id[target] = parent_fragment
        normal_axis[target] = axis
