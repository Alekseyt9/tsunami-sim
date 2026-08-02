"""Small V3 GPU kernels layered on top of the stable V2 solver.

V3 keeps inactive buildings as immovable SPH boundaries, but skips their
expensive bond traversal until enough facade particles receive water load.
"""

import warp as wp

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


@wp.kernel
def count_loaded_building_particles(
    rest_x: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    mass: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    load_acceleration_threshold: float,
    maximum_activation_elevation: float,
    hits: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    bid = building_id[i]
    if kind[i] != 0 and bid >= 0 and rest_x[i][1] <= maximum_activation_elevation:
        threshold = mass[i] * load_acceleration_threshold
        # The tsunami travels along +Z. Requiring forward load near the base
        # prevents isolated overhead/side spray from waking the whole graph.
        if solid_force[i][2] > threshold:
            wp.atomic_add(hits, bid, 1)


@wp.kernel
def activate_buildings_from_hits(
    hits: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    exposure_seconds: wp.array(dtype=float),
    minimum_hits: int,
    dt: float,
    required_exposure_seconds: float,
    exposure_decay_multiplier: float,
):
    bid = wp.tid()
    if active[bid] != 0:
        return
    if hits[bid] >= minimum_hits:
        exposure_seconds[bid] += dt
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
    base_fixed: wp.array(dtype=wp.int32),
    building_active: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    bid = building_id[i]
    if kind[i] != 0 and bid >= 0:
        if building_active[bid] != 0:
            fixed[i] = base_fixed[i]
        else:
            fixed[i] = 1


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
    fragment_id: wp.array(dtype=wp.int32),
    rigid_state: wp.array(dtype=wp.int32),
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
    local_damage = damage[i]
    gravity_fraction = local_damage * local_damage
    if body_rigid:
        gravity_fraction = 1.0
    force = solid_force[i] + wp.vec3(0.0, -9.81 * mass[i] * gravity_fraction, 0.0)
    hydro_loaded = wp.length(solid_force[i]) > mass[i] * 0.8
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

        if body_rigid and same_fragment:
            # A rigid cluster is projected from one body transform, so neither
            # springs nor self-contact are needed between its sample particles.
            continue
        if bonded and same_fragment:
            # An architectural chunk may deform, but it cannot dissolve into
            # individual lattice particles. This is the anti-dust constraint.
            strain = (dist - rest_dist) / wp.max(rest_dist, 1.0e-4)
            stiffness = wp.min(material_stiffness(material[i]), material_stiffness(material[j]))
            damping = 75000.0 * wp.dot(v[j] - v[i], delta / dist)
            force += (stiffness * internal_stiffness_multiplier * strain + damping) * (delta / dist) * radius[i] * radius[i]
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
            crack_front = hydro_loaded or (damage[j] > propagation_threshold and abs_strain > limit * 2.5)
            if abs_strain > limit and crack_front:
                normalized = (abs_strain - limit) / wp.max(limit, 1.0e-4)
                role_rate = wp.max(
                    structural_damage_rate_multiplier(structural_class[i]),
                    structural_damage_rate_multiplier(structural_class[j]),
                )
                increment = wp.min(
                    normalized * dt * damage_rate * role_rate,
                    max_damage_per_substep * role_rate,
                )
                local_damage += increment
            if local_damage < 1.0:
                stiffness = wp.min(material_stiffness(material[i]), material_stiffness(material[j]))
                damping = 50000.0 * wp.dot(v[j] - v[i], delta / dist)
                cohesion = (1.0 - local_damage) * (1.0 - local_damage)
                force += cohesion * (stiffness * strain + damping) * (delta / dist) * radius[i] * radius[i]
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
                force -= normal * (3.0e6 * penetration + 2600.0 * wp.min(closing, 0.0))

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
    body_center: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    contact_acceleration_peak: wp.array(dtype=float),
    query_radius: float,
    normal_damping: float,
    tangential_damping: float,
):
    """Pairwise rubble contact using rigid surface samples as the broadphase."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    fid = fragment_id[i]
    if kind[i] == 0 or fid < 0 or rigid_state[fid] == 0:
        return
    xi = x[i]
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        other = fragment_id[j]
        if j <= i or kind[j] == 0 or other < 0 or other == fid or rigid_state[other] == 0:
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
    if rigid_state[body] != 0 and contact_acceleration_peak[body] >= acceleration_threshold:
        rigid_state[body] = 0
        wp.atomic_add(reactivated_count, 0, 1)


@wp.kernel
def integrate_rigid_bodies(
    rigid_state: wp.array(dtype=wp.int32),
    body_center: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_linear_velocity: wp.array(dtype=wp.vec3),
    body_angular_velocity: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_inverse_inertia: wp.array(dtype=wp.mat33),
    body_force: wp.array2d(dtype=float),
    body_torque: wp.array2d(dtype=float),
    dt: float,
    linear_damping: float,
    angular_damping: float,
    maximum_angular_speed: float,
):
    body = wp.tid()
    if rigid_state[body] == 0 or body_mass[body] <= 0.0:
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
    angular_speed = wp.length(angular_velocity)
    if angular_speed > maximum_angular_speed:
        angular_velocity *= maximum_angular_speed / angular_speed
        angular_speed = maximum_angular_speed
    if angular_speed * dt > 1.0e-8:
        delta = wp.quat_from_axis_angle(angular_velocity / angular_speed, angular_speed * dt)
        orientation = wp.normalize(delta * orientation)
    body_center[body] += linear_velocity * dt
    body_orientation[body] = orientation
    body_linear_velocity[body] = linear_velocity
    body_angular_velocity[body] = angular_velocity


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
def compute_density_multirate(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
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
    xi = x[i]
    rhoi = float(0.0)
    query = wp.hash_grid_query(grid, xi, max_support)
    for j in query:
        rij = xi - x[j]
        r2 = wp.dot(rij, rij)
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
def compute_fluid_forces_multirate(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    time_level: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    deferred_impulse: wp.array2d(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    rest_density: float,
    sound_speed: float,
    max_density_ratio: float,
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
    rhoi = rho[i]
    gamma = 7.0
    stiffness = rest_density * sound_speed * sound_speed / gamma
    pi = stiffness * (wp.pow(wp.min(rhoi / rest_density, max_density_ratio), gamma) - 1.0)
    pi = wp.max(pi, -0.02 * stiffness)
    ai = wp.vec3(0.0, -9.81, 0.0)
    xsph = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, max_support)
    for j in query:
        if j == i:
            continue
        r = xi - x[j]
        dist = wp.length(r)
        support = 4.0 * wp.max(radius[i], radius[j])
        if dist >= support or dist <= 1.0e-5:
            continue
        rhoj = rho[j]
        pj = pi
        if kind[j] == 0:
            pj = stiffness * (wp.pow(wp.min(rhoj / rest_density, max_density_ratio), gamma) - 1.0)
            pj = wp.max(pj, -0.02 * stiffness)
        else:
            rhoj = rest_density
        grad = spiky_grad(r, dist, support)
        neighbour_mass = mass[j]
        if kind[j] != 0:
            neighbour_mass = rest_density * volume[j]
        pair_acc = -neighbour_mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * grad
        pair_acc += viscosity * neighbour_mass * (v[j] - vi) / rhoj * viscosity_laplacian(dist, support) / rhoi
        ai += pair_acc
        if kind[j] == 0:
            xsph += mass[j] / rhoj * (v[j] - vi) * poly6(wp.dot(r, r), support)
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
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    current_vertex: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
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
    return base


@wp.kernel
def raster_facade_color(
    vertex: wp.array(dtype=wp.vec3),
    rest_vertex: wp.array(dtype=wp.vec3),
    anchor: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    particle_damage: wp.array(dtype=float),
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
):
    triangle = wp.tid()
    panel = triangle // 2
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
    light = 0.34 + 0.66 * wp.abs(wp.dot(normal, wp.normalize(wp.vec3(-0.35, 0.72, -0.42))))
    anchor_base = panel * 4
    panel_damage = 0.25 * (
        particle_damage[anchor[anchor_base]] + particle_damage[anchor[anchor_base + 1]] +
        particle_damage[anchor[anchor_base + 2]] + particle_damage[anchor[anchor_base + 3]]
    )
    base = wp.lerp(base, wp.vec3(0.18, 0.055, 0.03), wp.clamp(panel_damage, 0.0, 1.0)) * light
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            fx = float(px) + 0.5; fy = float(py) + 0.5
            w0 = edge2(b, c, fx, fy) / area
            w1 = edge2(c, a, fx, fy) / area
            w2 = 1.0 - w0 - w1
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                index = py * width + px
                if z <= depth[index] + 0.02:
                    color[index] = base


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
    loaded = wp.length(solid_force[i]) > mass[i] * load_acceleration_trigger
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
        fragment_id[target] = parent_fragment
        normal_axis[target] = axis
