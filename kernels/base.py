"""CUDA kernels for DELUGE V2.

Fluid: variable-resolution weakly-compressible SPH (WCSPH).
Solid: bond-based particle lattice with strain damage and DEM contact.
Coupling: boundary-particle pressure with equal/opposite momentum transfer.
"""

import warp as wp

PI = wp.constant(3.141592653589793)


@wp.func
def poly6(r2: float, h: float) -> float:
    if r2 >= h * h:
        return 0.0
    x = h * h - r2
    return 315.0 / (64.0 * PI * wp.pow(h, 9.0)) * x * x * x


@wp.func
def spiky_grad(r: wp.vec3, length: float, h: float) -> wp.vec3:
    if length <= 1.0e-5 or length >= h:
        return wp.vec3(0.0)
    scale = -45.0 / (PI * wp.pow(h, 6.0)) * (h - length) * (h - length)
    return r * (scale / length)


@wp.func
def viscosity_laplacian(length: float, h: float) -> float:
    if length >= h:
        return 0.0
    return 45.0 / (PI * wp.pow(h, 6.0)) * (h - length)


@wp.kernel
def clear_vec3(values: wp.array(dtype=wp.vec3)):
    values[wp.tid()] = wp.vec3(0.0)


@wp.kernel
def copy_vec3(source: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec3)):
    target[wp.tid()] = source[wp.tid()]


@wp.kernel
def clear_gbuffer(
    normal: wp.array(dtype=wp.vec3),
    motion: wp.array(dtype=wp.vec2),
    material: wp.array(dtype=wp.int32),
    roughness: wp.array(dtype=float),
    metallic: wp.array(dtype=float),
):
    i = wp.tid()
    normal[i] = wp.vec3(0.0, 1.0, 0.0)
    motion[i] = wp.vec2(0.0, 0.0)
    material[i] = 0
    roughness[i] = 1.0
    metallic[i] = 0.0


@wp.kernel
def compute_density(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
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
    if kind[i] != 0:
        rho[i] = rest_density
        return

    xi = x[i]
    rhoi = float(0.0)
    query = wp.hash_grid_query(grid, xi, max_support)
    for j in query:
        rij = xi - x[j]
        r2 = wp.dot(rij, rij)
        # radius stores half the nominal spacing; a 4r support gives h=2*dx,
        # enough neighbours for a normalized 3-D poly6 estimate.
        support = 4.0 * wp.max(radius[i], radius[j])
        effective_mass = mass[j]
        if kind[j] != 0:
            # Boundary particles contribute displaced water volume, not their
            # structural material density.
            effective_mass = rest_density * volume[j]
        rhoi += effective_mass * poly6(r2, support)
    rhoi = wp.max(rhoi, rest_density * 0.15)
    reference = rho_reference[i]
    if reference <= 0.0:
        # Calibrate the quadrature error of this particle's current resolution
        # neighbourhood. This removes pressure impulses at a coarse/fine
        # interface while retaining subsequent physical compression ratios.
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
def compute_fluid_forces(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    rest_density: float,
    sound_speed: float,
    max_density_ratio: float,
    viscosity: float,
    xsph_strength: float,
    max_support: float,
    dt: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] != 0:
        return

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
        # Use the same h=2*spacing support as the density summation. The former
        # h=1.1*spacing reached only the six axial lattice neighbours in 3-D,
        # producing a noisy, under-resolved pressure gradient and artificial
        # vertical kinetic energy throughout otherwise calm water.
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
        neighbor_mass = mass[j]
        if kind[j] != 0:
            neighbor_mass = rest_density * volume[j]
        pair_acc = -neighbor_mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * grad
        pair_acc += viscosity * neighbor_mass * (v[j] - vi) / rhoj * viscosity_laplacian(dist, support) / rhoi
        ai += pair_acc

        if kind[j] == 0:
            # XSPH regularization removes high-frequency particle disorder
            # while preserving the velocity of the bulk flow.
            xsph += mass[j] / rhoj * (v[j] - vi) * poly6(wp.dot(r, r), support)

        if kind[j] != 0:
            # Newton's third law: feed boundary pressure back into the solid.
            wp.atomic_add(solid_force, j, -pair_acc * mass[i])

    ai += xsph * (xsph_strength / wp.max(dt, 1.0e-7))

    # Limit catastrophic WCSPH spikes without changing ordinary flow.
    a_len = wp.length(ai)
    if a_len > 8000.0:
        ai = ai * (8000.0 / a_len)
    acceleration[i] = ai


@wp.func
def material_stiffness(material: int) -> float:
    if material == 2:  # glass
        return 1.0e8
    if material == 3:  # reinforcement steel
        return 8.0e8
    if material == 4:  # light masonry / shop walls
        return 1.2e8
    if material == 5:  # wood
        return 6.0e7
    return 3.0e8       # reinforced concrete / masonry


@wp.func
def material_failure_strain(material: int) -> float:
    if material == 2:
        return 0.012
    if material == 3:
        return 0.11
    if material == 4:
        return 0.018
    if material == 5:
        return 0.070
    return 0.032


@wp.func
def deformable_contact_magnitude(
    penetration: float,
    closing_speed: float,
    stiffness: float,
    damping: float,
) -> float:
    return wp.max(
        stiffness * penetration + damping * wp.max(-closing_speed, 0.0),
        0.0,
    )


@wp.kernel
def compute_solid_forces(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    solid_force: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    max_support: float,
    dt: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] == 0:
        return
    if fixed[i] != 0:
        acceleration[i] = wp.vec3(0.0)
        return

    xi = x[i]
    ri = rest_x[i]
    local_damage = damage[i]
    # The rest lattice represents a completed building that is already in
    # static equilibrium under its own weight.  Applying gravity in full to
    # this zero-strain lattice would first require a separate static preload
    # solve; without it, thin one-particle slabs sag almost in free fall before
    # the wave arrives.  This is the incremental-dynamics equivalent of that
    # preload: intact material keeps its gravity reaction, while released
    # fragments progressively acquire their full weight as damage reaches 1.
    gravity_fraction = local_damage * local_damage
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
        # Include face diagonals (sqrt(2) * spacing). Axial bonds alone give a
        # rectangular lattice essentially no shear rigidity and let walls fold
        # like hinged panels.
        bond_range = 3.2 * wp.max(radius[i], radius[j])
        same_structure = building_id[i] == building_id[j]

        if same_structure and rest_dist < bond_range and local_damage < 1.0:
            strain = (dist - rest_dist) / wp.max(rest_dist, 1.0e-4)
            limit = wp.min(material_failure_strain(material[i]), material_failure_strain(material[j]))
            abs_strain = wp.abs(strain)
            # Gravity can pre-stress the discrete lattice before the tsunami
            # arrives. Fracture starts only from a hydrodynamically loaded
            # boundary particle, then propagates through already damaged bonds.
            crack_front = hydro_loaded or damage[j] > 0.02 or local_damage > 0.02
            if abs_strain > limit and crack_front:
                local_damage += (abs_strain - limit) * dt * 32.0
            if local_damage < 1.0:
                stiffness = wp.min(material_stiffness(material[i]), material_stiffness(material[j]))
                damping = 50000.0 * wp.dot(v[j] - v[i], delta / dist)
                force += (stiffness * strain + damping) * (delta / dist) * radius[i] * radius[i]
        else:
            # DEM contact for separated fragments and different buildings.
            contact = radius[i] + radius[j]
            if dist < contact:
                penetration = contact - dist
                normal = delta / dist
                closing = wp.dot(v[j] - v[i], normal)
                force -= normal * deformable_contact_magnitude(
                    penetration, closing, 3.0e6, 9000.0
                )

    damage[i] = wp.min(local_damage, 1.0)
    ai = force / wp.max(mass[i], 1.0)
    a_len = wp.length(ai)
    if a_len > 6000.0:
        ai *= 6000.0 / a_len
    acceleration[i] = ai


@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    dt: float,
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
    if fixed[i] != 0:
        return
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
            # Free-slip water boundary with a small physical bed drag. The old
            # solid-fragment contact multiplied tangential water velocity by
            # 0.78 every substep and bounced it upward, destroying horizontal
            # momentum while injecting artificial spray.
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
def refine_entering_fluid(
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    fluid_group_id: wp.array(dtype=wp.int32),
    wave_cohort: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    fluid_group_counter: wp.array(dtype=wp.int32),
    old_count: int,
    capacity: int,
    fine_radius: float,
    refine_z: float,
    surface_only: int,
    classified_surface_only: int,
    surface_minimum_y: float,
    turbulent_vertical_speed: float,
):
    i = wp.tid()
    if i >= old_count or kind[i] != 0 or radius[i] <= fine_radius * 1.25 or x[i][2] < refine_z:
        return
    if surface_only != 0:
        near_surface = x[i][1] >= surface_minimum_y
        if classified_surface_only != 0:
            near_surface = surface_mask[i] != 0
        turbulent = wp.abs(v[i][1]) >= turbulent_vertical_speed
        if not near_surface and not turbulent:
            return

    base = wp.atomic_add(count, 0, 7)
    if base + 6 >= capacity:
        return
    group_id = wp.atomic_add(fluid_group_counter, 0, 1)

    parent_x = x[i]
    parent_v = v[i]
    parent_cohort = wave_cohort[i]
    child_r = radius[i] * 0.5
    child_m = mass[i] * 0.125
    child_vol = volume[i] * 0.125
    # Child centers must be one child diameter apart. The former 0.55
    # multiplier overlapped children and produced a non-physical pressure burst.
    offsets = wp.vec3(-1.0, -1.0, -1.0) * child_r
    x[i] = parent_x + offsets
    rest_x[i] = x[i]
    radius[i] = child_r; mass[i] = child_m; volume[i] = child_vol
    rho_reference[i] = 0.0
    fluid_group_id[i] = group_id
    wave_cohort[i] = parent_cohort

    for c in range(7):
        idx = base + c
        code = c + 1
        ox = float((code & 1) * 2 - 1)
        oy = float(((code >> 1) & 1) * 2 - 1)
        oz = float(((code >> 2) & 1) * 2 - 1)
        x[idx] = parent_x + wp.vec3(ox, oy, oz) * child_r
        rest_x[idx] = x[idx]
        v[idx] = parent_v
        radius[idx] = child_r; mass[idx] = child_m; volume[idx] = child_vol
        kind[idx] = 0; material[idx] = 0; building_id[idx] = -1; fixed[idx] = 0; damage[idx] = 0.0
        rho_reference[idx] = 0.0
        fluid_group_id[idx] = group_id
        wave_cohort[idx] = parent_cohort


@wp.kernel
def clear_render(depth: wp.array(dtype=float), color: wp.array(dtype=wp.vec3), width: int, height: int):
    i = wp.tid()
    y = i // width
    t = float(y) / float(height)
    depth[i] = 1.0e9
    upper = wp.vec3(0.075, 0.16, 0.23)
    horizon = wp.vec3(0.46, 0.56, 0.60)
    lower = wp.vec3(0.23, 0.28, 0.29)
    if t < 0.62:
        color[i] = wp.lerp(upper, horizon, wp.smoothstep(0.0, 0.62, t))
    else:
        color[i] = wp.lerp(horizon, lower, wp.smoothstep(0.62, 1.0, t))


@wp.kernel
def clear_depth(depth: wp.array(dtype=float)):
    depth[wp.tid()] = 1.0e9


@wp.kernel
def clear_scalar(values: wp.array(dtype=float), value: float):
    values[wp.tid()] = value


@wp.kernel
def clear_int(values: wp.array(dtype=wp.int32)):
    values[wp.tid()] = 0


@wp.kernel
def count_damaged(kind: wp.array(dtype=wp.int32), damage: wp.array(dtype=float), counter: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if kind[i] != 0 and damage[i] > 0.05:
        wp.atomic_add(counter, 0, 1)


@wp.func
def project_point(p: wp.vec3, cam: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3, focal: float, width: int, height: int) -> wp.vec3:
    rel = p - cam
    z = wp.dot(rel, forward)
    x = float(width) * 0.5 + focal * wp.dot(rel, right) / wp.max(z, 0.001)
    y = float(height) * 0.5 - focal * wp.dot(rel, up) / wp.max(z, 0.001)
    return wp.vec3(x, y, z)


@wp.func
def physical_sky_radiance(
    direction: wp.vec3,
    sun_direction: wp.vec3,
    turbidity: float,
    sky_intensity: float,
    sun_intensity: float,
) -> wp.vec3:
    """Compact HDR daylight model shared by sky, IBL and water.

    This is an analytic clear/hazy sky approximation rather than an LDR
    background gradient.  It preserves a real HDR sun disc and horizon glow,
    which can then be reflected and passed through filmic tone mapping.
    """
    ray = wp.normalize(direction)
    sun = wp.normalize(sun_direction)
    elevation = wp.clamp(ray[1], -1.0, 1.0)
    haze = wp.clamp((turbidity - 1.5) / 7.0, 0.0, 1.0)
    horizon_amount = wp.exp(-wp.max(elevation, 0.0) * (4.2 - 1.6 * haze))
    zenith = wp.vec3(0.055, 0.145, 0.285) * (1.05 - 0.24 * haze)
    horizon = wp.vec3(0.54, 0.63, 0.66) * (0.82 + 0.34 * haze)
    ground = wp.vec3(0.105, 0.115, 0.105)
    sky = wp.lerp(zenith, horizon, horizon_amount)
    if elevation < 0.0:
        below = wp.smoothstep(-0.24, 0.0, elevation)
        sky = wp.lerp(ground, horizon, below)

    sun_cosine = wp.clamp(wp.dot(ray, sun), 0.0, 1.0)
    # A slightly enlarged disc remains sub-pixel stable at 360p inset views;
    # the halo carries the lower-frequency atmospheric forward scattering.
    sun_disc = wp.smoothstep(0.99955, 0.99993, sun_cosine)
    sun_halo = wp.pow(sun_cosine, 48.0 / (1.0 + 0.55 * haze))
    sun_color = wp.vec3(1.0, 0.86, 0.62)
    sky *= sky_intensity
    sky += sun_color * sun_intensity * (sun_disc * 5.5 + sun_halo * (0.055 + 0.10 * haze))
    return sky


@wp.kernel
def render_physical_sky(
    color: wp.array(dtype=wp.vec3),
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
    sun_direction: wp.vec3,
    turbidity: float,
    sky_intensity: float,
    sun_intensity: float,
):
    i = wp.tid()
    x = i % width
    y = i // width
    camera_x = (float(x) + 0.5 - float(width) * 0.5) / focal
    camera_y = -(float(y) + 0.5 - float(height) * 0.5) / focal
    ray = wp.normalize(forward + right * camera_x + up * camera_y)
    color[i] = physical_sky_radiance(
        ray, sun_direction, turbidity, sky_intensity, sun_intensity
    )


@wp.func
def aces_filmic_channel(value: float) -> float:
    x = wp.max(value, 0.0)
    return wp.clamp(
        x * (2.51 * x + 0.03) / wp.max(x * (2.43 * x + 0.59) + 0.14, 1.0e-6),
        0.0,
        1.0,
    )


@wp.kernel
def filmic_tonemap_color(
    hdr_color: wp.array(dtype=wp.vec3),
    display_color: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    exposure_ev: float,
    bloom_threshold: float,
    bloom_strength: float,
):
    """Resolve HDR after TAA with a compact glare gather and ACES curve."""
    i = wp.tid()
    x = i % width
    y = i // width
    bloom = wp.vec3(0.0)
    weight_sum = float(0.0)
    for sample in range(13):
        ox = 0
        oy = 0
        weight = 1.0
        if sample == 1: ox = 2
        elif sample == 2: ox = -2
        elif sample == 3: oy = 2
        elif sample == 4: oy = -2
        elif sample == 5: ox = 4; weight = 0.55
        elif sample == 6: ox = -4; weight = 0.55
        elif sample == 7: oy = 4; weight = 0.55
        elif sample == 8: oy = -4; weight = 0.55
        elif sample == 9: ox = 2; oy = 2; weight = 0.72
        elif sample == 10: ox = -2; oy = 2; weight = 0.72
        elif sample == 11: ox = 2; oy = -2; weight = 0.72
        elif sample == 12: ox = -2; oy = -2; weight = 0.72
        px = x + ox
        py = y + oy
        if px >= 0 and px < width and py >= 0 and py < height:
            value = hdr_color[py * width + px]
            luminance = wp.dot(value, wp.vec3(0.2126, 0.7152, 0.0722))
            bright = wp.max(luminance - bloom_threshold, 0.0) / wp.max(luminance, 1.0e-5)
            bloom += value * bright * weight
            weight_sum += weight
    bloom /= wp.max(weight_sum, 1.0)
    exposure = wp.pow(2.0, exposure_ev)
    value = (hdr_color[i] + bloom * bloom_strength) * exposure
    display_color[i] = wp.vec3(
        aces_filmic_channel(value[0]),
        aces_filmic_channel(value[1]),
        aces_filmic_channel(value[2]),
    )


@wp.kernel
def raster_depth(
    x: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float), kind: wp.array(dtype=wp.int32), depth: wp.array(dtype=float),
    cam: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
    focal: float, width: int, height: int,
):
    i = wp.tid()
    if kind[i] == 0:
        return
    p = project_point(x[i], cam, right, up, forward, focal, width, height)
    if p[2] <= 0.1:
        return
    rp = wp.clamp(focal * radius[i] / p[2] * 1.35, 1.0, 6.0)
    cx = int(p[0]); cy = int(p[1])
    for oy in range(-6, 7):
        for ox in range(-6, 7):
            px = cx + ox; py = cy + oy
            if px >= 0 and px < width and py >= 0 and py < height and float(ox * ox + oy * oy) <= rp * rp:
                wp.atomic_min(depth, py * width + px, p[2])


@wp.kernel
def raster_color(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32), material: wp.array(dtype=wp.int32), damage: wp.array(dtype=float),
    depth: wp.array(dtype=float), color: wp.array(dtype=wp.vec3),
    cam: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
    focal: float, width: int, height: int,
):
    i = wp.tid()
    if kind[i] == 0:
        return
    p = project_point(x[i], cam, right, up, forward, focal, width, height)
    if p[2] <= 0.1:
        return
    rp = wp.clamp(focal * radius[i] / p[2] * 1.35, 1.0, 6.0)
    cx = int(p[0]); cy = int(p[1])
    base = wp.vec3(0.52, 0.52, 0.49)
    if material[i] == 2:
        base = wp.vec3(0.20, 0.55, 0.68)
    if material[i] == 3:
        base = wp.vec3(0.18, 0.20, 0.21)
    if material[i] == 4:
        base = wp.vec3(0.55, 0.48, 0.40)
    if material[i] == 5:
        base = wp.vec3(0.32, 0.20, 0.11)
    # Preserve the physical material hue. Damage lowers brightness instead of
    # replacing every fragment with the old diagnostic brown overlay.
    base *= 1.0 - 0.38 * wp.clamp(damage[i], 0.0, 1.0)

    for oy in range(-6, 7):
        for ox in range(-6, 7):
            px = cx + ox; py = cy + oy
            rr = float(ox * ox + oy * oy)
            if px >= 0 and px < width and py >= 0 and py < height and rr <= rp * rp:
                index = py * width + px
                if p[2] <= depth[index] + 0.015:
                    nz = wp.sqrt(wp.max(0.0, 1.0 - rr / (rp * rp)))
                    light = 0.42 + 0.58 * nz
                    color[index] = base * light


@wp.kernel
def raster_water_depth(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float), kind: wp.array(dtype=wp.int32),
    depth: wp.array(dtype=float), back_depth: wp.array(dtype=float), foam_field: wp.array(dtype=float),
    cam: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
    focal: float, width: int, height: int,
):
    i = wp.tid()
    if kind[i] != 0:
        return
    p = project_point(x[i], cam, right, up, forward, focal, width, height)
    if p[2] <= 0.1:
        return
    # Overlapping splats form an implicit screen-space surface instead of
    # exposing individual simulation particles.
    world_r = radius[i] * 2.35
    rp = wp.clamp(focal * world_r / p[2], 1.25, 11.0)
    cx = int(p[0]); cy = int(p[1])
    for oy in range(-11, 12):
        for ox in range(-11, 12):
            px = cx + ox; py = cy + oy
            rr = float(ox * ox + oy * oy)
            if px >= 0 and px < width and py >= 0 and py < height and rr <= rp * rp:
                extent = world_r * wp.sqrt(wp.max(0.0, 1.0 - rr / (rp * rp)))
                sphere_z = p[2] - extent
                sphere_back_z = p[2] + extent
                pixel = py * width + px
                wp.atomic_min(depth, pixel, sphere_z)
                wp.atomic_max(back_depth, pixel, sphere_back_z)
                # Foam comes from energetic vertical/turbulent motion, not the
                # silhouette of every particle splat.
                speed = wp.length(v[i])
                foam = wp.smoothstep(2.5, 11.0, wp.abs(v[i][1]))
                foam += 0.45 * wp.smoothstep(20.0, 45.0, speed)
                wp.atomic_max(foam_field, pixel, wp.min(foam, 1.0))


@wp.kernel
def bilateral_depth(
    source: wp.array(dtype=float), target: wp.array(dtype=float), width: int, height: int,
    spatial_sigma: float, depth_sigma: float,
):
    i = wp.tid()
    x = i % width; y = i // width
    center = source[i]
    if center > 1.0e8:
        target[i] = center
        return
    weighted = float(0.0); weight_sum = float(0.0)
    for oy in range(-3, 4):
        for ox in range(-3, 4):
            px = x + ox; py = y + oy
            if px >= 0 and px < width and py >= 0 and py < height:
                sample = source[py * width + px]
                if sample < 1.0e8:
                    ds = float(ox * ox + oy * oy) / (2.0 * spatial_sigma * spatial_sigma)
                    dd = (sample - center) * (sample - center) / (2.0 * depth_sigma * depth_sigma)
                    w = wp.exp(-ds - dd)
                    weighted += sample * w; weight_sum += w
    target[i] = weighted / wp.max(weight_sum, 1.0e-6)


@wp.kernel
def bilateral_depth_axis(
    source: wp.array(dtype=float), target: wp.array(dtype=float), width: int, height: int,
    spatial_sigma: float, depth_sigma: float, axis: int,
):
    """Separable edge-aware smoothing with small-hole reconstruction."""
    i = wp.tid()
    x = i % width; y = i // width
    center = source[i]
    valid_center = center < 1.0e8
    if not valid_center:
        # Do not dilate isolated droplets into axis-aligned square patches.
        target[i] = center
        return
    weighted = float(0.0); weight_sum = float(0.0)
    for k in range(-3, 4):
        px = x; py = y
        if axis == 0:
            px = x + k
        else:
            py = y + k
        if px >= 0 and px < width and py >= 0 and py < height:
            sample = source[py * width + px]
            if sample < 1.0e8:
                ds = float(k * k) / (2.0 * spatial_sigma * spatial_sigma)
                dd = float(0.0)
                dd = (sample - center) * (sample - center) / (2.0 * depth_sigma * depth_sigma)
                w = wp.exp(-ds - dd)
                weighted += sample * w
                weight_sum += w
    if weight_sum > 1.0e-6:
        target[i] = weighted / weight_sum
    else:
        target[i] = 1.0e9


@wp.kernel
def temporal_stabilize_water_depth(
    current_depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    history_depth: wp.array(dtype=float),
    output_depth: wp.array(dtype=float),
    has_history: int,
    history_weight: float,
    disocclusion_threshold: float,
):
    """Static-camera temporal accumulation with splash/disocclusion rejection."""
    i = wp.tid()
    current = current_depth[i]
    previous = history_depth[i]
    result = current
    if current < 1.0e8 and has_history != 0 and previous < 1.0e8:
        threshold = wp.max(disocclusion_threshold, current * 0.0035)
        difference = wp.abs(current - previous)
        if difference < threshold:
            # Energetic foam and spray must remain responsive. Calm connected
            # water receives the strongest accumulation and loses lattice shimmer.
            foam = wp.clamp(foam_field[i], 0.0, 1.0)
            stable = 1.0 - wp.smoothstep(threshold * 0.25, threshold, difference)
            weight = wp.clamp(history_weight * stable * (1.0 - 0.72 * foam), 0.0, 0.94)
            result = current * (1.0 - weight) + previous * weight
    history_depth[i] = result
    output_depth[i] = result


@wp.kernel
def temporal_antialias_color(
    current_color: wp.array(dtype=wp.vec3),
    current_depth: wp.array(dtype=float),
    water_depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    motion: wp.array(dtype=wp.vec2),
    history_color: wp.array(dtype=wp.vec3),
    history_depth: wp.array(dtype=float),
    output_color: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    has_history: int,
    history_weight: float,
):
    """Motion-reprojected TAA with depth rejection and neighbourhood clipping."""
    i = wp.tid()
    x = i % width
    y = i // width
    current = current_color[i]
    depth = wp.min(current_depth[i], water_depth[i])
    result = current
    if has_history != 0:
        velocity = motion[i]
        previous_x = int(wp.round(float(x) - velocity[0]))
        previous_y = int(wp.round(float(y) - velocity[1]))
        if previous_x >= 0 and previous_x < width and previous_y >= 0 and previous_y < height:
            previous_index = previous_y * width + previous_x
            previous_depth = history_depth[previous_index]
            valid_depth = depth > 1.0e8 and previous_depth > 1.0e8
            if depth < 1.0e8 and previous_depth < 1.0e8:
                threshold = wp.max(0.30, depth * 0.004)
                valid_depth = wp.abs(depth - previous_depth) < threshold
            if valid_depth:
                minimum = current
                maximum = current
                for oy in range(-1, 2):
                    for ox in range(-1, 2):
                        px = x + ox; py = y + oy
                        if px >= 0 and px < width and py >= 0 and py < height:
                            sample = current_color[py * width + px]
                            minimum = wp.vec3(
                                wp.min(minimum[0], sample[0]),
                                wp.min(minimum[1], sample[1]),
                                wp.min(minimum[2], sample[2]),
                            )
                            maximum = wp.vec3(
                                wp.max(maximum[0], sample[0]),
                                wp.max(maximum[1], sample[1]),
                                wp.max(maximum[2], sample[2]),
                            )
                history = history_color[previous_index]
                history = wp.vec3(
                    wp.clamp(history[0], minimum[0], maximum[0]),
                    wp.clamp(history[1], minimum[1], maximum[1]),
                    wp.clamp(history[2], minimum[2], maximum[2]),
                )
                speed = wp.length(velocity)
                responsive = wp.clamp(speed / 8.0, 0.0, 1.0)
                responsive = wp.max(responsive, wp.clamp(foam_field[i], 0.0, 1.0) * 0.82)
                weight = wp.clamp(history_weight * (1.0 - responsive), 0.0, 0.94)
                result = current * (1.0 - weight) + history * weight
    output_color[i] = result
    history_color[i] = result
    history_depth[i] = depth


@wp.kernel
def shade_water_surface(
    water_depth: wp.array(dtype=float),
    water_back_depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    scene_depth: wp.array(dtype=float), color: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    absorption_scale: float,
    refraction_strength: float,
    absorption_coefficient: wp.vec3,
    scattering_coefficient: wp.vec3,
    phase_g: float,
    maximum_optical_depth: float,
    sun_direction: wp.vec3,
    sky_turbidity: float,
    sky_intensity: float,
    sun_intensity: float,
    ibl_strength: float,
):
    """Single-layer participating-medium water over the opaque HDR scene."""
    i = wp.tid()
    x = i % width; y = i // width
    z = water_depth[i]
    if z > 1.0e8 or z >= scene_depth[i]:
        return
    zl = z; zr = z; zu = z; zd = z
    if x > 0 and water_depth[i - 1] < 1.0e8: zl = water_depth[i - 1]
    if x + 1 < width and water_depth[i + 1] < 1.0e8: zr = water_depth[i + 1]
    if y > 0 and water_depth[i - width] < 1.0e8: zu = water_depth[i - width]
    if y + 1 < height and water_depth[i + width] < 1.0e8: zd = water_depth[i + width]

    # Reconstruct a real world-space normal from neighbouring camera-depth
    # samples. Unlike the previous screen-facing normal, a calm horizontal
    # surface now reflects the sky above it from every camera angle.
    left_position = (
        forward * zl
        + right * ((float(x - 1) + 0.5 - float(width) * 0.5) * zl / focal)
        + up * (-(float(y) + 0.5 - float(height) * 0.5) * zl / focal)
    )
    right_position = (
        forward * zr
        + right * ((float(x + 1) + 0.5 - float(width) * 0.5) * zr / focal)
        + up * (-(float(y) + 0.5 - float(height) * 0.5) * zr / focal)
    )
    upper_position = (
        forward * zu
        + right * ((float(x) + 0.5 - float(width) * 0.5) * zu / focal)
        + up * (-(float(y - 1) + 0.5 - float(height) * 0.5) * zu / focal)
    )
    lower_position = (
        forward * zd
        + right * ((float(x) + 0.5 - float(width) * 0.5) * zd / focal)
        + up * (-(float(y + 1) + 0.5 - float(height) * 0.5) * zd / focal)
    )
    tangent_x = right_position - left_position
    tangent_y = lower_position - upper_position
    normal = wp.normalize(wp.cross(tangent_x, tangent_y))
    camera_x = (float(x) + 0.5 - float(width) * 0.5) / focal
    camera_y = -(float(y) + 0.5 - float(height) * 0.5) / focal
    view_ray = wp.normalize(forward + right * camera_x + up * camera_y)
    view_direction = -view_ray
    if wp.dot(normal, view_direction) < 0.0:
        normal = -normal
    facing = wp.clamp(wp.dot(normal, view_direction), 0.0, 1.0)
    fresnel = 0.0204 + 0.9796 * wp.pow(1.0 - facing, 5.0)
    sun = wp.normalize(sun_direction)
    half_vector = wp.normalize(view_direction + sun)
    sparkle = wp.pow(wp.max(wp.dot(normal, half_vector), 0.0), 220.0)
    foam = wp.clamp(foam_field[i], 0.0, 1.0)
    incident_dot = wp.dot(view_ray, normal)
    reflection_ray = wp.normalize(view_ray - normal * (2.0 * incident_dot))
    sky_reflection = physical_sky_radiance(
        reflection_ray, sun, sky_turbidity, sky_intensity, sun_intensity
    ) * ibl_strength
    reflected_scene = sky_reflection
    reflection_found = float(0.0)
    # Preserve a conservative local SSR contribution for nearby silhouettes;
    # the analytic HDR sky supplies valid data outside screen space.
    for step in range(1, 17):
        py = y - step * 4
        if py >= 0 and reflection_found < 0.5:
            sample_index = py * width + x
            if scene_depth[sample_index] < 1.0e8 and water_depth[sample_index] > 1.0e8:
                reflected_scene = color[sample_index]
                reflection_found = 1.0
    reflection = wp.lerp(sky_reflection, reflected_scene, reflection_found * 0.58)

    normal_right = wp.dot(normal, right)
    normal_up = wp.dot(normal, up)
    refract_x = wp.clamp(
        x + int(-normal_right * refraction_strength / wp.max(facing, 0.25)), 0, width - 1
    )
    refract_y = wp.clamp(
        y + int(normal_up * refraction_strength / wp.max(facing, 0.25)), 0, height - 1
    )
    refract_index = refract_y * width + refract_x
    refracted_scene = color[refract_index]
    refracted_depth = scene_depth[refract_index]

    exit_depth = water_back_depth[i]
    if refracted_depth < 1.0e8 and refracted_depth > z:
        if exit_depth <= z:
            exit_depth = refracted_depth
        else:
            exit_depth = wp.min(exit_depth, refracted_depth)
    optical_path = 0.32 + foam * 0.18
    if exit_depth > z:
        forward_cosine = wp.max(wp.dot(view_ray, forward), 0.12)
        optical_path = (exit_depth - z) / forward_cosine
    optical_path = wp.clamp(
        optical_path * absorption_scale, 0.06, maximum_optical_depth
    )
    extinction = absorption_coefficient + scattering_coefficient
    transmission = wp.vec3(
        wp.exp(-extinction[0] * optical_path),
        wp.exp(-extinction[1] * optical_path),
        wp.exp(-extinction[2] * optical_path),
    )
    g = wp.clamp(phase_g, -0.85, 0.85)
    phase_denominator = wp.pow(
        wp.max(1.0 + g * g - 2.0 * g * wp.dot(view_ray, sun), 0.02), 1.5
    )
    phase = (1.0 - g * g) / phase_denominator
    scatter_tint = wp.vec3(0.018, 0.22, 0.31) * (0.30 + 0.34 * phase)
    in_scatter = wp.vec3(
        scatter_tint[0] * scattering_coefficient[0] / wp.max(extinction[0], 1.0e-5),
        scatter_tint[1] * scattering_coefficient[1] / wp.max(extinction[1], 1.0e-5),
        scatter_tint[2] * scattering_coefficient[2] / wp.max(extinction[2], 1.0e-5),
    ) * sky_intensity
    transmitted = wp.vec3(
        refracted_scene[0] * transmission[0] + in_scatter[0] * (1.0 - transmission[0]),
        refracted_scene[1] * transmission[1] + in_scatter[1] * (1.0 - transmission[1]),
        refracted_scene[2] * transmission[2] + in_scatter[2] * (1.0 - transmission[2]),
    )
    water = wp.lerp(transmitted, reflection, fresnel)
    water += wp.vec3(1.0, 0.83, 0.58) * sparkle * sun_intensity * 0.75
    # Foam is composed in its own material pass below; retain only the diffuse
    # subsurface brightening it contributes to the water immediately beneath.
    water = wp.lerp(water, wp.vec3(0.42, 0.52, 0.53), wp.sqrt(foam) * 0.12)
    color[i] = water


@wp.kernel
def composite_surface_foam(
    water_depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    scene_depth: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    time_s: float,
    foam_strength: float,
):
    """Render foam as a distinct rough, opaque surface material."""
    i = wp.tid()
    z = water_depth[i]
    if z > 1.0e8 or z >= scene_depth[i]:
        return
    x = i % width
    y = i // width
    foam = wp.clamp(foam_field[i] * foam_strength, 0.0, 1.0)
    # Deterministic moving breakup prevents one uniform white blanket while
    # retaining temporal coherence from frame to frame.
    phase = wp.sin(float(x) * 0.173 + float(y) * 0.119 + time_s * 2.7)
    breakup = 0.74 + 0.26 * (phase * 0.5 + 0.5)
    coverage = wp.smoothstep(0.20, 0.96, wp.sqrt(foam) * breakup)
    if coverage <= 0.0:
        return
    foam_color = wp.vec3(0.80, 0.86, 0.83)
    highlight = wp.pow(wp.clamp(foam, 0.0, 1.0), 2.0)
    foam_color += wp.vec3(0.16, 0.14, 0.10) * highlight
    color[i] = wp.lerp(color[i], foam_color, coverage * 0.72)


@wp.kernel
def composite_volumetric_atmosphere(
    scene_depth: wp.array(dtype=float),
    water_depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3),
    cam: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
    time_s: float,
    fog_density: float,
    fog_height_falloff: float,
    mist_strength: float,
    sun_direction: wp.vec3,
    sky_turbidity: float,
    sky_intensity: float,
    sun_intensity: float,
):
    """Height fog plus screen-space water dust sourced by energetic foam."""
    i = wp.tid()
    x = i % width
    y = i // width
    z = wp.min(scene_depth[i], water_depth[i])
    value = color[i]
    camera_x_normalized = (float(x) + 0.5 - float(width) * 0.5) / focal
    camera_y_normalized = -(float(y) + 0.5 - float(height) * 0.5) / focal
    view_ray = wp.normalize(
        forward + right * camera_x_normalized + up * camera_y_normalized
    )
    atmospheric_radiance = physical_sky_radiance(
        view_ray, sun_direction, sky_turbidity, sky_intensity, sun_intensity
    )
    if z < 1.0e8:
        camera_x = (float(x) + 0.5 - float(width) * 0.5) * z / focal
        camera_y = -(float(y) + 0.5 - float(height) * 0.5) * z / focal
        world_y = cam[1] + right[1] * camera_x + up[1] * camera_y + forward[1] * z
        height_density = wp.exp(-wp.max(world_y, 0.0) / wp.max(fog_height_falloff, 1.0))
        fog_amount = 1.0 - wp.exp(-wp.min(z, 600.0) * fog_density * height_density)
        fog_color = atmospheric_radiance * 0.58 + wp.vec3(0.055, 0.065, 0.062)
        value = wp.lerp(value, fog_color, wp.clamp(fog_amount, 0.0, 0.55))

    mist = wp.clamp(foam_field[i], 0.0, 1.0) * 1.4
    samples = int(1)
    # Sparse multi-radius gather approximates a short volume around breaking
    # crests. It is deliberately independent of surface-foam coverage.
    for ring in range(3):
        radius = 3 + ring * 5
        for direction in range(8):
            ox = 0; oy = 0
            if direction == 0: ox = radius
            elif direction == 1: ox = radius; oy = radius
            elif direction == 2: oy = radius
            elif direction == 3: ox = -radius; oy = radius
            elif direction == 4: ox = -radius
            elif direction == 5: ox = -radius; oy = -radius
            elif direction == 6: oy = -radius
            else: ox = radius; oy = -radius
            px = x + ox; py = y + oy
            if px >= 0 and px < width and py >= 0 and py < height:
                sample_index = py * width + px
                sample_foam = wp.clamp(foam_field[sample_index], 0.0, 1.0)
                sample_water = water_depth[sample_index]
                if sample_water < 1.0e8:
                    radial_weight = 1.0 / (1.0 + float(ring) * 0.85)
                    mist += sample_foam * radial_weight
                samples += 1
    noise = 0.82 + 0.18 * wp.sin(float(x) * 0.071 - float(y) * 0.053 + time_s * 1.9)
    mist_amount = wp.clamp(mist / float(samples) * mist_strength * noise, 0.0, 0.22)
    mist_color = atmospheric_radiance * 0.42 + wp.vec3(0.20, 0.24, 0.235)
    color[i] = wp.lerp(value, mist_color, mist_amount)


@wp.kernel
def apply_directional_screen_shadows(
    scene_depth: wp.array(dtype=float),
    water_depth: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3),
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
):
    """Approximate cast sunlight shadows by ray marching the opaque depth map.

    Unlike ambient occlusion, this follows the projected world-space sun ray,
    so towers, shops, cars, trees, and debris cast consistently oriented
    shadows onto terrain and the reconstructed water surface.
    """
    i = wp.tid()
    x = i % width
    y = i // width
    z = wp.min(scene_depth[i], water_depth[i])
    if z > 1.0e8:
        return
    sun = wp.normalize(wp.vec3(-0.38, 0.82, -0.35))
    sun_x = wp.dot(sun, right)
    sun_y = wp.dot(sun, up)
    sun_z = wp.dot(sun, forward)
    camera_x = (float(x) + 0.5 - float(width) * 0.5) * z / focal
    camera_y = -(float(y) + 0.5 - float(height) * 0.5) * z / focal
    shadow = float(0.0)
    # A 3.5 m step is fine enough not to jump across facade depth while the
    # 84 m ray still covers the tallest tower's ground shadow.
    for step in range(1, 25):
        distance = float(step) * 3.5
        ray_z = z + sun_z * distance
        if ray_z > 0.2 and shadow < 0.5:
            ray_x = camera_x + sun_x * distance
            ray_y = camera_y + sun_y * distance
            px = int(float(width) * 0.5 + focal * ray_x / ray_z)
            py = int(float(height) * 0.5 - focal * ray_y / ray_z)
            if px >= 0 and px < width and py >= 0 and py < height:
                blocker = scene_depth[py * width + px]
                depth_gap = ray_z - blocker
                # A finite depth thickness reduces false shadows from an
                # unrelated foreground silhouette in screen space.
                if blocker < 1.0e8 and depth_gap > 0.25 and depth_gap < 100.0:
                    shadow = 1.0 - float(step - 1) / 30.0
    if shadow > 0.0:
        warm_ambient = wp.vec3(0.50, 0.56, 0.60)
        current = color[i]
        shaded = wp.vec3(
            current[0] * warm_ambient[0],
            current[1] * warm_ambient[1],
            current[2] * warm_ambient[2],
        )
        color[i] = wp.lerp(current, shaded, 0.82 * shadow)


@wp.kernel
def apply_cinematic_postprocess(
    scene_depth: wp.array(dtype=float),
    water_depth: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
):
    """Small screen-space AO, wet-contact darkening, distance haze and vignette."""
    i = wp.tid()
    x = i % width; y = i // width
    z = wp.min(scene_depth[i], water_depth[i])
    value = color[i]
    if z < 1.0e8:
        occlusion = float(0.0)
        samples = int(0)
        # Three radii retain tight wheel/tree/debris contact while also
        # grounding large wall slabs. Rotating the sparse ring per pixel avoids
        # the old square 3x3 halo without adding a random texture lookup.
        phase = (x * 13 + y * 7) & 3
        for ring in range(3):
            radius = 2 + ring * 3
            for sample in range(8):
                direction = (sample + phase) & 7
                ox = 0; oy = 0
                if direction == 0: ox = radius
                elif direction == 1: ox = radius; oy = radius
                elif direction == 2: oy = radius
                elif direction == 3: ox = -radius; oy = radius
                elif direction == 4: ox = -radius
                elif direction == 5: ox = -radius; oy = -radius
                elif direction == 6: oy = -radius
                else: ox = radius; oy = -radius
                px = x + ox; py = y + oy
                if px >= 0 and px < width and py >= 0 and py < height:
                    neighbour = wp.min(scene_depth[py * width + px], water_depth[py * width + px])
                    depth_delta = z - neighbour
                    if neighbour < 1.0e8 and depth_delta > 0.16:
                        range_weight = 1.0 - wp.smoothstep(0.0, 12.0 + float(ring) * 5.0, depth_delta)
                        occlusion += wp.clamp(depth_delta / (2.5 + float(ring) * 3.0), 0.0, 1.0) * range_weight
                    samples += 1
        ao = 1.0 - 0.52 * occlusion / float(wp.max(samples, 1))
        value *= ao
        if scene_depth[i] < 1.0e8 and water_depth[i] < 1.0e8:
            separation = water_depth[i] - scene_depth[i]
            wet = 1.0 - wp.smoothstep(0.0, 4.0, wp.abs(separation))
            wet_value = wp.vec3(value[0] * 0.58, value[1] * 0.66, value[2] * 0.68)
            value = wp.lerp(value, wet_value, wet * 0.32)
    nx = (float(x) + 0.5) / float(width) * 2.0 - 1.0
    ny = (float(y) + 0.5) / float(height) * 2.0 - 1.0
    vignette = 1.0 - 0.13 * wp.clamp(nx * nx + ny * ny - 0.35, 0.0, 1.0)
    color[i] = value * vignette


@wp.kernel
def apply_screen_space_indirect_lighting(
    source_color: wp.array(dtype=wp.vec3),
    scene_depth: wp.array(dtype=float),
    water_depth: wp.array(dtype=float),
    gbuffer_normal: wp.array(dtype=wp.vec3),
    output_color: wp.array(dtype=wp.vec3),
    cam: wp.vec3,
    camera_right: wp.vec3,
    camera_up: wp.vec3,
    camera_forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
    strength: float,
    radius_pixels: int,
):
    """Deterministic one-bounce diffuse screen-space light transport.

    This is deliberately conservative: it adds nearby colour bleeding and
    sky-lit fill without replacing the physically based direct/IBL terms.
    Reading from a frozen source buffer avoids order-dependent feedback.
    """
    i = wp.tid()
    x = i % width
    y = i // width
    z = wp.min(scene_depth[i], water_depth[i])
    base = source_color[i]
    if z > 1.0e8 or strength <= 0.0:
        output_color[i] = base
        return
    normal = gbuffer_normal[i]
    if wp.length(normal) < 0.25:
        normal = wp.vec3(0.0, 1.0, 0.0)
    else:
        normal = wp.normalize(normal)
    camera_x = (float(x) + 0.5 - float(width) * 0.5) * z / focal
    camera_y = -(float(y) + 0.5 - float(height) * 0.5) * z / focal
    world = cam + camera_right * camera_x + camera_up * camera_y + camera_forward * z
    accumulated = wp.vec3(0.0)
    weight_sum = float(0.0)
    phase = (x * 17 + y * 11) & 7
    for ring in range(2):
        sample_radius = wp.max(2, radius_pixels * (ring + 1) // 2)
        for sample in range(8):
            direction_index = (sample + phase) & 7
            ox = 0
            oy = 0
            if direction_index == 0: ox = sample_radius
            elif direction_index == 1: ox = sample_radius; oy = sample_radius
            elif direction_index == 2: oy = sample_radius
            elif direction_index == 3: ox = -sample_radius; oy = sample_radius
            elif direction_index == 4: ox = -sample_radius
            elif direction_index == 5: ox = -sample_radius; oy = -sample_radius
            elif direction_index == 6: oy = -sample_radius
            else: ox = sample_radius; oy = -sample_radius
            px = x + ox
            py = y + oy
            if px < 0 or px >= width or py < 0 or py >= height:
                continue
            neighbour_index = py * width + px
            neighbour_z = wp.min(scene_depth[neighbour_index], water_depth[neighbour_index])
            if neighbour_z > 1.0e8 or wp.abs(neighbour_z - z) > 36.0:
                continue
            neighbour_x = (float(px) + 0.5 - float(width) * 0.5) * neighbour_z / focal
            neighbour_y = -(float(py) + 0.5 - float(height) * 0.5) * neighbour_z / focal
            neighbour_world = (
                cam + camera_right * neighbour_x + camera_up * neighbour_y
                + camera_forward * neighbour_z
            )
            delta = neighbour_world - world
            distance = wp.length(delta)
            if distance < 0.05:
                continue
            ray = delta / distance
            neighbour_normal = gbuffer_normal[neighbour_index]
            if wp.length(neighbour_normal) < 0.25:
                neighbour_normal = wp.vec3(0.0, 1.0, 0.0)
            else:
                neighbour_normal = wp.normalize(neighbour_normal)
            receiver = wp.max(wp.dot(normal, ray), 0.0)
            emitter = wp.max(wp.dot(neighbour_normal, -ray), 0.0)
            # A small isotropic term approximates unresolved diffuse transport
            # at contact edges where the screen-space direction is tangent.
            geometry = 0.12 + 0.88 * wp.sqrt(receiver * emitter)
            attenuation = geometry / (1.0 + 0.10 * distance + 0.012 * distance * distance)
            accumulated += source_color[neighbour_index] * attenuation
            weight_sum += attenuation
    if weight_sum > 1.0e-5:
        bounced = accumulated / weight_sum
        # Limit energy injection; this pass is a one-bounce approximation and
        # must not become an emissive blur in highly fragmented regions.
        bounced = wp.min(bounced, wp.vec3(2.5, 2.5, 2.5))
        output_color[i] = base + bounced * wp.clamp(strength, 0.0, 0.45)
    else:
        output_color[i] = base
