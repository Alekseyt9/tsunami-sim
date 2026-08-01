"""CUDA water, contact, fracture and rendering kernels for DELUGE V3.

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
    return 3.0e8       # reinforced concrete / masonry


@wp.func
def material_failure_strain(material: int) -> float:
    if material == 2:
        return 0.012
    if material == 3:
        return 0.11
    return 0.032


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
                force -= normal * (3.0e6 * penetration + 2600.0 * wp.min(closing, 0.0))

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
):
    i = wp.tid()
    if fixed[i] != 0:
        return
    vi = v[i] + acceleration[i] * dt
    if kind[i] != 0:
        vi *= wp.pow(0.9993, dt * 1000.0)
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
    old_count: int,
    capacity: int,
    fine_radius: float,
    refine_z: float,
):
    i = wp.tid()
    if i >= old_count or kind[i] != 0 or radius[i] <= fine_radius * 1.25 or x[i][2] < refine_z:
        return

    base = wp.atomic_add(count, 0, 7)
    if base + 6 >= capacity:
        return

    parent_x = x[i]
    parent_v = v[i]
    child_r = radius[i] * 0.5
    child_m = mass[i] * 0.125
    child_vol = volume[i] * 0.125
    # Child centers must be one child diameter apart. The former 0.55
    # multiplier overlapped children and produced a non-physical pressure burst.
    offsets = wp.vec3(-1.0, -1.0, -1.0) * child_r
    x[i] = parent_x + offsets
    radius[i] = child_r; mass[i] = child_m; volume[i] = child_vol
    rho_reference[i] = 0.0

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


@wp.kernel
def clear_render(depth: wp.array(dtype=float), color: wp.array(dtype=wp.vec3), width: int, height: int):
    i = wp.tid()
    y = i // width
    t = float(y) / float(height)
    depth[i] = 1.0e9
    color[i] = wp.vec3(0.08 + 0.16 * (1.0 - t), 0.14 + 0.18 * (1.0 - t), 0.18 + 0.20 * (1.0 - t))


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
    base = wp.lerp(base, wp.vec3(0.22, 0.045, 0.025), damage[i])

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
    depth: wp.array(dtype=float), foam_field: wp.array(dtype=float),
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
                sphere_z = p[2] - world_r * wp.sqrt(wp.max(0.0, 1.0 - rr / (rp * rp)))
                wp.atomic_min(depth, py * width + px, sphere_z)
                # Foam comes from energetic vertical/turbulent motion, not the
                # silhouette of every particle splat.
                speed = wp.length(v[i])
                foam = wp.smoothstep(2.5, 11.0, wp.abs(v[i][1]))
                foam += 0.45 * wp.smoothstep(20.0, 45.0, speed)
                wp.atomic_max(foam_field, py * width + px, wp.min(foam, 1.0))


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
def shade_water_surface(
    water_depth: wp.array(dtype=float), foam_field: wp.array(dtype=float),
    scene_depth: wp.array(dtype=float), color: wp.array(dtype=wp.vec3),
    width: int, height: int,
):
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
    dx = zr - zl; dy = zd - zu
    normal = wp.normalize(wp.vec3(-dx * 7.0, dy * 7.0, 1.0))
    fresnel = wp.pow(1.0 - wp.clamp(normal[2], 0.0, 1.0), 3.0)
    light = wp.clamp(wp.dot(normal, wp.normalize(wp.vec3(-0.35, 0.55, 0.76))), 0.0, 1.0)
    foam = wp.clamp(foam_field[i], 0.0, 1.0)
    water = wp.lerp(wp.vec3(0.018, 0.15, 0.21), wp.vec3(0.18, 0.39, 0.45), fresnel)
    water += wp.vec3(0.18, 0.24, 0.25) * light * 0.28
    water = wp.lerp(water, wp.vec3(0.78, 0.86, 0.84), foam * 0.78)
    behind = color[i]
    alpha = 0.76 + foam * 0.18
    color[i] = wp.lerp(behind, water, alpha)
