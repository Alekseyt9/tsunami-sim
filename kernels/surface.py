"""Sparse free-surface classification and anisotropic water rasterization."""

import warp as wp

from kernels.base import project_point


@wp.kernel
def classify_water_surface(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    rho_reference: wp.array(dtype=float),
    surface_mask: wp.array(dtype=wp.int32),
    surface_normal: wp.array(dtype=wp.vec3),
    foam_strength: wp.array(dtype=float),
    water_phase: wp.array(dtype=wp.int32),
    phase_candidate: wp.array(dtype=wp.int32),
    phase_candidate_age: wp.array(dtype=wp.int32),
    phase_transitions: wp.array(dtype=wp.int32),
    query_radius: float,
    minimum_neighbours: int,
    sheet_minimum_neighbours: int,
    sheet_thickness_ratio: float,
    droplet_maximum_neighbours: int,
    droplet_enter_classifications: int,
    droplet_exit_classifications: int,
    foam_decay: float,
    phase_separation_enabled: int,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if kind[i] != 0:
        surface_mask[i] = 0
        foam_strength[i] = 0.0
        water_phase[i] = 0
        phase_candidate[i] = 0
        phase_candidate_age[i] = 0
        return
    xi = x[i]
    vi = v[i]
    positive_x = int(0); negative_x = int(0)
    positive_y = int(0); negative_y = int(0)
    positive_z = int(0); negative_z = int(0)
    neighbours = int(0)
    gradient = wp.vec3(0.0)
    vorticity = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, xi, query_radius)
    for j in query:
        if j == i or kind[j] != 0:
            continue
        delta = x[j] - xi
        distance = wp.length(delta)
        support = 5.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support:
            continue
        neighbours += 1
        absolute = wp.vec3(wp.abs(delta[0]), wp.abs(delta[1]), wp.abs(delta[2]))
        if absolute[0] >= absolute[1] and absolute[0] >= absolute[2]:
            if delta[0] > 0.0: positive_x = 1
            else: negative_x = 1
        elif absolute[1] >= absolute[2]:
            if delta[1] > 0.0: positive_y = 1
            else: negative_y = 1
        else:
            if delta[2] > 0.0: positive_z = 1
            else: negative_z = 1
        weight = wp.max(1.0 - distance / support, 0.0)
        direction = delta / distance
        gradient -= direction * weight
        vorticity += wp.cross(v[j] - vi, direction) * (weight / wp.max(distance, 0.05))

    closed = positive_x * negative_x * positive_y * negative_y * positive_z * negative_z
    is_surface = closed == 0 or neighbours < minimum_neighbours
    candidate = int(0)
    if is_surface:
        surface_mask[i] = 1
        gradient_length = wp.length(gradient)
        normal = wp.vec3(0.0, 1.0, 0.0)
        if gradient_length > 1.0e-5:
            normal = gradient / gradient_length
        surface_normal[i] = normal
        vort = wp.length(vorticity) / float(wp.max(neighbours, 1))
        vertical = wp.abs(vi[1])
        # Calm bulk flow remains clear. Foam is reserved for genuine rotation,
        # overturning and detached energetic spray.
        foam_source = 0.70 * wp.smoothstep(2.0, 9.0, vort)
        foam_source += 0.55 * wp.smoothstep(4.0, 13.0, vertical)

        # A thin sheet has broad tangential support but little thickness along
        # its free-surface normal.  A second local pass is substantially more
        # stable than classifying every low-neighbour surface sample as spray:
        # the ordinary top of the wave has a deep inward normal extent, while
        # an overturning lamella is genuinely thin on both sides.
        if phase_separation_enabled != 0:
            normal_extent = float(0.0)
            tangent_extent = float(0.0)
            extent_query = wp.hash_grid_query(grid, xi, query_radius)
            for j in extent_query:
                if j == i or kind[j] != 0:
                    continue
                delta = x[j] - xi
                distance = wp.length(delta)
                support = 5.0 * wp.max(radius[i], radius[j])
                if distance <= 1.0e-5 or distance >= support:
                    continue
                along_normal = wp.abs(wp.dot(delta, normal))
                tangent = wp.sqrt(wp.max(distance * distance - along_normal * along_normal, 0.0))
                normal_extent = wp.max(normal_extent, along_normal)
                tangent_extent = wp.max(tangent_extent, tangent)

            isolated = neighbours <= droplet_maximum_neighbours
            thin = (
                neighbours >= sheet_minimum_neighbours
                and tangent_extent > radius[i] * 2.5
                and normal_extent < tangent_extent * sheet_thickness_ratio
            )
            if isolated:
                candidate = 2
                # A detached drop is not automatically whitewater.  Reserve
                # strong foam for fast drops that are also moving vertically;
                # otherwise distant spray becomes a field of bright capsules.
                spray_energy = wp.smoothstep(8.0, 18.0, wp.length(vi))
                spray_energy *= wp.smoothstep(2.0, 10.0, wp.abs(vi[1]))
                foam_source = wp.max(foam_source, 0.08 + 0.48 * spray_energy)
            elif thin:
                candidate = 1
                foam_source = wp.max(foam_source, 0.15 * wp.smoothstep(4.0, 12.0, wp.length(vi)))
        foam_strength[i] = wp.clamp(wp.max(foam_strength[i] * foam_decay, foam_source), 0.0, 1.0)
    else:
        surface_mask[i] = 0
        surface_normal[i] = wp.vec3(0.0, 1.0, 0.0)
        foam_strength[i] *= foam_decay

    if phase_separation_enabled == 0:
        if water_phase[i] == 2:
            rho_reference[i] = 0.0
        water_phase[i] = 0
        phase_candidate[i] = 0
        phase_candidate_age[i] = 0
        return

    # Core/sheet switches are harmless representation changes.  Entering or
    # leaving the ballistic mode changes the force model, so require a stable
    # candidate for several output classifications in both directions.
    current = water_phase[i]
    if current == 2 or candidate == 2:
        if phase_candidate[i] == candidate:
            phase_candidate_age[i] += 1
        else:
            phase_candidate[i] = candidate
            phase_candidate_age[i] = 1
        required = droplet_enter_classifications
        if current == 2:
            required = droplet_exit_classifications
        if phase_candidate_age[i] >= required:
            if current != candidate:
                if candidate == 2:
                    wp.atomic_add(phase_transitions, 0, 1)
                elif current == 2:
                    wp.atomic_add(phase_transitions, 1, 1)
                if candidate == 1:
                    wp.atomic_add(phase_transitions, 2, 1)
                if current == 1 and candidate != 1:
                    wp.atomic_add(phase_transitions, 3, 1)
            water_phase[i] = candidate
            phase_candidate_age[i] = 0
            if current == 2 and candidate != 2:
                # Rejoining SPH must recalibrate the density normalization at
                # the receiving free surface instead of retaining the
                # ballistic sentinel written by the density kernel.
                rho_reference[i] = 0.0
    else:
        if current != candidate:
            if candidate == 1:
                wp.atomic_add(phase_transitions, 2, 1)
            if current == 1 and candidate != 1:
                wp.atomic_add(phase_transitions, 3, 1)
        water_phase[i] = candidate
        phase_candidate[i] = candidate
        phase_candidate_age[i] = 0


@wp.kernel
def build_water_phase_masks(
    kind: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    connected_mask: wp.array(dtype=wp.int32),
    sheet_mask: wp.array(dtype=wp.int32),
    droplet_mask: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    connected_mask[i] = 0
    sheet_mask[i] = 0
    droplet_mask[i] = 0
    if kind[i] != 0 or surface_mask[i] == 0:
        return
    phase = water_phase[i]
    if phase == 2:
        droplet_mask[i] = 1
    elif phase == 1:
        sheet_mask[i] = 1
    else:
        connected_mask[i] = 1


@wp.kernel
def raster_anisotropic_water_depth(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    surface_normal: wp.array(dtype=wp.vec3),
    particle_foam: wp.array(dtype=float),
    water_phase: wp.array(dtype=wp.int32),
    depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    cam: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
    tangent_scale: float,
    normal_scale: float,
):
    i = wp.tid()
    if kind[i] != 0 or surface_mask[i] == 0:
        return
    p = project_point(x[i], cam, right, up, forward, focal, width, height)
    if p[2] <= 0.1:
        return
    normal = surface_normal[i]
    screen_normal = wp.vec2(wp.dot(normal, right), -wp.dot(normal, up))
    normal_length = wp.length(screen_normal)
    if normal_length > 1.0e-5:
        screen_normal /= normal_length
    else:
        screen_normal = wp.vec2(0.0, 1.0)
    screen_tangent = wp.vec2(-screen_normal[1], screen_normal[0])
    tangent_radius = wp.clamp(focal * radius[i] * tangent_scale / p[2], 1.35, 14.0)
    normal_radius = wp.clamp(focal * radius[i] * normal_scale / p[2], 1.1, 10.0)
    foam = particle_foam[i]
    phase = water_phase[i]
    if phase == 1:
        # An overturning lamella is a broad, thin optical sheet, not a row of
        # circular blobs.  Preserve its measured normal and flatten only the
        # render footprint; its SPH state is unchanged.
        tangent_radius = wp.clamp(tangent_radius * 1.55, 2.0, 14.0)
        normal_radius = wp.clamp(normal_radius * 0.48, 0.75, 4.0)
    if phase == 2 or foam > 0.35:
        # Energetic detached spray is motion-blurred along its trajectory.
        # Rendering it with the near-isotropic free-surface footprint makes
        # every sample look like a solid ball; a thin velocity-aligned streak
        # reads as spray/foam while preserving its simulated position.
        travel = wp.vec2(wp.dot(v[i], right), -wp.dot(v[i], up))
        travel_length = wp.length(travel)
        if travel_length > 1.0e-4:
            screen_tangent = travel / travel_length
            screen_normal = wp.vec2(-screen_tangent[1], screen_tangent[0])
        spray = wp.max(foam, 0.20 if phase == 2 else 0.0)
        speed_stretch = 1.0 + spray * wp.clamp(wp.length(v[i]) / 12.0, 0.0, 1.0)
        tangent_radius = wp.clamp(tangent_radius * speed_stretch, 1.35, 8.0)
        normal_radius = wp.clamp(normal_radius * (0.82 - 0.12 * spray), 0.7, 5.5)
    cx = int(p[0]); cy = int(p[1])
    for oy in range(-14, 15):
        for ox in range(-14, 15):
            pixel_offset = wp.vec2(float(ox), float(oy))
            tangent_coordinate = wp.dot(pixel_offset, screen_tangent) / tangent_radius
            normal_coordinate = wp.dot(pixel_offset, screen_normal) / normal_radius
            ellipse = tangent_coordinate * tangent_coordinate + normal_coordinate * normal_coordinate
            px = cx + ox; py = cy + oy
            if px >= 0 and px < width and py >= 0 and py < height and ellipse <= 1.0:
                surface_depth = p[2] - radius[i] * normal_scale * wp.sqrt(wp.max(0.0, 1.0 - ellipse))
                index = py * width + px
                wp.atomic_min(depth, index, surface_depth)
                wp.atomic_max(foam_field, index, foam)


@wp.kernel
def splat_sparse_surface_field(
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    field: wp.array3d(dtype=float),
    lower: wp.vec3,
    voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
):
    i = wp.tid()
    if kind[i] != 0 or surface_mask[i] == 0:
        return
    particle = x[i]
    center = (particle - lower) / voxel_size
    cx = int(wp.round(center[0])); cy = int(wp.round(center[1])); cz = int(wp.round(center[2]))
    sigma = wp.max(radius[i] * 1.45, voxel_size * 0.72)
    inverse_two_sigma2 = 0.5 / (sigma * sigma)
    nominal_spacing = radius[i] * 2.0
    # Gaussian quadrature normalized by kernel volume.  A uniform fine and a
    # uniform coarse lattice therefore reconstruct the same scalar density at
    # their interface instead of leaving a visible resolution ridge.
    volume_weight = wp.pow(nominal_spacing / sigma, 3.0)
    for dz in range(-2, 3):
        iz = cz + dz
        if iz < 0 or iz >= nz:
            continue
        for dy in range(-2, 3):
            iy = cy + dy
            if iy < 0 or iy >= ny:
                continue
            for dx in range(-2, 3):
                ix = cx + dx
                if ix < 0 or ix >= nx:
                    continue
                node = lower + wp.vec3(float(ix), float(iy), float(iz)) * voxel_size
                distance2 = wp.length_sq(node - particle)
                contribution = wp.exp(-distance2 * inverse_two_sigma2) * volume_weight
                if contribution > 0.005:
                    wp.atomic_add(field, ix, iy, iz, contribution)


@wp.kernel
def smooth_sparse_field_axis(
    source: wp.array3d(dtype=float),
    target: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    axis: int,
):
    ix, iy, iz = wp.tid()
    total = float(0.0)
    weight_sum = float(0.0)
    for offset in range(-2, 3):
        sx = ix; sy = iy; sz = iz
        if axis == 0: sx = ix + offset
        elif axis == 1: sy = iy + offset
        else: sz = iz + offset
        if sx >= 0 and sx < nx and sy >= 0 and sy < ny and sz >= 0 and sz < nz:
            weight = 1.0
            if offset == -1 or offset == 1: weight = 4.0
            elif offset == 0: weight = 6.0
            total += source[sx, sy, sz] * weight
            weight_sum += weight
    target[ix, iy, iz] = total / wp.max(weight_sum, 1.0)


@wp.kernel
def blend_sparse_fields(
    current: wp.array3d(dtype=float),
    previous: wp.array3d(dtype=float),
    previous_weight: float,
):
    """Temporally filter equal-domain scalar fields before meshing."""
    ix, iy, iz = wp.tid()
    current[ix, iy, iz] = (
        current[ix, iy, iz] * (1.0 - previous_weight)
        + previous[ix, iy, iz] * previous_weight
    )


@wp.func
def mesh_edge(a: wp.vec3, b: wp.vec3, px: float, py: float) -> float:
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


@wp.kernel
def raster_water_mesh_depth(
    vertex: wp.array(dtype=wp.vec3),
    index: wp.array(dtype=wp.int32),
    depth: wp.array(dtype=float),
    cam: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
):
    triangle = wp.tid()
    ia = index[triangle * 3]
    ib = index[triangle * 3 + 1]
    ic = index[triangle * 3 + 2]
    a = project_point(vertex[ia], cam, right, up, forward, focal, width, height)
    b = project_point(vertex[ib], cam, right, up, forward, focal, width, height)
    c = project_point(vertex[ic], cam, right, up, forward, focal, width, height)
    if a[2] <= 0.1 or b[2] <= 0.1 or c[2] <= 0.1:
        return
    min_x = wp.max(int(wp.floor(wp.min(a[0], wp.min(b[0], c[0])))), 0)
    max_x = wp.min(int(wp.ceil(wp.max(a[0], wp.max(b[0], c[0])))), width - 1)
    min_y = wp.max(int(wp.floor(wp.min(a[1], wp.min(b[1], c[1])))), 0)
    max_y = wp.min(int(wp.ceil(wp.max(a[1], wp.max(b[1], c[1])))), height - 1)
    # Degenerate or pathologically large projected triangles are rejected;
    # regular marching-cubes cells project to compact screen regions.
    if max_x < min_x or max_y < min_y or max_x - min_x > 96 or max_y - min_y > 96:
        return
    area = mesh_edge(a, b, c[0], c[1])
    if wp.abs(area) < 1.0e-7:
        return
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            fx = float(px) + 0.5; fy = float(py) + 0.5
            w0 = mesh_edge(b, c, fx, fy) / area
            w1 = mesh_edge(c, a, fx, fy) / area
            w2 = 1.0 - w0 - w1
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                wp.atomic_min(depth, py * width + px, z)


@wp.kernel
def raster_water_mesh_foam(
    x: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    particle_foam: wp.array(dtype=float),
    water_depth: wp.array(dtype=float),
    foam_field: wp.array(dtype=float),
    cam: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    width: int,
    height: int,
):
    i = wp.tid()
    foam = particle_foam[i]
    if kind[i] != 0 or surface_mask[i] == 0 or foam < 0.01:
        return
    p = project_point(x[i], cam, right, up, forward, focal, width, height)
    if p[2] <= 0.1:
        return
    pixel_radius = int(wp.clamp(focal * radius[i] * 2.2 / p[2], 1.0, 8.0))
    cx = int(p[0]); cy = int(p[1])
    for oy in range(-8, 9):
        for ox in range(-8, 9):
            px = cx + ox; py = cy + oy
            if px >= 0 and px < width and py >= 0 and py < height:
                if ox * ox + oy * oy <= pixel_radius * pixel_radius:
                    index = py * width + px
                    if wp.abs(water_depth[index] - p[2]) < radius[i] * 3.0:
                        wp.atomic_max(foam_field, index, foam)
