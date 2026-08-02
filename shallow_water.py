"""GPU far-field shallow-water model and conservative SPH interface coupling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp


@wp.func
def _flux_x(q: wp.vec3, gravity: float) -> wp.vec3:
    h = wp.max(q[0], 1.0e-5)
    u = q[1] / h
    w = q[2] / h
    return wp.vec3(q[1], q[1] * u + 0.5 * gravity * h * h, q[1] * w)


@wp.func
def _flux_z(q: wp.vec3, gravity: float) -> wp.vec3:
    h = wp.max(q[0], 1.0e-5)
    u = q[1] / h
    w = q[2] / h
    return wp.vec3(q[2], q[2] * u, q[2] * w + 0.5 * gravity * h * h)


@wp.func
def _rusanov_x(left: wp.vec3, right: wp.vec3, gravity: float) -> wp.vec3:
    hl = wp.max(left[0], 1.0e-5)
    hr = wp.max(right[0], 1.0e-5)
    speed = wp.max(
        wp.abs(left[1] / hl) + wp.sqrt(gravity * hl),
        wp.abs(right[1] / hr) + wp.sqrt(gravity * hr),
    )
    return 0.5 * (_flux_x(left, gravity) + _flux_x(right, gravity)) - 0.5 * speed * (right - left)


@wp.func
def _rusanov_z(lower: wp.vec3, upper: wp.vec3, gravity: float) -> wp.vec3:
    hl = wp.max(lower[0], 1.0e-5)
    hr = wp.max(upper[0], 1.0e-5)
    speed = wp.max(
        wp.abs(lower[2] / hl) + wp.sqrt(gravity * hl),
        wp.abs(upper[2] / hr) + wp.sqrt(gravity * hr),
    )
    return 0.5 * (_flux_z(lower, gravity) + _flux_z(upper, gravity)) - 0.5 * speed * (upper - lower)


@wp.kernel
def advance_shallow_water(
    current: wp.array2d(dtype=wp.vec3),
    updated: wp.array2d(dtype=wp.vec3),
    nx: int,
    nz: int,
    cell_size: float,
    dt: float,
    gravity: float,
    bed_drag: float,
    dry_depth: float,
):
    ix, iz = wp.tid()
    center = current[ix, iz]
    left = current[wp.max(ix - 1, 0), iz]
    right = current[wp.min(ix + 1, nx - 1), iz]
    back = current[ix, wp.max(iz - 1, 0)]
    front = current[ix, wp.min(iz + 1, nz - 1)]
    # Reflective ghost states match the particle solver's tank boundaries and
    # prevent a copied non-zero boundary velocity from becoming an artificial
    # infinite mass source.
    if ix == 0:
        left = wp.vec3(center[0], -center[1], center[2])
    if ix == nx - 1:
        right = wp.vec3(center[0], -center[1], center[2])
    if iz == 0:
        back = wp.vec3(center[0], center[1], -center[2])
    if iz == nz - 1:
        front = wp.vec3(center[0], center[1], -center[2])
    next_value = center - (dt / cell_size) * (
        _rusanov_x(center, right, gravity) - _rusanov_x(left, center, gravity)
        + _rusanov_z(center, front, gravity) - _rusanov_z(back, center, gravity)
    )
    h = wp.max(next_value[0], 0.0)
    if h <= dry_depth:
        updated[ix, iz] = wp.vec3(0.0, 0.0, 0.0)
    else:
        damping = wp.max(0.0, 1.0 - bed_drag * dt)
        updated[ix, iz] = wp.vec3(h, next_value[1] * damping, next_value[2] * damping)


@wp.kernel
def couple_sph_interface(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    acceleration: wp.array(dtype=wp.vec3),
    shallow: wp.array2d(dtype=wp.vec3),
    exchange_x: wp.array2d(dtype=float),
    exchange_z: wp.array2d(dtype=float),
    lower_x: float,
    lower_z: float,
    cell_size: float,
    nx: int,
    nz: int,
    interface_z: float,
    coupling_width: float,
    relaxation_rate: float,
    dt: float,
):
    i = wp.tid()
    if kind[i] != 0 or x[i][2] < interface_z or x[i][2] >= interface_z + coupling_width:
        return
    ix = wp.clamp(int(wp.floor((x[i][0] - lower_x) / cell_size)), 0, nx - 1)
    iz = wp.clamp(int(wp.floor((x[i][2] - lower_z) / cell_size)), 0, nz - 1)
    state = shallow[ix, iz]
    if state[0] <= 1.0e-4:
        return
    weight = 1.0 - (x[i][2] - interface_z) / coupling_width
    rate = relaxation_rate * weight * weight
    target_x = state[1] / state[0]
    target_z = state[2] / state[0]
    ax = (target_x - v[i][0]) * rate
    az = (target_z - v[i][2]) * rate
    acceleration[i] = acceleration[i] + wp.vec3(ax, 0.0, az)
    # The equal and opposite horizontal impulse is returned to the 2D cell.
    wp.atomic_add(exchange_x, ix, iz, -mass[i] * ax * dt)
    wp.atomic_add(exchange_z, ix, iz, -mass[i] * az * dt)


@wp.kernel
def apply_exchange_impulse(
    shallow: wp.array2d(dtype=wp.vec3),
    exchange_volume: wp.array2d(dtype=float),
    exchange_x: wp.array2d(dtype=float),
    exchange_z: wp.array2d(dtype=float),
    cell_area: float,
    density_times_area: float,
):
    ix, iz = wp.tid()
    state = shallow[ix, iz]
    shallow[ix, iz] = wp.vec3(
        wp.max(0.0, state[0] + exchange_volume[ix, iz] / cell_area),
        state[1] + exchange_x[ix, iz] / density_times_area,
        state[2] + exchange_z[ix, iz] / density_times_area,
    )
    exchange_volume[ix, iz] = 0.0
    exchange_x[ix, iz] = 0.0
    exchange_z[ix, iz] = 0.0


@wp.kernel
def emit_sph_interface_particles(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    material: wp.array(dtype=wp.int32),
    building_id: wp.array(dtype=wp.int32),
    structural_class: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    impact_impulse: wp.array(dtype=float),
    local_impact_active: wp.array(dtype=wp.int32),
    rho_reference: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    base_fixed: wp.array(dtype=wp.int32),
    fragment_id: wp.array(dtype=wp.int32),
    normal_axis: wp.array(dtype=wp.int32),
    time_level: wp.array(dtype=wp.int32),
    time_active: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    surface_normal: wp.array(dtype=wp.vec3),
    foam_strength: wp.array(dtype=float),
    fluid_group_id: wp.array(dtype=wp.int32),
    shallow: wp.array2d(dtype=wp.vec3),
    exchange_volume: wp.array2d(dtype=float),
    exchange_x: wp.array2d(dtype=float),
    exchange_z: wp.array2d(dtype=float),
    count: wp.array(dtype=wp.int32),
    old_count: int,
    capacity: int,
    emitter_nx: int,
    emitter_ny: int,
    lower_x: float,
    lower_z: float,
    interface_z: float,
    cell_size: float,
    shallow_nx: int,
    shallow_nz: int,
    particle_spacing: float,
    rest_density: float,
    minimum_emission_velocity: float,
):
    emitter_x, emitter_y = wp.tid()
    if emitter_x >= emitter_nx or emitter_y >= emitter_ny:
        return
    px = lower_x + (float(emitter_x) + 0.5) * particle_spacing
    py = (float(emitter_y) + 0.5) * particle_spacing
    pz = interface_z + 0.55 * particle_spacing
    sx = wp.clamp(int(wp.floor((px - lower_x) / cell_size)), 0, shallow_nx - 1)
    sz = wp.clamp(int(wp.floor((interface_z - lower_z) / cell_size)), 0, shallow_nz - 1)
    state = shallow[sx, sz]
    if state[0] <= py + 0.5 * particle_spacing:
        return
    position = wp.vec3(px, py, pz)
    occupied = int(0)
    query = wp.hash_grid_query(grid, position, particle_spacing * 0.62)
    for neighbour in query:
        if neighbour < old_count and kind[neighbour] == 0:
            if wp.length_sq(x[neighbour] - position) < particle_spacing * particle_spacing * 0.38:
                occupied = 1
                break
    if occupied != 0:
        return
    velocity_x = state[1] / wp.max(state[0], 1.0e-5)
    velocity_z = state[2] / wp.max(state[0], 1.0e-5)
    if velocity_z < minimum_emission_velocity:
        return
    target = wp.atomic_add(count, 0, 1)
    if target >= capacity:
        return
    particle_volume = particle_spacing * particle_spacing * particle_spacing
    particle_mass = particle_volume * rest_density
    velocity = wp.vec3(velocity_x, 0.0, velocity_z)
    x[target] = position
    rest_x[target] = position
    v[target] = velocity
    radius[target] = 0.5 * particle_spacing
    mass[target] = particle_mass
    volume[target] = particle_volume
    kind[target] = 0
    material[target] = 0
    building_id[target] = -1
    structural_class[target] = 0
    fixed[target] = 0
    damage[target] = 0.0
    impact_impulse[target] = 0.0
    local_impact_active[target] = 0
    rho_reference[target] = 0.0
    rho[target] = rest_density
    acceleration[target] = wp.vec3(0.0)
    solid_force[target] = wp.vec3(0.0)
    base_fixed[target] = 0
    fragment_id[target] = -1
    normal_axis[target] = -1
    time_level[target] = 0
    time_active[target] = 1
    surface_mask[target] = 0
    surface_normal[target] = wp.vec3(0.0)
    foam_strength[target] = 0.0
    fluid_group_id[target] = -1
    wp.atomic_add(exchange_volume, sx, sz, -particle_volume)
    wp.atomic_add(exchange_x, sx, sz, -particle_mass * velocity_x)
    wp.atomic_add(exchange_z, sx, sz, -particle_mass * velocity_z)


@wp.kernel
def mark_sph_return_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    keep: wp.array(dtype=wp.int32),
    exchange_volume: wp.array2d(dtype=float),
    exchange_x: wp.array2d(dtype=float),
    exchange_z: wp.array2d(dtype=float),
    merged_volume: wp.array(dtype=float),
    lower_x: float,
    lower_z: float,
    interface_z: float,
    cell_size: float,
    shallow_nx: int,
    shallow_nz: int,
    minimum_return_speed: float,
    forced_capture_depth: float,
):
    i = wp.tid()
    position = x[i]
    velocity = v[i]
    returning = (
        kind[i] == 0
        and position[2] < interface_z
        and (
            velocity[2] <= -minimum_return_speed
            or position[2] <= interface_z - forced_capture_depth
        )
    )
    if not returning:
        keep[i] = 1
        return
    keep[i] = 0
    ix = wp.clamp(int(wp.floor((position[0] - lower_x) / cell_size)), 0, shallow_nx - 1)
    iz = wp.clamp(int(wp.floor((interface_z - lower_z) / cell_size)), 0, shallow_nz - 1)
    wp.atomic_add(exchange_volume, ix, iz, volume[i])
    wp.atomic_add(exchange_x, ix, iz, mass[i] * velocity[0])
    wp.atomic_add(exchange_z, ix, iz, mass[i] * velocity[2])
    wp.atomic_add(merged_volume, 0, volume[i])


@wp.kernel
def compact_float_particles(
    source: wp.array(dtype=float),
    target: wp.array(dtype=float),
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if keep[i] != 0:
        target[offsets[i]] = source[i]


@wp.kernel
def compact_int_particles(
    source: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if keep[i] != 0:
        target[offsets[i]] = source[i]


@wp.kernel
def compact_vec3_particles(
    source: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if keep[i] != 0:
        target[offsets[i]] = source[i]


@wp.kernel
def compact_vec3_components(
    source: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if keep[i] != 0:
        destination = offsets[i]
        target[destination, 0] = source[i, 0]
        target[destination, 1] = source[i, 1]
        target[destination, 2] = source[i, 2]


@wp.kernel
def remap_particle_indices(
    indices: wp.array(dtype=wp.int32),
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    old_index = indices[i]
    if old_index >= 0 and keep[old_index] != 0:
        indices[i] = offsets[old_index]
    else:
        indices[i] = -1


class ShallowWaterFarField:
    """Coarse 2D conservative wave field behind the local 3D SPH window."""

    def __init__(self, cfg: dict, device, checkpoint: Path | None = None):
        policy = cfg["v3"].get("shallow_water", {})
        self.enabled = bool(policy.get("enabled", False))
        self.device = device
        self.cfg = policy
        self.cell_size = float(policy.get("cell_size", 2.0))
        self.lower_x = -0.5 * float(cfg["domain_width"])
        self.lower_z = float(cfg["reservoir_z_min"])
        self.upper_z = float(cfg["domain_z_max"])
        self.nx = max(3, int(math.ceil(float(cfg["domain_width"]) / self.cell_size)))
        self.nz = max(3, int(math.ceil((self.upper_z - self.lower_z) / self.cell_size)))
        self.interface_z = float(policy.get("sph_z_min", cfg["reservoir_z_min"]))
        self.update_interval = float(policy.get("update_interval", 0.008))
        self.accumulated_dt = 0.0
        host = np.zeros((self.nx, self.nz, 3), dtype=np.float32)
        depth = float(cfg["water_depth"])
        crest = float(cfg["wave_height"])
        wave_speed = float(cfg["wave_speed"])
        background = float(cfg.get("background_current", 0.0))
        reservoir_front = float(cfg["reservoir_z_max"])
        for iz in range(self.nz):
            z = self.lower_z + (iz + 0.5) * self.cell_size
            if z >= reservoir_front:
                continue
            elevation = crest * math.exp(-((z - reservoir_front + 5.0) / 7.5) ** 2)
            h = depth + elevation
            velocity_z = background + wave_speed * elevation / max(h, 1.0e-6)
            host[:, iz, 0] = h
            host[:, iz, 2] = h * velocity_z
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if "shallow_water_state" in saved and saved["shallow_water_state"].shape == host.shape:
                    host = saved["shallow_water_state"].astype(np.float32, copy=True)
                if "shallow_water_accumulated_dt" in saved:
                    self.accumulated_dt = float(saved["shallow_water_accumulated_dt"])
        self.state = wp.array(host, dtype=wp.vec3, device=device)
        self.updated = wp.zeros((self.nx, self.nz), dtype=wp.vec3, device=device)
        self.exchange_x = wp.zeros((self.nx, self.nz), dtype=float, device=device)
        self.exchange_z = wp.zeros((self.nx, self.nz), dtype=float, device=device)
        self.exchange_volume = wp.zeros((self.nx, self.nz), dtype=float, device=device)
        self.emitted_particles_total = 0
        self.emitted_volume_total = 0.0
        self.merged_particles_total = 0
        self.merged_volume_total = 0.0
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if "shallow_emitted_particles_total" in saved:
                    self.emitted_particles_total = int(saved["shallow_emitted_particles_total"])
                if "shallow_emitted_volume_total" in saved:
                    self.emitted_volume_total = float(saved["shallow_emitted_volume_total"])
                if "shallow_merged_particles_total" in saved:
                    self.merged_particles_total = int(saved["shallow_merged_particles_total"])
                if "shallow_merged_volume_total" in saved:
                    self.merged_volume_total = float(saved["shallow_merged_volume_total"])

    def couple(self, arrays: dict, count: int, dt: float):
        if not self.enabled:
            return
        wp.launch(
            couple_sph_interface, dim=count,
            inputs=[arrays["x"][:count], arrays["v"][:count], arrays["mass"][:count],
                    arrays["kind"][:count], arrays["acceleration"][:count], self.state,
                    self.exchange_x, self.exchange_z, self.lower_x, self.lower_z,
                    self.cell_size, self.nx, self.nz, self.interface_z,
                    float(self.cfg.get("coupling_width", 4.0)),
                    float(self.cfg.get("velocity_relaxation_rate", 1.5)), dt],
            device=self.device,
        )

    def advance(self, dt: float, rest_density: float):
        if not self.enabled:
            return
        self.accumulated_dt += dt
        if self.accumulated_dt + 1.0e-12 < self.update_interval:
            return
        step_dt = self.accumulated_dt
        self.accumulated_dt = 0.0
        self.commit_exchange(rest_density)
        maximum_dt = float(self.cfg.get("maximum_step", 0.02))
        substeps = max(1, int(math.ceil(step_dt / maximum_dt)))
        local_dt = step_dt / substeps
        for _ in range(substeps):
            wp.launch(
                advance_shallow_water, dim=(self.nx, self.nz),
                inputs=[self.state, self.updated, self.nx, self.nz, self.cell_size, local_dt,
                        9.81, float(self.cfg.get("bed_drag", 0.006)),
                        float(self.cfg.get("dry_depth", 0.02))], device=self.device,
            )
            self.state, self.updated = self.updated, self.state

    def commit_exchange(self, rest_density: float):
        """Apply pending SPH exchange without advancing the shallow grid.

        Emission happens at an output boundary. Committing it immediately keeps
        the 2D and 3D representations conservative in that same rendered frame
        and prevents checkpoints from storing a newly emitted particle while
        still retaining its source volume in the shallow field.
        """
        if not self.enabled:
            return
        wp.launch(
            apply_exchange_impulse, dim=(self.nx, self.nz),
            inputs=[self.state, self.exchange_volume, self.exchange_x, self.exchange_z,
                    self.cell_size * self.cell_size,
                    rest_density * self.cell_size * self.cell_size], device=self.device,
        )

    def surface_mesh(self):
        """Return the visible rear-field free surface up to the SPH overlap."""
        if not self.enabled or not bool(self.cfg.get("render_far_surface", True)):
            return None, None
        height = self.state.numpy()[:, :, 0]
        visible_nz = min(
            self.nz,
            max(2, int(math.ceil((self.interface_z - self.lower_z) / self.cell_size)) + 2),
        )
        vertices = np.empty((self.nx * visible_nz, 3), dtype=np.float32)
        for ix in range(self.nx):
            for iz in range(visible_nz):
                index = ix * visible_nz + iz
                vertices[index] = (
                    self.lower_x + (ix + 0.5) * self.cell_size,
                    height[ix, iz],
                    self.lower_z + (iz + 0.5) * self.cell_size,
                )
        triangles = []
        for ix in range(self.nx - 1):
            for iz in range(visible_nz - 1):
                a = ix * visible_nz + iz
                b = (ix + 1) * visible_nz + iz
                triangles.extend((a, b, a + 1, b, b + 1, a + 1))
        return vertices, np.asarray(triangles, dtype=np.int32)

    def stitched_surface_samples(self, sph_surface_positions: np.ndarray):
        """Create a shallow/SPH transition sheet for the common scalar field.

        These are reconstruction samples, not additional physical particles.
        Their height follows the 2D field in the rear and smoothly approaches
        a robust SPH free-surface height inside the overlap. Marching Cubes can
        therefore build one surface instead of two independently rasterized
        planes with a visible step between them.
        """
        if not self.enabled or not bool(self.cfg.get("stitch_surface", True)):
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        spacing = float(self.cfg.get("surface_sample_spacing", 1.3))
        coupling_width = float(self.cfg.get("coupling_width", 4.0))
        upper_z = self.interface_z + coupling_width
        xs = np.arange(
            self.lower_x + 0.5 * spacing,
            self.lower_x + self.nx * self.cell_size,
            spacing,
            dtype=np.float32,
        )
        zs = np.arange(
            self.lower_z + 0.5 * spacing,
            upper_z + 0.25 * spacing,
            spacing,
            dtype=np.float32,
        )
        state = self.state.numpy()
        heights = state[:, :, 0]
        ix = np.clip(((xs - self.lower_x) / self.cell_size).astype(np.int32), 0, self.nx - 1)
        interface_iz = int(np.clip((self.interface_z - self.lower_z) / self.cell_size, 0, self.nz - 1))
        target = heights[ix, interface_iz].astype(np.float32, copy=True)

        sph = np.asarray(sph_surface_positions, dtype=np.float32)
        if len(sph):
            band = (
                (sph[:, 2] >= self.interface_z)
                & (sph[:, 2] <= self.interface_z + 2.0 * coupling_width)
                & (sph[:, 1] > float(self.cfg.get("minimum_stitch_height", 4.0)))
            )
            band_points = sph[band]
            if len(band_points):
                bins = np.clip(
                    ((band_points[:, 0] - self.lower_x) / spacing).astype(np.int32),
                    0,
                    len(xs) - 1,
                )
                known = []
                for bin_index in np.unique(bins):
                    values = band_points[bins == bin_index, 1]
                    if len(values) >= 2:
                        target[bin_index] = float(np.quantile(values, 0.95))
                        known.append(int(bin_index))
                if known:
                    known = np.asarray(known, dtype=np.int32)
                    target = np.interp(
                        np.arange(len(xs), dtype=np.float32), known.astype(np.float32), target[known]
                    ).astype(np.float32)
        base_at_interface = heights[ix, interface_iz]
        maximum_delta = float(self.cfg.get("maximum_stitch_height_delta", 6.0))
        target = np.clip(target, base_at_interface - maximum_delta, base_at_interface + maximum_delta)

        samples = np.empty((len(xs) * len(zs), 3), dtype=np.float32)
        cursor = 0
        transition_start = self.interface_z - coupling_width
        transition_span = max(2.0 * coupling_width, 1.0e-5)
        for z in zs:
            iz = int(np.clip((z - self.lower_z) / self.cell_size, 0, self.nz - 1))
            shallow_height = heights[ix, iz]
            blend = float(np.clip((z - transition_start) / transition_span, 0.0, 1.0))
            blend = blend * blend * (3.0 - 2.0 * blend)
            row_height = shallow_height * (1.0 - blend) + target * blend
            count = len(xs)
            samples[cursor:cursor + count, 0] = xs
            samples[cursor:cursor + count, 1] = row_height
            samples[cursor:cursor + count, 2] = z
            cursor += count
        radius = np.full(len(samples), spacing * 0.5, dtype=np.float32)
        return samples, radius

    def diagnostics(self):
        host = self.state.numpy()
        area = self.cell_size * self.cell_size
        return {
            "shallow_water_cells": self.nx * self.nz,
            "shallow_water_wet_cells": int(np.count_nonzero(host[:, :, 0] > 0.02)),
            "shallow_water_volume_m3": float(np.sum(host[:, :, 0], dtype=np.float64) * area),
            "shallow_water_momentum_z": float(np.sum(host[:, :, 2], dtype=np.float64) * area),
            "shallow_emitted_particles": self.emitted_particles_total,
            "shallow_emitted_volume_m3": self.emitted_volume_total,
            "shallow_merged_particles": self.merged_particles_total,
            "shallow_merged_volume_m3": self.merged_volume_total,
            "shallow_net_transfer_volume_m3": self.emitted_volume_total - self.merged_volume_total,
        }
