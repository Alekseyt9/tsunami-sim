"""GPU far-field shallow-water model and conservative SPH interface coupling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp


def prepare_hysteretic_emission_quota(
    interface_state: np.ndarray,
    *,
    cell_size: float,
    emitter_spacing: float,
    emitter_nx: int,
    elapsed: float,
    residual_volume: np.ndarray,
    positive_age: np.ndarray,
    minimum_velocity: float,
    rearm_delay: float,
    ramp_seconds: float,
    maximum_quota_per_cell: int = 0,
    maximum_layers_per_frame: int = 0,
    emission_allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return bottom-up column quotas after a stable positive-flow interval.

    Return flow resets the per-cell clock.  Water is represented only in the
    shallow field during the dead time, so no mass is discarded.  When flow
    becomes positive again the quota ramps up without retaining a large
    scheduler backlog that could later emerge as one vertical particle wall.
    """
    state = np.asarray(interface_state, dtype=np.float64)
    residual = np.asarray(residual_volume, dtype=np.float64).copy()
    age = np.asarray(positive_age, dtype=np.float64).copy()
    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError("interface_state must have shape (nx, >=3)")
    if residual.shape != (len(state),) or age.shape != (len(state),):
        raise ValueError("emission state arrays must match shallow interface width")
    column_quota = np.zeros(max(int(emitter_nx), 1), dtype=np.int32)
    if elapsed <= 0.0:
        return column_quota, residual, age, 0

    depth = np.maximum(state[:, 0], 1.0e-6)
    discharge = state[:, 2]
    velocity = discharge / depth
    stable_positive = velocity >= max(float(minimum_velocity), 0.0)
    if emission_allowed is not None:
        allowed = np.asarray(emission_allowed, dtype=bool)
        if allowed.shape != (len(state),):
            raise ValueError("emission_allowed must match shallow interface width")
        stable_positive &= allowed
    age[stable_positive] += float(elapsed)
    age[~stable_positive] = 0.0
    # A residual is only the sub-particle fraction of a currently active
    # discharge.  Keeping it through reverse flow creates an artificial pulse
    # when the sign changes again.
    residual[~stable_positive] = 0.0

    delay = max(float(rearm_delay), 0.0)
    ramp_duration = max(float(ramp_seconds), 1.0e-6)
    ramp = np.clip((age - delay) / ramp_duration, 0.0, 1.0)
    positive_discharge = np.maximum(discharge, 0.0) * ramp
    particle_volume = float(emitter_spacing) ** 3
    requested_volume = positive_discharge * float(cell_size) * float(elapsed) + residual
    raw_quota = np.floor(requested_volume / particle_volume).astype(np.int64)
    # Preserve only a fractional scheduler residual.  Whole particles denied
    # by a rate cap remain physical shallow-water volume and are reconsidered
    # from the next frame's actual discharge instead of forming a backlog.
    residual = requested_volume - raw_quota.astype(np.float64) * particle_volume
    cell_quota = raw_quota.copy()
    if maximum_quota_per_cell > 0:
        cell_quota = np.minimum(cell_quota, int(maximum_quota_per_cell))

    emitter_indices = np.arange(len(column_quota), dtype=np.int32)
    emitter_x = (emitter_indices.astype(np.float64) + 0.5) * float(emitter_spacing)
    emitter_cell = np.clip(
        np.floor(emitter_x / float(cell_size)).astype(np.int32), 0, len(state) - 1
    )
    for cell in np.flatnonzero(cell_quota > 0):
        columns = emitter_indices[emitter_cell == cell]
        if len(columns) == 0:
            continue
        quota = int(cell_quota[cell])
        if maximum_layers_per_frame > 0:
            quota = min(quota, len(columns) * int(maximum_layers_per_frame))
        base, extra = divmod(quota, len(columns))
        column_quota[columns] = base
        if extra:
            column_quota[columns[:extra]] += 1
    return column_quota, residual, age, int(np.sum(column_quota, dtype=np.int64))


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
    sustained_inlet: int,
    inlet_depth: float,
    inlet_velocity_z: float,
    inlet_volume: wp.array(dtype=float),
    inlet_momentum_z: wp.array(dtype=float),
    transmissive_outlet: int,
    outlet_volume: wp.array(dtype=float),
    outlet_momentum_z: wp.array(dtype=float),
):
    ix, iz = wp.tid()
    center = current[ix, iz]
    left = current[wp.max(ix - 1, 0), iz]
    right = current[wp.min(ix + 1, nx - 1), iz]
    back = current[ix, wp.max(iz - 1, 0)]
    front = current[ix, wp.min(iz + 1, nz - 1)]
    # The legacy pulse uses a reflective offshore boundary.  A sustained
    # tsunami surge instead prescribes the incoming far-field state through a
    # Riemann ghost cell.  This supplies flux at the boundary rather than
    # repeatedly raising every cell in an offshore strip.
    if ix == 0:
        left = wp.vec3(center[0], -center[1], center[2])
    if ix == nx - 1:
        right = wp.vec3(center[0], -center[1], center[2])
    if iz == 0:
        if sustained_inlet != 0:
            back = wp.vec3(inlet_depth, 0.0, inlet_depth * inlet_velocity_z)
        else:
            back = wp.vec3(center[0], center[1], -center[2])
    if iz == nz - 1:
        if transmissive_outlet != 0:
            # Zero-gradient Riemann ghost: outgoing characteristics leave the
            # city instead of reflecting from an invisible wall.  This is the
            # standard transmissive finite-volume boundary for a downstream
            # domain whose detailed geometry lies outside the simulation.
            front = center
        else:
            front = wp.vec3(center[0], center[1], -center[2])
    back_flux = _rusanov_z(back, center, gravity)
    front_flux = _rusanov_z(center, front, gravity)
    next_value = center - (dt / cell_size) * (
        _rusanov_x(center, right, gravity) - _rusanov_x(left, center, gravity)
        + front_flux - back_flux
    )
    h = wp.max(next_value[0], 0.0)
    if h <= dry_depth:
        updated[ix, iz] = wp.vec3(0.0, 0.0, 0.0)
    else:
        damping = wp.max(0.0, 1.0 - bed_drag * dt)
        updated[ix, iz] = wp.vec3(h, next_value[1] * damping, next_value[2] * damping)
    if iz == 0 and sustained_inlet != 0:
        # Boundary flux is specific discharge (m2/s).  Multiplication by the
        # cell width and dt gives the signed volume/momentum entering this row.
        wp.atomic_add(inlet_volume, 0, back_flux[0] * cell_size * dt)
        wp.atomic_add(inlet_momentum_z, 0, back_flux[2] * cell_size * dt)
    if iz == nz - 1 and transmissive_outlet != 0:
        wp.atomic_add(outlet_volume, 0, front_flux[0] * cell_size * dt)
        wp.atomic_add(outlet_momentum_z, 0, front_flux[2] * cell_size * dt)


@wp.kernel
def inject_wave_train_pulse(
    state: wp.array2d(dtype=wp.vec3),
    injected_volume: wp.array(dtype=float),
    injected_momentum_z: wp.array(dtype=float),
    nx: int,
    nz: int,
    cell_size: float,
    dt: float,
    simulation_time: float,
    start_time: float,
    duration: float,
    pulse_height: float,
    pulse_speed: float,
    background_current: float,
    pulse_length: float,
):
    """Inject a finite, conservative long-wave slug at the offshore boundary."""
    ix, iz = wp.tid()
    if simulation_time < start_time or simulation_time >= start_time + duration:
        return
    z_from_boundary = (float(iz) + 0.5) * cell_size
    if z_from_boundary >= pulse_length:
        return
    phase = (simulation_time - start_time) / wp.max(duration, 1.0e-5)
    temporal = wp.sin(3.14159265 * phase)
    temporal_rate = (2.0 / wp.max(duration, 1.0e-5)) * temporal * temporal
    spatial = 0.5 + 0.5 * wp.cos(3.14159265 * z_from_boundary / pulse_length)
    dh = pulse_height * temporal_rate * spatial * dt
    if dh <= 0.0:
        return
    previous = state[ix, iz]
    # ``dh`` is only the tiny increment added during this solver substep.  The
    # previous expression multiplied the requested pulse speed by dh / h, so
    # as dt became smaller the second wave lost essentially all of its forward
    # impulse and degenerated into a slow water-level rise.  Treat the injected
    # layer as a finite incoming bore: its conserved volume enters at the
    # background current plus the configured pulse speed.  Existing cell
    # momentum is retained and the shallow-water solver then spreads the bore.
    injection_velocity = background_current + pulse_speed
    added_momentum = dh * injection_velocity
    state[ix, iz] = wp.vec3(previous[0] + dh, previous[1], previous[2] + added_momentum)
    area = cell_size * cell_size
    wp.atomic_add(injected_volume, 0, dh * area)
    wp.atomic_add(injected_momentum_z, 0, added_momentum * area)


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
    incoming_characteristic: int,
    incoming_relaxation_rate: float,
    minimum_incoming_velocity: float,
    incoming_impulse: wp.array(dtype=float),
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
    characteristic_impulse = float(0.0)
    if incoming_characteristic != 0 and target_z >= minimum_incoming_velocity:
        # An SPH inlet is an open boundary, not a closed overlap region.  Only
        # the incoming shallow-water characteristic is prescribed here: slow
        # or reflected particles are brought up to the incident bore speed,
        # while particles already travelling faster are not braked into an
        # artificial compression wall.  Returning particles are handled by
        # the conservative SPH -> shallow capture kernel below.
        az = wp.max(target_z - v[i][2], 0.0) * incoming_relaxation_rate * weight * weight
        characteristic_impulse = mass[i] * az * dt
    acceleration[i] = acceleration[i] + wp.vec3(ax, 0.0, az)
    # Tangential relaxation remains a conservative two-way exchange.  In the
    # legacy overlap mode the normal impulse is also returned to the shallow
    # cell.  For an open incoming-characteristic boundary it is pressure work
    # performed by the unresolved offshore reservoir and is recorded instead
    # of immediately cancelling the incident wave in a single grid cell.
    wp.atomic_add(exchange_x, ix, iz, -mass[i] * ax * dt)
    if incoming_characteristic != 0 and target_z >= minimum_incoming_velocity:
        wp.atomic_add(incoming_impulse, 0, characteristic_impulse)
    else:
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
    wave_cohort: wp.array(dtype=wp.int32),
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
    emission_quota: wp.array(dtype=wp.int32),
    use_flux_quota: int,
    cohort_id: int,
    cohort_emitted_volume: wp.array(dtype=float),
    cohort_emitted_momentum_z: wp.array(dtype=float),
):
    emitter_x = wp.tid()
    if emitter_x >= emitter_nx:
        return
    px = lower_x + (float(emitter_x) + 0.5) * particle_spacing
    pz = interface_z + 0.55 * particle_spacing
    sx = wp.clamp(int(wp.floor((px - lower_x) / cell_size)), 0, shallow_nx - 1)
    sz = wp.clamp(int(wp.floor((interface_z - lower_z) / cell_size)), 0, shallow_nz - 1)
    state = shallow[sx, sz]
    velocity_x = state[1] / wp.max(state[0], 1.0e-5)
    velocity_z = state[2] / wp.max(state[0], 1.0e-5)
    if velocity_z < minimum_emission_velocity:
        return
    allowed = emitter_ny
    if use_flux_quota != 0:
        allowed = emission_quota[emitter_x]
    if allowed <= 0:
        return
    particle_volume = particle_spacing * particle_spacing * particle_spacing
    particle_mass = particle_volume * rest_density
    velocity = wp.vec3(velocity_x, 0.0, velocity_z)
    emitted_column = int(0)
    # One thread owns one x-column and visits layers in ascending order.  The
    # old 2-D launch let arbitrary GPU threads consume a shared quota, which
    # selected disconnected particles anywhere up to the full shallow depth
    # and reconstructed them as a tall water wall after flow reversal.
    for emitter_y in range(emitter_ny):
        if emitted_column < allowed:
            py = (float(emitter_y) + 0.5) * particle_spacing
            if state[0] > py + 0.5 * particle_spacing:
                position = wp.vec3(px, py, pz)
                occupied = int(0)
                query = wp.hash_grid_query(grid, position, particle_spacing * 0.62)
                for neighbour in query:
                    if neighbour < old_count and kind[neighbour] == 0:
                        if wp.length_sq(x[neighbour] - position) < particle_spacing * particle_spacing * 0.38:
                            occupied = 1
                            break
                if occupied == 0:
                    target = wp.atomic_add(count, 0, 1)
                    if target < capacity:
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
                        wave_cohort[target] = cohort_id
                        if cohort_id > 0:
                            wp.atomic_add(cohort_emitted_volume, 0, particle_volume)
                            wp.atomic_add(cohort_emitted_momentum_z, 0, particle_mass * velocity_z)
                        wp.atomic_add(exchange_volume, sx, sz, -particle_volume)
                        wp.atomic_add(exchange_x, sx, sz, -particle_mass * velocity_x)
                        wp.atomic_add(exchange_z, sx, sz, -particle_mass * velocity_z)
                        emitted_column += 1


@wp.kernel
def mark_sph_return_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    wave_cohort: wp.array(dtype=wp.int32),
    keep: wp.array(dtype=wp.int32),
    exchange_volume: wp.array2d(dtype=float),
    exchange_x: wp.array2d(dtype=float),
    exchange_z: wp.array2d(dtype=float),
    return_volume_by_x: wp.array(dtype=float),
    merged_volume: wp.array(dtype=float),
    cohort_returned_volume: wp.array(dtype=float),
    cohort_returned_momentum_z: wp.array(dtype=float),
    lower_x: float,
    lower_z: float,
    interface_z: float,
    cell_size: float,
    shallow_nx: int,
    shallow_nz: int,
    minimum_return_speed: float,
    forced_capture_depth: float,
    reverse_capture_width: float,
):
    i = wp.tid()
    position = x[i]
    velocity = v[i]
    crossed_interface = (
        position[2] < interface_z
        and (
            velocity[2] <= -minimum_return_speed
            or position[2] <= interface_z - forced_capture_depth
        )
    )
    reflected_in_buffer = (
        reverse_capture_width > 0.0
        and position[2] >= interface_z
        and position[2] < interface_z + reverse_capture_width
        and velocity[2] <= -minimum_return_speed
    )
    returning = kind[i] == 0 and (crossed_interface or reflected_in_buffer)
    if not returning:
        keep[i] = 1
        return
    keep[i] = 0
    ix = wp.clamp(int(wp.floor((position[0] - lower_x) / cell_size)), 0, shallow_nx - 1)
    iz = wp.clamp(int(wp.floor((interface_z - lower_z) / cell_size)), 0, shallow_nz - 1)
    wp.atomic_add(exchange_volume, ix, iz, volume[i])
    wp.atomic_add(exchange_x, ix, iz, mass[i] * velocity[0])
    wp.atomic_add(exchange_z, ix, iz, mass[i] * velocity[2])
    wp.atomic_add(return_volume_by_x, ix, volume[i])
    wp.atomic_add(merged_volume, 0, volume[i])
    if wave_cohort[i] > 0:
        wp.atomic_add(cohort_returned_volume, 0, volume[i])
        wp.atomic_add(cohort_returned_momentum_z, 0, mass[i] * velocity[2])


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
        self.probe_rows_m = tuple(
            float(value) for value in policy.get("probe_rows_m", (16.0, 52.0, 91.0))
        )
        self.update_interval = float(policy.get("update_interval", 0.008))
        self.accumulated_dt = 0.0
        self.wave_train_injected_volume = 0.0
        self.wave_train_injected_momentum_z = 0.0
        self.downstream_outflow_volume = 0.0
        self.downstream_outflow_momentum_z = 0.0
        downstream_boundary = str(
            policy.get("downstream_boundary", "reflective")
        ).strip().lower()
        if downstream_boundary not in {"reflective", "transmissive", "open", "outflow"}:
            raise ValueError(
                "shallow_water downstream_boundary must be reflective or transmissive"
            )
        self.transmissive_outlet = downstream_boundary in {
            "transmissive", "open", "outflow"
        }
        incoming_impulse_initial = 0.0
        cohort_emitted_volume_initial = 0.0
        cohort_emitted_momentum_initial = 0.0
        cohort_returned_volume_initial = 0.0
        cohort_returned_momentum_initial = 0.0
        host = np.zeros((self.nx, self.nz, 3), dtype=np.float32)
        depth = float(cfg["water_depth"])
        self.base_depth = depth
        crest = float(cfg["wave_height"])
        wave_speed = float(cfg["wave_speed"])
        background = float(cfg.get("background_current", 0.0))
        reservoir_front = float(cfg["reservoir_z_max"])
        if self.enabled:
            for iz in range(self.nz):
                z = self.lower_z + (iz + 0.5) * self.cell_size
                if z >= reservoir_front:
                    continue
                elevation = crest * math.exp(-((z - reservoir_front + 5.0) / 7.5) ** 2)
                h = depth + elevation
                velocity_z = background + wave_speed * elevation / max(h, 1.0e-6)
                host[:, iz, 0] = h
                host[:, iz, 2] = h * velocity_z
        if self.enabled and checkpoint is not None and checkpoint.exists():
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
        self.flux_requested_particles_total = 0
        self.flux_emitted_particles_total = 0
        self.emission_residual_volume = np.zeros(self.nx, dtype=np.float64)
        self.emission_positive_age = np.zeros(self.nx, dtype=np.float64)
        self.emission_return_quiet_age = np.zeros(self.nx, dtype=np.float64)
        self.last_frame_return_volume_by_x = np.zeros(self.nx, dtype=np.float64)
        self.emission_blocked_cells = self.nx
        self.returning_cells = 0
        self.return_flow_quiet_age = 0.0
        self.last_frame_merged_volume = 0.0
        self.last_emission_time = 0.0
        self.merged_particles_total = 0
        self.merged_volume_total = 0.0
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if "shallow_emitted_particles_total" in saved:
                    self.emitted_particles_total = int(saved["shallow_emitted_particles_total"])
                if "shallow_emitted_volume_total" in saved:
                    self.emitted_volume_total = float(saved["shallow_emitted_volume_total"])
                if "shallow_flux_requested_particles_total" in saved:
                    self.flux_requested_particles_total = int(
                        saved["shallow_flux_requested_particles_total"]
                    )
                if "shallow_flux_emitted_particles_total" in saved:
                    self.flux_emitted_particles_total = int(
                        saved["shallow_flux_emitted_particles_total"]
                    )
                if (
                    "shallow_emission_residual_volume" in saved
                    and saved["shallow_emission_residual_volume"].shape == (self.nx,)
                ):
                    self.emission_residual_volume = saved[
                        "shallow_emission_residual_volume"
                    ].astype(np.float64, copy=True)
                if (
                    "shallow_emission_positive_age" in saved
                    and saved["shallow_emission_positive_age"].shape == (self.nx,)
                ):
                    self.emission_positive_age = saved[
                        "shallow_emission_positive_age"
                    ].astype(np.float64, copy=True)
                if (
                    "shallow_emission_return_quiet_age" in saved
                    and saved["shallow_emission_return_quiet_age"].shape == (self.nx,)
                ):
                    self.emission_return_quiet_age = saved[
                        "shallow_emission_return_quiet_age"
                    ].astype(np.float64, copy=True)
                if "shallow_return_flow_quiet_age" in saved:
                    self.return_flow_quiet_age = float(saved["shallow_return_flow_quiet_age"])
                if "shallow_merged_particles_total" in saved:
                    self.merged_particles_total = int(saved["shallow_merged_particles_total"])
                if "shallow_merged_volume_total" in saved:
                    self.merged_volume_total = float(saved["shallow_merged_volume_total"])
                if "wave_train_injected_volume" in saved:
                    self.wave_train_injected_volume = float(saved["wave_train_injected_volume"])
                if "wave_train_injected_momentum_z" in saved:
                    self.wave_train_injected_momentum_z = float(saved["wave_train_injected_momentum_z"])
                if "shallow_downstream_outflow_volume" in saved:
                    self.downstream_outflow_volume = float(
                        saved["shallow_downstream_outflow_volume"]
                    )
                if "shallow_downstream_outflow_momentum_z" in saved:
                    self.downstream_outflow_momentum_z = float(
                        saved["shallow_downstream_outflow_momentum_z"]
                    )
                if "shallow_incoming_boundary_impulse" in saved:
                    incoming_impulse_initial = float(saved["shallow_incoming_boundary_impulse"])
                if "wave_cohort_emitted_volume" in saved:
                    cohort_emitted_volume_initial = float(saved["wave_cohort_emitted_volume"])
                if "wave_cohort_emitted_momentum_z" in saved:
                    cohort_emitted_momentum_initial = float(
                        saved["wave_cohort_emitted_momentum_z"]
                    )
                if "wave_cohort_returned_volume" in saved:
                    cohort_returned_volume_initial = float(saved["wave_cohort_returned_volume"])
                if "wave_cohort_returned_momentum_z" in saved:
                    cohort_returned_momentum_initial = float(
                        saved["wave_cohort_returned_momentum_z"]
                    )
        self._wave_train_volume_step = wp.zeros(1, dtype=float, device=device)
        self._wave_train_momentum_step = wp.zeros(1, dtype=float, device=device)
        self._downstream_outflow_volume_step = wp.zeros(1, dtype=float, device=device)
        self._downstream_outflow_momentum_step = wp.zeros(1, dtype=float, device=device)
        self.emission_quota = wp.zeros(self.nx, dtype=wp.int32, device=device)
        self.return_volume_by_x = wp.zeros(self.nx, dtype=float, device=device)
        self.incoming_boundary_impulse = wp.array(
            np.asarray([incoming_impulse_initial], dtype=np.float32),
            dtype=float,
            device=device,
        )
        self.wave_cohort_emitted_volume = wp.array(
            np.asarray([cohort_emitted_volume_initial], dtype=np.float32),
            dtype=float, device=device,
        )
        self.wave_cohort_emitted_momentum_z = wp.array(
            np.asarray([cohort_emitted_momentum_initial], dtype=np.float32),
            dtype=float, device=device,
        )
        self.wave_cohort_returned_volume = wp.array(
            np.asarray([cohort_returned_volume_initial], dtype=np.float32),
            dtype=float, device=device,
        )
        self.wave_cohort_returned_momentum_z = wp.array(
            np.asarray([cohort_returned_momentum_initial], dtype=np.float32),
            dtype=float, device=device,
        )

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
                    float(self.cfg.get("velocity_relaxation_rate", 1.5)),
                    int(bool(self.cfg.get("incoming_characteristic", False))),
                    float(self.cfg.get("incoming_relaxation_rate", 6.0)),
                    float(self.cfg.get("minimum_incoming_velocity", 0.5)),
                    self.incoming_boundary_impulse, dt],
            device=self.device,
        )

    def advance(self, dt: float, rest_density: float, simulation_time: float = 0.0):
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
        wave_train = self.cfg.get("wave_train", {})
        for substep in range(substeps):
            local_time = simulation_time - step_dt + (substep + 1) * local_dt
            profile = str(wave_train.get("profile", "pulse")).strip().lower()
            sustained_enabled = bool(wave_train.get("enabled", False)) and profile in {
                "sustained", "sustained_surge", "surge",
            }
            inlet_active = False
            inlet_depth = float(wave_train.get("base_depth", self.cfg.get("base_depth", 0.0)))
            inlet_velocity = float(wave_train.get("background_current", 0.0))
            if sustained_enabled:
                start = float(wave_train.get("start_seconds", 0.0))
                ramp_up = max(float(wave_train.get("ramp_up_seconds", 2.0)), 1.0e-6)
                hold = max(float(wave_train.get("hold_seconds", 8.0)), 0.0)
                ramp_down = max(float(wave_train.get("ramp_down_seconds", 3.0)), 1.0e-6)
                elapsed = local_time - start
                total_duration = ramp_up + hold + ramp_down
                if 0.0 <= elapsed < total_duration:
                    inlet_active = True
                    if elapsed < ramp_up:
                        envelope = elapsed / ramp_up
                    elif elapsed < ramp_up + hold:
                        envelope = 1.0
                    else:
                        envelope = 1.0 - (elapsed - ramp_up - hold) / ramp_down
                    envelope = min(max(envelope, 0.0), 1.0)
                    # Smooth first derivative at both ends of the hydrograph so
                    # the source does not generate a numerical impulse.
                    envelope = envelope * envelope * (3.0 - 2.0 * envelope)
                    base_depth = float(
                        wave_train.get("base_depth", self.cfg.get("base_depth", 0.0))
                    )
                    if base_depth <= 0.0:
                        base_depth = float(wave_train.get("water_depth", 0.0))
                    if base_depth <= 0.0:
                        # The global water depth is copied into the policy by
                        # the solver constructor for compact generated configs.
                        base_depth = float(self.base_depth)
                    background = float(wave_train.get("background_current", 0.0))
                    target_velocity = float(
                        wave_train.get(
                            "target_velocity",
                            background + float(wave_train.get("speed", 14.0)),
                        )
                    )
                    inlet_depth = base_depth + float(wave_train.get("height", 8.0)) * envelope
                    inlet_velocity = background + (target_velocity - background) * envelope
            self._wave_train_volume_step.zero_()
            self._wave_train_momentum_step.zero_()
            self._downstream_outflow_volume_step.zero_()
            self._downstream_outflow_momentum_step.zero_()
            wp.launch(
                advance_shallow_water, dim=(self.nx, self.nz),
                inputs=[self.state, self.updated, self.nx, self.nz, self.cell_size, local_dt,
                        9.81, float(self.cfg.get("bed_drag", 0.006)),
                        float(self.cfg.get("dry_depth", 0.02)), int(inlet_active),
                        inlet_depth, inlet_velocity, self._wave_train_volume_step,
                        self._wave_train_momentum_step, int(self.transmissive_outlet),
                        self._downstream_outflow_volume_step,
                        self._downstream_outflow_momentum_step], device=self.device,
            )
            self.state, self.updated = self.updated, self.state
            if self.transmissive_outlet:
                self.downstream_outflow_volume += float(
                    self._downstream_outflow_volume_step.numpy()[0]
                )
                self.downstream_outflow_momentum_z += float(
                    self._downstream_outflow_momentum_step.numpy()[0]
                )
            if inlet_active:
                self.wave_train_injected_volume += float(
                    self._wave_train_volume_step.numpy()[0]
                )
                self.wave_train_injected_momentum_z += float(
                    self._wave_train_momentum_step.numpy()[0]
                )
            if bool(wave_train.get("enabled", False)) and not sustained_enabled:
                self._wave_train_volume_step.zero_()
                self._wave_train_momentum_step.zero_()
                wp.launch(
                    inject_wave_train_pulse, dim=(self.nx, self.nz),
                    inputs=[
                        self.state, self._wave_train_volume_step,
                        self._wave_train_momentum_step, self.nx, self.nz,
                        self.cell_size, local_dt, local_time,
                        float(wave_train.get("start_seconds", 7.0)),
                        float(wave_train.get("duration_seconds", 3.0)),
                        float(wave_train.get("height", 4.5)),
                        float(wave_train.get("speed", 14.0)),
                        float(wave_train.get("background_current", 5.0)),
                        float(wave_train.get("length_m", 30.0)),
                    ], device=self.device,
                )
                self.wave_train_injected_volume += float(
                    self._wave_train_volume_step.numpy()[0]
                )
                self.wave_train_injected_momentum_z += float(
                    self._wave_train_momentum_step.numpy()[0]
                )

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
        """Return a closed rear-field water volume up to the SPH overlap.

        A height-field top alone has no optical thickness, while feeding the
        complete rear field to Marching Cubes produces a thin floating shell.
        Top, bed and perimeter faces give the renderer a meaningful front/back
        depth pair. The short overlap is hidden by the transition field.
        """
        if not self.enabled or not bool(self.cfg.get("render_far_surface", True)):
            return None, None
        height = self.state.numpy()[:, :, 0]
        coupling_width = float(self.cfg.get("coupling_width", 4.0))
        transition_start = self.interface_z - coupling_width
        visible_nz = min(
            self.nz,
            max(
                2,
                int(math.ceil((transition_start - self.lower_z) / self.cell_size)) + 1,
            ),
        )
        surface_count = self.nx * visible_nz
        vertices = np.empty((surface_count * 2, 3), dtype=np.float32)
        bed_height = float(self.cfg.get("render_bed_height", 0.0))
        for ix in range(self.nx):
            for iz in range(visible_nz):
                index = ix * visible_nz + iz
                x = self.lower_x + (ix + 0.5) * self.cell_size
                z = self.lower_z + (iz + 0.5) * self.cell_size
                vertices[index] = (
                    x,
                    max(float(height[ix, iz]), bed_height),
                    z,
                )
                vertices[surface_count + index] = (
                    x,
                    bed_height,
                    z,
                )
        triangles: list[int] = []
        for ix in range(self.nx - 1):
            for iz in range(visible_nz - 1):
                a = ix * visible_nz + iz
                b = (ix + 1) * visible_nz + iz
                c = a + 1
                d = b + 1
                triangles.extend((a, b, c, b, d, c))
                ba = surface_count + a
                bb = surface_count + b
                bc = surface_count + c
                bd = surface_count + d
                triangles.extend((ba, bc, bb, bb, bc, bd))

        def append_side(a: int, b: int) -> None:
            ba = surface_count + a
            bb = surface_count + b
            triangles.extend((a, ba, b, b, ba, bb))

        for ix in range(self.nx - 1):
            append_side(ix * visible_nz, (ix + 1) * visible_nz)
            append_side(
                ix * visible_nz + visible_nz - 1,
                (ix + 1) * visible_nz + visible_nz - 1,
            )
        for iz in range(visible_nz - 1):
            append_side(iz, iz + 1)
            append_side(
                (self.nx - 1) * visible_nz + iz,
                (self.nx - 1) * visible_nz + iz + 1,
            )
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
        transition_start = self.interface_z - coupling_width
        overlap = max(
            spacing,
            float(self.cfg.get("stitch_overlap_m", max(self.cell_size, 2.0 * spacing))),
        )
        zs = np.arange(
            max(self.lower_z + 0.5 * spacing, transition_start - overlap),
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
        has_sph_target = False
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
                    has_sph_target = True
        if has_sph_target and len(target) >= 3:
            # Suppress individual-particle peaks without moving the interface
            # back toward the (potentially much taller) shallow column.
            padded = np.pad(target, (1, 1), mode="edge")
            target = (
                0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
            ).astype(np.float32)
        target = np.clip(
            target,
            float(self.cfg.get("render_bed_height", 0.0)),
            float(self.cfg.get("maximum_stitch_height", 48.0)),
        )

        samples = np.empty((len(xs) * len(zs), 3), dtype=np.float32)
        cursor = 0
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
        if not self.enabled:
            result = {
                "shallow_water_cells": 0,
                "shallow_water_wet_cells": 0,
                "shallow_water_volume_m3": 0.0,
                "shallow_water_momentum_z": 0.0,
                "shallow_emitted_particles": 0,
                "shallow_emitted_volume_m3": 0.0,
                "shallow_flux_requested_particles": 0,
                "shallow_flux_emitted_particles": 0,
                "shallow_flux_emission_efficiency": 0.0,
                "shallow_emission_rearmed_cells": 0,
                "shallow_emission_blocked_cells": 0,
                "shallow_returning_cells": 0,
                "shallow_return_flow_quiet_age_s": 0.0,
                "shallow_merged_particles": 0,
                "shallow_merged_volume_m3": 0.0,
                "shallow_net_transfer_volume_m3": 0.0,
                "wave_train_injected_volume_m3": 0.0,
                "wave_train_injected_momentum_z": 0.0,
                "shallow_downstream_outflow_volume_m3": 0.0,
                "shallow_downstream_outflow_momentum_z": 0.0,
                "coupling_incoming_boundary_impulse_kg_m_s": 0.0,
                "wave_cohort_emitted_volume_m3": 0.0,
                "wave_cohort_emitted_momentum_z_kg_m_s": 0.0,
                "wave_cohort_returned_volume_m3": 0.0,
                "wave_cohort_returned_momentum_z_kg_m_s": 0.0,
            }
            for row_index, row_z in enumerate(self.probe_rows_m, start=1):
                prefix = f"wave_row_{row_index}"
                result[f"{prefix}_z_m"] = row_z
                result[f"{prefix}_depth_mean_m"] = 0.0
                result[f"{prefix}_depth_max_m"] = 0.0
                result[f"{prefix}_velocity_mean_m_s"] = 0.0
                result[f"{prefix}_velocity_max_m_s"] = 0.0
                result[f"{prefix}_forward_discharge_m3_s"] = 0.0
                result[f"{prefix}_reverse_discharge_m3_s"] = 0.0
                result[f"{prefix}_specific_momentum_flux_m4_s2"] = 0.0
            return result
        host = self.state.numpy()
        area = self.cell_size * self.cell_size
        result = {
            "shallow_water_cells": self.nx * self.nz,
            "shallow_water_wet_cells": int(np.count_nonzero(host[:, :, 0] > 0.02)),
            "shallow_water_volume_m3": float(np.sum(host[:, :, 0], dtype=np.float64) * area),
            "shallow_water_momentum_z": float(np.sum(host[:, :, 2], dtype=np.float64) * area),
            "shallow_emitted_particles": self.emitted_particles_total,
            "shallow_emitted_volume_m3": self.emitted_volume_total,
            "shallow_flux_requested_particles": self.flux_requested_particles_total,
            "shallow_flux_emitted_particles": self.flux_emitted_particles_total,
            "shallow_flux_emission_efficiency": float(
                self.flux_emitted_particles_total
                / max(self.flux_requested_particles_total, 1)
            ),
            "shallow_emission_rearmed_cells": int(np.count_nonzero(
                self.emission_positive_age
                >= float(self.cfg.get("emission_rearm_delay_seconds", 0.35))
            )),
            "shallow_emission_blocked_cells": int(self.emission_blocked_cells),
            "shallow_returning_cells": int(self.returning_cells),
            "shallow_return_flow_quiet_age_s": self.return_flow_quiet_age,
            "shallow_merged_particles": self.merged_particles_total,
            "shallow_merged_volume_m3": self.merged_volume_total,
            "shallow_net_transfer_volume_m3": self.emitted_volume_total - self.merged_volume_total,
            "wave_train_injected_volume_m3": self.wave_train_injected_volume,
            "wave_train_injected_momentum_z": self.wave_train_injected_momentum_z,
            "shallow_downstream_outflow_volume_m3": self.downstream_outflow_volume,
            "shallow_downstream_outflow_momentum_z": self.downstream_outflow_momentum_z,
            "coupling_incoming_boundary_impulse_kg_m_s": float(
                self.incoming_boundary_impulse.numpy()[0]
            ),
            "wave_cohort_emitted_volume_m3": float(
                self.wave_cohort_emitted_volume.numpy()[0]
            ),
            "wave_cohort_emitted_momentum_z_kg_m_s": float(
                self.wave_cohort_emitted_momentum_z.numpy()[0]
            ),
            "wave_cohort_returned_volume_m3": float(
                self.wave_cohort_returned_volume.numpy()[0]
            ),
            "wave_cohort_returned_momentum_z_kg_m_s": float(
                self.wave_cohort_returned_momentum_z.numpy()[0]
            ),
        }
        dry_depth = float(self.cfg.get("dry_depth", 0.02))
        for row_index, row_z in enumerate(self.probe_rows_m, start=1):
            iz = int(np.clip(
                np.floor((row_z - self.lower_z) / self.cell_size),
                0,
                self.nz - 1,
            ))
            depth = host[:, iz, 0].astype(np.float64, copy=False)
            discharge = host[:, iz, 2].astype(np.float64, copy=False)
            velocity = np.divide(
                discharge,
                np.maximum(depth, dry_depth),
                out=np.zeros_like(discharge),
                where=depth > dry_depth,
            )
            # Integral across the domain width.  The specific momentum flux
            # omits density, so multiplying it by rho gives force in newtons.
            specific_flux = discharge * velocity + 0.5 * 9.81 * depth * depth
            prefix = f"wave_row_{row_index}"
            result[f"{prefix}_z_m"] = row_z
            result[f"{prefix}_depth_mean_m"] = float(np.mean(depth))
            result[f"{prefix}_depth_max_m"] = float(np.max(depth))
            result[f"{prefix}_velocity_mean_m_s"] = float(np.mean(velocity))
            result[f"{prefix}_velocity_max_m_s"] = float(np.max(velocity))
            result[f"{prefix}_forward_discharge_m3_s"] = float(
                np.sum(np.maximum(discharge, 0.0), dtype=np.float64) * self.cell_size
            )
            result[f"{prefix}_reverse_discharge_m3_s"] = float(
                np.sum(np.minimum(discharge, 0.0), dtype=np.float64) * self.cell_size
            )
            result[f"{prefix}_specific_momentum_flux_m4_s2"] = float(
                np.sum(specific_flux, dtype=np.float64) * self.cell_size
            )
        return result
