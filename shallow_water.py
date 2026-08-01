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
    exchange_x: wp.array2d(dtype=float),
    exchange_z: wp.array2d(dtype=float),
    density_times_area: float,
):
    ix, iz = wp.tid()
    state = shallow[ix, iz]
    shallow[ix, iz] = wp.vec3(
        state[0],
        state[1] + exchange_x[ix, iz] / density_times_area,
        state[2] + exchange_z[ix, iz] / density_times_area,
    )
    exchange_x[ix, iz] = 0.0
    exchange_z[ix, iz] = 0.0


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
        wp.launch(
            apply_exchange_impulse, dim=(self.nx, self.nz),
            inputs=[self.state, self.exchange_x, self.exchange_z,
                    rest_density * self.cell_size * self.cell_size], device=self.device,
        )
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

    def diagnostics(self):
        host = self.state.numpy()
        area = self.cell_size * self.cell_size
        return {
            "shallow_water_cells": self.nx * self.nz,
            "shallow_water_wet_cells": int(np.count_nonzero(host[:, :, 0] > 0.02)),
            "shallow_water_volume_m3": float(np.sum(host[:, :, 0], dtype=np.float64) * area),
            "shallow_water_momentum_z": float(np.sum(host[:, :, 2], dtype=np.float64) * area),
        }
