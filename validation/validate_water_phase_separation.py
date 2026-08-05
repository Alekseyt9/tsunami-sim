"""GPU regression for connected/sheet/foam/ballistic water separation."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import (  # noqa: E402
    compute_fluid_forces_multirate,
    precompute_sph_kernel_coefficients,
)
from kernels.surface import classify_water_surface  # noqa: E402


def classify(grid, arrays, count: int) -> None:
    x, v, radius, kind, rho_reference, mask, normal, foam, phase, candidate, age, transitions = arrays
    grid.build(x, 2.6)
    wp.launch(
        classify_water_surface, dim=count,
        inputs=[grid.id, x, v, radius, kind, rho_reference, mask, normal, foam, phase, candidate, age,
                transitions,
                2.6, 18, 8, 0.32, 5, 3, 2, 0.86, 0.64, 1], device="cuda:0",
    )


def main() -> None:
    wp.init()
    device = "cuda:0"
    bulk = np.asarray(
        [[x, y, z] for x in (-1.0, 0.0, 1.0) for y in (1.0, 2.0, 3.0)
         for z in (-1.0, 0.0, 1.0)], dtype=np.float32,
    )
    sheet = np.asarray(
        [[20.0 + x, 8.0, z] for x in (-2.0, -1.0, 0.0, 1.0, 2.0)
         for z in (-2.0, -1.0, 0.0, 1.0, 2.0)], dtype=np.float32,
    )
    droplet = np.asarray([[40.0, 12.0, 0.0]], dtype=np.float32)
    points = np.concatenate([bulk, sheet, droplet], axis=0)
    count = len(points)
    velocity_host = np.zeros_like(points)
    velocity_host[-1, 1] = 9.0
    x = wp.array(points, dtype=wp.vec3, device=device)
    v = wp.array(velocity_host, dtype=wp.vec3, device=device)
    radius = wp.array(np.full(count, 0.5, dtype=np.float32), dtype=float, device=device)
    kind = wp.zeros(count, dtype=wp.int32, device=device)
    rho_reference = wp.zeros(count, dtype=float, device=device)
    mask = wp.zeros(count, dtype=wp.int32, device=device)
    normal = wp.zeros(count, dtype=wp.vec3, device=device)
    foam = wp.zeros(count, dtype=float, device=device)
    phase = wp.zeros(count, dtype=wp.int32, device=device)
    candidate = wp.zeros(count, dtype=wp.int32, device=device)
    age = wp.zeros(count, dtype=wp.int32, device=device)
    transitions = wp.zeros(4, dtype=wp.int32, device=device)
    arrays = (x, v, radius, kind, rho_reference, mask, normal, foam, phase, candidate, age, transitions)
    grid = wp.HashGrid(64, 64, 64, device=device)

    mass_before = np.full(count, 125.0, dtype=np.float32)
    momentum_before = (mass_before[:, None] * velocity_host).sum(axis=0)
    for _ in range(2):
        classify(grid, arrays, count)
    if int(phase.numpy()[-1]) == 2:
        raise AssertionError("droplet entered ballistic mode before hysteresis completed")
    classify(grid, arrays, count)
    wp.synchronize_device(device)

    phase_host = phase.numpy()
    bulk_center = int(np.flatnonzero(np.all(points == (0.0, 2.0, 0.0), axis=1))[0])
    sheet_center = len(bulk) + len(sheet) // 2
    if phase_host[bulk_center] != 0:
        raise AssertionError("connected bulk particle left the core phase")
    if phase_host[sheet_center] != 1:
        raise AssertionError("one-layer lamella was not classified as a thin sheet")
    if phase_host[-1] != 2:
        raise AssertionError("isolated particle did not enter ballistic mode after hysteresis")
    if int(transitions.numpy()[0]) != 1:
        raise AssertionError("ballistic entry transition was not counted exactly once")
    if float(foam.numpy()[-1]) <= 0.05:
        raise AssertionError("energetic spray did not create render-only foam")

    # The ballistic force path is pressure-free but still integrates gravity.
    mass = wp.array(mass_before, dtype=float, device=device)
    volume = wp.array(np.full(count, 0.125, dtype=np.float32), dtype=float, device=device)
    rho = wp.array(np.full(count, 1000.0, dtype=np.float32), dtype=float, device=device)
    pressure = wp.zeros(count, dtype=float, device=device)
    inverse_density = wp.array(
        np.full(count, 1.0 / 1000.0, dtype=np.float32), dtype=float, device=device
    )
    mass_over_density = wp.array(
        mass_before / 1000.0, dtype=float, device=device
    )
    pressure_over_density_squared = wp.zeros(count, dtype=float, device=device)
    level = wp.zeros(count, dtype=wp.int32, device=device)
    active = wp.ones(count, dtype=wp.int32, device=device)
    deferred = wp.zeros((count, 3), dtype=float, device=device)
    acceleration = wp.zeros(count, dtype=wp.vec3, device=device)
    solid_force = wp.zeros(count, dtype=wp.vec3, device=device)
    hydraulic_boundary = wp.zeros(count, dtype=wp.int32, device=device)
    support = wp.zeros(count, dtype=float, device=device)
    support_squared = wp.zeros(count, dtype=float, device=device)
    poly6_coefficient = wp.zeros(count, dtype=float, device=device)
    spiky_coefficient = wp.zeros(count, dtype=float, device=device)
    viscosity_coefficient = wp.zeros(count, dtype=float, device=device)
    wp.launch(
        precompute_sph_kernel_coefficients, dim=count,
        inputs=[radius, support, support_squared, poly6_coefficient,
                spiky_coefficient, viscosity_coefficient], device=device,
    )
    grid.build(x, 2.6)
    wp.launch(
        compute_fluid_forces_multirate, dim=count,
        inputs=[grid.id, x, v, radius, support, support_squared,
                poly6_coefficient, spiky_coefficient, viscosity_coefficient,
                mass, volume, kind, hydraulic_boundary,
                phase, rho, pressure,
                inverse_density, mass_over_density, pressure_over_density_squared,
                level, active,
                deferred, acceleration, solid_force, 1000.0, 0.1, 0.02,
                2.6, 1.0e-4], device=device,
    )
    wp.synchronize_device(device)
    if not np.allclose(acceleration.numpy()[-1], (0.0, -9.81, 0.0), atol=1.0e-5):
        raise AssertionError("isolated ballistic droplet did not receive gravity-only acceleration")
    momentum_after_classification = (mass.numpy()[:, None] * v.numpy()).sum(axis=0)
    if not np.allclose(momentum_after_classification, momentum_before, atol=1.0e-5):
        raise AssertionError("phase classification changed particle momentum")

    # Put the drop back onto the connected free surface.  It must retain the
    # ballistic mode for one classification, then rejoin and request a fresh
    # SPH density normalization on the second.
    rejoined_points = points.copy()
    rejoined_points[-1] = (0.0, 4.0, 0.0)
    wp.copy(x, wp.array(rejoined_points, dtype=wp.vec3, device=device), count=count)
    classify(grid, arrays, count)
    if int(phase.numpy()[-1]) != 2:
        raise AssertionError("ballistic drop left its mode before exit hysteresis completed")
    classify(grid, arrays, count)
    wp.synchronize_device(device)
    if int(phase.numpy()[-1]) == 2 or abs(float(rho_reference.numpy()[-1])) > 1.0e-7:
        raise AssertionError("drop did not rejoin SPH with density recalibration")
    if int(transitions.numpy()[1]) != 1:
        raise AssertionError("SPH rejoin transition was not counted exactly once")

    print(
        "PASS: connected bulk, thin sheet and ballistic drop separate with hysteresis; "
        "classification preserves mass/momentum, rejoins SPH, and foam remains render-only"
    )


if __name__ == "__main__":
    main()
