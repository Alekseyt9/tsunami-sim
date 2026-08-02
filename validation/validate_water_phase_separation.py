"""GPU regression for connected/sheet/foam/ballistic water separation."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from hybrid_kernels import compute_fluid_forces_multirate  # noqa: E402
from surface_kernels import classify_water_surface  # noqa: E402


def classify(grid, arrays, count: int) -> None:
    x, v, radius, kind, mask, normal, foam, phase, candidate, age = arrays
    grid.build(x, 2.6)
    wp.launch(
        classify_water_surface, dim=count,
        inputs=[grid.id, x, v, radius, kind, mask, normal, foam, phase, candidate, age,
                2.6, 18, 8, 0.32, 5, 3, 2, 0.86], device="cuda:0",
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
    mask = wp.zeros(count, dtype=wp.int32, device=device)
    normal = wp.zeros(count, dtype=wp.vec3, device=device)
    foam = wp.zeros(count, dtype=float, device=device)
    phase = wp.zeros(count, dtype=wp.int32, device=device)
    candidate = wp.zeros(count, dtype=wp.int32, device=device)
    age = wp.zeros(count, dtype=wp.int32, device=device)
    arrays = (x, v, radius, kind, mask, normal, foam, phase, candidate, age)
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
    if float(foam.numpy()[-1]) <= 0.05:
        raise AssertionError("energetic spray did not create render-only foam")

    # The ballistic force path is pressure-free but still integrates gravity.
    mass = wp.array(mass_before, dtype=float, device=device)
    volume = wp.array(np.full(count, 0.125, dtype=np.float32), dtype=float, device=device)
    rho = wp.array(np.full(count, 1000.0, dtype=np.float32), dtype=float, device=device)
    level = wp.zeros(count, dtype=wp.int32, device=device)
    active = wp.ones(count, dtype=wp.int32, device=device)
    deferred = wp.zeros((count, 3), dtype=float, device=device)
    acceleration = wp.zeros(count, dtype=wp.vec3, device=device)
    solid_force = wp.zeros(count, dtype=wp.vec3, device=device)
    grid.build(x, 2.6)
    wp.launch(
        compute_fluid_forces_multirate, dim=count,
        inputs=[grid.id, x, v, radius, mass, volume, kind, phase, rho, level, active,
                deferred, acceleration, solid_force, 1000.0, 140.0, 1.08, 0.1, 0.02,
                2.6, 1.0e-4], device=device,
    )
    wp.synchronize_device(device)
    if not np.allclose(acceleration.numpy()[-1], (0.0, -9.81, 0.0), atol=1.0e-5):
        raise AssertionError("isolated ballistic droplet did not receive gravity-only acceleration")
    momentum_after_classification = (mass.numpy()[:, None] * v.numpy()).sum(axis=0)
    if not np.allclose(momentum_after_classification, momentum_before, atol=1.0e-5):
        raise AssertionError("phase classification changed particle momentum")

    print(
        "PASS: connected bulk, thin sheet and ballistic drop separate with hysteresis; "
        "classification preserves mass/momentum and foam remains render-only"
    )


if __name__ == "__main__":
    main()
