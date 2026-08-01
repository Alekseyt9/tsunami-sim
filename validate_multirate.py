"""CUDA conservation test for the V3 fast/slow fluid interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp


from hybrid_kernels import (  # noqa: E402
    compute_fluid_forces_multirate,
    consume_deferred_fluid_impulse,
    integrate_multirate,
    select_active_time_level,
)


def simulate(levels_host: np.ndarray, steps: int = 4):
    device = "cuda:0"
    position = wp.array(np.asarray([[-0.32, 10.0, 0.0], [0.32, 10.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    radius = wp.array(np.full(2, 0.5, dtype=np.float32), dtype=float, device=device)
    mass = wp.array(np.full(2, 1000.0, dtype=np.float32), dtype=float, device=device)
    volume = wp.array(np.ones(2, dtype=np.float32), dtype=float, device=device)
    kind = wp.array(np.zeros(2, dtype=np.int32), dtype=wp.int32, device=device)
    rho = wp.array(np.full(2, 1012.0, dtype=np.float32), dtype=float, device=device)
    levels = wp.array(levels_host.astype(np.int32), dtype=wp.int32, device=device)
    active = wp.ones(2, dtype=wp.int32, device=device)
    deferred = wp.zeros((2, 3), dtype=float, device=device)
    acceleration = wp.zeros(2, dtype=wp.vec3, device=device)
    solid_force = wp.zeros(2, dtype=wp.vec3, device=device)
    fixed = wp.zeros(2, dtype=wp.int32, device=device)
    grid = wp.HashGrid(16, 16, 16, device=device)
    dt = 1.0e-4
    for tick in range(steps):
        grid.build(position, 2.0)
        wp.launch(select_active_time_level, dim=2, inputs=[levels, kind, tick, active], device=device)
        wp.launch(
            compute_fluid_forces_multirate,
            dim=2,
            inputs=[
                grid.id, position, velocity, radius, mass, volume, kind, rho, levels, active,
                deferred, acceleration, solid_force, 1000.0, 140.0, 1.08, 0.0, 0.0, 2.0, dt,
            ],
            device=device,
        )
        wp.launch(
            consume_deferred_fluid_impulse,
            dim=2,
            inputs=[mass, kind, levels, active, deferred, acceleration, dt],
            device=device,
        )
        wp.launch(
            integrate_multirate,
            dim=2,
            inputs=[
                position, velocity, acceleration, kind, fixed, levels, active, dt,
                100.0, -100.0, 100.0, 100.0, 0.0,
            ],
            device=device,
        )
    wp.synchronize_device(device)
    return position.numpy(), velocity.numpy(), deferred.numpy()


def main() -> None:
    wp.init()
    reference_x, reference_v, _ = simulate(np.asarray([0, 0], dtype=np.int32))
    multirate_x, multirate_v, deferred = simulate(np.asarray([0, 1], dtype=np.int32))
    momentum = (multirate_v * 1000.0).sum(axis=0)
    expected_external_impulse = np.asarray([0.0, -2.0 * 1000.0 * 9.81 * 4.0e-4, 0.0])
    momentum_scale = max(float(np.linalg.norm(multirate_v * 1000.0, axis=1).sum()), 1.0)
    relative_momentum_error = float(np.linalg.norm(momentum - expected_external_impulse) / momentum_scale)
    velocity_error = float(np.max(np.abs(multirate_v - reference_v)))
    position_error = float(np.max(np.abs(multirate_x - reference_x)))
    if relative_momentum_error > 1.0e-5:
        raise AssertionError(f"fast/slow momentum error is {relative_momentum_error:.3e}")
    if np.max(np.abs(deferred)) > 1.0e-7:
        raise AssertionError("deferred impulse was not fully consumed at synchronization")
    if velocity_error > 0.01 or position_error > 1.0e-4:
        raise AssertionError("multirate trajectory diverged excessively from uniform stepping")
    print(
        f"PASS: fast/slow momentum error={relative_momentum_error:.3e}, "
        f"velocity delta={velocity_error:.3e} m/s, position delta={position_error:.3e} m"
    )


if __name__ == "__main__":
    main()
