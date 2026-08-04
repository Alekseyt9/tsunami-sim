"""CUDA regression test for momentum-preserving V3 rigid-cluster LOD."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import (  # noqa: E402
    accumulate_rigid_body_loads,
    clear_body_accumulators,
    integrate_rigid_bodies,
    scatter_rigid_particles,
)
from simulation.rigid_clusters import fit_rigid_cluster, fit_rigid_cluster_to_reference  # noqa: E402


def main() -> None:
    wp.init()
    device = "cuda:0"
    local = np.asarray(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float32,
    )
    center = np.asarray([3.0, 12.0, 7.0], dtype=np.float32)
    linear = np.asarray([2.0, -0.4, 1.25], dtype=np.float32)
    omega = np.asarray([0.3, -0.2, 0.45], dtype=np.float32)
    position = center + local
    velocity = linear + np.cross(omega[None, :], local)
    mass = np.linspace(2.0, 3.4, len(local), dtype=np.float32)
    fit = fit_rigid_cluster(position, velocity, mass)

    if fit.internal_velocity_rms > 2.0e-6:
        raise AssertionError(f"rigid fit residual is {fit.internal_velocity_rms}")
    momentum_linear = (mass[:, None] * velocity).sum(axis=0) / mass.sum()
    if not np.allclose(fit.linear_velocity, momentum_linear, atol=2.0e-6):
        raise AssertionError("linear momentum was not preserved by the fit")
    if not np.allclose(fit.angular_velocity, omega, atol=2.0e-6):
        raise AssertionError("angular momentum was not preserved by the fit")

    # A deformed cloud must recover the small undeformed collision shape while
    # keeping its current centre. This guards against multi-storey proxies
    # fitted from stretched particles late in a collapse.
    angle = np.deg2rad(27.0)
    rotation = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
         [-np.sin(angle), 0.0, np.cos(angle)]], dtype=np.float32
    )
    distorted = center + (rotation @ local.T).T
    distorted[0] += np.asarray([12.0, 0.0, 0.0], dtype=np.float32)
    reference_fit = fit_rigid_cluster_to_reference(distorted, center + local, velocity, mass)
    reference_extent = np.ptp(reference_fit.local_positions, axis=0)
    if float(np.max(reference_extent)) > 2.001:
        raise AssertionError(f"reference fit retained a stretched extent: {reference_extent}")
    if not np.all(np.isfinite(reference_fit.orientation)):
        raise AssertionError("reference fit produced an invalid orientation")
    expected_center = (mass[:, None] * distorted).sum(axis=0) / mass.sum()
    if not np.allclose(reference_fit.center, expected_center, atol=2.0e-6):
        raise AssertionError("reference fit did not preserve the current mass centre")

    count = len(local)
    acceleration_value = np.asarray([0.8, -1.5, 0.35], dtype=np.float32)
    acceleration = np.repeat(acceleration_value[None, :], count, axis=0)
    x = wp.array(position, dtype=wp.vec3, device=device)
    v = wp.array(velocity, dtype=wp.vec3, device=device)
    radius = wp.array(np.full(count, 0.2, dtype=np.float32), dtype=float, device=device)
    mass_gpu = wp.array(mass, dtype=float, device=device)
    kind = wp.array(np.ones(count, dtype=np.int32), dtype=wp.int32, device=device)
    fragment = wp.array(np.zeros(count, dtype=np.int32), dtype=wp.int32, device=device)
    rigid_state = wp.array(np.ones(1, dtype=np.int32), dtype=wp.int32, device=device)
    rigid_terminal = wp.zeros(1, dtype=wp.int32, device=device)
    proxy_enabled = wp.zeros(1, dtype=wp.int32, device=device)
    acceleration_gpu = wp.array(acceleration, dtype=wp.vec3, device=device)
    local_gpu = wp.array(fit.local_positions, dtype=wp.vec3, device=device)
    body_center = wp.array(fit.center[None, :], dtype=wp.vec3, device=device)
    body_orientation = wp.array(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.quat, device=device)
    body_linear = wp.array(fit.linear_velocity[None, :], dtype=wp.vec3, device=device)
    body_angular = wp.array(fit.angular_velocity[None, :], dtype=wp.vec3, device=device)
    body_mass = wp.array(np.asarray([fit.mass], dtype=np.float32), dtype=float, device=device)
    body_inverse_inertia = wp.array(fit.inverse_inertia[None, :, :], dtype=wp.mat33, device=device)
    body_half_extent = wp.array(
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    body_force = wp.zeros((1, 3), dtype=float, device=device)
    body_torque = wp.zeros((1, 3), dtype=float, device=device)

    dt = 0.01
    wp.launch(clear_body_accumulators, dim=1, inputs=[body_force, body_torque], device=device)
    wp.launch(
        accumulate_rigid_body_loads,
        dim=count,
        inputs=[
            x, v, radius, mass_gpu, kind, fragment, rigid_state, rigid_terminal,
            proxy_enabled, acceleration_gpu,
            body_center, body_force, body_torque, 100.0, -100.0, 100.0, 100.0,
            4.0e6, 1.8e4, 0,
        ],
        device=device,
    )
    wp.launch(
        integrate_rigid_bodies,
        dim=1,
        inputs=[
            rigid_state, body_center, body_orientation, body_linear, body_angular,
            body_mass, body_inverse_inertia, body_half_extent,
            body_force, body_torque, dt,
            0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 100.0,
        ],
        device=device,
    )
    wp.launch(
        scatter_rigid_particles,
        dim=count,
        inputs=[
            x, v, kind, fragment, rigid_state, rigid_terminal, proxy_enabled,
            local_gpu, body_center, body_orientation, body_linear, body_angular, 0,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    expected_velocity = fit.linear_velocity + acceleration_value * dt
    expected_center = fit.center + expected_velocity * dt
    actual_velocity = body_linear.numpy()[0]
    actual_center = body_center.numpy()[0]
    if not np.allclose(actual_velocity, expected_velocity, atol=3.0e-6):
        raise AssertionError(f"body velocity {actual_velocity}, expected {expected_velocity}")
    if not np.allclose(actual_center, expected_center, atol=3.0e-6):
        raise AssertionError(f"body center {actual_center}, expected {expected_center}")

    moved = x.numpy()
    initial_distances = np.linalg.norm(position[:, None, :] - position[None, :, :], axis=2)
    final_distances = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=2)
    shape_error = float(np.max(np.abs(final_distances - initial_distances)))
    if shape_error > 3.0e-5:
        raise AssertionError(f"rigid shape drift is {shape_error}")
    momentum_error = float(np.max(np.abs(fit.mass * actual_velocity - fit.mass * expected_velocity)))

    # A long fragment must use a size-dependent angular cap.  Otherwise even
    # a nominal 18 rad/s limit launches a ten-metre tip at cinematic speeds.
    wp.copy(
        body_linear,
        wp.array(np.asarray([[40.0, 20.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device),
    )
    wp.copy(
        body_angular,
        wp.array(np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device),
    )
    wp.copy(
        body_half_extent,
        wp.array(np.asarray([[10.0, 1.0, 1.0]], dtype=np.float32), dtype=wp.vec3, device=device),
    )
    wp.copy(
        body_mass,
        wp.array(np.asarray([500000.0], dtype=np.float32), dtype=float, device=device),
    )
    wp.launch(clear_body_accumulators, dim=1, inputs=[body_force, body_torque], device=device)
    wp.launch(
        integrate_rigid_bodies,
        dim=1,
        inputs=[
            rigid_state, body_center, body_orientation, body_linear, body_angular,
            body_mass, body_inverse_inertia, body_half_extent, body_force, body_torque, dt,
            0.0, 0.0, 3.0, 22.0, 6.0, 50000.0, 1.5, 10.0,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    capped_linear = body_linear.numpy()[0]
    capped_angular = body_angular.numpy()[0]
    expected_heavy_upward_limit = 6.0 * np.sqrt(50000.0 / 500000.0)
    if (
        np.linalg.norm(capped_linear) > 22.0001
        or capped_linear[1] > expected_heavy_upward_limit + 1.0e-4
    ):
        raise AssertionError(f"rigid linear speed cap failed: {capped_linear}")
    expected_angular_limit = 10.0 / np.linalg.norm([10.0, 1.0, 1.0])
    if np.linalg.norm(capped_angular) > expected_angular_limit + 1.0e-4:
        raise AssertionError(f"size-dependent rigid tip-speed cap failed: {capped_angular}")
    print(f"PASS: CUDA rigid cluster kept 8-particle shape; max distance error={shape_error:.3e}")
    print(
        f"linear momentum error={momentum_error:.3e}; fit residual={fit.internal_velocity_rms:.3e} m/s; "
        f"capped |v|={np.linalg.norm(capped_linear):.2f}, |omega|={np.linalg.norm(capped_angular):.2f}"
    )


if __name__ == "__main__":
    main()
