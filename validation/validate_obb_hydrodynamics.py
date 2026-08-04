"""CUDA regression for conservative SPH <-> terminal-OBB coupling."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import (
    accumulate_terminal_proxy_fluid_contacts_bvh,
    accumulate_terminal_proxy_fluid_contacts_cached,
    accumulate_terminal_proxy_fluid_contacts_grid,
    apply_terminal_obb_hydrodynamics,
    build_terminal_proxy_fluid_contact_cache_bvh,
    scatter_terminal_proxy_fluid_contact_cache_active,
    update_rigid_proxy_bounds,
    update_terminal_obb_quadrature,
)


def cross_sum(points: np.ndarray, forces: np.ndarray) -> np.ndarray:
    return np.cross(points.astype(np.float64), forces.astype(np.float64)).sum(axis=0)


def run_case(
    device: str, time_level_value: int, evaluation_stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    rigid = wp.ones(1, dtype=wp.int32, device=device)
    terminal = wp.ones(1, dtype=wp.int32, device=device)
    proxy = wp.ones(1, dtype=wp.int32, device=device)
    local_center = wp.zeros(1, dtype=wp.vec3, device=device)
    half_extent = wp.array([[1.0, 0.5, 0.5]], dtype=wp.vec3, device=device)
    body_center = wp.array([[0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    orientation = wp.array(
        [[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device=device
    )
    linear = wp.array([[0.2, 0.0, 0.0]], dtype=wp.vec3, device=device)
    angular = wp.array([[0.0, 0.3, 0.0]], dtype=wp.vec3, device=device)
    body_mass = wp.array([4000.0], dtype=float, device=device)

    quadrature_active = wp.zeros(24, dtype=wp.int32, device=device)
    quadrature_body = wp.zeros(24, dtype=wp.int32, device=device)
    quadrature_position = wp.zeros(24, dtype=wp.vec3, device=device)
    quadrature_normal = wp.zeros(24, dtype=wp.vec3, device=device)
    quadrature_velocity = wp.zeros(24, dtype=wp.vec3, device=device)
    quadrature_area = wp.zeros(24, dtype=float, device=device)
    quadrature_occupancy = wp.ones(24, dtype=float, device=device)
    wp.launch(
        update_terminal_obb_quadrature,
        dim=24,
        inputs=[
            rigid, terminal, proxy, quadrature_occupancy, local_center,
            half_extent, body_center,
            orientation, linear, angular, quadrature_active, quadrature_body,
            quadrature_position, quadrature_normal, quadrature_velocity,
            quadrature_area,
        ],
        device=device,
    )

    gauss = 0.5 / np.sqrt(3.0)
    fluid_x_host = np.asarray(
        [[1.12, 1.0 + sy * gauss, sz * gauss]
         for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float32,
    )
    count = len(fluid_x_host)
    fluid_x = wp.array(fluid_x_host, dtype=wp.vec3, device=device)
    fluid_v = wp.array(
        np.tile(np.asarray([4.0, 0.6, 0.2], dtype=np.float32), (count, 1)),
        dtype=wp.vec3, device=device,
    )
    radius = wp.array(np.full(count, 0.12, dtype=np.float32), dtype=float, device=device)
    volume_host = np.full(count, 0.12 ** 3, dtype=np.float32)
    volume = wp.array(volume_host, dtype=float, device=device)
    mass_host = volume_host * 1000.0
    mass = wp.array(mass_host, dtype=float, device=device)
    kind = wp.zeros(count, dtype=wp.int32, device=device)
    phase = wp.zeros(count, dtype=wp.int32, device=device)
    pressure = wp.array(np.full(count, 20_000.0, dtype=np.float32), dtype=float, device=device)
    time_level = wp.array(
        np.full(count, time_level_value, dtype=np.int32), dtype=wp.int32, device=device
    )
    time_active = wp.ones(count, dtype=wp.int32, device=device)
    acceleration = wp.zeros(count, dtype=wp.vec3, device=device)
    body_force = wp.zeros((1, 3), dtype=float, device=device)
    body_torque = wp.zeros((1, 3), dtype=float, device=device)
    active_counter = wp.zeros(1, dtype=wp.int32, device=device)
    wet_counter = wp.zeros(1, dtype=wp.int32, device=device)
    grid = wp.HashGrid(16, 16, 16, device=device)
    grid.build(fluid_x, 0.60)
    wp.launch(
        apply_terminal_obb_hydrodynamics,
        dim=24,
        inputs=[
            grid.id, fluid_x, fluid_v, radius, mass, volume, kind, phase,
            pressure, time_level, time_active, quadrature_active,
            quadrature_body, quadrature_position, quadrature_normal,
            quadrature_velocity, quadrature_area, body_center, body_mass,
            body_force, body_torque, acceleration, active_counter, wet_counter,
            1000.0, 0.65, 45.0, 0.60, evaluation_stride,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    if int(active_counter.numpy()[0]) != 24:
        raise AssertionError("terminal OBB did not generate exactly 24 quadrature samples")
    wet = int(wet_counter.numpy()[0])
    if wet < 4 or wet >= 24:
        raise AssertionError(f"unexpected wet quadrature count: {wet}")
    area_host = quadrature_area.numpy()
    np.testing.assert_allclose(area_host.sum(), 10.0, rtol=1.0e-6, atol=1.0e-6)

    fluid_force = acceleration.numpy() * mass_host[:, None]
    force_host = body_force.numpy()[0]
    torque_host = body_torque.numpy()[0]
    stride = 1 << time_level_value
    np.testing.assert_allclose(
        force_host + fluid_force.sum(axis=0) * stride,
        0.0, rtol=3.0e-5, atol=3.0e-3,
    )
    total_angular_impulse_rate = (
        torque_host
        + np.cross(body_center.numpy()[0], force_host)
        + cross_sum(fluid_x_host, fluid_force * stride)
    )
    np.testing.assert_allclose(
        total_angular_impulse_rate, 0.0, rtol=3.0e-5, atol=5.0e-3
    )
    if np.linalg.norm(force_host) < 1.0:
        raise AssertionError("wet OBB received no meaningful hydrodynamic load")
    return force_host, torque_host


def run_analytic_contact_case(
    device: str, point: np.ndarray, time_level_value: int,
    terminal_value: int = 1, broadphase: str = "particle_bvh",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rigid = wp.ones(1, dtype=wp.int32, device=device)
    terminal = wp.array([terminal_value], dtype=wp.int32, device=device)
    proxy = wp.ones(1, dtype=wp.int32, device=device)
    local_center = wp.zeros(1, dtype=wp.vec3, device=device)
    half_extent = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=device)
    body_center = wp.zeros(1, dtype=wp.vec3, device=device)
    orientation = wp.array(
        [[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device=device
    )
    body_linear = wp.zeros(1, dtype=wp.vec3, device=device)
    body_angular = wp.zeros(1, dtype=wp.vec3, device=device)
    bounds_lower = wp.zeros(1, dtype=wp.vec3, device=device)
    bounds_upper = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        update_rigid_proxy_bounds, dim=1,
        inputs=[rigid, proxy, local_center, half_extent, body_center,
                orientation, bounds_lower, bounds_upper, 0.0], device=device,
    )
    wp.synchronize_device(device)
    bvh = wp.Bvh(bounds_lower, bounds_upper, constructor="lbvh")
    x = wp.array(np.asarray([point], dtype=np.float32), dtype=wp.vec3, device=device)
    velocity = wp.array([[-3.0, 1.0, 0.5]], dtype=wp.vec3, device=device)
    radius = wp.array([0.2], dtype=float, device=device)
    mass = wp.array([10.0], dtype=float, device=device)
    kind = wp.zeros(1, dtype=wp.int32, device=device)
    fluid_particle = wp.zeros(1, dtype=wp.int32, device=device)
    # Ballistic droplets must hit the analytical proxy too.
    phase = wp.array([2], dtype=wp.int32, device=device)
    time_level = wp.array([time_level_value], dtype=wp.int32, device=device)
    time_active = wp.ones(1, dtype=wp.int32, device=device)
    body_force = wp.zeros((1, 3), dtype=float, device=device)
    body_torque = wp.zeros((1, 3), dtype=float, device=device)
    acceleration = wp.zeros(1, dtype=wp.vec3, device=device)
    candidate_counter = wp.zeros(1, dtype=wp.int32, device=device)
    contact_counter = wp.zeros(1, dtype=wp.int32, device=device)
    query_particle_counter = wp.zeros(1, dtype=wp.int32, device=device)
    global_lower = wp.array(
        np.asarray([[-1.0, -1.0, -1.0]], dtype=np.float32),
        dtype=float, device=device,
    )
    global_upper = wp.array(
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        dtype=float, device=device,
    )
    if broadphase == "fluid_grid":
        fluid_grid = wp.HashGrid(16, 16, 16, device=device)
        fluid_grid.build(x, 0.6)
        wp.launch(
            accumulate_terminal_proxy_fluid_contacts_grid, dim=1,
            inputs=[
                fluid_grid.id, x, fluid_particle, x, velocity, radius, mass,
                time_level, time_active, rigid, terminal, proxy, local_center,
                half_extent, body_center, orientation, body_linear,
                body_angular, body_force, body_torque, acceleration,
                candidate_counter, contact_counter, 2.0e5, 3000.0, 500.0,
                0.08, 0.25, 2500.0, 0.0, 0.6,
            ], device=device,
        )
    elif broadphase == "cached_particle_bvh":
        quadrature_occupancy = wp.ones(24, dtype=float, device=device)
        cache_body = wp.zeros((1, 4), dtype=wp.int32, device=device)
        cache_body_count = wp.zeros(1, dtype=wp.int32, device=device)
        cache_active_flag = wp.zeros(1, dtype=wp.int32, device=device)
        cache_active_offset = wp.zeros(1, dtype=wp.int32, device=device)
        cache_active_slot = wp.zeros(1, dtype=wp.int32, device=device)
        overflow = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            build_terminal_proxy_fluid_contact_cache_bvh, dim=1,
            inputs=[
                bvh.id, x, radius, kind, fluid_particle, rigid, terminal,
                proxy, global_lower, global_upper, cache_body,
                cache_body_count, cache_active_flag, query_particle_counter,
                candidate_counter, overflow, 0.0, 0.1, 4,
            ], device=device,
        )
        wp.utils.array_scan(
            cache_active_flag, cache_active_offset, inclusive=False
        )
        wp.launch(
            scatter_terminal_proxy_fluid_contact_cache_active, dim=1,
            inputs=[cache_active_flag, cache_active_offset, cache_active_slot],
            device=device,
        )
        wp.launch(
            accumulate_terminal_proxy_fluid_contacts_cached, dim=1,
            inputs=[
                x, velocity, radius, mass, kind, fluid_particle, time_level,
                time_active, rigid, terminal, proxy, quadrature_occupancy,
                local_center, half_extent,
                body_center, orientation, body_linear, body_angular, body_force,
                body_torque, acceleration, cache_body, cache_body_count,
                cache_active_slot, contact_counter, 2.0e5, 3000.0, 500.0,
                0.08, 0.25, 2500.0, 0.0, 1,
            ], device=device,
        )
        if int(overflow.numpy()[0]) != 0:
            raise AssertionError("isolated analytical contact cache overflowed")
    else:
        wp.launch(
            accumulate_terminal_proxy_fluid_contacts_bvh, dim=1,
            inputs=[
                bvh.id, x, velocity, radius, mass, kind, fluid_particle, phase,
                time_level, time_active, rigid, terminal, proxy, local_center,
                half_extent, body_center, orientation, body_linear, body_angular,
                body_force, body_torque, acceleration, global_lower,
                global_upper, query_particle_counter, candidate_counter,
                contact_counter, 2.0e5, 3000.0, 500.0, 0.08, 0.25, 2500.0, 0.0,
                1,
            ], device=device,
        )
    wp.synchronize_device(device)
    if terminal_value:
        if int(candidate_counter.numpy()[0]) != 1 or int(contact_counter.numpy()[0]) != 1:
            raise AssertionError("expanded OBB BVH missed a fluid contact")
    elif int(contact_counter.numpy()[0]) != 0:
        raise AssertionError("analytical contact included a non-terminal proxy")
    return acceleration.numpy()[0], body_force.numpy()[0], body_torque.numpy()[0]


def main() -> None:
    wp.init()
    device = "cuda:0"
    base_force, base_torque = run_case(device, 0)
    slow_force, slow_torque = run_case(device, 2)
    sparse_force, sparse_torque = run_case(device, 0, 4)
    np.testing.assert_allclose(slow_force, base_force * 4.0, rtol=3.0e-5, atol=5.0e-3)
    np.testing.assert_allclose(slow_torque, base_torque * 4.0, rtol=3.0e-5, atol=5.0e-3)
    np.testing.assert_allclose(sparse_force, base_force * 4.0, rtol=3.0e-5, atol=5.0e-3)
    np.testing.assert_allclose(sparse_torque, base_torque * 4.0, rtol=3.0e-5, atol=5.0e-3)
    outside_point = np.asarray([1.10, 0.20, 0.0], dtype=np.float32)
    contact_acceleration, contact_body_force, contact_body_torque = (
        run_analytic_contact_case(device, outside_point, 0)
    )
    fluid_force = contact_acceleration * 10.0
    np.testing.assert_allclose(
        contact_body_force + fluid_force, 0.0, rtol=3.0e-5, atol=3.0e-3
    )
    np.testing.assert_allclose(
        contact_body_torque + np.cross(outside_point, fluid_force),
        0.0, rtol=3.0e-5, atol=3.0e-3,
    )
    if contact_acceleration[0] <= 0.0 or contact_acceleration[1] >= 0.0:
        raise AssertionError("analytical OBB normal/friction direction is wrong")
    grid_acceleration, grid_body_force, grid_body_torque = (
        run_analytic_contact_case(
            device, outside_point, 0, broadphase="fluid_grid"
        )
    )
    np.testing.assert_allclose(grid_acceleration, contact_acceleration, rtol=3.0e-5)
    np.testing.assert_allclose(grid_body_force, contact_body_force, rtol=3.0e-5)
    np.testing.assert_allclose(grid_body_torque, contact_body_torque, rtol=3.0e-5)
    cached_acceleration, cached_body_force, cached_body_torque = (
        run_analytic_contact_case(
            device, outside_point, 0, broadphase="cached_particle_bvh"
        )
    )
    np.testing.assert_allclose(cached_acceleration, contact_acceleration, rtol=3.0e-5)
    np.testing.assert_allclose(cached_body_force, contact_body_force, rtol=3.0e-5)
    np.testing.assert_allclose(cached_body_torque, contact_body_torque, rtol=3.0e-5)
    slow_acceleration, slow_body_force, _ = run_analytic_contact_case(
        device, outside_point, 2
    )
    np.testing.assert_allclose(slow_acceleration, contact_acceleration, rtol=3.0e-5)
    np.testing.assert_allclose(slow_body_force, contact_body_force * 4.0, rtol=3.0e-5)
    inside_acceleration, _, _ = run_analytic_contact_case(
        device, np.asarray([0.0, 0.0, 0.0], dtype=np.float32), 0
    )
    if inside_acceleration[0] <= 0.0:
        raise AssertionError("particle inside OBB was not expelled through nearest face")
    ignored_acceleration, _, _ = run_analytic_contact_case(
        device, outside_point, 0, terminal_value=0
    )
    np.testing.assert_allclose(ignored_acceleration, 0.0, atol=1.0e-8)
    print(
        "PASS: OBB quadrature/contact conserves linear/angular impulse, preserves "
        "multirate stride, validates the persistent candidate cache, catches "
        "ballistic water, and expels interior particles"
    )


if __name__ == "__main__":
    main()
