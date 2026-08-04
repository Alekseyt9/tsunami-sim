"""CUDA regression for convex OBB rigid-debris collision proxies."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import (
    accumulate_rigid_sample_bottom,
    accumulate_terminal_proxy_bottom,
    accumulate_rigid_gravity,
    accumulate_rigid_proxy_boundaries,
    accumulate_rigid_proxy_contacts,
    accumulate_rigid_proxy_contacts_bvh,
    clear_body_accumulators,
    clear_rigid_sample_bottom,
    project_rigid_samples_above_ground,
    reactivate_rigid_after_impact,
    update_rigid_proxy_bounds,
)
from simulation.rigid_clusters import fit_rigid_collision_proxy


def main() -> None:
    wp.init()
    device = "cuda:0"
    local_samples = np.asarray(
        [[-0.8, -0.3, -0.4], [0.9, 0.35, 0.45], [0.2, -0.1, 0.0]], dtype=np.float32
    )
    fitted = fit_rigid_collision_proxy(
        local_samples, np.full(3, 0.1, dtype=np.float32),
        np.asarray([1, 1, 3], dtype=np.int32), np.asarray([5.0, 5.0, 1.0]), 0.7,
    )
    lower = fitted.local_center - fitted.half_extent
    upper = fitted.local_center + fitted.half_extent
    if np.any(local_samples < lower) or np.any(local_samples > upper) or fitted.material != 1:
        raise AssertionError("fitted proxy does not enclose its weighted structural samples")

    rigid = wp.ones(2, dtype=wp.int32, device=device)
    enabled = wp.ones(2, dtype=wp.int32, device=device)
    local_center = wp.zeros(2, dtype=wp.vec3, device=device)
    extent = wp.array([[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=wp.vec3, device=device)
    material = wp.ones(2, dtype=wp.int32, device=device)
    center = wp.array([[0.0, 1.0, 0.0], [1.6, 1.0, 0.0]], dtype=wp.vec3, device=device)
    orientation = wp.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device=device
    )
    linear = wp.array([[0.0, 0.0, 0.0], [-8.0, 0.0, 2.0]], dtype=wp.vec3, device=device)
    angular = wp.zeros(2, dtype=wp.vec3, device=device)
    mass = wp.array([1000.0, 1000.0], dtype=float, device=device)
    force = wp.zeros((2, 3), dtype=float, device=device)
    torque = wp.zeros((2, 3), dtype=float, device=device)
    peak = wp.zeros(2, dtype=float, device=device)
    pair_left = wp.array([0], dtype=wp.int32, device=device)
    pair_right = wp.array([1], dtype=wp.int32, device=device)
    wp.launch(clear_body_accumulators, dim=2, inputs=[force, torque], device=device)
    wp.launch(
        accumulate_rigid_proxy_contacts, dim=1,
        inputs=[pair_left, pair_right, rigid, enabled, local_center, extent, material,
                center, orientation, linear, angular, mass, force, torque, peak,
                3200.0, 0.9, 1800.0, 0.10, 45.0, 6.0], device=device,
    )
    wp.synchronize_device(device)
    force_host = force.numpy()
    torque_host = torque.numpy()
    np.testing.assert_allclose(force_host[0] + force_host[1], 0.0, atol=1.0e-4)
    center_host = center.numpy()
    world_torque = (
        torque_host[0] + np.cross(center_host[0], force_host[0])
        + torque_host[1] + np.cross(center_host[1], force_host[1])
    )
    np.testing.assert_allclose(world_torque, 0.0, atol=2.0e-3)
    if force_host[0, 0] >= 0.0 or force_host[0, 2] <= 0.0:
        raise AssertionError(f"proxy normal/friction direction is wrong: {force_host[0]}")
    if not np.all((peak.numpy() > 30.0) & (peak.numpy() <= 45.0001)):
        raise AssertionError("proxy impact did not reach the deformable-reactivation threshold")

    # The GPU BVH broadphase must produce the same narrowphase forces as the
    # legacy all-pairs path while visiting only overlapping AABBs.
    legacy_force = force_host.copy()
    legacy_torque = torque_host.copy()
    legacy_peak = peak.numpy().copy()
    force.zero_()
    torque.zero_()
    peak.zero_()
    bounds_lower = wp.zeros(2, dtype=wp.vec3, device=device)
    bounds_upper = wp.zeros(2, dtype=wp.vec3, device=device)
    candidate_count = wp.zeros(1, dtype=wp.int32, device=device)
    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        update_rigid_proxy_bounds, dim=2,
        inputs=[rigid, enabled, local_center, extent, center, orientation,
                bounds_lower, bounds_upper, 0.03], device=device,
    )
    wp.synchronize_device(device)
    bvh = wp.Bvh(bounds_lower, bounds_upper, constructor="lbvh")
    wp.launch(
        accumulate_rigid_proxy_contacts_bvh, dim=2,
        inputs=[bvh.id, bounds_lower, bounds_upper, rigid, enabled, local_center,
                extent, material, center, orientation, linear, angular, mass,
                force, torque, peak, candidate_count, contact_count,
                3200.0, 0.9, 1800.0, 0.10, 45.0, 6.0], device=device,
    )
    wp.synchronize_device(device)
    np.testing.assert_allclose(force.numpy(), legacy_force, rtol=2.0e-5, atol=1.0e-3)
    np.testing.assert_allclose(torque.numpy(), legacy_torque, rtol=2.0e-5, atol=1.0e-3)
    np.testing.assert_allclose(peak.numpy(), legacy_peak, rtol=2.0e-5, atol=1.0e-5)
    if int(candidate_count.numpy()[0]) != 1 or int(contact_count.numpy()[0]) != 1:
        raise AssertionError("BVH did not emit exactly one overlapping proxy pair")

    # A separate body crosses the ground only through its proxy. The body-level
    # contact must push it upward without requiring any particle boundary splats.
    ground_center = wp.array([[0.0, 0.35, 0.0]], dtype=wp.vec3, device=device)
    ground_force = wp.zeros((1, 3), dtype=float, device=device)
    ground_torque = wp.zeros((1, 3), dtype=float, device=device)
    wp.launch(
        accumulate_rigid_proxy_boundaries, dim=1,
        inputs=[rigid, enabled, local_center, extent, material, ground_center, orientation,
                linear, angular, mass, ground_force, ground_torque,
                100.0, -100.0, 100.0, 100.0,
                4.0e6, 1.8e4, 0.9, 1800.0, 0.10, 45.0], device=device,
    )
    wp.synchronize_device(device)
    if ground_force.numpy()[0, 1] <= 0.0:
        raise AssertionError("ground proxy did not generate an upward support force")

    # Sample shedding must retain weight and an analytical ground floor even
    # though no render particle contributes body loads or sample minima.
    terminal_force = wp.zeros((1, 3), dtype=float, device=device)
    terminal_flag_one = wp.ones(1, dtype=wp.int32, device=device)
    proxy_one = wp.ones(1, dtype=wp.int32, device=device)
    mass_one = wp.array([1000.0], dtype=float, device=device)
    wp.launch(
        accumulate_rigid_gravity, dim=1,
        inputs=[terminal_flag_one, mass_one, terminal_force, 9.81], device=device,
    )
    proxy_bottom = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        accumulate_terminal_proxy_bottom, dim=1,
        inputs=[terminal_flag_one, terminal_flag_one, proxy_one, local_center,
                extent, ground_center, orientation, proxy_bottom], device=device,
    )
    wp.synchronize_device(device)
    np.testing.assert_allclose(terminal_force.numpy()[0, 1], -9810.0, atol=1.0e-3)
    np.testing.assert_allclose(proxy_bottom.numpy()[0], -0.15, atol=1.0e-6)

    # Penalty contact alone used to let a very massive body keep tunnelling
    # once its capped spring force fell below its weight. The post-contact
    # sample-union projection must place the actual lowest sample on y=0.
    penetrating_center = wp.array([[0.0, -1.0, 0.0]], dtype=wp.vec3, device=device)
    penetrating_linear = wp.array([[2.0, -5.0, 1.0]], dtype=wp.vec3, device=device)
    penetrating_angular = wp.array([[0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    sample_bottom = wp.zeros(1, dtype=float, device=device)
    sample_radius = wp.array([0.2], dtype=float, device=device)
    sample_kind = wp.ones(1, dtype=wp.int32, device=device)
    sample_fragment = wp.zeros(1, dtype=wp.int32, device=device)
    sample_local = wp.zeros(1, dtype=wp.vec3, device=device)
    one_rigid = wp.ones(1, dtype=wp.int32, device=device)
    one_terminal = wp.zeros(1, dtype=wp.int32, device=device)
    one_proxy_disabled = wp.zeros(1, dtype=wp.int32, device=device)
    one_orientation = wp.array(
        [[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device=device
    )
    wp.launch(clear_rigid_sample_bottom, dim=1, inputs=[sample_bottom], device=device)
    wp.launch(
        accumulate_rigid_sample_bottom, dim=1,
        inputs=[sample_radius, sample_kind, sample_fragment, one_rigid,
                one_terminal, one_proxy_disabled, sample_local,
                penetrating_center, one_orientation, sample_bottom, 0], device=device,
    )
    wp.launch(
        project_rigid_samples_above_ground, dim=1,
        inputs=[one_rigid, sample_bottom, penetrating_center, penetrating_linear,
                penetrating_angular, 0.985], device=device,
    )
    wp.synchronize_device(device)
    projected_center = penetrating_center.numpy()[0]
    projected_velocity = penetrating_linear.numpy()[0]
    if projected_center[1] < 0.19999 or projected_velocity[1] < -1.0e-6:
        raise AssertionError(
            f"rigid sample ground projection failed: center={projected_center}, "
            f"velocity={projected_velocity}"
        )

    reactivated = wp.zeros(1, dtype=wp.int32, device=device)
    deferred = wp.zeros(2, dtype=wp.int32, device=device)
    rigid_age = wp.zeros(2, dtype=wp.int32, device=device)
    terminal = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        reactivate_rigid_after_impact, dim=2,
        inputs=[rigid, terminal, rigid_age, peak, reactivated, deferred,
                32.0, 100, 50.0], device=device,
    )
    wp.synchronize_device(device)
    if np.any(rigid.numpy() != 1) or int(np.sum(deferred.numpy())) != 2:
        raise AssertionError("fresh rigid fragments bypassed the reactivation dwell")
    rigid_age.assign(np.full(2, 100, dtype=np.int32))
    wp.launch(
        reactivate_rigid_after_impact, dim=2,
        inputs=[rigid, terminal, rigid_age, peak, reactivated, deferred,
                32.0, 100, 50.0], device=device,
    )
    wp.synchronize_device(device)
    if np.any(rigid.numpy() != 0) or int(reactivated.numpy()[0]) != 2:
        raise AssertionError("strong proxy contact did not restore deformable fragments")
    terminal_rigid = wp.ones(1, dtype=wp.int32, device=device)
    terminal_flag = wp.ones(1, dtype=wp.int32, device=device)
    terminal_age = wp.array([100], dtype=wp.int32, device=device)
    terminal_peak = wp.array([800.0], dtype=float, device=device)
    terminal_reactivated = wp.zeros(1, dtype=wp.int32, device=device)
    terminal_deferred = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        reactivate_rigid_after_impact, dim=1,
        inputs=[terminal_rigid, terminal_flag, terminal_age, terminal_peak,
                terminal_reactivated, terminal_deferred, 32.0, 0, 96.0],
        device=device,
    )
    wp.synchronize_device(device)
    if int(terminal_rigid.numpy()[0]) != 1 or int(terminal_reactivated.numpy()[0]) != 0:
        raise AssertionError("terminal plastic rubble expanded back to deformable state")
    print(
        "PASS: fitted convex OBB encloses its samples; SAT contact conserves force, "
        "applies friction/ground support, dwell-gates cohesive reactivation, and locks plastic rubble"
    )


if __name__ == "__main__":
    main()
