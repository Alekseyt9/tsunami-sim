"""CUDA regression for equal/opposite frictional rubble contacts."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from hybrid_kernels import (
    accumulate_rigid_contacts,
    clear_body_accumulators,
    reactivate_rigid_after_impact,
)


def main():
    wp.init()
    device = "cuda:0"
    position_host = np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    position = wp.array(position_host, dtype=wp.vec3, device=device)
    velocity = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=wp.vec3, device=device)
    radius = wp.array([0.6, 0.6], dtype=float, device=device)
    kind = wp.ones(2, dtype=wp.int32, device=device)
    material = wp.array([1, 1], dtype=wp.int32, device=device)
    fragment = wp.array([0, 1], dtype=wp.int32, device=device)
    rigid = wp.ones(2, dtype=wp.int32, device=device)
    centers = wp.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=wp.vec3, device=device)
    body_mass = wp.array([1000.0, 1000.0], dtype=float, device=device)
    force = wp.zeros((2, 3), dtype=float, device=device)
    torque = wp.zeros((2, 3), dtype=float, device=device)
    peak = wp.zeros(2, dtype=float, device=device)
    reactivated = wp.zeros(1, dtype=wp.int32, device=device)
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.build(position, 1.5)
    wp.launch(clear_body_accumulators, dim=2, inputs=[force, torque], device=device)
    wp.launch(
        accumulate_rigid_contacts, dim=2,
        inputs=[grid.id, position, velocity, radius, kind, material, fragment, rigid,
                centers, body_mass, force, torque, peak, 1.5, 3200.0, 1800.0],
        device=device,
    )
    wp.synchronize_device(device)
    force_host = force.numpy()
    torque_host = torque.numpy()
    np.testing.assert_allclose(force_host[0] + force_host[1], 0.0, atol=1.0e-4)
    np.testing.assert_allclose(torque_host[0] + torque_host[1], 0.0, atol=1.0e-4)
    if force_host[0, 0] >= 0.0 or force_host[0, 2] <= 0.0:
        raise AssertionError(f"unexpected normal/friction directions: {force_host[0]}")
    if not np.all(peak.numpy() > 120.0):
        raise AssertionError("impact peak did not reach deformable-reactivation threshold")
    wp.launch(
        reactivate_rigid_after_impact, dim=2,
        inputs=[rigid, peak, reactivated, 120.0], device=device,
    )
    wp.synchronize_device(device)
    if np.any(rigid.numpy() != 0) or int(reactivated.numpy()[0]) != 2:
        raise AssertionError("strong contact did not reactivate both cohesive fragments")
    print(
        f"PASS: rigid contact conserves force/torque, applies friction and reactivates "
        f"2 fragments; peak={peak.numpy().min():.1f} m/s^2"
    )


if __name__ == "__main__":
    main()
