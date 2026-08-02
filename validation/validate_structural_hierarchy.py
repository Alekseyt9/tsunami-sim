"""CUDA regression for structural-role fracture hierarchy."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent

from hybrid_kernels import (  # noqa: E402
    accumulate_building_damage,
    collapse_gravity_fraction,
    deformable_contact_magnitude,
    facade_support_loss_rate,
    structural_damage_rate_multiplier,
    structural_failure_strain_multiplier,
)


@wp.kernel
def sample_contact_damping(
    closing_speeds: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    i = wp.tid()
    result[i] = deformable_contact_magnitude(
        0.02, closing_speeds[i], 3.0e6, 9000.0
    )


@wp.kernel
def sample_hierarchy(
    roles: wp.array(dtype=wp.int32),
    failure: wp.array(dtype=float),
    damage_rate: wp.array(dtype=float),
):
    i = wp.tid()
    failure[i] = structural_failure_strain_multiplier(roles[i])
    damage_rate[i] = structural_damage_rate_multiplier(roles[i])


@wp.kernel
def sample_collapse_gravity(
    damage_integral: wp.array(dtype=float),
    structural_volume: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    i = wp.tid()
    result[i] = collapse_gravity_fraction(
        damage_integral[i], structural_volume[i], 0.015, 0.10
    )


@wp.kernel
def sample_facade_support_loss(
    roles: wp.array(dtype=wp.int32),
    elevations: wp.array(dtype=float),
    collapse: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    i = wp.tid()
    result[i] = facade_support_loss_rate(
        roles[i], elevations[i], collapse[i], 4.0, 0.75, 2.5
    )


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    closing = wp.array(
        np.asarray([-20.0, 0.0, 20.0], dtype=np.float32), dtype=float, device=device
    )
    contact = wp.zeros(3, dtype=float, device=device)
    wp.launch(sample_contact_damping, dim=3, inputs=[closing, contact], device=device)
    contact_host = contact.numpy()
    if not (
        contact_host[0] > contact_host[1]
        and contact_host[1] == contact_host[2]
        and contact_host[2] > 0.0
    ):
        raise AssertionError(
            f"deformable contact damping is not dissipative/non-attractive: {contact_host}"
        )
    # Glass, wall, slab, beam, column, core: weak facade to ductile core.
    role_values = np.asarray([6, 2, 1, 3, 4, 5], dtype=np.int32)
    roles = wp.array(role_values, dtype=wp.int32, device=device)
    failure = wp.zeros(len(role_values), dtype=float, device=device)
    damage_rate = wp.zeros(len(role_values), dtype=float, device=device)
    wp.launch(sample_hierarchy, dim=len(role_values), inputs=[roles, failure, damage_rate], device=device)
    failure_host = failure.numpy()
    damage_rate_host = damage_rate.numpy()
    np.testing.assert_allclose(failure_host, [0.65, 1.0, 1.25, 1.6, 1.9, 2.2], atol=1.0e-6)
    np.testing.assert_allclose(damage_rate_host, [1.4, 1.0, 0.75, 0.55, 0.4, 0.3], atol=1.0e-6)
    if not np.all(np.diff(failure_host) > 0.0):
        raise AssertionError(f"failure hierarchy is not monotonic: {failure_host}")
    if not np.all(np.diff(damage_rate_host) < 0.0):
        raise AssertionError(f"damage-rate hierarchy is not monotonic: {damage_rate_host}")

    kind = wp.ones(3, dtype=wp.int32, device=device)
    building = wp.array(np.asarray([0, 0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    volume = wp.array(np.asarray([2.0, 3.0, 4.0], dtype=np.float32), dtype=float, device=device)
    damage = wp.array(np.asarray([0.25, 0.5, 1.0], dtype=np.float32), dtype=float, device=device)
    integral = wp.zeros(2, dtype=float, device=device)
    wp.launch(
        accumulate_building_damage, dim=3,
        inputs=[kind, building, volume, damage, integral], device=device,
    )
    np.testing.assert_allclose(integral.numpy(), [2.0, 4.0], atol=1.0e-6)

    collapse_damage = wp.array(
        np.asarray([0.0, 0.0575, 0.10], dtype=np.float32), dtype=float, device=device
    )
    collapse_volume = wp.ones(3, dtype=float, device=device)
    collapse_result = wp.zeros(3, dtype=float, device=device)
    wp.launch(
        sample_collapse_gravity, dim=3,
        inputs=[collapse_damage, collapse_volume, collapse_result], device=device,
    )
    np.testing.assert_allclose(collapse_result.numpy(), [0.0, 0.5, 1.0], atol=1.0e-6)
    support_roles = wp.array(
        np.asarray([2, 5, 2, 2, 6], dtype=np.int32), dtype=wp.int32, device=device
    )
    support_elevation = wp.array(
        np.asarray([3.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32), dtype=float, device=device
    )
    support_collapse = wp.array(
        np.asarray([1.0, 1.0, 0.75, 1.0, 1.0], dtype=np.float32), dtype=float, device=device
    )
    support_result = wp.zeros(5, dtype=float, device=device)
    wp.launch(
        sample_facade_support_loss, dim=5,
        inputs=[support_roles, support_elevation, support_collapse, support_result], device=device,
    )
    np.testing.assert_allclose(support_result.numpy(), [0.0, 0.0, 0.0, 2.5, 2.5], atol=1.0e-6)
    print(
        "PASS: glass < wall < slab < beam < column < core; "
        "gravity/support loss begin only after causal building damage"
    )


if __name__ == "__main__":
    main()
