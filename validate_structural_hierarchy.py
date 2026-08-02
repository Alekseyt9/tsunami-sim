"""CUDA regression for structural-role fracture hierarchy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent

from hybrid_kernels import (  # noqa: E402
    structural_damage_rate_multiplier,
    structural_failure_strain_multiplier,
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


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
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
    print("PASS: glass < wall < slab < beam < column < core fracture resistance on CUDA")


if __name__ == "__main__":
    main()
