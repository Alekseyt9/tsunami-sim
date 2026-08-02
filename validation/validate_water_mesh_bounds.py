"""Regression for late spray expanding the main water reconstruction box."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np

from deluge_v3 import limit_water_core_height, robust_axis_bounds


def main() -> None:
    rng = np.random.default_rng(42)
    core = rng.uniform((-70.0, 0.0, -54.0), (70.0, 21.0, 72.0), size=(100_000, 3)).astype(np.float32)
    spray = rng.uniform((-65.0, 35.0, 90.0), (65.0, 88.0, 130.0), size=(200, 3)).astype(np.float32)
    positions = np.concatenate((core, spray), axis=0)
    lower, upper = robust_axis_bounds(positions, [0.0, 0.0, 0.0], [1.0, 0.995, 0.9975])
    excluded = np.any((positions < lower) | (positions > upper), axis=1)
    if upper[1] > 22.0 or upper[2] > 75.0:
        raise AssertionError(f"spray still controls robust upper bound: {upper}")
    if lower[0] > -69.0 or upper[0] < 69.0:
        raise AssertionError(f"main reservoir width was cropped: x=[{lower[0]}, {upper[0]}]")
    if int(np.count_nonzero(excluded)) < len(spray):
        raise AssertionError("not all synthetic detached spray was excluded from the mesh box")

    uncapped_lower = np.array([-70.0, 0.0, -54.0], dtype=np.float32)
    uncapped_upper = np.array([70.0, 78.0, 130.0], dtype=np.float32)
    capped_lower, capped_upper = limit_water_core_height(
        uncapped_lower, uncapped_upper, 42.0
    )
    np.testing.assert_array_equal(capped_lower, uncapped_lower)
    np.testing.assert_allclose(capped_upper, [70.0, 42.0, 130.0])
    print(
        f"PASS: robust mesh box upper={upper}; excluded {int(np.count_nonzero(excluded)):,}/"
        f"{len(positions):,} samples without cropping reservoir width; "
        f"connected core capped at y={capped_upper[1]:.1f} m"
    )


if __name__ == "__main__":
    main()
