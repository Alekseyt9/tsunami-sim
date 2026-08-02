"""CPU validation for temporally stable mesh bounds and splash-brick selection."""

from __future__ import annotations

import numpy as np

from deluge_v3 import hysteretic_bounds, select_splash_bricks


def main():
    previous_lower = np.array([-10.0, -1.0, -20.0], dtype=np.float32)
    previous_upper = np.array([10.0, 25.0, 50.0], dtype=np.float32)

    # New spray expands immediately.
    lower, upper = hysteretic_bounds(
        previous_lower, previous_upper,
        np.array([-14.0, 0.0, -18.0]),
        np.array([9.0, 31.0, 48.0]),
        0.125,
    )
    np.testing.assert_allclose(lower, [-14.0, -0.875, -19.75])
    np.testing.assert_allclose(upper, [9.875, 31.0, 49.75])

    # A dense sheet enters, a formerly active thinning sheet remains, and
    # isolated droplets never allocate a local 3D field.
    dense = np.tile(np.array([[1.0, 3.0, 1.0]], dtype=np.float32), (52, 1))
    retained = np.tile(np.array([[13.0, 4.0, 1.0]], dtype=np.float32), (27, 1))
    droplets = np.array([[30.0 + i * 12.0, 20.0, 2.0] for i in range(8)], dtype=np.float32)
    positions = np.concatenate([dense, retained, droplets], axis=0)
    excluded = np.ones(len(positions), dtype=bool)
    keys, counts = select_splash_bricks(
        positions, excluded, 12.0, 48, 24, {(1, 0, 0)}, 6
    )
    assert keys == [(0, 0, 0), (1, 0, 0)], keys
    assert counts[(0, 0, 0)] == 52 and counts[(1, 0, 0)] == 27

    print("PASS: immediate domain expansion, gradual shrink and dense splash-brick hysteresis")


if __name__ == "__main__":
    main()
