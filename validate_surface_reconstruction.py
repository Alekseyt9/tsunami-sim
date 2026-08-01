"""CUDA regression for sparse free-surface classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp


from surface_kernels import classify_water_surface  # noqa: E402


def main() -> None:
    wp.init()
    device = "cuda:0"
    points = np.asarray(
        [[x, y, z] for x in (-1.0, 0.0, 1.0) for y in (1.0, 2.0, 3.0) for z in (-1.0, 0.0, 1.0)],
        dtype=np.float32,
    )
    count = len(points)
    x = wp.array(points, dtype=wp.vec3, device=device)
    v = wp.zeros(count, dtype=wp.vec3, device=device)
    radius = wp.array(np.full(count, 0.5, dtype=np.float32), dtype=float, device=device)
    kind = wp.zeros(count, dtype=wp.int32, device=device)
    mask = wp.zeros(count, dtype=wp.int32, device=device)
    normal = wp.zeros(count, dtype=wp.vec3, device=device)
    foam = wp.zeros(count, dtype=float, device=device)
    grid = wp.HashGrid(16, 16, 16, device=device)
    grid.build(x, 2.6)
    wp.launch(
        classify_water_surface,
        dim=count,
        inputs=[grid.id, x, v, radius, kind, mask, normal, foam, 2.6, 18],
        device=device,
    )
    wp.synchronize_device(device)
    mask_host = mask.numpy()
    normal_host = normal.numpy()
    center_index = int(np.flatnonzero(np.all(points == (0.0, 2.0, 0.0), axis=1))[0])
    top_index = int(np.flatnonzero(np.all(points == (0.0, 3.0, 0.0), axis=1))[0])
    if mask_host[center_index] != 0:
        raise AssertionError("fully surrounded center particle was classified as surface")
    if mask_host[top_index] != 1 or normal_host[top_index, 1] < 0.8:
        raise AssertionError("top free-surface particle or its normal is incorrect")
    if float(np.max(foam.numpy())) > 1.0e-6:
        raise AssertionError("calm water generated foam")
    print(
        f"PASS: sparse surface selected {int(mask_host.sum())}/{count} samples; "
        f"top normal={normal_host[top_index]} and calm foam=0"
    )


if __name__ == "__main__":
    main()
