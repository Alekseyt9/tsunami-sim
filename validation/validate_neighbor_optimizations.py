"""GPU checks for hydraulic masks and core/halo Verlet collection."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import (  # noqa: E402
    count_fluid_verlet_neighbors,
    fill_fluid_verlet_neighbors,
    finalize_verlet_rebuild,
    precompute_sph_kernel_coefficients,
    update_hydraulic_boundary_mask,
)


def main() -> None:
    wp.init()
    device = "cuda:0"
    points = np.asarray(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.2, 0.0, 0.0),
         (0.5, 0.0, 0.0), (0.7, 0.0, 0.0)], dtype=np.float32,
    )
    count = len(points)
    x = wp.array(points, dtype=wp.vec3, device=device)
    kind = wp.array(np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
                    dtype=wp.int32, device=device)
    hydraulic = wp.array(np.asarray([0, 0, 0, 1, 0], dtype=np.int32),
                         dtype=wp.int32, device=device)
    grid = wp.HashGrid(16, 16, 16, device=device)
    grid.build(x, 2.0)
    fluid_particle = wp.array(np.asarray([0, 1, 2], dtype=np.int32),
                              dtype=wp.int32, device=device)
    fluid_count = 3
    neighbour_count = wp.zeros(fluid_count, dtype=wp.int32, device=device)
    neighbour_offset = wp.zeros(fluid_count, dtype=wp.int32, device=device)
    wp.launch(
        count_fluid_verlet_neighbors, dim=fluid_count,
        inputs=[grid.id, x, fluid_particle, kind, hydraulic,
                neighbour_count, 2.0, 2.3],
        device=device,
    )
    wp.utils.array_scan(neighbour_count, neighbour_offset, inclusive=False)
    wp.synchronize_device(device)
    count_host = neighbour_count.numpy()
    offset_host = neighbour_offset.numpy()
    total = int(offset_host[-1] + count_host[-1])
    neighbour_index = wp.zeros(max(total, 1), dtype=wp.int32, device=device)
    entries = wp.zeros(1, dtype=wp.int32, device=device)
    overflow = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        finalize_verlet_rebuild, dim=1,
        inputs=[neighbour_count, neighbour_offset, fluid_count, max(total, 1),
                entries, overflow], device=device,
    )
    wp.launch(
        fill_fluid_verlet_neighbors, dim=fluid_count,
        inputs=[grid.id, x, fluid_particle, kind, hydraulic, neighbour_offset,
                neighbour_index, max(total, 1), overflow, 2.0, 2.3],
        device=device,
    )
    wp.synchronize_device(device)
    first = set(neighbour_index.numpy()[offset_host[0]:offset_host[0] + count_host[0]])
    if first != {0, 1, 2, 3}:
        raise AssertionError(f"unexpected core/halo neighbours: {sorted(first)}")
    if int(entries.numpy()[0]) != total or int(overflow.numpy()[0]) != 0:
        raise AssertionError("GPU-side Verlet finalization produced invalid counters")

    radius = wp.array(np.asarray([0.25, 0.50], dtype=np.float32),
                      dtype=float, device=device)
    support = wp.zeros(2, dtype=float, device=device)
    support_squared = wp.zeros(2, dtype=float, device=device)
    poly6_coefficient = wp.zeros(2, dtype=float, device=device)
    spiky_coefficient = wp.zeros(2, dtype=float, device=device)
    viscosity_coefficient = wp.zeros(2, dtype=float, device=device)
    wp.launch(
        precompute_sph_kernel_coefficients, dim=2,
        inputs=[radius, support, support_squared, poly6_coefficient,
                spiky_coefficient, viscosity_coefficient], device=device,
    )
    if not np.allclose(support.numpy(), [1.0, 2.0], rtol=1e-6):
        raise AssertionError("SPH support lookup was not updated from radius")

    base = wp.array(np.asarray([0, 0, 0, 1, 0], dtype=np.int32),
                    dtype=wp.int32, device=device)
    damage = wp.zeros(count, dtype=float, device=device)
    fragment = wp.array(np.asarray([-1, -1, -1, 0, 1], dtype=np.int32),
                        dtype=wp.int32, device=device)
    rigid = wp.zeros(2, dtype=wp.int32, device=device)
    output = wp.zeros(count, dtype=wp.int32, device=device)
    wp.launch(
        update_hydraulic_boundary_mask, dim=count,
        inputs=[kind, base, damage, fragment, rigid, output, 0.18], device=device,
    )
    if tuple(output.numpy()[3:]) != (1, 0):
        raise AssertionError("intact interior particle was exposed")
    wp.copy(damage, wp.array(np.asarray([0, 0, 0, 0, 0.2], dtype=np.float32),
                             dtype=float, device=device))
    wp.launch(
        update_hydraulic_boundary_mask, dim=count,
        inputs=[kind, base, damage, fragment, rigid, output, 0.18], device=device,
    )
    if tuple(output.numpy()[3:]) != (1, 1):
        raise AssertionError("damaged interior particle was not exposed")
    print("PASS: fluid-only Verlet, async counters, SPH coefficient cache, and hydraulic masks")


if __name__ == "__main__":
    main()
