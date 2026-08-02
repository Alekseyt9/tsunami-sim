"""CUDA regression for the compact scalar field and marching-cubes mesh."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from surface_kernels import smooth_sparse_field_axis, splat_sparse_surface_field  # noqa: E402


def component_count(vertex_count: int, faces: np.ndarray) -> int:
    parent = np.arange(vertex_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(int(a), int(b)); union(int(b), int(c)); union(int(c), int(a))
    used = np.unique(faces)
    return len({find(int(index)) for index in used})


def main() -> None:
    wp.init()
    device = "cuda:0"
    spacing = 1.0
    points = np.asarray(
        [[x, y, z] for x in np.arange(-4.0, 4.1, spacing)
                   for y in np.arange(1.0, 5.1, spacing)
                   for z in np.arange(-4.0, 4.1, spacing)],
        dtype=np.float32,
    )
    count = len(points)
    lower = np.asarray([-6.0, -1.0, -6.0], dtype=np.float32)
    upper = np.asarray([6.0, 7.0, 6.0], dtype=np.float32)
    voxel = 0.5
    shape = tuple((np.rint((upper - lower) / voxel).astype(np.int32) + 1).tolist())
    nx, ny, nz = shape
    x = wp.array(points, dtype=wp.vec3, device=device)
    radius = wp.array(np.full(count, 0.5, dtype=np.float32), dtype=float, device=device)
    kind = wp.zeros(count, dtype=wp.int32, device=device)
    mask = wp.ones(count, dtype=wp.int32, device=device)
    field = wp.zeros(shape, dtype=float, device=device)
    temporary = wp.zeros(shape, dtype=float, device=device)
    wp.launch(
        splat_sparse_surface_field,
        dim=count,
        inputs=[x, radius, kind, mask, field, wp.vec3(*lower), voxel, nx, ny, nz],
        device=device,
    )
    source, target = field, temporary
    for axis in range(3):
        wp.launch(
            smooth_sparse_field_axis,
            dim=shape,
            inputs=[source, target, nx, ny, nz, axis],
            device=device,
        )
        source, target = target, source
    vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
        source,
        threshold=5.0,
        domain_bounds_lower_corner=tuple(float(v) for v in lower),
        domain_bounds_upper_corner=tuple(float(v) for v in upper),
    )
    wp.synchronize_device(device)
    vertex = vertices.numpy()
    faces = indices.numpy().reshape(-1, 3)
    if len(vertex) == 0 or len(faces) == 0 or not np.isfinite(vertex).all():
        raise AssertionError("marching cubes returned an empty or invalid mesh")
    if faces.min() < 0 or faces.max() >= len(vertex):
        raise AssertionError("mesh contains an invalid index")
    components = component_count(len(vertex), faces)
    if components != 1:
        raise AssertionError(f"expected one connected water body, got {components}")
    triangle_area2 = np.linalg.norm(
        np.cross(vertex[faces[:, 1]] - vertex[faces[:, 0]], vertex[faces[:, 2]] - vertex[faces[:, 0]]), axis=1
    )
    if float(np.min(triangle_area2)) <= 1.0e-8:
        raise AssertionError("mesh contains a degenerate triangle")
    min_y, max_y = float(vertex[:, 1].min()), float(vertex[:, 1].max())
    # Regression for the visually "hollow wave" failure: the reconstructed
    # volume must include both its bottom closure and a free surface above the
    # highest row of particle centres, rather than only a side/inner shell.
    if min_y >= 1.0 or max_y <= 5.0:
        raise AssertionError(
            f"water volume is missing a closure layer: y bounds are [{min_y:.3f}, {max_y:.3f}]"
        )
    top_vertex_count = int(np.count_nonzero(vertex[:, 1] > 5.0))
    if top_vertex_count < 16:
        raise AssertionError(f"free-surface cap is undersampled: only {top_vertex_count} top vertices")
    print(
        f"PASS: connected water mesh has {len(vertex):,} vertices / {len(faces):,} triangles; "
        f"field={shape} ({int(np.prod(shape)):,} nodes); y=[{min_y:.2f}, {max_y:.2f}]"
    )


if __name__ == "__main__":
    main()
