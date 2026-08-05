"""CUDA regression for motion-reprojected colour TAA."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.base import temporal_antialias_color


def vec3_array(values: list[tuple[float, float, float]], device: str) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.vec3, device=device)


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    width, height = 4, 1
    first_values = [(0.05, 0.05, 0.05), (1.0, 0.2, 0.1), (0.05, 0.05, 0.05), (0.1, 0.1, 0.1)]
    first = vec3_array(first_values, device)
    scene_depth = wp.array(np.asarray([10.0, 10.0, 10.0, 20.0], dtype=np.float32), dtype=float, device=device)
    water_depth = wp.array(np.full(4, 1.0e9, dtype=np.float32), dtype=float, device=device)
    foam = wp.array(np.zeros(4, dtype=np.float32), dtype=float, device=device)
    motion = wp.array(np.zeros((4, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    history_color = wp.empty(4, dtype=wp.vec3, device=device)
    history_depth = wp.empty(4, dtype=float, device=device)
    output = wp.empty(4, dtype=wp.vec3, device=device)
    wp.launch(
        temporal_antialias_color, dim=4,
        inputs=[first, scene_depth, water_depth, foam, motion, history_color, history_depth,
                output, width, height, 0, 0.86], device=device,
    )
    wp.synchronize_device(device)
    if not np.allclose(output.numpy(), np.asarray(first_values, dtype=np.float32)):
        raise AssertionError("first TAA frame was not copied exactly")

    second_values = [(0.05, 0.05, 0.05), (0.05, 0.05, 0.05), (1.0, 0.2, 0.1), (0.9, 0.9, 0.9)]
    second = vec3_array(second_values, device)
    moving = np.zeros((4, 2), dtype=np.float32)
    moving[2, 0] = 1.0
    motion = wp.array(moving, dtype=wp.vec2, device=device)
    second_depth_values = np.asarray([10.0, 10.0, 10.0, 38.0], dtype=np.float32)
    second_depth = wp.array(second_depth_values, dtype=float, device=device)
    wp.launch(
        temporal_antialias_color, dim=4,
        inputs=[second, second_depth, water_depth, foam, motion, history_color, history_depth,
                output, width, height, 1, 0.86], device=device,
    )
    wp.synchronize_device(device)
    result = output.numpy()
    # The red object moved from x=1 to x=2. Reprojection should preserve it,
    # while x=3 changed depth and must reject its old dark history.
    if float(result[2, 0]) < 0.90:
        raise AssertionError(f"motion reprojection lost the moving object: {result[2]}")
    if not np.allclose(result[3], np.asarray(second_values[3]), atol=1.0e-5):
        raise AssertionError(f"depth disocclusion retained stale colour: {result[3]}")
    print("PASS: TAA reprojects motion, clips history and rejects depth disocclusions")


if __name__ == "__main__":
    main()
