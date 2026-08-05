"""CUDA regression for water-depth history and disocclusion rejection."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.base import temporal_stabilize_water_depth


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    history = wp.empty(4, dtype=float, device=device)
    output = wp.empty(4, dtype=float, device=device)
    foam = wp.array(np.zeros(4, dtype=np.float32), dtype=float, device=device)
    first = wp.array(
        np.asarray([10.0, 20.0, 1.0e9, 12.0], dtype=np.float32),
        dtype=float,
        device=device,
    )
    wp.launch(
        temporal_stabilize_water_depth,
        dim=4,
        inputs=[first, foam, history, output, 0, 0.80, 0.75],
        device=device,
    )
    wp.synchronize_device(device)
    if not np.allclose(output.numpy(), first.numpy()):
        raise AssertionError("first temporal frame was not copied exactly")

    second_values = np.asarray([10.4, 25.0, 1.0e9, 12.5], dtype=np.float32)
    second = wp.array(second_values, dtype=float, device=device)
    energetic_foam = wp.array(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), dtype=float, device=device
    )
    wp.launch(
        temporal_stabilize_water_depth,
        dim=4,
        inputs=[second, energetic_foam, history, output, 1, 0.80, 0.75],
        device=device,
    )
    wp.synchronize_device(device)
    result = output.numpy()
    if not (10.0 < float(result[0]) < 10.4):
        raise AssertionError(f"calm water did not accumulate history: {result[0]}")
    if not np.isclose(result[1], 25.0):
        raise AssertionError(f"disocclusion retained stale depth: {result[1]}")
    if result[2] < 1.0e8:
        raise AssertionError("invalid background depth became water")
    calm_delta = abs(float(result[0]) - 10.0)
    foam_delta = abs(float(result[3]) - 12.0)
    if foam_delta <= calm_delta:
        raise AssertionError("foam did not reduce temporal history weight")
    print(
        "PASS: calm water accumulates history; discontinuities reject it and foam stays responsive"
    )


if __name__ == "__main__":
    main()
