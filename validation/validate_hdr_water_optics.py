"""CUDA regression for HDR sky, filmic resolve and depth-aware water optics."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.base import filmic_tonemap_color, render_physical_sky, shade_water_surface


def vec3_array(values, device: str) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.vec3, device=device)


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # The HDR sky must retain a sun value above display white; otherwise bloom
    # and filmic highlight roll-off receive no real dynamic range.
    sky = wp.empty(15, dtype=wp.vec3, device=device)
    wp.launch(
        render_physical_sky,
        dim=15,
        inputs=[
            sky,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
            5.0,
            5,
            3,
            wp.vec3(0.0, 0.0, 1.0),
            3.2,
            1.0,
            3.1,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    sky_host = sky.numpy()
    if float(np.max(sky_host)) <= 1.0:
        raise AssertionError("physical sky lost its HDR sun range")
    if float(np.max(sky_host[7])) <= float(np.max(sky_host[0])):
        raise AssertionError("sun-facing sky pixel is not brighter than the off-axis sky")

    hdr = vec3_array(
        [(0.18, 0.18, 0.18), (1.0, 1.0, 1.0), (4.0, 4.0, 4.0), (16.0, 16.0, 16.0)],
        device,
    )
    display = wp.empty(4, dtype=wp.vec3, device=device)
    wp.launch(
        filmic_tonemap_color,
        dim=4,
        inputs=[hdr, display, 4, 1, 0.0, 1.15, 0.0],
        device=device,
    )
    wp.synchronize_device(device)
    filmic = display.numpy()[:, 0]
    if not np.all(np.diff(filmic) > 0.0):
        raise AssertionError(f"filmic curve is not strictly monotonic: {filmic}")
    if not np.all((filmic >= 0.0) & (filmic <= 1.0)):
        raise AssertionError(f"filmic output escaped display range: {filmic}")
    if float(filmic[2]) >= 0.9999 or float(filmic[3] - filmic[2]) <= 0.005:
        raise AssertionError(f"HDR highlights were clipped instead of rolled off: {filmic}")

    # Equal white backgrounds viewed through increasingly thick water must
    # absorb red faster than blue. Disable IBL/sun here to isolate transport.
    width, height = 3, 1
    front = wp.array(np.full(3, 5.0, dtype=np.float32), dtype=float, device=device)
    back = wp.array(np.asarray([5.5, 9.0, 15.0], dtype=np.float32), dtype=float, device=device)
    foam = wp.array(np.zeros(3, dtype=np.float32), dtype=float, device=device)
    scene_depth = wp.array(np.full(3, 20.0, dtype=np.float32), dtype=float, device=device)
    scene_color = vec3_array([(1.0, 1.0, 1.0)] * 3, device)
    wp.launch(
        shade_water_surface,
        dim=3,
        inputs=[
            front,
            back,
            foam,
            scene_depth,
            scene_color,
            width,
            height,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
            4.0,
            1.0,
            0.0,
            wp.vec3(0.17, 0.045, 0.018),
            wp.vec3(0.012, 0.032, 0.055),
            0.35,
            18.0,
            wp.vec3(0.0, 1.0, 0.0),
            3.2,
            0.0,
            0.0,
            0.0,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    water = scene_color.numpy()
    red_blue_ratio = water[:, 0] / np.maximum(water[:, 2], 1.0e-6)
    if not np.all(np.diff(red_blue_ratio) < 0.0):
        raise AssertionError(
            f"optical thickness did not preferentially absorb red: {water}"
        )
    if not np.all(np.isfinite(water)):
        raise AssertionError("water transport produced non-finite radiance")

    print(
        "PASS: HDR sky retains the sun, ACES rolls highlights, and front/back "
        "water thickness drives wavelength-dependent transport"
    )


if __name__ == "__main__":
    main()
