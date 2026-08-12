"""GPU regression: ballistic SPH samples render as mist, not giant drops."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.surface import raster_anisotropic_water_depth  # noqa: E402


def render_sample(device: str, phase_value: int, mist_only: bool) -> tuple[np.ndarray, np.ndarray]:
    width = height = 64
    pixel_count = width * height
    position = wp.array(np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device)
    velocity = wp.array(np.asarray(((1.0, 2.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device)
    radius = wp.array(np.asarray((0.5,), dtype=np.float32), dtype=float, device=device)
    kind = wp.zeros(1, dtype=wp.int32, device=device)
    surface = wp.ones(1, dtype=wp.int32, device=device)
    normal = wp.array(np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device)
    foam_particle = wp.zeros(1, dtype=float, device=device)
    phase = wp.array(np.asarray((phase_value,), dtype=np.int32), dtype=wp.int32, device=device)
    depth = wp.full(pixel_count, 1.0e9, dtype=float, device=device)
    back_depth = wp.zeros(pixel_count, dtype=float, device=device)
    foam = wp.zeros(pixel_count, dtype=float, device=device)
    wp.launch(
        raster_anisotropic_water_depth, dim=1,
        inputs=[
            position, velocity, radius, kind, surface, normal, foam_particle, phase,
            depth, back_depth, foam, wp.vec3(0.0, 0.0, -5.0),
            wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0), 50.0, width, height, 2.8, 2.45,
            int(mist_only),
        ], device=device,
    )
    wp.synchronize_device(device)
    return depth.numpy(), foam.numpy()


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    connected_depth, _ = render_sample(device, phase_value=0, mist_only=True)
    droplet_depth, droplet_foam = render_sample(device, phase_value=2, mist_only=True)
    legacy_depth, _ = render_sample(device, phase_value=2, mist_only=False)
    if not np.any(connected_depth < 1.0e8):
        raise AssertionError("connected water stopped writing refractive depth")
    if np.any(droplet_depth < 1.0e8):
        raise AssertionError("ballistic sample still produced giant refractive droplet geometry")
    if not np.any(droplet_foam >= 0.48):
        raise AssertionError("hidden ballistic sample did not feed the foam/mist field")
    if not np.any(legacy_depth < 1.0e8):
        raise AssertionError("diagnostic legacy droplet switch no longer restores depth")
    print(
        "PASS: connected water keeps depth; ballistic sample writes foam/mist only; "
        "legacy diagnostic geometry remains switchable"
    )


if __name__ == "__main__":
    main()
