"""CUDA regression for surface/turbulence selective SPH refinement."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels import refine_entering_fluid


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    capacity = 24
    old_count = 3
    position_host = np.zeros((capacity, 3), dtype=np.float32)
    position_host[:old_count] = ((0.0, 5.0, 2.0), (1.0, 12.0, 2.0), (2.0, 5.0, 2.0))
    velocity_host = np.zeros((capacity, 3), dtype=np.float32)
    velocity_host[2, 1] = 3.0
    radius_host = np.zeros(capacity, dtype=np.float32); radius_host[:old_count] = 0.5
    mass_host = np.zeros(capacity, dtype=np.float32); mass_host[:old_count] = 1000.0
    volume_host = np.zeros(capacity, dtype=np.float32); volume_host[:old_count] = 1.0
    x = wp.array(position_host, dtype=wp.vec3, device=device)
    rest = wp.array(position_host, dtype=wp.vec3, device=device)
    velocity = wp.array(velocity_host, dtype=wp.vec3, device=device)
    radius = wp.array(radius_host, dtype=float, device=device)
    mass = wp.array(mass_host, dtype=float, device=device)
    volume = wp.array(volume_host, dtype=float, device=device)
    integer = lambda: wp.zeros(capacity, dtype=wp.int32, device=device)
    scalar = lambda: wp.zeros(capacity, dtype=float, device=device)
    count = wp.array(np.asarray([old_count], dtype=np.int32), dtype=wp.int32, device=device)
    wp.launch(
        refine_entering_fluid, dim=old_count,
        inputs=[x, rest, velocity, radius, mass, volume, integer(), integer(), integer(),
                integer(), scalar(), scalar(), count, old_count, capacity, 0.25, 1.0,
                1, 11.5, 2.5], device=device,
    )
    wp.synchronize_device(device)
    final_count = int(count.numpy()[0])
    if final_count != 17:
        raise AssertionError(f"expected surface+turbulent split only (17 particles), got {final_count}")
    if abs(float(radius.numpy()[0]) - 0.5) > 1.0e-6:
        raise AssertionError("calm interior particle was incorrectly refined")
    if abs(float(mass.numpy()[:final_count].sum()) - 3000.0) > 1.0e-3:
        raise AssertionError("adaptive surface refinement did not conserve mass")
    print("PASS: calm interior remains coarse; surface and turbulent particles refine 1->8 with conserved mass")


if __name__ == "__main__":
    main()
