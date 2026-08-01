"""CUDA regression for the V3.4 far-field model and SPH overlap coupling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

from shallow_water import ShallowWaterFarField


HERE = Path(__file__).resolve().parent


def main():
    wp.init()
    device = "cuda:0"
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    solver = ShallowWaterFarField(cfg, device)
    before = solver.diagnostics()
    for _ in range(125):
        solver.advance(0.008, float(cfg["rest_density"]))
    wp.synchronize_device(device)
    after = solver.diagnostics()
    volume_error = abs(after["shallow_water_volume_m3"] / before["shallow_water_volume_m3"] - 1.0)
    if volume_error >= 0.02:
        raise AssertionError(f"shallow-water volume drift is {volume_error:.3%}")

    # Verify the overlap transfers an exactly opposite horizontal impulse to
    # the 2D field instead of injecting momentum into both representations.
    position = np.asarray([[0.0, 3.0, solver.interface_z + 0.5]], dtype=np.float32)
    velocity = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.asarray([1250.0], dtype=np.float32)
    arrays = {
        "x": wp.array(position, dtype=wp.vec3, device=device),
        "v": wp.array(velocity, dtype=wp.vec3, device=device),
        "mass": wp.array(mass, dtype=float, device=device),
        "kind": wp.zeros(1, dtype=wp.int32, device=device),
        "acceleration": wp.zeros(1, dtype=wp.vec3, device=device),
    }
    dt = 0.001
    solver.couple(arrays, 1, dt)
    wp.synchronize_device(device)
    particle_impulse = mass[0] * arrays["acceleration"].numpy()[0, (0, 2)] * dt
    field_impulse = np.asarray(
        [solver.exchange_x.numpy().sum(), solver.exchange_z.numpy().sum()], dtype=np.float32
    )
    impulse_residual = float(np.linalg.norm(particle_impulse + field_impulse))
    if impulse_residual > 1.0e-5:
        raise AssertionError(f"SPH/2D exchange residual is {impulse_residual:.3e} kg m/s")
    if float(np.linalg.norm(particle_impulse)) <= 1.0e-6:
        raise AssertionError("interface coupling applied no test impulse")

    print(
        f"PASS: shallow field volume drift={volume_error:.3%} after 1 s; "
        f"SPH/2D impulse residual={impulse_residual:.3e} kg m/s; "
        f"grid={solver.nx}x{solver.nz}"
    )


if __name__ == "__main__":
    main()
