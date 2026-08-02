"""CUDA regression for spray-energy and sustained building-activation gates."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
from pathlib import Path
import warp as wp

HERE = Path(__file__).resolve().parent

from hybrid_kernels import (
    accumulate_material_impact,
    activate_buildings_from_hits,
    apply_building_activity,
    count_loaded_building_particles,
)
from kernels import integrate
from scene import STRUCT_CORE, STRUCT_GLASS, STRUCT_WALL


def validate_velocity_guard(device: str) -> None:
    position = wp.array(
        np.asarray([[0.0, 10.0, 0.0]] * 6, dtype=np.float32), dtype=wp.vec3, device=device
    )
    velocity = wp.array(
        np.asarray(
            [
                [100.0, 0.0, 0.0], [0.0, 80.0, 0.0],
                [0.0, -80.0, 0.0], [19.0, 5.0, 0.0],
                [100.0, 0.0, 0.0], [0.0, 80.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    acceleration = wp.zeros(6, dtype=wp.vec3, device=device)
    kind = wp.array(np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32), dtype=wp.int32, device=device)
    fixed = wp.zeros(6, dtype=wp.int32, device=device)
    wp.launch(
        integrate,
        dim=6,
        inputs=[
            position, velocity, acceleration, kind, fixed, 0.001,
            1000.0, -1000.0, 1000.0, 1000.0, 0.0, 30.0, 18.0, 40.0, 12.0,
        ],
        device=device,
    )
    result = velocity.numpy()
    fluid_speed = np.linalg.norm(result[:4], axis=1)
    solid_speed = np.linalg.norm(result[4:], axis=1)
    if float(fluid_speed.max()) > 30.0001 or float(np.abs(result[:4, 1]).max()) > 18.0001:
        raise AssertionError(f"fluid energy guard failed: velocity={result[:4]}")
    np.testing.assert_allclose(result[3], [19.0, 5.0, 0.0], atol=1.0e-6)
    if float(solid_speed.max()) > 40.0001 or float(result[4:, 1].max()) > 12.0001:
        raise AssertionError(f"solid energy guard failed: velocity={result[4:]}")


def count_hits(device: str, elevation: float, force_z: float) -> int:
    count = 16
    rest = np.zeros((count, 3), dtype=np.float32)
    rest[:, 1] = elevation
    force = np.zeros((count, 3), dtype=np.float32)
    force[:, 2] = force_z
    hits = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        count_loaded_building_particles,
        dim=count,
        inputs=[
            wp.array(rest, dtype=wp.vec3, device=device),
            wp.ones(count, dtype=wp.int32, device=device),
            wp.zeros(count, dtype=wp.int32, device=device),
            wp.array(np.full(count, 100.0, dtype=np.float32), dtype=float, device=device),
            wp.array(force, dtype=wp.vec3, device=device),
            5.0, 8.0, hits,
        ],
        device=device,
    )
    return int(hits.numpy()[0])


def validate_activation_gate(device: str) -> None:
    if count_hits(device, 20.0, 1000.0) != 0:
        raise AssertionError("high spray was counted as a foundation load")
    if count_hits(device, 2.0, -1000.0) != 0:
        raise AssertionError("reverse/overhead load was counted as the incoming front")
    if count_hits(device, 2.0, 1000.0) != 16:
        raise AssertionError("sustained lower-facade front was not counted")

    hits = wp.array(np.asarray([16], dtype=np.int32), dtype=wp.int32, device=device)
    active = wp.zeros(1, dtype=wp.int32, device=device)
    exposure = wp.zeros(1, dtype=float, device=device)
    for _ in range(3):
        wp.launch(
            activate_buildings_from_hits,
            dim=1,
            inputs=[hits, active, exposure, 12, 0.005, 0.02, 4.0],
            device=device,
        )
    if int(active.numpy()[0]) != 0:
        raise AssertionError("a sub-threshold transient activated the building")
    wp.launch(
        activate_buildings_from_hits,
        dim=1,
        inputs=[hits, active, exposure, 12, 0.005, 0.02, 4.0],
        device=device,
    )
    if int(active.numpy()[0]) != 1:
        raise AssertionError("a sustained tsunami-front load did not activate the building")


def validate_material_impact_gate(device: str) -> None:
    roles = wp.array(
        np.asarray([STRUCT_GLASS, STRUCT_WALL, STRUCT_CORE], dtype=np.int32),
        dtype=wp.int32, device=device,
    )
    kind = wp.ones(3, dtype=wp.int32, device=device)
    mass = wp.array(np.full(3, 100.0, dtype=np.float32), dtype=float, device=device)
    force = wp.array(
        np.asarray([[2000.0, 0.0, 0.0]] * 3, dtype=np.float32),
        dtype=wp.vec3, device=device,
    )
    impulse = wp.zeros(3, dtype=float, device=device)
    local = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        accumulate_material_impact, dim=3,
        inputs=[kind, roles, mass, force, impulse, local, 0.01], device=device,
    )
    if local.numpy().tolist() != [1, 0, 0]:
        raise AssertionError(f"isolated impulse did not release only glass: {local.numpy()}")
    fixed = wp.ones(3, dtype=wp.int32, device=device)
    wp.launch(
        apply_building_activity, dim=3,
        inputs=[kind, wp.zeros(3, dtype=wp.int32, device=device), roles,
                wp.zeros(3, dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device), local, fixed], device=device,
    )
    if fixed.numpy().tolist() != [0, 1, 1]:
        raise AssertionError(f"local glass release woke concrete or the whole building: {fixed.numpy()}")


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    validate_velocity_guard(device)
    validate_activation_gate(device)
    validate_material_impact_gate(device)
    print(
        "PASS: fluid <=30 m/s / vertical <=18 m/s; solids <=40 m/s / upward <=12 m/s; "
        "only sustained lower-facade +Z load activates; isolated impact releases only glass"
    )


if __name__ == "__main__":
    main()
