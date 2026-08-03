"""CUDA regression for spray-energy and sustained building-activation gates."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
from pathlib import Path
import warp as wp

HERE = Path(__file__).resolve().parent

from kernels.hybrid import (
    accumulate_material_impact,
    activate_buildings_from_load,
    apply_building_activity,
    accumulate_loaded_building_volume,
    material_impact_damage_drive,
    preloaded_structure_gravity_fraction,
)
from kernels.base import integrate
from simulation.scene import STRUCT_CORE, STRUCT_GLASS, STRUCT_WALL


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


@wp.kernel
def evaluate_impact_drive(
    role: wp.array(dtype=wp.int32),
    impulse: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    i = wp.tid()
    result[i] = material_impact_damage_drive(role[i], impulse[i])


@wp.kernel
def evaluate_dynamic_gravity(
    building: wp.array(dtype=wp.int32),
    support_loss: wp.array(dtype=float),
    rigid: wp.array(dtype=wp.int32),
    result: wp.array(dtype=float),
):
    i = wp.tid()
    result[i] = preloaded_structure_gravity_fraction(
        building[i], support_loss[i], rigid[i] != 0
    )


def loaded_volume(device: str, elevation: float, force_z: float, count: int = 16,
                  particle_volume: float = 1.0) -> float:
    rest = np.zeros((count, 3), dtype=np.float32)
    rest[:, 1] = elevation
    force = np.zeros((count, 3), dtype=np.float32)
    force[:, 2] = force_z
    result = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        accumulate_loaded_building_volume,
        dim=count,
        inputs=[
            wp.array(rest, dtype=wp.vec3, device=device),
            wp.ones(count, dtype=wp.int32, device=device),
            wp.zeros(count, dtype=wp.int32, device=device),
            wp.array(np.full(count, 100.0, dtype=np.float32), dtype=float, device=device),
            wp.array(np.full(count, particle_volume, dtype=np.float32), dtype=float, device=device),
            wp.array(force, dtype=wp.vec3, device=device),
            8.0, 8.0, result,
        ],
        device=device,
    )
    return float(result.numpy()[0])


def validate_activation_gate(device: str) -> None:
    if loaded_volume(device, 20.0, 1000.0) != 0.0:
        raise AssertionError("high spray was counted as a foundation load")
    if loaded_volume(device, 2.0, -1000.0) != 0.0:
        raise AssertionError("reverse/overhead load was counted as the incoming front")
    if loaded_volume(device, 2.0, 1000.0) != 16.0:
        raise AssertionError("sustained lower-facade front was not counted")
    coarse = loaded_volume(device, 2.0, 1000.0, count=16, particle_volume=0.5)
    refined = loaded_volume(device, 2.0, 1000.0, count=32, particle_volume=0.25)
    if abs(coarse - refined) > 1.0e-6:
        raise AssertionError("adaptive refinement changed the physical activation load")

    load = wp.array(np.asarray([0.8], dtype=np.float32), dtype=float, device=device)
    eligible = wp.array(np.asarray([10.0], dtype=np.float32), dtype=float, device=device)
    active = wp.zeros(1, dtype=wp.int32, device=device)
    exposure = wp.zeros(1, dtype=float, device=device)
    for _ in range(4):
        wp.launch(
            activate_buildings_from_load,
            dim=1,
            inputs=[load, eligible, active, exposure, 0.08, 0.05, 0.25, 2.0],
            device=device,
        )
    if int(active.numpy()[0]) != 0:
        raise AssertionError("a sub-threshold transient activated the building")
    wp.launch(
        activate_buildings_from_load,
        dim=1,
        inputs=[load, eligible, active, exposure, 0.08, 0.05, 0.25, 2.0],
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
    debris_force = wp.zeros(3, dtype=wp.vec3, device=device)
    impulse = wp.zeros(3, dtype=float, device=device)
    local = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        accumulate_material_impact, dim=3,
        inputs=[kind, roles, mass, force, debris_force, impulse, local, 0.01], device=device,
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

    drive_role = wp.array(
        np.asarray([STRUCT_WALL, STRUCT_WALL, STRUCT_WALL], dtype=np.int32),
        dtype=wp.int32, device=device,
    )
    drive_impulse = wp.array(
        np.asarray([0.20, 0.22, 0.60], dtype=np.float32), dtype=float, device=device,
    )
    drive = wp.zeros(3, dtype=float, device=device)
    wp.launch(
        evaluate_impact_drive, dim=3,
        inputs=[drive_role, drive_impulse, drive], device=device,
    )
    actual = drive.numpy()
    np.testing.assert_allclose(actual, [0.0, 0.05, 1.0], atol=1.0e-5)

    gravity = wp.zeros(5, dtype=float, device=device)
    wp.launch(
        evaluate_dynamic_gravity, dim=5,
        inputs=[
            wp.array(np.asarray([0, 0, 0, -1, 0], dtype=np.int32), dtype=wp.int32, device=device),
            wp.array(np.asarray([0.0, 0.4, 1.0, 0.0, 0.0], dtype=np.float32), dtype=float, device=device),
            wp.array(np.asarray([0, 0, 0, 0, 1], dtype=np.int32), dtype=wp.int32, device=device),
            gravity,
        ], device=device,
    )
    np.testing.assert_allclose(gravity.numpy(), [0.0, 0.4, 1.0, 1.0, 1.0], atol=1.0e-6)


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    validate_velocity_guard(device)
    validate_activation_gate(device)
    validate_material_impact_gate(device)
    print(
        "PASS: fluid <=30 m/s / vertical <=18 m/s; solids <=40 m/s / upward <=12 m/s; "
        "only sustained volume-normalized lower-facade +Z load activates; "
        "impact fracture drive is continuous; waking preserves static preload; "
        "isolated impact releases only glass"
    )


if __name__ == "__main__":
    main()
