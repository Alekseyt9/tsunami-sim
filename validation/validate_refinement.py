"""CUDA regression test for V3 adaptive structural particle splitting."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from simulation.scene import (  # noqa: E402
    STRUCT_BEAM,
    STRUCT_COLUMN,
    STRUCT_CORE,
    STRUCT_GLASS,
    STRUCT_SLAB,
    STRUCT_WALL,
)
from kernels.hybrid import refine_impacted_solids  # noqa: E402


def gpu_array(values: np.ndarray, dtype, device: str):
    return wp.array(values, dtype=dtype, device=device)


def main() -> None:
    wp.init()
    device = "cuda:0"
    capacity = 64
    roles = np.asarray(
        [STRUCT_SLAB, STRUCT_WALL, STRUCT_BEAM, STRUCT_COLUMN, STRUCT_CORE, STRUCT_GLASS],
        dtype=np.int32,
    )
    expected_children = {
        STRUCT_SLAB: 4,
        STRUCT_WALL: 4,
        STRUCT_BEAM: 2,
        STRUCT_COLUMN: 2,
        STRUCT_CORE: 8,
        STRUCT_GLASS: 4,
    }
    parent_count = len(roles)

    positions = np.zeros((capacity, 3), dtype=np.float32)
    velocities = np.zeros((capacity, 3), dtype=np.float32)
    masses = np.zeros(capacity, dtype=np.float32)
    volumes = np.zeros(capacity, dtype=np.float32)
    for index in range(parent_count):
        positions[index] = (index * 10.0, 12.0 + index, 4.0)
        velocities[index] = (0.25 * index, -0.1 * index, 1.0 + 0.05 * index)
        masses[index] = 100.0 + 17.0 * index
        volumes[index] = 0.8 + 0.13 * index

    original_mass = masses[:parent_count].copy()
    original_volume = volumes[:parent_count].copy()
    original_momentum = original_mass[:, None] * velocities[:parent_count]
    original_com = positions[:parent_count].copy()

    def floats(values=None):
        host = np.zeros(capacity, dtype=np.float32) if values is None else values
        return gpu_array(host, float, device)

    def ints(values=None):
        host = np.zeros(capacity, dtype=np.int32) if values is None else values
        return gpu_array(host, wp.int32, device)

    x = gpu_array(positions, wp.vec3, device)
    rest_x = gpu_array(positions.copy(), wp.vec3, device)
    velocity = gpu_array(velocities, wp.vec3, device)
    radius_host = np.zeros(capacity, dtype=np.float32)
    radius_host[:parent_count] = 0.624
    radius = floats(radius_host)
    mass = floats(masses)
    volume = floats(volumes)
    kind_host = np.zeros(capacity, dtype=np.int32); kind_host[:parent_count] = 1
    kind = ints(kind_host)
    material_host = np.zeros(capacity, dtype=np.int32); material_host[:parent_count] = 1
    material = ints(material_host)
    role_host = np.zeros(capacity, dtype=np.int32); role_host[:parent_count] = roles
    structural_class = ints(role_host)
    building_host = np.full(capacity, -1, dtype=np.int32); building_host[:parent_count] = np.arange(parent_count)
    building_id = ints(building_host)
    fixed = ints()
    base_fixed = ints()
    damage = floats()
    rho_reference = floats()
    solid_force = gpu_array(np.zeros((capacity, 3), dtype=np.float32), wp.vec3, device)
    impact_impulse = floats()
    local_impact_active = ints()
    fragment_host = np.full(capacity, -1, dtype=np.int32); fragment_host[:parent_count] = np.arange(parent_count)
    fragment_id = ints(fragment_host)
    axis_host = np.full(capacity, -1, dtype=np.int32); axis_host[:parent_count] = (1, 0, 0, 1, 2, 2)
    normal_axis = ints(axis_host)
    preimpact = gpu_array(np.ones(parent_count, dtype=np.int32), wp.int32, device)
    counters = gpu_array(np.zeros(7, dtype=np.int32), wp.int32, device)
    rigid_state = gpu_array(np.zeros(parent_count, dtype=np.int32), wp.int32, device)
    count = gpu_array(np.asarray([parent_count], dtype=np.int32), wp.int32, device)

    wp.launch(
        refine_impacted_solids,
        dim=parent_count,
        inputs=[
            x, rest_x, velocity, radius, mass, volume, kind, material, structural_class,
            building_id, fixed, base_fixed, damage, rho_reference, solid_force,
            impact_impulse, local_impact_active,
            fragment_id, normal_axis, rigid_state, preimpact, counters, count, parent_count, capacity,
            0.195, 0.0975, 0.39, 0.08, 25.0,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    final_count = int(count.numpy()[0])
    out_role = structural_class.numpy()[:final_count]
    out_mass = mass.numpy()[:final_count]
    out_volume = volume.numpy()[:final_count]
    out_velocity = velocity.numpy()[:final_count]
    out_position = x.numpy()[:final_count]
    out_counters = counters.numpy()

    expected_total = sum(expected_children.values())
    if final_count != expected_total:
        raise AssertionError(f"particle count {final_count}, expected {expected_total}")

    worst_mass = 0.0
    worst_volume = 0.0
    worst_momentum = 0.0
    worst_com = 0.0
    for parent_index, role in enumerate(roles):
        selection = out_role == role
        actual_children = int(np.count_nonzero(selection))
        if actual_children != expected_children[int(role)]:
            raise AssertionError(
                f"role {role}: {actual_children} children, expected {expected_children[int(role)]}"
            )
        if out_counters[int(role)] != 1:
            raise AssertionError(f"role {role}: refinement counter is {out_counters[int(role)]}")
        role_mass = float(out_mass[selection].sum(dtype=np.float64))
        role_volume = float(out_volume[selection].sum(dtype=np.float64))
        role_momentum = (out_mass[selection, None] * out_velocity[selection]).sum(axis=0, dtype=np.float64)
        role_com = (
            out_mass[selection, None] * out_position[selection]
        ).sum(axis=0, dtype=np.float64) / role_mass
        worst_mass = max(worst_mass, abs(role_mass - original_mass[parent_index]))
        worst_volume = max(worst_volume, abs(role_volume - original_volume[parent_index]))
        worst_momentum = max(
            worst_momentum,
            float(np.max(np.abs(role_momentum - original_momentum[parent_index]))),
        )
        worst_com = max(worst_com, float(np.max(np.abs(role_com - original_com[parent_index]))))

    tolerance = 2.0e-5
    if max(worst_mass, worst_volume, worst_momentum, worst_com) > tolerance:
        raise AssertionError("adaptive split failed a conservation check")
    print(f"PASS: {parent_count} parents -> {final_count} typed children on {device}")
    print(
        "max absolute errors: "
        f"mass={worst_mass:.3e}, volume={worst_volume:.3e}, "
        f"momentum={worst_momentum:.3e}, center_of_mass={worst_com:.3e}"
    )


if __name__ == "__main__":
    main()
