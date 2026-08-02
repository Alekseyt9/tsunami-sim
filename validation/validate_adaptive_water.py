"""CUDA regression for surface/turbulence selective SPH refinement."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels import refine_entering_fluid
from hybrid_kernels import apply_conservative_fluid_merges
from hybrid_model import select_conservative_fluid_merges


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
    group_id = wp.array(np.full(capacity, -1, dtype=np.int32), dtype=wp.int32, device=device)
    group_counter = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        refine_entering_fluid, dim=old_count,
        inputs=[x, rest, velocity, radius, mass, volume, integer(), integer(), integer(),
                integer(), scalar(), scalar(), count, group_id, group_counter,
                old_count, capacity, 0.25, 1.0,
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
    groups = group_id.numpy()[:final_count]
    valid_groups, group_sizes = np.unique(groups[groups >= 0], return_counts=True)
    if len(valid_groups) != 2 or not np.all(group_sizes == 8):
        raise AssertionError(f"refined particles did not retain two complete sibling octets: {group_sizes}")

    # Only a complete, calm octet below the free-surface band may merge back.
    sibling_position = np.asarray([
        (sx, 5.0 + sy, sz)
        for sx in (-0.25, 0.25) for sy in (-0.25, 0.25) for sz in (-0.25, 0.25)
    ], dtype=np.float32)
    sibling_velocity = np.tile(np.asarray((1.25, 0.1, 3.5), dtype=np.float32), (8, 1))
    sibling_mass = np.full(8, 125.0, dtype=np.float32)
    sibling_volume = np.full(8, 0.125, dtype=np.float32)
    merge = select_conservative_fluid_merges(
        np.full(8, 17, dtype=np.int32), np.zeros(8, dtype=np.int32),
        sibling_position, sibling_velocity, sibling_mass, sibling_volume,
        np.full(8, 0.25, dtype=np.float32), maximum_y=10.0,
        maximum_vertical_speed=0.8, maximum_velocity_rms=0.6,
        maximum_span=0.9, maximum_fine_radius=0.3125,
    )
    if len(merge["representatives"]) != 1 or len(merge["removed"]) != 7:
        raise AssertionError("complete calm sibling octet did not merge")
    if abs(float(merge["mass"][0]) - 1000.0) > 1.0e-5:
        raise AssertionError("reverse adaptation did not conserve mass")
    if not np.allclose(merge["velocity"][0], sibling_velocity[0], atol=1.0e-6):
        raise AssertionError("reverse adaptation did not conserve momentum")
    merge_x = wp.array(sibling_position, dtype=wp.vec3, device=device)
    merge_rest = wp.array(sibling_position, dtype=wp.vec3, device=device)
    merge_velocity = wp.array(sibling_velocity, dtype=wp.vec3, device=device)
    merge_mass_gpu = wp.array(sibling_mass, dtype=float, device=device)
    merge_volume_gpu = wp.array(sibling_volume, dtype=float, device=device)
    merge_radius_gpu = wp.array(np.full(8, 0.25, dtype=np.float32), dtype=float, device=device)
    merge_group_gpu = wp.array(np.full(8, 17, dtype=np.int32), dtype=wp.int32, device=device)
    merge_scalar = lambda: wp.zeros(8, dtype=float, device=device)
    merge_integer = lambda: wp.zeros(8, dtype=wp.int32, device=device)
    merge_vector = lambda: wp.zeros(8, dtype=wp.vec3, device=device)
    wp.launch(
        apply_conservative_fluid_merges, dim=1,
        inputs=[
            wp.array(merge["representatives"], dtype=wp.int32, device=device),
            wp.array(merge["position"], dtype=wp.vec3, device=device),
            wp.array(merge["velocity"], dtype=wp.vec3, device=device),
            wp.array(merge["mass"], dtype=float, device=device),
            wp.array(merge["volume"], dtype=float, device=device),
            wp.array(merge["radius"], dtype=float, device=device),
            merge_x, merge_rest, merge_velocity, merge_mass_gpu, merge_volume_gpu,
            merge_radius_gpu, merge_scalar(), merge_scalar(), merge_vector(), merge_vector(),
            merge_group_gpu, merge_integer(), merge_vector(), merge_scalar(),
        ], device=device,
    )
    wp.synchronize_device(device)
    representative = int(merge["representatives"][0])
    if abs(float(merge_mass_gpu.numpy()[representative]) - 1000.0) > 1.0e-5:
        raise AssertionError("GPU merge state did not receive conserved mass")
    if int(merge_group_gpu.numpy()[representative]) != -1:
        raise AssertionError("GPU merge representative retained a stale sibling ID")
    sibling_velocity[0, 1] = 2.0
    rejected = select_conservative_fluid_merges(
        np.full(8, 17, dtype=np.int32), np.zeros(8, dtype=np.int32),
        sibling_position, sibling_velocity, sibling_mass, sibling_volume,
        np.full(8, 0.25, dtype=np.float32), maximum_y=10.0,
        maximum_vertical_speed=0.8, maximum_velocity_rms=0.6,
        maximum_span=0.9, maximum_fine_radius=0.3125,
    )
    if len(rejected["representatives"]) != 0:
        raise AssertionError("turbulent sibling group was incorrectly merged")
    print(
        "PASS: surface/turbulent refinement preserves sibling octets; calm complete octets "
        "merge conservatively while turbulence blocks coarsening"
    )


if __name__ == "__main__":
    main()
