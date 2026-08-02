"""CUDA regression for the V3.4 far-field model and SPH overlap coupling."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np
import warp as wp

from shallow_water import ShallowWaterFarField, emit_sph_interface_particles


HERE = Path(__file__).resolve().parent.parent


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

    # Filling an empty SPH site must be a representation transfer, not a
    # source: the 3D particle receives exactly the volume and horizontal
    # momentum removed from the 2D cell.
    capacity = 4
    old_count = 1
    scalar = lambda: wp.zeros(capacity, dtype=float, device=device)
    integer = lambda: wp.zeros(capacity, dtype=wp.int32, device=device)
    vector = lambda: wp.zeros(capacity, dtype=wp.vec3, device=device)
    emit_arrays = {
        "x": vector(), "rest_x": vector(), "v": vector(), "radius": scalar(),
        "mass": scalar(), "volume": scalar(), "kind": integer(), "material": integer(),
        "building_id": integer(), "structural_class": integer(), "fixed": integer(),
        "damage": scalar(), "impact_impulse": scalar(), "local_impact_active": integer(),
        "rho_reference": scalar(), "rho": scalar(),
        "acceleration": vector(), "solid_force": vector(), "base_fixed": integer(),
        "fragment_id": integer(), "normal_axis": integer(), "time_level": integer(),
        "time_active": integer(), "surface_mask": integer(), "surface_normal": vector(),
        "foam_strength": scalar(),
        "fluid_group_id": wp.array(
            np.full(capacity, -1, dtype=np.int32), dtype=wp.int32, device=device
        ),
    }
    emit_arrays["x"] = wp.array(
        np.asarray([[0.0, 0.0, 0.0]] * capacity, dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    emit_arrays["rest_x"] = wp.clone(emit_arrays["x"])
    exchange_volume = wp.zeros((solver.nx, solver.nz), dtype=float, device=device)
    exchange_x = wp.zeros((solver.nx, solver.nz), dtype=float, device=device)
    exchange_z = wp.zeros((solver.nx, solver.nz), dtype=float, device=device)
    counter = wp.array(np.asarray([old_count], dtype=np.int32), dtype=wp.int32, device=device)
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.build(emit_arrays["x"][:old_count], 1.0)
    wp.launch(
        emit_sph_interface_particles,
        dim=(1, 1),
        inputs=[
            grid.id, emit_arrays["x"], emit_arrays["rest_x"], emit_arrays["v"],
            emit_arrays["radius"], emit_arrays["mass"], emit_arrays["volume"],
            emit_arrays["kind"], emit_arrays["material"], emit_arrays["building_id"],
            emit_arrays["structural_class"], emit_arrays["fixed"], emit_arrays["damage"],
            emit_arrays["impact_impulse"], emit_arrays["local_impact_active"],
            emit_arrays["rho_reference"], emit_arrays["rho"], emit_arrays["acceleration"],
            emit_arrays["solid_force"], emit_arrays["base_fixed"], emit_arrays["fragment_id"],
            emit_arrays["normal_axis"], emit_arrays["time_level"], emit_arrays["time_active"],
            emit_arrays["surface_mask"], emit_arrays["surface_normal"],
            emit_arrays["foam_strength"], emit_arrays["fluid_group_id"],
            solver.state, exchange_volume, exchange_x, exchange_z,
            counter, old_count, capacity, 1, 1, solver.lower_x, solver.lower_z,
            solver.interface_z, solver.cell_size, solver.nx, solver.nz, 1.0,
            float(cfg["rest_density"]), 0.25,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    if int(counter.numpy()[0]) != old_count + 1:
        raise AssertionError("empty shallow/SPH interface site emitted no particle")
    new_mass = float(emit_arrays["mass"].numpy()[old_count])
    new_volume = float(emit_arrays["volume"].numpy()[old_count])
    new_velocity = emit_arrays["v"].numpy()[old_count]
    volume_residual = abs(new_volume + float(exchange_volume.numpy().sum()))
    momentum_residual = float(np.linalg.norm(
        new_mass * new_velocity[[0, 2]]
        + np.asarray([exchange_x.numpy().sum(), exchange_z.numpy().sum()], dtype=np.float32)
    ))
    if volume_residual > 1.0e-6:
        raise AssertionError(f"SPH emission volume residual is {volume_residual:.3e} m3")
    if momentum_residual > 1.0e-3:
        raise AssertionError(f"SPH emission momentum residual is {momentum_residual:.3e} kg m/s")
    if int(emit_arrays["fluid_group_id"].numpy()[old_count]) != -1:
        raise AssertionError("fresh shallow/SPH particle inherited an adaptive sibling group")

    # Rejected slow sites must not reserve an uninitialized particle slot.
    rejected_counter = wp.array(
        np.asarray([old_count], dtype=np.int32), dtype=wp.int32, device=device
    )
    wp.launch(
        emit_sph_interface_particles,
        dim=(1, 1),
        inputs=[
            grid.id, emit_arrays["x"], emit_arrays["rest_x"], emit_arrays["v"],
            emit_arrays["radius"], emit_arrays["mass"], emit_arrays["volume"],
            emit_arrays["kind"], emit_arrays["material"], emit_arrays["building_id"],
            emit_arrays["structural_class"], emit_arrays["fixed"], emit_arrays["damage"],
            emit_arrays["impact_impulse"], emit_arrays["local_impact_active"],
            emit_arrays["rho_reference"], emit_arrays["rho"], emit_arrays["acceleration"],
            emit_arrays["solid_force"], emit_arrays["base_fixed"], emit_arrays["fragment_id"],
            emit_arrays["normal_axis"], emit_arrays["time_level"], emit_arrays["time_active"],
            emit_arrays["surface_mask"], emit_arrays["surface_normal"],
            emit_arrays["foam_strength"], emit_arrays["fluid_group_id"],
            solver.state, exchange_volume, exchange_x, exchange_z,
            rejected_counter, old_count, capacity, 1, 1, solver.lower_x, solver.lower_z,
            solver.interface_z, solver.cell_size, solver.nx, solver.nz, 1.0,
            float(cfg["rest_density"]), 100.0,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    if int(rejected_counter.numpy()[0]) != old_count:
        raise AssertionError("rejected shallow/SPH emission left a zero-valued particle slot")

    print(
        f"PASS: shallow field volume drift={volume_error:.3%} after 1 s; "
        f"SPH/2D impulse residual={impulse_residual:.3e} kg m/s; "
        f"emission volume residual={volume_residual:.3e} m3; "
        f"emission momentum residual={momentum_residual:.3e} kg m/s; "
        f"grid={solver.nx}x{solver.nz}"
    )


if __name__ == "__main__":
    main()
