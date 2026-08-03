"""Microbenchmark baseline and V3.2 kernels without rendering or FFmpeg."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path
import time

import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver
from kernels.base import clear_vec3, compute_density, compute_fluid_forces
from kernels.hybrid import (
    compute_density_multirate,
    compute_fluid_forces_multirate,
    consume_deferred_fluid_impulse,
    select_active_time_level,
)


def measure(enabled: bool, output: Path, substeps: int = 8) -> float:
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    cfg["v3"]["multirate"]["enabled"] = enabled
    solver = HybridDelugeSolver(cfg, output)
    # Warm up lazy kernels before measuring.
    solver.substep(float(cfg["dt"]))
    wp.synchronize_device(solver.device)
    started = time.perf_counter()
    for _ in range(substeps):
        solver.substep(float(cfg["dt"]))
    wp.synchronize_device(solver.device)
    elapsed = time.perf_counter() - started
    return elapsed / substeps


def main() -> None:
    wp.init()
    output_root = HERE / "outputs" / "multirate_microbenchmark_20260801"
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = measure(False, output_root / "baseline")
    multirate = measure(True, output_root / "multirate")
    print(f"baseline={baseline * 1000.0:.3f} ms/substep")
    print(f"multirate={multirate * 1000.0:.3f} ms/substep")
    print(f"speedup={baseline / multirate:.3f}x")

    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    solver = HybridDelugeSolver(cfg, output_root / "profile")
    a = solver.arrays
    view = a["x"][:solver.count]
    solver.grid.build(view, solver.max_support)

    def timed(name, kernel, dim, inputs):
        started = time.perf_counter()
        wp.launch(kernel, dim=dim, inputs=inputs, device=solver.device)
        wp.synchronize_device(solver.device)
        print(f"{name}={(time.perf_counter() - started) * 1000.0:.3f} ms")

    timed("select", select_active_time_level, solver.count,
          [solver.time_level[:solver.count], a["kind"][:solver.count], 3, solver.time_active[:solver.count]])
    common_density = [solver.grid.id, view, a["radius"][:solver.count], a["mass"][:solver.count],
                      a["volume"][:solver.count], a["kind"][:solver.count]]
    density_tail = [a["rho"][:solver.count], a["rho_reference"][:solver.count], float(cfg["rest_density"]),
                    float(cfg["sound_speed"]), float(cfg["water_depth"]), float(cfg["wave_height"]),
                    float(cfg["reservoir_z_max"]), solver.max_support]
    timed("density_baseline", compute_density, solver.count, common_density + density_tail)
    timed("density_multirate", compute_density_multirate, solver.count,
          common_density[:3] + [solver.sph_kernel_support_squared[:solver.count],
                                solver.sph_poly6_coefficient[:solver.count]] +
          common_density[3:] + [solver.hydraulic_boundary[:solver.count],
                            a["water_phase"][:solver.count],
                            solver.time_active[:solver.count]] + density_tail)
    wp.launch(clear_vec3, dim=solver.count, inputs=[a["solid_force"][:solver.count]], device=solver.device)
    force_common = [solver.grid.id, view, a["v"][:solver.count], a["radius"][:solver.count],
                    solver.sph_kernel_support[:solver.count],
                    solver.sph_kernel_support_squared[:solver.count],
                    solver.sph_poly6_coefficient[:solver.count],
                    solver.sph_spiky_coefficient[:solver.count],
                    solver.sph_viscosity_coefficient[:solver.count],
                    a["mass"][:solver.count], a["volume"][:solver.count], a["kind"][:solver.count],
                    a["rho"][:solver.count]]
    force_tail = [a["acceleration"][:solver.count], a["solid_force"][:solver.count],
                  float(cfg["rest_density"]), float(cfg["sound_speed"]), float(cfg["max_density_ratio"]),
                  float(cfg["viscosity"]), float(cfg["xsph_strength"]), solver.max_support, float(cfg["dt"])]
    timed("force_baseline", compute_fluid_forces, solver.count,
          force_common[:4] + force_common[9:] + force_tail)
    wp.launch(clear_vec3, dim=solver.count, inputs=[a["solid_force"][:solver.count]], device=solver.device)
    timed("force_multirate", compute_fluid_forces_multirate, solver.count,
          force_common[:12] + [solver.hydraulic_boundary[:solver.count],
                               a["water_phase"][:solver.count]] + force_common[12:] +
          [solver.fluid_pressure[:solver.count],
           solver.fluid_inverse_density[:solver.count],
           solver.fluid_mass_over_density[:solver.count],
           solver.fluid_pressure_over_density_squared[:solver.count]] +
          [solver.time_level[:solver.count], solver.time_active[:solver.count],
                          solver.deferred_fluid_impulse] + force_tail[:3] +
          force_tail[5:])
    timed("consume", consume_deferred_fluid_impulse, solver.count,
          [a["mass"][:solver.count], a["kind"][:solver.count], solver.time_level[:solver.count],
           solver.time_active[:solver.count], solver.deferred_fluid_impulse,
           a["acceleration"][:solver.count], float(cfg["dt"])])


if __name__ == "__main__":
    main()
