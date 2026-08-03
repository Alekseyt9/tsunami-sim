"""Sweep conservative narrow-band eligibility on early and late checkpoints."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver
from simulation.experimental_optimizations import NarrowBandVolumePreparation


DEFAULT_RUN = HERE / "outputs" / "v3_106_production_15s_24fps_20260803"


def audit_case(
    solver: HybridDelugeSolver,
    cfg: dict,
    detail_distance: float,
    velocity_rms: float,
) -> dict:
    policy = copy.deepcopy(cfg["v3"]["narrow_band_volume"])
    policy.update({
        "enabled": True,
        "mode": "diagnostic",
        "detail_distance": detail_distance,
        "maximum_velocity_rms": velocity_rms,
    })
    narrow = NarrowBandVolumePreparation(
        policy, cfg, solver.capacity, solver.device
    )
    started = time.perf_counter()
    narrow.analyze(solver.arrays, solver.count)
    wp.synchronize_device(solver.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    kind = solver.arrays["kind"][:solver.count].numpy()
    mask = narrow.interior_mask[:solver.count].numpy() != 0
    mass = solver.arrays["mass"][:solver.count].numpy()
    volume = solver.arrays["volume"][:solver.count].numpy()
    velocity = solver.arrays["v"][:solver.count].numpy()
    particle_momentum = np.sum(
        mass[mask, None] * velocity[mask], axis=0, dtype=np.float64
    )
    grid_momentum = np.asarray([
        np.sum(narrow.grid_momentum_x.numpy(), dtype=np.float64),
        np.sum(narrow.grid_momentum_y.numpy(), dtype=np.float64),
        np.sum(narrow.grid_momentum_z.numpy(), dtype=np.float64),
    ])
    diagnostics = narrow.diagnostics(int(np.count_nonzero(kind == 0)))
    return {
        "requested_detail_distance_m": detail_distance,
        "detail_distance_m": narrow.detail_distance,
        "neighbour_grid_dims": list(narrow.neighbour_grid_dims),
        "maximum_velocity_rms_m_s": velocity_rms,
        "audit_wall_ms": elapsed_ms,
        **diagnostics,
        "mass_error_kg": diagnostics["narrow_band_grid_mass_kg"]
        - float(np.sum(mass[mask], dtype=np.float64)),
        "volume_error_m3": diagnostics["narrow_band_grid_volume_m3"]
        - float(np.sum(volume[mask], dtype=np.float64)),
        "momentum_error_kg_m_s": (grid_momentum - particle_momentum).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_114_narrow_band_sweep_20260803.json",
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()
    report: dict[str, object] = {"checkpoints": {}}
    for label, frame in (("early", 24), ("late", 336)):
        variant = copy.deepcopy(cfg)
        variant["v3"]["implicit_fluid"]["enabled"] = False
        variant["v3"]["narrow_band_volume"]["enabled"] = False
        checkpoint = args.run / "checkpoints" / f"state_{frame:05d}.npz"
        solver = HybridDelugeSolver(variant, args.output.parent / label, checkpoint)
        solver.update_water_surface()
        cases = []
        for detail_distance in (0.75, 1.0, 1.5, 2.0):
            for velocity_rms in (1.0, 3.0, 6.0):
                cases.append(audit_case(
                    solver, variant, detail_distance, velocity_rms
                ))
        report["checkpoints"][label] = {
            "checkpoint": str(checkpoint.resolve()),
            "cases": cases,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
