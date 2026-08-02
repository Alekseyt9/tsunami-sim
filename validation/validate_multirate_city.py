"""Compare full-city V3.2 state with uniform small stepping."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver


def run(enabled: bool, output: Path, substeps: int):
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    cfg["v3"]["multirate"]["enabled"] = enabled
    # Exercise the deformable graph without a water impact.  This catches a
    # false local-support classification that would make an unlocked facade
    # sag or damage itself before the wave reaches the building.
    cfg["v3"]["initially_active_buildings"] = [0]
    solver = HybridDelugeSolver(cfg, output)
    dt = float(cfg["dt"])
    for _ in range(substeps):
        solver.substep(dt)
    wp.synchronize_device(solver.device)
    kind = solver.arrays["kind"][:solver.count].numpy()
    building = solver.arrays["building_id"][:solver.count].numpy()
    damage = solver.arrays["damage"][:solver.count].numpy()
    position = solver.arrays["x"][:solver.count].numpy()
    rest_position = solver.arrays["rest_x"][:solver.count].numpy()
    fluid = kind == 0
    unlocked = (kind != 0) & (building == 0)
    return (
        position[fluid],
        solver.arrays["v"][:solver.count].numpy()[fluid],
        solver.arrays["mass"][:solver.count].numpy()[fluid],
        float(damage[unlocked].max(initial=0.0)),
        float(np.sqrt(np.mean(np.sum((position[unlocked] - rest_position[unlocked]) ** 2, axis=1)))),
    )


def main() -> None:
    wp.init()
    root = HERE / "outputs" / "multirate_city_validation_20260801"
    root.mkdir(parents=True, exist_ok=True)
    substeps = 128  # divisible by the longest stride
    reference_x, reference_v, mass, reference_damage, reference_solid_rms = run(
        False, root / "baseline", substeps
    )
    multirate_x, multirate_v, multirate_mass, multirate_damage, multirate_solid_rms = run(
        True, root / "multirate", substeps
    )
    if not np.array_equal(mass, multirate_mass):
        raise AssertionError("fluid mass layout changed")
    reference_momentum = (mass[:, None] * reference_v).sum(axis=0, dtype=np.float64)
    multirate_momentum = (mass[:, None] * multirate_v).sum(axis=0, dtype=np.float64)
    momentum_scale = max(float(np.linalg.norm(reference_momentum)), 1.0)
    momentum_error = float(np.linalg.norm(multirate_momentum - reference_momentum) / momentum_scale)
    reference_energy = float((0.5 * mass * np.sum(reference_v * reference_v, axis=1)).sum(dtype=np.float64))
    multirate_energy = float((0.5 * mass * np.sum(multirate_v * multirate_v, axis=1)).sum(dtype=np.float64))
    energy_error = abs(multirate_energy - reference_energy) / max(reference_energy, 1.0)
    position_rms = float(np.sqrt(np.mean(np.sum((multirate_x - reference_x) ** 2, axis=1))))
    if momentum_error > 0.01:
        raise AssertionError(f"city momentum difference is {momentum_error:.3%}")
    if energy_error > 0.02:
        raise AssertionError(f"city kinetic-energy difference is {energy_error:.3%}")
    if max(reference_damage, multirate_damage) > 1.0e-6:
        raise AssertionError(
            "an unlocked dry building accumulated damage before water impact: "
            f"baseline={reference_damage:.3e}, multirate={multirate_damage:.3e}"
        )
    if max(reference_solid_rms, multirate_solid_rms) > 1.0e-4:
        raise AssertionError(
            "an unlocked dry building moved under a false support-loss signal: "
            f"baseline={reference_solid_rms:.3e} m, multirate={multirate_solid_rms:.3e} m"
        )
    print(
        f"PASS: city momentum delta={momentum_error:.4%}, energy delta={energy_error:.4%}, "
        f"position RMS={position_rms:.3e} m after {substeps} substeps; "
        f"unlocked dry building damage=0 and RMS<={max(reference_solid_rms, multirate_solid_rms):.3e} m"
    )


if __name__ == "__main__":
    main()
