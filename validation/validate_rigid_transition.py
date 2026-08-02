"""End-to-end validation of deformable-to-rigid conversion in HybridDelugeSolver."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver


def main() -> None:
    wp.init()
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    cfg["v3"]["rigid_clusters"]["scan_every_frames"] = 1
    cfg["v3"]["rigid_clusters"]["required_quiet_scans"] = 1
    output = HERE / "outputs" / "rigid_transition_validation_20260801"
    output.mkdir(parents=True, exist_ok=True)
    solver = HybridDelugeSolver(cfg, output)

    fragment = solver.fragment_id[:solver.count].numpy()
    kind = solver.arrays["kind"][:solver.count].numpy()
    base_fixed = solver.base_fixed[:solver.count].numpy()
    chosen = -1
    chosen_indices = np.empty(0, dtype=np.int64)
    for fid in range(solver.fragment_count):
        indices = np.flatnonzero((fragment == fid) & (kind != 0))
        if len(indices) >= 6 and not np.any(base_fixed[indices] != 0):
            chosen = fid
            chosen_indices = indices
            break
    if chosen < 0:
        raise AssertionError("no non-foundation fragment available for transition test")

    damage = solver.arrays["damage"].numpy()
    damage[chosen_indices] = 1.0
    solver.arrays["damage"] = wp.array(damage, dtype=float, device=solver.device)
    fixed = solver.arrays["fixed"].numpy()
    fixed[chosen_indices] = 0
    solver.arrays["fixed"] = wp.array(fixed, dtype=wp.int32, device=solver.device)
    before = solver.arrays["x"][:solver.count].numpy()[chosen_indices]
    before_distances = np.linalg.norm(before[:, None, :] - before[None, :, :], axis=2)

    solver.update_rigid_clusters()
    if solver.rigid_state_host[chosen] != 1:
        raise AssertionError("eligible fragment was not converted to a rigid cluster")
    expected_mass = float(solver.arrays["mass"][:solver.count].numpy()[chosen_indices].sum(dtype=np.float64))
    actual_mass = float(solver.body_mass.numpy()[chosen])
    if abs(expected_mass - actual_mass) > max(1.0e-4, expected_mass * 2.0e-6):
        raise AssertionError("body mass changed during conversion")

    solver.substep(0.00012)
    after = solver.arrays["x"][:solver.count].numpy()[chosen_indices]
    after_distances = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=2)
    shape_error = float(np.max(np.abs(after_distances - before_distances)))
    if shape_error > 5.0e-4:
        raise AssertionError(f"shape changed after full solver substep: {shape_error}")
    print(
        f"PASS: fragment {chosen} converted end-to-end ({len(chosen_indices)} particles, "
        f"mass={actual_mass:.3f} kg, shape error={shape_error:.3e} m)"
    )


if __name__ == "__main__":
    main()
