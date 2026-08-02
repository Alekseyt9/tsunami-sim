"""Classify and render water phases on the late production frame-96 checkpoint."""

from __future__ import annotations

from _bootstrap import ROOT

import copy
import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver  # noqa: E402


def main() -> None:
    wp.init()
    checkpoint = (
        ROOT / "outputs" / "v3_21_proxy_ab_checkpoint96_20260802" /
        "migrated" / "checkpoints" / "state_00096.npz"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"late migrated checkpoint is absent: {checkpoint}")
    with np.load(checkpoint, allow_pickle=False) as saved:
        cfg = copy.deepcopy(json.loads(str(saved["config"])))
    cfg["v3"]["water_mesh"]["enabled"] = False
    cfg["render"]["view_width"] = 320
    cfg["render"]["view_height"] = 180
    cfg["render"]["views"] = {"original": cfg["render"]["views"]["original"]}
    output = ROOT / "outputs" / "validation_water_phase_checkpoint96"
    solver = HybridDelugeSolver(cfg, output, checkpoint)

    mass_before = float(np.sum(solver.arrays["mass"][:solver.count].numpy(), dtype=np.float64))
    momentum_before = np.sum(
        solver.arrays["mass"][:solver.count].numpy()[:, None]
        * solver.arrays["v"][:solver.count].numpy(), axis=0, dtype=np.float64,
    )
    for _ in range(3):
        solver.update_water_surface()
    wp.synchronize_device(solver.device)
    phase = solver.arrays["water_phase"][:solver.count].numpy()
    kind = solver.arrays["kind"][:solver.count].numpy()
    mask = solver.arrays["water_surface_mask"][:solver.count].numpy() != 0
    fluid_surface = (kind == 0) & mask
    counts = [int(np.count_nonzero(fluid_surface & (phase == value))) for value in range(3)]
    if counts[0] == 0:
        raise AssertionError("late checkpoint has no connected surface after phase separation")

    mass_after = float(np.sum(solver.arrays["mass"][:solver.count].numpy(), dtype=np.float64))
    momentum_after = np.sum(
        solver.arrays["mass"][:solver.count].numpy()[:, None]
        * solver.arrays["v"][:solver.count].numpy(), axis=0, dtype=np.float64,
    )
    if mass_after != mass_before or not np.array_equal(momentum_after, momentum_before):
        raise AssertionError("classification modified production mass or momentum")
    image = solver.renderer.render(solver.arrays, solver.count, None, 96, solver.time, {})
    if image.shape != (180, 320, 3) or not np.any(image):
        raise AssertionError("phase-aware water renderer produced an invalid frame")
    print(
        f"PASS: checkpoint 96 phases connected={counts[0]:,}, sheet={counts[1]:,}, "
        f"ballistic={counts[2]:,}; 320x180 phase-aware render is valid"
    )


if __name__ == "__main__":
    main()
