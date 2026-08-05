"""End-to-end validation of deformable-to-rigid conversion in HybridDelugeSolver."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver
from simulation.rigid_clusters import limit_rigid_release_motion


def main() -> None:
    wp.init()
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    cfg["v3"]["rigid_clusters"]["scan_every_frames"] = 1
    cfg["v3"]["rigid_clusters"]["required_quiet_scans"] = 1
    early = cfg["v3"]["rigid_clusters"]["early_rigidification"]
    early["enabled"] = True
    early["scan_every_frames"] = 1
    early["minimum_detached_scans"] = 1
    early["required_quiet_scans"] = 1
    early["release_damage_fraction"] = 0.0
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
    # Give the fragment a small, still plausible deformation. Conversion must
    # preserve this exact visible pose instead of snapping facade anchors back
    # to the undeformed Kabsch reference shape.
    positions = solver.arrays["x"].numpy()
    center = np.mean(positions[chosen_indices], axis=0)
    local = positions[chosen_indices] - center
    local[:, 0] += 0.035 * local[:, 1]
    local[:, 2] *= 1.015
    positions[chosen_indices] = center + local
    solver.arrays["x"] = wp.array(positions, dtype=wp.vec3, device=solver.device)
    before = positions[chosen_indices].copy()
    before_distances = np.linalg.norm(before[:, None, :] - before[None, :, :], axis=2)

    solver.update_rigid_clusters()
    if solver.rigid_state_host[chosen] != 0:
        raise AssertionError("supported fragment converted to rigid rubble prematurely")
    solver.fragment_support_host[chosen] = 0.0
    solver.fragment_support = wp.array(
        solver.fragment_support_host, dtype=float, device=solver.device
    )
    solver.update_rigid_clusters()
    if solver.rigid_state_host[chosen] != 1:
        raise AssertionError("detached eligible fragment was not converted to a rigid cluster")
    if solver.rigid_proxy_enabled_host[chosen] != 1:
        raise AssertionError("eligible rigid fragment did not receive a collision proxy")
    solver._sync_rigid_render_samples()
    transition_after = solver.arrays["x"][:solver.count].numpy()[chosen_indices]
    transition_error = float(np.max(np.linalg.norm(transition_after - before, axis=1)))
    if transition_error > 5.0e-5:
        raise AssertionError(
            f"visible particle cloud jumped during rigid conversion: {transition_error}"
        )
    proxy_extent = solver.rigid_proxy_half_extent_host[chosen]
    if np.any(proxy_extent <= 0.0):
        raise AssertionError(
            f"invalid collision proxy extent: {proxy_extent}"
        )
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
    heavy_linear, heavy_angular = limit_rigid_release_motion(
        np.asarray((1.0, 8.2, 3.0), dtype=np.float32),
        np.asarray((0.0, 8.0, 0.0), dtype=np.float32),
        355000.0,
        np.asarray((8.0, 1.3, 3.0), dtype=np.float32),
        22.0, 6.0, 50000.0, 1.5, 3.0, 10.0,
    )
    expected_upward_limit = 6.0 * np.sqrt(50000.0 / 355000.0)
    if heavy_linear[1] > expected_upward_limit + 1.0e-5:
        raise AssertionError("heavy rigid release retained an impossible upward speed")
    if np.linalg.norm(heavy_angular) * np.linalg.norm((8.0, 1.3, 3.0)) > 10.0001:
        raise AssertionError("large rigid release retained an impossible tip speed")
    print(
        f"PASS: fragment {chosen} converted end-to-end ({len(chosen_indices)} particles, "
        f"mass={actual_mass:.3f} kg, proxy half-extent={proxy_extent.tolist()}, "
        f"transition error={transition_error:.3e} m, shape error={shape_error:.3e} m, "
        f"355 t upward limit={heavy_linear[1]:.3f} m/s)"
    )


if __name__ == "__main__":
    main()
