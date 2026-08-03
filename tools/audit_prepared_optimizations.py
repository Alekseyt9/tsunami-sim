"""Audit prepared implicit, early-rigid, and narrow-band paths on a checkpoint."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "optimization_audit")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["v3"]["implicit_fluid"]["enabled"] = True
    cfg["v3"]["narrow_band_volume"]["enabled"] = True
    cfg["v3"]["narrow_band_volume"]["analyze_every_frames"] = 1
    cfg["v3"]["rigid_clusters"]["early_rigidification"]["enabled"] = True
    args.output.mkdir(parents=True, exist_ok=True)

    wp.init()
    solver = HybridDelugeSolver(cfg, args.output, args.checkpoint)
    solver.update_water_surface()
    narrow = solver.narrow_band_volume_preparation
    narrow.analyze(solver.arrays, solver.count)
    wp.synchronize_device(solver.device)

    kind = solver.arrays["kind"][:solver.count].numpy()
    mask = narrow.interior_mask[:solver.count].numpy() != 0
    mass = solver.arrays["mass"][:solver.count].numpy()
    volume = solver.arrays["volume"][:solver.count].numpy()
    velocity = solver.arrays["v"][:solver.count].numpy()
    position = solver.arrays["x"][:solver.count].numpy()
    surface_mask = solver.arrays["water_surface_mask"][:solver.count].numpy() != 0
    water_phase = solver.arrays["water_phase"][:solver.count].numpy()
    acceleration = solver.arrays["acceleration"][:solver.count].numpy()
    radius = solver.arrays["radius"][:solver.count].numpy()

    particle_mass = float(np.sum(mass[mask], dtype=np.float64))
    particle_volume = float(np.sum(volume[mask], dtype=np.float64))
    particle_momentum = np.sum(
        mass[mask, None] * velocity[mask], axis=0, dtype=np.float64
    )
    grid_momentum = np.asarray(
        [
            np.sum(narrow.grid_momentum_x.numpy(), dtype=np.float64),
            np.sum(narrow.grid_momentum_y.numpy(), dtype=np.float64),
            np.sum(narrow.grid_momentum_z.numpy(), dtype=np.float64),
        ]
    )
    grid_mass = float(np.sum(narrow.grid_mass.numpy(), dtype=np.float64))
    grid_volume = float(np.sum(narrow.grid_volume.numpy(), dtype=np.float64))

    implicit = solver.implicit_fluid_preparation.analyze(
        radius, velocity, acceleration, kind,
        float(cfg["dt"]), float(cfg["output_fps"]),
    )

    rigid_before = int(np.count_nonzero(solver.rigid_state.numpy()))
    required_scans = max(
        2,
        int(cfg["v3"]["rigid_clusters"]["early_rigidification"].get(
            "minimum_detached_scans", 2
        )),
    )
    for _ in range(required_scans):
        # Force each audit call onto a configured scan boundary without
        # advancing physics. This measures eligibility only in the private
        # checkpoint instance.
        solver.rigid_stats_calls = 0
        solver.v3_cfg["rigid_clusters"]["early_rigidification"][
            "scan_every_frames"
        ] = 1
        solver.update_rigid_clusters()
    rigid_after = int(np.count_nonzero(solver.rigid_state.numpy()))
    rigid_state = solver.rigid_state.numpy()
    fragment = solver.fragment_id[:solver.count].numpy()
    body_linear = solver.body_linear_velocity.numpy()
    body_angular = solver.body_angular_velocity.numpy()
    body_extent = solver.rigid_proxy_half_extent.numpy()
    sleeping_policy = cfg["v3"]["rigid_clusters"].get("sleeping", {})
    sleeping_candidates = 0
    sleeping_candidate_particles = 0
    solid_fragment_indices = np.flatnonzero((fragment >= 0) & (kind != 0))
    fragment_order = np.argsort(fragment[solid_fragment_indices], kind="stable")
    sorted_particle_indices = solid_fragment_indices[fragment_order]
    sorted_fragments = fragment[sorted_particle_indices]
    for fid in np.flatnonzero(rigid_state != 0):
        first = int(np.searchsorted(sorted_fragments, fid, side="left"))
        last = int(np.searchsorted(sorted_fragments, fid, side="right"))
        indices = sorted_particle_indices[first:last]
        if len(indices) == 0:
            continue
        bottom = float(np.min(position[indices, 1] - radius[indices]))
        linear_speed = float(np.linalg.norm(body_linear[fid]))
        tip_speed = float(np.linalg.norm(body_angular[fid]) * max(np.linalg.norm(body_extent[fid]), 0.25))
        if (
            bottom <= float(sleeping_policy.get("ground_margin", 0.05))
            and linear_speed <= float(sleeping_policy.get("maximum_linear_speed", 0.12))
            and tip_speed <= float(sleeping_policy.get("maximum_tip_speed", 0.18))
        ):
            sleeping_candidates += 1
            sleeping_candidate_particles += len(indices)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "particles": solver.count,
        "fluid_particles": int(np.count_nonzero(kind == 0)),
        "fluid_surface_particles": int(np.count_nonzero((kind == 0) & surface_mask)),
        "connected_fluid_particles": int(np.count_nonzero((kind == 0) & (water_phase == 0))),
        "thin_sheet_particles": int(np.count_nonzero((kind == 0) & (water_phase == 1))),
        "ballistic_droplet_particles": int(np.count_nonzero((kind == 0) & (water_phase == 2))),
        **implicit,
        **narrow.diagnostics(int(np.count_nonzero(kind == 0))),
        "narrow_band_particle_mass_kg": particle_mass,
        "narrow_band_grid_mass_error_kg": grid_mass - particle_mass,
        "narrow_band_particle_volume_m3": particle_volume,
        "narrow_band_grid_volume_error_m3": grid_volume - particle_volume,
        "narrow_band_grid_momentum_error_kg_m_s": (
            grid_momentum - particle_momentum
        ).tolist(),
        "early_rigid_clusters_before": rigid_before,
        "early_rigid_clusters_after": rigid_after,
        "early_rigid_eligible_conversions": rigid_after - rigid_before,
        "sleeping_rigid_candidate_clusters": sleeping_candidates,
        "sleeping_rigid_candidate_particles": sleeping_candidate_particles,
    }
    report_path = args.output / "optimization_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
