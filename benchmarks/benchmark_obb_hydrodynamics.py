"""A/B the conservative OBB hydrodynamic path from one late checkpoint.

This gate deliberately keeps the original terminal particles alive. It tests
coupling stability, impulse symmetry and overhead before sample shedding is
allowed to change the production neighbour cloud.
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver, load_run_config


DEFAULT_CHECKPOINT = (
    ROOT / "outputs" / "terminal_plastic_rubble_ab_checkpoint288_14f"
    / "hysteresis" / "checkpoints" / "state_00302.npz"
)


def rms(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def axis_rms(values: np.ndarray) -> list[float]:
    if not values.size:
        return [0.0, 0.0, 0.0]
    return [rms(values[:, axis]) for axis in range(3)]


def run_variant(
    config: dict, checkpoint: Path, output: Path, enabled: bool,
    shed_terminal_samples: bool, frames: int
) -> tuple[dict, dict[str, np.ndarray]]:
    cfg = copy.deepcopy(config)
    cfg["checkpoint_every_frames"] = 0
    policy = cfg["v3"]["rigid_clusters"].setdefault("proxy_hydrodynamics", {})
    policy["enabled"] = enabled
    policy["shed_terminal_samples"] = shed_terminal_samples
    if not enabled:
        policy.setdefault("occupancy", {})["enabled"] = False
    policy.setdefault("analytic_contact", {})["enabled"] = enabled
    output.mkdir(parents=True, exist_ok=True)
    solver = HybridDelugeSolver(cfg, output, checkpoint)
    fps = float(cfg["output_fps"])
    substeps = int(math.ceil((1.0 / fps) / float(cfg["dt"])))
    dt = (1.0 / fps) / substeps
    rows: list[dict] = []
    for frame in range(solver.start_frame, solver.start_frame + frames):
        started = time.perf_counter()
        for _ in range(substeps):
            solver.substep(dt)
        wp.synchronize_device(solver.device)
        elapsed = time.perf_counter() - started
        stats = solver.stats()
        rows.append({
            "frame": int(frame),
            "sim_time_seconds": float(solver.time),
            "wall_seconds": float(elapsed),
            "particles": int(solver.count),
            "rigid_clusters": int(stats.get("rigid_clusters", 0)),
            "terminal_rigid_clusters": int(stats.get("terminal_rigid_clusters", 0)),
            "quadrature_active": int(stats.get("rigid_obb_quadrature_active", 0)),
            "quadrature_wet": int(stats.get("rigid_obb_quadrature_wet", 0)),
            "analytic_contact_candidates": int(
                stats.get("rigid_obb_fluid_contact_candidates", 0)
            ),
            "analytic_query_particles": int(
                stats.get("rigid_obb_fluid_query_particles", 0)
            ),
            "analytic_contacts": int(
                stats.get("rigid_obb_fluid_contacts", 0)
            ),
            "analytic_contact_cache_active_particles": int(
                stats.get("rigid_obb_contact_cache_active_particles", 0)
            ),
            "fluid_verlet_entries": int(stats.get("fluid_verlet_entries", 0)),
            "fluid_volume_m3": float(stats.get("fluid_volume_m3", 0.0)),
            "shallow_water_volume_m3": float(
                stats.get("shallow_water_volume_m3", 0.0)
            ),
            "damage_integral_wall_m3": float(
                stats.get("damage_integral_wall_m3", 0.0)
            ),
            "damage_integral_core_m3": float(
                stats.get("damage_integral_core_m3", 0.0)
            ),
        })
    state = {
        "position": solver.arrays["x"][:solver.count].numpy(),
        "velocity": solver.arrays["v"][:solver.count].numpy(),
        "body_center": solver.body_center.numpy(),
        "body_velocity": solver.body_linear_velocity.numpy(),
        "rigid_state": solver.rigid_state.numpy(),
    }
    report = {
        "enabled": enabled,
        "shed_terminal_samples": shed_terminal_samples,
        "keeps_render_checkpoint_samples": True,
        "keeps_samples_in_sph_boundary": not shed_terminal_samples,
        "frames": rows,
        "wall_seconds_total": float(sum(row["wall_seconds"] for row in rows)),
        "wall_seconds_median": float(statistics.median(
            row["wall_seconds"] for row in rows
        )),
    }
    del solver
    gc.collect()
    return report, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config_v3_sustained_surge_30s.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "obb_hydrodynamics_ab_checkpoint302",
    )
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--candidate-cache", action="store_true",
        help="enable the conservative persistent particle/OBB candidate cache",
    )
    parser.add_argument(
        "--occupancy", action="store_true",
        help="enable 24-cell material occupancy on terminal OBBs",
    )
    parser.add_argument(
        "--cache-refit-evaluations", type=int, default=None,
        help="override cached BVH refresh interval measured in contact evaluations",
    )
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    wp.init()
    cfg = load_run_config(args.config.resolve())
    analytic = cfg["v3"]["rigid_clusters"]["proxy_hydrodynamics"].setdefault(
        "analytic_contact", {}
    )
    analytic.setdefault("candidate_cache", {})["enabled"] = bool(
        args.candidate_cache
    )
    cfg["v3"]["rigid_clusters"]["proxy_hydrodynamics"].setdefault(
        "occupancy", {}
    )["enabled"] = bool(args.occupancy)
    if args.cache_refit_evaluations is not None:
        if args.cache_refit_evaluations < 1:
            raise ValueError("--cache-refit-evaluations must be positive")
        analytic["bvh_refit_every_substeps"] = int(
            args.cache_refit_evaluations
        )
    baseline, baseline_state = run_variant(
        cfg, checkpoint, args.output / "baseline", False, False, args.frames
    )
    coupled, coupled_state = run_variant(
        cfg, checkpoint, args.output / "coupled", True, False, args.frames
    )
    shed, shed_state = run_variant(
        cfg, checkpoint, args.output / "shed", True, True, args.frames
    )
    count = min(len(baseline_state["position"]), len(coupled_state["position"]))
    body_count = min(
        len(baseline_state["rigid_state"]), len(coupled_state["rigid_state"])
    )
    common_rigid = (
        (baseline_state["rigid_state"][:body_count] != 0)
        & (coupled_state["rigid_state"][:body_count] != 0)
    )
    shed_count = min(len(baseline_state["position"]), len(shed_state["position"]))
    shed_body_count = min(
        len(baseline_state["rigid_state"]), len(shed_state["rigid_state"])
    )
    shed_common_rigid = (
        (baseline_state["rigid_state"][:shed_body_count] != 0)
        & (shed_state["rigid_state"][:shed_body_count] != 0)
    )
    coupled_shed_count = min(
        len(coupled_state["position"]), len(shed_state["position"])
    )
    coupled_shed_body_count = min(
        len(coupled_state["rigid_state"]), len(shed_state["rigid_state"])
    )
    coupled_shed_common_rigid = (
        (coupled_state["rigid_state"][:coupled_shed_body_count] != 0)
        & (shed_state["rigid_state"][:coupled_shed_body_count] != 0)
    )
    report = {
        "checkpoint": str(checkpoint),
        "config": str(args.config.resolve()),
        "baseline": baseline,
        "coupled": coupled,
        "shed": shed,
        "comparison": {
            "total_runtime_ratio_coupled_over_baseline": (
                coupled["wall_seconds_total"]
                / max(baseline["wall_seconds_total"], 1.0e-9)
            ),
            "particle_position_rms_m": rms(
                coupled_state["position"][:count] - baseline_state["position"][:count]
            ),
            "particle_velocity_rms_m_s": rms(
                coupled_state["velocity"][:count] - baseline_state["velocity"][:count]
            ),
            "common_rigid_bodies": int(np.count_nonzero(common_rigid)),
            "rigid_center_rms_m": rms(
                coupled_state["body_center"][:body_count][common_rigid]
                - baseline_state["body_center"][:body_count][common_rigid]
            ),
            "rigid_velocity_rms_m_s": rms(
                coupled_state["body_velocity"][:body_count][common_rigid]
                - baseline_state["body_velocity"][:body_count][common_rigid]
            ),
            "shed_total_runtime_ratio_over_baseline": (
                shed["wall_seconds_total"]
                / max(baseline["wall_seconds_total"], 1.0e-9)
            ),
            "shed_particle_position_rms_m": rms(
                shed_state["position"][:shed_count]
                - baseline_state["position"][:shed_count]
            ),
            "shed_particle_velocity_rms_m_s": rms(
                shed_state["velocity"][:shed_count]
                - baseline_state["velocity"][:shed_count]
            ),
            "shed_common_rigid_bodies": int(np.count_nonzero(shed_common_rigid)),
            "shed_rigid_center_rms_m": rms(
                shed_state["body_center"][:shed_body_count][shed_common_rigid]
                - baseline_state["body_center"][:shed_body_count][shed_common_rigid]
            ),
            "shed_rigid_center_axis_rms_m": axis_rms(
                shed_state["body_center"][:shed_body_count][shed_common_rigid]
                - baseline_state["body_center"][:shed_body_count][shed_common_rigid]
            ),
            "shed_rigid_velocity_rms_m_s": rms(
                shed_state["body_velocity"][:shed_body_count][shed_common_rigid]
                - baseline_state["body_velocity"][:shed_body_count][shed_common_rigid]
            ),
            "shed_rigid_velocity_axis_rms_m_s": axis_rms(
                shed_state["body_velocity"][:shed_body_count][shed_common_rigid]
                - baseline_state["body_velocity"][:shed_body_count][shed_common_rigid]
            ),
            "shed_runtime_ratio_over_coupled": (
                shed["wall_seconds_total"]
                / max(coupled["wall_seconds_total"], 1.0e-9)
            ),
            "shed_vs_coupled_particle_position_rms_m": rms(
                shed_state["position"][:coupled_shed_count]
                - coupled_state["position"][:coupled_shed_count]
            ),
            "shed_vs_coupled_particle_velocity_rms_m_s": rms(
                shed_state["velocity"][:coupled_shed_count]
                - coupled_state["velocity"][:coupled_shed_count]
            ),
            "shed_vs_coupled_common_rigid_bodies": int(
                np.count_nonzero(coupled_shed_common_rigid)
            ),
            "shed_vs_coupled_rigid_center_rms_m": rms(
                shed_state["body_center"][:coupled_shed_body_count][coupled_shed_common_rigid]
                - coupled_state["body_center"][:coupled_shed_body_count][coupled_shed_common_rigid]
            ),
            "shed_vs_coupled_rigid_center_axis_rms_m": axis_rms(
                shed_state["body_center"][:coupled_shed_body_count][coupled_shed_common_rigid]
                - coupled_state["body_center"][:coupled_shed_body_count][coupled_shed_common_rigid]
            ),
            "shed_vs_coupled_rigid_velocity_rms_m_s": rms(
                shed_state["body_velocity"][:coupled_shed_body_count][coupled_shed_common_rigid]
                - coupled_state["body_velocity"][:coupled_shed_body_count][coupled_shed_common_rigid]
            ),
            "shed_vs_coupled_rigid_velocity_axis_rms_m_s": axis_rms(
                shed_state["body_velocity"][:coupled_shed_body_count][coupled_shed_common_rigid]
                - coupled_state["body_velocity"][:coupled_shed_body_count][coupled_shed_common_rigid]
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
