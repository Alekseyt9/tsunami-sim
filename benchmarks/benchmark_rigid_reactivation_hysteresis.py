"""A/B rigid/deformable reactivation hysteresis from one identical checkpoint."""

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
    ROOT / "outputs" / "sustained_surge_optimized_8_to15_video"
    / "checkpoints" / "state_00288.npz"
)


def run_variant(
    base_cfg: dict,
    checkpoint: Path,
    output: Path,
    enabled: bool,
    terminal_plastic_collapse: bool,
    frame_count: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    cfg = copy.deepcopy(base_cfg)
    cfg["checkpoint_every_frames"] = 0
    policy = cfg["v3"]["rigid_clusters"].setdefault(
        "reactivation_hysteresis", {}
    )
    policy["enabled"] = enabled
    policy.setdefault("minimum_dwell_seconds", 0.10)
    policy.setdefault("severe_acceleration_multiplier", 3.0)
    cfg["v3"]["rigid_clusters"]["terminal_plastic_collapse"] = bool(
        terminal_plastic_collapse
    )
    output.mkdir(parents=True, exist_ok=True)

    solver = HybridDelugeSolver(cfg, output, checkpoint)
    fps = int(cfg["output_fps"])
    substeps = int(math.ceil((1.0 / fps) / float(cfg["dt"])))
    dt = (1.0 / fps) / substeps
    rows: list[dict] = []
    for frame in range(solver.start_frame, solver.start_frame + frame_count):
        started = time.perf_counter()
        for _ in range(substeps):
            solver.substep(dt)
        if (
            bool(cfg.get("adaptive_refinement", True))
            and frame > 0
            and frame % int(cfg.get("refine_every_frames", 8)) == 0
        ):
            solver.refine()
        stats = solver.stats()
        wp.synchronize_device(solver.device)
        rows.append({
            "frame": frame,
            "sim_time_seconds": float(solver.time),
            "wall_seconds": float(time.perf_counter() - started),
            "particles": int(solver.count),
            "rigid_clusters": int(stats.get("rigid_clusters", 0)),
            "rigid_particles": int(stats.get("rigid_particles", 0)),
            "rigid_reactivated_fragments": int(
                stats.get("rigid_reactivated_fragments", 0)
            ),
            "rigid_reactivation_deferred_substeps": int(
                stats.get("rigid_reactivation_deferred_substeps", 0)
            ),
            "rigid_hysteresis_protected_clusters": int(
                stats.get("rigid_hysteresis_protected_clusters", 0)
            ),
            "terminal_rigid_clusters": int(
                stats.get("terminal_rigid_clusters", 0)
            ),
            "released_fragments": int(stats.get("released_fragments", 0)),
            "unsupported_fragments": int(stats.get("unsupported_fragments", 0)),
            "damage_integral_wall_m3": float(
                stats.get("damage_integral_wall_m3", 0.0)
            ),
            "damage_integral_core_m3": float(
                stats.get("damage_integral_core_m3", 0.0)
            ),
            "fluid_volume_m3": float(stats.get("fluid_volume_m3", 0.0)),
            "shallow_water_volume_m3": float(
                stats.get("shallow_water_volume_m3", 0.0)
            ),
        })

    solver.save_checkpoint(rows[-1]["frame"])

    state = {
        "position": solver.arrays["x"][:solver.count].numpy(),
        "velocity": solver.arrays["v"][:solver.count].numpy(),
        "damage": solver.arrays["damage"][:solver.count].numpy(),
        "kind": solver.arrays["kind"][:solver.count].numpy(),
        "rigid_state": solver.rigid_state.numpy(),
        "body_center": solver.body_center.numpy(),
        "body_velocity": solver.body_linear_velocity.numpy(),
    }
    normal_frames = [
        row["wall_seconds"] for row in rows
        if row["frame"] % int(cfg.get("refine_every_frames", 8)) != 0
    ]
    report = {
        "enabled": enabled,
        "terminal_plastic_collapse": terminal_plastic_collapse,
        "minimum_dwell_seconds": float(policy["minimum_dwell_seconds"]),
        "severe_acceleration_multiplier": float(
            policy["severe_acceleration_multiplier"]
        ),
        "frames": rows,
        "wall_seconds_total": float(sum(row["wall_seconds"] for row in rows)),
        "wall_seconds_normal_median": float(statistics.median(normal_frames)),
        "reactivations_during_run": (
            rows[-1]["rigid_reactivated_fragments"]
            - rows[0]["rigid_reactivated_fragments"]
        ),
        "deferred_contact_substeps_during_run": (
            rows[-1]["rigid_reactivation_deferred_substeps"]
            - rows[0]["rigid_reactivation_deferred_substeps"]
        ),
    }
    del solver
    gc.collect()
    return report, state


def rms(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def compare_states(baseline: dict[str, np.ndarray], tuned: dict[str, np.ndarray]) -> dict:
    count = min(len(baseline["kind"]), len(tuned["kind"]))
    position_delta = tuned["position"][:count] - baseline["position"][:count]
    velocity_delta = tuned["velocity"][:count] - baseline["velocity"][:count]
    damage_delta = tuned["damage"][:count] - baseline["damage"][:count]
    body_count = min(len(baseline["rigid_state"]), len(tuned["rigid_state"]))
    active_in_both = (
        (baseline["rigid_state"][:body_count] != 0)
        & (tuned["rigid_state"][:body_count] != 0)
    )
    body_center_delta = (
        tuned["body_center"][:body_count] - baseline["body_center"][:body_count]
    )
    body_velocity_delta = (
        tuned["body_velocity"][:body_count] - baseline["body_velocity"][:body_count]
    )
    return {
        "compared_particles": count,
        "particle_position_rms_m": rms(position_delta),
        "particle_position_max_m": float(
            np.max(np.linalg.norm(position_delta, axis=1))
        ),
        "particle_velocity_rms_m_s": rms(velocity_delta),
        "particle_damage_rms": rms(damage_delta),
        "rigid_state_changes": int(np.count_nonzero(
            baseline["rigid_state"][:body_count]
            != tuned["rigid_state"][:body_count]
        )),
        "common_rigid_bodies": int(np.count_nonzero(active_in_both)),
        "common_rigid_center_rms_m": rms(body_center_delta[active_in_both]),
        "common_rigid_velocity_rms_m_s": rms(
            body_velocity_delta[active_in_both]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config_v3_sustained_surge_30s.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "rigid_reactivation_hysteresis_ab",
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--only-hysteresis", action="store_true",
        help="Run only the tuned branch for a longer dwell-window audit",
    )
    parser.add_argument(
        "--disable-hysteresis", action="store_true",
        help="Disable dwell hysteresis in the tuned branch",
    )
    parser.add_argument(
        "--terminal-plastic-collapse", action="store_true",
        help="Keep plastic-collapse rubble permanently rigid in the tuned branch",
    )
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    wp.init()
    cfg = load_run_config(args.config.resolve())
    baseline = baseline_state = None
    if not args.only_hysteresis:
        baseline, baseline_state = run_variant(
            cfg, checkpoint, args.output / "baseline", False, False, args.frames
        )
    tuned, tuned_state = run_variant(
        cfg, checkpoint, args.output / "hysteresis",
        not args.disable_hysteresis,
        args.terminal_plastic_collapse,
        args.frames,
    )
    report = {
        "checkpoint": str(checkpoint),
        "config": str(args.config.resolve()),
        "hysteresis": tuned,
    }
    if baseline is not None and baseline_state is not None:
        report["baseline"] = baseline
        report["comparison"] = {
            "normal_frame_speedup": (
                baseline["wall_seconds_normal_median"]
                / max(tuned["wall_seconds_normal_median"], 1.0e-9)
            ),
            "total_run_speedup": (
                baseline["wall_seconds_total"]
                / max(tuned["wall_seconds_total"], 1.0e-9)
            ),
            **compare_states(baseline_state, tuned_state),
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
