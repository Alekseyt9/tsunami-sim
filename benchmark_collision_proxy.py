"""A/B benchmark rigid collision proxies from one identical late checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver
from solver_base import compose_quad_view


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE.parent / ".publish" / "tsunami-sim" / "outputs"
    / "v3_7_prefix97_for_v3_8_20260802" / "checkpoints" / "state_00096.npz"
)


def checkpoint_config(path: Path) -> tuple[dict, int]:
    with np.load(path, allow_pickle=False) as state:
        cfg = json.loads(str(state["config"]))
        frame = int(state["frame"])
    proxy = cfg.setdefault("v3", {}).setdefault("rigid_clusters", {}).setdefault(
        "collision_proxy", {}
    )
    proxy.update({
        "enabled": True,
        "minimum_particles": int(proxy.get("minimum_particles", 12)),
        "padding_scale": float(proxy.get("padding_scale", 0.70)),
        "maximum_penetration": float(proxy.get("maximum_penetration", 0.35)),
    })
    return cfg, frame


def migrate_checkpoint(source: Path, cfg: dict, frame: int, output: Path) -> Path:
    migrated = output / "migrated" / "checkpoints" / f"state_{frame:05d}.npz"
    v3_migrated = migrated.with_name("v3_" + migrated.name)
    if migrated.exists() and v3_migrated.exists():
        with np.load(v3_migrated, allow_pickle=False) as state:
            if "support_edge_fragments" in state and "rigid_proxy_enabled" in state:
                print(f"Reusing migrated checkpoint: {migrated}")
                return migrated
    solver = HybridDelugeSolver(copy.deepcopy(cfg), output / "migrated", source)
    solver.save_checkpoint(frame)
    del solver
    gc.collect()
    return migrated


def run_variant(
    name: str,
    enabled: bool,
    cfg: dict,
    checkpoint: Path,
    output: Path,
    profile_substeps: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg["v3"]["rigid_clusters"]["collision_proxy"]["enabled"] = enabled
    solver = HybridDelugeSolver(variant_cfg, output / name, checkpoint)
    substeps_per_frame = int(math.ceil(
        (1.0 / float(variant_cfg["output_fps"])) / float(variant_cfg["dt"])
    ))
    dt = (1.0 / float(variant_cfg["output_fps"])) / substeps_per_frame

    # Warm lazy CUDA modules identically; this step is excluded from timing but
    # remains in both trajectories.
    solver.substep(dt)
    wp.synchronize_device(solver.device)
    totals: dict[str, float] = defaultdict(float)
    for _ in range(profile_substeps):
        with wp.ScopedTimer(
            f"{name}_substep", print=False, synchronize=True,
            cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN,
        ) as timer:
            solver.substep(dt)
        for timing in timer.timing_results:
            totals[timing.name] += timing.elapsed

    wp.synchronize_device(solver.device)
    physics_started = time.perf_counter()
    for _ in range(substeps_per_frame):
        solver.substep(dt)
    wp.synchronize_device(solver.device)
    physics_frame_ms = (time.perf_counter() - physics_started) * 1000.0

    kind = solver.arrays["kind"][:solver.count].numpy()
    solid = kind != 0
    trajectory = {
        "solid_position": solver.arrays["x"][:solver.count].numpy()[solid],
        "solid_velocity": solver.arrays["v"][:solver.count].numpy()[solid],
        "solid_damage": solver.arrays["damage"][:solver.count].numpy()[solid],
        "body_center": solver.body_center.numpy(),
        "body_linear_velocity": solver.body_linear_velocity.numpy(),
        "rigid_state": solver.rigid_state.numpy(),
    }

    post_started = time.perf_counter()
    stats = solver.stats()
    rendered = {
        view: renderer.render(solver.arrays, solver.count, None, 0, solver.time, stats)
        for view, renderer in solver.renderers.items()
    }
    render_cfg = variant_cfg["render"]
    compose_quad_view(
        rendered,
        list(render_cfg.get("quad_order", rendered.keys())),
        int(render_cfg["width"]), int(render_cfg["height"]),
    )
    wp.synchronize_device(solver.device)
    post_frame_ms = (time.perf_counter() - post_started) * 1000.0

    average_kernel = {
        key: value / profile_substeps for key, value in sorted(totals.items())
    }
    contact_names = (
        "accumulate_rigid_body_loads", "accumulate_rigid_contacts",
        "accumulate_rigid_proxy_boundaries", "accumulate_rigid_proxy_contacts",
    )
    structural_names = ("compute_clustered_solid_forces",) + contact_names
    result = {
        "name": name,
        "proxy_enabled": enabled,
        "particles": solver.count,
        "rigid_bodies": int(np.count_nonzero(trajectory["rigid_state"])),
        "rigid_collision_proxies": int(stats.get("rigid_collision_proxies", 0)),
        "rigid_proxy_pairs": int(stats.get("rigid_proxy_pairs", 0)),
        "substeps_per_output_frame": substeps_per_frame,
        "dt_seconds": dt,
        "profile_substeps": profile_substeps,
        "profile_average_kernel_ms": average_kernel,
        "profile_contact_ms": float(sum(
            value for key, value in average_kernel.items()
            if any(token in key for token in contact_names)
        )),
        "profile_structural_and_contact_ms": float(sum(
            value for key, value in average_kernel.items()
            if any(token in key for token in structural_names)
        )),
        "physics_output_frame_ms": physics_frame_ms,
        "stats_render_compose_ms": post_frame_ms,
        "complete_output_frame_ms": physics_frame_ms + post_frame_ms,
    }
    del solver
    gc.collect()
    return result, trajectory


def trajectory_delta(off: dict[str, np.ndarray], on: dict[str, np.ndarray]) -> dict:
    result: dict[str, float | int] = {}
    for name in ("solid_position", "solid_velocity"):
        delta = on[name] - off[name]
        magnitude = np.linalg.norm(delta, axis=1)
        result[f"{name}_rms"] = float(np.sqrt(np.mean(magnitude * magnitude)))
        result[f"{name}_max"] = float(np.max(magnitude))
    damage_delta = np.abs(on["solid_damage"] - off["solid_damage"])
    result["solid_damage_max"] = float(np.max(damage_delta))
    active = (off["rigid_state"] != 0) | (on["rigid_state"] != 0)
    result["compared_rigid_bodies"] = int(np.count_nonzero(active))
    if np.any(active):
        center_delta = np.linalg.norm(on["body_center"][active] - off["body_center"][active], axis=1)
        velocity_delta = np.linalg.norm(
            on["body_linear_velocity"][active] - off["body_linear_velocity"][active], axis=1
        )
        result["rigid_center_rms_m"] = float(np.sqrt(np.mean(center_delta * center_delta)))
        result["rigid_center_max_m"] = float(np.max(center_delta))
        result["rigid_velocity_rms_m_s"] = float(np.sqrt(np.mean(velocity_delta * velocity_delta)))
        result["rigid_velocity_max_m_s"] = float(np.max(velocity_delta))
    result["rigid_state_changes"] = int(np.count_nonzero(off["rigid_state"] != on["rigid_state"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_21_proxy_ab_checkpoint96_20260802",
    )
    parser.add_argument("--profile-substeps", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    cfg, frame = checkpoint_config(args.checkpoint)
    migrated = migrate_checkpoint(args.checkpoint, cfg, frame, args.output)
    off_result, off_trajectory = run_variant(
        "proxy_off", False, cfg, migrated, args.output, args.profile_substeps
    )
    on_result, on_trajectory = run_variant(
        "proxy_on", True, cfg, migrated, args.output, args.profile_substeps
    )
    comparison = trajectory_delta(off_trajectory, on_trajectory)
    comparison.update({
        "contact_speedup": (
            off_result["profile_contact_ms"] / on_result["profile_contact_ms"]
            if on_result["profile_contact_ms"] > 0.0 else None
        ),
        "physics_frame_speedup": (
            off_result["physics_output_frame_ms"] / on_result["physics_output_frame_ms"]
        ),
        "complete_frame_speedup": (
            off_result["complete_output_frame_ms"] / on_result["complete_output_frame_ms"]
        ),
    })
    report = {
        "source_checkpoint": str(args.checkpoint),
        "migrated_checkpoint": str(migrated),
        "proxy_off": off_result,
        "proxy_on": on_result,
        "comparison": comparison,
    }
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
