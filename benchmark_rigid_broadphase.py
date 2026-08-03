"""A/B benchmark legacy all-pairs and GPU-BVH rigid collision broadphases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE / "outputs" / "v3_45_cinematic_environment_25s_24fps_20260803"
    / "checkpoints" / "state_00528.npz"
)


def run_variant(
    name: str,
    broadphase: str,
    cfg: dict,
    checkpoint: Path,
    output: Path,
    profile_substeps: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    variant = copy.deepcopy(cfg)
    proxy = variant["v3"]["rigid_clusters"]["collision_proxy"]
    proxy["enabled"] = True
    proxy["broadphase"] = broadphase
    # Rendering/surface extraction is not part of a physics-substep benchmark.
    variant["v3"]["water_mesh"]["enabled"] = False
    solver = HybridDelugeSolver(variant, output / name, checkpoint)
    substeps_per_frame = int(math.ceil(
        (1.0 / float(variant["output_fps"])) / float(variant["dt"])
    ))
    dt = (1.0 / float(variant["output_fps"])) / substeps_per_frame

    # Compile/lazily load all modules before timing.
    solver.substep(dt)
    wp.synchronize_device(solver.device)
    kernel_totals: dict[str, float] = defaultdict(float)
    wall_started = time.perf_counter()
    for _ in range(profile_substeps):
        with wp.ScopedTimer(
            name, print=False, synchronize=True,
            cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN,
        ) as timer:
            solver.substep(dt)
        for timing in timer.timing_results:
            kernel_totals[timing.name] += timing.elapsed
    wp.synchronize_device(solver.device)
    wall_ms = (time.perf_counter() - wall_started) * 1000.0 / profile_substeps

    state = solver.rigid_state.numpy()
    active = state != 0
    result = {
        "name": name,
        "broadphase": broadphase,
        "profile_substeps": profile_substeps,
        "average_wall_ms_per_substep": wall_ms,
        "average_gpu_kernel_ms_per_substep": float(
            sum(kernel_totals.values()) / profile_substeps
        ),
        "average_kernel_ms": {
            key: value / profile_substeps for key, value in sorted(kernel_totals.items())
        },
        "active_rigid_proxies": solver.rigid_proxy_active_count,
        "possible_all_pairs": solver.rigid_proxy_all_pair_count,
        "last_bvh_candidates": (
            int(solver.rigid_proxy_bvh_candidate_counter.numpy()[0])
            if broadphase == "bvh" else None
        ),
        "last_bvh_contacts": (
            int(solver.rigid_proxy_bvh_contact_counter.numpy()[0])
            if broadphase == "bvh" else None
        ),
    }
    trajectory = {
        "center": solver.body_center.numpy()[active],
        "linear_velocity": solver.body_linear_velocity.numpy()[active],
        "orientation": solver.body_orientation.numpy()[active],
        "state": state,
    }
    return result, trajectory


def compare_trajectories(
    all_pairs: dict[str, np.ndarray], bvh: dict[str, np.ndarray]
) -> dict[str, float | int]:
    if np.any(all_pairs["state"] != bvh["state"]):
        state_changes = int(np.count_nonzero(all_pairs["state"] != bvh["state"]))
    else:
        state_changes = 0
    result: dict[str, float | int] = {"rigid_state_changes": state_changes}
    for key in ("center", "linear_velocity", "orientation"):
        delta = np.linalg.norm(bvh[key] - all_pairs[key], axis=1)
        result[f"{key}_rms"] = float(np.sqrt(np.mean(delta * delta)))
        result[f"{key}_max"] = float(np.max(delta))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_54_rigid_bvh_ab_20260803",
    )
    parser.add_argument("--profile-substeps", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()
    legacy, legacy_trajectory = run_variant(
        "all_pairs", "all_pairs", cfg, args.checkpoint, args.output,
        args.profile_substeps,
    )
    bvh, bvh_trajectory = run_variant(
        "gpu_bvh", "bvh", cfg, args.checkpoint, args.output,
        args.profile_substeps,
    )
    comparison = compare_trajectories(legacy_trajectory, bvh_trajectory)
    comparison["wall_speedup"] = (
        legacy["average_wall_ms_per_substep"] / bvh["average_wall_ms_per_substep"]
    )
    comparison["gpu_kernel_speedup"] = (
        legacy["average_gpu_kernel_ms_per_substep"]
        / bvh["average_gpu_kernel_ms_per_substep"]
    )
    report = {"all_pairs": legacy, "gpu_bvh": bvh, "comparison": comparison}
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
