"""A/B benchmark legacy hash-grid structural bonds against persistent GPU CSR."""

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
    name: str, enabled: bool, cfg: dict, checkpoint: Path,
    output: Path, profile_substeps: int,
    compact_contacts: bool | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    variant = copy.deepcopy(cfg)
    variant["v3"].setdefault("structural_adjacency", {})["enabled"] = enabled
    if compact_contacts is not None:
        variant["v3"]["structural_adjacency"][
            "compact_contact_candidates"
        ] = compact_contacts
    variant["v3"]["rigid_clusters"]["collision_proxy"]["broadphase"] = "all_pairs"
    variant["v3"]["water_mesh"]["enabled"] = False
    solver = HybridDelugeSolver(variant, output / name, checkpoint)
    substeps_per_frame = int(math.ceil(
        (1.0 / float(variant["output_fps"])) / float(variant["dt"])
    ))
    dt = (1.0 / float(variant["output_fps"])) / substeps_per_frame
    solver.substep(dt)
    wp.synchronize_device(solver.device)
    totals: dict[str, float] = defaultdict(float)
    started = time.perf_counter()
    for _ in range(profile_substeps):
        with wp.ScopedTimer(
            name, print=False, synchronize=True,
            cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN,
        ) as timer:
            solver.substep(dt)
        for timing in timer.timing_results:
            totals[timing.name] += timing.elapsed
    wp.synchronize_device(solver.device)
    wall_ms = (time.perf_counter() - started) * 1000.0 / profile_substeps
    average = {key: value / profile_substeps for key, value in sorted(totals.items())}
    kind = solver.arrays["kind"][:solver.count].numpy()
    solid = kind != 0
    result = {
        "name": name,
        "adjacency_enabled": enabled,
        "compact_contact_candidates": solver.compact_contact_candidates_enabled,
        "particles": solver.count,
        "solid_particles": int(np.count_nonzero(solid)),
        "adjacency_entries": solver.structural_adjacency_entries,
        "adjacency_rebuild_ms": solver.structural_adjacency_last_rebuild_ms,
        "last_contact_candidates": int(
            solver.deformable_contact_candidate_count.numpy()[0]
        ) if solver.compact_contact_candidates_enabled else solver.count,
        "average_wall_ms_per_substep": wall_ms,
        "average_gpu_kernel_ms_per_substep": float(sum(average.values())),
        "average_kernel_ms": average,
    }
    trajectory = {
        "position": solver.arrays["x"][:solver.count].numpy()[solid],
        "velocity": solver.arrays["v"][:solver.count].numpy()[solid],
        "damage": solver.arrays["damage"][:solver.count].numpy()[solid],
        "rigid_state": solver.rigid_state.numpy(),
    }
    return result, trajectory


def compare(legacy: dict[str, np.ndarray], csr: dict[str, np.ndarray]) -> dict:
    result: dict[str, float | int] = {}
    for key in ("position", "velocity"):
        magnitude = np.linalg.norm(csr[key] - legacy[key], axis=1)
        result[f"{key}_rms"] = float(np.sqrt(np.mean(magnitude * magnitude)))
        result[f"{key}_max"] = float(np.max(magnitude))
    damage = np.abs(csr["damage"] - legacy["damage"])
    result["damage_rms"] = float(np.sqrt(np.mean(damage * damage)))
    result["damage_max"] = float(np.max(damage))
    result["rigid_state_changes"] = int(np.count_nonzero(
        csr["rigid_state"] != legacy["rigid_state"]
    ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_60_structural_adjacency_ab_20260803",
    )
    parser.add_argument("--profile-substeps", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()
    legacy, legacy_trajectory = run_variant(
        "legacy_hash_grid", False, cfg, args.checkpoint, args.output,
        args.profile_substeps,
    )
    csr, csr_trajectory = run_variant(
        "gpu_csr", True, cfg, args.checkpoint, args.output,
        args.profile_substeps,
    )
    comparison = compare(legacy_trajectory, csr_trajectory)
    comparison["wall_speedup"] = (
        legacy["average_wall_ms_per_substep"] / csr["average_wall_ms_per_substep"]
    )
    comparison["gpu_kernel_speedup"] = (
        legacy["average_gpu_kernel_ms_per_substep"]
        / csr["average_gpu_kernel_ms_per_substep"]
    )
    report = {"legacy": legacy, "gpu_csr": csr, "comparison": comparison}
    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
