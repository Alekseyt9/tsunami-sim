"""A/B benchmark early deformable-to-rigid conversion on one checkpoint."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver


DEFAULT_CHECKPOINT = (
    HERE / "outputs" / "v3_106_production_15s_24fps_20260803"
    / "checkpoints" / "state_00336.npz"
)


def prepare_variant(
    cfg: dict, checkpoint: Path, output: Path, enabled: bool
) -> tuple[HybridDelugeSolver, dict]:
    variant = copy.deepcopy(cfg)
    variant["v3"]["implicit_fluid"]["enabled"] = False
    variant["v3"]["narrow_band_volume"]["enabled"] = False
    early = variant["v3"]["rigid_clusters"]["early_rigidification"]
    early["enabled"] = enabled
    early["scan_every_frames"] = 1
    solver = HybridDelugeSolver(variant, output, checkpoint)
    state_before = solver.rigid_state.numpy()
    rigid_before = int(np.count_nonzero(state_before))
    rigid_particles_before = int(np.count_nonzero(
        (solver.fragment_host >= 0)
        & (state_before[np.maximum(solver.fragment_host, 0)] != 0)
    ))
    if enabled:
        required = max(2, int(early.get("minimum_detached_scans", 2)))
        for _ in range(required):
            solver.rigid_stats_calls = 0
            solver.update_rigid_clusters()
    state_after = solver.rigid_state.numpy()
    rigid_after = int(np.count_nonzero(state_after))
    rigid_particles_after = int(np.count_nonzero(
        (solver.fragment_host >= 0)
        & (state_after[np.maximum(solver.fragment_host, 0)] != 0)
    ))
    return solver, {
        "enabled": enabled,
        "rigid_clusters_before": rigid_before,
        "rigid_clusters_after": rigid_after,
        "converted_clusters": rigid_after - rigid_before,
        "rigid_particles_before": rigid_particles_before,
        "rigid_particles_after": rigid_particles_after,
        "converted_particles": rigid_particles_after - rigid_particles_before,
    }


def profile(solver: HybridDelugeSolver, dt: float, substeps: int) -> dict:
    solver.substep(dt)
    wp.synchronize_device(solver.device)
    wall_ms: list[float] = []
    kernel_totals: dict[str, float] = {}
    for _ in range(substeps):
        started = time.perf_counter()
        with wp.ScopedTimer(
            "early_rigid_substep", print=False, synchronize=True,
            cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN,
        ) as timer:
            solver.substep(dt)
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        for timing in timer.timing_results:
            kernel_totals[timing.name] = (
                kernel_totals.get(timing.name, 0.0) + timing.elapsed
            )
    average = {
        key: value / substeps for key, value in sorted(kernel_totals.items())
    }
    structural_tokens = (
        "compute_clustered_solid_forces", "deformable_contacts",
        "collect_deformable_contact", "finalize_deformable_acceleration",
    )
    rigid_tokens = (
        "rigid_body", "rigid_contact", "rigid_proxy", "scatter_rigid",
    )
    positions = solver.arrays["x"][:solver.count].numpy()
    velocities = solver.arrays["v"][:solver.count].numpy()
    return {
        "profile_substeps": substeps,
        "wall_substep_mean_ms": float(statistics.fmean(wall_ms)),
        "wall_substep_median_ms": float(statistics.median(wall_ms)),
        "profile_structural_ms": float(sum(
            value for key, value in average.items()
            if any(token in key for token in structural_tokens)
        )),
        "profile_rigid_ms": float(sum(
            value for key, value in average.items()
            if any(token in key for token in rigid_tokens)
        )),
        "profile_average_kernel_ms": average,
        "finite_positions": bool(np.isfinite(positions).all()),
        "finite_velocities": bool(np.isfinite(velocities).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_109_early_rigid_ab_20260803",
    )
    parser.add_argument("--profile-substeps", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()
    reports: dict[str, dict] = {}
    for name, enabled in (("baseline", False), ("early_rigid", True)):
        solver, transition = prepare_variant(
            cfg, args.checkpoint, args.output / name, enabled
        )
        reports[name] = {
            **transition,
            **profile(solver, float(cfg["dt"]), args.profile_substeps),
        }
    baseline = reports["baseline"]
    optimized = reports["early_rigid"]
    comparison = {
        "wall_substep_speedup": (
            baseline["wall_substep_median_ms"]
            / optimized["wall_substep_median_ms"]
        ),
        "structural_kernel_speedup": (
            baseline["profile_structural_ms"]
            / max(optimized["profile_structural_ms"], 1.0e-9)
        ),
        "structural_plus_rigid_speedup": (
            (baseline["profile_structural_ms"] + baseline["profile_rigid_ms"])
            / max(
                optimized["profile_structural_ms"]
                + optimized["profile_rigid_ms"],
                1.0e-9,
            )
        ),
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "baseline": baseline,
        "early_rigid": optimized,
        "comparison": comparison,
    }
    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
