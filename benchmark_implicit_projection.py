"""Checkpoint benchmark for the experimental unequal-mass DFSPH projection."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
import warp as wp

from deluge_v3 import HERE, HybridDelugeSolver


DEFAULT_RUN = HERE / "outputs" / "v3_106_production_15s_24fps_20260803"


def run_variant(
    cfg: dict,
    checkpoint: Path,
    output: Path,
    dt: float,
    steps: int,
    implicit: bool,
    pressure_iterations: int,
    divergence_projection: bool,
    selective_compression: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    variant = copy.deepcopy(cfg)
    policy = variant["v3"]["implicit_fluid"]
    policy["enabled"] = implicit
    policy["mode"] = "density_projection" if implicit else "diagnostic"
    if implicit:
        policy["minimum_pressure_iterations"] = pressure_iterations
        policy["maximum_pressure_iterations"] = pressure_iterations
        policy["divergence_projection"] = divergence_projection
        policy.setdefault("selective_compression", {})["enabled"] = (
            selective_compression
        )
    variant["v3"]["narrow_band_volume"]["enabled"] = False
    variant["v3"]["rigid_clusters"]["early_rigidification"]["enabled"] = False
    warmup = HybridDelugeSolver(variant, output / "_warmup", checkpoint)
    # Compile and build transient neighbour state on a disposable instance.
    # The measured solver must start from exactly the checkpoint; advancing
    # each variant by a different warm-up dt invalidated older A/B deltas.
    warmup.substep(dt)
    wp.synchronize_device(warmup.device)
    del warmup
    gc.collect()
    solver = HybridDelugeSolver(variant, output, checkpoint)
    wall_ms: list[float] = []
    for _ in range(steps):
        started = time.perf_counter()
        solver.substep(dt)
        wp.synchronize_device(solver.device)
        wall_ms.append((time.perf_counter() - started) * 1000.0)

    kind = solver.arrays["kind"][:solver.count].numpy()
    fluid = kind == 0
    position = solver.arrays["x"][:solver.count].numpy()
    velocity = solver.arrays["v"][:solver.count].numpy()
    density_ratio = (
        solver.arrays["rho"][:solver.count].numpy()[fluid]
        / float(variant["rest_density"])
    )
    diagnostics = solver.implicit_fluid_preparation.execution_diagnostics()
    median_ms = float(statistics.median(wall_ms))
    report = {
        "implicit": implicit,
        "divergence_projection": bool(implicit and divergence_projection),
        "selective_compression": bool(implicit and selective_compression),
        "dt_s": dt,
        "steps": steps,
        "simulated_duration_s": dt * steps,
        "wall_substep_mean_ms": float(statistics.fmean(wall_ms)),
        "wall_substep_median_ms": median_ms,
        "wall_seconds_per_simulated_second": median_ms / (1000.0 * dt),
        "finite_positions": bool(np.isfinite(position).all()),
        "finite_velocities": bool(np.isfinite(velocity).all()),
        "fluid_speed_max_m_s": float(np.max(np.linalg.norm(velocity[fluid], axis=1))),
        "density_ratio_percentiles": {
            "p50": float(np.percentile(density_ratio, 50.0)),
            "p95": float(np.percentile(density_ratio, 95.0)),
            "p99": float(np.percentile(density_ratio, 99.0)),
            "max": float(np.max(density_ratio)),
        },
        **diagnostics,
    }
    return report, {
        "position": position,
        "velocity": velocity,
        "damage": solver.arrays["damage"][:solver.count].numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_110_implicit_projection_ab_20260803",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--implicit-dt", type=float, default=0.0006)
    parser.add_argument("--pressure-iterations", type=int, default=4)
    parser.add_argument(
        "--disable-divergence", action="store_true",
        help="Benchmark the density projection without divergence correction.",
    )
    parser.add_argument(
        "--disable-selective", action="store_true",
        help="Run the projection over every fluid particle.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()

    report: dict[str, object] = {"checkpoints": {}}
    for label, frame in (("early", 24), ("late", 336)):
        checkpoint = args.run / "checkpoints" / f"state_{frame:05d}.npz"
        variants = {}
        states: dict[str, dict[str, np.ndarray]] = {}
        target_duration = max(1, args.steps) * args.implicit_dt
        for name, enabled, dt in (
            ("baseline", False, float(cfg["dt"])),
            ("unequal_mass_dfsph", True, args.implicit_dt),
        ):
            variant_steps = max(1, int(round(target_duration / dt)))
            variants[name], states[name] = run_variant(
                cfg, checkpoint, args.output / label / name,
                dt, variant_steps, enabled,
                max(1, args.pressure_iterations),
                not args.disable_divergence,
                not args.disable_selective,
            )
        baseline = variants["baseline"]
        projected = variants["unequal_mass_dfsph"]
        variants["comparison"] = {
            "wall_per_simulated_second_speedup": (
                baseline["wall_seconds_per_simulated_second"]
                / max(projected["wall_seconds_per_simulated_second"], 1.0e-12)
            ),
            "dt_multiplier": args.implicit_dt / float(cfg["dt"]),
            "equal_simulated_duration_s": target_duration,
        }
        position_delta = np.linalg.norm(
            states["unequal_mass_dfsph"]["position"]
            - states["baseline"]["position"], axis=1,
        )
        velocity_delta = np.linalg.norm(
            states["unequal_mass_dfsph"]["velocity"]
            - states["baseline"]["velocity"], axis=1,
        )
        damage_delta = np.abs(
            states["unequal_mass_dfsph"]["damage"]
            - states["baseline"]["damage"]
        )
        variants["comparison"].update({
            "position_delta_rms_m": float(np.sqrt(np.mean(position_delta ** 2))),
            "position_delta_p95_m": float(np.percentile(position_delta, 95.0)),
            "velocity_delta_rms_m_s": float(np.sqrt(np.mean(velocity_delta ** 2))),
            "velocity_delta_p95_m_s": float(np.percentile(velocity_delta, 95.0)),
            "damage_delta_mean": float(np.mean(damage_delta)),
            "damage_delta_max": float(np.max(damage_delta)),
        })
        report["checkpoints"][label] = {
            "checkpoint": str(checkpoint.resolve()),
            **variants,
        }

    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
