"""Equal one-second WCSPH/DFSPH checkpoint audit without video rendering."""

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

from deluge_v3 import HERE, HybridDelugeSolver


DEFAULT_RUN = HERE / "outputs" / "v3_106_production_15s_24fps_20260803"
STRUCTURAL_FLOAT_METRICS = (
    "damage_integral_slab_m3",
    "damage_integral_wall_m3",
    "damage_integral_beam_m3",
    "damage_integral_column_m3",
    "damage_integral_core_m3",
    "damage_integral_glass_m3",
    "damaged_slab_volume_m3",
    "damaged_wall_volume_m3",
    "damaged_beam_volume_m3",
    "damaged_column_volume_m3",
    "damaged_core_volume_m3",
    "damaged_glass_volume_m3",
)


def configure(cfg: dict, implicit: bool) -> dict:
    variant = copy.deepcopy(cfg)
    policy = variant["v3"]["implicit_fluid"]
    policy["enabled"] = implicit
    policy["mode"] = "density_projection" if implicit else "diagnostic"
    variant["v3"]["narrow_band_volume"]["enabled"] = False
    variant["v3"]["rigid_clusters"]["early_rigidification"]["enabled"] = False
    return variant


def combined_water(stats: dict) -> float:
    return float(stats.get("fluid_volume_m3", 0.0)) + float(
        stats.get("shallow_water_volume_m3", 0.0)
    )


def structural_snapshot(stats: dict) -> dict:
    result = {
        "active_buildings": int(stats.get("active_buildings", 0)),
        "released_fragments": int(stats.get("released_fragments", 0)),
        "unsupported_fragments": int(stats.get("unsupported_fragments", 0)),
        "rigid_clusters": int(stats.get("rigid_clusters", 0)),
        "combined_water_volume_m3": combined_water(stats),
        "wave_train_injected_volume_m3": float(
            stats.get("wave_train_injected_volume_m3", 0.0)
        ),
        "fluid_momentum_z_kg_m_s": float(
            stats.get("fluid_momentum_z_kg_m_s", 0.0)
        ),
        "fluid_height_p99_m": float(stats.get("fluid_height_p99_m", 0.0)),
        "fluid_height_p999_m": float(stats.get("fluid_height_p999_m", 0.0)),
        "fluid_height_max_m": float(stats.get("fluid_height_max_m", 0.0)),
    }
    for key in STRUCTURAL_FLOAT_METRICS:
        result[key] = float(stats.get(key, 0.0))
    result["damage_integral_total_m3"] = sum(
        result[key] for key in STRUCTURAL_FLOAT_METRICS
        if key.startswith("damage_integral_")
    )
    return result


def run_variant(
    cfg: dict,
    checkpoint: Path,
    output: Path,
    implicit: bool,
    duration: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    variant = configure(cfg, implicit)
    dt_limit = (
        float(variant["v3"]["implicit_fluid"]["maximum_dt"])
        if implicit else float(variant["dt"])
    )
    fps = int(variant["output_fps"])
    frames = max(1, int(round(duration * fps)))
    frame_dt = duration / frames
    substeps_per_frame = int(math.ceil(frame_dt / dt_limit))
    dt = frame_dt / substeps_per_frame

    # Compile every path on a disposable solver. The measured solver starts
    # from the checkpoint again, avoiding the old unequal warm-up time bug.
    warmup = HybridDelugeSolver(variant, output / "_warmup", checkpoint)
    warmup.substep(dt)
    wp.synchronize_device(warmup.device)
    warmup.stats()
    wp.synchronize_device(warmup.device)
    del warmup
    gc.collect()

    solver = HybridDelugeSolver(variant, output, checkpoint)
    initial_stats = solver.stats()
    wp.synchronize_device(solver.device)
    initial = structural_snapshot(initial_stats)
    frame_rows: list[dict] = []
    physics_ms: list[float] = []
    for local_frame in range(frames):
        frame_started = time.perf_counter()
        physics_started = time.perf_counter()
        for _ in range(substeps_per_frame):
            solver.substep(dt)
        wp.synchronize_device(solver.device)
        frame_physics_ms = (time.perf_counter() - physics_started) * 1000.0
        absolute_frame = solver.start_frame + local_frame
        if (
            bool(variant.get("adaptive_refinement", True))
            and absolute_frame > 0
            and absolute_frame % int(variant.get("refine_every_frames", 8)) == 0
        ):
            solver.refine()
        stats = solver.stats()
        wp.synchronize_device(solver.device)
        frame_total_ms = (time.perf_counter() - frame_started) * 1000.0
        physics_ms.append(frame_physics_ms)
        frame_rows.append({
            "frame": local_frame,
            "simulated_time_s": (local_frame + 1) * frame_dt,
            "particle_count": solver.count,
            "physics_ms": frame_physics_ms,
            "full_no_render_frame_ms": frame_total_ms,
            **structural_snapshot(stats),
        })

    count = solver.count
    kind = solver.arrays["kind"][:count].numpy()
    solid = kind != 0
    position = solver.arrays["x"][:count].numpy()[solid]
    velocity = solver.arrays["v"][:count].numpy()[solid]
    damage = solver.arrays["damage"][:count].numpy()[solid]
    rest_position = solver.arrays["rest_x"][:count].numpy()[solid]
    building = solver.arrays["building_id"][:count].numpy()[solid]
    structural_class = solver.arrays["structural_class"][:count].numpy()[solid]
    final = structural_snapshot(stats)
    injected = (
        final["wave_train_injected_volume_m3"]
        - initial["wave_train_injected_volume_m3"]
    )
    expected_final_water = initial["combined_water_volume_m3"] + injected
    report = {
        "implicit": implicit,
        "duration_s": duration,
        "frames": frames,
        "substeps_per_frame": substeps_per_frame,
        "dt_s": dt,
        "total_substeps": frames * substeps_per_frame,
        "physics_ms_per_substep_mean": float(
            statistics.fmean(physics_ms) / substeps_per_frame
        ),
        "full_no_render_frame_ms_mean": float(statistics.fmean(
            row["full_no_render_frame_ms"] for row in frame_rows
        )),
        "full_no_render_frame_ms_median": float(statistics.median(
            row["full_no_render_frame_ms"] for row in frame_rows
        )),
        "finite_positions": bool(np.isfinite(position).all()),
        "finite_velocities": bool(np.isfinite(velocity).all()),
        "initial": initial,
        "final": final,
        "injected_water_volume_m3": injected,
        "combined_water_volume_drift_fraction": (
            final["combined_water_volume_m3"] / expected_final_water - 1.0
            if expected_final_water > 0.0 else 0.0
        ),
        "frame_rows": frame_rows,
        **solver.implicit_fluid_preparation.execution_diagnostics(),
    }
    return report, {
        "rest_position": rest_position,
        "building": building,
        "structural_class": structural_class,
        "position": position,
        "velocity": velocity,
        "damage": damage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint-frame", type=int, default=336)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs" / "v3_111_divergence_one_second_ab_20260804",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = (
        args.run / "checkpoints" / f"state_{args.checkpoint_frame:05d}.npz"
    )
    wp.init()
    report: dict[str, object] = {
        "checkpoint": str(checkpoint.resolve()),
        "duration_s": args.duration,
        "variants": {},
    }
    states: dict[str, dict[str, np.ndarray]] = {}
    for name, implicit in (("wcsph", False), ("dfsph_selective", True)):
        variant_report, states[name] = run_variant(
            cfg, checkpoint, args.output / name, implicit, args.duration
        )
        report["variants"][name] = variant_report
        (args.output / "comparison.partial.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    key_dtype = np.dtype([
        ("building", np.int32), ("class", np.int32),
        ("x_mm", np.int32), ("y_mm", np.int32), ("z_mm", np.int32),
    ])
    keys: dict[str, np.ndarray] = {}
    for name in ("wcsph", "dfsph_selective"):
        state = states[name]
        key = np.empty(len(state["position"]), dtype=key_dtype)
        key["building"] = state["building"]
        key["class"] = state["structural_class"]
        rest_mm = np.rint(state["rest_position"] * 1000.0).astype(np.int32)
        key["x_mm"], key["y_mm"], key["z_mm"] = rest_mm.T
        keys[name] = key
    _, baseline_index, projected_index = np.intersect1d(
        keys["wcsph"], keys["dfsph_selective"], return_indices=True
    )
    position_delta = np.linalg.norm(
        states["dfsph_selective"]["position"][projected_index]
        - states["wcsph"]["position"][baseline_index], axis=1,
    )
    velocity_delta = np.linalg.norm(
        states["dfsph_selective"]["velocity"][projected_index]
        - states["wcsph"]["velocity"][baseline_index], axis=1,
    )
    damage_delta = np.abs(
        states["dfsph_selective"]["damage"][projected_index]
        - states["wcsph"]["damage"][baseline_index]
    )
    baseline = report["variants"]["wcsph"]
    projected = report["variants"]["dfsph_selective"]
    report["comparison"] = {
        "physics_speedup": (
            baseline["physics_ms_per_substep_mean"] * baseline["total_substeps"]
            / max(
                projected["physics_ms_per_substep_mean"]
                * projected["total_substeps"],
                1.0e-12,
            )
        ),
        "full_no_render_frame_speedup": (
            baseline["full_no_render_frame_ms_mean"]
            / max(projected["full_no_render_frame_ms_mean"], 1.0e-12)
        ),
        "matched_structural_particles": int(len(baseline_index)),
        "position_delta_rms_m": float(np.sqrt(np.mean(position_delta ** 2))),
        "position_delta_p95_m": float(np.percentile(position_delta, 95.0)),
        "velocity_delta_rms_m_s": float(np.sqrt(np.mean(velocity_delta ** 2))),
        "velocity_delta_p95_m_s": float(np.percentile(velocity_delta, 95.0)),
        "damage_delta_mean": float(np.mean(damage_delta)),
        "damage_delta_max": float(np.max(damage_delta)),
        "active_buildings_delta": (
            projected["final"]["active_buildings"]
            - baseline["final"]["active_buildings"]
        ),
        "released_fragments_delta": (
            projected["final"]["released_fragments"]
            - baseline["final"]["released_fragments"]
        ),
        "damage_integral_total_delta_m3": (
            projected["final"]["damage_integral_total_m3"]
            - baseline["final"]["damage_integral_total_m3"]
        ),
    }
    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
