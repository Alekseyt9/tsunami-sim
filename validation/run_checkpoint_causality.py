"""Advance a production checkpoint without rasterizing intermediate frames.

This keeps the production output cadence, refinement cadence, substeps and
support-graph updates intact.  It is intended for causal A/B checks where the
expensive four-view raster is useful only at the final state.
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver


def building_snapshot(solver: HybridDelugeSolver) -> list[dict]:
    building = solver.arrays["building_id"][:solver.count].numpy()
    kind = solver.arrays["kind"][:solver.count].numpy()
    damage = solver.arrays["damage"][:solver.count].numpy()
    impulse = solver.arrays["material_impact_impulse"][:solver.count].numpy()
    fragment = solver.fragment_id[:solver.count].numpy()
    active = solver.building_active.numpy()
    exposure = solver.building_activation_exposure.numpy()
    debris_exposure = solver.building_debris_activation_exposure.numpy()
    debris_volume = solver.building_debris_impacted_volume.numpy()
    structural_volume = solver.building_structural_volume.numpy()
    debris_fraction = np.divide(debris_volume, np.maximum(structural_volume, 1.0e-6))
    debris_peak = solver.building_debris_peak_acceleration.numpy()
    loaded_volume = solver.activation_loaded_volume.numpy()
    eligible_volume = solver.building_activation_base_volume.numpy()
    loaded_fraction = np.divide(
        loaded_volume,
        np.maximum(eligible_volume, 1.0e-6),
    )
    rows = []
    for bid in range(solver.building_count):
        mask = (building == bid) & (kind != 0)
        fragment_ids = np.unique(fragment[mask & (fragment >= 0)])
        support_lost = (
            float(np.mean(solver.fragment_support_host[fragment_ids] < 0.5))
            if len(fragment_ids) else 0.0
        )
        rows.append({
            "building": bid,
            "active": bool(active[bid]),
            "activation_exposure_seconds": float(exposure[bid]),
            "debris_activation_exposure_seconds": float(debris_exposure[bid]),
            "debris_impacted_volume_fraction": float(debris_fraction[bid]),
            "debris_peak_acceleration_m_s2": float(debris_peak[bid]),
            "loaded_base_fraction": float(loaded_fraction[bid]),
            "damaged_above_002_fraction": float(np.mean(damage[mask] > 0.02)),
            "mean_damage": float(np.mean(damage[mask])),
            "maximum_impact_impulse": float(np.max(impulse[mask])),
            "support_lost_fraction": support_lost,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config_used.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    wp.init()
    solver = HybridDelugeSolver(cfg, args.output, args.resume)
    fps = float(cfg["output_fps"])
    substeps = int(math.ceil((1.0 / fps) / float(cfg["dt"])))
    dt = (1.0 / fps) / substeps
    metrics_path = args.output / "headless_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for frame in range(solver.start_frame, args.frames):
        started = time.perf_counter()
        for _ in range(substeps):
            solver.substep(dt)
        if frame > 0 and frame % int(cfg.get("refine_every_frames", 8)) == 0:
            solver.refine()
        stats = solver.stats()
        wp.synchronize_device(solver.device)
        loaded_fraction = np.divide(
            solver.activation_loaded_volume.numpy(),
            np.maximum(solver.building_activation_base_volume.numpy(), 1.0e-6),
        )
        row = {
            "frame": frame,
            "sim_time_seconds": solver.time,
            "particles": solver.count,
            "damaged_particles": int(stats["damaged"]),
            "active_buildings": int(stats.get("active_buildings", 0)),
            "released_fragments": int(stats.get("released_fragments", 0)),
            "unsupported_fragments": int(stats.get("unsupported_fragments", 0)),
            "loaded_base_fraction": loaded_fraction.tolist(),
            "loaded_base_fraction_max": float(np.max(loaded_fraction)),
            "wall_seconds": time.perf_counter() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[{frame + 1:05d}/{args.frames:05d}] t={solver.time:8.4f}s "
            f"particles={solver.count:,} damaged={stats['damaged']:,} "
            f"active={row['active_buildings']} wall={row['wall_seconds']:.2f}s",
            flush=True,
        )
        checkpoint_every = int(cfg.get("checkpoint_every_frames", 0))
        if checkpoint_every and frame > 0 and frame % checkpoint_every == 0:
            solver.save_checkpoint(frame)

    final_frame = args.frames - 1
    solver.save_checkpoint(final_frame)
    summary = {
        "frame": final_frame,
        "sim_time_seconds": solver.time,
        "particles": solver.count,
        "buildings": building_snapshot(solver),
    }
    (args.output / "causality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
