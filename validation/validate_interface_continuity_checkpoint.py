"""Replay a production checkpoint and audit continuous shallow->SPH inflow."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import warp as wp

from deluge_v3 import HybridDelugeSolver, load_run_config  # noqa: E402


HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_city_clear_surge_30s_x1_7.json")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "interface_continuity_check")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(args.config.resolve())
    wp.init()
    solver = HybridDelugeSolver(cfg, args.output, args.checkpoint.resolve())
    fps = int(cfg["output_fps"])
    substeps = int(math.ceil((1.0 / fps) / float(cfg["dt"])))
    dt = (1.0 / fps) / substeps
    frame_count = max(1, int(math.ceil(float(args.seconds) * fps)))
    start_emitted = solver.shallow_water.emitted_particles_total
    start_merged = solver.shallow_water.merged_particles_total
    rows = []
    for local_frame in range(frame_count):
        frame = solver.start_frame + local_frame
        for _ in range(substeps):
            solver.substep(dt)
        if (
            bool(cfg.get("adaptive_refinement", True))
            and frame > 0
            and frame % int(cfg.get("refine_every_frames", 8)) == 0
        ):
            solver.refine()
        stats = solver.stats()
        rows.append({
            "frame": frame,
            "time": float(solver.time),
            "emitted": int(stats["shallow_emitted_particles"]),
            "merged": int(stats["shallow_merged_particles"]),
            "blocked_cells": int(stats["shallow_emission_blocked_cells"]),
            "returning_cells": int(stats["shallow_returning_cells"]),
            "coupling_velocity": float(stats["coupling_sph_velocity_volume_mean_m_s"]),
            "row1_discharge": float(stats["wave_row_1_forward_discharge_m3_s"]),
        })
    emitted_delta = solver.shallow_water.emitted_particles_total - start_emitted
    merged_delta = solver.shallow_water.merged_particles_total - start_merged
    unblocked_frames = sum(row["blocked_cells"] < solver.shallow_water.nx for row in rows)
    result = {
        "start_time": rows[0]["time"] - 1.0 / fps,
        "end_time": rows[-1]["time"],
        "emitted_delta": emitted_delta,
        "merged_delta": merged_delta,
        "unblocked_frames": unblocked_frames,
        "shallow_cells_across": solver.shallow_water.nx,
        "coupling_velocity_peak": max(row["coupling_velocity"] for row in rows),
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if emitted_delta <= 0:
        raise AssertionError("positive sustained discharge produced no SPH particles")
    if unblocked_frames < frame_count // 2:
        raise AssertionError("local returns still block most of the full-width inlet")
    if max(row["row1_discharge"] for row in rows) < 1000.0:
        raise AssertionError("checkpoint does not contain the sustained surge")
    print("PASS: local return cells no longer starve the complete SPH inlet")


if __name__ == "__main__":
    main()
