"""Short V3.21-style surface vs V3.22 phase-separated water A/B run."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE / "outputs" / "v3_21_proxy_ab_checkpoint96_20260802" /
    "migrated" / "checkpoints" / "state_00096.npz"
)
CURRENT_CONFIG = HERE / "config_v3_rtx5070.json"


def apply_current_runtime_safety(checkpoint_cfg: dict) -> dict:
    """Use current water/energy policy without changing checkpoint topology."""
    cfg = copy.deepcopy(checkpoint_cfg)
    current = json.loads(CURRENT_CONFIG.read_text(encoding="utf-8"))
    for key in (
        "maximum_fluid_speed", "maximum_fluid_vertical_speed",
        "maximum_solid_speed", "maximum_solid_upward_speed", "fluid_bed_drag",
    ):
        cfg[key] = copy.deepcopy(current[key])
    # Fragment clustering and scene spacing must remain checkpoint-compatible.
    # Surface classification and meshing are topology-independent runtime policy.
    cfg["v3"]["water_surface"] = copy.deepcopy(current["v3"]["water_surface"])
    cfg["v3"]["water_mesh"] = copy.deepcopy(current["v3"]["water_mesh"])
    return cfg


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def capture(solver: HybridDelugeSolver) -> dict[str, np.ndarray]:
    kind = solver.arrays["kind"][:solver.count].numpy()
    return {
        "kind": kind,
        "position": solver.arrays["x"][:solver.count].numpy(),
        "velocity": solver.arrays["v"][:solver.count].numpy(),
        "damage": solver.arrays["damage"][:solver.count].numpy(),
        "mass": solver.arrays["mass"][:solver.count].numpy(),
        "phase": solver.arrays["water_phase"][:solver.count].numpy(),
    }


def run_variant(base_cfg: dict, checkpoint: Path, output: Path, enabled: bool,
                start_frame: int, frame_count: int) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    cfg = copy.deepcopy(base_cfg)
    cfg["duration_seconds"] = float(start_frame + frame_count) / float(cfg["output_fps"])
    cfg["output_basename"] = "water_phase_on" if enabled else "water_phase_off"
    cfg["checkpoint_every_frames"] = 0
    cfg["v3"]["water_surface"]["phase_separation_enabled"] = bool(enabled)
    cfg["render"].update({
        "output_mode": "video",
        "width": 1280,
        "height": 720,
        "view_width": 640,
        "view_height": 360,
        "view_layout": "quad",
        "progressive_fragment_seconds": 0.5,
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    solver = HybridDelugeSolver(cfg, output, checkpoint)
    if solver.start_frame != start_frame:
        raise AssertionError(f"expected resume frame {start_frame}, got {solver.start_frame}")
    solver.run()
    state = capture(solver)
    summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
    rows = load_rows(output / "frame_metrics.jsonl")
    if len(rows) != frame_count:
        raise AssertionError(f"expected {frame_count} rows, got {len(rows)}")
    del solver
    gc.collect()
    return summary, state, rows


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))


def summarize_rows(rows: list[dict]) -> dict:
    first, last = rows[0], rows[-1]
    combined = [
        row.get("fluid_volume_m3", 0.0) + row.get("shallow_water_volume_m3", 0.0)
        for row in rows
    ]
    keys = (
        "connected_surface_particles", "thin_sheet_particles",
        "ballistic_droplet_particles", "foam_particles", "surface_water_particles",
        "water_phase_droplet_entries", "water_phase_droplet_rejoins",
        "water_phase_sheet_entries", "water_phase_sheet_returns",
        "water_phase_droplet_entries_total", "water_phase_droplet_rejoins_total",
        "water_phase_sheet_entries_total", "water_phase_sheet_returns_total",
        "water_mesh_vertices", "water_mesh_triangles", "water_field_nodes",
        "water_mesh_voxel_millimeters", "water_mesh_lod_changes",
        "fluid_height_max_m", "fluid_height_p999_m", "fluid_vertical_speed_max_m_s",
        "water_surface_classify_ms", "water_mesh_total_ms", "wall_seconds",
        "gpu_memory_used_mib", "damaged_particles", "active_buildings",
    )
    result = {
        "first_frame": int(first["frame"]),
        "last_frame": int(last["frame"]),
        "combined_water_volume_initial_m3": float(combined[0]),
        "combined_water_volume_final_m3": float(combined[-1]),
        "combined_water_volume_drift_fraction": float(combined[-1] / combined[0] - 1.0),
    }
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values:
            result[key] = {
                "first": values[0], "last": values[-1],
                "minimum": min(values), "maximum": max(values),
                "average": float(np.mean(values, dtype=np.float64)),
            }
    return result


def compare_states(off: dict[str, np.ndarray], on: dict[str, np.ndarray]) -> dict:
    count = min(len(off["kind"]), len(on["kind"]))
    same_kind = off["kind"][:count] == on["kind"][:count]
    solid = same_kind & (off["kind"][:count] != 0)
    fluid = same_kind & (off["kind"][:count] == 0)
    position_delta = on["position"][:count] - off["position"][:count]
    velocity_delta = on["velocity"][:count] - off["velocity"][:count]
    result = {
        "phase_off_particles": int(len(off["kind"])),
        "phase_on_particles": int(len(on["kind"])),
        "compared_particles": int(count),
        "fluid_position_rms_m": rms(position_delta[fluid]) if np.any(fluid) else 0.0,
        "fluid_position_max_m": float(np.max(np.linalg.norm(position_delta[fluid], axis=1))) if np.any(fluid) else 0.0,
        "fluid_velocity_rms_m_s": rms(velocity_delta[fluid]) if np.any(fluid) else 0.0,
        "solid_position_rms_m": rms(position_delta[solid]) if np.any(solid) else 0.0,
        "solid_position_max_m": float(np.max(np.linalg.norm(position_delta[solid], axis=1))) if np.any(solid) else 0.0,
        "solid_velocity_rms_m_s": rms(velocity_delta[solid]) if np.any(solid) else 0.0,
        "solid_damage_max": float(np.max(np.abs(on["damage"][:count][solid] - off["damage"][:count][solid]))) if np.any(solid) else 0.0,
        "off_total_mass_kg": float(np.sum(off["mass"], dtype=np.float64)),
        "on_total_mass_kg": float(np.sum(on["mass"], dtype=np.float64)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path,
                        default=HERE / "outputs" / "v3_22_water_phase_ab_checkpoint96_12f_20260802")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--only-on", action="store_true",
                        help="Run only the tuned phase-separated branch")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as saved:
        cfg = apply_current_runtime_safety(json.loads(str(saved["config"])))
        start_frame = int(saved["frame"]) + 1
    wp.init()
    off_summary = off_state = off_rows = None
    if not args.only_on:
        off_summary, off_state, off_rows = run_variant(
            cfg, checkpoint, args.output / "phase_off", False, start_frame, args.frames
        )
    on_summary, on_state, on_rows = run_variant(
        cfg, checkpoint, args.output / "phase_on", True, start_frame, args.frames
    )
    report = {
        "checkpoint": str(checkpoint),
        "frames": int(args.frames),
        "runtime_safety_overlay": {
            "maximum_fluid_speed": cfg["maximum_fluid_speed"],
            "maximum_fluid_vertical_speed": cfg["maximum_fluid_vertical_speed"],
            "maximum_solid_speed": cfg["maximum_solid_speed"],
            "maximum_solid_upward_speed": cfg["maximum_solid_upward_speed"],
            "maximum_core_height": cfg["v3"]["water_mesh"].get("maximum_core_height"),
        },
        "phase_on_video": str((args.output / "phase_on" / f"water_phase_on_segment_{start_frame:05d}.mp4").resolve()),
        "phase_on_summary": on_summary,
        "phase_on_metrics": summarize_rows(on_rows),
    }
    if not args.only_on:
        report.update({
            "phase_off_video": str((args.output / "phase_off" / f"water_phase_off_segment_{start_frame:05d}.mp4").resolve()),
            "phase_off_summary": off_summary,
            "phase_off_metrics": summarize_rows(off_rows),
            "trajectory_comparison": compare_states(off_state, on_state),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
