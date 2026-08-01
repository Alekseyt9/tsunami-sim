"""Validate a completed DELUGE V3 production directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def probe_video(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_name,width,height,nb_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--maximum-volume-drift", type=float, default=0.02)
    args = parser.parse_args()

    output = args.output.resolve()
    metrics = load_json_lines(output / "frame_metrics.jsonl")
    if not metrics:
        raise AssertionError("frame_metrics.jsonl is empty")
    expected = args.expected_frames or len(metrics)
    if len(metrics) != expected:
        raise AssertionError(f"expected {expected} metric rows, found {len(metrics)}")
    if [row["frame"] for row in metrics] != list(range(expected)):
        raise AssertionError("frame metrics are missing or out of order")

    video = output / "deluge_v3.mp4"
    if not video.exists() or video.stat().st_size <= 0:
        raise AssertionError("deluge_v3.mp4 is missing or empty")
    probe = probe_video(video)
    stream = probe["streams"][0]
    video_frames = int(stream["nb_frames"])
    if video_frames != expected:
        raise AssertionError(f"video has {video_frames} frames, expected {expected}")

    first = metrics[0]
    last = metrics[-1]
    initial_volume = first["fluid_volume_m3"] + first["shallow_water_volume_m3"]
    final_volume = last["fluid_volume_m3"] + last["shallow_water_volume_m3"]
    volume_drift = final_volume / initial_volume - 1.0
    if abs(volume_drift) > args.maximum_volume_drift:
        raise AssertionError(f"combined 2D/3D water drift is {volume_drift:.3%}")

    late_start = min(85, max(0, expected - 15))
    late = metrics[late_start:]
    late_vertices = [row["water_mesh_vertices"] for row in late]
    if min(late_vertices) < 5_000:
        raise AssertionError(f"late water mesh collapsed to {min(late_vertices):,} vertices")
    voxels = {row["water_mesh_voxel_millimeters"] for row in metrics}
    if len(voxels) != 1:
        raise AssertionError(f"water voxel LOD changed during the run: {sorted(voxels)}")
    if max(row["water_mesh_lod_changes"] for row in metrics) != 0:
        raise AssertionError("water mesh reported an unexpected LOD transition")

    summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
    if int(summary["output_frames"]) != expected:
        raise AssertionError("benchmark summary frame count does not match metrics")
    print(
        f"PASS: {expected} frames / {float(probe['format']['duration']):.4f} s; "
        f"video={video.stat().st_size / 1024**2:.2f} MiB; "
        f"combined water drift={volume_drift:.3%}; "
        f"late mesh={min(late_vertices):,}-{max(late_vertices):,} vertices; "
        f"voxel={next(iter(voxels)) / 1000.0:.2f} m"
    )


if __name__ == "__main__":
    main()
