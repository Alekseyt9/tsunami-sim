"""Assemble a frame-zero prefix and a checkpoint-resumed DELUGE run."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def video_for(run: Path, start_frame: int, basename: str) -> Path:
    segment = f"_segment_{start_frame:05d}" if start_frame else ""
    return run / f"{basename}{segment}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("continuation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    prefix = args.prefix.resolve()
    continuation = args.continuation.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    prefix_rows = load_rows(prefix / "frame_metrics.jsonl")
    continuation_rows = load_rows(continuation / "frame_metrics.jsonl")
    if not prefix_rows or not continuation_rows:
        raise ValueError("both runs must contain frame metrics")
    split_frame = int(continuation_rows[0]["frame"])
    if [int(row["frame"]) for row in prefix_rows] != list(range(split_frame)):
        raise ValueError(f"prefix must contain exactly frames 0-{split_frame - 1}")
    if [int(row["frame"]) for row in continuation_rows] != list(
        range(split_frame, split_frame + len(continuation_rows))
    ):
        raise ValueError("continuation frame sequence is incomplete")

    config = json.loads((continuation / "config_used.json").read_text(encoding="utf-8"))
    basename = str(config.get("output_basename", "deluge_v3"))
    prefix_video = video_for(prefix, 0, basename)
    continuation_video = video_for(continuation, split_frame, basename)
    if not prefix_video.exists() or not continuation_video.exists():
        raise FileNotFoundError("a source MP4 is missing")

    concat_list = output / "concat_sources.txt"
    concat_list.write_text(
        "\n".join(
            f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
            for path in (prefix_video, continuation_video)
        ) + "\n",
        encoding="utf-8",
    )
    final_video = output / f"{basename}.mp4"
    temporary_video = output / f".{basename}.assembling.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", "-movflags", "+faststart",
            str(temporary_video),
        ],
        check=True,
    )
    temporary_video.replace(final_video)

    rows = prefix_rows + continuation_rows
    (output / "frame_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    shutil.copy2(continuation / "config_used.json", output / "config_used.json")

    wall = [float(row["wall_seconds"]) for row in rows]
    combined_summary = {
        "device": "NVIDIA GeForce RTX 5070",
        "resolution": [int(config["render"]["width"]), int(config["render"]["height"])],
        "views": list(config["render"].get("views", ["original", "front", "side", "top"])),
        "view_count": 4,
        "view_layout": config["render"].get("view_layout", "quad"),
        "output_fps": int(config["output_fps"]),
        "output_frames": len(rows),
        "simulated_seconds": float(rows[-1]["sim_time_seconds"]),
        "initial_particles": int(rows[0]["particles"]),
        "peak_particles": max(int(row["particles"]) for row in rows),
        "peak_fluid_particles": max(int(row["fluid_particles"]) for row in rows),
        "peak_damaged_particles": max(int(row["damaged_particles"]) for row in rows),
        "simulation_and_render_wall_seconds": sum(wall),
        "average_output_frame_wall_seconds": statistics.fmean(wall),
        "median_output_frame_wall_seconds": statistics.median(wall),
        "max_output_frame_wall_seconds": max(wall),
        "peak_gpu_memory_used_mib": max(float(row["gpu_memory_used_mib"]) for row in rows),
        "assembly_split_frame": split_frame,
        "prefix_source": str(prefix),
        "continuation_source": str(continuation),
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"assembled {len(prefix_rows)} + {len(continuation_rows)} = {len(rows)} frames; "
        f"video={final_video} ({final_video.stat().st_size / 1024**2:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
