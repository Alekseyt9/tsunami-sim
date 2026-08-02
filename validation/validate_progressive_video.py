"""Regression: a completed second of fragmented MP4 is readable before close."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


from renderer import StreamingVideoWriter  # noqa: E402


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "stream=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def main() -> None:
    fps = 24
    with tempfile.TemporaryDirectory(prefix="deluge_progressive_", ignore_cleanup_errors=True) as folder:
        output = Path(folder) / "live.mp4"
        writer = StreamingVideoWriter(output, 320, 180, fps, "h264_nvenc", 1.0)
        live_duration = 0.0
        try:
            for frame in range(fps + 1):
                image = np.zeros((180, 320, 3), dtype=np.uint8)
                image[:, :, 0] = (frame * 9) % 255
                image[:, :, 1] = np.arange(320, dtype=np.uint16)[None, :] % 255
                writer.write(image)
            for _ in range(40):
                time.sleep(0.1)
                live_duration = probe_duration(output)
                if live_duration >= 0.99:
                    break
            if not output.exists() or output.stat().st_size <= 0:
                raise AssertionError("public MP4 is still absent or zero-sized after one completed second")
            for frame in range(fps + 1, 50):
                writer.write(np.full((180, 320, 3), frame, dtype=np.uint8))
        finally:
            writer.close()
        final_frames = int(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(output),
                ],
                text=True,
            ).strip()
        )
        if live_duration < 0.99:
            raise AssertionError(f"open MP4 exposed only {live_duration:.3f} seconds")
        if final_frames != 50:
            raise AssertionError(f"final MP4 has {final_frames} frames instead of 50")
        print(
            f"PASS: non-zero open NVENC MP4 exposed {live_duration:.3f}s; "
            f"finalized file has {final_frames} frames"
        )


if __name__ == "__main__":
    main()
