"""Profile DELUGE V3 CUDA kernels at a fresh scene or checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tempfile

import warp as wp

from deluge_v3 import HybridDelugeSolver


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_v3_rtx5070.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    wp.init()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    substeps = int(math.ceil((1.0 / float(cfg["output_fps"])) / float(cfg["dt"])))
    dt = (1.0 / float(cfg["output_fps"])) / substeps
    with tempfile.TemporaryDirectory(prefix="deluge_profile_") as temporary:
        solver = HybridDelugeSolver(cfg, Path(temporary), args.checkpoint)
        # Warm all conditional kernels once before collecting timings.
        solver.substep(dt)
        wp.synchronize_device(solver.device)
        totals: dict[str, float] = defaultdict(float)
        calls: dict[str, int] = defaultdict(int)
        wall_total = 0.0
        for _ in range(args.iterations):
            with wp.ScopedTimer(
                "substep", print=False, synchronize=True,
                cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN,
            ) as timer:
                solver.substep(dt)
            wall_total += timer.elapsed
            for result in timer.timing_results:
                totals[result.name] += result.elapsed
                calls[result.name] += 1
        print(
            f"Profile: particles={solver.count:,}; iterations={args.iterations}; "
            f"wall={wall_total / args.iterations:.3f} ms/substep"
        )
        for name in sorted(totals, key=totals.get, reverse=True):
            per_step = totals[name] / args.iterations
            share = 100.0 * totals[name] / max(sum(totals.values()), 1.0e-9)
            print(f"{per_step:9.3f} ms  {share:6.2f}%  {calls[name]:4d} calls  {name}")


if __name__ == "__main__":
    main()
