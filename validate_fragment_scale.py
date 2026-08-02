"""CPU regression for architectural-scale anti-dust fragments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from scene import ParticleScene  # noqa: E402
from hybrid_model import build_fragment_ids  # noqa: E402


def main() -> None:
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    cfg["solid_spacing"] = float(cfg["v3"]["solid_refinement"]["coarse_spacing"])
    scene = ParticleScene(rest_density=float(cfg["rest_density"]))
    scene.add_city(cfg)
    arrays = scene.as_numpy()
    fragment_id, counts = build_fragment_ids(
        arrays["rest_x"], arrays["kind"], arrays["building_id"], cfg
    )
    if not 2500 <= len(counts) <= 3500:
        raise AssertionError(f"unexpected architectural fragment count: {len(counts):,}")
    if int(counts.min()) < 9 or float(np.median(counts)) < 18.0:
        raise AssertionError(
            f"anti-dust fragments are too small: min={counts.min()}, median={np.median(counts):.1f}"
        )
    if np.any(fragment_id[arrays["kind"] != 0] < 0):
        raise AssertionError("one or more structural particles has no cohesive fragment")
    print(
        f"PASS: {len(counts):,} apartment-scale fragments; "
        f"coarse count min/median/p90/max={counts.min()}/{np.median(counts):.0f}/"
        f"{np.quantile(counts, 0.9):.0f}/{counts.max()}"
    )


if __name__ == "__main__":
    main()
