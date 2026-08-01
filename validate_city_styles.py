"""Regression for physical building profiles and facade palette diversity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "v2_particle_solver"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from scene import ParticleScene, building_profile  # noqa: E402
from hybrid_model import build_facade_skin  # noqa: E402


def main() -> None:
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    styles = cfg["building_styles"]
    signatures = set()
    changed_profiles = 0
    for spec, style in zip(cfg["buildings"], styles):
        _cx, _cz, width, depth, height = map(float, spec)
        slices = tuple(
            tuple(round(value, 3) for value in building_profile(style, width, depth, height, ratio * height))
            for ratio in (0.15, 0.55, 0.88)
        )
        signatures.add(slices)
        changed_profiles += int(slices[0] != slices[-1])
    if len(signatures) < 10 or changed_profiles < 10:
        raise AssertionError(f"city profiles are insufficiently varied: {len(signatures)} signatures")

    scene_cfg = dict(cfg)
    scene_cfg["solid_spacing"] = float(cfg["v3"]["solid_refinement"]["coarse_spacing"])
    scene = ParticleScene(rest_density=float(cfg["rest_density"]))
    counts = scene.add_city(scene_cfg)
    if len(counts) != len(cfg["buildings"]) or min(counts) <= 0:
        raise AssertionError("one or more styled buildings has no physical lattice")

    skin = build_facade_skin(cfg)
    material = skin["material"]
    wall_palettes = set((material[(material >= 10) & (material < 20)] % 10).tolist())
    glass_palettes = set((material[(material >= 20) & (material < 30)] % 10).tolist())
    roof_palettes = set((material[(material >= 30) & (material < 40)] % 10).tolist())
    if len(wall_palettes) < 6 or len(glass_palettes) < 6 or len(roof_palettes) < 6:
        raise AssertionError("not all six facade palettes are represented")
    if not np.isfinite(skin["vertex"]).all():
        raise AssertionError("styled facade contains invalid vertices")
    print(
        f"PASS: {len(signatures)} physical silhouettes, {sum(counts):,} coarse structural particles, "
        f"{len(material):,} panels and six wall/glass/roof palettes"
    )


if __name__ == "__main__":
    main()
