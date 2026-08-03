"""Regression for physical building profiles and facade palette diversity."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent.parent

from simulation.scene import ParticleScene, building_profile, environment_layout  # noqa: E402
from simulation.hybrid_model import (  # noqa: E402
    build_facade_skin,
    build_fragment_cell_faces,
    build_fragment_debris_skin,
    build_fragment_ids,
)
from simulation.scene import STRUCT_WALL  # noqa: E402


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
    layout = environment_layout(cfg)
    shop_back = max(
        float(shop["center"][1]) + 0.5 * float(shop["size"][1])
        for shop in layout["small_buildings"]
    )
    first_row_front = min(
        float(spec[1]) - 0.5 * float(spec[3]) for spec in cfg["buildings"][:5]
    )
    if shop_back >= first_row_front - 1.5:
        raise AssertionError(
            "foreground shops do not leave a visible water corridor before the first row"
        )

    scene_cfg = dict(cfg)
    scene_cfg["solid_spacing"] = float(cfg["v3"]["solid_refinement"]["coarse_spacing"])
    scene = ParticleScene(rest_density=float(cfg["rest_density"]))
    counts = scene.add_city(scene_cfg)
    if len(counts) != len(cfg["buildings"]) or min(counts) <= 0:
        raise AssertionError("one or more styled buildings has no physical lattice")
    state = scene.as_numpy()
    fragment_id, fragment_counts = build_fragment_ids(
        state["rest_x"], state["kind"], state["building_id"], scene_cfg,
        state["structural_class"],
    )
    debris = build_fragment_debris_skin(
        scene_cfg, state["rest_x"], state["kind"], state["building_id"], fragment_id,
        state["radius"], state["structural_class"],
    )
    if not np.all(debris["panel_mode"] == 1) or set(debris["owner_fragment"]) != set(range(len(fragment_counts))):
        raise AssertionError("debris cell surfaces are not bound to every cohesive fragment")
    if not np.isfinite(debris["vertex"]).all():
        raise AssertionError("fragment debris cell surface contains invalid vertices")
    debris_face_count = np.bincount(debris["owner_fragment"], minlength=len(fragment_counts))
    if np.any(debris_face_count < 4):
        raise AssertionError("one or more fragment cell union is not a closed volume")
    if np.any(np.all(np.isclose(debris["vertex"][:, 2], debris["vertex"][:, 3]), axis=1)):
        raise AssertionError("legacy convex-hull triangles remain in the debris skin")

    # An L-shaped set must retain its empty fourth cell. A global ConvexHull
    # incorrectly covers that void with one solid polygon.
    l_shape = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32)
    l_faces = build_fragment_cell_faces(
        l_shape, np.full(3, 0.48, dtype=np.float32),
        np.full(3, STRUCT_WALL, dtype=np.int32), 0, 1.0,
    )
    missing_center = np.asarray((1.0, 1.0), dtype=np.float32)
    for center, size, normal, _material in l_faces:
        if normal[2] < 0.5:
            continue
        lower = center[:2] - size[:2] * 0.5
        upper = center[:2] + size[:2] * 0.5
        if np.all(missing_center > lower + 1.0e-4) and np.all(missing_center < upper - 1.0e-4):
            raise AssertionError("cell-union debris surface filled an L-shaped void")

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
        f"{len(material):,} facade panels, {len(debris['material']):,} hidden cell-union faces "
        f"for {len(fragment_counts):,} cohesive fragments "
        "and six wall/glass/roof palettes"
    )


if __name__ == "__main__":
    main()
