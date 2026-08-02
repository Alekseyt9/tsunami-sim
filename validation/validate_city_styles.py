"""Regression for physical building profiles and facade palette diversity."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent.parent

from scene import ParticleScene, building_profile  # noqa: E402
from hybrid_model import build_facade_skin, build_fragment_debris_skin, build_fragment_ids  # noqa: E402


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
        raise AssertionError("debris hulls are not bound to every cohesive fragment")
    if not np.isfinite(debris["vertex"]).all():
        raise AssertionError("fragment debris skin contains invalid vertices")
    debris_face_count = np.bincount(debris["owner_fragment"], minlength=len(fragment_counts))
    if np.any(debris_face_count < 4):
        raise AssertionError("one or more fragment debris hulls is not a closed volume")
    convex_mask = np.all(
        np.isclose(debris["vertex"][:, 2], debris["vertex"][:, 3], atol=1.0e-6), axis=1
    )
    convex_fragments = np.unique(debris["owner_fragment"][convex_mask])
    if len(convex_fragments) < int(0.98 * len(fragment_counts)):
        raise AssertionError("too many fragment hulls fell back to axis-aligned boxes")
    # Qhull triangles form a watertight surface: every undirected edge must
    # belong to exactly two faces. Check representative fragments across the city.
    sample_fragments = np.linspace(0, len(fragment_counts) - 1, 32, dtype=np.int32)
    for fragment in sample_fragments:
        triangles = debris["vertex"][(debris["owner_fragment"] == fragment) & convex_mask, :3]
        edge_count: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
        for triangle in triangles:
            keys = [tuple(np.round(point, 5)) for point in triangle]
            for edge in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
                canonical = tuple(sorted(edge))
                edge_count[canonical] = edge_count.get(canonical, 0) + 1
        if edge_count and any(count != 2 for count in edge_count.values()):
            raise AssertionError(f"fragment {fragment} convex debris hull is not watertight")

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
        f"{len(material):,} facade panels, {len(debris['material']):,} hidden convex debris triangles "
        f"({len(convex_fragments):,}/{len(fragment_counts):,} convex fragments) "
        "and six wall/glass/roof palettes"
    )


if __name__ == "__main__":
    main()
