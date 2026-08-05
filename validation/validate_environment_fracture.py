"""Validate foreground placement and fragment topology for dynamic props."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json

import numpy as np

from simulation.hybrid_model import (
    build_fragment_ids,
    build_fragment_support_graph,
    build_refinement_axes,
)
from simulation.scene import ParticleScene, environment_layout


def main() -> None:
    cfg = json.loads((ROOT / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    layout = environment_layout(cfg)
    foreground = [car for car in layout["cars"] if float(car["center"][1]) < 0.0]
    if len(foreground) < 10:
        raise AssertionError("cars were not moved into the foreground wave corridor")

    scene = ParticleScene(rest_density=float(cfg["rest_density"]))
    counts = scene.add_environment(cfg)
    state = scene.as_numpy()
    if counts["cars"] < len(layout["cars"]) * 50:
        raise AssertionError("cars still use the legacy twelve-particle proxy")
    if counts["trees"] < len(layout["trees"]) * 40:
        raise AssertionError("trees do not have a resolved breakable trunk/crown lattice")

    fragment, fragment_counts = build_fragment_ids(
        state["rest_x"], state["kind"], state["building_id"], cfg,
        state["structural_class"],
    )
    dynamic_environment = (state["building_id"] < 0) & (state["building_id"] != -9000)
    if np.any(fragment[dynamic_environment] < 0):
        raise AssertionError("one or more dynamic environment particles lacks a fragment")
    for object_id in np.unique(state["building_id"][dynamic_environment]):
        object_fragments = np.unique(fragment[state["building_id"] == object_id])
        if len(object_fragments) < 2:
            raise AssertionError(f"environment object {object_id} is still one rigid box")

    axes = build_refinement_axes(
        state["rest_x"], state["kind"], state["building_id"],
        float(cfg["solid_spacing"]), state["structural_class"], cfg,
    )
    if np.count_nonzero(axes[dynamic_environment] >= 0) < 0.90 * np.count_nonzero(dynamic_environment):
        raise AssertionError("environment sheet/beam axes were not reconstructed")
    graph = build_fragment_support_graph(
        state["rest_x"], state["radius"], state["kind"], state["building_id"],
        state["fixed"], fragment, 12,
    )
    if len(graph.edge_fragments) == 0 or not np.any(graph.anchored_fragments):
        raise AssertionError("environment support graph has no breakable edges or ground anchors")
    print(
        f"PASS: {len(foreground)} foreground cars; {counts['cars']:,} car and "
        f"{counts['trees']:,} tree particles; {len(fragment_counts):,} prop/shop fragments"
    )


if __name__ == "__main__":
    main()
