"""Render one V3 checkpoint without rerunning physics (diagnostic utility)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import warp as wp

HERE = Path(__file__).resolve().parent

from solver_base import compose_hero_insets, compose_quad_view  # noqa: E402
from hybrid_renderer import HybridRenderer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--skin", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    render = cfg["render"]
    wp.init()
    device = cfg.get("device", "cuda:0")
    with np.load(args.state, allow_pickle=False) as state:
        host = {name: state[name].copy() for name in ("x", "rest_x", "v", "radius", "kind", "damage")}
        time_s = float(state["time"])
        frame = int(state["frame"])
    arrays = {
        "x": wp.array(host["x"], dtype=wp.vec3, device=device),
        "rest_x": wp.array(host["rest_x"], dtype=wp.vec3, device=device),
        "v": wp.array(host["v"], dtype=wp.vec3, device=device),
        "radius": wp.array(host["radius"], dtype=float, device=device),
        "kind": wp.array(host["kind"], dtype=wp.int32, device=device),
        "damage": wp.array(host["damage"], dtype=float, device=device),
    }
    solid = int(np.count_nonzero(host["kind"] != 0))
    stats = {
        "fluid": len(host["kind"]) - solid,
        "solid": solid,
        "damaged": int(np.count_nonzero((host["kind"] != 0) & (host["damage"] > 0.05))),
        "cohesive_fragments": 0,
        "released_fragments": 0,
    }
    view_width = int(render.get("view_width", render["width"]))
    view_height = int(render.get("view_height", render["height"]))
    frames = {}
    for name, camera in render["views"].items():
        renderer = HybridRenderer(
            int(camera.get("render_width", view_width)),
            int(camera.get("render_height", view_height)),
            camera, device, args.skin, name,
            float(render.get("maximum_panel_stretch", 1.8)),
        )
        frames[name] = renderer.render(arrays, len(host["kind"]), None, frame, time_s, stats)
    compose = (
        compose_hero_insets
        if render.get("view_layout") == "hero_insets"
        else compose_quad_view
    )
    quad = compose(
        frames, list(render["quad_order"]), int(render["width"]), int(render["height"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quad).save(args.output, compress_level=2)
    print(args.output)


if __name__ == "__main__":
    main()
