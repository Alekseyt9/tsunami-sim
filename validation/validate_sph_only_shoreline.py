"""Regression for a shallow-water-free SPH reservoir aligned to the land edge."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import load_run_config  # noqa: E402
from simulation.hybrid_model import build_environment_skin  # noqa: E402
from simulation.scene import ParticleScene  # noqa: E402
from simulation.shallow_water import ShallowWaterFarField  # noqa: E402


def main() -> None:
    cfg = load_run_config(Path(ROOT) / "config_v3_city_sph_only_30s.json")
    policy = cfg["v3"]["shallow_water"]
    if bool(policy["enabled"]):
        raise AssertionError("SPH-only production preset still enables shallow water")
    if any(bool(policy[name]) for name in ("replace_far_sph", "render_far_surface", "stitch_surface", "emit_sph", "merge_sph")):
        raise AssertionError("a shallow-water coupling/render path remains enabled")

    scene = ParticleScene(rest_density=float(cfg["rest_density"]))
    scene.add_water(cfg)
    position = np.asarray(scene.positions, dtype=np.float32)
    radius = np.asarray(scene.radii, dtype=np.float32)
    water_front = float(np.max(position[:, 2] + radius))
    water_back = float(np.min(position[:, 2] - radius))
    skin = build_environment_skin(cfg)
    terrain = np.flatnonzero(skin["material"] == 90)
    if len(terrain) != 1:
        raise AssertionError(f"expected one terrain panel, found {len(terrain)}")
    terrain_start = float(np.min(skin["vertex"][terrain[0], :, 2]))
    if abs(water_front - terrain_start) > 0.06:
        raise AssertionError(
            f"water/land seam is misregistered: water={water_front}, land={terrain_start}"
        )
    if len(position) >= int(cfg["max_particles"]) // 2:
        raise AssertionError("initial offshore SPH reservoir consumes too much particle capacity")

    wp.init()
    far_field = ShallowWaterFarField(cfg, "cpu")
    diagnostics = far_field.diagnostics()
    if diagnostics["shallow_water_cells"] != 0 or diagnostics["shallow_water_volume_m3"] != 0.0:
        raise AssertionError("disabled shallow-water model still reports live state")
    peak_speed = float(cfg["background_current"]) + float(cfg["wave_speed"]) * float(
        cfg["wave_height"]
    ) / (float(cfg["water_depth"]) + float(cfg["wave_height"]))
    print(
        f"PASS: {len(position):,} SPH water particles span {water_back:.1f}..{water_front:.1f} m; "
        f"land starts at {terrain_start:.1f} m; shallow state is zero; crest speed={peak_speed:.2f} m/s"
    )


if __name__ == "__main__":
    main()
