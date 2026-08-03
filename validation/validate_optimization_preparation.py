"""CPU/static checks for the three prepared high-impact optimization paths."""

from __future__ import annotations

from _bootstrap import ROOT

import json

import numpy as np
import warp as wp

from simulation.experimental_optimizations import (  # noqa: E402
    ImplicitFluidPreparation,
    NarrowBandVolumePreparation,
)
from kernels.hybrid import update_rigid_sleep_state  # noqa: E402


def main() -> None:
    wp.init()
    cfg = json.loads((ROOT / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    v3 = cfg["v3"]
    for section in (
        "implicit_fluid", "narrow_band_volume",
    ):
        if section not in v3:
            raise AssertionError(f"missing prepared configuration section {section}")
        if bool(v3[section].get("enabled", True)):
            raise AssertionError(f"experimental section {section} changed production defaults")
    early = v3["rigid_clusters"].get("early_rigidification", {})
    if not early or bool(early.get("enabled", True)):
        raise AssertionError("early rigidification is missing or enabled in production")
    sleeping = v3["rigid_clusters"].get("sleeping", {})
    if not sleeping or bool(sleeping.get("enabled", True)):
        raise AssertionError("rigid sleeping is missing or enabled in production")

    implicit_policy = dict(v3["implicit_fluid"])
    implicit_policy["enabled"] = True
    preparation = ImplicitFluidPreparation(implicit_policy, 8, "cpu")
    radius = np.full(8, 0.5, dtype=np.float32)
    velocity = np.zeros((8, 3), dtype=np.float32)
    velocity[:, 2] = 10.0
    acceleration = np.zeros((8, 3), dtype=np.float32)
    acceleration[:, 1] = -9.81
    kind = np.zeros(8, dtype=np.int32)
    diagnostics = preparation.analyze(
        radius, velocity, acceleration, kind,
        float(cfg["dt"]), float(cfg["output_fps"]),
    )
    if diagnostics["implicit_recommended_dt_s"] <= float(cfg["dt"]):
        raise AssertionError("implicit CFL audit did not expose a larger stable target step")
    if diagnostics["implicit_predicted_substeps_per_frame"] >= 348:
        raise AssertionError("implicit preparation did not reduce predicted substeps")

    narrow = NarrowBandVolumePreparation(
        v3["narrow_band_volume"], cfg, 8, "cpu"
    )
    if narrow.enabled or narrow.nx <= 0 or narrow.ny <= 0 or narrow.nz <= 0:
        raise AssertionError("disabled narrow-band preparation has invalid grid metadata")
    narrow_policy = dict(v3["narrow_band_volume"])
    narrow_policy.update({"enabled": True, "detail_distance": 1.0})
    narrow_enabled = NarrowBandVolumePreparation(
        narrow_policy, cfg, 8, "cpu"
    )
    span = np.asarray([
        float(cfg["domain_width"]), float(cfg["domain_y_max"]),
        float(cfg["domain_z_max"]) - float(cfg["reservoir_z_min"]),
    ])
    required_dims = np.maximum(
        16, np.ceil(span / narrow_enabled.detail_distance).astype(np.int32) + 8
    )
    if np.any(np.asarray(narrow_enabled.neighbour_grid_dims) < required_dims):
        raise AssertionError("narrow-band neighbour HashGrid aliases the domain")

    state = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")
    quiet = wp.zeros(1, dtype=wp.int32, device="cpu")
    bottom = wp.array(np.asarray([0.0], dtype=np.float32), dtype=float, device="cpu")
    linear = wp.zeros(1, dtype=wp.vec3, device="cpu")
    angular = wp.zeros(1, dtype=wp.vec3, device="cpu")
    extent = wp.array(
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32), dtype=wp.vec3, device="cpu"
    )
    force = wp.array(
        np.asarray([[0.0, -9810.0, 0.0]], dtype=np.float32), dtype=float, device="cpu"
    )
    mass = wp.array(np.asarray([1000.0], dtype=np.float32), dtype=float, device="cpu")
    contact = wp.zeros(1, dtype=float, device="cpu")
    transitions = wp.zeros(2, dtype=wp.int32, device="cpu")
    sleep_inputs = [
        state, quiet, bottom, linear, angular, extent, force, mass, contact,
        transitions, 2, 0.05, 0.12, 0.18, 18.0, 18.0,
    ]
    for _ in range(2):
        wp.launch(update_rigid_sleep_state, dim=1, inputs=sleep_inputs, device="cpu")
    if state.numpy()[0] != 2:
        raise AssertionError("quiet grounded rigid proxy did not enter sleep state")
    force.assign(np.asarray([[20000.0, -9810.0, 0.0]], dtype=np.float32))
    wp.launch(update_rigid_sleep_state, dim=1, inputs=sleep_inputs, device="cpu")
    if state.numpy()[0] != 1 or transitions.numpy().tolist() != [1, 1]:
        raise AssertionError("loaded sleeping proxy did not wake as a rigid body")

    print(
        "PASS: implicit buffers/CFL audit, early-rigid/sleep-wake policy, and "
        "narrow-band coarse-grid metadata are prepared without changing production defaults"
    )


if __name__ == "__main__":
    main()
