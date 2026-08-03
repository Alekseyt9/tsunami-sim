"""Validate finite conservative secondary-wave injection on CUDA."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np
import warp as wp

from shallow_water import ShallowWaterFarField


def main() -> None:
    cfg = json.loads((ROOT / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    wp.init()
    field = ShallowWaterFarField(cfg, cfg.get("device", "cuda:0"))
    dt = float(field.update_interval)
    time_s = 0.0
    while time_s < 10.25:
        time_s += dt
        field.advance(dt, float(cfg["rest_density"]), time_s)
    state = field.state.numpy()
    if np.any(~np.isfinite(state)) or np.any(state[:, :, 0] < 0.0):
        raise AssertionError("secondary wave produced invalid or negative shallow-water state")
    injected = float(field.wave_train_injected_volume)
    momentum = float(field.wave_train_injected_momentum_z)
    if not 7000.0 <= injected <= 12500.0:
        raise AssertionError(f"unexpected injected volume: {injected:.3f} m3")
    if momentum <= 0.0:
        raise AssertionError("secondary wave did not inject forward momentum")
    print(
        f"PASS: finite second pulse injected {injected:,.1f} m3 and "
        f"{momentum:,.1f} m4/s without negative depth"
    )


if __name__ == "__main__":
    main()
