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
    wave_train = cfg["v3"]["shallow_water"]["wave_train"]
    end_time = float(wave_train["start_seconds"]) + float(wave_train["duration_seconds"])
    while time_s < end_time + 0.25:
        time_s += dt
        field.advance(dt, float(cfg["rest_density"]), time_s)
    state = field.state.numpy()
    if np.any(~np.isfinite(state)) or np.any(state[:, :, 0] < 0.0):
        raise AssertionError("secondary wave produced invalid or negative shallow-water state")
    injected = float(field.wave_train_injected_volume)
    momentum = float(field.wave_train_injected_momentum_z)
    if not 8000.0 <= injected <= 12500.0:
        raise AssertionError(f"unexpected injected volume: {injected:.3f} m3")
    transport_velocity = momentum / max(injected, 1.0e-9)
    expected_velocity = float(wave_train["background_current"]) + float(wave_train["speed"])
    if abs(transport_velocity - expected_velocity) > 0.05:
        raise AssertionError(
            f"secondary wave transport velocity is {transport_velocity:.3f} m/s, "
            f"expected {expected_velocity:.3f} m/s"
        )
    print(
        f"PASS: finite second pulse injected {injected:,.1f} m3 and "
        f"{momentum:,.1f} m4/s ({transport_velocity:.2f} m/s transport) "
        "without negative depth"
    )


if __name__ == "__main__":
    main()
