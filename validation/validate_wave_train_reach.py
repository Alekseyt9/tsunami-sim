"""A/B-check that the secondary bore reaches all three city rows."""

from __future__ import annotations

from _bootstrap import ROOT

import copy
import json

import numpy as np
import warp as wp

from shallow_water import ShallowWaterFarField


def main() -> None:
    cfg = json.loads((ROOT / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    control_cfg = copy.deepcopy(cfg)
    control_cfg["v3"]["shallow_water"]["wave_train"]["enabled"] = False
    wp.init()
    device = cfg.get("device", "cuda:0")
    pulse = ShallowWaterFarField(cfg, device)
    control = ShallowWaterFarField(control_cfg, device)
    dt = float(pulse.update_interval)
    stop_time = 25.0
    sample_every = 5
    rows = (16.0, 52.0, 91.0)
    peak_depth = {row: 0.0 for row in rows}
    peak_velocity = {row: 0.0 for row in rows}
    arrival = {row: None for row in rows}
    time_s = 0.0
    tick = 0
    while time_s < stop_time - 0.5 * dt:
        time_s += dt
        tick += 1
        pulse.advance(dt, float(cfg["rest_density"]), time_s)
        control.advance(dt, float(cfg["rest_density"]), time_s)
        if tick % sample_every != 0:
            continue
        pulse_state = pulse.state.numpy()
        control_state = control.state.numpy()
        for row in rows:
            iz = int(np.clip(
                np.floor((row - pulse.lower_z) / pulse.cell_size),
                0, pulse.nz - 1,
            ))
            hp = pulse_state[:, iz, 0]
            hc = control_state[:, iz, 0]
            depth_delta = float(np.mean(hp - hc, dtype=np.float64))
            vp = pulse_state[:, iz, 2] / np.maximum(hp, 1.0e-5)
            vc = control_state[:, iz, 2] / np.maximum(hc, 1.0e-5)
            velocity_delta = float(np.mean(vp - vc, dtype=np.float64))
            peak_depth[row] = max(peak_depth[row], depth_delta)
            peak_velocity[row] = max(peak_velocity[row], velocity_delta)
            if arrival[row] is None and depth_delta >= 0.15:
                arrival[row] = time_s

    state = pulse.state.numpy()
    if not np.isfinite(state).all() or np.any(state[:, :, 0] < 0.0):
        raise AssertionError("secondary-wave reach test produced invalid state")
    for row in rows:
        if arrival[row] is None:
            raise AssertionError(
                f"secondary bore never reached row z={row:.0f} m; "
                f"peak depth delta={peak_depth[row]:.3f} m"
            )
    if peak_depth[91.0] < 0.35 or peak_velocity[91.0] < 0.35:
        raise AssertionError(
            "secondary bore reaches the rear row too weakly: "
            f"dh={peak_depth[91.0]:.3f} m, dv={peak_velocity[91.0]:.3f} m/s"
        )
    print(
        "PASS: secondary bore A/B reach "
        + ", ".join(
            f"z={row:.0f} m at {arrival[row]:.2f} s "
            f"(peak dh={peak_depth[row]:.2f} m, dv={peak_velocity[row]:.2f} m/s)"
            for row in rows
        )
    )


if __name__ == "__main__":
    main()
