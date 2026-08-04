"""Validate the long-duration Riemann-boundary tsunami surge preset."""

from __future__ import annotations

from _bootstrap import ROOT

import copy
import json
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import load_run_config
from simulation.shallow_water import ShallowWaterFarField


def main() -> None:
    config_path = ROOT / "config_v3_sustained_surge_30s.json"
    cfg = load_run_config(config_path)
    control_cfg = copy.deepcopy(cfg)
    control_cfg["v3"]["shallow_water"]["wave_train"]["enabled"] = False
    device = cfg.get("device", "cuda:0")
    wp.init()
    surge = ShallowWaterFarField(cfg, device)
    control = ShallowWaterFarField(control_cfg, device)
    dt = float(cfg["v3"]["shallow_water"]["update_interval"])
    stop_time = 24.0
    sample_interval = 0.04
    next_sample = 0.0
    trace: list[dict[str, float]] = []
    time_s = 0.0
    while time_s < stop_time - 0.5 * dt:
        time_s += dt
        surge.advance(dt, float(cfg["rest_density"]), time_s)
        control.advance(dt, float(cfg["rest_density"]), time_s)
        if time_s + 1.0e-9 >= next_sample:
            current = surge.diagnostics()
            baseline = control.diagnostics()
            row: dict[str, float] = {"time_s": time_s}
            for row_index in range(1, 4):
                for suffix in (
                    "velocity_mean_m_s",
                    "forward_discharge_m3_s",
                    "specific_momentum_flux_m4_s2",
                ):
                    key = f"wave_row_{row_index}_{suffix}"
                    row[key] = float(current[key] - baseline[key])
            trace.append(row)
            next_sample += sample_interval

    diagnostics = surge.diagnostics()
    control_diagnostics = control.diagnostics()
    state = surge.state.numpy()
    if not np.isfinite(state).all() or float(np.min(state[:, :, 0])) < -1.0e-6:
        raise AssertionError("sustained surge produced invalid/negative water depth")
    volume_delta = (
        diagnostics["shallow_water_volume_m3"]
        - control_diagnostics["shallow_water_volume_m3"]
    )
    injected = diagnostics["wave_train_injected_volume_m3"]
    relative_error = abs(volume_delta - injected) / max(abs(injected), 1.0)
    if relative_error > 2.0e-3:
        raise AssertionError(
            f"surge boundary is not conservative: delta={volume_delta}, source={injected}"
        )

    row_summaries = []
    for row_index in range(1, 4):
        discharge = np.asarray(
            [item[f"wave_row_{row_index}_forward_discharge_m3_s"] for item in trace],
            dtype=np.float64,
        )
        peak = float(np.max(discharge))
        threshold = max(0.25 * peak, 1.0)
        active = np.flatnonzero(discharge >= threshold)
        row_summaries.append({
            "row": row_index,
            "arrival_s": float(trace[int(active[0])]["time_s"]) if len(active) else None,
            "peak_forward_discharge_m3_s": peak,
            "significant_duration_s": float(len(active) * sample_interval),
        })

    rear = row_summaries[-1]
    if rear["arrival_s"] is None or float(rear["arrival_s"]) > 18.0:
        raise AssertionError(f"surge did not reach the rear-row probe in time: {rear}")
    if float(rear["significant_duration_s"]) < 5.0:
        raise AssertionError(f"rear-row surge is too brief: {rear}")

    report = {
        "config": str(config_path),
        "stop_time_s": stop_time,
        "injected_volume_m3": injected,
        "injected_momentum_z": diagnostics["wave_train_injected_momentum_z"],
        "volume_conservation_relative_error": relative_error,
        "rows": row_summaries,
    }
    output = ROOT / "outputs" / "sustained_surge_shallow_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS: sustained surge is conservative and remains loaded at the rear probe")


if __name__ == "__main__":
    main()
