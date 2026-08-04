"""Fast shallow-water A/B sweep for a visibly separate secondary bore.

This benchmark deliberately excludes buildings and SPH.  It rejects wave
trains that cannot deliver a finite rear-row impulse before an expensive
checkpoint causality run is attempted.  Promotion still requires the full
water/structure checkpoint test; an open-field pass is not sufficient.
"""

from __future__ import annotations

from _bootstrap import ROOT

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import warp as wp

from simulation.shallow_water import ShallowWaterFarField


VARIANTS = {
    "production": {},
    "delayed_same_energy": {
        "start_seconds": 12.0,
    },
    "delayed_balanced": {
        "start_seconds": 12.0,
        "duration_seconds": 1.50,
        "height": 11.0,
        "speed": 21.0,
        "background_current": 7.0,
        "length_m": 20.0,
    },
    "delayed_moderate": {
        "start_seconds": 12.0,
        "duration_seconds": 1.75,
        "height": 10.5,
        "speed": 21.0,
        "background_current": 7.0,
        "length_m": 23.0,
    },
    "delayed_compromise": {
        "start_seconds": 12.0,
        "duration_seconds": 1.75,
        "height": 10.0,
        "speed": 22.0,
        "background_current": 7.0,
        "length_m": 23.0,
    },
    "delayed_momentum": {
        "start_seconds": 12.0,
        "duration_seconds": 1.50,
        "height": 10.0,
        "speed": 23.0,
        "background_current": 7.0,
        "length_m": 20.0,
    },
    "delayed_long_bore": {
        "start_seconds": 12.0,
        "duration_seconds": 2.00,
        "height": 10.5,
        "speed": 22.0,
        "background_current": 7.0,
        "length_m": 26.0,
    },
    "delayed_penetration": {
        "start_seconds": 12.0,
        "duration_seconds": 1.75,
        "height": 12.0,
        "speed": 22.0,
        "background_current": 7.0,
        "length_m": 22.0,
    },
}


def field_config(base: dict, overrides: dict, enabled: bool = True) -> dict:
    cfg = copy.deepcopy(base)
    wave = cfg["v3"]["shallow_water"]["wave_train"]
    wave.update(overrides)
    wave["enabled"] = enabled
    return cfg


def advance_trace(cfg: dict, stop_time: float, sample_every: int) -> list[dict]:
    field = ShallowWaterFarField(cfg, cfg.get("device", "cuda:0"))
    dt = float(field.update_interval)
    rest_density = float(cfg["rest_density"])
    rows = []
    time_s = 0.0
    tick = 0
    while time_s < stop_time - 0.5 * dt:
        time_s += dt
        tick += 1
        field.advance(dt, rest_density, time_s)
        if tick % sample_every:
            continue
        row = field.diagnostics()
        row["time_s"] = time_s
        rows.append(row)
    return rows


def summarize(name: str, trace: list[dict], control: list[dict], probe_count: int) -> dict:
    if len(trace) != len(control):
        raise AssertionError("variant/control trace lengths differ")
    sample_dt = float(trace[1]["time_s"] - trace[0]["time_s"])
    result = {
        "variant": name,
        "injected_volume_m3": float(trace[-1]["wave_train_injected_volume_m3"]),
        "injected_momentum_z": float(trace[-1]["wave_train_injected_momentum_z"]),
        "transport_velocity_m_s": float(
            trace[-1]["wave_train_injected_momentum_z"]
            / max(trace[-1]["wave_train_injected_volume_m3"], 1.0e-9)
        ),
        "rows": [],
    }
    for row_index in range(1, probe_count + 1):
        prefix = f"wave_row_{row_index}"
        time = np.asarray([row["time_s"] for row in trace], dtype=np.float64)
        depth = np.asarray(
            [row[f"{prefix}_depth_mean_m"] - base[f"{prefix}_depth_mean_m"]
             for row, base in zip(trace, control)], dtype=np.float64
        )
        velocity = np.asarray(
            [row[f"{prefix}_velocity_mean_m_s"] - base[f"{prefix}_velocity_mean_m_s"]
             for row, base in zip(trace, control)], dtype=np.float64
        )
        discharge = np.asarray(
            [row[f"{prefix}_forward_discharge_m3_s"]
             - base[f"{prefix}_forward_discharge_m3_s"]
             for row, base in zip(trace, control)], dtype=np.float64
        )
        flux = np.asarray(
            [row[f"{prefix}_specific_momentum_flux_m4_s2"]
             - base[f"{prefix}_specific_momentum_flux_m4_s2"]
             for row, base in zip(trace, control)], dtype=np.float64
        )
        arrived = np.flatnonzero(depth >= 0.15)
        significant = (depth >= 0.35) & (velocity >= 0.35)
        result["rows"].append({
            "row": row_index,
            "z_m": float(trace[0][f"{prefix}_z_m"]),
            "arrival_s": float(time[arrived[0]]) if len(arrived) else None,
            "peak_depth_delta_m": float(np.max(depth)),
            "peak_velocity_delta_m_s": float(np.max(velocity)),
            "peak_forward_discharge_delta_m3_s": float(np.max(discharge)),
            "peak_specific_momentum_flux_delta_m4_s2": float(np.max(flux)),
            "positive_specific_impulse_m4_s": float(
                np.sum(np.maximum(flux, 0.0), dtype=np.float64) * sample_dt
            ),
            "significant_load_duration_s": float(np.count_nonzero(significant) * sample_dt),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config_v3_rtx5070.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-time", type=float, default=30.0)
    parser.add_argument("--sample-every", type=int, default=5)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / "outputs" / datetime.now().strftime(
        "secondary_wave_sweep_%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=True)
    wp.init()

    control = advance_trace(field_config(cfg, {}, enabled=False), args.stop_time, args.sample_every)
    summaries = []
    traces = {}
    probe_count = len(cfg["v3"]["shallow_water"].get("probe_rows_m", (16.0, 52.0, 91.0)))
    for name, overrides in VARIANTS.items():
        variant_cfg = field_config(cfg, overrides)
        (output / f"config_{name}.json").write_text(
            json.dumps(variant_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        trace = advance_trace(variant_cfg, args.stop_time, args.sample_every)
        traces[name] = trace
        summaries.append(summarize(name, trace, control, probe_count))

    selected = field_config(cfg, VARIANTS["delayed_moderate"])
    for coupling_rate in (4.0, 8.0):
        coupling_cfg = copy.deepcopy(selected)
        coupling_cfg["v3"]["shallow_water"]["velocity_relaxation_rate"] = coupling_rate
        (output / f"config_delayed_moderate_coupling_{coupling_rate:g}.json").write_text(
            json.dumps(coupling_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    wide_coupling_cfg = copy.deepcopy(selected)
    wide_coupling_cfg["v3"]["shallow_water"]["coupling_width"] = 12.0
    wide_coupling_cfg["v3"]["shallow_water"]["velocity_relaxation_rate"] = 2.5
    (output / "config_delayed_moderate_coupling_wide.json").write_text(
        json.dumps(wide_coupling_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    characteristic_cfg = copy.deepcopy(selected)
    characteristic_policy = characteristic_cfg["v3"]["shallow_water"]
    characteristic_policy["coupling_width"] = 4.0
    characteristic_policy["velocity_relaxation_rate"] = 1.5
    characteristic_policy["incoming_characteristic"] = True
    characteristic_policy["incoming_relaxation_rate"] = 6.0
    characteristic_policy["minimum_incoming_velocity"] = 0.5
    (output / "config_delayed_moderate_characteristic_inflow.json").write_text(
        json.dumps(characteristic_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    characteristic_capture_cfg = copy.deepcopy(characteristic_cfg)
    characteristic_capture_cfg["v3"]["shallow_water"]["reverse_capture_width"] = 2.0
    (output / "config_delayed_moderate_characteristic_capture.json").write_text(
        json.dumps(characteristic_capture_cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    flux_quota_cfg = copy.deepcopy(characteristic_capture_cfg)
    flux_quota_cfg["v3"]["shallow_water"]["flux_quota_enabled"] = True
    (output / "config_delayed_moderate_flux_quota.json").write_text(
        json.dumps(flux_quota_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flux_low_drag_cfg = copy.deepcopy(flux_quota_cfg)
    flux_low_drag_cfg["fluid_bed_drag"] = 0.03
    (output / "config_delayed_moderate_flux_quota_low_drag.json").write_text(
        json.dumps(flux_low_drag_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    near_interface_cfg = copy.deepcopy(flux_quota_cfg)
    near_interface_policy = near_interface_cfg["v3"]["shallow_water"]
    near_interface_policy["sph_z_min"] = -8.0
    near_interface_policy["wave_cohort"] = {
        "enabled": True,
        "id": 1,
        "start_seconds": 14.25,
        "end_seconds": 17.25,
    }
    (output / "config_delayed_moderate_flux_quota_near_interface.json").write_text(
        json.dumps(near_interface_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    minimum_vortex_band_cfg = copy.deepcopy(flux_quota_cfg)
    minimum_vortex_band_cfg["v3"]["shallow_water"]["sph_z_min"] = -4.0
    (output / "config_delayed_moderate_flux_quota_minimum_vortex_band.json").write_text(
        json.dumps(minimum_vortex_band_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    production = next(item for item in summaries if item["variant"] == "production")
    production_rear_impulse = production["rows"][-1]["positive_specific_impulse_m4_s"]
    production_front_peak = production["rows"][0]["peak_depth_delta_m"]
    for item in summaries:
        item["rear_impulse_ratio_vs_production"] = float(
            item["rows"][-1]["positive_specific_impulse_m4_s"]
            / max(production_rear_impulse, 1.0e-9)
        )
        item["front_peak_depth_ratio_vs_production"] = float(
            item["rows"][0]["peak_depth_delta_m"]
            / max(production_front_peak, 1.0e-9)
        )

    report = {
        "config": str(args.config.resolve()),
        "stop_time_s": args.stop_time,
        "sample_every_updates": args.sample_every,
        "variants": summaries,
        "promotion_gate": {
            "rear_impulse_ratio_minimum": 1.20,
            "front_peak_depth_ratio_maximum": 1.35,
            "requires_full_checkpoint_causality_test": True,
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for item in summaries:
        rear = item["rows"][-1]
        print(
            f"{item['variant']}: volume={item['injected_volume_m3']:.1f} m3, "
            f"rear arrival={rear['arrival_s']}, rear impulse="
            f"{item['rear_impulse_ratio_vs_production']:.3f}x, "
            f"front peak={item['front_peak_depth_ratio_vs_production']:.3f}x"
        )
    print(output / "comparison.json")


if __name__ == "__main__":
    main()
