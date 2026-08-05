"""Regression checks for reversal-safe shallow-to-SPH emission scheduling."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np

from simulation.shallow_water import prepare_hysteretic_emission_quota


def main() -> None:
    state = np.asarray(((10.0, 0.0, -50.0), (10.0, 0.0, -50.0)), dtype=np.float64)
    residual = np.asarray((0.75, 0.50), dtype=np.float64)
    age = np.asarray((2.0, 2.0), dtype=np.float64)
    quota, residual, age, requested = prepare_hysteretic_emission_quota(
        state, cell_size=2.0, emitter_spacing=1.0, emitter_nx=4, elapsed=0.1,
        residual_volume=residual, positive_age=age, minimum_velocity=0.25,
        rearm_delay=0.35, ramp_seconds=0.65, maximum_layers_per_frame=2,
    )
    if requested != 0 or np.any(quota) or np.any(age) or np.any(residual):
        raise AssertionError("reverse flow did not disarm and clear emission scheduling")

    state[:, 2] = 100.0
    for _ in range(3):
        quota, residual, age, requested = prepare_hysteretic_emission_quota(
            state, cell_size=2.0, emitter_spacing=1.0, emitter_nx=4, elapsed=0.1,
            residual_volume=residual, positive_age=age, minimum_velocity=0.25,
            rearm_delay=0.35, ramp_seconds=0.65, maximum_layers_per_frame=2,
        )
        if requested != 0 or np.any(quota):
            raise AssertionError("interface re-emitted before the positive-flow hold time")

    # A very large rebound cannot create more than two bottom-up layers per
    # emitter column, and the scheduler retains no whole-particle backlog.
    state[:, 2] = 1000.0
    quota, residual, age, requested = prepare_hysteretic_emission_quota(
        state, cell_size=2.0, emitter_spacing=1.0, emitter_nx=4, elapsed=0.5,
        residual_volume=residual, positive_age=age, minimum_velocity=0.25,
        rearm_delay=0.35, ramp_seconds=0.65, maximum_layers_per_frame=2,
    )
    if requested != 8 or np.any(quota != 2):
        raise AssertionError(f"column layer cap failed: quota={quota.tolist()}")
    if np.any(residual < 0.0) or np.any(residual >= 1.0):
        raise AssertionError("emission scheduler retained a whole-particle backlog")
    print("PASS: reverse flow disarms emission; restart is delayed, ramped and layer-capped")


if __name__ == "__main__":
    main()
