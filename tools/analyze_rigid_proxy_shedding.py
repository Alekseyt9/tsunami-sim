"""Audit which rigid particle samples could safely become proxy-only state."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np


def companion_v3_checkpoint(state_path: Path) -> Path:
    if state_path.name.startswith("v3_state_"):
        return state_path
    return state_path.with_name(f"v3_{state_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--minimum-quiet-scans", type=int, default=3)
    parser.add_argument("--minimum-detached-scans", type=int, default=2)
    parser.add_argument("--maximum-linear-speed", type=float, default=2.0)
    parser.add_argument("--maximum-angular-speed", type=float, default=1.0)
    parser.add_argument("--quadrature-samples-per-body", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    state_path = args.checkpoint.resolve()
    v3_path = companion_v3_checkpoint(state_path)
    if not state_path.exists() or not v3_path.exists():
        raise FileNotFoundError(f"checkpoint pair is incomplete: {state_path}, {v3_path}")

    with np.load(state_path, allow_pickle=False) as state, np.load(
        v3_path, allow_pickle=False
    ) as v3:
        fragment = v3["fragment_id"].astype(np.int64, copy=False)
        rigid_state = v3["rigid_state"].astype(np.int32, copy=False)
        rigid_terminal = (
            v3["rigid_terminal"].astype(np.int32, copy=False)
            if "rigid_terminal" in v3
            else np.zeros_like(rigid_state)
        )
        proxy_enabled = v3["rigid_proxy_enabled"].astype(np.int32, copy=False)
        quiet = v3["rigid_quiet_scans"].astype(np.int32, copy=False)
        detached = v3["rigid_detached_scans"].astype(np.int32, copy=False)
        linear_speed = np.linalg.norm(v3["body_linear_velocity"], axis=1)
        angular_speed = np.linalg.norm(v3["body_angular_velocity"], axis=1)
        particle_count = np.bincount(
            fragment[fragment >= 0], minlength=len(rigid_state)
        ).astype(np.int64)
        particle_volume = np.bincount(
            fragment[fragment >= 0],
            weights=state["volume"][fragment >= 0].astype(np.float64, copy=False),
            minlength=len(rigid_state),
        )
        particle_mass = np.bincount(
            fragment[fragment >= 0],
            weights=state["mass"][fragment >= 0].astype(np.float64, copy=False),
            minlength=len(rigid_state),
        )

    active = (rigid_state != 0) & (proxy_enabled != 0)
    terminal_active = active & (rigid_terminal != 0)
    candidate = (
        active
        & (quiet >= max(0, args.minimum_quiet_scans))
        & (detached >= max(0, args.minimum_detached_scans))
        & (linear_speed <= args.maximum_linear_speed)
        & (angular_speed <= args.maximum_angular_speed)
    )
    quadrature_samples = max(6, int(args.quadrature_samples_per_body))
    terminal_quadrature_count = int(np.count_nonzero(terminal_active)) * quadrature_samples
    terminal_particle_count = int(
        np.sum(particle_count[terminal_active], dtype=np.int64)
    )
    report = {
        "state_checkpoint": str(state_path),
        "v3_checkpoint": str(v3_path),
        "policy": {
            "minimum_quiet_scans": args.minimum_quiet_scans,
            "minimum_detached_scans": args.minimum_detached_scans,
            "maximum_linear_speed_m_s": args.maximum_linear_speed,
            "maximum_angular_speed_rad_s": args.maximum_angular_speed,
            "quadrature_samples_per_body": quadrature_samples,
        },
        "active_proxy_bodies": int(np.count_nonzero(active)),
        "active_proxy_particles": int(np.sum(particle_count[active], dtype=np.int64)),
        "terminal_proxy_bodies": int(np.count_nonzero(terminal_active)),
        "terminal_proxy_particles": terminal_particle_count,
        "terminal_proxy_volume_m3": float(
            np.sum(particle_volume[terminal_active], dtype=np.float64)
        ),
        "terminal_proxy_mass_kg": float(
            np.sum(particle_mass[terminal_active], dtype=np.float64)
        ),
        "terminal_obb_quadrature_samples": terminal_quadrature_count,
        "terminal_sample_reduction": terminal_particle_count - terminal_quadrature_count,
        "terminal_sample_reduction_fraction": float(
            1.0 - terminal_quadrature_count / max(terminal_particle_count, 1)
        ),
        "candidate_bodies": int(np.count_nonzero(candidate)),
        "candidate_particles": int(np.sum(particle_count[candidate], dtype=np.int64)),
        "candidate_volume_m3": float(np.sum(particle_volume[candidate], dtype=np.float64)),
        "candidate_mass_kg": float(np.sum(particle_mass[candidate], dtype=np.float64)),
        "candidate_particle_fraction": float(
            np.sum(particle_count[candidate], dtype=np.float64)
            / max(np.sum(particle_count[active], dtype=np.float64), 1.0)
        ),
        "requires_before_physical_shedding": [
            "SPH-to-OBB hydrodynamic pressure/drag with equal body reaction",
            "fragment-owned render skin independent of live particle indices",
            "checkpointed proxy dwell age and local rehydration on fracture-level impact",
        ],
    }
    output = args.output or state_path.with_name("rigid_proxy_shedding_analysis.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == "__main__":
    main()
