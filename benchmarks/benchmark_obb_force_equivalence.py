"""Compare one-substep rigid loads from samples and terminal OBB proxies."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver, load_run_config


DEFAULT_CHECKPOINT = (
    ROOT / "outputs" / "terminal_plastic_rubble_ab_checkpoint288_14f"
    / "hysteresis" / "checkpoints" / "state_00302.npz"
)


def run_once(
    config: dict, checkpoint: Path, output: Path, proxy: bool, steps: int,
    quadrature_force: bool = True, analytic_force: bool = True,
) -> dict:
    cfg = copy.deepcopy(config)
    cfg["checkpoint_every_frames"] = 0
    policy = cfg["v3"]["rigid_clusters"]["proxy_hydrodynamics"]
    policy["enabled"] = proxy
    policy["shed_terminal_samples"] = proxy
    policy.setdefault("occupancy", {})["enabled"] = bool(
        proxy and policy.get("occupancy", {}).get("enabled", False)
    )
    analytic = policy["analytic_contact"]
    analytic["enabled"] = proxy
    analytic.setdefault("candidate_cache", {})["enabled"] = proxy
    if proxy and not quadrature_force:
        policy["maximum_body_acceleration"] = 0.0
    if proxy and not analytic_force:
        analytic["stiffness"] = 0.0
        analytic["normal_damping"] = 0.0
        analytic["tangential_damping"] = 0.0
    solver = HybridDelugeSolver(cfg, output, checkpoint)
    substeps = int(math.ceil((1.0 / float(cfg["output_fps"])) / float(cfg["dt"])))
    dt = (1.0 / float(cfg["output_fps"])) / substeps
    force_impulse = np.zeros_like(solver.body_force.numpy(), dtype=np.float64)
    torque_impulse = np.zeros_like(solver.body_torque.numpy(), dtype=np.float64)
    contact_sum = 0
    candidate_peak = 0
    cache_active_peak = 0
    for _ in range(steps):
        solver.substep(dt)
        wp.synchronize_device(solver.device)
        force_impulse += solver.body_force.numpy().astype(np.float64) * dt
        torque_impulse += solver.body_torque.numpy().astype(np.float64) * dt
        contact_sum += int(solver.rigid_proxy_fluid_contact_counter.numpy()[0])
        candidate_peak = max(
            candidate_peak,
            int(solver.rigid_proxy_fluid_candidate_counter.numpy()[0]),
        )
        cache_active_peak = max(
            cache_active_peak,
            int(solver.rigid_proxy_fluid_contact_cache_active_count),
        )
    result = {
        "body_force_average": force_impulse / (dt * steps),
        "body_torque_average": torque_impulse / (dt * steps),
        "body_mass": solver.body_mass.numpy(),
        "body_center": solver.body_center.numpy(),
        "rigid_state": solver.rigid_state.numpy(),
        "terminal": solver.rigid_terminal.numpy(),
        "proxy_enabled": solver.rigid_proxy_enabled.numpy(),
        "contact_cache_active": cache_active_peak,
        "contact_candidates": candidate_peak,
        "contacts": contact_sum,
        "dt": dt,
        "steps": steps,
    }
    del solver
    gc.collect()
    return result


def vector_rms(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(values.astype(np.float64) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config_v3_sustained_surge_30s.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "obb_force_equivalence_checkpoint302",
    )
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument(
        "--decompose", action="store_true",
        help="also run quadrature-only and analytical-contact-only variants",
    )
    parser.add_argument("--occupancy", action="store_true")
    args = parser.parse_args()
    wp.init()
    cfg = load_run_config(args.config.resolve())
    analytic = cfg["v3"]["rigid_clusters"]["proxy_hydrodynamics"][
        "analytic_contact"
    ]
    analytic["bvh_refit_every_substeps"] = 4
    analytic.setdefault("candidate_cache", {})[
        "maximum_bodies_per_particle"
    ] = 48
    cfg["v3"]["rigid_clusters"]["proxy_hydrodynamics"].setdefault(
        "occupancy", {}
    )["enabled"] = bool(args.occupancy)
    if args.substeps < 8:
        raise ValueError("--substeps must cover at least one contact stride (8)")
    baseline = run_once(
        cfg, args.checkpoint.resolve(), args.output / "baseline", False,
        args.substeps,
    )
    proxy = run_once(
        cfg, args.checkpoint.resolve(), args.output / "proxy", True,
        args.substeps,
    )
    component_states = {}
    if args.decompose:
        component_states["quadrature_only"] = run_once(
            cfg, args.checkpoint.resolve(), args.output / "quadrature_only",
            True, args.substeps, quadrature_force=True, analytic_force=False,
        )
        component_states["analytic_contact_only"] = run_once(
            cfg, args.checkpoint.resolve(), args.output / "analytic_contact_only",
            True, args.substeps, quadrature_force=False, analytic_force=True,
        )
    count = min(len(baseline["rigid_state"]), len(proxy["rigid_state"]))
    mask = (
        (baseline["rigid_state"][:count] != 0)
        & (baseline["terminal"][:count] != 0)
        & (baseline["proxy_enabled"][:count] != 0)
        & (proxy["rigid_state"][:count] != 0)
    )
    gravity = np.zeros((count, 3), dtype=np.float64)
    gravity[:, 1] = -baseline["body_mass"][:count].astype(np.float64) * 9.81
    base_external = baseline["body_force_average"][:count] - gravity
    proxy_external = proxy["body_force_average"][:count] - gravity
    base_selected = base_external[mask]
    proxy_selected = proxy_external[mask]
    delta = proxy_selected - base_selected
    base_norm = np.linalg.norm(base_selected, axis=1)
    proxy_norm = np.linalg.norm(proxy_selected, axis=1)
    meaningful = base_norm > 100.0
    dot = np.sum(base_selected[meaningful] * proxy_selected[meaningful], axis=1)
    cosine = dot / np.maximum(
        base_norm[meaningful] * proxy_norm[meaningful], 1.0e-9
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "dt": baseline["dt"],
        "substeps": args.substeps,
        "window_seconds": baseline["dt"] * args.substeps,
        "terminal_proxy_bodies": int(np.count_nonzero(mask)),
        "meaningfully_loaded_baseline_bodies": int(np.count_nonzero(meaningful)),
        "baseline_external_force_vector_rms_n": vector_rms(base_selected),
        "proxy_external_force_vector_rms_n": vector_rms(proxy_selected),
        "force_delta_vector_rms_n": vector_rms(delta),
        "total_baseline_external_force_n": base_selected.sum(axis=0).tolist(),
        "total_proxy_external_force_n": proxy_selected.sum(axis=0).tolist(),
        "total_force_delta_n": delta.sum(axis=0).tolist(),
        "meaningful_force_direction_cosine_median": (
            float(np.median(cosine)) if cosine.size else 0.0
        ),
        "meaningful_force_magnitude_ratio_median": (
            float(np.median(
                proxy_norm[meaningful] / np.maximum(base_norm[meaningful], 1.0e-9)
            )) if np.any(meaningful) else 0.0
        ),
        "proxy_contact_cache_active_particles": proxy["contact_cache_active"],
        "proxy_contact_candidates": proxy["contact_candidates"],
        "proxy_contacts": proxy["contacts"],
    }
    for name, state in component_states.items():
        component_external = state["body_force_average"][:count] - gravity
        selected = component_external[mask]
        report[name] = {
            "external_force_vector_rms_n": vector_rms(selected),
            "total_external_force_n": selected.sum(axis=0).tolist(),
            "delta_from_baseline_vector_rms_n": vector_rms(
                selected - base_selected
            ),
            "contact_cache_active_particles": state["contact_cache_active"],
            "contact_candidates": state["contact_candidates"],
            "contacts": state["contacts"],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
