"""Estimate adaptive 1-4 OBB decomposition for sparse terminal rigid bodies."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_CHECKPOINT = (
    ROOT / "outputs" / "terminal_plastic_rubble_ab_checkpoint288_14f"
    / "hysteresis" / "checkpoints" / "state_00302.npz"
)


def box_for(
    indices: np.ndarray, local: np.ndarray, radius: np.ndarray,
    padding_scale: float,
) -> tuple:
    points = local[indices]
    padding = max(float(np.median(radius[indices])) * padding_scale, 1.0e-4)
    lower = np.min(points, axis=0) - padding
    upper = np.max(points, axis=0) + padding
    extent = np.maximum(upper - lower, 1.0e-5)
    return lower, upper, float(np.prod(extent))


def decompose(
    indices: np.ndarray, local: np.ndarray, radius: np.ndarray,
    volume: np.ndarray, target_ratio: float, maximum_boxes: int,
    padding_scale: float,
) -> list[np.ndarray]:
    groups = [indices]
    while len(groups) < maximum_boxes:
        ratios = []
        for group in groups:
            _, _, box_volume = box_for(group, local, radius, padding_scale)
            material_volume = float(np.sum(volume[group], dtype=np.float64))
            ratios.append(box_volume / max(material_volume, 1.0e-9))
        split_group = int(np.argmax(ratios))
        if ratios[split_group] <= target_ratio:
            break
        group = groups[split_group]
        if len(group) < 16:
            break
        points = local[group]
        axis = int(np.argmax(np.ptp(points, axis=0)))
        order = np.argsort(points[:, axis], kind="stable")
        ordered = group[order]
        weights = volume[ordered].astype(np.float64)
        midpoint = float(np.sum(weights)) * 0.5
        cut = int(np.searchsorted(np.cumsum(weights), midpoint)) + 1
        cut = min(max(cut, 4), len(ordered) - 4)
        left, right = ordered[:cut], ordered[cut:]
        if not len(left) or not len(right):
            break
        groups[split_group:split_group + 1] = [left, right]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--target-fill-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-boxes", type=int, default=4)
    parser.add_argument("--padding-scale", type=float, default=0.7)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "terminal_proxy_decomposition_checkpoint302.json",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    v3_checkpoint = checkpoint.with_name(f"v3_{checkpoint.name}")
    with np.load(checkpoint, allow_pickle=False) as saved:
        volume = saved["volume"].astype(np.float64)
        radius = saved["radius"].astype(np.float64)
        kind = saved["kind"]
    with np.load(v3_checkpoint, allow_pickle=False) as saved:
        fragment = saved["fragment_id"]
        local = saved["rigid_local_position"].astype(np.float64)
        rigid = saved["rigid_state"]
        terminal = saved["rigid_terminal"]
        proxy = saved["rigid_proxy_enabled"]
        old_extent = saved["rigid_proxy_half_extent"].astype(np.float64)
    body_mask = (rigid != 0) & (terminal != 0) & (proxy != 0)
    body_ids = np.flatnonzero(body_mask)
    counts = np.bincount(
        fragment[(fragment >= 0) & (kind != 0)], minlength=len(rigid)
    )
    old_total_volume = 0.0
    new_total_volume = 0.0
    material_total_volume = 0.0
    box_counts = []
    final_ratios = []
    for body in body_ids:
        indices = np.flatnonzero((fragment == body) & (kind != 0))
        material_volume = float(np.sum(volume[indices], dtype=np.float64))
        groups = decompose(
            indices, local, radius, volume, args.target_fill_ratio,
            args.maximum_boxes, args.padding_scale,
        )
        new_volume = sum(
            box_for(group, local, radius, args.padding_scale)[2]
            for group in groups
        )
        old_volume = float(8.0 * np.prod(old_extent[body]))
        old_total_volume += old_volume
        new_total_volume += new_volume
        material_total_volume += material_volume
        box_counts.append(len(groups))
        final_ratios.append(new_volume / max(material_volume, 1.0e-9))
    box_counts_array = np.asarray(box_counts, dtype=np.int32)
    final_ratios_array = np.asarray(final_ratios, dtype=np.float64)
    total_boxes = int(np.sum(box_counts_array))
    terminal_samples = int(np.sum(counts[body_ids]))
    quadrature_samples = total_boxes * 24
    report = {
        "checkpoint": str(checkpoint),
        "terminal_bodies": int(len(body_ids)),
        "terminal_particle_samples": terminal_samples,
        "target_box_to_material_ratio": args.target_fill_ratio,
        "maximum_boxes_per_body": args.maximum_boxes,
        "padding_scale": args.padding_scale,
        "adaptive_boxes": total_boxes,
        "boxes_per_body_histogram": {
            str(value): int(np.count_nonzero(box_counts_array == value))
            for value in range(1, args.maximum_boxes + 1)
        },
        "quadrature_samples_at_24_per_box": quadrature_samples,
        "quadrature_reduction_from_terminal_particles_percent": (
            100.0 * (1.0 - quadrature_samples / max(terminal_samples, 1))
        ),
        "old_obb_volume_m3": old_total_volume,
        "adaptive_obb_volume_m3": new_total_volume,
        "material_particle_volume_m3": material_total_volume,
        "old_volume_to_material_ratio": (
            old_total_volume / max(material_total_volume, 1.0e-9)
        ),
        "adaptive_volume_to_material_ratio": (
            new_total_volume / max(material_total_volume, 1.0e-9)
        ),
        "adaptive_body_ratio_percentiles": {
            str(q): float(np.percentile(final_ratios_array, q))
            for q in (0, 25, 50, 75, 90, 95, 99, 100)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
