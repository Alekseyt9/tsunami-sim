"""Summarize passive second-wave particle cohort transport from headless metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no metrics in {args.metrics}")
    required = (
        "wave_cohort_emitted_volume_m3",
        "wave_cohort_emitted_momentum_z_kg_m_s",
        "wave_cohort_returned_volume_m3",
        "wave_cohort_volume_m3",
        "wave_cohort_momentum_z_kg_m_s",
    )
    missing = [key for key in required if key not in rows[-1]]
    if missing:
        raise ValueError(f"metrics do not contain cohort fields: {missing}")

    final = rows[-1]
    emitted_volume = float(final["wave_cohort_emitted_volume_m3"])
    emitted_momentum = float(final["wave_cohort_emitted_momentum_z_kg_m_s"])
    peak_current = max(rows, key=lambda row: float(row["wave_cohort_momentum_z_kg_m_s"]))
    summary: dict[str, object] = {
        "metrics": str(args.metrics.resolve()),
        "frames": len(rows),
        "start_time_s": float(rows[0]["sim_time_seconds"]),
        "end_time_s": float(final["sim_time_seconds"]),
        "emitted_volume_m3": emitted_volume,
        "emitted_momentum_z_kg_m_s": emitted_momentum,
        "returned_volume_m3": float(final["wave_cohort_returned_volume_m3"]),
        "returned_volume_fraction": (
            float(final["wave_cohort_returned_volume_m3"]) / max(emitted_volume, 1.0e-9)
        ),
        "returned_momentum_z_kg_m_s": float(
            final["wave_cohort_returned_momentum_z_kg_m_s"]
        ),
        "remaining_volume_m3": float(final["wave_cohort_volume_m3"]),
        "remaining_momentum_z_kg_m_s": float(final["wave_cohort_momentum_z_kg_m_s"]),
        "peak_remaining_momentum_z_kg_m_s": float(
            peak_current["wave_cohort_momentum_z_kg_m_s"]
        ),
        "peak_remaining_momentum_time_s": float(peak_current["sim_time_seconds"]),
        "rows": [],
    }
    row_summaries: list[dict[str, object]] = []
    for row_index in range(1, 4):
        prefix = f"wave_cohort_row_{row_index}"
        arrivals = [row for row in rows if int(row[f"{prefix}_particles"]) > 0]
        peak_forward = max(
            rows, key=lambda row: float(row[f"{prefix}_forward_momentum_kg_m_s"])
        )
        peak_volume = max(rows, key=lambda row: float(row[f"{prefix}_volume_m3"]))
        peak_forward_momentum = float(peak_forward[f"{prefix}_forward_momentum_kg_m_s"])
        row_summaries.append({
            "row": row_index,
            "first_arrival_time_s": (
                float(arrivals[0]["sim_time_seconds"]) if arrivals else None
            ),
            "peak_volume_m3": float(peak_volume[f"{prefix}_volume_m3"]),
            "peak_volume_time_s": float(peak_volume["sim_time_seconds"]),
            "peak_forward_momentum_kg_m_s": peak_forward_momentum,
            "peak_forward_momentum_time_s": float(peak_forward["sim_time_seconds"]),
            "peak_forward_momentum_fraction_of_emitted": (
                peak_forward_momentum / max(emitted_momentum, 1.0e-9)
            ),
        })
    summary["rows"] = row_summaries

    output = args.output or args.metrics.with_name("cohort_summary.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
