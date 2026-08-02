"""End-to-end checkpoint regression for irreversible fragment crack energy."""

from __future__ import annotations

from _bootstrap import ROOT

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver


def fragment_maximum(solver: HybridDelugeSolver, edge_energy: np.ndarray) -> np.ndarray:
    result = np.zeros(solver.fragment_count, dtype=np.float32)
    if len(edge_energy):
        edge = solver.fragment_support_graph.edge_fragments
        np.maximum.at(result, edge[:, 0], edge_energy)
        np.maximum.at(result, edge[:, 1], edge_energy)
    return result


def main() -> None:
    with (ROOT / "config_v3_impact_validation.json").open("r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    with TemporaryDirectory(prefix="deluge_v3_fracture_checkpoint_") as temporary:
        output = Path(temporary) / "initial"
        solver = HybridDelugeSolver(cfg, output)
        if len(solver.fragment_edge_fracture_energy_host) < 3:
            raise AssertionError("production support graph unexpectedly has fewer than three edges")
        expected = solver.fragment_edge_fracture_energy_host.copy()
        expected[:3] = np.asarray((0.17, 0.43, 0.81), dtype=np.float32)
        solver.fragment_edge_fracture_energy_host = expected
        solver.fragment_fracture_energy_host = fragment_maximum(solver, expected)
        solver.fragment_fracture_energy = wp.array(
            solver.fragment_fracture_energy_host, dtype=float, device=solver.device
        )
        solver.rigid_proxy_enabled_host[5:7] = 1
        solver.rigid_proxy_local_center_host[5] = (0.2, -0.1, 0.4)
        solver.rigid_proxy_half_extent_host[5] = (1.5, 0.7, 2.0)
        solver.rigid_proxy_material_host[5] = 3
        solver.save_checkpoint(7)

        base_checkpoint = output / "checkpoints" / "state_00007.npz"
        resumed = HybridDelugeSolver(cfg, Path(temporary) / "resumed", base_checkpoint)
        restored = resumed.fragment_edge_fracture_energy_host
        if not np.array_equal(restored, expected):
            error = float(np.max(np.abs(restored - expected)))
            raise AssertionError(f"fracture energy changed across checkpoint restore: {error:.3e}")
        if (
            resumed.rigid_proxy_enabled_host[5:7].tolist() != [1, 1]
            or not np.array_equal(
                resumed.rigid_proxy_half_extent_host[5], np.asarray((1.5, 0.7, 2.0), dtype=np.float32)
            )
            or resumed.rigid_proxy_material_host[5] != 3
        ):
            raise AssertionError("rigid collision proxy changed across checkpoint restore")
        print(
            f"PASS: checkpoint preserved {len(restored):,} irreversible edge-energy values; "
            f"seed={restored[:3].tolist()}"
        )


if __name__ == "__main__":
    main()
