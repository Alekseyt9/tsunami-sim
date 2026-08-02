"""CPU regression for foundation-connected architectural fragment support."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from hybrid_model import (  # noqa: E402
    build_fragment_support_graph,
    evaluate_fragment_fracture_energy,
    evaluate_fragment_support,
)


def main() -> None:
    rest = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)), dtype=np.float32)
    radius = np.full(3, 0.5, dtype=np.float32)
    kind = np.ones(3, dtype=np.int32)
    building = np.zeros(3, dtype=np.int32)
    fixed = np.asarray((1, 0, 0), dtype=np.int32)
    fragment = np.asarray((0, 1, 2), dtype=np.int32)
    graph = build_fragment_support_graph(rest, radius, kind, building, fixed, fragment, 4)
    if len(graph.edge_fragments) != 2:
        raise AssertionError(f"expected two load-path edges, got {graph.edge_fragments.tolist()}")

    damage = np.zeros(3, dtype=np.float32)
    material = np.ones(3, dtype=np.int32)
    role = np.full(3, 2, dtype=np.int32)
    supported, intact = evaluate_fragment_support(graph, rest.copy(), damage)
    if not np.all(supported) or not np.all(intact):
        raise AssertionError("intact fragment chain did not reach the foundation")

    damaged = damage.copy(); damaged[2] = 1.0
    supported, intact = evaluate_fragment_support(graph, rest.copy(), damaged)
    if supported.tolist() != [True, True, False] or int(np.count_nonzero(intact)) != 1:
        raise AssertionError(f"failed boundary did not detach upper fragment: {supported}, {intact}")

    displaced = rest.copy(); displaced[2, 0] += 2.0
    supported, intact = evaluate_fragment_support(graph, displaced, damage, maximum_stretch=1.6)
    if supported.tolist() != [True, True, False] or int(np.count_nonzero(intact)) != 1:
        raise AssertionError("overstretched boundary remained a valid load path")

    opening = rest.copy(); opening[2, 0] += 0.06
    edge_energy, fragment_energy = evaluate_fragment_fracture_energy(
        graph, opening, damage, material, role
    )
    supported, intact = evaluate_fragment_support(graph, opening, damage, maximum_stretch=1.6)
    if not np.all(intact) or not (0.0 < edge_energy[1] < 1.0):
        raise AssertionError(
            f"subcritical boundary opening did not create a progressive crack: {edge_energy}"
        )
    healed_edge, healed_fragment = evaluate_fragment_fracture_energy(
        graph, rest.copy(), damage, material, role, edge_energy
    )
    if not np.array_equal(healed_edge, edge_energy) or np.any(healed_fragment < fragment_energy):
        raise AssertionError("fracture energy healed after elastic unloading")
    failed_edge, _ = evaluate_fragment_fracture_energy(
        graph, rest.copy(), damaged, material, role, healed_edge
    )
    if failed_edge[1] < 0.99:
        raise AssertionError("fully damaged boundary did not retain full fracture energy")
    print(
        "PASS: support is causal; subcritical tensile energy opens an irreversible crack "
        "and failed boundaries retain full fracture energy"
    )


if __name__ == "__main__":
    main()
