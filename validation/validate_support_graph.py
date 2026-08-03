"""CPU regression for foundation-connected architectural fragment support."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from simulation.hybrid_model import (  # noqa: E402
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

    # An undirected graph can falsely support fragment 3 by travelling from
    # the foundation up to fragment 2, back down to 3, then upward again.
    # Directed gravity paths must stop at that downward edge.
    detour_graph = type(graph)(
        edge_fragments=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        sample_offsets=np.arange(5, dtype=np.int32),
        sample_pairs=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        sample_rest_length=np.full(4, 3.0, dtype=np.float32),
        anchored_fragments=np.asarray((True, False, False, False, False)),
    )
    detour_position = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 6.0, 0.0),
         (1.0, 3.0, 0.0), (1.0, 6.0, 0.0)), dtype=np.float32,
    )
    detour_center = detour_position.copy()
    detour_role = np.full(5, 2, dtype=np.int32)
    directed, _ = evaluate_fragment_support(
        detour_graph, detour_position, np.zeros(5, dtype=np.float32),
        maximum_stretch=2.0,
        fragment_rest_center=detour_center,
        fragment_role=detour_role,
    )
    if directed.tolist() != [True, True, True, False, False]:
        raise AssertionError(f"directed support admitted an up-down detour: {directed}")

    # A horizontal diaphragm cannot transfer load across an unlimited chain,
    # while many slightly offset vertical storeys must not consume that budget.
    lateral_graph = type(graph)(
        edge_fragments=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        sample_offsets=np.arange(5, dtype=np.int32),
        sample_pairs=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        sample_rest_length=np.full(4, 4.0, dtype=np.float32),
        anchored_fragments=np.asarray((True, False, False, False, False)),
    )
    lateral_position = np.asarray(
        ((0.0, 0.0, 0.0), (0.2, 3.0, 0.0), (0.4, 6.0, 0.0),
         (4.4, 6.0, 0.0), (8.4, 6.0, 0.0)), dtype=np.float32,
    )
    lateral_role = np.full(5, 1, dtype=np.int32)
    limited, _ = evaluate_fragment_support(
        lateral_graph, lateral_position, np.zeros(5, dtype=np.float32),
        maximum_stretch=2.0,
        fragment_rest_center=lateral_position,
        fragment_role=lateral_role,
        maximum_lateral_transfer=6.0,
    )
    if limited.tolist() != [True, True, True, True, False]:
        raise AssertionError(f"lateral load-transfer budget failed: {limited}")

    capacity_graph = type(graph)(
        edge_fragments=np.asarray(((0, 1),), dtype=np.int32),
        sample_offsets=np.asarray((0, 4), dtype=np.int32),
        sample_pairs=np.asarray(((0, 1), (2, 3), (4, 5), (6, 7)), dtype=np.int32),
        sample_rest_length=np.ones(4, dtype=np.float32),
        anchored_fragments=np.asarray((True, False)),
    )
    capacity_position = np.asarray(
        ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
         (2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0)), dtype=np.float32,
    )
    capacity_damage = np.zeros(8, dtype=np.float32)
    capacity_damage[[5, 7]] = 1.0
    capacity_support, capacity_edges = evaluate_fragment_support(
        capacity_graph, capacity_position, capacity_damage,
        minimum_intact_sample_fraction=0.25,
        fragment_rest_center=np.asarray(((1.5, 0.0, 0.0), (1.5, 1.0, 0.0)), dtype=np.float32),
        fragment_role=np.asarray((4, 4), dtype=np.int32),
        minimum_load_capacity_fraction=0.75,
    )
    if not bool(capacity_edges[0]) or capacity_support.tolist() != [True, False]:
        raise AssertionError(
            f"a half-destroyed boundary retained full load capacity: {capacity_support}"
        )
    print(
        "PASS: support is causal and directed upward; subcritical tensile energy opens "
        "an irreversible crack and failed boundaries retain full fracture energy"
    )


if __name__ == "__main__":
    main()
