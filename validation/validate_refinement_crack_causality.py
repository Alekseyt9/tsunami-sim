"""Regression: adaptive structural LOD must not create facade cracks."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np

from hybrid_model import (
    FragmentSupportGraph,
    evaluate_fragment_fracture_energy,
    rebaseline_fragment_support_graph,
)


def main() -> None:
    # One representative inter-fragment boundary sample.  Refinement retains
    # index 0 for a child but moves it in rest and world space by the same LOD
    # offset.  With the stale parent length this used to look fully fractured.
    graph = FragmentSupportGraph(
        edge_fragments=np.asarray([[0, 1]], dtype=np.int32),
        sample_offsets=np.asarray([0, 1], dtype=np.int32),
        sample_pairs=np.asarray([[0, 1]], dtype=np.int32),
        sample_rest_length=np.asarray([1.0], dtype=np.float32),
        anchored_fragments=np.asarray([True, False]),
    )
    refined_rest = np.asarray([[-0.52, 0.0, 0.0], [1.52, 0.0, 0.0]], dtype=np.float32)
    refined_world = refined_rest.copy()
    damage = np.zeros(2, dtype=np.float32)
    material = np.ones(2, dtype=np.int32)
    role = np.full(2, 2, dtype=np.int32)

    stale_energy, _ = evaluate_fragment_fracture_energy(
        graph, refined_world, damage, material, role
    )
    if float(stale_energy[0]) < 0.99:
        raise AssertionError("test fixture no longer reproduces the stale-length crack")

    rebased = rebaseline_fragment_support_graph(graph, refined_rest)
    corrected_energy, _ = evaluate_fragment_fracture_energy(
        rebased, refined_world, damage, material, role
    )
    if float(corrected_energy[0]) != 0.0:
        raise AssertionError(
            f"rest-preserving refinement created crack energy {corrected_energy[0]:.6f}"
        )
    if rebased.edge_fragments is not graph.edge_fragments:
        raise AssertionError("rebaseline unexpectedly rebuilt edge topology")
    print(
        "PASS: a 1.04 m refinement offset changes the sparse rest length but "
        "creates zero physical crack energy"
    )


if __name__ == "__main__":
    main()
