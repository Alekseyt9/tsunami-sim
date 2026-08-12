"""GPU regression for bounded, finite-lifetime facade glass micro-shards."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import update_glass_micro_shards, update_glass_shatter_state  # noqa: E402


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    vertex_host = np.asarray(
        ((0.0, 2.0, 0.0), (0.0, 4.0, 0.0), (2.0, 4.0, 0.0), (2.0, 2.0, 0.0)),
        dtype=np.float32,
    )
    vertex = wp.array(vertex_host, dtype=wp.vec3, device=device)
    previous = wp.array(vertex_host - np.asarray((0.0, 0.0, 0.05), dtype=np.float32), dtype=wp.vec3, device=device)
    panels = wp.array(np.asarray((0,), dtype=np.int32), dtype=wp.int32, device=device)
    anchor = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    damage = wp.ones(4, dtype=float, device=device)
    birth = wp.array(np.asarray((-1.0,), dtype=np.float32), dtype=float, device=device)
    origin = wp.zeros(1, dtype=wp.vec3, device=device)
    basis_u = wp.zeros(1, dtype=wp.vec3, device=device)
    basis_v = wp.zeros(1, dtype=wp.vec3, device=device)
    velocity = wp.zeros(1, dtype=wp.vec3, device=device)
    shards_per_panel = 4
    shard_count = shards_per_panel
    shard_vertex = wp.zeros(shard_count * 3, dtype=wp.vec3, device=device)
    previous_shard_vertex = wp.zeros(shard_count * 3, dtype=wp.vec3, device=device)
    active = wp.zeros(shard_count, dtype=wp.int32, device=device)

    wp.launch(
        update_glass_shatter_state, dim=1,
        inputs=[panels, vertex, previous, anchor, damage, birth, origin, basis_u, basis_v,
                velocity, 5.0, 1.0 / 24.0, 0.42, 0, 3.2], device=device,
    )
    common = [
        panels, birth, origin, basis_u, basis_v, velocity, shard_vertex,
        previous_shard_vertex, active, shards_per_panel,
    ]
    wp.launch(
        update_glass_micro_shards, dim=shard_count,
        inputs=[*common, 5.5, 1.0 / 24.0, 1.4, 3.2, 2.2, 0.065, 1.0], device=device,
    )
    wp.synchronize_device(device)
    live = int(np.count_nonzero(active.numpy()))
    if live != shard_count:
        raise AssertionError(f"newly shattered pane produced {live}/{shard_count} live shards")
    if len(shard_vertex) != shards_per_panel * 3:
        raise AssertionError("glass shard storage is not fixed-capacity")

    wp.launch(
        update_glass_micro_shards, dim=shard_count,
        inputs=[*common, 8.3, 1.0 / 24.0, 1.4, 3.2, 2.2, 0.065, 1.0], device=device,
    )
    wp.synchronize_device(device)
    expired = int(np.count_nonzero(active.numpy()))
    if expired != 0:
        raise AssertionError(f"{expired} micro-shards survived their maximum lifetime")

    # Renderer state is intentionally not part of the multi-gigabyte physics
    # checkpoint. On resume, already-broken panes are marked as expired rather
    # than spuriously emitting another city-wide burst of glass.
    resume_birth = wp.array(np.asarray((-1.0,), dtype=np.float32), dtype=float, device=device)
    wp.launch(
        update_glass_shatter_state, dim=1,
        inputs=[panels, vertex, previous, anchor, damage, resume_birth, origin, basis_u, basis_v,
                velocity, 12.0, 1.0 / 24.0, 0.42, 1, 3.2], device=device,
    )
    resume_common = [
        panels, resume_birth, origin, basis_u, basis_v, velocity, shard_vertex,
        previous_shard_vertex, active, shards_per_panel,
    ]
    wp.launch(
        update_glass_micro_shards, dim=shard_count,
        inputs=[*resume_common, 12.1, 1.0 / 24.0, 1.4, 3.2, 2.2, 0.065, 1.0], device=device,
    )
    wp.synchronize_device(device)
    if int(np.count_nonzero(active.numpy())) != 0:
        raise AssertionError("checkpoint resume re-emitted already expired glass shards")
    print(
        "PASS: a shattered pane uses four preallocated GPU micro-shards; "
        "all are retired by 3.3 seconds, and resume does not re-emit them"
    )


if __name__ == "__main__":
    main()
