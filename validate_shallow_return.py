"""End-to-end CUDA validation of conservative SPH-to-shallow return flow."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import warp as wp

from deluge_v3 import HybridDelugeSolver


HERE = Path(__file__).resolve().parent


@wp.kernel
def place_returning_particle(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    index: int,
    interface_z: float,
):
    position = x[index]
    x[index] = wp.vec3(position[0], 2.0, interface_z - 0.2)
    v[index] = wp.vec3(-1.25, 0.0, -2.0)


def combined_volume(solver: HybridDelugeSolver) -> float:
    kind = solver.arrays["kind"][:solver.count].numpy()
    volume = solver.arrays["volume"][:solver.count].numpy()
    return float(np.sum(volume[kind == 0], dtype=np.float64)) + float(
        solver.shallow_water.diagnostics()["shallow_water_volume_m3"]
    )


def main() -> None:
    wp.init()
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    cfg["render"]["progressive_fragment_seconds"] = 0.0
    with tempfile.TemporaryDirectory(prefix="deluge_return_") as temporary:
        solver = HybridDelugeSolver(cfg, Path(temporary))
        old_count = solver.count
        if int(solver.arrays["kind"][0:1].numpy()[0]) != 0:
            raise AssertionError("test expects the first scene particle to be water")
        successor_x = solver.arrays["x"][1:2].numpy()[0].copy()
        successor_fragment = int(solver.fragment_id[1:2].numpy()[0])
        first_renderer = next(iter(solver.renderers.values()))
        old_anchor = int(first_renderer.anchor[0:1].numpy()[0])
        particle_volume = float(solver.arrays["volume"][0:1].numpy()[0])
        particle_mass = float(solver.arrays["mass"][0:1].numpy()[0])
        before_volume = combined_volume(solver)
        before_momentum = solver.shallow_water.diagnostics()["shallow_water_momentum_z"]
        wp.launch(
            place_returning_particle, dim=1,
            inputs=[solver.arrays["x"], solver.arrays["v"], 0,
                    solver.shallow_water.interface_z], device=solver.device,
        )
        solver._merge_sph_interface_particles()
        wp.synchronize_device(solver.device)

        if solver.count != old_count - 1:
            raise AssertionError(f"return compaction removed {old_count - solver.count} particles, expected 1")
        if solver.shallow_water.merged_particles_total != 1:
            raise AssertionError("return-flow diagnostic did not count the merged particle")
        if abs(solver.shallow_water.merged_volume_total - particle_volume) > 1.0e-6:
            raise AssertionError("return-flow diagnostic recorded the wrong particle volume")
        after_volume = combined_volume(solver)
        volume_residual = abs(after_volume - before_volume)
        if volume_residual > 2.0e-3:
            raise AssertionError(f"combined return-flow volume residual is {volume_residual:.3e} m3")
        after_momentum = solver.shallow_water.diagnostics()["shallow_water_momentum_z"]
        momentum_residual = abs(
            (after_momentum - before_momentum) * float(cfg["rest_density"])
            - particle_mass * -2.0
        )
        if momentum_residual > 2.0:
            raise AssertionError(f"return-flow momentum residual is {momentum_residual:.3e} kg m/s")
        compacted_x = solver.arrays["x"][0:1].numpy()[0]
        if not np.allclose(compacted_x, successor_x, atol=1.0e-6):
            raise AssertionError("stable GPU compaction did not move particle 1 into slot 0")
        if int(solver.fragment_id[0:1].numpy()[0]) != successor_fragment:
            raise AssertionError("fragment IDs lost alignment during return-flow compaction")
        if len(solver.fragment_host) != solver.count:
            raise AssertionError("host fragment mapping was not resized after GPU compaction")
        if int(solver.fragment_id[solver.count:solver.count + 1].numpy()[0]) != -1:
            raise AssertionError("compacted fragment tail is not initialized for future water refinement")
        if int(solver.normal_axis[solver.count:solver.count + 1].numpy()[0]) != -1:
            raise AssertionError("compacted normal-axis tail is not initialized")
        if int(solver.time_active[solver.count:solver.count + 1].numpy()[0]) != 1:
            raise AssertionError("compacted multirate tail is not active for future emitted particles")
        new_anchor = int(first_renderer.anchor[0:1].numpy()[0])
        if new_anchor != old_anchor - 1:
            raise AssertionError(
                f"facade anchor was not remapped with particle compaction: {old_anchor}->{new_anchor}"
            )

        print(
            f"PASS: returned 1 particle / {particle_volume:.3f} m3; "
            f"volume residual={volume_residual:.3e} m3; "
            f"momentum residual={momentum_residual:.3e} kg m/s; "
            f"stable compaction {old_count:,}->{solver.count:,}"
        )


if __name__ == "__main__":
    main()
