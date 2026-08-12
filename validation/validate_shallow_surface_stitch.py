"""Regression for the closed shallow-water volume and narrow SPH seam."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from deluge_v3 import load_run_config  # noqa: E402
from simulation.shallow_water import ShallowWaterFarField  # noqa: E402


def main() -> None:
    cfg = load_run_config(ROOT / "config_v3_city_clear_surge_30s_x1_7.json")
    policy = cfg["v3"]["shallow_water"]
    if bool(policy.get("render_far_surface", True)):
        raise AssertionError("city preset exposes the reduced-order far-water volume")
    # The closed representation remains available for dedicated offshore
    # views, while the city profile deliberately renders only its narrow seam.
    policy["render_far_surface"] = True
    wp.init()
    device = cfg.get("device", "cuda:0")
    shallow = ShallowWaterFarField(cfg, device)

    # Exaggerate the 2-D column and provide a deliberately lower SPH surface.
    # The seam must end at the SPH height instead of retaining a floating
    # shallow-water slab many metres above it.
    state = shallow.state.numpy()
    state[:, :, 0] = 25.0
    shallow.state = wp.array(state, dtype=wp.vec3, device=device)
    xs = np.arange(
        shallow.lower_x + 0.5,
        shallow.lower_x + shallow.nx * shallow.cell_size,
        1.0,
        dtype=np.float32,
    )
    sph = np.asarray(
        [
            (x, y, shallow.interface_z + z)
            for x in xs
            for y in (8.7, 9.0)
            for z in (0.5, 1.5)
        ],
        dtype=np.float32,
    )
    stitch, _ = shallow.stitched_surface_samples(sph)
    unique_x = np.unique(stitch[:, 0])
    unique_z = np.unique(stitch[:, 2])
    rows = stitch.reshape(len(unique_z), len(unique_x), 3)
    if float(np.median(rows[-1, :, 1])) > 10.0:
        raise AssertionError("transition surface remained attached to the taller shallow column")
    maximum_span = (
        2.0 * float(policy["coupling_width"])
        + float(policy["stitch_overlap_m"])
        + 2.0 * float(policy["surface_sample_spacing"])
    )
    if float(np.ptp(unique_z)) > maximum_span:
        raise AssertionError("stitch samples still cover the complete far-water domain")

    vertices, indices = shallow.surface_mesh()
    triangles = indices.reshape(-1, 3)
    edges = np.concatenate(
        [triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]], axis=0
    )
    edges.sort(axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    if not np.all(incidence == 2):
        raise AssertionError("shallow-water render mesh is not a closed manifold")
    surface_count = len(vertices) // 2
    top = vertices[:surface_count]
    bed = vertices[surface_count:]
    if not np.allclose(top[:, (0, 2)], bed[:, (0, 2)]):
        raise AssertionError("top and bed of shallow-water volume are misregistered")
    if float(np.min(top[:, 1] - bed[:, 1])) < -1.0e-6:
        raise AssertionError("shallow-water top fell below its render bed")
    if float(np.max(top[:, 2])) < float(np.min(unique_z)):
        raise AssertionError("closed far volume does not overlap the SPH transition seam")

    print(
        "PASS: shallow water is a closed volume; the narrow seam overlaps it "
        f"and converges to SPH height ({len(vertices):,} vertices, "
        f"{len(stitch):,} stitch samples)"
    )


if __name__ == "__main__":
    main()
