"""CUDA regression for anisotropic water, 2048 shadows and indirect light."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import json
from pathlib import Path

import numpy as np
import warp as wp

from kernels.base import apply_screen_space_indirect_lighting, clear_depth  # noqa: E402
from kernels.hybrid import (  # noqa: E402
    apply_cascaded_shadow_maps,
    raster_facade_shadow_depth,
)
from kernels.surface import splat_anisotropic_surface_field  # noqa: E402


HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    cfg = json.loads((HERE / "config_v3_rtx5070.json").read_text(encoding="utf-8"))
    shadows = cfg["render"]["cascaded_shadows"]
    sun_direction = np.asarray(cfg["render"]["physical_sky"]["sun_direction"], dtype=np.float32)
    sun_direction /= np.linalg.norm(sun_direction)
    building_z = sorted(float(building[1]) for building in cfg["buildings"])
    row_groups: list[list[float]] = []
    for z in building_z:
        if not row_groups or z - row_groups[-1][-1] > 10.0:
            row_groups.append([z])
        else:
            row_groups[-1].append(z)
    row_centres = np.asarray([np.mean(group) for group in row_groups], dtype=np.float32)
    row_gap = float(np.median(np.diff(row_centres)))
    median_height = float(np.median([building[4] for building in cfg["buildings"]]))
    longitudinal_shadow_reach = median_height * abs(float(sun_direction[2])) / max(
        float(sun_direction[1]), 1.0e-6
    )
    if longitudinal_shadow_reach <= row_gap:
        raise AssertionError(
            "production sun cannot cast one building row onto the next: "
            f"reach={longitudinal_shadow_reach:.1f} m, gap={row_gap:.1f} m"
        )
    if int(shadows["cascade_count"]) != 4 or int(shadows["resolution"]) != 2048:
        raise AssertionError("production shadows are not four 2048 cascades")
    if len(shadows["splits_m"]) != 4 or shadows.get("pcf") != "5x5":
        raise AssertionError("four-cascade split/PCF configuration is incomplete")
    views = cfg["render"]["views"]
    if int(views["original"]["shadow_resolution"]) != 2048:
        raise AssertionError("hero view lost its 2048 shadow maps")
    if any(int(views[name]["shadow_resolution"]) < 1024 for name in ("front", "side", "top")):
        raise AssertionError("inset shadow maps dropped below 1024")

    # A window-heavy tower must still cast a building-scale direct shadow.
    # This exercises the actual raster/lookup kernels rather than only checking
    # that the JSON says shadows are enabled.
    shadow_resolution = 64
    blocker_host = np.asarray(
        ((-2.0, 5.0, -2.0), (2.0, 5.0, -2.0), (2.0, 5.0, 2.0), (-2.0, 5.0, 2.0)),
        dtype=np.float32,
    )
    blocker = wp.array(blocker_host, dtype=wp.vec3, device=device)
    blocker_anchor = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    blocker_damage = wp.zeros(4, dtype=float, device=device)
    shadow_depth = wp.empty(shadow_resolution * shadow_resolution, dtype=float, device=device)
    wp.launch(clear_depth, dim=len(shadow_depth), inputs=[shadow_depth], device=device)
    wp.launch(
        raster_facade_shadow_depth,
        dim=2,
        inputs=[
            blocker,
            blocker,
            blocker_anchor,
            wp.array(np.asarray((20,), dtype=np.int32), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.array(np.asarray((-1,), dtype=np.int32), dtype=wp.int32, device=device),
            wp.ones(1, dtype=float, device=device),
            blocker_damage,
            shadow_depth,
            0,
            wp.vec3(0.0, 10.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
            wp.vec3(0.0, -1.0, 0.0),
            wp.vec2(10.0, 10.0),
            shadow_resolution,
            2.0,
        ],
        device=device,
    )
    # A road is a receiver, not an occluding copy of itself.
    road_shadow = wp.empty(shadow_resolution * shadow_resolution, dtype=float, device=device)
    wp.launch(clear_depth, dim=len(road_shadow), inputs=[road_shadow], device=device)
    wp.launch(
        raster_facade_shadow_depth,
        dim=2,
        inputs=[
            blocker,
            blocker,
            blocker_anchor,
            wp.array(np.asarray((90,), dtype=np.int32), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.array(np.asarray((-1,), dtype=np.int32), dtype=wp.int32, device=device),
            wp.ones(1, dtype=float, device=device),
            blocker_damage,
            road_shadow,
            0,
            wp.vec3(0.0, 10.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
            wp.vec3(0.0, -1.0, 0.0),
            wp.vec2(10.0, 10.0),
            shadow_resolution,
            2.0,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    if np.any(road_shadow.numpy() < 1.0e8):
        raise AssertionError("terrain/road receiver polluted the shadow map")
    shadow_width = shadow_height = 9
    shadow_pixels = shadow_width * shadow_height
    receiver_depth_host = np.full(shadow_pixels, 1.0e9, dtype=np.float32)
    receiver_depth_host[4 * shadow_width + 4] = 10.0
    receiver_normal_host = np.zeros((shadow_pixels, 3), dtype=np.float32)
    receiver_normal_host[4 * shadow_width + 4] = (0.0, 1.0, 0.0)
    receiver_color_host = np.ones((shadow_pixels, 3), dtype=np.float32)
    receiver_color = wp.array(receiver_color_host, dtype=wp.vec3, device=device)
    wp.launch(
        apply_cascaded_shadow_maps,
        dim=shadow_pixels,
        inputs=[
            wp.array(receiver_depth_host, dtype=float, device=device),
            wp.array(np.full(shadow_pixels, 1.0e9, dtype=np.float32), dtype=float, device=device),
            wp.array(receiver_normal_host, dtype=wp.vec3, device=device),
            receiver_color,
            wp.vec3(0.0, 0.0, -10.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
            8.0,
            shadow_width,
            shadow_height,
            shadow_depth,
            wp.array(np.asarray(((0.0, 10.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device),
            wp.array(np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device),
            wp.array(np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32), dtype=wp.vec3, device=device),
            wp.array(np.asarray(((0.0, -1.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device),
            wp.array(np.asarray(((10.0, 10.0),), dtype=np.float32), dtype=wp.vec2, device=device),
            wp.array(np.asarray((20.0,), dtype=np.float32), dtype=float, device=device),
            1,
            shadow_resolution,
            float(shadows["strength"]),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    shadowed = receiver_color.numpy()[4 * shadow_width + 4]
    if float(np.max(shadowed)) > 0.72:
        raise AssertionError(f"glass facade did not cast a visible cascade shadow: {shadowed}")

    shape = (25, 25, 25)
    voxel = 0.30
    lower = np.asarray((-3.6, -3.6, -3.6), dtype=np.float32)
    x = wp.array(np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device)
    v = wp.array(np.asarray(((10.0, 0.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device)
    radius = wp.array(np.asarray((0.5,), dtype=np.float32), dtype=float, device=device)
    kind = wp.zeros(1, dtype=wp.int32, device=device)
    mask = wp.ones(1, dtype=wp.int32, device=device)
    normal = wp.array(np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32), dtype=wp.vec3, device=device)
    field = wp.zeros(shape, dtype=float, device=device)
    wp.launch(
        splat_anisotropic_surface_field,
        dim=1,
        inputs=[x, v, radius, kind, mask, normal, field, wp.vec3(*lower), voxel,
                *shape, 1.72, 1.38, 0.82, 0.025],
        device=device,
    )
    wp.synchronize_device(device)
    values = field.numpy()
    coordinates = lower[0] + np.arange(shape[0], dtype=np.float32) * voxel
    total = float(values.sum())
    variances = []
    for axis in range(3):
        marginal = values.sum(axis=tuple(index for index in range(3) if index != axis))
        mean = float(np.sum(marginal * coordinates) / total)
        variances.append(float(np.sum(marginal * np.square(coordinates - mean)) / total))
    if not (variances[0] > variances[2] * 1.08 and variances[2] > variances[1] * 1.20):
        raise AssertionError(f"water kernel is not flow-aligned and surface-flat: {variances}")

    width = height = 9
    count = width * height
    source_host = np.zeros((count, 3), dtype=np.float32)
    source_host[:] = (0.04, 0.04, 0.04)
    center = 4 * width + 4
    emitter = 4 * width + 6
    source_host[emitter] = (1.2, 0.12, 0.06)
    depth_host = np.full(count, 1.0e9, dtype=np.float32)
    depth_host[center] = 5.0
    depth_host[emitter] = 5.0
    normal_host = np.zeros((count, 3), dtype=np.float32)
    normal_host[center] = (1.0, 0.0, 0.0)
    normal_host[emitter] = (-1.0, 0.0, 0.0)
    source = wp.array(source_host, dtype=wp.vec3, device=device)
    scene_depth = wp.array(depth_host, dtype=float, device=device)
    water_depth = wp.array(np.full(count, 1.0e9, dtype=np.float32), dtype=float, device=device)
    normals = wp.array(normal_host, dtype=wp.vec3, device=device)
    output = wp.empty(count, dtype=wp.vec3, device=device)
    wp.launch(
        apply_screen_space_indirect_lighting,
        dim=count,
        inputs=[source, scene_depth, water_depth, normals, output,
                wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0),
                8.0, width, height, 0.20, 4],
        device=device,
    )
    wp.synchronize_device(device)
    indirect = output.numpy()[center]
    if indirect[0] <= source_host[center, 0] + 0.05 or indirect[0] <= indirect[1] * 1.8:
        raise AssertionError(f"nearby red surface did not produce bounded colour bleeding: {indirect}")
    if not np.all(np.isfinite(output.numpy())):
        raise AssertionError("indirect-lighting pass produced non-finite radiance")

    print(
        "PASS: anisotropic SDF is flow-aligned/normal-thin, indirect colour bounce works, "
        "and tinted glass casts a visible four-cascade PCF shadow"
    )


if __name__ == "__main__":
    main()
