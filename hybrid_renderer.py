"""V3 facade renderer: deformable quads instead of solid particle circles."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import warp as wp

from kernels import (
    bilateral_depth_axis,
    clear_depth,
    clear_render,
    clear_scalar,
    raster_water_depth,
    shade_water_surface,
)
from renderer import ParticleRenderer
from hybrid_kernels import deform_facade_vertices, raster_facade_color, raster_facade_depth
from surface_kernels import raster_anisotropic_water_depth, raster_water_mesh_depth


class HybridRenderer(ParticleRenderer):
    def __init__(self, width: int, height: int, camera: dict, device: str, skin_path: Path,
                 view_name: str = "main", maximum_panel_stretch: float = 1.8,
                 water_tangent_scale: float = 2.8, water_normal_scale: float = 2.45,
                 crack_strength: float = 1.0):
        super().__init__(width, height, camera, device)
        self.view_name = view_name
        self.maximum_panel_stretch = float(maximum_panel_stretch)
        self.water_tangent_scale = float(water_tangent_scale)
        self.water_normal_scale = float(water_normal_scale)
        self.crack_strength = max(0.0, float(crack_strength))
        with np.load(skin_path, allow_pickle=False) as skin:
            rest_vertex = skin["vertex"].reshape(-1, 3).copy()
            anchor = skin["anchor"].reshape(-1).copy()
            material = skin["material"].copy()
            panel_mode = skin["panel_mode"].copy() if "panel_mode" in skin else np.zeros(len(material), dtype=np.int32)
            owner_fragment = skin["owner_fragment"].copy() if "owner_fragment" in skin else np.full(len(material), -1, dtype=np.int32)
        self.panel_count = len(material)
        self.rest_vertex = wp.array(rest_vertex, dtype=wp.vec3, device=device)
        self.current_vertex = wp.empty(len(rest_vertex), dtype=wp.vec3, device=device)
        self.anchor = wp.array(anchor, dtype=wp.int32, device=device)
        self.panel_material = wp.array(material, dtype=wp.int32, device=device)
        self.panel_mode = wp.array(panel_mode, dtype=wp.int32, device=device)
        self.owner_fragment = wp.array(owner_fragment, dtype=wp.int32, device=device)
        fragment_count = int(owner_fragment[owner_fragment >= 0].max()) + 1 if np.any(owner_fragment >= 0) else 1
        self.fragment_support = wp.ones(fragment_count, dtype=float, device=device)
        self.fragment_fracture_energy = wp.zeros(fragment_count, dtype=float, device=device)

    def render(self, arrays: dict, count: int, output_path: Path | None, frame: int, time_s: float, stats: dict):
        pixel_count = self.width * self.height
        wp.launch(clear_render, dim=pixel_count, inputs=[self.depth, self.color, self.width, self.height], device=self.device)
        wp.launch(clear_depth, dim=pixel_count, inputs=[self.water_depth], device=self.device)
        wp.launch(clear_scalar, dim=pixel_count, inputs=[self.water_foam, 0.0], device=self.device)
        common = [
            wp.vec3(*self.cam), wp.vec3(*self.right), wp.vec3(*self.up), wp.vec3(*self.forward),
            self.focal, self.width, self.height,
        ]

        wp.launch(
            deform_facade_vertices, dim=self.panel_count * 4,
            inputs=[self.rest_vertex, self.anchor, self.panel_mode, self.owner_fragment,
                    self.fragment_support, arrays["x"], arrays["rest_x"], self.current_vertex],
            device=self.device,
        )
        wp.launch(
            raster_facade_depth, dim=self.panel_count * 2,
            inputs=[self.current_vertex, self.rest_vertex, self.panel_mode, self.owner_fragment,
                    self.fragment_support, self.depth, *common, self.maximum_panel_stretch], device=self.device,
        )
        if "water_mesh_indices" in arrays and len(arrays["water_mesh_indices"]) >= 3:
            wp.launch(
                raster_water_mesh_depth, dim=len(arrays["water_mesh_indices"]) // 3,
                inputs=[arrays["water_mesh_vertices"], arrays["water_mesh_indices"],
                        self.water_depth, *common], device=self.device,
            )
            # Marching cubes supplies a connected base volume.  Surface SPH
            # samples add only the nearest sub-voxel free-surface layer; atomic
            # depth composition closes thin/missing top regions without
            # exposing particle spheres after bilateral reconstruction.
            wp.launch(
                raster_anisotropic_water_depth, dim=count,
                inputs=[arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count], arrays["kind"][:count],
                        arrays["water_surface_mask"][:count], arrays["water_surface_normal"][:count],
                        arrays["water_foam_strength"][:count], arrays["water_phase"][:count],
                        self.water_depth, self.water_foam, *common,
                        self.water_tangent_scale, self.water_normal_scale], device=self.device,
            )
        elif "water_surface_mask" in arrays:
            wp.launch(
                raster_anisotropic_water_depth, dim=count,
                inputs=[arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count], arrays["kind"][:count],
                        arrays["water_surface_mask"][:count], arrays["water_surface_normal"][:count],
                        arrays["water_foam_strength"][:count], arrays["water_phase"][:count],
                        self.water_depth, self.water_foam, *common,
                        self.water_tangent_scale, self.water_normal_scale], device=self.device,
            )
        else:
            wp.launch(
                raster_water_depth, dim=count,
                inputs=[arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count], arrays["kind"][:count],
                        self.water_depth, self.water_foam, *common], device=self.device,
            )

        smooth_source, smooth_target = self.water_depth, self.water_temp
        # Sparse anisotropic samples need a wider reconstruction kernel than
        # the old all-particle splats.  The larger depth sigma removes lattice
        # banding while the bilateral term still stops at silhouettes.
        for _ in range(4):
            wp.launch(
                bilateral_depth_axis, dim=pixel_count,
                inputs=[smooth_source, smooth_target, self.width, self.height, 3.2, 1.8, 0], device=self.device,
            )
            smooth_source, smooth_target = smooth_target, smooth_source
            wp.launch(
                bilateral_depth_axis, dim=pixel_count,
                inputs=[smooth_source, smooth_target, self.width, self.height, 3.2, 1.8, 1], device=self.device,
            )
            smooth_source, smooth_target = smooth_target, smooth_source

        wp.launch(
            raster_facade_color, dim=self.panel_count * 2,
            inputs=[self.current_vertex, self.rest_vertex, self.anchor, self.panel_material,
                    self.panel_mode, self.owner_fragment, self.fragment_support, arrays["damage"],
                    self.fragment_fracture_energy,
                    self.depth, self.color, *common, self.maximum_panel_stretch,
                    self.crack_strength], device=self.device,
        )
        wp.launch(
            shade_water_surface, dim=pixel_count,
            inputs=[smooth_source, self.water_foam, self.depth, self.color, self.width, self.height], device=self.device,
        )
        wp.synchronize_device(self.device)

        rgb = self.color.numpy().reshape(self.height, self.width, 3)
        rgb = np.clip(np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, "RGB")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, compress_level=2)
        return np.asarray(image)
