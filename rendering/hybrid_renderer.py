"""V3 facade renderer: deformable quads instead of solid particle circles."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import warp as wp

from kernels.base import (
    apply_directional_screen_shadows,
    bilateral_depth_axis,
    apply_cinematic_postprocess,
    apply_screen_space_indirect_lighting,
    clear_depth,
    clear_gbuffer,
    clear_render,
    clear_scalar,
    composite_surface_foam,
    composite_volumetric_atmosphere,
    copy_vec3,
    filmic_tonemap_color,
    raster_water_depth,
    render_physical_sky,
    shade_water_surface,
    temporal_stabilize_water_depth,
    temporal_antialias_color,
)
from rendering.renderer import ParticleRenderer
from kernels.hybrid import (
    apply_cascaded_shadow_maps,
    deform_facade_vertices,
    raster_facade_color,
    raster_facade_depth,
    raster_facade_shadow_depth,
)
from kernels.surface import raster_anisotropic_water_depth, raster_water_mesh_depth


class HybridRenderer(ParticleRenderer):
    def __init__(self, width: int, height: int, camera: dict, device: str, skin_path: Path,
                 view_name: str = "main", maximum_panel_stretch: float = 1.8,
                 water_tangent_scale: float = 2.8, water_normal_scale: float = 2.45,
                 crack_strength: float = 1.0,
                 architectural_overlay_tolerance: float = 0.9,
                 water_absorption_scale: float = 1.0,
                 water_refraction_strength: float = 8.0,
                 temporal_history_weight: float = 0.78,
                 temporal_disocclusion_threshold: float = 0.75,
                 foam_strength: float = 0.72,
                 fog_density: float = 0.0016,
                 fog_height_falloff: float = 52.0,
                 mist_strength: float = 2.2,
                 taa_history_weight: float = 0.86,
                 cascaded_shadows: bool = True,
                 shadow_resolution: int = 512,
                 shadow_splits: tuple[float, float, float] = (90.0, 220.0, 480.0),
                 shadow_strength: float = 0.78,
                 indirect_lighting_enabled: bool = True,
                 indirect_lighting_strength: float = 0.16,
                 indirect_lighting_radius_pixels: int = 14,
                 hdr_enabled: bool = True,
                 physical_sky_enabled: bool = True,
                 hdr_exposure_ev: float = -0.35,
                 bloom_threshold: float = 1.15,
                 bloom_strength: float = 0.11,
                 sky_turbidity: float = 3.2,
                 sky_intensity: float = 1.0,
                 sun_direction: tuple[float, float, float] = (-0.38, 0.82, -0.35),
                 sun_intensity: float = 3.1,
                 ibl_strength: float = 0.72,
                 water_absorption: tuple[float, float, float] = (0.17, 0.045, 0.018),
                 water_scattering: tuple[float, float, float] = (0.012, 0.032, 0.055),
                 water_phase_g: float = 0.35,
                 water_maximum_optical_depth: float = 18.0):
        super().__init__(width, height, camera, device)
        self.view_name = view_name
        self.maximum_panel_stretch = float(maximum_panel_stretch)
        self.architectural_overlay_tolerance = float(architectural_overlay_tolerance)
        self.water_tangent_scale = float(water_tangent_scale)
        self.water_normal_scale = float(water_normal_scale)
        self.crack_strength = max(0.0, float(crack_strength))
        self.water_absorption_scale = max(0.0, float(water_absorption_scale))
        self.water_refraction_strength = max(0.0, float(water_refraction_strength))
        self.temporal_history_weight = max(0.0, min(float(temporal_history_weight), 0.94))
        self.temporal_disocclusion_threshold = max(0.05, float(temporal_disocclusion_threshold))
        self.foam_strength = max(0.0, float(foam_strength))
        self.fog_density = max(0.0, float(fog_density))
        self.fog_height_falloff = max(1.0, float(fog_height_falloff))
        self.mist_strength = max(0.0, float(mist_strength))
        self.taa_history_weight = max(0.0, min(float(taa_history_weight), 0.94))
        self.cascaded_shadows = bool(cascaded_shadows)
        self.shadow_resolution = max(128, int(shadow_resolution))
        self.shadow_strength = max(0.0, min(float(shadow_strength), 1.0))
        self.indirect_lighting_enabled = bool(indirect_lighting_enabled)
        self.indirect_lighting_strength = max(0.0, min(float(indirect_lighting_strength), 0.45))
        self.indirect_lighting_radius_pixels = max(2, int(indirect_lighting_radius_pixels))
        self.hdr_enabled = bool(hdr_enabled)
        self.physical_sky_enabled = bool(physical_sky_enabled)
        self.hdr_exposure_ev = float(hdr_exposure_ev)
        self.bloom_threshold = max(0.0, float(bloom_threshold))
        self.bloom_strength = max(0.0, float(bloom_strength))
        self.sky_turbidity = max(1.0, float(sky_turbidity))
        self.sky_intensity = max(0.0, float(sky_intensity))
        sun = np.asarray(sun_direction, dtype=np.float32)
        sun /= max(float(np.linalg.norm(sun)), 1.0e-6)
        self.sun_direction = sun
        self.sun_intensity = max(0.0, float(sun_intensity))
        self.ibl_strength = max(0.0, float(ibl_strength))
        self.water_absorption = tuple(max(0.0, float(v)) for v in water_absorption)
        self.water_scattering = tuple(max(0.0, float(v)) for v in water_scattering)
        self.water_phase_g = max(-0.85, min(float(water_phase_g), 0.85))
        self.water_maximum_optical_depth = max(0.1, float(water_maximum_optical_depth))
        pixel_count = width * height
        self.water_history = wp.empty(pixel_count, dtype=float, device=device)
        self.water_temporal = wp.empty(pixel_count, dtype=float, device=device)
        wp.launch(clear_depth, dim=pixel_count, inputs=[self.water_history], device=device)
        self.water_history_valid = False
        self.gbuffer_normal = wp.empty(pixel_count, dtype=wp.vec3, device=device)
        self.gbuffer_motion = wp.empty(pixel_count, dtype=wp.vec2, device=device)
        self.gbuffer_material = wp.empty(pixel_count, dtype=wp.int32, device=device)
        self.gbuffer_roughness = wp.empty(pixel_count, dtype=float, device=device)
        self.gbuffer_metallic = wp.empty(pixel_count, dtype=float, device=device)
        self.taa_history_color = wp.empty(pixel_count, dtype=wp.vec3, device=device)
        self.taa_history_depth = wp.empty(pixel_count, dtype=float, device=device)
        self.taa_output = wp.empty(pixel_count, dtype=wp.vec3, device=device)
        self.display_color = wp.empty(pixel_count, dtype=wp.vec3, device=device)
        self.lighting_source = wp.empty(pixel_count, dtype=wp.vec3, device=device)
        wp.launch(clear_depth, dim=pixel_count, inputs=[self.taa_history_depth], device=device)
        self.taa_history_valid = False
        with np.load(skin_path, allow_pickle=False) as skin:
            rest_vertex = skin["vertex"].reshape(-1, 3).copy()
            anchor = skin["anchor"].reshape(-1).copy()
            material = skin["material"].copy()
            panel_mode = skin["panel_mode"].copy() if "panel_mode" in skin else np.zeros(len(material), dtype=np.int32)
            owner_fragment = skin["owner_fragment"].copy() if "owner_fragment" in skin else np.full(len(material), -1, dtype=np.int32)
        self.panel_count = len(material)
        self.rest_vertex = wp.array(rest_vertex, dtype=wp.vec3, device=device)
        self.current_vertex = wp.empty(len(rest_vertex), dtype=wp.vec3, device=device)
        self.previous_vertex = wp.array(rest_vertex, dtype=wp.vec3, device=device)
        self.anchor = wp.array(anchor, dtype=wp.int32, device=device)
        self.panel_material = wp.array(material, dtype=wp.int32, device=device)
        self.panel_mode = wp.array(panel_mode, dtype=wp.int32, device=device)
        self.owner_fragment = wp.array(owner_fragment, dtype=wp.int32, device=device)
        material_family = material // 10

        def triangle_order(mask: np.ndarray):
            panels = np.flatnonzero(mask).astype(np.int32, copy=False)
            order = np.empty(len(panels) * 2, dtype=np.int32)
            order[0::2] = panels * 2
            order[1::2] = panels * 2 + 1
            return wp.array(order, dtype=wp.int32, device=device)

        # Three compact index lists preserve material priority without making
        # every launch scan and reject the complete (potentially 600k+) skin.
        self.color_passes = (
            ("debris", triangle_order(panel_mode != 0)),
            ("opaque", triangle_order((panel_mode == 0) & (material_family != 2))),
            ("glass", triangle_order((panel_mode == 0) & (material_family == 2))),
        )
        fragment_count = int(owner_fragment[owner_fragment >= 0].max()) + 1 if np.any(owner_fragment >= 0) else 1
        self.fragment_support = wp.ones(fragment_count, dtype=float, device=device)
        self.fragment_fracture_energy = wp.zeros(fragment_count, dtype=float, device=device)
        self._initialize_shadow_cascades(tuple(float(value) for value in shadow_splits))

    def _initialize_shadow_cascades(self, splits: tuple[float, ...]) -> None:
        """Build stable orthographic sunlight volumes around camera-frustum slices."""
        splits_array = np.maximum.accumulate(np.asarray(splits, dtype=np.float32))
        cascade_count = min(4, len(splits_array))
        tan_half_fov = 0.5 * float(self.height) / max(float(self.focal), 1.0e-6)
        aspect = float(self.width) / max(float(self.height), 1.0)
        sun = np.asarray(self.sun_direction, dtype=np.float32)
        sun /= np.linalg.norm(sun)
        light_forward = -sun
        light_right = np.cross(light_forward, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
        light_right /= np.linalg.norm(light_right)
        light_up = np.cross(light_right, light_forward)
        light_up /= np.linalg.norm(light_up)
        origins: list[np.ndarray] = []
        rights: list[np.ndarray] = []
        ups: list[np.ndarray] = []
        forwards: list[np.ndarray] = []
        extents: list[tuple[float, float]] = []
        near = 1.0
        for far in splits_array[:cascade_count]:
            corners = []
            for depth in (near, float(far)):
                half_height = depth * tan_half_fov
                half_width = half_height * aspect
                center = self.cam + self.forward * depth
                for sy in (-1.0, 1.0):
                    for sx in (-1.0, 1.0):
                        corners.append(center + self.right * (sx * half_width) + self.up * (sy * half_height))
            corners_array = np.asarray(corners, dtype=np.float32)
            center = np.mean(corners_array, axis=0)
            relative = corners_array - center[None, :]
            half_x = float(np.max(np.abs(relative @ light_right))) + 24.0
            half_y = float(np.max(np.abs(relative @ light_up))) + 24.0
            # Square, texel-snapped cascades reduce shimmering as geometry moves.
            half_extent = max(half_x, half_y)
            texel = 2.0 * half_extent / float(self.shadow_resolution)
            light_x = round(float(np.dot(center, light_right)) / texel) * texel
            light_y = round(float(np.dot(center, light_up)) / texel) * texel
            snapped_center = center + light_right * (light_x - float(np.dot(center, light_right)))
            snapped_center += light_up * (light_y - float(np.dot(center, light_up)))
            origins.append((snapped_center - light_forward * 420.0).astype(np.float32))
            rights.append(light_right.copy())
            ups.append(light_up.copy())
            forwards.append(light_forward.copy())
            extents.append((half_extent, half_extent))
            near = float(far)
        self.shadow_cascade_count = cascade_count
        self.shadow_origins_host = np.asarray(origins, dtype=np.float32)
        self.shadow_rights_host = np.asarray(rights, dtype=np.float32)
        self.shadow_ups_host = np.asarray(ups, dtype=np.float32)
        self.shadow_forwards_host = np.asarray(forwards, dtype=np.float32)
        self.shadow_extents_host = np.asarray(extents, dtype=np.float32)
        self.shadow_origins = wp.array(self.shadow_origins_host, dtype=wp.vec3, device=self.device)
        self.shadow_rights = wp.array(self.shadow_rights_host, dtype=wp.vec3, device=self.device)
        self.shadow_ups = wp.array(self.shadow_ups_host, dtype=wp.vec3, device=self.device)
        self.shadow_forwards = wp.array(self.shadow_forwards_host, dtype=wp.vec3, device=self.device)
        self.shadow_extents = wp.array(self.shadow_extents_host, dtype=wp.vec2, device=self.device)
        self.shadow_far_splits = wp.array(splits_array[:cascade_count], dtype=float, device=self.device)
        self.shadow_depth = wp.empty(
            cascade_count * self.shadow_resolution * self.shadow_resolution,
            dtype=float,
            device=self.device,
        )

    def render(self, arrays: dict, count: int, output_path: Path | None, frame: int, time_s: float, stats: dict):
        pixel_count = self.width * self.height
        wp.launch(clear_render, dim=pixel_count, inputs=[self.depth, self.color, self.width, self.height], device=self.device)
        if self.physical_sky_enabled:
            wp.launch(
                render_physical_sky, dim=pixel_count,
                inputs=[
                    self.color, wp.vec3(*self.right), wp.vec3(*self.up), wp.vec3(*self.forward),
                    self.focal, self.width, self.height, wp.vec3(*self.sun_direction),
                    self.sky_turbidity, self.sky_intensity, self.sun_intensity,
                ], device=self.device,
            )
        wp.launch(clear_depth, dim=pixel_count, inputs=[self.water_depth], device=self.device)
        wp.launch(clear_scalar, dim=pixel_count, inputs=[self.water_back_depth, 0.0], device=self.device)
        wp.launch(clear_scalar, dim=pixel_count, inputs=[self.water_foam, 0.0], device=self.device)
        wp.launch(
            clear_gbuffer, dim=pixel_count,
            inputs=[self.gbuffer_normal, self.gbuffer_motion, self.gbuffer_material,
                    self.gbuffer_roughness, self.gbuffer_metallic], device=self.device,
        )
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
        if self.cascaded_shadows:
            wp.launch(
                clear_depth, dim=len(self.shadow_depth), inputs=[self.shadow_depth], device=self.device,
            )
            shadow_pixels = self.shadow_resolution * self.shadow_resolution
            for cascade in range(self.shadow_cascade_count):
                wp.launch(
                    raster_facade_shadow_depth, dim=self.panel_count * 2,
                    inputs=[
                        self.current_vertex, self.rest_vertex, self.panel_material,
                        self.panel_mode, self.owner_fragment, self.fragment_support,
                        self.shadow_depth, cascade * shadow_pixels,
                        wp.vec3(*self.shadow_origins_host[cascade]),
                        wp.vec3(*self.shadow_rights_host[cascade]),
                        wp.vec3(*self.shadow_ups_host[cascade]),
                        wp.vec3(*self.shadow_forwards_host[cascade]),
                        wp.vec2(*self.shadow_extents_host[cascade]), self.shadow_resolution,
                        self.maximum_panel_stretch,
                    ], device=self.device,
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
                        self.water_depth, self.water_back_depth, *common], device=self.device,
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
                        self.water_depth, self.water_back_depth, self.water_foam, *common,
                        self.water_tangent_scale, self.water_normal_scale], device=self.device,
            )
        elif "water_surface_mask" in arrays:
            wp.launch(
                raster_anisotropic_water_depth, dim=count,
                inputs=[arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count], arrays["kind"][:count],
                        arrays["water_surface_mask"][:count], arrays["water_surface_normal"][:count],
                        arrays["water_foam_strength"][:count], arrays["water_phase"][:count],
                        self.water_depth, self.water_back_depth, self.water_foam, *common,
                        self.water_tangent_scale, self.water_normal_scale], device=self.device,
            )
        else:
            wp.launch(
                raster_water_depth, dim=count,
                inputs=[arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count], arrays["kind"][:count],
                        self.water_depth, self.water_back_depth, self.water_foam, *common], device=self.device,
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
            temporal_stabilize_water_depth, dim=pixel_count,
            inputs=[smooth_source, self.water_foam, self.water_history, self.water_temporal,
                    int(self.water_history_valid), self.temporal_history_weight,
                    self.temporal_disocclusion_threshold], device=self.device,
        )
        smooth_source = self.water_temporal
        self.water_history_valid = True

        # Interior/cut surfaces first, then authored opaque panels, with glass
        # last.  Wall backing and its window are only 9 cm apart and both pass
        # the conservative depth tolerance.  Drawing all authored materials in
        # one launch therefore allowed the backing wall to race with and
        # randomly overwrite the window.  Ordered family passes make facade
        # layering deterministic without letting debris cover the windows.
        for _pass_name, triangle_order in self.color_passes:
            if len(triangle_order) == 0:
                continue
            wp.launch(
                raster_facade_color, dim=len(triangle_order),
                inputs=[self.current_vertex, self.previous_vertex, self.rest_vertex,
                        self.anchor, self.panel_material,
                        self.panel_mode, self.owner_fragment, self.fragment_support, arrays["damage"],
                        self.fragment_fracture_energy, triangle_order,
                        self.depth, self.color, self.gbuffer_normal, self.gbuffer_motion,
                        self.gbuffer_material, self.gbuffer_roughness, self.gbuffer_metallic,
                        *common, self.maximum_panel_stretch,
                        self.crack_strength, self.architectural_overlay_tolerance,
                        wp.vec3(*self.sun_direction), self.sky_turbidity, self.sky_intensity,
                        self.sun_intensity, self.ibl_strength], device=self.device,
            )
        wp.launch(
            shade_water_surface, dim=pixel_count,
            inputs=[
                smooth_source, self.water_back_depth, self.water_foam, self.depth, self.color,
                self.width, self.height, wp.vec3(*self.right), wp.vec3(*self.up),
                wp.vec3(*self.forward), self.focal, self.water_absorption_scale,
                self.water_refraction_strength, wp.vec3(*self.water_absorption),
                wp.vec3(*self.water_scattering), self.water_phase_g,
                self.water_maximum_optical_depth, wp.vec3(*self.sun_direction),
                self.sky_turbidity, self.sky_intensity, self.sun_intensity,
                self.ibl_strength,
            ], device=self.device,
        )
        wp.launch(
            composite_surface_foam, dim=pixel_count,
            inputs=[smooth_source, self.water_foam, self.depth, self.color,
                    self.width, self.height, float(time_s), self.foam_strength], device=self.device,
        )
        if self.cascaded_shadows:
            wp.launch(
                apply_cascaded_shadow_maps, dim=pixel_count,
                inputs=[
                    self.depth, smooth_source, self.gbuffer_normal, self.color,
                    wp.vec3(*self.cam), wp.vec3(*self.right), wp.vec3(*self.up), wp.vec3(*self.forward),
                    self.focal, self.width, self.height, self.shadow_depth,
                    self.shadow_origins, self.shadow_rights, self.shadow_ups, self.shadow_forwards,
                    self.shadow_extents, self.shadow_far_splits, self.shadow_cascade_count,
                    self.shadow_resolution, self.shadow_strength, wp.vec3(*self.sun_direction),
                ], device=self.device,
            )
        else:
            wp.launch(
                apply_directional_screen_shadows, dim=pixel_count,
                inputs=[
                    self.depth, smooth_source, self.color,
                    wp.vec3(*self.right), wp.vec3(*self.up), wp.vec3(*self.forward),
                    self.focal, self.width, self.height,
                ], device=self.device,
            )
        if self.indirect_lighting_enabled:
            wp.launch(
                copy_vec3, dim=pixel_count,
                inputs=[self.color, self.lighting_source], device=self.device,
            )
            wp.launch(
                apply_screen_space_indirect_lighting, dim=pixel_count,
                inputs=[
                    self.lighting_source, self.depth, smooth_source, self.gbuffer_normal,
                    self.color, wp.vec3(*self.cam), wp.vec3(*self.right), wp.vec3(*self.up),
                    wp.vec3(*self.forward), self.focal, self.width, self.height,
                    self.indirect_lighting_strength, self.indirect_lighting_radius_pixels,
                ], device=self.device,
            )
        wp.launch(
            apply_cinematic_postprocess, dim=pixel_count,
            inputs=[self.depth, smooth_source, self.color, self.width, self.height], device=self.device,
        )
        wp.launch(
            composite_volumetric_atmosphere, dim=pixel_count,
            inputs=[
                self.depth, smooth_source, self.water_foam, self.color,
                wp.vec3(*self.cam), wp.vec3(*self.right), wp.vec3(*self.up), wp.vec3(*self.forward),
                self.focal, self.width, self.height, float(time_s), self.fog_density,
                self.fog_height_falloff, self.mist_strength, wp.vec3(*self.sun_direction),
                self.sky_turbidity, self.sky_intensity, self.sun_intensity,
            ], device=self.device,
        )
        wp.launch(
            temporal_antialias_color, dim=pixel_count,
            inputs=[self.color, self.depth, smooth_source, self.water_foam,
                    self.gbuffer_motion, self.taa_history_color, self.taa_history_depth,
                    self.taa_output, self.width, self.height, int(self.taa_history_valid),
                    self.taa_history_weight], device=self.device,
        )
        self.taa_history_valid = True
        if self.hdr_enabled:
            wp.launch(
                filmic_tonemap_color, dim=pixel_count,
                inputs=[
                    self.taa_output, self.display_color, self.width, self.height,
                    self.hdr_exposure_ev, self.bloom_threshold, self.bloom_strength,
                ], device=self.device,
            )
        wp.launch(
            copy_vec3, dim=self.panel_count * 4,
            inputs=[self.current_vertex, self.previous_vertex], device=self.device,
        )
        wp.synchronize_device(self.device)

        resolved_color = self.display_color if self.hdr_enabled else self.taa_output
        rgb = resolved_color.numpy().reshape(self.height, self.width, 3)
        rgb = np.clip(np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, "RGB")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, compress_level=2)
        return np.asarray(image)
