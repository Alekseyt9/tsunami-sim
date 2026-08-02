"""Shared standalone CUDA particle solver used by DELUGE V3.

The solver deliberately prioritizes physical state and reproducible offline
frames over real-time playback. See README.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import warp as wp

from kernels import (
    clear_int, clear_vec3, compute_density, compute_fluid_forces, compute_solid_forces, count_damaged,
    integrate, refine_entering_fluid,
)
from renderer import ParticleRenderer, StreamingVideoWriter, encode_video
from scene import ParticleScene


def compose_quad_view(frames: dict[str, np.ndarray], order: list[str], width: int, height: int) -> np.ndarray:
    """Compose four synchronized RGB views into one 2x2 output frame."""
    if len(order) != 4 or any(name not in frames for name in order):
        raise ValueError("Quad layout requires exactly four named views")
    slot_width = width // 2
    slot_height = height // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for index, name in enumerate(order):
        rgb = frames[name]
        if rgb.shape[1] != slot_width or rgb.shape[0] != slot_height:
            rgb = np.asarray(Image.fromarray(rgb).resize((slot_width, slot_height), Image.Resampling.BILINEAR))
        x0 = (index % 2) * slot_width
        y0 = (index // 2) * slot_height
        canvas[y0:y0 + slot_height, x0:x0 + slot_width] = rgb
    # Thin separators make camera boundaries unambiguous after compression.
    canvas[max(0, slot_height - 1):min(height, slot_height + 1), :] = 8
    canvas[:, max(0, slot_width - 1):min(width, slot_width + 1)] = 8
    return canvas


class DelugeSolver:
    def __init__(self, cfg: dict, output: Path, resume: Path | None = None):
        self.cfg = cfg
        self.output = output
        self.frames_dir = output / "frames"
        self.checkpoint_dir = output / "checkpoints"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = cfg.get("device", "cuda:0")
        self.capacity = int(cfg["max_particles"])
        self.time = 0.0
        self.start_frame = 0

        if resume:
            initial = self.load_checkpoint(resume)
        else:
            scene = ParticleScene(rest_density=float(cfg["rest_density"]))
            scene.add_water(cfg)
            building_counts = scene.add_city(cfg)
            initial = scene.as_numpy()
            print(f"Water + solids: {len(initial['x']):,} particles; building lattices: {building_counts}")

        self.count = len(initial["x"])
        self.solid_count = int(np.count_nonzero(initial["kind"] != 0))
        if self.count > self.capacity:
            raise RuntimeError(f"Initial particle count {self.count:,} exceeds max_particles={self.capacity:,}")
        self.arrays = self.allocate(initial)
        group_ids = np.asarray(initial.get("fluid_group_id", ()), dtype=np.int32)
        next_group_id = int(group_ids[group_ids >= 0].max()) + 1 if np.any(group_ids >= 0) else 0
        self.fluid_group_counter = wp.array(
            np.asarray([next_group_id], dtype=np.int32), dtype=wp.int32, device=self.device
        )

        fluid_support = float(cfg.get("uniform_water_spacing", cfg["coarse_spacing"])) * 2.0
        # The widest structural bond is 3.2 * particle radius, while scene
        # particles use radius=0.48 * spacing.  A 1.55*spacing query therefore
        # keeps every possible bond (1.536*spacing) without scanning the old,
        # unnecessarily large 1.8*spacing neighbourhood on every substep.
        solid_support = float(cfg["solid_spacing"]) * float(cfg.get("solid_query_scale", 1.55))
        max_support = max(fluid_support, solid_support)
        self.max_support = max_support
        dim_x = max(16, int(float(cfg["domain_width"]) / max_support) + 8)
        dim_y = max(16, int(float(cfg["domain_y_max"]) / max_support) + 8)
        dim_z = max(16, int((float(cfg["domain_z_max"]) - float(cfg["reservoir_z_min"])) / max_support) + 8)
        self.grid = wp.HashGrid(dim_x, dim_y, dim_z, device=self.device)
        render = cfg["render"]
        self.renderer = ParticleRenderer(int(render["width"]), int(render["height"]), render["camera"], self.device)

    def allocate(self, initial: dict):
        arrays = {}
        specs = {
            "x": (wp.vec3, np.float32, (self.capacity, 3)),
            "v": (wp.vec3, np.float32, (self.capacity, 3)),
            "rest_x": (wp.vec3, np.float32, (self.capacity, 3)),
            "radius": (float, np.float32, (self.capacity,)),
            "mass": (float, np.float32, (self.capacity,)),
            "volume": (float, np.float32, (self.capacity,)),
            "kind": (wp.int32, np.int32, (self.capacity,)),
            "material": (wp.int32, np.int32, (self.capacity,)),
            "building_id": (wp.int32, np.int32, (self.capacity,)),
            "structural_class": (wp.int32, np.int32, (self.capacity,)),
            "fixed": (wp.int32, np.int32, (self.capacity,)),
            "damage": (float, np.float32, (self.capacity,)),
            "rho_reference": (float, np.float32, (self.capacity,)),
            "fluid_group_id": (wp.int32, np.int32, (self.capacity,)),
        }
        for name, (dtype, np_dtype, shape) in specs.items():
            host = np.zeros(shape, dtype=np_dtype)
            if name == "fluid_group_id":
                host.fill(-1)
            if name in initial:
                host[:self.count] = initial[name]
            arrays[name] = wp.array(host, dtype=dtype, device=self.device)
        arrays["rho"] = wp.zeros(self.capacity, dtype=float, device=self.device)
        arrays["acceleration"] = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        arrays["solid_force"] = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        arrays["damage_counter"] = wp.zeros(1, dtype=wp.int32, device=self.device)
        return arrays

    def substep(self, dt: float):
        a = self.arrays
        view = a["x"][:self.count]
        self.grid.build(view, self.max_support)
        wp.launch(clear_vec3, dim=self.count, inputs=[a["solid_force"][:self.count]], device=self.device)
        wp.launch(
            compute_density, dim=self.count,
            inputs=[self.grid.id, view, a["radius"][:self.count], a["mass"][:self.count], a["volume"][:self.count],
                    a["kind"][:self.count], a["rho"][:self.count], a["rho_reference"][:self.count],
                    float(self.cfg["rest_density"]), float(self.cfg["sound_speed"]),
                    float(self.cfg["water_depth"]), float(self.cfg["wave_height"]),
                    float(self.cfg["reservoir_z_max"]), self.max_support],
            device=self.device,
        )
        wp.launch(
            compute_fluid_forces, dim=self.count,
            inputs=[self.grid.id, view, a["v"][:self.count], a["radius"][:self.count], a["mass"][:self.count], a["volume"][:self.count],
                    a["kind"][:self.count], a["rho"][:self.count], a["acceleration"][:self.count],
                    a["solid_force"][:self.count], float(self.cfg["rest_density"]), float(self.cfg["sound_speed"]),
                    float(self.cfg.get("max_density_ratio", 1.08)),
                    float(self.cfg["viscosity"]), float(self.cfg.get("xsph_strength", 0.0)), self.max_support, dt],
            device=self.device,
        )
        wp.launch(
            compute_solid_forces, dim=self.count,
            inputs=[self.grid.id, view, a["rest_x"][:self.count], a["v"][:self.count], a["radius"][:self.count],
                    a["mass"][:self.count], a["kind"][:self.count], a["material"][:self.count],
                    a["building_id"][:self.count], a["fixed"][:self.count], a["damage"][:self.count],
                    a["solid_force"][:self.count], a["acceleration"][:self.count], self.max_support, dt],
            device=self.device,
        )
        wp.launch(
            integrate, dim=self.count,
            inputs=[view, a["v"][:self.count], a["acceleration"][:self.count], a["kind"][:self.count],
                    a["fixed"][:self.count], dt, float(self.cfg["domain_width"]) * 0.5,
                    float(self.cfg["reservoir_z_min"]), float(self.cfg["domain_z_max"]), float(self.cfg["domain_y_max"]),
                    float(self.cfg.get("fluid_bed_drag", 0.12)),
                    float(self.cfg.get("maximum_fluid_speed", 0.0)),
                    float(self.cfg.get("maximum_fluid_vertical_speed", 0.0)),
                    float(self.cfg.get("maximum_solid_speed", 0.0)),
                    float(self.cfg.get("maximum_solid_upward_speed", 0.0))],
            device=self.device,
        )
        self.time += dt

    def refine(self):
        if self.count >= self.capacity:
            return
        old_count = self.count
        count_device = wp.array(np.asarray([old_count], dtype=np.int32), dtype=wp.int32, device=self.device)
        a = self.arrays
        wp.launch(
            refine_entering_fluid, dim=old_count,
            inputs=[a["x"], a["rest_x"], a["v"], a["radius"], a["mass"], a["volume"], a["kind"],
                    a["material"], a["building_id"], a["fixed"], a["damage"], a["rho_reference"], count_device,
                    a["fluid_group_id"], self.fluid_group_counter,
                    old_count, self.capacity, float(self.cfg["fine_spacing"]) * 0.5, float(self.cfg["refine_z"]),
                    int(bool(self.cfg.get("adaptive_surface_only", False))),
                    float(self.cfg["water_depth"]) - float(self.cfg.get("fine_surface_band", 0.0)),
                    float(self.cfg.get("refine_vertical_speed", 2.5))],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        self.count = min(int(count_device.numpy()[0]), self.capacity)
        if self.count != old_count:
            print(f"  adaptive split: {old_count:,} -> {self.count:,} particles")

    def stats(self):
        counter = self.arrays["damage_counter"]
        wp.launch(clear_int, dim=1, inputs=[counter], device=self.device)
        wp.launch(count_damaged, dim=self.count,
                  inputs=[self.arrays["kind"][:self.count], self.arrays["damage"][:self.count], counter],
                  device=self.device)
        damaged = int(counter.numpy()[0])
        return {"fluid": self.count - self.solid_count, "solid": self.solid_count, "damaged": damaged}

    def save_checkpoint(self, frame: int):
        data = {name: arr[:self.count].numpy() for name, arr in self.arrays.items()
                if name not in ("rho", "acceleration", "solid_force", "damage_counter",
                                "water_surface_mask", "water_surface_normal", "water_foam_strength",
                                "water_mesh_vertices", "water_mesh_indices")}
        data.update(time=np.asarray(self.time), frame=np.asarray(frame), config=np.asarray(json.dumps(self.cfg)))
        path = self.checkpoint_dir / f"state_{frame:05d}.npz"
        np.savez_compressed(path, **data)
        print(f"  checkpoint: {path.name} ({path.stat().st_size / 1024**2:.1f} MiB)")

    def load_checkpoint(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.time = float(data["time"])
            self.start_frame = int(data["frame"]) + 1
            names = ("x", "v", "rest_x", "radius", "mass", "volume", "kind", "material", "building_id", "fixed", "damage")
            result = {name: data[name].copy() for name in names}
            result["structural_class"] = (
                data["structural_class"].copy()
                if "structural_class" in data
                else np.zeros(len(result["x"]), dtype=np.int32)
            )
            result["rho_reference"] = data["rho_reference"].copy() if "rho_reference" in data else np.zeros(len(result["x"]), dtype=np.float32)
            result["fluid_group_id"] = (
                data["fluid_group_id"].copy()
                if "fluid_group_id" in data
                else np.full(len(result["x"]), -1, dtype=np.int32)
            )
            return result

    def run(self, smoke: bool = False, no_video: bool = False):
        fps = int(self.cfg["output_fps"])
        total_frames = int(math.ceil(float(self.cfg["duration_seconds"]) * fps))
        dt_limit = float(self.cfg["dt"])
        substeps = int(math.ceil((1.0 / fps) / dt_limit))
        dt = (1.0 / fps) / substeps
        if smoke:
            total_frames = 2
            substeps = 2
            dt = min(dt, 0.001)
            print("SMOKE MODE: two rendered frames, two CUDA substeps each")
        print(f"Device={self.device}; frames={total_frames}; substeps/frame={substeps}; dt={dt:.6f}s; capacity={self.capacity:,}")
        render_times = []
        benchmark_rows = []
        metrics_path = self.output / "frame_metrics.jsonl"
        if self.start_frame == 0:
            metrics_path.write_text("", encoding="utf-8")
        simulation_started = time.perf_counter()
        gpu_device = wp.get_device(self.device)
        render_cfg = self.cfg["render"]
        output_mode = "png" if smoke or no_video else render_cfg.get("output_mode", "png")
        view_renderers = getattr(self, "renderers", {"main": self.renderer})
        multiple_views = len(view_renderers) > 1
        quad_layout = multiple_views and render_cfg.get("view_layout", "separate") == "quad"
        quad_order = list(render_cfg.get("quad_order", view_renderers.keys()))
        stream_writers = {}
        if output_mode == "video":
            output_basename = str(self.cfg.get("output_basename", "deluge_v3"))
            if quad_layout:
                segment = f"_segment_{self.start_frame:05d}" if self.start_frame != 0 else ""
                stream_writers["quad"] = StreamingVideoWriter(
                    self.output / f"{output_basename}{segment}.mp4",
                    int(render_cfg["width"]), int(render_cfg["height"]), fps,
                    render_cfg.get("video_codec", "h264_nvenc"),
                    float(render_cfg.get("progressive_fragment_seconds", 0.0)),
                )
            else:
                for view_name, view_renderer in view_renderers.items():
                    suffix = f"_{view_name}" if multiple_views or view_name != "main" else ""
                    segment = f"_segment_{self.start_frame:05d}" if self.start_frame != 0 else ""
                    stream_name = f"{output_basename}{suffix}{segment}.mp4"
                    stream_writers[view_name] = StreamingVideoWriter(
                        self.output / stream_name, int(view_renderer.width), int(view_renderer.height), fps,
                        render_cfg.get("video_codec", "h264_nvenc"),
                        float(render_cfg.get("progressive_fragment_seconds", 0.0)),
                    )

        for frame in range(self.start_frame, total_frames):
            started = time.perf_counter()
            for _ in range(substeps):
                self.substep(dt)
            refinement_enabled = bool(self.cfg.get("adaptive_refinement", True))
            if refinement_enabled and ((smoke and frame == 1) or (frame > 0 and frame % int(self.cfg.get("refine_every_frames", 8)) == 0)):
                self.refine()
            stats = self.stats()
            rendered_views = {}
            for view_name, view_renderer in view_renderers.items():
                if output_mode == "png" and not quad_layout:
                    view_dir = self.frames_dir / view_name if multiple_views else self.frames_dir
                    png_path = view_dir / f"frame_{frame:05d}.png"
                else:
                    png_path = None
                rgb = view_renderer.render(self.arrays, self.count, png_path, frame, self.time, stats)
                rendered_views[view_name] = rgb
                if not quad_layout and view_name in stream_writers:
                    stream_writers[view_name].write(rgb)
            if quad_layout:
                quad_rgb = compose_quad_view(
                    rendered_views, quad_order, int(render_cfg["width"]), int(render_cfg["height"])
                )
                if "quad" in stream_writers:
                    stream_writers["quad"].write(quad_rgb)
                elif output_mode == "png":
                    Image.fromarray(quad_rgb).save(self.frames_dir / f"frame_{frame:05d}.png", compress_level=2)
            wp.synchronize_device(self.device)
            elapsed = time.perf_counter() - started
            render_times.append(elapsed)
            gpu_used_mib = (gpu_device.total_memory - gpu_device.free_memory) / 1024**2
            row = {
                "frame": frame,
                "sim_time_seconds": self.time,
                "particles": self.count,
                "fluid_particles": stats["fluid"],
                "solid_particles": stats["solid"],
                "damaged_particles": stats["damaged"],
                "wall_seconds": elapsed,
                "substeps": substeps,
                "milliseconds_per_substep": elapsed * 1000.0 / substeps,
                "gpu_memory_used_mib": gpu_used_mib,
            }
            for optional_stat in ("active_buildings", "released_fragments", "rigid_clusters", "rigid_particles",
                                  "rigid_reactivated_fragments",
                                  "unsupported_fragments", "support_graph_edges", "support_graph_intact_edges",
                                  "invalid_zero_volume_particles",
                                  "local_impact_glass_particles",
                                  "fine_fluid_particles", "coarse_fluid_particles",
                                  "adaptive_merged_groups", "adaptive_merged_particles",
                                  "time_level_0_particles", "time_level_1_particles", "time_level_2_particles",
                                  "surface_water_particles", "water_mesh_vertices", "water_mesh_triangles",
                                  "water_field_nodes", "water_field_nx", "water_field_ny", "water_field_nz",
                                  "water_mesh_excluded_surface_particles",
                                  "water_mesh_voxel_millimeters", "water_mesh_lod_changes",
                                  "water_splash_bricks", "water_splash_mesh_vertices",
                                  "water_stitch_surface_samples", "shallow_water_cells",
                                  "shallow_water_wet_cells", "shallow_emitted_particles",
                                  "shallow_merged_particles", "fluid_particles_above_30m",
                                  "fluid_particles_above_42m", "fluid_particles_above_60m",
                                  "damaged_slab_particles", "damaged_wall_particles",
                                  "damaged_beam_particles", "damaged_column_particles",
                                  "damaged_core_particles", "damaged_glass_particles",
                                  "released_slab_fragments", "released_wall_fragments",
                                  "released_beam_fragments", "released_column_fragments",
                                  "released_core_fragments", "released_glass_fragments",
                                  "collapse_gravity_buildings"):
                if optional_stat in stats:
                    row[optional_stat] = int(stats[optional_stat])
            for optional_stat in (
                "fluid_volume_m3", "fluid_momentum_z_kg_m_s", "shallow_water_volume_m3",
                "shallow_water_momentum_z", "shallow_emitted_volume_m3",
                "shallow_merged_volume_m3", "shallow_net_transfer_volume_m3",
                "water_surface_classify_ms", "water_mesh_preprocess_ms", "water_mesh_field_ms",
                "water_mesh_marching_cubes_ms", "water_mesh_splash_ms", "water_mesh_total_ms",
                "material_impact_impulse_max_m_s",
                "fine_fluid_volume_percent",
                "water_mesh_span_x_m", "water_mesh_span_y_m", "water_mesh_span_z_m",
                "water_mesh_lower_x_m", "water_mesh_lower_y_m", "water_mesh_lower_z_m",
                "water_mesh_upper_x_m", "water_mesh_upper_y_m", "water_mesh_upper_z_m",
                "water_mesh_core_lower_x_m", "water_mesh_core_lower_y_m",
                "water_mesh_core_lower_z_m", "water_mesh_core_upper_x_m",
                "water_mesh_core_upper_y_m", "water_mesh_core_upper_z_m",
                "fluid_height_p99_m", "fluid_height_p999_m", "fluid_height_max_m",
                "fluid_vertical_speed_max_m_s",
                "solid_speed_p99_m_s", "solid_speed_max_m_s",
                "solid_upward_speed_max_m_s", "solid_mass_upward_above_10m_s_percent",
                "structural_slab_volume_m3", "structural_wall_volume_m3",
                "structural_beam_volume_m3", "structural_column_volume_m3",
                "structural_core_volume_m3", "structural_glass_volume_m3",
                "damaged_slab_volume_m3", "damaged_wall_volume_m3",
                "damaged_beam_volume_m3", "damaged_column_volume_m3",
                "damaged_core_volume_m3", "damaged_glass_volume_m3",
                "damage_integral_slab_m3", "damage_integral_wall_m3",
                "damage_integral_beam_m3", "damage_integral_column_m3",
                "damage_integral_core_m3", "damage_integral_glass_m3",
                "structural_collapse_gravity_max",
            ):
                if optional_stat in stats:
                    row[optional_stat] = float(stats[optional_stat])
            benchmark_rows.append(row)
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{frame+1:05d}/{total_frames:05d}] t={self.time:8.4f}s  particles={self.count:9,d}  damage={stats['damaged']:7,d}  wall={elapsed:7.2f}s  VRAM={gpu_used_mib:7.0f}MiB")
            checkpoint_every = int(self.cfg.get("checkpoint_every_frames", 0))
            if checkpoint_every and frame > 0 and frame % checkpoint_every == 0:
                self.save_checkpoint(frame)

        simulation_elapsed = time.perf_counter() - simulation_started
        encoding_elapsed = 0.0
        if stream_writers:
            encoding_started = time.perf_counter()
            for writer in stream_writers.values():
                writer.close()
            encoding_elapsed = time.perf_counter() - encoding_started
        elif not smoke and not no_video:
            encoding_started = time.perf_counter()
            output_basename = str(self.cfg.get("output_basename", "deluge_v3"))
            encode_targets = [("", self.frames_dir)] if quad_layout else [
                (f"_{view_name}" if multiple_views or view_name != "main" else "",
                 self.frames_dir / view_name if multiple_views else self.frames_dir)
                for view_name in view_renderers
            ]
            for suffix, view_dir in encode_targets:
                video = self.output / f"{output_basename}{suffix}.mp4"
                print(f"Encoding {video} ...")
                encode_video(view_dir, video, fps)
            encoding_elapsed = time.perf_counter() - encoding_started
        if benchmark_rows:
            summary = {
                "device": gpu_device.name,
                "cuda_arch": gpu_device.arch,
                "resolution": [int(self.cfg["render"]["width"]), int(self.cfg["render"]["height"])],
                "views": list(view_renderers.keys()),
                "view_count": len(view_renderers),
                "view_layout": "quad" if quad_layout else "separate",
                "output_fps": fps,
                "output_frames": len(benchmark_rows),
                "simulated_seconds": self.time,
                "substeps_per_frame": substeps,
                "dt_seconds": dt,
                "initial_particles": benchmark_rows[0]["particles"],
                "peak_particles": max(row["particles"] for row in benchmark_rows),
                "peak_fluid_particles": max(row["fluid_particles"] for row in benchmark_rows),
                "solid_particles": benchmark_rows[-1]["solid_particles"],
                "peak_damaged_particles": max(row["damaged_particles"] for row in benchmark_rows),
                "simulation_and_render_wall_seconds": simulation_elapsed,
                "video_encoding_wall_seconds": encoding_elapsed,
                "total_wall_seconds": simulation_elapsed + encoding_elapsed,
                "average_output_frame_wall_seconds": sum(render_times) / len(render_times),
                "median_output_frame_wall_seconds": float(np.median(render_times)),
                "max_output_frame_wall_seconds": max(render_times),
                "average_milliseconds_per_substep": 1000.0 * sum(render_times) / len(render_times) / substeps,
                "peak_gpu_memory_used_mib": max(row["gpu_memory_used_mib"] for row in benchmark_rows),
            }
            for optional_stat in ("active_buildings", "released_fragments", "rigid_clusters", "rigid_particles",
                                  "rigid_reactivated_fragments",
                                  "unsupported_fragments", "support_graph_edges", "support_graph_intact_edges",
                                  "invalid_zero_volume_particles",
                                  "local_impact_glass_particles",
                                  "fine_fluid_particles", "coarse_fluid_particles",
                                  "adaptive_merged_groups", "adaptive_merged_particles",
                                  "time_level_0_particles", "time_level_1_particles", "time_level_2_particles",
                                  "surface_water_particles", "water_mesh_vertices", "water_mesh_triangles",
                                  "water_field_nodes", "water_mesh_lod_changes", "water_splash_bricks",
                                  "water_splash_mesh_vertices", "water_stitch_surface_samples",
                                  "shallow_water_cells", "shallow_water_wet_cells",
                                  "shallow_emitted_particles", "shallow_merged_particles"):
                if optional_stat in benchmark_rows[0]:
                    summary[f"peak_{optional_stat}"] = max(row.get(optional_stat, 0) for row in benchmark_rows)
            if all(
                key in benchmark_rows[0] and key in benchmark_rows[-1]
                for key in ("fluid_volume_m3", "shallow_water_volume_m3")
            ):
                initial_water_volume = (
                    benchmark_rows[0]["fluid_volume_m3"]
                    + benchmark_rows[0]["shallow_water_volume_m3"]
                )
                final_water_volume = (
                    benchmark_rows[-1]["fluid_volume_m3"]
                    + benchmark_rows[-1]["shallow_water_volume_m3"]
                )
                summary["initial_combined_water_volume_m3"] = initial_water_volume
                summary["final_combined_water_volume_m3"] = final_water_volume
                summary["combined_water_volume_drift_fraction"] = (
                    final_water_volume / initial_water_volume - 1.0
                    if initial_water_volume > 0.0 else 0.0
                )
                summary["shallow_emitted_volume_m3"] = benchmark_rows[-1].get(
                    "shallow_emitted_volume_m3", 0.0
                )
            (self.output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if render_times:
            print(f"Complete. Average output-frame wall time: {sum(render_times)/len(render_times):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="DELUGE V2 offline CUDA particle simulation")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_preview.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Compile and test all CUDA stages using two frames")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or Path(__file__).with_name("outputs") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    wp.init()
    device = wp.get_device(cfg.get("device", "cuda:0"))
    if not device.is_cuda:
        raise RuntimeError("V2 requires an NVIDIA CUDA device")
    print(f"DELUGE V2 on {device.name}; CUDA compute {device.arch}; free memory is managed by Warp mempool")
    solver = DelugeSolver(cfg, output, args.resume)
    solver.run(smoke=args.smoke, no_video=args.no_video)


if __name__ == "__main__":
    main()
