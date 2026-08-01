"""GPU point-splat renderer and frame-to-video encoder."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import warp as wp

from kernels import (
    bilateral_depth_axis, clear_depth, clear_render, clear_scalar, raster_color, raster_depth,
    raster_water_depth, shade_water_surface,
)


class ParticleRenderer:
    def __init__(self, width: int, height: int, camera: dict, device: str):
        self.width = width
        self.height = height
        self.device = device
        self.depth = wp.empty(width * height, dtype=float, device=device)
        self.water_depth = wp.empty(width * height, dtype=float, device=device)
        self.water_temp = wp.empty(width * height, dtype=float, device=device)
        self.water_foam = wp.empty(width * height, dtype=float, device=device)
        self.color = wp.empty(width * height, dtype=wp.vec3, device=device)
        self.cam = np.asarray(camera["position"], dtype=np.float32)
        target = np.asarray(camera["target"], dtype=np.float32)
        forward = target - self.cam
        forward /= np.linalg.norm(forward)
        world_up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        right = np.cross(forward, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, forward); up /= np.linalg.norm(up)
        self.forward, self.right, self.up = forward, right, up
        fov = math.radians(float(camera.get("fov_degrees", 48.0)))
        self.focal = 0.5 * height / math.tan(fov * 0.5)

    def render(self, arrays: dict, count: int, output_path: Path | None, frame: int, time_s: float, stats: dict):
        wp.launch(clear_render, dim=self.width * self.height, inputs=[self.depth, self.color, self.width, self.height], device=self.device)
        wp.launch(clear_depth, dim=self.width * self.height, inputs=[self.water_depth], device=self.device)
        wp.launch(clear_scalar, dim=self.width * self.height, inputs=[self.water_foam, 0.0], device=self.device)
        common = [
            wp.vec3(*self.cam), wp.vec3(*self.right), wp.vec3(*self.up), wp.vec3(*self.forward),
            self.focal, self.width, self.height,
        ]
        wp.launch(
            raster_depth,
            dim=count,
            inputs=[arrays["x"][:count], arrays["radius"][:count], arrays["kind"][:count], self.depth, *common],
            device=self.device,
        )
        wp.launch(
            raster_water_depth,
            dim=count,
            inputs=[arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count], arrays["kind"][:count],
                    self.water_depth, self.water_foam, *common],
            device=self.device,
        )
        # Three separable bilateral iterations cost 6*7 taps per pixel instead
        # of 5*49 taps in the former square filter. Invalid pixels are filled
        # from nearby valid depth, closing small particle-splat holes.
        smooth_source, smooth_target = self.water_depth, self.water_temp
        for _ in range(3):
            wp.launch(
                bilateral_depth_axis, dim=self.width * self.height,
                inputs=[smooth_source, smooth_target, self.width, self.height, 2.4, 0.62, 0], device=self.device,
            )
            smooth_source, smooth_target = smooth_target, smooth_source
            wp.launch(
                bilateral_depth_axis, dim=self.width * self.height,
                inputs=[smooth_source, smooth_target, self.width, self.height, 2.4, 0.62, 1], device=self.device,
            )
            smooth_source, smooth_target = smooth_target, smooth_source
        # Opaque geometry is composed first. The reconstructed translucent
        # water surface is then blended only where its depth is closer. The old
        # order let the solid pass overwrite water even when water was in front.
        wp.launch(
            raster_color,
            dim=count,
            inputs=[
                arrays["x"][:count], arrays["v"][:count], arrays["radius"][:count],
                arrays["kind"][:count], arrays["material"][:count], arrays["damage"][:count],
                self.depth, self.color, *common,
            ],
            device=self.device,
        )
        wp.launch(
            shade_water_surface, dim=self.width * self.height,
            inputs=[smooth_source, self.water_foam, self.depth, self.color, self.width, self.height], device=self.device,
        )
        wp.synchronize_device(self.device)
        rgb = self.color.numpy().reshape(self.height, self.width, 3)
        rgb = np.clip(np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, "RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((20, 18, 420, 88), fill=(4, 14, 18, 190), outline=(80, 214, 228, 95), width=1)
        draw.text((36, 30), "DELUGE V3 / CUDA PARTICLE SOLVER", fill=(220, 241, 244, 255))
        draw.text((36, 54), f"FRAME {frame:05d}   T+{time_s:07.3f}s   {count:,} PARTICLES", fill=(102, 206, 217, 255))
        draw.rectangle((self.width - 330, 18, self.width - 20, 112), fill=(4, 14, 18, 190), outline=(255, 255, 255, 45), width=1)
        draw.text((self.width - 312, 30), f"WATER  {stats['fluid']:,}", fill=(92, 198, 215, 255))
        draw.text((self.width - 312, 53), f"SOLID  {stats['solid']:,}", fill=(190, 194, 188, 255))
        draw.text((self.width - 312, 76), f"DAMAGE {stats['damaged']:,}", fill=(232, 112, 76, 255))
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, compress_level=2)
        return rgb


class StreamingVideoWriter:
    """Stream RGB frames to recoverable video, optionally as live fMP4 fragments."""

    def __init__(self, output_file: Path, width: int, height: int, fps: int, codec: str = "h264_nvenc",
                 progressive_fragment_seconds: float = 0.0):
        self.output_file = output_file
        self.progressive_fragment_seconds = max(0.0, float(progressive_fragment_seconds))
        self.progressive = self.progressive_fragment_seconds > 0.0
        self.fragment_frames = max(1, int(round(fps * self.progressive_fragment_seconds))) if self.progressive else 0
        # Progressive mode writes directly to the requested MP4.  Its empty
        # moov plus completed moof/mdat fragments can be opened while FFmpeg
        # is still receiving later frames.  Legacy mode retains recoverable MKV.
        self.temp_file = output_file if self.progressive else output_file.with_suffix(".stream.mkv")
        self.log_path = output_file.with_suffix(".ffmpeg.log")
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.frames_written = 0
        command = [
            self.ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-an", "-c:v", codec,
        ]
        if codec == "h264_nvenc":
            command += ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
        else:
            command += ["-preset", "medium", "-crf", "17"]
        command += ["-pix_fmt", "yuv420p"]
        if self.progressive:
            fragment_us = max(1, int(round(self.progressive_fragment_seconds * 1_000_000.0)))
            if codec == "h264_nvenc":
                command += ["-zerolatency", "1", "-delay", "0", "-rc-lookahead", "0"]
            else:
                command += ["-tune", "zerolatency"]
            command += [
                "-g", str(self.fragment_frames),
                "-bf", "0",
                "-force_key_frames", f"expr:gte(t,n_forced*{self.progressive_fragment_seconds:.9f})",
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                "-frag_duration", str(fragment_us),
                "-flush_packets", "1",
            ]
        command += [str(self.temp_file)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log_file)

    def write(self, rgb: np.ndarray):
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is closed")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frames_written += 1
        # A keyframe at the start of the next interval closes the preceding
        # fragment.  Flush the Python pipe around both sides of that boundary
        # so the playable file appears as soon as one simulated second exists.
        if self.progressive and self.frames_written % self.fragment_frames in (0, 1):
            self.process.stdin.flush()

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        self.log_file.close()
        if code != 0:
            raise RuntimeError(f"Streaming FFmpeg failed with exit code {code}; see {self.log_path}")
        remux_source = self.temp_file
        remux_target = self.output_file.with_suffix(".finalizing.mp4") if self.progressive else self.output_file
        subprocess.run(
            [self.ffmpeg, "-y", "-i", str(remux_source), "-c", "copy", "-movflags", "+faststart", str(remux_target)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if self.progressive:
            os.replace(remux_target, self.output_file)
        else:
            self.temp_file.unlink(missing_ok=True)


def encode_video(frames_dir: Path, output_file: Path, fps: int):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_file),
    ]
    subprocess.run(command, check=True)
