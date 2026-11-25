"""High-performance multi-mode vertical clipper with hardware acceleration.

Features:
- Dual-engine encoding: Auto-detects Intel Arc Quick Sync Video (h264_qsv)
  with Direct3D11 hardware acceleration, falling back gracefully to multi-threaded libx264.
- Broadcast audio mastering: EBU R128 loudness normalization (-14 LUFS, -1.5 dBFS True Peak),
  highpass rumble cut (80 Hz), and vocal presence boost (3.2 kHz).
- Studio reframing modes: split_stack, left_focus, right_focus, smart_focus, center, fit_blur, zoom_blur.
- Dynamic aspect ratios: 9:16 (vertical), 1:1 (square), 4:5 (portrait), 16:9 (landscape).
- Mobile streaming optimization: BT.709 broadcast colorimetry and moov atom faststart.
"""

import io
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from ..config import LOCAL_OUTPUT_DIR, RAW_CAPTIONS_DIR
from ..clip_registry import ClipRegistry
from ..diversity_types import CandidateHighlight, FramingMode, HighlightArchetype, ZoomPlan
from ..engine.self_healing import CorruptedVideoError, SelfHealingVideoSanitizer
from ..face_tracker import FaceTracker
from ..framing_analyzer import FramingAnalyzer, generate_pan_crop_expression, smooth_speaker_tracking
from ..video.color_matrix_fixer import ColorMatrixFixer
from ..viral_zoom_analyzer import ViralZoomAnalyzer
from .subtitles import generate_ass_file


def detect_hardware_encoder() -> Tuple[str, List[str], str]:
    """Probe system hardware to select optimal video encoder.

    Returns:
        Tuple of (encoder_name, extra_cli_flags, pixel_format_filter).
    """
    return get_encoder_config(encoder_choice="auto", resolution="4k")


def get_encoder_config(encoder_choice: str = "auto", resolution: str = "4k") -> Tuple[str, List[str], str]:
    """Select optimal video encoder and high-bitrate flags for broadcast quality.

    Args:
        encoder_choice: Encoder selection ('auto', 'qsv', 'cpu').
        resolution: Target resolution profile ('4k' or '1080p').

    Returns:
        Tuple of (encoder_name, extra_cli_flags, pixel_format_filter).
    """
    is_4k = (resolution or "4k").lower() == "4k"

    if encoder_choice in ("auto", "qsv"):
        probe_cmd = [
            "ffmpeg", "-v", "error",
            "-f", "lavfi",
            "-i", "color=c=black:s=128x128:r=30:d=0.1",
            "-pix_fmt", "nv12",
            "-c:v", "h264_qsv",
            "-b:v", "1M",
            "-f", "null", "-"
        ]
        try:
            res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                if is_4k:
                    qsv_flags = ["-global_quality", "16", "-b:v", "35M", "-maxrate", "50M", "-bufsize", "60M", "-look_ahead", "1"]
                else:
                    qsv_flags = ["-global_quality", "18", "-b:v", "15M", "-maxrate", "25M", "-bufsize", "30M", "-look_ahead", "1"]
                return "h264_qsv", qsv_flags, "format=nv12"
        except Exception:
            pass

    if is_4k:
        cpu_flags = ["-preset", "medium", "-crf", "16", "-b:v", "30M", "-maxrate", "45M", "-bufsize", "60M", "-threads", "0"]
    else:
        cpu_flags = ["-preset", "slow", "-crf", "17", "-b:v", "15M", "-maxrate", "25M", "-bufsize", "30M", "-threads", "0"]
    return "libx264", cpu_flags, "format=yuv420p"


def get_source_dimensions(media_path: str) -> Tuple[int, int]:
    """Retrieve video dimensions via ffprobe.

    Args:
        media_path: Path to source media file.

    Returns:
        Tuple of (width, height) in pixels.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        media_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    parts = res.stdout.strip().split("x")
    return int(parts[0]), int(parts[1])


def get_target_canvas_dimensions(aspect_ratio: str = "9:16", resolution: str = "4k") -> Tuple[int, int]:
    """Compute target render canvas dimensions for requested aspect ratio and resolution profile.

    Args:
        aspect_ratio: Target aspect ratio ('9:16', '1:1', '4:5', '16:9').
        resolution: Target resolution profile ('4k' or '1080p').

    Returns:
        Tuple of (width, height) in pixels.
    """
    is_4k = (resolution or "4k").lower() == "4k"
    norm_ratio = (aspect_ratio or "9:16").strip()

    if norm_ratio == "1:1":
        dim = 2160 if is_4k else 1080
        return dim, dim
    if norm_ratio == "4:5":
        return (1728, 2160) if is_4k else (864, 1080)
    if norm_ratio == "16:9":
        return (3840, 2160) if is_4k else (1920, 1080)

    return (2160, 3840) if is_4k else (1080, 1920)


def extract_representative_keyframe(source_path: str, timestamp: float) -> Optional[Image.Image]:
    """Extract a single video frame as a PIL Image at specified timestamp.

    Args:
        source_path: Local media path.
        timestamp: Presentation timestamp in seconds.

    Returns:
        PIL Image instance or None if extraction fails.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{max(0.0, timestamp):.3f}",
        "-i", source_path,
        "-vframes", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return Image.open(io.BytesIO(res.stdout)).convert("RGB")
    except Exception:
        return None


def build_ffmpeg_filtergraph(
    src_w: int,
    src_h: int,
    framing: str = "split_stack",
    ass_path: Optional[str] = None,
    pixel_format: str = "format=nv12",
    resolution: str = "4k",
    aspect_ratio: str = "9:16",
    framing_params: Optional[Dict] = None,
) -> str:
    """Construct FFmpeg video filtergraph with framing, separator, zoom, and subtitle burn-in.

    Args:
        src_w: Source video width in pixels.
        src_h: Source video height in pixels.
        framing: Framing mode (split_stack, left_focus, right_focus, smart_focus, center, fit_blur, zoom_blur).
        ass_path: Optional path to ASS subtitle file to overlay.
        pixel_format: Pixel format filter string for encoder compatibility.
        resolution: Target resolution profile ('4k' for 2160x3840 or '1080p' for 1080x1920).
        aspect_ratio: Target aspect ratio ('9:16', '1:1', '4:5', '16:9').
        framing_params: Optional dictionary of fine-tuned spatial coordinates, ratios, and dynamic zoom plans.

    Returns:
        FFmpeg filtergraph string.
    """
    escaped_ass = ""
    if ass_path:
        norm_path = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")
        escaped_ass = f",ass='{norm_path}'"

    is_4k = (resolution or "4k").lower() == "4k"
    out_w, out_h = get_target_canvas_dimensions(aspect_ratio=aspect_ratio, resolution=resolution)

    target_ratio = out_w / out_h
    crop_w = int(src_h * target_ratio)
    crop_w = min(src_w, max(2, crop_w - (crop_w % 2)))

    zoom_plan = framing_params.get("zoom_plan", None) if framing_params else None
    focal_x = framing_params.get("focal_x", 0.5) if framing_params else 0.5
    focal_y = framing_params.get("focal_y", 0.45) if framing_params else 0.45

    color_matrix_filter = framing_params.get("color_matrix_filter", None) if framing_params else None
    cm_prefix = f"{color_matrix_filter}," if color_matrix_filter else ""

    if framing == "split_stack":
        half_h = out_h // 2
        sub_ratio = framing_params.get("sub_h_ratio", 0.72) if framing_params else 0.72
        sub_h = int(src_h * sub_ratio)
        sub_w = int(sub_h * (out_w / half_h))
        sub_w = min(src_w, max(2, sub_w - (sub_w % 2)))
        sub_h = max(2, sub_h - (sub_h % 2))

        def_x_left = max(0, int(src_w * 0.08))
        def_x_right = min(src_w - sub_w, int(src_w * 0.46))
        def_y = int((src_h - sub_h) * 0.35)

        x_top = framing_params.get("x_top", def_x_left) if framing_params else def_x_left
        x_bot = framing_params.get("x_bot", def_x_right) if framing_params else def_x_right
        y_top = framing_params.get("y_top", def_y) if framing_params else def_y
        y_bot = framing_params.get("y_bot", def_y) if framing_params else def_y

        line_y = half_h - 1
        line_thick = 4 if is_4k else 2

        zoom_plan_top = framing_params.get("zoom_plan_top", zoom_plan) if framing_params else zoom_plan
        zoom_plan_bot = framing_params.get("zoom_plan_bot", None) if framing_params else None

        crop_top = ViralZoomAnalyzer.build_crop_filter_with_zoom(sub_w, sub_h, x_top, y_top, zoom_plan_top, focal_x, focal_y)
        crop_bot = ViralZoomAnalyzer.build_crop_filter_with_zoom(sub_w, sub_h, x_bot, y_bot, zoom_plan_bot, focal_x, focal_y)

        if color_matrix_filter:
            filtergraph = (
                f"[0:v]{color_matrix_filter},split=2[v_top_in][v_bot_in]; "
                f"[v_top_in]{crop_top},scale={out_w}:{half_h}:flags=lanczos[top]; "
                f"[v_bot_in]{crop_bot},scale={out_w}:{half_h}:flags=lanczos[bot]; "
                f"[top][bot]vstack=inputs=2,drawbox=y={line_y}:w={out_w}:h={line_thick}:color=black@0.9:t=fill"
                f"{escaped_ass},{pixel_format}[v_out]"
            )
        else:
            filtergraph = (
                f"[0:v]{crop_top},scale={out_w}:{half_h}:flags=lanczos[top]; "
                f"[0:v]{crop_bot},scale={out_w}:{half_h}:flags=lanczos[bot]; "
                f"[top][bot]vstack=inputs=2,drawbox=y={line_y}:w={out_w}:h={line_thick}:color=black@0.9:t=fill"
                f"{escaped_ass},{pixel_format}[v_out]"
            )
        return filtergraph

    if framing == "fit_blur":
        box_blur = "50:10" if is_4k else "25:5"
        if color_matrix_filter:
            filtergraph = (
                f"[0:v]{color_matrix_filter},split=2[v_fb_bg][v_fb_fg]; "
                f"[v_fb_bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur={box_blur},eq=brightness=-0.30[bg]; "
                f"[v_fb_fg]scale={out_w}:-1:flags=lanczos[fg]; "
                f"[bg][fg]overlay=0:(H-h)/2"
                f"{escaped_ass},{pixel_format}[v_out]"
            )
        else:
            filtergraph = (
                f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur={box_blur},eq=brightness=-0.30[bg]; "
                f"[0:v]scale={out_w}:-1:flags=lanczos[fg]; "
                f"[bg][fg]overlay=0:(H-h)/2"
                f"{escaped_ass},{pixel_format}[v_out]"
            )
        return filtergraph

    if framing == "zoom_blur":
        box_blur = "50:10" if is_4k else "25:5"
        fg_crop_w = int(src_w * 0.625)
        fg_crop_h = int(fg_crop_w * (1.0 / target_ratio))
        fg_crop_w = max(2, fg_crop_w - (fg_crop_w % 2))
        fg_crop_h = max(2, fg_crop_h - (fg_crop_h % 2))
        fg_crop_x = int(src_w * 0.208)
        fg_crop_y = int(src_h * 0.138)
        fg_h = int(out_w * (1.0 / target_ratio))
        fg_h = max(2, fg_h - (fg_h % 2))

        if color_matrix_filter:
            filtergraph = (
                f"[0:v]{color_matrix_filter},split=2[v_zb_bg][v_zb_fg]; "
                f"[v_zb_bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur={box_blur},eq=brightness=-0.35[bg]; "
                f"[v_zb_fg]crop={fg_crop_w}:{fg_crop_h}:{fg_crop_x}:{fg_crop_y},scale={out_w}:{fg_h}:flags=lanczos[fg]; "
                f"[bg][fg]overlay=0:(H-h)/2"
                f"{escaped_ass},{pixel_format}[v_out]"
            )
        else:
            filtergraph = (
                f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur={box_blur},eq=brightness=-0.35[bg]; "
                f"[0:v]crop={fg_crop_w}:{fg_crop_h}:{fg_crop_x}:{fg_crop_y},scale={out_w}:{fg_h}:flags=lanczos[fg]; "
                f"[bg][fg]overlay=0:(H-h)/2"
                f"{escaped_ass},{pixel_format}[v_out]"
            )
        return filtergraph

    if framing == "left_focus":
        x_pos = max(0, min(src_w - crop_w, int(src_w * 0.18)))
        crop_filter = ViralZoomAnalyzer.build_crop_filter_with_zoom(crop_w, src_h, x_pos, 0, zoom_plan, focal_x, focal_y)
        return f"{cm_prefix}{crop_filter},scale={out_w}:{out_h}:flags=lanczos{escaped_ass},{pixel_format}"

    if framing == "right_focus":
        x_pos = max(0, min(src_w - crop_w, int(src_w * 0.48)))
        crop_filter = ViralZoomAnalyzer.build_crop_filter_with_zoom(crop_w, src_h, x_pos, 0, zoom_plan, focal_x, focal_y)
        return f"{cm_prefix}{crop_filter},scale={out_w}:{out_h}:flags=lanczos{escaped_ass},{pixel_format}"

    if framing == "smart_focus":
        pan_expr = framing_params.get("pan_expr", None) if framing_params else None
        strip_letterbox = framing_params.get("strip_letterbox", False) if framing_params else False
        if strip_letterbox:
            active_h = framing_params.get("active_h", int(src_h * 0.88)) if framing_params else int(src_h * 0.88)
            active_h = max(2, active_h - (active_h % 2))
            active_crop_w = int(active_h * target_ratio)
            active_crop_w = min(src_w, max(2, active_crop_w - (active_crop_w % 2)))
            y_offset = framing_params.get("y_offset", (src_h - active_h) // 2) if framing_params else (src_h - active_h) // 2
            if pan_expr:
                crop_filter = f"crop=w={active_crop_w}:h={active_h}:x='{pan_expr}':y={y_offset}:exact=1"
            else:
                x_ratio = framing_params.get("x_ratio", 0.30) if framing_params else 0.30
                x_pos = max(0, min(src_w - active_crop_w, int(src_w * x_ratio)))
                crop_filter = ViralZoomAnalyzer.build_crop_filter_with_zoom(active_crop_w, active_h, x_pos, y_offset, zoom_plan, focal_x, focal_y)
            return f"{cm_prefix}{crop_filter},scale={out_w}:{out_h}:flags=lanczos{escaped_ass},{pixel_format}"
        if pan_expr:
            crop_filter = f"crop=w={crop_w}:h={src_h}:x='{pan_expr}':y=0:exact=1"
            return f"{cm_prefix}{crop_filter},scale={out_w}:{out_h}:flags=lanczos{escaped_ass},{pixel_format}"
        x_ratio = framing_params.get("x_ratio", 0.28) if framing_params else 0.28
        x_pos = max(0, min(src_w - crop_w, int(src_w * x_ratio)))
        crop_filter = ViralZoomAnalyzer.build_crop_filter_with_zoom(crop_w, src_h, x_pos, 0, zoom_plan, focal_x, focal_y)
        return f"{cm_prefix}{crop_filter},scale={out_w}:{out_h}:flags=lanczos{escaped_ass},{pixel_format}"

    x_pos = (src_w - crop_w) // 2
    crop_filter = ViralZoomAnalyzer.build_crop_filter_with_zoom(crop_w, src_h, x_pos, 0, zoom_plan, focal_x, focal_y)
    return f"{cm_prefix}{crop_filter},scale={out_w}:{out_h}:flags=lanczos{escaped_ass},{pixel_format}"


def render_short_clip(
    source_path: str,
    start_time: float,
    end_time: float,
    out_path: str,
    framing: str = "split_stack",
    burn_subtitles: bool = True,
    master_audio: bool = True,
    encoder_choice: str = "auto",
    resolution: str = "4k",
    aspect_ratio: str = "9:16",
    framing_params: Optional[Dict] = None,
    words: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render a mobile short video with hardware acceleration and broadcast audio.

    Args:
        source_path: Local media file path.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        out_path: Output file destination path.
        framing: Framing strategy (split_stack, left_focus, right_focus, smart_focus, center, fit_blur, zoom_blur).
        burn_subtitles: Whether to burn animated karaoke subtitles.
        master_audio: Whether to apply EBU R128 loudness normalization and EQ.
        encoder_choice: Encoder selection ('auto', 'qsv', 'cpu').
        resolution: Output resolution profile ('4k' for 2160x3840 or '1080p' for 1080x1920).
        aspect_ratio: Target aspect ratio ('9:16', '1:1', '4:5', '16:9').
        framing_params: Optional dictionary of fine-tuned spatial coordinates and ratios.
        words: Optional pre-extracted word timing dictionaries for subtitle generation.

    Returns:
        Path to rendered .mp4 file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    source_path = SelfHealingVideoSanitizer.ensure_playable_or_heal(source_path)
    src_w, src_h = get_source_dimensions(source_path)

    if framing_params is None:
        framing_params = {}
    if "color_matrix_filter" not in framing_params:
        cm_meta = ColorMatrixFixer.probe_color_metadata(source_path)
        cm_filter = ColorMatrixFixer.build_color_matrix_filter(cm_meta)
        if cm_filter:
            framing_params["color_matrix_filter"] = cm_filter

    is_4k = (resolution or "4k").lower() == "4k"
    target_w, target_h = get_target_canvas_dimensions(aspect_ratio=aspect_ratio, resolution=resolution)

    encoder, extra_enc_args, pix_fmt = get_encoder_config(encoder_choice, resolution=resolution)

    ass_path: Optional[str] = None
    if burn_subtitles:
        stem = Path(out_path).stem
        ass_path = os.path.join(RAW_CAPTIONS_DIR, f"{stem}_captions.ass")
        default_margin_v = 940 if framing == "split_stack" else 520
        margin_v = framing_params.get("margin_v", default_margin_v) if framing_params else default_margin_v
        generate_ass_file(
            input_media=source_path,
            output_ass=ass_path,
            margin_v=margin_v,
            target_w=target_w,
            target_h=target_h,
            words=words,
            clip_start=start_time,
            clip_end=end_time,
        )

    filtergraph = build_ffmpeg_filtergraph(
        src_w, src_h,
        framing=framing,
        ass_path=ass_path,
        pixel_format=pix_fmt,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        framing_params=framing_params,
    )

    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", source_path
    ]

    if framing in ["split_stack", "fit_blur", "zoom_blur"]:
        cmd.extend([
            "-filter_complex", filtergraph,
            "-map", "[v_out]",
            "-map", "0:a"
        ])
    else:
        cmd.extend([
            "-vf", filtergraph
        ])

    if master_audio:
        fade_out_st = max(0.0, duration - 0.03)
        af_chain = (
            f"afade=t=in:ss=0:d=0.02,afade=t=out:st={fade_out_st:.3f}:d=0.03,"
            "highpass=f=80,equalizer=f=3200:t=q:w=1.5:g=3.0,loudnorm=I=-14:LRA=11:TP=-1.5,aresample=48000"
        )
        cmd.extend([
            "-af", af_chain,
            "-c:a", "aac",
            "-b:a", "192k"
        ])
    else:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k"
        ])

    cmd.extend([
        "-c:v", encoder,
        *extra_enc_args,
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-movflags", "+faststart",
        out_path
    ])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        if encoder == "h264_qsv":
            print(f"[clip/local] WARNING: Hardware encoder h264_qsv failed ({err}). Falling back to libx264 software encoder.", flush=True)
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass

            sw_filtergraph = build_ffmpeg_filtergraph(
                src_w, src_h,
                framing=framing,
                ass_path=ass_path,
                pixel_format="format=yuv420p",
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                framing_params=framing_params,
            )

            if is_4k:
                sw_flags = ["-preset", "medium", "-crf", "16", "-b:v", "30M", "-maxrate", "45M", "-bufsize", "60M", "-threads", "0"]
            else:
                sw_flags = ["-preset", "slow", "-crf", "17", "-b:v", "15M", "-maxrate", "25M", "-bufsize", "30M", "-threads", "0"]

            sw_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start_time:.3f}",
                "-t", f"{duration:.3f}",
                "-i", source_path
            ]
            if framing in ["split_stack", "fit_blur", "zoom_blur"]:
                sw_cmd.extend([
                    "-filter_complex", sw_filtergraph,
                    "-map", "[v_out]",
                    "-map", "0:a"
                ])
            else:
                sw_cmd.extend([
                    "-vf", sw_filtergraph
                ])

            if master_audio:
                fade_out_st = max(0.0, duration - 0.03)
                af_chain = (
                    f"afade=t=in:ss=0:d=0.02,afade=t=out:st={fade_out_st:.3f}:d=0.03,"
                    "highpass=f=80,equalizer=f=3200:t=q:w=1.5:g=3.0,loudnorm=I=-14:LRA=11:TP=-1.5,aresample=48000"
                )
                sw_cmd.extend([
                    "-af", af_chain,
                    "-c:a", "aac",
                    "-b:a", "192k"
                ])
            else:
                sw_cmd.extend([
                    "-c:a", "aac",
                    "-b:a", "192k"
                ])

            sw_cmd.extend([
                "-c:v", "libx264",
                *sw_flags,
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
                "-movflags", "+faststart",
                out_path
            ])
            try:
                subprocess.run(sw_cmd, check=True)
            except subprocess.CalledProcessError as sw_err:
                repaired_ok, repaired_path = SelfHealingVideoSanitizer.repair_container(source_path)
                if repaired_ok and repaired_path != source_path:
                    print(f"[clip/local] Re-indexed corrupt container to {repaired_path}. Retrying render...", flush=True)
                    return render_short_clip(
                        source_path=repaired_path,
                        start_time=start_time,
                        end_time=end_time,
                        out_path=out_path,
                        framing=framing,
                        burn_subtitles=burn_subtitles,
                        master_audio=master_audio,
                        encoder_choice="cpu",
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        framing_params=framing_params,
                        words=words,
                    )
                SelfHealingVideoSanitizer.route_to_dlq(source_path, reason=str(sw_err))
                raise CorruptedVideoError(f"Rendering failed and container repair could not salvage asset: {sw_err}")
        else:
            repaired_ok, repaired_path = SelfHealingVideoSanitizer.repair_container(source_path)
            if repaired_ok and repaired_path != source_path:
                print(f"[clip/local] Re-indexed corrupt container to {repaired_path}. Retrying render...", flush=True)
                return render_short_clip(
                    source_path=repaired_path,
                    start_time=start_time,
                    end_time=end_time,
                    out_path=out_path,
                    framing=framing,
                    burn_subtitles=burn_subtitles,
                    master_audio=master_audio,
                    encoder_choice="cpu",
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    framing_params=framing_params,
                    words=words,
                )
            SelfHealingVideoSanitizer.route_to_dlq(source_path, reason=str(err))
            raise CorruptedVideoError(f"Rendering failed and container repair could not salvage asset: {err}")

    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    framing: str = "split_stack",
    burn_subtitles: bool = True,
    master_audio: bool = True,
    out_dir: Optional[str] = None,
    video_id: Optional[str] = None,
    resolution: str = "4k",
    transcript_segments: Optional[List[Dict]] = None,
    transcript_words: Optional[List[Dict]] = None,
    auto_framing: bool = True,
    auto_zoom: bool = True,
) -> List[Dict]:
    """Render highlighted segments into standalone social video clips.

    Args:
        source_path: Local media file path.
        highlights: List of highlight dictionaries with timing boundaries.
        aspect_ratio: Target aspect ratio ('9:16', '1:1', '4:5', '16:9').
        framing: Framing mode string.
        burn_subtitles: Subtitle overlay toggle.
        master_audio: Audio normalization and EQ toggle.
        out_dir: Destination directory path.
        video_id: Persistent identifier override.
        resolution: Target resolution profile ('4k' or '1080p').
        transcript_segments: Full list of transcript segment dictionaries.
        transcript_words: Full list of pre-extracted word timing dictionaries.
        auto_framing: Intelligent framing recommendation toggle.
        auto_zoom: Dynamic zoom plan toggle.

    Returns:
        List of processed highlight dictionaries with output file paths.
    """
    out_dir = os.path.abspath(os.path.expanduser(out_dir or LOCAL_OUTPUT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    registry = ClipRegistry()
    vid = video_id or Path(source_path).stem
    src_dims = get_source_dimensions(source_path)

    framing_analyzer = FramingAnalyzer(target_resolution=resolution) if auto_framing else None
    zoom_analyzer = ViralZoomAnalyzer() if auto_zoom else None

    for i, h in enumerate(highlights, 1):
        archetype_str = str(h.get("archetype", "HIGHLIGHT")).upper()
        archetype_enum = HighlightArchetype.from_string(archetype_str)
        clip_framing = h.get("framing", framing)
        clip_framing_params = dict(h.get("framing_params") or {})

        candidate = CandidateHighlight(
            title=str(h.get("title", "")),
            raw_start=float(h["start_time"]),
            raw_end=float(h["end_time"]),
            score=int(h.get("score", 0)),
            archetype=archetype_enum,
            hook_sentence=str(h.get("hook_sentence", "")),
            virality_reason=str(h.get("virality_reason", "")),
            snapped_start=float(h.get("snapped_start", h["start_time"])),
            snapped_end=float(h.get("snapped_end", h["end_time"])),
            is_snapped=bool(h.get("is_snapped", False)),
        )

        if auto_framing and framing_analyzer and (clip_framing in ("auto", None) or not clip_framing_params):
            sample_time = candidate.snapped_start + (candidate.snapped_end - candidate.snapped_start) * 0.25
            keyframe = extract_representative_keyframe(source_path, sample_time)
            rec = framing_analyzer.recommend_framing(
                highlight=candidate,
                transcript_segments=transcript_segments,
                source_dimensions=src_dims,
                keyframe=keyframe,
            )
            if clip_framing in ("auto", None):
                clip_framing = rec.mode
            clip_framing_params = {**rec.params, **clip_framing_params}

        if auto_zoom and zoom_analyzer and "zoom_plan" not in clip_framing_params:
            z_plan = zoom_analyzer.plan_zooms(candidate)
            if z_plan.events:
                clip_framing_params["zoom_plan"] = z_plan

        if auto_framing and (clip_framing in ("auto", "smart_focus") or clip_framing_params.get("active_face_tracking", False)):
            try:
                face_tracker = FaceTracker()
                face_detections = face_tracker.sample_video_faces(
                    media_path=source_path,
                    start_time=candidate.snapped_start,
                    end_time=candidate.snapped_end,
                    fps_sample=2.0
                )
                if face_detections:
                    smoothed = smooth_speaker_tracking(face_detections, frame_width=src_dims[0])
                    t_w, t_h = get_target_canvas_dimensions(aspect_ratio=aspect_ratio, resolution=resolution)
                    t_ratio = t_w / t_h
                    c_w = int(src_dims[1] * t_ratio)
                    c_w = min(src_dims[0], max(2, c_w - (c_w % 2)))
                    pan_expr = generate_pan_crop_expression(smoothed, crop_w=c_w, src_w=src_dims[0])
                    clip_framing_params["pan_expr"] = pan_expr
                    clip_framing_params["smoothed_tracking"] = smoothed
                    if clip_framing in ("auto", None):
                        clip_framing = "smart_focus"
            except Exception as e:
                print(f"[clip/local] Face tracking notice: {e}", flush=True)

        seq = registry.get_next_sequence_number(vid)
        out_path = os.path.join(out_dir, f"short_{vid}_{seq:03d}_{archetype_str}.mp4")
        print(f"[clip/local] Rendering {i}/{len(highlights)}: {h.get('title', '(untitled)')} [{clip_framing}] ({resolution}) -> {Path(out_path).name}", flush=True)
        try:
            render_short_clip(
                source_path=source_path,
                start_time=float(h["start_time"]),
                end_time=float(h["end_time"]),
                out_path=out_path,
                framing=clip_framing,
                burn_subtitles=burn_subtitles,
                master_audio=master_audio,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                framing_params=clip_framing_params,
                words=transcript_words,
            )
            registry.register_clip(
                video_id=vid,
                start_time=float(h["start_time"]),
                end_time=float(h["end_time"]),
                guard_band=45.0,
                archetype=archetype_str,
                score=int(h.get("score", 0)),
                title=str(h.get("title", "")),
                hook_sentence=str(h.get("hook_sentence", "")),
                file_path=out_path
            )
            results.append({
                **h,
                "clip_url": out_path,
                "framing": clip_framing,
                "framing_params": clip_framing_params,
            })
        except Exception as e:
            print(f"[clip/local] Error on clip {i}: {e}", flush=True)
            results.append({
                **h,
                "clip_url": None,
                "error": str(e),
                "dlq_path": getattr(e, "dlq_path", None),
            })

    return results
