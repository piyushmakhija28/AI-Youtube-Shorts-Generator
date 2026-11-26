"""Real-world validation script for prosody-reactive kinetic typography on YouTube footage.

Tests acoustic feature extraction, dynamic color temperature mapping, squash-and-stretch
spring physics tags, and end-to-end 9:16 mobile short video rendering on actual 4K media.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List

from shorts_generator.local.clipper import (
    get_source_dimensions,
    render_short_clip,
)
from shorts_generator.local.subtitles import generate_ass_file
from shorts_generator.video.prosody_typography import (
    KineticStyleConfig,
    ProsodyKineticTypography,
    ProsodyWordMetric,
)


def probe_rendered_short(file_path: str) -> Dict[str, Any]:
    """Inspect output video stream attributes using ffprobe.

    Args:
        file_path: Local path to rendered MP4 video.

    Returns:
        Dictionary containing width, height, duration, vcodec, acodec, and file size.

    Raises:
        RuntimeError: If ffprobe execution fails or output JSON is invalid.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate",
        "-of", "json",
        file_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr}")

    data = json.loads(res.stdout)
    v_stream = next((s for s in data.get("streams", []) if s.get("width")), {})
    a_stream = next((s for s in data.get("streams", []) if not s.get("width")), {})

    return {
        "width": int(v_stream.get("width", 0)),
        "height": int(v_stream.get("height", 0)),
        "vcodec": v_stream.get("codec_name", "unknown"),
        "acodec": a_stream.get("codec_name", "unknown"),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "size_bytes": int(data.get("format", {}).get("size", 0)),
    }


def run_prosody_real_video_verification() -> int:
    """Execute end-to-end prosody typography test on real 4K YouTube video footage.

    Returns:
        Exit code 0 on complete success, 1 on assertion failure.
    """
    raw_video = "output/raw/video/ep1_raw_4k_moment1.mp4"
    transcript_file = "output/raw/captions/ep1_transcript.json"
    out_dir = "output/real_test"
    out_ass = os.path.join(out_dir, "prosody_real_short.ass")
    out_video = os.path.join(out_dir, "prosody_real_short_9x16.mp4")

    print("=" * 80)
    print("PROSODY KINETIC TYPOGRAPHY: REAL YOUTUBE VIDEO VERIFICATION")
    print(f"Target Video:     {raw_video}")
    print(f"Transcript Data:  {transcript_file}")
    print("=" * 80)

    if not os.path.exists(raw_video) or not os.path.exists(transcript_file):
        print(f"[Error] Required asset missing: {raw_video} or {transcript_file}")
        return 1

    src_w, src_h = get_source_dimensions(raw_video)
    print(f"[1/5] Source dimensions verified: {src_w}x{src_h}")

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    clip_start = 14.0
    clip_end = 27.5
    raw_segments = transcript_data.get("segments", [])

    selected_words: List[Dict[str, Any]] = []
    for s in raw_segments:
        if s["end"] < clip_start or s["start"] > clip_end:
            continue
        for w in s.get("words", []):
            w_st = float(w["start"])
            w_et = float(w["end"])
            if w_st >= clip_start and w_et <= clip_end:
                selected_words.append({
                    "word": w["word"].strip().upper(),
                    "start": max(0.0, w_st - clip_start),
                    "end": max(0.05, w_et - clip_start),
                })

    print(f"[2/5] Extracted {len(selected_words)} dialogue words in target temporal window [{clip_start}s - {clip_end}s]")

    analyzer = ProsodyKineticTypography()
    metrics = analyzer.analyze_clip_words(
        words=selected_words,
        media_path=raw_video,
        clip_start=clip_start,
        clip_end=clip_end,
    )

    print("\n[3/5] Acoustic Prosody Scorecard on Real Dialogue:")
    print(f"{'WORD':<14} | {'TIME (s)':<11} | {'RMS':<7} | {'PITCH':<7} | {'EMPHASIS':<8} | {'SCALE':<5} | {'COLOR':<10}")
    print("-" * 75)

    color_counts: Dict[str, int] = {}
    for m in metrics:
        color_counts[m.active_color] = color_counts.get(m.active_color, 0) + 1
        time_str = f"{m.start:.2f}-{m.end:.2f}"
        print(
            f"{m.word:<14} | {time_str:<11} | {m.rms_energy:<7.3f} | {m.pitch_hz:<7.1f} | "
            f"{m.emphasis_score:<8.2f} | {m.scale_factor:<5} | {m.active_color:<10}"
        )

    print("-" * 75)
    print(f"Color distribution on real speech: {color_counts}")

    assert len(color_counts) >= 2, "Expected diverse emotional color distribution across speech dynamic range"

    print("\n[4/5] Generating kinetic ASS subtitles with prosody transforms...")
    os.makedirs(out_dir, exist_ok=True)
    generated_ass = generate_ass_file(
        input_media=raw_video,
        output_ass=out_ass,
        words=selected_words,
        clip_start=0.0,
        clip_end=(clip_end - clip_start),
        enable_prosody=True,
    )

    with open(generated_ass, "r", encoding="utf-8") as f:
        ass_content = f.read()

    assert "[Script Info]" in ass_content, "Missing ASS header in generated script"
    assert "\\c&H" in ass_content, "Missing dynamic color codes in ASS dialogue"
    assert "\\fscx" in ass_content, "Missing scale transforms in ASS dialogue"
    print(f"[ASS File] Successfully generated at {generated_ass} ({len(ass_content)} bytes)")

    print("\n[5/5] Rendering 9:16 vertical short with burned prosody subtitles...")
    if os.path.exists(out_video):
        os.remove(out_video)

    rendered_path = render_short_clip(
        source_path=raw_video,
        start_time=clip_start,
        end_time=clip_end,
        out_path=out_video,
        framing="center",
        burn_subtitles=True,
        master_audio=True,
        encoder_choice="cpu",
        resolution="1080p",
        aspect_ratio="9:16",
        words=selected_words,
    )

    meta = probe_rendered_short(rendered_path)
    print("\n" + "=" * 80)
    print("RENDERED MEDIA QUALITY AUDIT (ffprobe):")
    print(f"  Destination Path:  {rendered_path}")
    print(f"  Dimensions:        {meta['width']}x{meta['height']} (Target: 1080x1920)")
    print(f"  Video Codec:       {meta['vcodec']}")
    print(f"  Audio Codec:       {meta['acodec']}")
    print(f"  Duration:          {meta['duration']:.2f}s")
    print(f"  File Size:         {meta['size_bytes'] / (1024 * 1024):.2f} MB")
    print("=" * 80)

    assert meta["width"] == 1080, f"Expected 1080 width, got {meta['width']}"
    assert meta["height"] == 1920, f"Expected 1920 height, got {meta['height']}"
    assert meta["duration"] > 10.0, f"Expected >10s duration, got {meta['duration']}"
    assert meta["size_bytes"] > 500000, "Rendered video file is abnormally small"

    print("\n[SUCCESS] REAL VIDEO VERIFICATION COMPLETE: ALL ASSERTIONS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(run_prosody_real_video_verification())
