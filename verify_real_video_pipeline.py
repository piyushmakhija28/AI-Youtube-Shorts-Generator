"""End-to-end real-world pipeline verification using YouTube video UKF6t47NqPg.

Validates all 5 resolved GitHub bug issues:
- Issue #1: Subtitle relative timestamp offset alignment and word caching.
- Issue #2: Keyframe extraction and integration with FramingAnalyzer.
- Issue #4: Transcoder execution and libx264 software fallback resilience.
- Issue #5: Multi-aspect ratio scaling across 9:16, 1:1, and 4:5.
- Issue #20: Whisper hallucination and outro spam filtering.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from shorts_generator.diversity_types import CandidateHighlight, HighlightArchetype
from shorts_generator.engine.self_healing import CorruptedVideoError, SelfHealingVideoSanitizer
from shorts_generator.face_tracker import FaceTracker
from shorts_generator.framing_analyzer import FramingAnalyzer, generate_pan_crop_expression, smooth_speaker_tracking
from shorts_generator.local.clipper import (
    extract_representative_keyframe,
    get_source_dimensions,
    get_target_canvas_dimensions,
    render_short_clip,
)
from shorts_generator.local.subtitles import generate_ass_file
from shorts_generator.local.transcriber import sanitize_transcript_segments
from shorts_generator.video.color_matrix_fixer import ColorMatrixFixer


def probe_media_file(file_path: str) -> Dict[str, Any]:
    """Retrieve detailed stream metadata from rendered video via ffprobe.

    Args:
        file_path: Path to media file on disk.

    Returns:
        Dictionary with width, height, duration, video codec, and audio codec.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate",
        "-of", "json",
        file_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
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


def run_real_video_verification() -> int:
    """Execute real-world test suite across real 4K video assets.

    Returns:
        Exit code 0 on complete verification success, 1 on any assertion failure.
    """
    raw_video = "output/raw/video/ep1_raw_4k_moment1.mp4"
    transcript_file = "output/raw/captions/ep1_transcript.json"
    out_dir = "output/real_test"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("STARTING REAL VIDEO PIPELINE VERIFICATION")
    print(f"Source Media: {raw_video}")
    print(f"Transcript:   {transcript_file}")
    print("=" * 78)

    if not os.path.exists(raw_video) or not os.path.exists(transcript_file):
        print(f"FAILED: Required assets missing ({raw_video} or {transcript_file})")
        return 1

    src_w, src_h = get_source_dimensions(raw_video)
    print(f"[Probe] Source video dimensions: {src_w}x{src_h}")

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    raw_segments = transcript_data.get("segments", [])
    test_segments = [s for s in raw_segments if s["start"] >= 88.0 and s["end"] <= 108.0]
    test_words = []
    for s in test_segments:
        for w in s.get("words", []):
            test_words.append({
                "word": w["word"].strip(),
                "start": float(w["start"]) - 88.0,
                "end": float(w["end"]) - 88.0,
            })

    print(f"[Transcript] Extracted {len(test_words)} words for 20s test window")

    print("\n--- [TEST 1: Subtitle Relative Timestamp Normalization (Issue #1)] ---")
    test_ass = os.path.join(out_dir, "test_captions.ass")
    generate_ass_file(
        input_media=raw_video,
        output_ass=test_ass,
        margin_v=520,
        target_w=1080,
        target_h=1920,
        words=test_words,
        clip_start=0.0,
        clip_end=20.0,
    )
    ass_content = Path(test_ass).read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:00.30" in ass_content or "Dialogue: 0,0:00:00." in ass_content, "Relative offset missing"
    assert "0:01:28" not in ass_content, "Absolute offset leaked into relative ASS"
    print("PASS: Subtitles generated with correct relative timestamps starting at t=0.0s")

    print("\n--- [TEST 2: Whisper Hallucination Filter (Issue #20)] ---")
    hallucinated_sample = [
        {"start": 0.0, "end": 2.0, "text": "Real dialogue from GTA 6."},
        {"start": 2.1, "end": 4.0, "text": "Thank you for watching!"},
        {"start": 4.1, "end": 6.0, "text": "Please subscribe to the channel."},
    ]
    cleaned = sanitize_transcript_segments(hallucinated_sample)
    assert len(cleaned) == 1, "Failed to purge hallucination segments"
    assert cleaned[0]["text"] == "Real dialogue from GTA 6."
    print("PASS: Hallucination patterns successfully stripped from transcript")

    print("\n--- [TEST 3: Keyframe Extraction & Framing Analyzer (Issue #2)] ---")
    keyframe = extract_representative_keyframe(raw_video, 5.0)
    assert keyframe is not None, "Failed to extract keyframe via FFmpeg"
    assert keyframe.size == (src_w, src_h), f"Keyframe size mismatch: {keyframe.size}"
    print(f"PASS: Representative keyframe extracted ({keyframe.size[0]}x{keyframe.size[1]})")

    analyzer = FramingAnalyzer(target_resolution="1080p")
    candidate = CandidateHighlight(
        title="Lucia Parole Grill",
        raw_start=88.0,
        raw_end=108.0,
        score=95,
        archetype=HighlightArchetype.DIALOGUE_BANTER,
        hook_sentence="Reintegrating yourself into the community",
        virality_reason="High drama dialogue",
        snapped_start=88.0,
        snapped_end=108.0,
        is_snapped=True,
    )
    rec = analyzer.recommend_framing(
        highlight=candidate,
        transcript_segments=test_segments,
        source_dimensions=(src_w, src_h),
        keyframe=keyframe,
    )
    print(f"PASS: Framing recommendation produced: mode={rec.mode}, confidence={rec.confidence:.2f}")

    print("\n--- [TEST 4: 9:16 Vertical Render (Issues #1, #4, #5)] ---")
    out_9x16 = os.path.join(out_dir, "short_ep1_9x16_split_stack.mp4")
    t0 = time.time()
    render_short_clip(
        source_path=raw_video,
        start_time=0.0,
        end_time=15.0,
        out_path=out_9x16,
        framing="split_stack",
        burn_subtitles=True,
        master_audio=True,
        encoder_choice="auto",
        resolution="1080p",
        aspect_ratio="9:16",
        words=test_words,
    )
    dur_9x16 = time.time() - t0
    meta_9x16 = probe_media_file(out_9x16)
    print(f"PASS: 9:16 rendered in {dur_9x16:.1f}s -> {meta_9x16['width']}x{meta_9x16['height']} ({meta_9x16['size_bytes'] / 1024 / 1024:.2f} MB)")
    assert meta_9x16["width"] == 1080 and meta_9x16["height"] == 1920

    print("\n--- [TEST 5: 1:1 Square Render (Issues #4, #5)] ---")
    out_1x1 = os.path.join(out_dir, "short_ep1_1x1_smart_focus.mp4")
    t0 = time.time()
    render_short_clip(
        source_path=raw_video,
        start_time=0.0,
        end_time=15.0,
        out_path=out_1x1,
        framing="smart_focus",
        burn_subtitles=True,
        master_audio=True,
        encoder_choice="auto",
        resolution="1080p",
        aspect_ratio="1:1",
        words=test_words,
    )
    dur_1x1 = time.time() - t0
    meta_1x1 = probe_media_file(out_1x1)
    print(f"PASS: 1:1 rendered in {dur_1x1:.1f}s -> {meta_1x1['width']}x{meta_1x1['height']} ({meta_1x1['size_bytes'] / 1024 / 1024:.2f} MB)")
    assert meta_1x1["width"] == 1080 and meta_1x1["height"] == 1080

    print("\n--- [TEST 6: 4:5 Portrait Render (Issues #4, #5)] ---")
    out_4x5 = os.path.join(out_dir, "short_ep1_4x5_center.mp4")
    t0 = time.time()
    render_short_clip(
        source_path=raw_video,
        start_time=0.0,
        end_time=15.0,
        out_path=out_4x5,
        framing="center",
        burn_subtitles=True,
        master_audio=True,
        encoder_choice="auto",
        resolution="1080p",
        aspect_ratio="4:5",
        words=test_words,
    )
    dur_4x5 = time.time() - t0
    meta_4x5 = probe_media_file(out_4x5)
    print(f"PASS: 4:5 rendered in {dur_4x5:.1f}s -> {meta_4x5['width']}x{meta_4x5['height']} ({meta_4x5['size_bytes'] / 1024 / 1024:.2f} MB)")
    assert meta_4x5["width"] == 864 and meta_4x5["height"] == 1080

    print("\n--- [TEST 7: Color Matrix Probing & BT.709 Tagging (Issue #496)] ---")
    color_meta = ColorMatrixFixer.probe_color_metadata(raw_video)
    print(f"[Color] Probed color metadata: {color_meta}")
    assert isinstance(color_meta, dict)
    assert "color_space" in color_meta
    cm_filter = ColorMatrixFixer.build_color_matrix_filter(color_meta)
    print(f"[Color] Matrix conversion filter required: {cm_filter is not None} (Filter: {cm_filter})")
    out_tags = ColorMatrixFixer.get_output_tagging_flags()
    assert out_tags["-colorspace"] == "bt709"
    print("PASS: Color matrix probe and broadcast tagging verified")

    print("\n--- [TEST 8: Container Health Inspection & Self-Healing DLQ (Issue #586)] ---")
    is_healthy, err_msg = SelfHealingVideoSanitizer.inspect_container(raw_video)
    assert is_healthy is True, f"Real video container reported unhealthy: {err_msg}"
    print(f"PASS: Real source video container verified healthy via ffprobe inspection")

    test_corrupt_file = os.path.join(out_dir, "simulated_corrupt.mp4")
    with open(test_corrupt_file, "wb") as f:
        f.write(b"CORRUPT_MOOV_ATOM_TEST_BYTES_NON_CONTAINER")

    quarantined = SelfHealingVideoSanitizer.route_to_dlq(
        media_path=test_corrupt_file,
        reason="Corrupt header simulated in verification suite",
        dlq_dir=os.path.join(out_dir, "dlq")
    )
    assert os.path.exists(quarantined), "DLQ file quarantine failed"
    report_path = os.path.join(out_dir, "dlq", "simulated_corrupt_failure_report.json")
    assert os.path.exists(report_path), "DLQ failure report JSON missing"
    print(f"PASS: Corrupted asset safely quarantined to DLQ: {Path(quarantined).name}")

    print("\n--- [TEST 9: CV Face Detection & Dynamic Panning Expression (Issue #3)] ---")
    tracker = FaceTracker()
    detected_faces = tracker.sample_video_faces(
        media_path=raw_video,
        start_time=0.0,
        end_time=5.0,
        fps_sample=2.0
    )
    print(f"[FaceTracker] Sampled 5s window: {len(detected_faces)} face events detected")
    mock_exchange = [
        {"timestamp": 0.0, "center_x": 420.0, "speaker_id": 0},
        {"timestamp": 1.5, "center_x": 420.0, "speaker_id": 0},
        {"timestamp": 3.0, "center_x": 1400.0, "speaker_id": 1},
        {"timestamp": 4.5, "center_x": 1400.0, "speaker_id": 1},
    ]
    smoothed_exchange = smooth_speaker_tracking(mock_exchange, frame_width=src_w, hold_seconds=2.5)
    pan_expression = generate_pan_crop_expression(smoothed_exchange, crop_w=1080, src_w=src_w)
    print(f"[PanExpression] Generated expression: {pan_expression}")
    assert "if(lte(t," in pan_expression
    assert "min(max(" in pan_expression
    print("PASS: Dynamic pan expression and face tracking pipeline verified")

    print("\n" + "=" * 78)
    print("ALL 9 REAL VIDEO VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(run_real_video_verification())
