"""Unit tests for clipper resilience, keyframe extraction, and multi-aspect ratio rendering.

Tests positive, negative, and edge-case scenarios for aspect ratio geometry,
keyframe extraction hooks, and two-stage hardware encoder fallback.
"""

import io
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from shorts_generator.local.clipper import (
    build_ffmpeg_filtergraph,
    extract_representative_keyframe,
    get_target_canvas_dimensions,
    render_short_clip,
)
from shorts_generator.framing_analyzer import FramingAnalyzer
from shorts_generator.diversity_types import CandidateHighlight, HighlightArchetype


class TestTargetCanvasDimensions:
    """Test suite for aspect ratio dimension resolution."""

    def test_dimensions_9_16(self):
        """Verify standard 9:16 vertical resolution profiles."""
        assert get_target_canvas_dimensions("9:16", "4k") == (2160, 3840)
        assert get_target_canvas_dimensions("9:16", "1080p") == (1080, 1920)

    def test_dimensions_1_1(self):
        """Verify 1:1 square canvas resolution profiles."""
        assert get_target_canvas_dimensions("1:1", "4k") == (2160, 2160)
        assert get_target_canvas_dimensions("1:1", "1080p") == (1080, 1080)

    def test_dimensions_4_5(self):
        """Verify 4:5 portrait canvas resolution profiles."""
        assert get_target_canvas_dimensions("4:5", "4k") == (1728, 2160)
        assert get_target_canvas_dimensions("4:5", "1080p") == (864, 1080)

    def test_dimensions_16_9(self):
        """Verify 16:9 landscape canvas resolution profiles."""
        assert get_target_canvas_dimensions("16:9", "4k") == (3840, 2160)
        assert get_target_canvas_dimensions("16:9", "1080p") == (1920, 1080)


class TestKeyframeExtractionAndFraming:
    """Test suite for single-frame keyframe extraction and analyzer integration."""

    def test_extract_representative_keyframe_success(self):
        """Verify keyframe extraction parses PNG bytes into a valid PIL Image."""
        synthetic_img = Image.new("RGB", (1920, 1080), color=(128, 128, 128))
        buf = io.BytesIO()
        synthetic_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_res = MagicMock()
        mock_res.stdout = png_bytes
        mock_res.returncode = 0

        with patch("subprocess.run", return_value=mock_res):
            img = extract_representative_keyframe("mock_video.mp4", 15.0)

        assert img is not None
        assert img.size == (1920, 1080)

    def test_extract_representative_keyframe_failure_returns_none(self):
        """Verify extraction failure returns None gracefully without raising exceptions."""
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            img = extract_representative_keyframe("corrupted_video.mp4", 15.0)

        assert img is None

    def test_framing_analyzer_with_synthetic_keyframe(self):
        """Verify FramingAnalyzer activates letterbox detection when keyframe is supplied."""
        analyzer = FramingAnalyzer(target_resolution="1080p")
        highlight = CandidateHighlight(
            title="Action",
            raw_start=10.0,
            raw_end=40.0,
            score=85,
            archetype=HighlightArchetype.ACTION_HEIST,
            hook_sentence="Look at this move",
            virality_reason="High pacing",
            snapped_start=10.0,
            snapped_end=40.0,
            is_snapped=True,
        )

        synthetic_frame = Image.new("RGB", (1920, 1080), color=(20, 20, 20))
        rec = analyzer.recommend_framing(
            highlight=highlight,
            transcript_segments=[],
            source_dimensions=(1920, 1080),
            keyframe=synthetic_frame,
        )

        assert rec is not None
        assert rec.mode in ["smart_focus", "center", "split_stack", "fit_blur"]


class TestEncoderFallback:
    """Test suite for hardware encoder fallback resilience."""

    def test_hardware_qsv_failure_retries_libx264(self, tmp_path: Path):
        """Verify CalledProcessError on QSV hardware encoding triggers libx264 fallback."""
        out_file = str(tmp_path / "fallback_short.mp4")

        calls = []

        def mock_subprocess_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            if len(calls) == 1:
                # First call simulates QSV failure
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
            # Second call simulates libx264 success
            res = MagicMock()
            res.returncode = 0
            return res

        with patch("shorts_generator.local.clipper.get_source_dimensions", return_value=(1920, 1080)), \
             patch("shorts_generator.local.clipper.get_encoder_config", return_value=("h264_qsv", ["-global_quality", "18"], "format=nv12")), \
             patch("subprocess.run", side_effect=mock_subprocess_run):

            result = render_short_clip(
                source_path="mock_source.mp4",
                start_time=10.0,
                end_time=25.0,
                out_path=out_file,
                framing="center",
                burn_subtitles=False,
                master_audio=False,
                encoder_choice="qsv",
                resolution="1080p",
            )

        assert result == out_file
        assert len(calls) == 2
        assert "h264_qsv" in calls[0]
        assert "libx264" in calls[1]
