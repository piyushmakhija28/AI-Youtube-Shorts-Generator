"""Unit tests for subtitle relative timestamp synchronization and word caching.

Tests positive, negative, and edge-case scenarios for ASS generation,
timestamp normalization, and Whisper hallucination suppression.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from shorts_generator.local.subtitles import (
    build_ass_script,
    format_ass_timestamp,
    generate_ass_file,
    group_into_chunks,
    sanitize_dialogue,
)
from shorts_generator.local.transcriber import sanitize_transcript_segments


class TestSubtitleTimestampNormalization:
    """Test suite for word-level timestamp offset normalization."""

    def test_relative_timestamp_offset_normalization(self, tmp_path: Path):
        """Verify words with absolute timestamps are normalized to relative clip offsets."""
        ass_path = str(tmp_path / "test_offset.ass")
        words = [
            {"word": "HELLO", "start": 332.10, "end": 333.20},
            {"word": "WORLD", "start": 333.30, "end": 334.50},
        ]

        generate_ass_file(
            input_media="mock.mp4",
            output_ass=ass_path,
            margin_v=520,
            target_w=1080,
            target_h=1920,
            words=words,
            clip_start=332.00,
            clip_end=362.00,
        )

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Dialogue: 0,0:00:00.10,0:00:01.30,Default,,0,0,0,,{\\c&H0000FFFF&\\fscx115\\fscy115}HELLO" in content
        assert "0:05:32" not in content

    def test_zero_whisper_calls_when_words_provided(self, tmp_path: Path):
        """Assert faster-whisper is never invoked when pre-cached words are supplied."""
        ass_path = str(tmp_path / "test_cache.ass")
        words = [{"word": "CACHED", "start": 10.0, "end": 11.0}]

        with patch("shorts_generator.local.subtitles.WhisperModel") as mock_whisper:
            generate_ass_file(
                input_media="mock.mp4",
                output_ass=ass_path,
                words=words,
                clip_start=10.0,
                clip_end=20.0,
            )
            mock_whisper.assert_not_called()

        assert Path(ass_path).exists()

    def test_empty_words_list_generates_valid_empty_ass(self, tmp_path: Path):
        """Verify passing an empty list produces a valid ASS header with no dialogue events."""
        ass_path = str(tmp_path / "test_empty.ass")
        generate_ass_file(
            input_media="mock.mp4",
            output_ass=ass_path,
            words=[],
            clip_start=0.0,
            clip_end=30.0,
        )

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "Dialogue:" not in content

    def test_words_outside_clip_boundary_filtered(self, tmp_path: Path):
        """Verify words preceding clip_start or succeeding clip_end are dropped."""
        ass_path = str(tmp_path / "test_filter.ass")
        words = [
            {"word": "BEFORE", "start": 5.0, "end": 8.0},
            {"word": "INSIDE", "start": 12.0, "end": 14.0},
            {"word": "AFTER", "start": 35.0, "end": 38.0},
        ]

        generate_ass_file(
            input_media="mock.mp4",
            output_ass=ass_path,
            words=words,
            clip_start=10.0,
            clip_end=30.0,
        )

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "INSIDE" in content
        assert "BEFORE" not in content
        assert "AFTER" not in content

    def test_word_straddling_clip_start_boundary(self, tmp_path: Path):
        """Verify a word starting before clip_start has its relative start clamped to 0.0s."""
        ass_path = str(tmp_path / "test_straddle.ass")
        words = [
            {"word": "STRADDLE", "start": 9.50, "end": 11.20},
        ]

        generate_ass_file(
            input_media="mock.mp4",
            output_ass=ass_path,
            words=words,
            clip_start=10.00,
            clip_end=20.00,
        )

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Dialogue: 0,0:00:00.00,0:00:01.40" in content

    def test_microsecond_word_duration_padded(self, tmp_path: Path):
        """Verify ultra-short words are padded to a minimum of 0.05 seconds."""
        ass_path = str(tmp_path / "test_micro.ass")
        words = [
            {"word": "QUICK", "start": 10.00, "end": 10.01},
        ]

        generate_ass_file(
            input_media="mock.mp4",
            output_ass=ass_path,
            words=words,
            clip_start=10.00,
            clip_end=20.00,
        )

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Dialogue: 0,0:00:00.00,0:00:00.25" in content


class TestWhisperHallucinationFilter:
    """Test suite for Whisper silence loop and outro suppression."""

    def test_outro_spam_filtered(self):
        """Verify common outro patterns are stripped from segments."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Welcome to the video."},
            {"start": 2.5, "end": 4.0, "text": "Thank you for watching!"},
            {"start": 4.1, "end": 6.0, "text": "Please subscribe to the channel."},
            {"start": 6.2, "end": 8.0, "text": "Here is the main takeaway."},
        ]

        cleaned = sanitize_transcript_segments(segments)
        assert len(cleaned) == 2
        assert cleaned[0]["text"] == "Welcome to the video."
        assert cleaned[1]["text"] == "Here is the main takeaway."

    def test_empty_and_whitespace_segments_dropped(self):
        """Verify empty and whitespace segments are pruned."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "Real dialogue."},
        ]

        cleaned = sanitize_transcript_segments(segments)
        assert len(cleaned) == 1
        assert cleaned[0]["text"] == "Real dialogue."
