"""Triad unit and integration test suite for prosody-reactive kinetic typography.

Validates acoustic feature extraction (RMS energy and F0 pitch), squash-and-stretch
spring transforms, dynamic color mapping, error resilience, and libass script generation.
"""

import math
import os
import tempfile
import wave
from typing import Any, Dict, List
import numpy as np
import pytest

from shorts_generator.local.subtitles import (
    build_ass_script,
    generate_ass_file,
    group_into_chunks,
)
from shorts_generator.video.prosody_typography import (
    KineticStyleConfig,
    ProsodyKineticTypography,
    ProsodyWordMetric,
)


def _generate_synthetic_wav(
    file_path: str,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    frequency_hz: float = 220.0,
    amplitude: float = 0.5,
) -> str:
    """Generate a clean synthetic sine wave WAV file for deterministic acoustic testing.

    Args:
        file_path: Target destination path on disk.
        duration_sec: Audio duration in seconds.
        sample_rate: Sampling frequency in Hertz.
        frequency_hz: Tone frequency in Hertz.
        amplitude: Sine wave peak amplitude normalized in range [0.0, 1.0].

    Returns:
        Absolute path to the created WAV file.
    """
    total_samples = int(duration_sec * sample_rate)
    time_points = np.linspace(0, duration_sec, total_samples, endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency_hz * time_points)
    pcm_int16 = (waveform * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_int16.tobytes())

    return os.path.abspath(file_path)


class TestProsodyPositiveScenarios:
    """Positive execution path tests for prosody typography feature extraction and rendering."""

    def test_compute_rms_energy_sine_wave(self) -> None:
        """Verify RMS calculation matches theoretical amplitude divided by sqrt(2)."""
        sample_rate = 16000
        duration = 0.5
        amplitude = 0.8
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        samples = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        rms = ProsodyKineticTypography.compute_rms_energy(samples)
        expected_rms = amplitude / math.sqrt(2)
        assert abs(rms - expected_rms) < 0.02

    def test_estimate_f0_pitch_vocal_harmonic(self) -> None:
        """Verify normalized cross-correlation tracks vocal pitch accurately within 5 Hz tolerance."""
        sample_rate = 16000
        duration = 0.3
        target_pitch = 210.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        samples = (0.6 * np.sin(2 * np.pi * target_pitch * t)).astype(np.float32)

        detected_pitch = ProsodyKineticTypography.estimate_f0_pitch(samples, sample_rate=sample_rate)
        assert abs(detected_pitch - target_pitch) < 5.0

    def test_analyze_word_prosody_explosive_punchline(self) -> None:
        """Verify high-amplitude audio triggers explosive color code and squash-stretch transforms."""
        analyzer = ProsodyKineticTypography()
        sample_rate = 16000
        duration = 0.4
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        high_energy_samples = (0.95 * np.sin(2 * np.pi * 320.0 * t)).astype(np.float32)

        metric = analyzer.analyze_word_prosody(
            word="BOOM",
            start_time=0.0,
            end_time=0.4,
            audio_samples=high_energy_samples,
            sample_rate=sample_rate,
            global_rms_baseline=0.02,
            global_rms_peak=0.30,
        )

        assert metric.word == "BOOM"
        assert metric.emphasis_score >= 0.70
        assert metric.scale_factor >= 135
        assert metric.active_color == analyzer.config.color_explosive
        assert "\\t(" in metric.kinetic_tags
        assert "\\fscx" in metric.kinetic_tags

    def test_analyze_word_prosody_neutral_speech(self) -> None:
        """Verify low-amplitude audio generates baseline scale and neutral yellow color."""
        analyzer = ProsodyKineticTypography()
        sample_rate = 16000
        duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        quiet_samples = (0.02 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

        metric = analyzer.analyze_word_prosody(
            word="whisper",
            start_time=0.0,
            end_time=0.3,
            audio_samples=quiet_samples,
            sample_rate=sample_rate,
            global_rms_baseline=0.02,
            global_rms_peak=0.30,
        )

        assert metric.emphasis_score < 0.20
        assert metric.scale_factor == analyzer.config.min_scale
        assert metric.active_color == analyzer.config.color_neutral
        assert "\\t(" not in metric.kinetic_tags

    def test_build_ass_script_with_metrics(self) -> None:
        """Verify build_ass_script injects kinetic tags and dynamic colors when metrics are passed."""
        chunks = [[{"word": "SUPER", "start": 1.0, "end": 1.4}]]
        metric = ProsodyWordMetric(
            word="SUPER",
            start=1.0,
            end=1.4,
            rms_energy=0.35,
            pitch_hz=280.0,
            emphasis_score=0.9,
            scale_factor=142,
            active_color="&H001155FF",
            outline_color="&H00000000",
            kinetic_tags="\\t(0,70,\\fscx165\\fscy120)\\t(70,160,\\fscx125\\fscy145)",
        )

        ass_text = build_ass_script(chunks, metrics=[metric])
        assert "[Script Info]" in ass_text
        assert "[Events]" in ass_text
        assert "\\c&H001155FF&" in ass_text
        assert "\\t(0,70,\\fscx165\\fscy120)" in ass_text
        assert "SUPER" in ass_text

    def test_generate_ass_file_end_to_end(self, tmp_path) -> None:
        """Verify end-to-end ASS generation with synthetic audio and word list creates valid output."""
        wav_path = str(tmp_path / "voice_sample.wav")
        out_ass = str(tmp_path / "subtitles.ass")
        _generate_synthetic_wav(wav_path, duration_sec=2.0, frequency_hz=200.0, amplitude=0.7)

        test_words = [
            {"word": "VIRAL", "start": 0.2, "end": 0.6},
            {"word": "SHORTS", "start": 0.7, "end": 1.1},
        ]

        result_path = generate_ass_file(
            input_media=wav_path,
            output_ass=out_ass,
            words=test_words,
            clip_start=0.0,
            clip_end=2.0,
            enable_prosody=True,
        )

        assert os.path.exists(result_path)
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "VIRAL" in content
        assert "SHORTS" in content
        assert "[V4+ Styles]" in content


class TestProsodyNegativeScenarios:
    """Negative and failure recovery path tests verifying graceful degradation."""

    def test_extract_pcm_nonexistent_file_raises(self) -> None:
        """Verify FileNotFoundError is raised when attempting to read nonexistent media file."""
        with pytest.raises(FileNotFoundError):
            ProsodyKineticTypography.extract_pcm_from_media(
                media_path="nonexistent_phantom_track.wav",
                start_time=0.0,
                duration=1.0,
            )

    def test_analyze_clip_words_fallback_on_unreadable_file(self) -> None:
        """Verify analyze_clip_words returns safe baseline metrics when audio extraction fails."""
        analyzer = ProsodyKineticTypography()
        words = [{"word": "FALLBACK", "start": 0.0, "end": 0.5}]

        metrics = analyzer.analyze_clip_words(
            words=words,
            media_path="invalid_path_to_audio_file.wav",
            clip_start=0.0,
            clip_end=1.0,
        )

        assert len(metrics) == 1
        assert metrics[0].word == "FALLBACK"
        assert metrics[0].scale_factor == analyzer.config.min_scale
        assert metrics[0].active_color == analyzer.config.color_neutral

    def test_estimate_f0_too_short_samples(self) -> None:
        """Verify pitch estimation returns 0.0 Hz safely when sample array is too short."""
        sample_rate = 16000
        short_samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        pitch = ProsodyKineticTypography.estimate_f0_pitch(short_samples, sample_rate=sample_rate)
        assert pitch == 0.0

    def test_estimate_f0_unvoiced_white_noise(self) -> None:
        """Verify pitch estimation returns 0.0 Hz for unvoiced pseudo-random noise."""
        np.random.seed(42)
        noise = np.random.uniform(-0.1, 0.1, 1600).astype(np.float32)
        pitch = ProsodyKineticTypography.estimate_f0_pitch(noise, sample_rate=16000)
        assert pitch == 0.0


class TestProsodyEdgeCases:
    """Boundary condition and edge case testing for extreme inputs."""

    def test_compute_rms_energy_pure_silence(self) -> None:
        """Verify RMS computation on all-zero array returns 0.0 without numerical exceptions."""
        silence = np.zeros(1600, dtype=np.float32)
        rms = ProsodyKineticTypography.compute_rms_energy(silence)
        assert rms == 0.0

    def test_compute_rms_energy_empty_array(self) -> None:
        """Verify RMS computation on empty array returns 0.0."""
        empty = np.zeros(0, dtype=np.float32)
        rms = ProsodyKineticTypography.compute_rms_energy(empty)
        assert rms == 0.0

    def test_analyze_word_prosody_clipped_maximum_amplitude(self) -> None:
        """Verify maximum clamped square wave amplitude evaluates cleanly at emphasis 1.0."""
        analyzer = ProsodyKineticTypography()
        clipped = np.ones(800, dtype=np.float32)
        metric = analyzer.analyze_word_prosody(
            word="MAX",
            start_time=0.0,
            end_time=0.05,
            audio_samples=clipped,
            sample_rate=16000,
            global_rms_baseline=0.05,
            global_rms_peak=0.50,
        )
        assert metric.emphasis_score == 1.0
        assert metric.scale_factor == analyzer.config.max_scale
        assert metric.active_color == analyzer.config.color_explosive

    def test_analyze_clip_words_empty_input(self) -> None:
        """Verify empty word list produces empty metric list."""
        analyzer = ProsodyKineticTypography()
        metrics = analyzer.analyze_clip_words(
            words=[],
            media_path="dummy.wav",
            clip_start=0.0,
            clip_end=1.0,
        )
        assert metrics == []

    def test_build_ass_script_special_character_escaping(self) -> None:
        """Verify subtitle formatting preserves syntax stability when text contains special ASS symbols."""
        chunks = [[{"word": "{DANGER} \\TEST/", "start": 0.0, "end": 0.5}]]
        ass_content = build_ass_script(chunks)
        assert "Dialogue: 0," in ass_content
        assert "\\c&H0000FFFF&" in ass_content

    def test_microsecond_word_duration(self) -> None:
        """Verify extremely short 5ms word slice produces non-empty output and valid timestamps."""
        analyzer = ProsodyKineticTypography()
        samples = np.zeros(1600, dtype=np.float32)
        metric = analyzer.analyze_word_prosody(
            word="A",
            start_time=0.100,
            end_time=0.105,
            audio_samples=samples,
            sample_rate=16000,
        )
        assert metric.word == "A"
        assert metric.scale_factor >= analyzer.config.min_scale
