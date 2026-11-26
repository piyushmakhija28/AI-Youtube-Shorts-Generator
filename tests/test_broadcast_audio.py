"""Triad unit and integration test suite for broadcast compliance multitrack audio mastering.

Validates ITU-R BS.1770-4 K-weighting pre-filtering in DF2T, 4x True Peak brickwall
limiting, dynamic sidechain auto-ducking, EBU R128 loudness convergence, and
resilience against corrupted or clipping audio stems.
"""

import math
import os
import tempfile
from typing import List, Tuple
import wave
import numpy as np
import pytest

from shorts_generator.audio import (
    AudioMasteringError,
    AudioStem,
    BroadcastAudioMaster,
    LoudnessMetrics,
    LoudnessStandard,
    MasteringConfig,
    MasteringResult,
    StemType,
    StemValidationError,
)


def _generate_synthetic_stem(
    file_path: str,
    duration_sec: float = 2.0,
    sample_rate: int = 48000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
) -> str:
    """Generate a clean synthetic PCM WAV file for deterministic audio testing.

    Args:
        file_path: Filesystem path to create.
        duration_sec: Stem audio duration in seconds.
        sample_rate: Sampling frequency in Hertz.
        frequency_hz: Tone frequency in Hertz.
        amplitude: Peak amplitude normalized in range [0.0, 1.0].

    Returns:
        Absolute filesystem path to the created WAV file.
    """
    total_samples = int(duration_sec * sample_rate)
    time_pts = np.linspace(0, duration_sec, total_samples, endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency_hz * time_pts)
    pcm_int16 = (waveform * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return os.path.abspath(file_path)


def _generate_speech_like_stem(
    file_path: str,
    duration_sec: float = 2.5,
    sample_rate: int = 48000,
) -> str:
    """Generate a speech-like audio stem with bursts of vocal energy separated by pauses.

    Args:
        file_path: Destination path on disk.
        duration_sec: Audio duration in seconds.
        sample_rate: Sampling frequency in Hertz.

    Returns:
        Absolute filesystem path to the created WAV file.
    """
    total_samples = int(duration_sec * sample_rate)
    time_pts = np.linspace(0, duration_sec, total_samples, endpoint=False)
    carrier = 0.6 * np.sin(2 * np.pi * 220.0 * time_pts)

    envelope = np.zeros(total_samples, dtype=np.float32)
    p1_start = int(0.2 * sample_rate)
    p1_end = int(1.0 * sample_rate)
    p2_start = int(1.4 * sample_rate)
    p2_end = int(2.2 * sample_rate)

    envelope[p1_start:p1_end] = 1.0
    envelope[p2_start:p2_end] = 1.0

    waveform = carrier * envelope
    pcm_int16 = (waveform * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return os.path.abspath(file_path)


class TestBroadcastAudioMasterPositive:
    """Positive test scenarios verifying compliance with broadcast loudness specifications."""

    def test_master_multitrack_4_stems_convergence(self, tmp_path: str) -> None:
        """Verify that mastering 4 stems achieves target -14 LUFS and True Peak ceiling."""
        d_path = _generate_speech_like_stem(os.path.join(str(tmp_path), "dialogue.wav"))
        b_path = _generate_synthetic_stem(os.path.join(str(tmp_path), "bgm.wav"), frequency_hz=300.0, amplitude=0.3)
        s_path = _generate_synthetic_stem(os.path.join(str(tmp_path), "sfx.wav"), frequency_hz=800.0, amplitude=0.2)
        a_path = _generate_synthetic_stem(os.path.join(str(tmp_path), "ambience.wav"), frequency_hz=150.0, amplitude=0.05)

        stems = [
            AudioStem(stem_type=StemType.DIALOGUE, file_path=d_path, source_gain_db=0.0, ducking_priority=1),
            AudioStem(stem_type=StemType.BGM, file_path=b_path, source_gain_db=-3.0, ducking_priority=3),
            AudioStem(stem_type=StemType.SFX, file_path=s_path, source_gain_db=-6.0, ducking_priority=2),
            AudioStem(stem_type=StemType.AMBIENCE, file_path=a_path, source_gain_db=-12.0, ducking_priority=3),
        ]

        master = BroadcastAudioMaster(MasteringConfig(target_lufs=-14.0, true_peak_ceiling_dbfs=-1.0))
        out_path = os.path.join(str(tmp_path), "mastered_4stems.wav")
        result = master.master_multitrack(stems, out_path)

        assert os.path.exists(result.output_path)
        assert result.stems_processed == 4
        assert abs(result.metrics.integrated_lufs - (-14.0)) <= 0.6
        assert result.metrics.true_peak_dbfs <= -0.9
        assert result.is_compliant is True

    def test_sidechain_ducking_attenuates_bgm_during_speech(self) -> None:
        """Verify that dynamic sidechain auto-ducking attenuates background music when dialogue speaks."""
        sample_rate = 48000
        duration_sec = 2.0
        total_samples = int(duration_sec * sample_rate)

        dialogue = np.zeros(total_samples, dtype=np.float32)
        speech_start = int(0.5 * sample_rate)
        speech_end = int(1.5 * sample_rate)
        dialogue[speech_start:speech_end] = 0.5 * np.sin(2 * np.pi * 300.0 * np.linspace(0, 1.0, speech_end - speech_start))

        bgm = 0.4 * np.ones(total_samples, dtype=np.float32)

        master = BroadcastAudioMaster(MasteringConfig(ducking_depth_db=-10.0, ducking_threshold_dbfs=-25.0))
        ducked_bgm = master.apply_sidechain_ducking(dialogue, bgm, sample_rate)

        quiet_rms = np.sqrt(np.mean(ducked_bgm[speech_start + 500 : speech_end - 500] ** 2))
        unattenuated_rms = np.sqrt(np.mean(ducked_bgm[: speech_start - 500] ** 2))

        attenuation_db = 20.0 * math.log10(quiet_rms / unattenuated_rms)
        assert attenuation_db <= -7.0

    def test_k_weighting_filter_biquads(self) -> None:
        """Verify ITU-R BS.1770-4 Stage 1 high-shelf and Stage 2 high-pass filter responses."""
        master = BroadcastAudioMaster()
        sample_rate = 48000
        n_samples = 48000
        t = np.linspace(0, 1.0, n_samples, endpoint=False)

        high_freq_tone = 0.5 * np.sin(2 * np.pi * 3000.0 * t)
        filtered_high = master.apply_k_weighting_filter(high_freq_tone, sample_rate)
        rms_in_high = np.sqrt(np.mean(high_freq_tone**2))
        rms_out_high = np.sqrt(np.mean(filtered_high**2))
        boost_db = 20.0 * math.log10(rms_out_high / rms_in_high)
        assert boost_db > 2.0

        low_freq_tone = 0.5 * np.sin(2 * np.pi * 20.0 * t)
        filtered_low = master.apply_k_weighting_filter(low_freq_tone, sample_rate)
        rms_in_low = np.sqrt(np.mean(low_freq_tone**2))
        rms_out_low = np.sqrt(np.mean(filtered_low**2))
        cut_db = 20.0 * math.log10(rms_out_low / rms_in_low)
        assert cut_db < -3.0

    def test_true_peak_limiter_inter_sample_clamping(self) -> None:
        """Verify that 4x oversampled True Peak brickwall limiter clamps signals exceeding ceiling."""
        master = BroadcastAudioMaster()
        sample_rate = 48000
        n_samples = 24000
        t = np.linspace(0, 0.5, n_samples, endpoint=False)
        clipping_signal = 1.8 * np.sin(2 * np.pi * 1000.0 * t)

        limited = master.limit_true_peak(clipping_signal, sample_rate, ceiling_dbfs=-1.0)
        oversampled = np.interp(
            np.linspace(0, len(limited), len(limited) * 4, endpoint=False),
            np.arange(len(limited)),
            limited,
        )
        max_tp = np.max(np.abs(oversampled))
        tp_dbfs = 20.0 * math.log10(max_tp)
        assert tp_dbfs <= -0.95

    def test_pure_python_dsp_mastering_fallback(self, tmp_path: str) -> None:
        """Verify that Python DSP fallback produces compliant output when circuit breaker is active."""
        d_path = _generate_speech_like_stem(os.path.join(str(tmp_path), "dialogue_py.wav"))
        b_path = _generate_synthetic_stem(os.path.join(str(tmp_path), "bgm_py.wav"), frequency_hz=250.0, amplitude=0.4)

        stems = [
            AudioStem(stem_type=StemType.DIALOGUE, file_path=d_path),
            AudioStem(stem_type=StemType.BGM, file_path=b_path),
        ]

        master = BroadcastAudioMaster(MasteringConfig(target_lufs=-14.0, true_peak_ceiling_dbfs=-1.0))
        master._circuit_breaker_tripped = True
        out_path = os.path.join(str(tmp_path), "mastered_python_fallback.wav")
        result = master.master_multitrack(stems, out_path)

        assert os.path.exists(result.output_path)
        assert abs(result.metrics.integrated_lufs - (-14.0)) <= 0.6
        assert result.metrics.true_peak_dbfs <= -0.9
        assert result.is_compliant is True


class TestBroadcastAudioMasterNegative:
    """Negative test scenarios verifying deterministic error handling and validation."""

    def test_empty_stems_raises_validation_error(self, tmp_path: str) -> None:
        """Verify that passing an empty list of stems raises StemValidationError."""
        master = BroadcastAudioMaster()
        out_path = os.path.join(str(tmp_path), "output.wav")
        with pytest.raises(StemValidationError, match="At least one audio stem"):
            master.master_multitrack([], out_path)

    def test_missing_stem_file_raises_validation_error(self, tmp_path: str) -> None:
        """Verify that a non-existent stem file raises StemValidationError."""
        master = BroadcastAudioMaster()
        out_path = os.path.join(str(tmp_path), "output.wav")
        ghost_stem = AudioStem(stem_type=StemType.DIALOGUE, file_path="C:/nonexistent/path/ghost.wav")
        with pytest.raises(StemValidationError, match="Audio stem file not found"):
            master.master_multitrack([ghost_stem], out_path)

    def test_measure_nonexistent_audio_raises_error(self) -> None:
        """Verify that measuring an unreadable audio file raises AudioMasteringError."""
        master = BroadcastAudioMaster()
        with pytest.raises(AudioMasteringError, match="Target audio file does not exist"):
            master.measure_loudness("C:/nonexistent/file.wav")

    def test_corrupt_audio_file_raises_error(self, tmp_path: str) -> None:
        """Verify that reading a corrupted WAV file cleanly raises AudioMasteringError."""
        corrupt_path = os.path.join(str(tmp_path), "corrupt.wav")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_WAV_HEADER_DATA")

        master = BroadcastAudioMaster()
        with pytest.raises(AudioMasteringError, match="Failed to read WAV file"):
            master.measure_loudness(corrupt_path)


class TestBroadcastAudioMasterEdgeCases:
    """Edge-case tests verifying behavior with extreme dynamics, single stems, and mismatches."""

    def test_single_stem_mastering(self, tmp_path: str) -> None:
        """Verify that mastering operates correctly when only a single stem is supplied."""
        solo_path = _generate_speech_like_stem(os.path.join(str(tmp_path), "solo.wav"))
        stem = AudioStem(stem_type=StemType.DIALOGUE, file_path=solo_path)

        master = BroadcastAudioMaster(MasteringConfig(target_lufs=-14.0))
        out_path = os.path.join(str(tmp_path), "mastered_solo.wav")
        result = master.master_multitrack([stem], out_path)

        assert os.path.exists(result.output_path)
        assert result.stems_processed == 1
        assert abs(result.metrics.integrated_lufs - (-14.0)) <= 0.6

    def test_severely_clipping_input_audio(self, tmp_path: str) -> None:
        """Verify that input audio with extreme clipping (+6 dBFS) is safely brought into spec."""
        clip_path = _generate_synthetic_stem(
            os.path.join(str(tmp_path), "clipped.wav"),
            amplitude=2.0,
        )
        stem = AudioStem(stem_type=StemType.SFX, file_path=clip_path)

        master = BroadcastAudioMaster(MasteringConfig(target_lufs=-14.0, true_peak_ceiling_dbfs=-1.0))
        out_path = os.path.join(str(tmp_path), "mastered_clipped.wav")
        result = master.master_multitrack([stem], out_path)

        assert os.path.exists(result.output_path)
        assert result.metrics.true_peak_dbfs <= -0.9
        assert not np.isnan(result.metrics.integrated_lufs)

    def test_near_silent_audio_mastering(self, tmp_path: str) -> None:
        """Verify that near-silent audio (-50 LUFS) is processed without division by zero."""
        quiet_path = _generate_synthetic_stem(
            os.path.join(str(tmp_path), "quiet.wav"),
            amplitude=0.001,
        )
        stem = AudioStem(stem_type=StemType.AMBIENCE, file_path=quiet_path)

        master = BroadcastAudioMaster(MasteringConfig(target_lufs=-14.0))
        out_path = os.path.join(str(tmp_path), "mastered_quiet.wav")
        result = master.master_multitrack([stem], out_path)

        assert os.path.exists(result.output_path)
        assert not np.isnan(result.metrics.integrated_lufs)

    def test_different_durations_multitrack_alignment(self, tmp_path: str) -> None:
        """Verify that stems with varying durations are blended and ducked without truncation."""
        d_path = _generate_speech_like_stem(os.path.join(str(tmp_path), "short_dialogue.wav"), duration_sec=1.0)
        b_path = _generate_synthetic_stem(os.path.join(str(tmp_path), "long_bgm.wav"), duration_sec=3.0)

        stems = [
            AudioStem(stem_type=StemType.DIALOGUE, file_path=d_path),
            AudioStem(stem_type=StemType.BGM, file_path=b_path),
        ]

        master = BroadcastAudioMaster(MasteringConfig(target_lufs=-14.0))
        out_path = os.path.join(str(tmp_path), "mastered_varying_lengths.wav")
        result = master.master_multitrack(stems, out_path)

        assert os.path.exists(result.output_path)
        assert result.stems_processed == 2

    def test_sample_rate_resampling_conversion(self, tmp_path: str) -> None:
        """Verify that a stem recorded at 44100 Hz is automatically adapted to 48000 Hz master."""
        sr_44k_path = _generate_synthetic_stem(
            os.path.join(str(tmp_path), "stem_44k.wav"),
            sample_rate=44100,
            duration_sec=1.5,
        )
        stem = AudioStem(stem_type=StemType.DIALOGUE, file_path=sr_44k_path)

        master = BroadcastAudioMaster(MasteringConfig(sample_rate=48000, target_lufs=-14.0))
        out_path = os.path.join(str(tmp_path), "mastered_resampled.wav")
        result = master.master_multitrack([stem], out_path)

        assert os.path.exists(result.output_path)
        with wave.open(result.output_path, "rb") as wf:
            assert wf.getframerate() == 48000

    def test_zero_length_buffers_handling(self) -> None:
        """Verify that zero-length sample buffers return gracefully without exceptions."""
        master = BroadcastAudioMaster()
        empty = np.array([], dtype=np.float32)

        ducked = master.apply_sidechain_ducking(empty, empty, 48000)
        assert len(ducked) == 0

        filtered = master.apply_k_weighting_filter(empty, 48000)
        assert len(filtered) == 0

        limited = master.limit_true_peak(empty, 48000)
        assert len(limited) == 0
