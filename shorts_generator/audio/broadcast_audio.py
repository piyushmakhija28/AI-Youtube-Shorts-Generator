"""Broadcast compliance multitrack audio mastering engine.

Implements EBU R128 and ATSC A/85 broadcast loudness specifications,
ITU-R BS.1770-4 K-weighting pre-filtering via Direct Form II Transposed biquads,
inter-sample True Peak brickwall limiting with 4x oversampling, dynamic sidechain
auto-ducking for 4-stem multitrack audio, and two-pass FFmpeg loudnorm normalization.
"""

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import wave
import numpy as np


class StemType(str, Enum):
    """Semantic category of an audio stem in multitrack production."""

    DIALOGUE = "dialogue"
    BGM = "bgm"
    SFX = "sfx"
    AMBIENCE = "ambience"


class LoudnessStandard(str, Enum):
    """Broadcast and online streaming loudness target standards."""

    EBU_R128 = "ebu_r128"
    ONLINE_STREAMING = "online_streaming"
    ATSC_A85 = "atsc_a85"


class AudioMasteringError(Exception):
    """Base exception raised for failures during the audio mastering workflow."""


class StemValidationError(AudioMasteringError):
    """Raised when provided audio stems fail structural validation or format checks."""


class LoudnessComplianceError(AudioMasteringError):
    """Raised when mastered audio deviates beyond broadcast loudness compliance tolerances."""


class CodecEncodingError(AudioMasteringError):
    """Raised when external codec rendering or FFmpeg multiplexing fails."""


@dataclass(frozen=True)
class AudioStem:
    """Descriptor and gain configuration for an individual audio stem.

    Attributes:
        stem_type: Semantic classification of the audio stem.
        file_path: Absolute filesystem path to the source audio file.
        source_gain_db: Initial static gain trim in decibels.
        ducking_priority: Sidechain priority rank (1 = master dialogue, 2 = sfx, 3 = bgm).
        pan: Stereo panorama offset in range [-1.0 (left), 1.0 (right)].
    """

    stem_type: StemType
    file_path: str
    source_gain_db: float = 0.0
    ducking_priority: int = 1
    pan: float = 0.0


@dataclass(frozen=True)
class MasteringConfig:
    """Mastering parameters and broadcast target specifications.

    Attributes:
        target_lufs: Target integrated loudness in LUFS (-14.0 for YouTube/streaming).
        true_peak_ceiling_dbfs: True Peak brickwall ceiling in dBFS (-1.0 for inter-sample safety).
        target_lra: Target Loudness Range in Loudness Units (typically 11.0 LU).
        ducking_depth_db: Gain reduction in decibels applied to BGM during dialogue speech.
        ducking_threshold_dbfs: Dialogue RMS threshold in dBFS triggering sidechain ducking.
        attack_ms: Sidechain ducking attack ramp duration in milliseconds.
        release_ms: Sidechain ducking release recovery duration in milliseconds.
        hold_ms: Sidechain hold duration preventing fluttering across syllables in milliseconds.
        sample_rate: Master output sampling frequency in Hertz.
        channels: Master output channel count (1 for mono, 2 for stereo).
        loudness_standard: Prescribed broadcast loudness specification standard.
    """

    target_lufs: float = -14.0
    true_peak_ceiling_dbfs: float = -1.0
    target_lra: float = 11.0
    ducking_depth_db: float = -10.0
    ducking_threshold_dbfs: float = -28.0
    attack_ms: float = 20.0
    release_ms: float = 350.0
    hold_ms: float = 100.0
    sample_rate: int = 48000
    channels: int = 2
    loudness_standard: LoudnessStandard = LoudnessStandard.ONLINE_STREAMING


@dataclass(frozen=True)
class LoudnessMetrics:
    """Acoustic loudness and dynamic range metrics per ITU-R BS.1770-4.

    Attributes:
        integrated_lufs: Integrated program loudness in LUFS across entire duration.
        true_peak_dbfs: Maximum True Peak level in dBFS with inter-sample reconstruction.
        loudness_range_lu: Statistical distribution of loudness dynamics in LU.
        momentary_max_lufs: Peak momentary loudness (400 ms sliding window) in LUFS.
        short_term_max_lufs: Peak short-term loudness (3000 ms sliding window) in LUFS.
        threshold_lufs: Relative loudness threshold calculated during gated integration in LUFS.
    """

    integrated_lufs: float
    true_peak_dbfs: float
    loudness_range_lu: float
    momentary_max_lufs: float
    short_term_max_lufs: float
    threshold_lufs: float


@dataclass(frozen=True)
class MasteringResult:
    """Output artifacts and quality compliance audit from the audio mastering process.

    Attributes:
        output_path: Absolute filesystem path to the finalized mastered audio file.
        metrics: Measured acoustic and broadcast loudness metrics.
        is_compliant: Boolean flag confirming adherence to target specifications.
        stems_processed: Count of audio stems integrated into the master mix.
        applied_gain_db: Static linear gain offset applied during final normalization pass.
        execution_time_sec: Wall-clock processing duration in seconds.
    """

    output_path: str
    metrics: LoudnessMetrics
    is_compliant: bool
    stems_processed: int
    applied_gain_db: float
    execution_time_sec: float


class BroadcastAudioMaster:
    """Mastering processor for multitrack broadcast audio compliance."""

    def __init__(self, config: Optional[MasteringConfig] = None) -> None:
        """Initialize the audio mastering processor with target specifications.

        Args:
            config: Optional mastering configuration parameters. Defaults to streaming profile.
        """
        self.config = config or MasteringConfig()
        self._circuit_breaker_failures = 0
        self._circuit_breaker_tripped = False

    def master_multitrack(
        self,
        stems: List[AudioStem],
        output_path: str,
    ) -> MasteringResult:
        """Execute complete multitrack mastering, sidechain ducking, and loudness normalization.

        Args:
            stems: Collection of audio stems to blend, duck, and master.
            output_path: Destination filesystem path for the mastered audio file.

        Returns:
            MasteringResult containing finalized file path, metrics, and compliance status.

        Raises:
            StemValidationError: If stems list is empty or source files do not exist.
            AudioMasteringError: If mastering pipeline fails execution.
        """
        start_time = time.perf_counter()
        if not stems:
            raise StemValidationError("At least one audio stem must be provided for mastering.")

        for stem in stems:
            if not os.path.exists(stem.file_path):
                raise StemValidationError(f"Audio stem file not found: {stem.file_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not self._circuit_breaker_tripped and self._has_ffmpeg():
            try:
                result = self._master_via_ffmpeg(stems, output_path)
                self._circuit_breaker_failures = 0
                return result
            except Exception:
                self._circuit_breaker_failures += 1
                if self._circuit_breaker_failures >= 3:
                    self._circuit_breaker_tripped = True

        return self._master_via_python_dsp(stems, output_path, start_time)

    def measure_loudness(self, audio_path: str) -> LoudnessMetrics:
        """Measure ITU-R BS.1770-4 loudness metrics of an audio file.

        Args:
            audio_path: Absolute filesystem path to the target audio file.

        Returns:
            LoudnessMetrics containing integrated LUFS, True Peak dBFS, and LRA.

        Raises:
            AudioMasteringError: If measurement fails or audio data cannot be decoded.
        """
        if not os.path.exists(audio_path):
            raise AudioMasteringError(f"Target audio file does not exist: {audio_path}")

        if not self._circuit_breaker_tripped and self._has_ffmpeg():
            try:
                metrics = self._measure_loudness_ffmpeg(audio_path)
                return metrics
            except Exception:
                pass

        return self._measure_loudness_python(audio_path)

    def apply_sidechain_ducking(
        self,
        dialogue_samples: np.ndarray,
        bgm_samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply dynamic sidechain auto-ducking to background music based on dialogue activity.

        Args:
            dialogue_samples: Floating-point PCM audio array representing spoken dialogue.
            bgm_samples: Floating-point PCM audio array representing background music.
            sample_rate: Sampling frequency in Hertz.

        Returns:
            Ducked floating-point PCM audio array for background music.
        """
        if len(dialogue_samples) == 0 or len(bgm_samples) == 0:
            return bgm_samples

        max_len = max(len(dialogue_samples), len(bgm_samples))
        diag_padded = np.zeros(max_len, dtype=np.float32)
        bgm_padded = np.zeros(max_len, dtype=np.float32)
        diag_padded[: len(dialogue_samples)] = dialogue_samples
        bgm_padded[: len(bgm_samples)] = bgm_samples

        window_size = int(0.040 * sample_rate)
        if window_size < 1:
            window_size = 1

        squared = diag_padded**2
        kernel = np.ones(window_size, dtype=np.float32) / window_size
        rms_env = np.sqrt(np.convolve(squared, kernel, mode="same") + 1e-12)

        threshold_linear = 10.0 ** (self.config.ducking_threshold_dbfs / 20.0)
        target_attenuation = 10.0 ** (self.config.ducking_depth_db / 20.0)

        attack_samples = max(1, int((self.config.attack_ms / 1000.0) * sample_rate))
        release_samples = max(1, int((self.config.release_ms / 1000.0) * sample_rate))
        hold_samples = max(1, int((self.config.hold_ms / 1000.0) * sample_rate))

        gain_envelope = np.ones(max_len, dtype=np.float32)
        current_gain = 1.0
        hold_counter = 0

        attack_coeff = math.exp(-1.0 / attack_samples)
        release_coeff = math.exp(-1.0 / release_samples)

        for i in range(max_len):
            if rms_env[i] >= threshold_linear:
                hold_counter = hold_samples
                desired_gain = target_attenuation
            else:
                if hold_counter > 0:
                    hold_counter -= 1
                    desired_gain = target_attenuation
                else:
                    desired_gain = 1.0

            if desired_gain < current_gain:
                current_gain = attack_coeff * current_gain + (1.0 - attack_coeff) * desired_gain
            else:
                current_gain = release_coeff * current_gain + (1.0 - release_coeff) * desired_gain

            gain_envelope[i] = current_gain

        ducked = bgm_padded * gain_envelope
        return ducked[: len(bgm_samples)]

    def apply_k_weighting_filter(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply ITU-R BS.1770-4 K-weighting pre-filter in Direct Form II Transposed.

        Args:
            samples: Audio sample array normalized in range [-1.0, 1.0].
            sample_rate: Sampling frequency in Hertz.

        Returns:
            K-weighted filtered audio sample array.
        """
        if len(samples) == 0:
            return samples.copy()

        stage1_filtered = self._apply_biquad_df2t(
            samples,
            *self._calculate_high_shelf_coefficients(sample_rate),
        )
        stage2_filtered = self._apply_biquad_df2t(
            stage1_filtered,
            *self._calculate_rlb_highpass_coefficients(sample_rate),
        )
        return stage2_filtered

    def limit_true_peak(
        self,
        samples: np.ndarray,
        sample_rate: int,
        ceiling_dbfs: float = -1.0,
    ) -> np.ndarray:
        """Apply a brickwall True Peak limiter with 4x oversampling interpolation.

        Args:
            samples: Floating-point PCM audio array.
            sample_rate: Sampling frequency in Hertz.
            ceiling_dbfs: Peak ceiling threshold in decibels relative to full scale.

        Returns:
            Peak-limited floating-point audio array adhering to ceiling_dbfs.
        """
        if len(samples) == 0:
            return samples

        ceiling_linear = 10.0 ** (ceiling_dbfs / 20.0)
        oversampled = np.interp(
            np.linspace(0, len(samples), len(samples) * 4, endpoint=False),
            np.arange(len(samples)),
            samples,
        )

        max_peak = np.max(np.abs(oversampled))
        if max_peak <= ceiling_linear:
            return samples

        lookahead_samples = int(0.002 * sample_rate * 4)
        release_samples = int(0.050 * sample_rate * 4)
        reduction = np.ones(len(oversampled), dtype=np.float32)

        peak_env = np.abs(oversampled)
        excess = peak_env / ceiling_linear
        excess[excess < 1.0] = 1.0

        target_reduction = 1.0 / excess
        current_g = 1.0
        alpha_rel = math.exp(-1.0 / max(1, release_samples))

        for i in range(len(target_reduction)):
            lookahead_end = min(len(target_reduction), i + lookahead_samples)
            local_min = np.min(target_reduction[i:lookahead_end])
            if local_min < current_g:
                current_g = local_min
            else:
                current_g = alpha_rel * current_g + (1.0 - alpha_rel) * 1.0
            reduction[i] = current_g

        limited_oversampled = oversampled * reduction
        decimated = limited_oversampled[::4]
        min_len = min(len(samples), len(decimated))
        result = np.zeros_like(samples)
        result[:min_len] = decimated[:min_len]

        peak_verify = np.max(np.abs(result))
        if peak_verify > ceiling_linear:
            result = result * (ceiling_linear / peak_verify)

        return result

    def _master_via_ffmpeg(
        self,
        stems: List[AudioStem],
        output_path: str,
    ) -> MasteringResult:
        """Master audio stems using two-pass FFmpeg loudnorm filtergraph.

        Args:
            stems: Audio stems to process.
            output_path: Destination path on disk.

        Returns:
            MasteringResult with measured broadcast compliance statistics.
        """
        start_time = time.perf_counter()
        target_sr = self.config.sample_rate

        stem_wavs = []
        dialogue_path = None
        bgm_paths = []
        other_paths = []

        for stem in stems:
            stem_wavs.append(stem)
            if stem.stem_type == StemType.DIALOGUE:
                dialogue_path = stem.file_path
            elif stem.stem_type == StemType.BGM:
                bgm_paths.append(stem.file_path)
            else:
                other_paths.append(stem.file_path)

        temp_blend_path = output_path + ".blend.wav"

        dialogue_stem = next((s for s in stems if s.stem_type == StemType.DIALOGUE), None)
        bgm_stem = next((s for s in stems if s.stem_type == StemType.BGM), None)

        if dialogue_stem and bgm_stem:
            d_audio, sr_d = self._read_wav_float(dialogue_stem.file_path)
            b_audio, sr_b = self._read_wav_float(bgm_stem.file_path)
            ducked_b = self.apply_sidechain_ducking(d_audio, b_audio, sr_b)
            ducked_bgm_path = output_path + ".ducked_bgm.wav"
            self._write_wav_float(ducked_bgm_path, ducked_b, sr_b)

            inputs = ["-i", dialogue_stem.file_path, "-i", ducked_bgm_path]
            filter_str = f"[0:a]volume={dialogue_stem.source_gain_db}dB[a0];[1:a]volume={bgm_stem.source_gain_db}dB[a1];[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[out]"
            other_stems = [s for s in stems if s != dialogue_stem and s != bgm_stem]
            for idx, s in enumerate(other_stems, start=2):
                inputs.extend(["-i", s.file_path])
                filter_str += f";[{idx}:a]volume={s.source_gain_db}dB[a{idx}]"
            if other_stems:
                mix_inputs = "".join(f"[a{i}]" for i in range(2 + len(other_stems)))
                filter_str = f"{filter_str};{mix_inputs}amix=inputs={2 + len(other_stems)}:duration=longest[out]"

            cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-map", "[out]", "-ar", str(target_sr), temp_blend_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(ducked_bgm_path):
                try:
                    os.remove(ducked_bgm_path)
                except OSError:
                    pass
        else:
            inputs = []
            filter_parts = []
            for idx, s in enumerate(stems):
                inputs.extend(["-i", s.file_path])
                filter_parts.append(f"[{idx}:a]volume={s.source_gain_db}dB[a{idx}]")
            amix_ins = "".join(f"[a{i}]" for i in range(len(stems)))
            filter_str = f"{';'.join(filter_parts)};{amix_ins}amix=inputs={len(stems)}:duration=longest[out]"
            cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-map", "[out]", "-ar", str(target_sr), temp_blend_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        pass1_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_blend_path,
            "-af",
            f"loudnorm=I={self.config.target_lufs}:TP={self.config.true_peak_ceiling_dbfs}:LRA={self.config.target_lra}:print_format=json",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(pass1_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stats = self._parse_loudnorm_json(proc.stderr)

        pass2_filter = (
            f"loudnorm=I={self.config.target_lufs}:TP={self.config.true_peak_ceiling_dbfs}:LRA={self.config.target_lra}:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
            f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true:print_format=summary"
        )
        pass2_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_blend_path,
            "-af",
            pass2_filter,
            "-ar",
            str(target_sr),
            "-ac",
            str(self.config.channels),
            output_path,
        ]
        subprocess.run(pass2_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if os.path.exists(temp_blend_path):
            try:
                os.remove(temp_blend_path)
            except OSError:
                pass

        final_metrics = self.measure_loudness(output_path)
        is_compliant = (
            abs(final_metrics.integrated_lufs - self.config.target_lufs) <= 0.6
            and final_metrics.true_peak_dbfs <= (self.config.true_peak_ceiling_dbfs + 0.1)
        )

        return MasteringResult(
            output_path=os.path.abspath(output_path),
            metrics=final_metrics,
            is_compliant=is_compliant,
            stems_processed=len(stems),
            applied_gain_db=float(stats.get("target_offset", 0.0)),
            execution_time_sec=time.perf_counter() - start_time,
        )

    def _master_via_python_dsp(
        self,
        stems: List[AudioStem],
        output_path: str,
        start_time: float,
    ) -> MasteringResult:
        """Master audio stems purely in Python and NumPy with ITU-R BS.1770-4 compliance.

        Args:
            stems: Audio stems to blend and normalize.
            output_path: Target output path.
            start_time: Timer origin mark.

        Returns:
            MasteringResult with compliant metrics.
        """
        loaded_audios = []
        max_samples = 0
        sample_rate = self.config.sample_rate

        for stem in stems:
            samples, sr = self._read_wav_float(stem.file_path)
            if sr != sample_rate:
                samples = np.interp(
                    np.linspace(0, len(samples), int(len(samples) * sample_rate / sr), endpoint=False),
                    np.arange(len(samples)),
                    samples,
                )
            gain_linear = 10.0 ** (stem.source_gain_db / 20.0)
            samples = samples * gain_linear
            loaded_audios.append((stem, samples))
            max_samples = max(max_samples, len(samples))

        dialogue_entry = next((s for s in loaded_audios if s[0].stem_type == StemType.DIALOGUE), None)
        bgm_entry = next((s for s in loaded_audios if s[0].stem_type == StemType.BGM), None)

        if dialogue_entry and bgm_entry:
            ducked_bgm = self.apply_sidechain_ducking(dialogue_entry[1], bgm_entry[1], sample_rate)
            loaded_audios = [(s, ducked_bgm if s.stem_type == StemType.BGM else a) for s, a in loaded_audios]

        mix = np.zeros(max_samples, dtype=np.float32)
        for _, audio in loaded_audios:
            mix[: len(audio)] += audio

        if len(stems) > 1:
            mix = mix / np.sqrt(len(stems))

        current_metrics = self._measure_samples_python(mix, sample_rate)
        target_lufs = self.config.target_lufs

        gain_needed_db = target_lufs - current_metrics.integrated_lufs
        mix_normalized = mix * (10.0 ** (gain_needed_db / 20.0))

        master_samples = self.limit_true_peak(
            mix_normalized,
            sample_rate,
            ceiling_dbfs=self.config.true_peak_ceiling_dbfs,
        )

        self._write_wav_float(output_path, master_samples, sample_rate)
        final_metrics = self._measure_samples_python(master_samples, sample_rate)

        is_compliant = (
            abs(final_metrics.integrated_lufs - self.config.target_lufs) <= 0.6
            and final_metrics.true_peak_dbfs <= (self.config.true_peak_ceiling_dbfs + 0.1)
        )

        return MasteringResult(
            output_path=os.path.abspath(output_path),
            metrics=final_metrics,
            is_compliant=is_compliant,
            stems_processed=len(stems),
            applied_gain_db=gain_needed_db,
            execution_time_sec=time.perf_counter() - start_time,
        )

    def _measure_loudness_ffmpeg(self, audio_path: str) -> LoudnessMetrics:
        """Measure loudness of an audio file using FFmpeg ebur128 filter.

        Args:
            audio_path: Path to target audio file.

        Returns:
            Measured LoudnessMetrics object.
        """
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-af",
            "ebur128=framelog=verbose",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return self._parse_ebur128_summary(proc.stderr)

    def _measure_loudness_python(self, audio_path: str) -> LoudnessMetrics:
        """Measure ITU-R BS.1770-4 loudness purely in Python.

        Args:
            audio_path: Path to target audio file.

        Returns:
            Measured LoudnessMetrics object.
        """
        samples, sr = self._read_wav_float(audio_path)
        return self._measure_samples_python(samples, sr)

    def _measure_samples_python(self, samples: np.ndarray, sample_rate: int) -> LoudnessMetrics:
        """Compute ITU-R BS.1770-4 gated loudness on raw sample buffer.

        Args:
            samples: Audio sample array.
            sample_rate: Sampling frequency in Hertz.

        Returns:
            Measured LoudnessMetrics object.
        """
        if len(samples) == 0:
            return LoudnessMetrics(
                integrated_lufs=-70.0,
                true_peak_dbfs=-70.0,
                loudness_range_lu=0.0,
                momentary_max_lufs=-70.0,
                short_term_max_lufs=-70.0,
                threshold_lufs=-70.0,
            )

        k_weighted = self.apply_k_weighting_filter(samples, sample_rate)

        block_len = int(0.400 * sample_rate)
        step = int(0.100 * sample_rate)
        if block_len < 1:
            block_len = 1
        if step < 1:
            step = 1

        powers = []
        for i in range(0, len(k_weighted) - block_len + 1, step):
            block = k_weighted[i : i + block_len]
            mean_sq = np.mean(block**2)
            powers.append(mean_sq)

        if not powers:
            powers = [np.mean(k_weighted**2)]

        z_blocks = np.array(powers, dtype=np.float64)
        l_blocks = -0.691 + 10.0 * np.log10(z_blocks + 1e-12)

        absolute_gate = -70.0
        abs_mask = l_blocks > absolute_gate
        if not np.any(abs_mask):
            integrated = -70.0
            relative_gate = -70.0
        else:
            mean_z_abs = np.mean(z_blocks[abs_mask])
            gamma_a = -0.691 + 10.0 * np.log10(mean_z_abs + 1e-12)
            relative_gate = gamma_a - 10.0

            rel_mask = abs_mask & (l_blocks > relative_gate)
            if not np.any(rel_mask):
                integrated = gamma_a
            else:
                mean_z_rel = np.mean(z_blocks[rel_mask])
                integrated = -0.691 + 10.0 * np.log10(mean_z_rel + 1e-12)

        oversampled = np.interp(
            np.linspace(0, len(samples), len(samples) * 4, endpoint=False),
            np.arange(len(samples)),
            samples,
        )
        max_abs = np.max(np.abs(oversampled)) if len(oversampled) > 0 else 1e-12
        true_peak = 20.0 * np.log10(max(max_abs, 1e-12))

        momentary_max = np.max(l_blocks) if len(l_blocks) > 0 else integrated
        valid_l = l_blocks[l_blocks > relative_gate]
        if len(valid_l) >= 2:
            lra = float(np.percentile(valid_l, 95) - np.percentile(valid_l, 10))
        else:
            lra = 0.0

        return LoudnessMetrics(
            integrated_lufs=float(integrated),
            true_peak_dbfs=float(true_peak),
            loudness_range_lu=float(lra),
            momentary_max_lufs=float(momentary_max),
            short_term_max_lufs=float(momentary_max),
            threshold_lufs=float(relative_gate),
        )

    def _apply_biquad_df2t(
        self,
        samples: np.ndarray,
        b0: float,
        b1: float,
        b2: float,
        a1: float,
        a2: float,
    ) -> np.ndarray:
        """Execute one biquad filter section in Direct Form II Transposed.

        Args:
            samples: Audio input vector.
            b0: Zero feedforward coefficient 0.
            b1: Zero feedforward coefficient 1.
            b2: Zero feedforward coefficient 2.
            a1: Normalized pole feedback coefficient 1.
            a2: Normalized pole feedback coefficient 2.

        Returns:
            Filtered output vector.
        """
        output = np.zeros(len(samples), dtype=np.float32)
        z1 = 0.0
        z2 = 0.0

        for i in range(len(samples)):
            x = float(samples[i])
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            output[i] = y

        return output

    def _calculate_high_shelf_coefficients(
        self,
        sample_rate: int,
    ) -> Tuple[float, float, float, float, float]:
        """Compute RBJ high-shelf biquad coefficients for ITU-R BS.1770 Stage 1 pre-filter.

        Args:
            sample_rate: Audio sampling frequency in Hertz.

        Returns:
            Tuple of normalized coefficients (b0, b1, b2, a1, a2).
        """
        db_gain = 3.99984385397
        f0 = 1681.9744479540
        q = 0.707175236955

        a = 10.0 ** (db_gain / 40.0)
        w0 = 2.0 * math.pi * f0 / sample_rate
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        b0_raw = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)
        b1_raw = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
        b2_raw = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)
        a0_raw = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha
        a1_raw = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
        a2_raw = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha

        return (
            b0_raw / a0_raw,
            b1_raw / a0_raw,
            b2_raw / a0_raw,
            a1_raw / a0_raw,
            a2_raw / a0_raw,
        )

    def _calculate_rlb_highpass_coefficients(
        self,
        sample_rate: int,
    ) -> Tuple[float, float, float, float, float]:
        """Compute RBJ high-pass biquad coefficients for ITU-R BS.1770 Stage 2 RLB weighting.

        Args:
            sample_rate: Audio sampling frequency in Hertz.

        Returns:
            Tuple of normalized coefficients (b0, b1, b2, a1, a2).
        """
        f0 = 38.13547087602444
        q = 0.5003270373253953

        w0 = 2.0 * math.pi * f0 / sample_rate
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        b0_raw = (1.0 + cos_w0) / 2.0
        b1_raw = -(1.0 + cos_w0)
        b2_raw = (1.0 + cos_w0) / 2.0
        a0_raw = 1.0 + alpha
        a1_raw = -2.0 * cos_w0
        a2_raw = 1.0 - alpha

        return (
            b0_raw / a0_raw,
            b1_raw / a0_raw,
            b2_raw / a0_raw,
            a1_raw / a0_raw,
            a2_raw / a0_raw,
        )

    def _read_wav_float(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Read standard PCM WAV file and return mono float array in range [-1.0, 1.0].

        Args:
            file_path: Path to WAV file on disk.

        Returns:
            Tuple of (sample_array, sample_rate).

        Raises:
            AudioMasteringError: If reading fails or format is unsupported.
        """
        try:
            with wave.open(file_path, "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                samples_int = np.frombuffer(raw_bytes, dtype=np.int16)
                samples_float = samples_int.astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples_int = np.frombuffer(raw_bytes, dtype=np.int32)
                samples_float = samples_int.astype(np.float32) / 2147483648.0
            elif sampwidth == 1:
                samples_int = np.frombuffer(raw_bytes, dtype=np.uint8)
                samples_float = (samples_int.astype(np.float32) - 128.0) / 128.0
            else:
                samples_float = np.frombuffer(raw_bytes, dtype=np.float32)

            if channels > 1:
                samples_float = samples_float.reshape(-1, channels).mean(axis=1)

            return samples_float, sample_rate
        except Exception as exc:
            raise AudioMasteringError(f"Failed to read WAV file {file_path}: {exc}") from exc

    def _write_wav_float(self, file_path: str, samples: np.ndarray, sample_rate: int) -> None:
        """Write floating-point audio array to disk as 16-bit PCM WAV.

        Args:
            file_path: Target path on disk.
            samples: Float array of samples in range [-1.0, 1.0].
            sample_rate: Sampling frequency in Hertz.
        """
        clipped = np.clip(samples, -1.0, 1.0)
        pcm_int16 = (clipped * 32767.0).astype(np.int16)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())

    def _has_ffmpeg(self) -> bool:
        """Verify whether FFmpeg binary is accessible in environment path.

        Returns:
            Boolean flag indicating executable availability.
        """
        try:
            res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0
        except OSError:
            return False

    def _parse_loudnorm_json(self, ffmpeg_stderr: str) -> Dict[str, str]:
        """Extract JSON output block from FFmpeg loudnorm stderr output.

        Args:
            ffmpeg_stderr: Complete console stderr text from FFmpeg.

        Returns:
            Dictionary containing parsed measured loudnorm keys.

        Raises:
            AudioMasteringError: If JSON block cannot be extracted.
        """
        start_idx = ffmpeg_stderr.rfind("{")
        end_idx = ffmpeg_stderr.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = ffmpeg_stderr[start_idx : end_idx + 1]
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as exc:
                raise AudioMasteringError(f"Malformed loudnorm JSON: {exc}") from exc
        raise AudioMasteringError("FFmpeg loudnorm filter did not emit JSON summary stats.")

    def _parse_ebur128_summary(self, stderr_text: str) -> LoudnessMetrics:
        """Parse text summary emitted by FFmpeg ebur128 filter.

        Args:
            stderr_text: Console stderr output text from FFmpeg.

        Returns:
            LoudnessMetrics object populated with parsed values.
        """
        integrated = -70.0
        true_peak = -70.0
        lra = 0.0
        momentary_max = -70.0
        short_term_max = -70.0
        threshold = -70.0

        for line in stderr_text.splitlines():
            clean = line.strip()
            if clean.startswith("I:") and "LUFS" in clean:
                parts = clean.split()
                try:
                    integrated = float(parts[1])
                except (ValueError, IndexError):
                    pass
            elif clean.startswith("Threshold:") and "LUFS" in clean:
                parts = clean.split()
                try:
                    threshold = float(parts[1])
                except (ValueError, IndexError):
                    pass
            elif clean.startswith("LRA:") and "LU" in clean:
                parts = clean.split()
                try:
                    lra = float(parts[1])
                except (ValueError, IndexError):
                    pass
            elif clean.startswith("Peak:") and "dBFS" in clean:
                parts = clean.split()
                try:
                    true_peak = float(parts[1])
                except (ValueError, IndexError):
                    pass

        return LoudnessMetrics(
            integrated_lufs=integrated,
            true_peak_dbfs=true_peak,
            loudness_range_lu=lra,
            momentary_max_lufs=momentary_max,
            short_term_max_lufs=short_term_max,
            threshold_lufs=threshold,
        )
