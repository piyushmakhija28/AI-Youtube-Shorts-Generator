"""Prosody-reactive kinetic typography engine for dynamic subtitle generation.

Extracts acoustic prosody (RMS energy and fundamental vocal pitch F0) per spoken
syllable/word, and synthesizes dynamic squash-and-stretch spring physics into
Advanced SubStation Alpha (.ass) dialogue events for high-retention short-form video.
"""

from dataclasses import dataclass
import math
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class KineticStyleConfig:
    """Configuration parameters for prosody-driven kinetic subtitle animations.

    Attributes:
        min_scale: Baseline font scale percentage for neutral speech.
        max_scale: Maximum font scale percentage for high-energy punchlines.
        energy_threshold_low: Normalized RMS energy floor below which speech is considered neutral.
        energy_threshold_high: Normalized RMS energy ceiling for peak vocal emphasis.
        squash_ratio: Horizontal scale multiplier during initial impact attack phase.
        stretch_ratio: Vertical scale multiplier during rebound stretch phase.
        attack_ms: Duration of initial squash attack in milliseconds.
        decay_ms: Duration of elastic rebound before settling in milliseconds.
        color_neutral: ASS hex color code for standard speech.
        color_excited: ASS hex color code for medium excitement or elevated pitch.
        color_explosive: ASS hex color code for high-intensity vocal shouts.
    """

    min_scale: int = 112
    max_scale: int = 145
    energy_threshold_low: float = 0.15
    energy_threshold_high: float = 0.65
    squash_ratio: float = 1.18
    stretch_ratio: float = 0.88
    attack_ms: int = 70
    decay_ms: int = 90
    color_neutral: str = "&H0000FFFF"
    color_excited: str = "&H0000FF66"
    color_explosive: str = "&H001155FF"


@dataclass(frozen=True)
class ProsodyWordMetric:
    """Acoustic prosody metrics and calculated kinetic visual styling for a spoken word.

    Attributes:
        word: Sanitized text token.
        start: Start timestamp in seconds relative to clip origin.
        end: End timestamp in seconds relative to clip origin.
        rms_energy: Raw root-mean-square amplitude of the audio segment.
        pitch_hz: Estimated fundamental vocal pitch frequency in Hertz.
        emphasis_score: Normalized emphasis score in the range [0.0, 1.0].
        scale_factor: Calculated target font scale percentage.
        active_color: ASS color code for active word highlighting.
        outline_color: ASS color code for stroke outline.
        kinetic_tags: Synthesized ASS transform tags for squash-and-stretch spring animation.
    """

    word: str
    start: float
    end: float
    rms_energy: float
    pitch_hz: float
    emphasis_score: float
    scale_factor: int
    active_color: str
    outline_color: str
    kinetic_tags: str


class ProsodyKineticTypography:
    """Audio DSP analyzer and ASS kinetic animation generator for spoken dialogue."""

    def __init__(self, config: Optional[KineticStyleConfig] = None) -> None:
        """Initialize the prosody typography engine with custom or default styling configuration.

        Args:
            config: Optional KineticStyleConfig instance.
        """
        self.config = config or KineticStyleConfig()

    @staticmethod
    def extract_pcm_from_media(
        media_path: str,
        start_time: float,
        duration: float,
        sample_rate: int = 16000,
    ) -> Tuple[np.ndarray, int]:
        """Extract a mono 16-bit PCM audio slice from local media via FFmpeg pipe.

        Args:
            media_path: Local media file path.
            start_time: Start timestamp in seconds.
            duration: Desired audio duration in seconds.
            sample_rate: Audio sampling frequency in Hertz.

        Returns:
            Tuple of (NumPy 1D array of float32 samples in range [-1.0, 1.0], sample_rate).

        Raises:
            FileNotFoundError: If the source media path does not exist.
            RuntimeError: If FFmpeg fails to extract raw audio stream.
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        duration = max(0.1, duration)
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", f"{max(0.0, start_time):.3f}",
            "-t", f"{duration:.3f}",
            "-i", media_path,
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg PCM extraction failed with exit code {res.returncode}")

        raw_bytes = res.stdout
        if not raw_bytes:
            return np.zeros(0, dtype=np.float32), sample_rate

        samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        samples_float = samples_int16.astype(np.float32) / 32768.0
        return samples_float, sample_rate

    @staticmethod
    def compute_rms_energy(samples: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) volume energy for an audio sample array.

        Args:
            samples: 1D NumPy array of audio amplitude samples.

        Returns:
            Scalar RMS energy float value. Returns 0.0 for empty or silent arrays.
        """
        if samples.size == 0:
            return 0.0
        mean_sq = float(np.mean(samples ** 2))
        return float(math.sqrt(max(0.0, mean_sq)))

    @staticmethod
    def estimate_f0_pitch(
        samples: np.ndarray,
        sample_rate: int = 16000,
        f_min: float = 65.0,
        f_max: float = 480.0,
    ) -> float:
        """Estimate fundamental vocal frequency (F0) using Normalized Cross-Correlation (NCCF).

        Args:
            samples: 1D NumPy array of audio amplitude samples.
            sample_rate: Sampling frequency in Hertz.
            f_min: Minimum human vocal frequency in Hertz.
            f_max: Maximum human vocal frequency in Hertz.

        Returns:
            Estimated pitch in Hertz, or 0.0 if unvoiced or below detection threshold.
        """
        if samples.size < int(sample_rate / f_min):
            return 0.0

        window = samples.astype(np.float32)
        norm = float(np.linalg.norm(window))
        if norm < 1e-4:
            return 0.0

        window = window - float(np.mean(window))
        corr = np.correlate(window, window, mode="full")
        midpoint = len(window) - 1
        lags = corr[midpoint:]

        min_lag = int(sample_rate / f_max)
        max_lag = min(len(lags) - 1, int(sample_rate / f_min))
        if min_lag >= max_lag:
            return 0.0

        search_region = lags[min_lag:max_lag + 1]
        peak_idx = int(np.argmax(search_region)) + min_lag
        peak_val = float(lags[peak_idx])

        if lags[0] <= 1e-6:
            return 0.0

        normalized_peak = peak_val / float(lags[0])
        if normalized_peak < 0.28:
            return 0.0

        return float(sample_rate / peak_idx)

    def analyze_word_prosody(
        self,
        word: str,
        start_time: float,
        end_time: float,
        audio_samples: np.ndarray,
        sample_rate: int = 16000,
        global_rms_baseline: Optional[float] = None,
        global_rms_peak: Optional[float] = None,
    ) -> ProsodyWordMetric:
        """Analyze a temporal word slice and generate animated kinetic typography metrics.

        Args:
            word: Spoken word text.
            start_time: Word start timestamp in seconds.
            end_time: Word end timestamp in seconds.
            audio_samples: Full audio buffer covering the clip or speech section.
            sample_rate: Audio sampling frequency in Hertz.
            global_rms_baseline: Optional global floor RMS energy for normalization.
            global_rms_peak: Optional global peak RMS energy for normalization.

        Returns:
            ProsodyWordMetric instance containing acoustic metrics and visual ASS tags.
        """
        total_len = len(audio_samples)
        pad_sec = 0.04
        start_idx = max(0, int((start_time - pad_sec) * sample_rate))
        end_idx = min(total_len, max(start_idx + 1, int((end_time + pad_sec) * sample_rate)))

        slice_data = audio_samples[start_idx:end_idx] if start_idx < total_len else np.zeros(0, dtype=np.float32)

        rms = self.compute_rms_energy(slice_data)
        pitch = self.estimate_f0_pitch(slice_data, sample_rate=sample_rate)

        e_min = global_rms_baseline if global_rms_baseline is not None else 0.015
        e_max = global_rms_peak if global_rms_peak is not None else 0.18
        e_range = max(1e-4, e_max - e_min)
        emphasis = float(np.clip((rms - e_min) / e_range, 0.0, 1.0))

        if pitch > 240.0:
            emphasis = min(1.0, emphasis + 0.12)

        scale_span = self.config.max_scale - self.config.min_scale
        scale_factor = int(round(self.config.min_scale + scale_span * emphasis))

        if emphasis >= self.config.energy_threshold_high:
            active_color = self.config.color_explosive
            outline_color = "&H00000000"
            kinetic_tags = self._build_squash_and_stretch_tags(scale_factor, high_intensity=True)
        elif emphasis >= self.config.energy_threshold_low:
            active_color = self.config.color_excited
            outline_color = "&H00000000"
            kinetic_tags = self._build_squash_and_stretch_tags(scale_factor, high_intensity=False)
        else:
            active_color = self.config.color_neutral
            outline_color = "&H00000000"
            kinetic_tags = f"\\fscx{scale_factor}\\fscy{scale_factor}"

        return ProsodyWordMetric(
            word=word,
            start=start_time,
            end=end_time,
            rms_energy=rms,
            pitch_hz=pitch,
            emphasis_score=emphasis,
            scale_factor=scale_factor,
            active_color=active_color,
            outline_color=outline_color,
            kinetic_tags=kinetic_tags,
        )

    def _build_squash_and_stretch_tags(self, target_scale: int, high_intensity: bool) -> str:
        """Synthesize ASS transform tags implementing damped harmonic squash and stretch.

        Args:
            target_scale: Target settled font scale percentage.
            high_intensity: Whether to apply aggressive elastic recoil for shouts.

        Returns:
            Formatted ASS inline tag string containing transform keyframes.
        """
        if high_intensity:
            t1 = self.config.attack_ms
            t2 = t1 + self.config.decay_ms
            t3 = t2 + 80
            squash_x = int(round(target_scale * self.config.squash_ratio))
            squash_y = int(round(target_scale * self.config.stretch_ratio))
            stretch_x = int(round(target_scale * 0.90))
            stretch_y = int(round(target_scale * 1.14))
            return (
                f"\\t(0,{t1},\\fscx{squash_x}\\fscy{squash_y})"
                f"\\t({t1},{t2},\\fscx{stretch_x}\\fscy{stretch_y})"
                f"\\t({t2},{t3},\\fscx{target_scale}\\fscy{target_scale})"
            )
        else:
            t1 = self.config.attack_ms
            t2 = t1 + self.config.decay_ms
            pop_x = int(round(target_scale * 1.08))
            pop_y = int(round(target_scale * 0.96))
            return (
                f"\\t(0,{t1},\\fscx{pop_x}\\fscy{pop_y})"
                f"\\t({t1},{t2},\\fscx{target_scale}\\fscy{target_scale})"
            )

    def analyze_clip_words(
        self,
        words: List[Dict[str, Any]],
        media_path: str,
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
    ) -> List[ProsodyWordMetric]:
        """Analyze an entire list of words against media audio and return timed metrics.

        Args:
            words: List of word dictionaries with 'word', 'start', and 'end' keys.
            media_path: Local path to source media containing audio.
            clip_start: Start timestamp in seconds of the clip slice.
            clip_end: End timestamp in seconds of the clip slice.

        Returns:
            List of ProsodyWordMetric instances corresponding to each in-bounds word.
        """
        if not words:
            return []

        clip_duration = (clip_end - clip_start) if clip_end is not None else 60.0
        try:
            samples, sample_rate = self.extract_pcm_from_media(media_path, clip_start, clip_duration)
        except Exception:
            return [
                ProsodyWordMetric(
                    word=str(w.get("word", "")),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    rms_energy=0.05,
                    pitch_hz=120.0,
                    emphasis_score=0.2,
                    scale_factor=self.config.min_scale,
                    active_color=self.config.color_neutral,
                    outline_color="&H00000000",
                    kinetic_tags=f"\\fscx{self.config.min_scale}\\fscy{self.config.min_scale}",
                )
                for w in words
            ]

        energies: List[float] = []
        for w in words:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", 0.0))
            s_idx = max(0, int(w_start * sample_rate))
            e_idx = min(len(samples), max(s_idx + 1, int(w_end * sample_rate)))
            if s_idx < len(samples):
                energies.append(self.compute_rms_energy(samples[s_idx:e_idx]))

        base_rms = float(np.percentile(energies, 15)) if energies else 0.015
        peak_rms = float(np.percentile(energies, 95)) if energies else 0.18

        metrics: List[ProsodyWordMetric] = []
        for w in words:
            w_text = str(w.get("word", ""))
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", 0.0))
            metric = self.analyze_word_prosody(
                word=w_text,
                start_time=w_start,
                end_time=w_end,
                audio_samples=samples,
                sample_rate=sample_rate,
                global_rms_baseline=base_rms,
                global_rms_peak=peak_rms,
            )
            metrics.append(metric)

        return metrics
