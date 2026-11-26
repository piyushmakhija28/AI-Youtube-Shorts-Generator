"""Broadcast-compliance multitrack audio mastering package.

Provides broadcast-grade loudness normalization (EBU R128, ATSC A/85, Online Streaming),
True Peak inter-sample brickwall limiting, dynamic sidechain auto-ducking, and
4-stem multitrack gain staging.
"""

from .broadcast_audio import (
    AudioMasteringError,
    AudioStem,
    BroadcastAudioMaster,
    CodecEncodingError,
    LoudnessComplianceError,
    LoudnessMetrics,
    LoudnessStandard,
    MasteringConfig,
    MasteringResult,
    StemType,
    StemValidationError,
)

__all__ = [
    "AudioMasteringError",
    "AudioStem",
    "BroadcastAudioMaster",
    "CodecEncodingError",
    "LoudnessComplianceError",
    "LoudnessMetrics",
    "LoudnessStandard",
    "MasteringConfig",
    "MasteringResult",
    "StemType",
    "StemValidationError",
]
