"""Automated broadcast color matrix detection and correction engine.

Probes video color space metadata via ffprobe and generates lossless
color matrix translation filters (e.g. BT.601 to BT.709) to prevent
washed-out blacks, gamut clipping, and hue shifts during transcoding.
"""

import json
import os
import subprocess
from typing import Dict, Optional


class ColorMatrixFixer:
    """Probes and corrects video stream color matrix mismatches."""

    BT601_SPACES = {
        "smpte170m",
        "bt470bg",
        "bt470_6m",
        "bt601",
        "bt601-6-525",
        "bt601-6-625",
    }

    @staticmethod
    def probe_color_metadata(media_path: str) -> Dict[str, str]:
        """Probe video stream colorimetry attributes using ffprobe.

        Args:
            media_path: Path to local video media asset.

        Returns:
            Dictionary containing 'color_space', 'color_transfer', 'color_primaries',
            and 'color_range'. Returns empty strings for unpopulated tags.
        """
        if not os.path.exists(media_path):
            return {"color_space": "", "color_transfer": "", "color_primaries": "", "color_range": ""}

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=color_space,color_transfer,color_primaries,color_range",
            "-of", "json",
            media_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(res.stdout or "{}")
            streams = data.get("streams", [])
            if not streams:
                return {"color_space": "", "color_transfer": "", "color_primaries": "", "color_range": ""}
            st = streams[0]
            return {
                "color_space": str(st.get("color_space") or "").lower().strip(),
                "color_transfer": str(st.get("color_transfer") or "").lower().strip(),
                "color_primaries": str(st.get("color_primaries") or "").lower().strip(),
                "color_range": str(st.get("color_range") or "").lower().strip(),
            }
        except Exception:
            return {"color_space": "", "color_transfer": "", "color_primaries": "", "color_range": ""}

    @classmethod
    def is_bt601(cls, metadata: Dict[str, str]) -> bool:
        """Evaluate if video stream metadata represents a BT.601 matrix.

        Args:
            metadata: Probed video color metadata dictionary.

        Returns:
            True if any colorimetry attribute signals BT.601 / NTSC / PAL color spaces.
        """
        space = metadata.get("color_space", "")
        transfer = metadata.get("color_transfer", "")
        primaries = metadata.get("color_primaries", "")

        return (
            space in cls.BT601_SPACES
            or transfer in cls.BT601_SPACES
            or primaries in cls.BT601_SPACES
        )

    @classmethod
    def build_color_matrix_filter(cls, metadata: Dict[str, str]) -> Optional[str]:
        """Construct FFmpeg scale filter expression to convert BT.601 to BT.709.

        Args:
            metadata: Probed video color metadata dictionary.

        Returns:
            FFmpeg scale filter string if conversion is required, otherwise None.
        """
        if not cls.is_bt601(metadata):
            return None

        in_range = "limited"
        if metadata.get("color_range") in ("pc", "full"):
            in_range = "full"

        return f"scale=in_color_matrix=bt601:out_color_matrix=bt709:in_range={in_range}:out_range=limited"

    @classmethod
    def get_output_tagging_flags(cls) -> Dict[str, str]:
        """Return standardized broadcast BT.709 container tagging parameters.

        Returns:
            Dictionary of FFmpeg output CLI flag pairs.
        """
        return {
            "-colorspace": "bt709",
            "-color_primaries": "bt709",
            "-color_trc": "bt709",
            "-color_range": "tv",
        }
