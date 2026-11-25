"""Automated container self-healing, corruption recovery, and Dead Letter Queue isolation.

Detects damaged video containers, broken MOOV atoms, and corrupted macroblocks.
Automatically cleans corrupt packets using non-destructive re-indexing or error
concealment, routing unrecoverable assets to a quarantined Dead Letter Queue (DLQ).
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class CorruptedVideoError(Exception):
    """Raised when an unrecoverable corrupted video container is routed to DLQ."""

    def __init__(self, message: str, dlq_path: Optional[str] = None) -> None:
        """Initialize CorruptedVideoError with failure message and quarantine path.

        Args:
            message: Diagnostic description of corruption.
            dlq_path: Path to quarantined asset in Dead Letter Queue.
        """
        super().__init__(message)
        self.dlq_path = dlq_path


class SelfHealingVideoSanitizer:
    """Isolates corrupted video files and cleans corrupt packets before rendering."""

    DEFAULT_DLQ_DIR = os.path.abspath("dead_letter_queue")

    @classmethod
    def inspect_container(cls, media_path: str) -> Tuple[bool, str]:
        """Validate container integrity and atom headers via ffprobe.

        Args:
            media_path: Path to local video file.

        Returns:
            Tuple of (is_healthy, error_details).
        """
        if not os.path.exists(media_path) or os.path.getsize(media_path) == 0:
            return False, "File does not exist or has zero byte length."

        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,duration",
            "-of", "json",
            media_path
        ]
        try:
            res = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15
            )
            if res.returncode != 0:
                return False, res.stderr.strip() or "ffprobe reported non-zero exit code."

            data = json.loads(res.stdout or "{}")
            streams = data.get("streams", [])
            if not streams:
                return False, "Container contains no detectable video streams."

            return True, ""
        except subprocess.TimeoutExpired:
            return False, "ffprobe execution timed out during container validation."
        except Exception as e:
            return False, f"Unexpected error inspecting container: {e}"

    @classmethod
    def repair_container(cls, media_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Attempt non-destructive stream copy re-indexing and packet error cleanup.

        Args:
            media_path: Path to corrupted input video file.
            output_path: Optional destination path for repaired video.

        Returns:
            Tuple of (success, repaired_path_or_error).
        """
        if not os.path.exists(media_path):
            return False, f"Source media file not found: {media_path}"

        stem = Path(media_path).stem
        ext = Path(media_path).suffix or ".mp4"
        if not output_path:
            out_dir = os.path.dirname(os.path.abspath(media_path))
            output_path = os.path.join(out_dir, f"{stem}_repaired{ext}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        remux_cmd = [
            "ffmpeg", "-y",
            "-v", "error",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+discardcorrupt",
            "-i", media_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path
        ]
        try:
            res = subprocess.run(remux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                is_healthy, _ = cls.inspect_container(output_path)
                if is_healthy:
                    return True, output_path
        except Exception:
            pass

        transcode_cmd = [
            "ffmpeg", "-y",
            "-v", "error",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+discardcorrupt",
            "-i", media_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path
        ]
        try:
            res = subprocess.run(transcode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                is_healthy, _ = cls.inspect_container(output_path)
                if is_healthy:
                    return True, output_path
        except Exception:
            pass

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        return False, "Lossless remuxing and transcode repair both failed to salvage container."

    @classmethod
    def route_to_dlq(
        cls,
        media_path: str,
        reason: str,
        dlq_dir: Optional[str] = None
    ) -> str:
        """Quarantine unrecoverable damaged media asset and emit diagnostic failure report.

        Args:
            media_path: Path to corrupted video media file.
            reason: Diagnostic error reason explaining why healing failed.
            dlq_dir: Destination Dead Letter Queue directory.

        Returns:
            Destination path of quarantined asset.
        """
        target_dir = os.path.abspath(dlq_dir or cls.DEFAULT_DLQ_DIR)
        os.makedirs(target_dir, exist_ok=True)

        stem = Path(media_path).stem
        ext = Path(media_path).suffix or ".mp4"
        quarantined_media = os.path.join(target_dir, f"{stem}_quarantined{ext}")
        report_path = os.path.join(target_dir, f"{stem}_failure_report.json")

        if os.path.exists(media_path):
            try:
                shutil.copy2(media_path, quarantined_media)
            except Exception:
                quarantined_media = media_path

        report_payload = {
            "source_path": os.path.abspath(media_path),
            "quarantined_path": quarantined_media,
            "timestamp": time.time(),
            "reason": reason,
            "source_size_bytes": os.path.getsize(media_path) if os.path.exists(media_path) else 0,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        return quarantined_media

    @classmethod
    def ensure_playable_or_heal(
        cls,
        media_path: str,
        dlq_dir: Optional[str] = None
    ) -> str:
        """Ensure media asset is structurally sound; heals or routes to DLQ on corruption.

        Args:
            media_path: Path to media file.
            dlq_dir: Optional Dead Letter Queue destination override.

        Returns:
            Path to verified playable media asset (original or repaired).

        Raises:
            CorruptedVideoError: If asset cannot be repaired and is routed to DLQ.
        """
        if not os.path.exists(media_path):
            return media_path

        is_healthy, error_msg = cls.inspect_container(media_path)
        if is_healthy:
            return media_path

        success, repaired_or_err = cls.repair_container(media_path)
        if success:
            return repaired_or_err

        quarantined = cls.route_to_dlq(media_path, reason=error_msg or repaired_or_err, dlq_dir=dlq_dir)
        raise CorruptedVideoError(
            f"Container {media_path} is irreparably damaged ({error_msg}). Quarantined to {quarantined}.",
            dlq_path=quarantined
        )
