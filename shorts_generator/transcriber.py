"""Transcription via MuAPI /openai-whisper.

Sends a hosted media URL to MuAPI's Whisper endpoint and returns the segment
shape expected by the highlight generator: {duration, segments[start,end,text]}.
The API runs verbose_json server-side, so we get per-segment timestamps for free.
"""
import json
import re
from typing import Dict, List, Optional

from . import muapi

HALLUCINATION_PATTERNS = [
    re.compile(r"thank you for watching", re.IGNORECASE),
    re.compile(r"thanks for watching", re.IGNORECASE),
    re.compile(r"please subscribe", re.IGNORECASE),
    re.compile(r"subtitles by", re.IGNORECASE),
    re.compile(r"translated by", re.IGNORECASE),
    re.compile(r"like and subscribe", re.IGNORECASE),
    re.compile(r"see you in the next video", re.IGNORECASE),
]


def sanitize_transcript_segments(segments: List[Dict]) -> List[Dict]:
    """Filter out known Whisper hallucination loops and outro spam.

    Args:
        segments: Raw list of segment dictionaries from Whisper.

    Returns:
        Cleaned list of segments with hallucinated content removed.
    """
    cleaned: List[Dict] = []
    for s in segments:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        if any(p.search(text) for p in HALLUCINATION_PATTERNS):
            continue
        cleaned.append(s)
    return cleaned


def _coerce_verbose(raw) -> Dict:
    """The /openai-whisper result can land as a dict or a JSON string depending on
    how the worker stored it. Normalise to a dict with `duration` and `segments`."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_verbose_payload(result: Dict) -> Dict:
    """MuAPI wraps results inconsistently across endpoints. Hunt for the
    verbose_json blob (which has `segments` + `duration`)."""
    for key in ("output", "result", "outputs"):
        v = result.get(key)
        if isinstance(v, dict) and "segments" in v:
            return v
        if isinstance(v, list) and v:
            first = v[0]
            decoded = _coerce_verbose(first)
            if "segments" in decoded:
                return decoded
        if isinstance(v, str):
            decoded = _coerce_verbose(v)
            if "segments" in decoded:
                return decoded

    if "segments" in result:
        return result

    raise RuntimeError(f"Could not find Whisper segments in MuAPI response: {result}")


def transcribe(media_url: str, language: Optional[str] = None) -> Dict:
    """Run MuAPI /openai-whisper on a hosted media URL.

    Returns {duration: float, segments: [{start, end, text}, ...]} so it slots
    straight into the highlight generator.
    """
    print(f"[transcribe] muapi /openai-whisper on {media_url}", flush=True)
    payload = {
        "audio_url": media_url,
        "response_format": "verbose_json",
    }
    if language:
        payload["language"] = language

    result = muapi.run("openai-whisper", payload, label="openai-whisper")
    verbose = _extract_verbose_payload(result)

    raw_segments = []
    for s in verbose.get("segments") or []:
        raw_segments.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
        })

    segments = sanitize_transcript_segments(raw_segments)
    duration = float(verbose.get("duration") or (segments[-1]["end"] if segments else 0.0))
    print(f"[transcribe] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": segments}
