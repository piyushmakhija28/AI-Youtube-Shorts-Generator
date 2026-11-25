"""End-to-end orchestrator for video shorts generation.

Two execution modes:
- mode="local": yt-dlp + faster-whisper + OpenAI/Gemini + Intel Arc QSV / FFmpeg hardware pipeline.
- mode="api": MuAPI cloud proxy.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .config import LLM_PROVIDER, require_api_key, require_gemini_key, require_openai_key
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _extract_video_id(url_or_path: str) -> str:
    """Extract YouTube video identifier or fallback to path stem.

    Args:
        url_or_path: YouTube URL string or filesystem path.

    Returns:
        Extracted identifier string.
    """
    m = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url_or_path)
    if m:
        return m.group(1)
    return Path(url_or_path).stem


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    framing: str = "split_stack",
    burn_subtitles: bool = True,
    master_audio: bool = True,
    video_id: Optional[str] = None,
    guard_band: float = 45.0,
    resolution: str = "4k",
    auto_zoom: bool = True,
) -> Dict:
    """Execute full local processing pipeline on the current machine.

    Args:
        youtube_url: YouTube URL or local video path.
        num_clips: Number of top highlights to render.
        aspect_ratio: Target aspect ratio string.
        download_format: Video quality shorthand.
        language: Language code hint for Whisper.
        framing: Framing strategy (split_stack, left_focus, right_focus, smart_focus, center).
        burn_subtitles: Whether to burn dynamic karaoke subtitles.
        master_audio: Whether to apply EBU R128 audio mastering.
        video_id: Optional video identifier for persistent exclusion.
        guard_band: Minimum temporal separation between clips in seconds.
        resolution: Output resolution profile ('4k' for 2160x3840 or '1080p' for 1080x1920).
        auto_zoom: Whether dynamic zoom plans should be applied.

    Returns:
        Structured pipeline result dictionary.
    """
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    vid = video_id or _extract_video_id(youtube_url)
    source_path = download_youtube_local(youtube_url, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(
        transcript,
        num_clips=num_clips,
        llm_fn=call_local_llm,
        video_id=vid,
        guard_band=guard_band
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates [{framing}] ({resolution})", flush=True)

    shorts = crop_highlights_local(
        source_path,
        top,
        aspect_ratio=aspect_ratio,
        framing=framing,
        burn_subtitles=burn_subtitles,
        master_audio=master_audio,
        video_id=vid,
        resolution=resolution,
        transcript_segments=transcript.get("segments"),
        transcript_words=transcript.get("words"),
        auto_framing=(framing in ("auto", "smart", "split_stack")),
        auto_zoom=auto_zoom,
    )

    return {
        "mode": "local",
        "video_id": vid,
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    """Execute processing via external MuAPI cloud services.

    Args:
        youtube_url: Hosted media URL.
        num_clips: Number of top highlights to generate.
        aspect_ratio: Target aspect ratio string.
        download_format: Target format identifier.
        language: Language code hint.

    Returns:
        Structured pipeline output dictionary.
    """
    require_api_key()
    download_url = download_youtube(youtube_url, download_format)
    transcript = transcribe(download_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech, "
            "or the download URL may be inaccessible."
        )

    vid = _extract_video_id(youtube_url)
    highlights_result = get_highlights(
        transcript,
        num_clips=num_clips,
        llm_fn=call_muapi_llm,
        video_id=vid
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/api] cropping top {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(
        download_url,
        top,
        aspect_ratio=aspect_ratio,
        video_id=vid
    )

    return {
        "mode": "api",
        "video_id": vid,
        "source_video_url": download_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 5,
    aspect_ratio: str = "9:16",
    mode: str = "local",
    download_format: str = "best",
    language: Optional[str] = None,
    framing: str = "split_stack",
    burn_subtitles: bool = True,
    master_audio: bool = True,
    guard_band: float = 45.0,
    video_id: Optional[str] = None,
    resolution: str = "4k",
    auto_zoom: bool = True,
    timeout: Optional[int] = None,
) -> Dict:
    """Entry point for shorts generation pipeline with timeout safeguard.

    Args:
        youtube_url: YouTube URL string or local file path.
        num_clips: Number of highlights to return.
        aspect_ratio: Target aspect ratio ('9:16', '1:1', '4:5', '16:9').
        mode: Processing mode ('local' or 'api').
        download_format: Shorthand quality selector.
        language: Language code hint.
        framing: Framing strategy.
        burn_subtitles: Subtitle overlay toggle.
        master_audio: EBU R128 audio mastering toggle.
        guard_band: Temporal distance guard in seconds.
        video_id: Persistent identifier override.
        resolution: Output profile ('4k' or '1080p').
        auto_zoom: Dynamic zoom plan toggle.
        timeout: Execution timeout in seconds.

    Returns:
        Structured output dictionary with shorts metadata.
    """
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language)

    if mode == "local":
        if LLM_PROVIDER in ("arc_runner", "intel_arc", "arc"):
            pass
        elif LLM_PROVIDER == "gemini":
            require_gemini_key()
        else:
            require_openai_key()

        def _execute():
            return _run_local(
                youtube_url=youtube_url,
                num_clips=num_clips,
                aspect_ratio=aspect_ratio,
                download_format=download_format,
                language=language,
                framing=framing,
                burn_subtitles=burn_subtitles,
                master_audio=master_audio,
                video_id=video_id,
                guard_band=guard_band,
                resolution=resolution,
                auto_zoom=auto_zoom,
            )

        if timeout and timeout > 0:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute)
                try:
                    return future.result(timeout=float(timeout))
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Pipeline execution exceeded timeout of {timeout}s.")
        return _execute()

    raise ValueError(f"Unknown mode: {mode!r}. Must be 'local' or 'api'.")
