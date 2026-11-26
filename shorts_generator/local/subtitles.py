"""Dynamic animated subtitle engine for short-form vertical video.

Generates Advanced SubStation Alpha (.ass) subtitles with karaoke-style
active word pop and high-contrast styling for YouTube Shorts and TikTok.
"""

import io
import os
import subprocess
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel


def format_ass_timestamp(seconds: float) -> str:
    """Convert a timestamp in seconds to standard ASS format (H:MM:SS.cs).

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted timestamp string.
    """
    total_centis = int(round(max(0.0, seconds) * 100))
    hours = total_centis // 360000
    remainder = total_centis % 360000
    minutes = remainder // 6000
    remainder %= 6000
    secs = remainder // 100
    centis = remainder % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def sanitize_dialogue(text: str) -> str:
    """Sanitize subtitle text by escaping ASS control symbols and uppercase normalization.

    Args:
        text: Raw transcribed string.

    Returns:
        Safe uppercase subtitle string.
    """
    clean = text.replace("{", "(").replace("}", ")").replace("\\", "/")
    clean = clean.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    return clean.strip().upper()


def extract_word_timings_slice(
    media_path: str,
    start_time: float,
    end_time: float,
    model_size: str = "base.en",
) -> List[Dict[str, Any]]:
    """Extract word timestamps from an audio slice using faster-whisper.

    Args:
        media_path: Local media path.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        model_size: Faster-whisper model identifier.

    Returns:
        List of dictionaries with sanitized word, start timestamp, and end timestamp relative to start_time.
    """
    duration = max(0.5, end_time - start_time)
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", "-"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio_stream = io.BytesIO(res.stdout)

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_stream, word_timestamps=True, beam_size=5)

    words: List[Dict[str, Any]] = []
    for segment in segments:
        if not segment.words:
            continue
        for item in segment.words:
            w_text = sanitize_dialogue(item.word)
            if not w_text:
                continue
            words.append({
                "word": w_text,
                "start": max(0.0, item.start),
                "end": max(item.start + 0.05, item.end)
            })
    return words


def extract_word_timings(media_path: str, model_size: str = "base.en") -> List[Dict[str, Any]]:
    """Transcribe media to retrieve word-level boundaries and probabilities.

    Args:
        media_path: Local media path.
        model_size: Faster-whisper model identifier.

    Returns:
        List of dictionaries with sanitized word, start timestamp, and end timestamp.
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(media_path, word_timestamps=True, beam_size=5)

    words: List[Dict[str, Any]] = []
    for segment in segments:
        if not segment.words:
            continue
        for item in segment.words:
            w_text = sanitize_dialogue(item.word)
            if not w_text:
                continue
            words.append({
                "word": w_text,
                "start": max(0.0, item.start),
                "end": max(item.start + 0.05, item.end)
            })
    return words


def group_into_chunks(
    words: List[Dict[str, Any]],
    max_words: int = 3,
    max_gap_sec: float = 0.45,
) -> List[List[Dict[str, Any]]]:
    """Group words into short conversational chunks for fast-paced mobile viewing.

    Args:
        words: List of word timing dictionaries.
        max_words: Maximum words per displayed subtitle line.
        max_gap_sec: Maximum pause duration before splitting into a new chunk.

    Returns:
        List of word chunks.
    """
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for word in words:
        if not current:
            current.append(word)
            continue

        prev = current[-1]
        gap = word["start"] - prev["end"]
        punct = any(prev["word"].endswith(p) for p in [".", "!", "?", ","])

        if len(current) >= max_words or gap > max_gap_sec or punct:
            chunks.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(current)

    return chunks


def build_ass_script(
    chunks: List[List[Dict[str, Any]]],
    margin_v: int = 520,
    font_name: str = "Impact",
    font_size: int = 78,
    active_color: str = "&H0000FFFF",
    inactive_color: str = "&H00FFFFFF",
    outline_color: str = "&H00000000",
    outline_width: int = 7,
    target_w: int = 1080,
    target_h: int = 1920,
    metrics: Optional[List[Any]] = None,
) -> str:
    """Build complete ASS document with active-word karaoke styling and prosody kinetic animation.

    Args:
        chunks: List of word chunks.
        margin_v: Vertical margin from bottom edge in reference coordinates.
        font_name: Font face name.
        font_size: Point size of the subtitle text in 1080x1920 reference.
        active_color: ASS color code for currently spoken word when metric not specified.
        inactive_color: ASS color code for surrounding chunk words.
        outline_color: ASS color code for text stroke outline.
        outline_width: Pixel width of the stroke outline.
        target_w: Target canvas width in pixels.
        target_h: Target canvas height in pixels.
        metrics: Optional list of ProsodyWordMetric instances matching words.

    Returns:
        Full UTF-8 ASS file content string.
    """
    scale = target_h / 1920.0
    effective_font_size = int(round(font_size * scale))
    effective_outline = max(1, int(round(outline_width * scale)))
    effective_margin_v = int(round(margin_v * scale))
    effective_margin_lr = int(round(60 * scale))

    metric_map: Dict[Tuple[float, str], Any] = {}
    if metrics:
        for m in metrics:
            metric_map[(round(getattr(m, "start", 0.0), 2), getattr(m, "word", ""))] = m

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {target_w}\n"
        f"PlayResY: {target_h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{effective_font_size},{inactive_color},{active_color},{outline_color},&H90000000,-1,0,0,0,100,100,2,0,1,{effective_outline},3,2,{effective_margin_lr},{effective_margin_lr},{effective_margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue

        chunk_end = chunk[-1]["end"] + 0.20
        for i, word_item in enumerate(chunk):
            start_t = word_item["start"]
            if i < len(chunk) - 1:
                end_t = max(start_t + 0.08, chunk[i + 1]["start"])
            else:
                end_t = max(start_t + 0.15, chunk_end)

            start_str = format_ass_timestamp(start_t)
            end_str = format_ass_timestamp(end_t)

            tokens: List[str] = []
            for j, token in enumerate(chunk):
                w = token["word"]
                if j == i:
                    m = metric_map.get((round(token["start"], 2), w))
                    if m is not None:
                        act_c = getattr(m, "active_color", active_color)
                        k_tags = getattr(m, "kinetic_tags", "\\fscx115\\fscy115")
                        tokens.append(f"{{\\c{act_c}&{k_tags}}}{w}{{\\c{inactive_color}&\\fscx100\\fscy100}}")
                    else:
                        tokens.append(f"{{\\c{active_color}&\\fscx115\\fscy115}}{w}{{\\c{inactive_color}&\\fscx100\\fscy100}}")
                else:
                    tokens.append(w)

            line_text = " ".join(tokens)
            events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{line_text}")

    return header + "\n".join(events) + "\n"


def generate_ass_file(
    input_media: str,
    output_ass: str,
    margin_v: int = 520,
    target_w: int = 1080,
    target_h: int = 1920,
    words: Optional[List[Dict[str, Any]]] = None,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    audio_path: Optional[str] = None,
    enable_prosody: bool = True,
    prosody_config: Optional[Any] = None,
) -> str:
    """Generate and write an ASS subtitle file from input media or cached words.

    Args:
        input_media: Path to video or audio source.
        output_ass: Path to write destination .ass file.
        margin_v: Vertical margin from bottom edge.
        target_w: Canvas target width in pixels.
        target_h: Canvas target height in pixels.
        words: Optional pre-extracted word timing dictionaries.
        clip_start: Start timestamp of the clip in seconds for relative offset.
        clip_end: End timestamp of the clip in seconds.
        audio_path: Optional path to dedicated audio media file for prosody extraction.
        enable_prosody: Whether to enable acoustic prosody-driven kinetic styling.
        prosody_config: Optional KineticStyleConfig instance.

    Returns:
        Absolute path to the created ASS file.
    """
    actual_end = clip_end if clip_end is not None else (clip_start + 60.0)
    if words is not None:
        clip_words: List[Dict[str, Any]] = []
        for w in words:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", 0.0))
            if clip_end is not None and w_start > clip_end:
                continue
            if w_end < clip_start:
                continue
            rel_start = max(0.0, w_start - clip_start)
            rel_end = max(rel_start + 0.05, w_end - clip_start)
            w_text = sanitize_dialogue(str(w.get("word", "")))
            if w_text:
                clip_words.append({
                    "word": w_text,
                    "start": rel_start,
                    "end": rel_end,
                })
        selected_words = clip_words
    else:
        selected_words = extract_word_timings_slice(input_media, clip_start, actual_end)

    metrics = None
    if enable_prosody and selected_words:
        try:
            from ..video.prosody_typography import ProsodyKineticTypography
            source_audio = audio_path if (audio_path and os.path.exists(audio_path)) else input_media
            if os.path.exists(source_audio):
                analyzer = ProsodyKineticTypography(config=prosody_config)
                metrics = analyzer.analyze_clip_words(
                    words=selected_words,
                    media_path=source_audio,
                    clip_start=clip_start,
                    clip_end=actual_end,
                )
        except Exception:
            metrics = None

    chunks = group_into_chunks(selected_words)
    content = build_ass_script(
        chunks,
        margin_v=margin_v,
        target_w=target_w,
        target_h=target_h,
        metrics=metrics,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_ass)), exist_ok=True)
    with open(output_ass, "w", encoding="utf-8") as f:
        f.write(content)

    return os.path.abspath(output_ass)
