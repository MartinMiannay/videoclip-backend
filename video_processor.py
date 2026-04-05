"""
video_processor.py — Core pipeline for short-form clip extraction.

Flow:
  1. Transcribe audio with local Whisper (word-level timestamps)
  2. Ask Claude to select the best clips (≤16)
  3. For each clip: detect speakers, render crop filter, burn subtitles,
     overlay title card + music, append CTA outro
  4. Upload to storage and update MongoDB project state
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MEDIAPIPE_AVAILABLE = True
except Exception as _mp_err:
    logging.getLogger(__name__).warning(
        "MediaPipe unavailable (%s) — falling back to OpenCV Haar cascade face detection",
        _mp_err,
    )
    _MEDIAPIPE_AVAILABLE = False
import whisperx
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PARALLEL_CLIPS = 3

FRAME_W = 1080
FRAME_H = 1920
SAFE_MARGIN = 40          # px from each edge
MAX_TITLE_W = FRAME_W - SAFE_MARGIN * 2   # 1000 px

# Subtitle style
SUB_FONT = "Arial-Bold"
SUB_FONTSIZE = 52
SUB_BORDER_W = 2          # thin black outline
SUB_BOX_BORDER = 8        # padding around subtitle background box (px)
SUB_Y_RATIO = 0.72        # vertical position as fraction of frame height
SUB_SHIFT_MS = 0.100      # shift card 100 ms earlier than word start
SUB_SILENCE_GAP = 0.3     # gap in seconds that means silence (no card)
SUB_MIN_CARD_GAP = 0.05   # minimum gap enforced between consecutive subtitle cards

_FILLER_WORDS = {"hm", "hum", "euh", "eh", "mmm", "ah"}  # isolated filler sounds to suppress

# Set False to skip all drawtext rendering (subtitles + title card).
# Disable while debugging FFmpeg filter parse errors; re-enable once rendering works.
SUBTITLES_ENABLED = True

# Face detection thresholds
FACE_CONF_THRESHOLD = 0.7

ROOT_DIR = Path(__file__).parent
MUSIC_DIR = ROOT_DIR / "music"
ASSETS_DIR = ROOT_DIR / "assets"

_FACE_MODEL_PATH = ROOT_DIR / "blaze_face_short_range.tflite"
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

CTA_PATH = ASSETS_DIR / "cta_outro.mov"
CTA_PREENCODED_PATH = ASSETS_DIR / "cta_preencoded.mp4"

import sys as _sys
if _sys.platform == "win32":
    # Drive-letter paths (C:/...) break FFmpeg's filter parser because `:` is a
    # delimiter.  Copy the font to assets/ at import time and reference it by a
    # path relative to ROOT_DIR — no colon, no issue.
    _win_font_src = Path("C:/Windows/Fonts/arialbd.ttf")
    _win_font_dst = ASSETS_DIR / "arialbd.ttf"
    if not _win_font_dst.exists() and _win_font_src.exists():
        import shutil as _shutil
        ASSETS_DIR.mkdir(exist_ok=True)
        _shutil.copy2(_win_font_src, _win_font_dst)
    FONT_PATH = "assets/arialbd.ttf"   # relative to ROOT_DIR, passed as ffmpeg cwd
else:
    FONT_PATH = "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"

CLIP_MIN_DURATION = 30   # seconds
CLIP_MAX_DURATION = 90   # seconds

# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------

SEGMENT_PROMPT = """\
You are an expert content analyst for French business and entrepreneurship interviews. \
Segment the transcript into 15–20 distinct themes.

Rules:
- You MUST return at least 15 themes. If you find fewer you are being too coarse — \
  split sections further. A 60-minute interview has at least 18 distinct moments.
- Each theme is one topic, story, opinion, or standalone moment (question answered, \
  anecdote told, claim made, number revealed).
- Themes can be as short as 30 seconds. Do NOT merge adjacent distinct topics.
- Themes must be non-overlapping and together cover the full interview.
- Provide a short French label (≤ 8 words) and a one-sentence description.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no commentary:
{
  "themes": [
    {
      "start": <float seconds>,
      "end": <float seconds>,
      "theme": "<short French label>",
      "description": "<one sentence>"
    }
  ]
}
"""

BOUNDARY_PROMPT = """\
You are a precision video editor for French short-form business content.

You will receive transcript excerpts for ALL themes from an interview. \
Select the best 12–14 clips total — one per theme, choosing the themes with \
the strongest viral potential. Every selected theme MUST have a clip; do not \
skip a theme you have chosen.

START RULES — the clip must open on:
- A surprising stat, bold claim, provocative question, or mid-tension moment
- The exact word where the tension or insight begins
- NEVER a greeting, filler phrase ("donc", "voilà", "eh bien"), or scene-setting \
  sentence

END RULES — cut immediately after:
- The key insight or punchline lands
- Before the speaker pivots to explanation, context, or a new point

DURATION: Aim for 30–90 seconds. If the best moment for a theme is slightly \
outside that range, include it anyway — always return the best clip for each \
selected theme even if imperfect.

VIRAL HOOK TITLE EXAMPLES:
- "Embaucher ma femme c'est une bonne idée ?"
- "Un téléphone pro ? Seulement quand on gagne 1 000 000 €"
- "Pourquoi faire du black vous appauvrit ?"
- "Il fait 500 000€/an et voyage pendant 5 mois !"
- "Je ne peux pas survivre avec un loyer de 15 000 €"

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no commentary:
{
  "clips": [
    {
      "theme": "<exact theme label as provided>",
      "start": <float seconds>,
      "end": <float seconds>,
      "title": "<French hook title, max 60 chars>",
      "hook": "<one sentence explaining why this clip stops a scroll>",
      "virality_score": <integer 1–10, how likely this clip goes viral on TikTok/Reels>
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float   # seconds
    end: float     # seconds
    precise: bool = True


@dataclass
class SubtitleCard:
    """2–3 words shown as a single subtitle card."""
    words: list[Word]
    display_start: float   # shifted earlier by SUB_SHIFT_MS
    display_end: float     # exact end of last word

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class ClipSpec:
    clip_id: str
    project_id: str
    start: float
    end: float
    title: str
    hook: str
    words: list[Word]          # words that fall within this clip's time range
    subtitle_cards: list[SubtitleCard] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Transcription — local Whisper
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, language: str = "fr") -> list[Word]:
    """
    Transcribe audio with WhisperX (large-v2 model) and return
    word-level timestamps via forced alignment.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Loading WhisperX large-v2 model on %s…", device)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    logger.info("Transcribing %s …", audio_path)
    result = model.transcribe(audio_path, language=language)

    logger.info("Aligning word timestamps…")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio_path, device,
        return_char_alignments=False,
    )

    words: list[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            raw = w.get("word", "").strip()
            if not raw:
                continue
            # whisperx may omit timestamps on rare unaligned words — skip those
            if "start" not in w or "end" not in w:
                continue
            words.append(Word(
                text=raw,
                start=float(w["start"]),
                end=float(w["end"]),
                precise=True,
            ))

    logger.info("Transcription complete — %d words", len(words))
    return words


def words_to_transcript_text(words: list[Word]) -> str:
    """Flat transcript with timestamps for Claude, one sentence per line."""
    if not words:
        return ""

    lines: list[str] = []
    sentence: list[str] = []
    sentence_start = words[0].start

    for i, w in enumerate(words):
        sentence.append(w.text)
        if w.text.endswith((".", "?", "!", "…", "...")):
            ts = f"[{sentence_start:.1f}s–{w.end:.1f}s]"
            lines.append(f"{ts} {''.join(sentence)}")
            sentence = []
            if i + 1 < len(words):
                sentence_start = words[i + 1].start

    if sentence:
        ts = f"[{sentence_start:.1f}s–{words[-1].end:.1f}s]"
        lines.append(f"{ts} {''.join(sentence)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Subtitle card grouping (computed once, reused everywhere)
# ---------------------------------------------------------------------------

def build_subtitle_cards(words: list[Word], clip_start: float) -> list[SubtitleCard]:
    """
    Group words into 2–3 word cards with precise display timing.

    Rules:
    - Filler words (_FILLER_WORDS) are stripped before grouping
    - 2–3 words per card
    - Card appears SUB_SHIFT_MS before the first word starts
    - Card disappears exactly when the last word ends
    - Never bridge a silence gap > SUB_SILENCE_GAP seconds
    - A minimum gap of SUB_MIN_CARD_GAP is enforced between consecutive cards
    - Timestamps are relative to clip_start (i.e. 0 = start of clip)
    """
    if not words:
        return []

    # Strip isolated filler sounds so they never appear as subtitles
    words = [w for w in words if w.text.strip().lower() not in _FILLER_WORDS]
    if not words:
        return []

    cards: list[SubtitleCard] = []
    group: list[Word] = []

    def flush(grp: list[Word]) -> None:
        if not grp:
            return
        raw_start = grp[0].start - clip_start
        raw_end = grp[-1].end - clip_start
        cards.append(SubtitleCard(
            words=grp,
            display_start=max(0.0, raw_start - SUB_SHIFT_MS),
            display_end=raw_end,
        ))

    for word in words:
        if group:
            gap = word.start - group[-1].end
            if gap > SUB_SILENCE_GAP:
                flush(group)
                group = [word]
                continue

        group.append(word)

        if len(group) >= 3:
            flush(group)
            group = []

    flush(group)

    # Enforce minimum gap between consecutive cards to prevent overlap
    for i in range(1, len(cards)):
        min_start = cards[i - 1].display_end + SUB_MIN_CARD_GAP
        if cards[i].display_start < min_start:
            cards[i].display_start = min_start

    return cards


# ---------------------------------------------------------------------------
# 3. Claude clip selection pipeline (5 steps)
# ---------------------------------------------------------------------------

_CLAUDE_RATE_LIMIT_SLEEP = 60   # seconds to wait after each Claude call

def _claude_json(system: str, user: str, label: str) -> Any:
    """Single Claude API call → parsed JSON. Sleeps after the call to avoid rate limits."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Claude [%s] sending request — user msg %d chars", label, len(user))
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = message.content[0].text.strip()
    logger.info(
        "Claude [%s] raw response (%d chars, stop_reason=%s): %.500s",
        label, len(raw), message.stop_reason, raw,
    )
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        result = json.loads(raw)
        logger.info("Claude [%s] JSON parse OK", label)
    except json.JSONDecodeError as exc:
        logger.error("Claude [%s] JSON parse FAILED: %s", label, exc)
        raise ValueError(f"Claude [{label}] invalid JSON: {exc}") from exc
    logger.info("Claude [%s] done — sleeping %ds to respect rate limit…", label, _CLAUDE_RATE_LIMIT_SLEEP)
    time.sleep(_CLAUDE_RATE_LIMIT_SLEEP)
    return result


def segment_themes_with_claude(words: list[Word]) -> list[dict[str, Any]]:
    """
    Step 1 — Thematic segmentation.
    Returns [{start, end, theme, description}, ...] (15–20 entries).
    """
    transcript = words_to_transcript_text(words)
    logger.info(
        "Step 1 — thematic segmentation — transcript: %d words, %d chars",
        len(words), len(transcript),
    )
    data = _claude_json(SEGMENT_PROMPT, transcript, "segmentation")
    themes = data.get("themes", [])
    logger.info(
        "  Found %d themes:%s",
        len(themes),
        "".join(
            f"\n    [{i+1:2d}] {t.get('theme', '?')!r:50s} "
            f"{t.get('start', 0):.1f}s–{t.get('end', 0):.1f}s "
            f"({t.get('end', 0) - t.get('start', 0):.0f}s)"
            for i, t in enumerate(themes)
        ),
    )
    if len(themes) < 15:
        logger.warning(
            "  !! Only %d themes returned (expected ≥ 15). Consider re-running.", len(themes)
        )
    return themes


_TOP_CLIPS = 14            # keep the N most viral clips


def find_clip_boundaries_with_claude(
    themes: list[dict[str, Any]],
    words: list[Word],
) -> list[dict[str, Any]]:
    """
    Step 2 — Send all themes to Claude in one call and ask for the best
    12–14 clips ranked by virality. Returns up to _TOP_CLIPS clips after
    duration validation and hook-snapping to the first spoken word.
    """
    logger.info("Step 2 — finding clip boundaries for %d themes (single call)…", len(themes))

    _EXCERPT_MAX_CHARS = 2000
    parts: list[str] = []
    for t in themes:
        theme_words = [w for w in words if t["start"] <= w.start <= t["end"]]
        excerpt = words_to_transcript_text(theme_words)
        if len(excerpt) > _EXCERPT_MAX_CHARS:
            excerpt = excerpt[:_EXCERPT_MAX_CHARS] + "…"
        parts.append(
            f'Theme: "{t["theme"]}" [{t["start"]:.1f}s–{t["end"]:.1f}s]\n{excerpt}'
        )
    user_msg = "\n\n---\n\n".join(parts)

    data = _claude_json(BOUNDARY_PROMPT, user_msg, "boundaries")
    all_raw_clips = data.get("clips", [])
    logger.info("  Claude returned %d raw clips for %d themes", len(all_raw_clips), len(themes))

    # Enforce one clip per theme at the code level
    seen_themes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in all_raw_clips:
        label = c.get("theme", "")
        if label in seen_themes:
            logger.info(
                "  Clip '%s' dropped — theme '%s' already has a clip",
                c.get("title", "?"), label,
            )
            continue
        seen_themes.add(label)
        deduped.append(c)

    valid: list[dict[str, Any]] = []
    for c in deduped:
        start = float(c.get("start", 0))
        end = float(c.get("end", 0))

        # Snap start to first spoken word to avoid opening on silence
        first_word = next((w for w in words if w.start >= start and w.start <= end), None)
        if first_word and first_word.start > start:
            logger.info(
                "  Clip '%s': snapping start %.3fs → %.3fs (%.2fs silence removed)",
                c.get("title", "?"), start, first_word.start, first_word.start - start,
            )
            start = first_word.start
            c = {**c, "start": start}

        duration = end - start
        if duration < CLIP_MIN_DURATION or duration > CLIP_MAX_DURATION:
            logger.warning(
                "  Clip '%s' skipped — duration %.1fs out of range [%d–%d]",
                c.get("title", "?"), duration, CLIP_MIN_DURATION, CLIP_MAX_DURATION,
            )
            continue
        valid.append(c)

    # Rank by virality_score and keep the top _TOP_CLIPS
    valid.sort(key=lambda c: float(c.get("virality_score", 0)), reverse=True)
    if len(valid) > _TOP_CLIPS:
        logger.info("  Keeping top %d clips by virality_score (dropping %d)",
                    _TOP_CLIPS, len(valid) - _TOP_CLIPS)
        valid = valid[:_TOP_CLIPS]

    logger.info(
        "  Generated %d clips from %d themes: %s",
        len(valid), len(themes),
        [f'"{c.get("title", "?")}" score={c.get("virality_score", "?")} ({float(c["end"]) - float(c["start"]):.0f}s)' for c in valid],
    )
    return valid


# Silence trimming ────────────────────────────────────────────────────────────

_SILENCE_TRIM_THRESHOLD = 1.5   # seconds

def trim_clip_silences(
    clips: list[dict[str, Any]],
    words: list[Word],
) -> list[dict[str, Any]]:
    """
    Step 4 — Internal silence trimming (no API call).
    For each clip:
    - Advance start forward to the first word if a leading silence > threshold exists.
    - Pull end back to the last word's end if a trailing silence > threshold exists.
    Clips that become too short after trimming are dropped.
    """
    logger.info("Silence trimming — %d clips…", len(clips))
    trimmed: list[dict[str, Any]] = []

    for c in clips:
        title = c.get("title", "?")
        logger.info("  Trimming silences for clip '%s'…", title)
        start = float(c["start"])
        end = float(c["end"])
        clip_words = [w for w in words if start <= w.start <= end]

        if not clip_words:
            logger.warning("  Clip '%s' has no words — dropping", title)
            continue

        new_start = start
        new_end = end

        # Trim leading silence
        if clip_words[0].start - start > _SILENCE_TRIM_THRESHOLD:
            new_start = clip_words[0].start
            logger.debug(
                "  Clip '%s': trimmed leading silence %.1fs → start %.1fs",
                c.get("title", "?"), clip_words[0].start - start, new_start,
            )

        # Trim trailing silence
        if end - clip_words[-1].end > _SILENCE_TRIM_THRESHOLD:
            new_end = clip_words[-1].end + 0.3   # small breath after last word
            logger.debug(
                "  Clip '%s': trimmed trailing silence %.1fs → end %.1fs",
                c.get("title", "?"), end - clip_words[-1].end, new_end,
            )

        duration = new_end - new_start
        if duration < CLIP_MIN_DURATION:
            logger.warning(
                "  Clip '%s' dropped after silence trim — duration %.1fs < %ds",
                c.get("title", "?"), duration, CLIP_MIN_DURATION,
            )
            continue

        trimmed.append({**c, "start": new_start, "end": new_end})

    logger.info("  → %d clips after silence trimming", len(trimmed))
    return trimmed


# ---------------------------------------------------------------------------
# 4. Face / speaker detection (MediaPipe)
# ---------------------------------------------------------------------------

def _ensure_face_model() -> None:
    if not _FACE_MODEL_PATH.exists():
        import urllib.request
        logger.info("Downloading mediapipe face detection model to %s", _FACE_MODEL_PATH)
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)


def _collect_faces_mediapipe(video_path: str, sample_every_n_frames: int) -> list[dict]:
    """Collect all face detections across sampled frames. Returns [{cx, area}, ...]."""
    import cv2

    _ensure_face_model()
    detector = _mp_vision.FaceDetector.create_from_options(
        _mp_vision.FaceDetectorOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=str(_FACE_MODEL_PATH)),
            min_detection_confidence=FACE_CONF_THRESHOLD,
        )
    )

    cap = cv2.VideoCapture(video_path)
    faces: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)
                if result.detections:
                    for det in result.detections:
                        bbox = det.bounding_box
                        faces.append({
                            "cx": bbox.origin_x + bbox.width // 2,
                            "area": bbox.width * bbox.height,
                        })
            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    return faces


def _collect_faces_opencv(video_path: str, sample_every_n_frames: int) -> list[dict]:
    """Collect all face detections across sampled frames. Returns [{cx, area}, ...]."""
    import cv2

    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)
    faces: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                detected = frontal_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
                )
                for (x, y, w, h) in (detected if len(detected) else []):
                    faces.append({"cx": int(x + w // 2), "area": int(w * h)})
            frame_idx += 1
    finally:
        cap.release()

    return faces


def detect_speakers(video_path: str, sample_every_n_frames: int = 15) -> list[int]:
    """
    Return X center positions of the up to two largest faces found in the video,
    sorted left to right. Returns [] (no faces), [x] (one speaker), or
    [x_left, x_right] (two speakers).
    """
    if _MEDIAPIPE_AVAILABLE:
        try:
            faces = _collect_faces_mediapipe(video_path, sample_every_n_frames)
        except Exception as e:
            logger.warning("MediaPipe face detection failed (%s), falling back to OpenCV", e)
            faces = _collect_faces_opencv(video_path, sample_every_n_frames)
    else:
        faces = _collect_faces_opencv(video_path, sample_every_n_frames)

    if not faces:
        logger.info("No faces detected in %s", video_path)
        return []

    # Sort by area descending, pick up to 2 spatially distinct faces
    MIN_X_SEP = 100  # px — faces closer than this are the same person
    speakers: list[int] = []
    for face in sorted(faces, key=lambda f: f["area"], reverse=True):
        if not any(abs(face["cx"] - sx) < MIN_X_SEP for sx in speakers):
            speakers.append(face["cx"])
        if len(speakers) == 2:
            break

    result = sorted(speakers)
    logger.info("detect_speakers: %d speaker(s) at X=%s in %s", len(result), result, video_path)
    return result


# ---------------------------------------------------------------------------
# 5. Dynamic crop filter (speaker switching)
# ---------------------------------------------------------------------------

def build_dynamic_crop_filter(
    speaker_xs: list[int],
    source_w: int,
    source_h: int,
    clip_start: float,
    clip_end: float,
    words: list[Word],
) -> str:
    """
    Build an FFmpeg crop+scale filter string.
    With two speakers, switches between their X positions at speech pauses.
    With one speaker, stays on that face. With none, center crops.
    """
    crop_w = source_h * 9 // 16

    def clamp_x(cx: int) -> int:
        return max(0, min(source_w - crop_w, cx - crop_w // 2))

    if not speaker_xs:
        x = (source_w - crop_w) // 2
        return f"crop={crop_w}:{source_h}:{x}:0,scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    if len(speaker_xs) == 1:
        x = clamp_x(speaker_xs[0])
        return f"crop={crop_w}:{source_h}:{x}:0,scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    # Two speakers — switch on speech pauses
    x0, x1 = clamp_x(speaker_xs[0]), clamp_x(speaker_xs[1])
    clip_words = [w for w in words if clip_start <= w.start <= clip_end]
    pause_times: list[float] = []
    for i in range(len(clip_words) - 1):
        if clip_words[i + 1].start - clip_words[i].end >= SUB_SILENCE_GAP:
            pause_times.append(clip_words[i].end - clip_start)

    if not pause_times:
        return f"crop={crop_w}:{source_h}:{x0}:0,scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    # Build a nested FFmpeg if(lt(t,T),X,…) expression that toggles x0/x1 at each pause.
    # Commas inside if() must be escaped as \, for FFmpeg's filter parser.
    xs = [x0, x1]
    expr = str(xs[len(pause_times) % 2])
    for i, t in reversed(list(enumerate(pause_times))):
        expr = f"if(lt(t\\,{t:.3f})\\,{xs[i % 2]}\\,{expr})"

    logger.info(
        "switch_crop: %d pause(s) in clip [%.2f–%.2f], x0=%d x1=%d",
        len(pause_times), clip_start, clip_end, x0, x1,
    )
    return f"crop={crop_w}:{source_h}:{expr}:0,scale={FRAME_W}:{FRAME_H}:flags=lanczos"


# ---------------------------------------------------------------------------
# 6. Title card rendering
# ---------------------------------------------------------------------------

def _measure_text_width(text: str, fontsize: int) -> int:
    """
    Approximate pixel width of text at given fontsize.
    Uses a simple heuristic: average ~0.6 × fontsize per character.
    """
    return int(len(text) * fontsize * 0.6)


def _wrap_title(title: str, max_width: int, fontsize: int) -> tuple[list[str], int]:
    """
    Break title into at most 2 lines at word boundaries.
    Reduces fontsize until all lines fit within max_width.
    Never splits mid-word, never removes spaces.

    Returns (lines, final_fontsize).
    """
    words = title.split(" ")
    fs = fontsize

    while fs >= 24:
        # Try fitting on 1 line
        if _measure_text_width(title, fs) <= max_width:
            return [title], fs

        # Try splitting into 2 lines
        best_split: tuple[list[str], int] | None = None
        for i in range(1, len(words)):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])
            w1 = _measure_text_width(line1, fs)
            w2 = _measure_text_width(line2, fs)
            if w1 <= max_width and w2 <= max_width:
                # Pick the split that balances line lengths
                balance = abs(len(line1) - len(line2))
                if best_split is None or balance < best_split[1]:
                    best_split = ([line1, line2], balance)

        if best_split is not None:
            return best_split[0], fs

        fs -= 2  # reduce and retry

    # Last resort: single line at minimum size
    return [title], fs


def build_title_card_filter(title: str, duration: float) -> list[str]:
    """
    Build FFmpeg drawtext filter fragments for the title card (one per line).
    Returns a list of individual drawtext filter strings (not joined).
    """
    display_duration = min(3.0, duration)
    lines, fontsize = _wrap_title(title, MAX_TITLE_W, 72)

    line_height = int(fontsize * 1.4)
    card_top_y = SAFE_MARGIN + 60   # 60px below top safe zone

    filters: list[str] = []
    for i, line in enumerate(lines):
        y = card_top_y + i * line_height
        filters.append(
            f"drawtext=fontfile={FONT_PATH}"
            f":fontsize={fontsize}"
            f":fontcolor=black"
            f":text='{_escape_drawtext(line)}'"
            f":x=(w-text_w)/2"
            f":y={y}"
            f":box=1"
            f":boxcolor=white@0.92"
            f":boxborderw={SUB_BOX_BORDER}"
            f":enable='between(t\\,0\\,{display_duration:.3f})'"
        )

    return filters


# ---------------------------------------------------------------------------
# 7. Subtitle drawtext filters
# ---------------------------------------------------------------------------

def _escape_drawtext(text: str) -> str:
    """Escape special characters for FFmpeg drawtext text option.

    Order matters: backslash first so we don't double-escape later substitutions.
    % must be escaped as %% — FFmpeg drawtext expands %{pts} etc. at render time,
    so a bare % (e.g. "20%") corrupts the filter string and causes cascading parse
    failures that manifest as "Missing ')' in between(t".
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("%", "%%")
    text = text.replace("'", "\u2019")
    text = text.replace(":", "\\:")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    return text


def build_subtitle_filters(cards: list[SubtitleCard]) -> list[str]:
    """
    Build FFmpeg drawtext filter fragments for all subtitle cards.
    Returns a list of individual drawtext filter strings (not joined),
    so callers can chunk them into batches before passing to FFmpeg.
    """
    if not cards:
        return []

    sub_y = int(FRAME_H * SUB_Y_RATIO)
    filters: list[str] = []

    for card in cards:
        text = _escape_drawtext(card.text)
        t_start = f"{card.display_start:.3f}"
        t_end = f"{card.display_end:.3f}"
        filters.append(
            f"drawtext=fontfile={FONT_PATH}"
            f":fontsize={SUB_FONTSIZE}"
            f":fontcolor=white"
            f":text='{text}'"
            f":x=(w-text_w)/2"
            f":y={sub_y}"
            f":bordercolor=black"
            f":borderw={SUB_BORDER_W}"
            f":enable='between(t\\,{t_start}\\,{t_end})'"
        )

    return filters


# ---------------------------------------------------------------------------
# 8. Music selection
# ---------------------------------------------------------------------------

def pick_music_track() -> str | None:
    """Return a random .mp3 from the music directory, or None if empty."""
    tracks = list(MUSIC_DIR.glob("*.mp3"))
    if not tracks:
        logger.warning("No music tracks found in %s", MUSIC_DIR)
        return None
    return str(random.choice(tracks))


# ---------------------------------------------------------------------------
# 9. CTA append (fast concat)
# ---------------------------------------------------------------------------

def append_cta_fast(main_clip: str, output_path: str) -> None:
    """
    Concat main_clip + CTA outro using filter_complex concat.
    Only called after main_clip is fully written and non-empty.
    Raises if either file is missing or too small.
    """
    if not os.path.exists(main_clip):
        raise FileNotFoundError(f"Main clip not found: {main_clip}")
    if os.path.getsize(main_clip) < 100_000:
        raise ValueError(f"Main clip too small ({os.path.getsize(main_clip)} bytes): {main_clip}")
    cta_file = CTA_PREENCODED_PATH if CTA_PREENCODED_PATH.exists() else CTA_PATH
    if not cta_file.exists():
        raise FileNotFoundError(f"CTA outro not found: {cta_file}")

    _run_ffmpeg([
        "-i", main_clip,
        "-i", str(cta_file),
        "-filter_complex", "[0:v]setsar=1[v0];[1:v]setsar=1[v1];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "4.0",
        "-preset", "fast", "-crf", "18",
        "-r", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-y", output_path,
    ], desc="CTA append")


def preencode_cta() -> None:
    """
    Re-encode the CTA outro to exactly match the clip output format:
      1080×1920, H.264 baseline, AAC 192k, 30 fps, yuv420p
    so that concat demuxer can use -c copy without any re-encode at append time.
    Called once at server startup.  Skips if already done.
    """
    if CTA_PREENCODED_PATH.exists() and CTA_PREENCODED_PATH.stat().st_size > 100_000:
        logger.info("Pre-encoded CTA already exists — skipping")
        return
    if not CTA_PATH.exists():
        logger.warning("CTA source not found at %s — skipping pre-encode", CTA_PATH)
        return

    logger.info("Pre-encoding CTA outro…")
    _run_ffmpeg([
        "-i", str(CTA_PATH),
        "-vf", f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
               f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "4.0",
        "-preset", "fast", "-crf", "18",
        "-r", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-y", str(CTA_PREENCODED_PATH),
    ], desc="preencode CTA")
    logger.info(
        "CTA pre-encoded — %.1f MB",
        CTA_PREENCODED_PATH.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# 10. FFmpeg helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(args: list[str], desc: str = "") -> None:
    """Run FFmpeg with the given arguments. Raises on non-zero exit."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"] + args
    logger.info("FFmpeg [%s]: %s", desc, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace',
                            cwd=str(ROOT_DIR))
    if result.returncode != 0:
        logger.error("FFmpeg error [%s]:\n%s", desc, result.stderr)
        raise RuntimeError(f"FFmpeg failed ({desc}): {result.stderr[-800:]}")


def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Return (width, height) of a video file using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, encoding='utf-8', errors='replace',
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract mono 16kHz WAV audio — optimal for Whisper."""
    _run_ffmpeg([
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-y", audio_path,
    ], desc="extract audio")


# ---------------------------------------------------------------------------
# 11. Single clip renderer
# ---------------------------------------------------------------------------

async def render_clip(
    spec: ClipSpec,
    source_video: str,
    source_w: int,
    source_h: int,
    detections: list[dict],
    output_dir: str,
) -> str:
    """
    Render one clip to disk.  Returns the path to the final .mp4 file.

    Steps:
      a) Trim source video to clip window
      b) Apply dynamic crop → 1080×1920
      c) Burn subtitle cards
      d) Overlay title card
      e) Mix background music (ducked under speech)
      f) Append CTA outro
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _render_clip_sync, spec, source_video,
                                      source_w, source_h, detections, output_dir)


def _render_clip_sync(
    spec: ClipSpec,
    source_video: str,
    source_w: int,
    source_h: int,
    detections: list[dict],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"clip_{spec.clip_id}_")

    try:
        duration = spec.end - spec.start

        # --- Step A: trim ---
        trimmed = os.path.join(tmp_dir, "trimmed.mp4")
        _run_ffmpeg([
            "-ss", str(spec.start),
            "-to", str(spec.end),
            "-i", source_video,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-y", trimmed,
        ], desc=f"trim {spec.clip_id}")

        # --- Step B: crop to 9:16 ---
        crop_filter = build_dynamic_crop_filter(
            detections, source_w, source_h,
            spec.start, spec.end, spec.words,
        )
        cropped = os.path.join(tmp_dir, "cropped.mp4")
        _run_ffmpeg([
            "-i", trimmed,
            "-vf", crop_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-y", cropped,
        ], desc=f"crop {spec.clip_id}")

        # --- Step C+D: burn subtitles + title card ---
        text_burned = os.path.join(tmp_dir, "text.mp4")
        if SUBTITLES_ENABLED:
            # Each drawtext filter contains commas inside between(t,X,Y), so we
            # keep them as a list and chunk the list — never split a joined string.
            all_drawtext: list[str] = (
                build_subtitle_filters(spec.subtitle_cards)
                + build_title_card_filter(spec.title, duration)
            )
            _CHUNK = 10
            chunks = [all_drawtext[i:i + _CHUNK] for i in range(0, len(all_drawtext), _CHUNK)]
            if not chunks:
                chunks = [["null"]]
            logger.info(
                "Subtitle render: %d drawtext filters → %d pass(es) of ≤%d",
                len(all_drawtext), len(chunks), _CHUNK,
            )
            current_input = cropped
            for pass_idx, chunk in enumerate(chunks):
                is_last = pass_idx == len(chunks) - 1
                out = text_burned if is_last else os.path.join(tmp_dir, f"sub_{pass_idx}.mp4")
                vf = ",".join(chunk)
                logger.info("Pass %d/%d vf (full): %s", pass_idx + 1, len(chunks), vf)
                _run_ffmpeg([
                    "-i", current_input,
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "fast" if is_last else "ultrafast",
                    "-crf",  "18"    if is_last else "0",
                    "-c:a", "copy",
                    "-y", out,
                ], desc=f"text {spec.clip_id} {pass_idx + 1}/{len(chunks)}")
                current_input = out
        else:
            logger.info("Subtitles disabled (SUBTITLES_ENABLED=False) — copying cropped video")
            import shutil as _shutil
            _shutil.copy2(cropped, text_burned)

        # --- Step E: mix background music ---
        music_track = pick_music_track()
        if music_track:
            with_music = os.path.join(tmp_dir, "music.mp4")
            _run_ffmpeg([
                "-i", text_burned,
                "-stream_loop", "-1", "-i", music_track,
                "-filter_complex",
                (
                    "[0:a]volume=1.0[speech];"
                    "[1:a]volume=0.12,atrim=0:duration={dur}[music];"
                    "[speech][music]amix=inputs=2:duration=first[aout]"
                ).format(dur=duration),
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                "-y", with_music,
            ], desc=f"music {spec.clip_id}")
            pre_cta = with_music
        else:
            pre_cta = text_burned

        # --- Step F: append CTA ---
        final = os.path.join(output_dir, f"{spec.clip_id}.mp4")
        try:
            append_cta_fast(pre_cta, final)
        except Exception as exc:
            logger.warning("CTA append failed (%s), using clip without CTA: %s", spec.clip_id, exc)
            import shutil
            shutil.copy2(pre_cta, final)

        # Verify output
        if not os.path.exists(final):
            raise FileNotFoundError(f"Output file not created: {final}")
        size = os.path.getsize(final)
        if size < 100_000:
            raise ValueError(f"Output file too small ({size} bytes): {final}")

        logger.info("Clip %s rendered — %.1f MB", spec.clip_id, size / 1e6)
        return final

    finally:
        # Clean up temp dir (but not output)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 12. Main pipeline orchestrator
# ---------------------------------------------------------------------------

async def process_video_pipeline(project_id: str, db: AsyncIOMotorDatabase) -> None:
    """
    Full pipeline entry point.  Called by server.py via asyncio.create_task().

    Reads local_video_path from the project document.
    Handles both fresh runs and retries (skips clips that are already 'done').
    Updates MongoDB using the server's field schema:
      status, processing_step, processing_progress, processing_details,
      short_clips[].{id, caption, status, storage_path, error}
    """
    from storage import put_file as _put_file

    projects = db["projects"]

    async def set_progress(
        step: str,
        progress: float,
        details: str = "",
        status: str = "processing",
    ) -> None:
        await projects.update_one(
            {"id": project_id},
            {"$set": {
                "status": status,
                "processing_step": step,
                "processing_progress": progress,
                "processing_details": details,
            }},
        )

    async def update_clip(clip_id: str, **fields: Any) -> None:
        await projects.update_one(
            {"id": project_id, "short_clips.id": clip_id},
            {"$set": {f"short_clips.$.{k}": v for k, v in fields.items()}},
        )

    output_dir: str | None = None
    try:
        # ----------------------------------------------------------------
        # Load project
        # ----------------------------------------------------------------
        logger.info("Pipeline started for project %s", project_id)
        doc = await projects.find_one({"id": project_id})
        if not doc:
            logger.error("Project %s not found", project_id)
            return

        source_video_path = doc.get("local_video_path", "")
        logger.info("Video path: %s | exists: %s", source_video_path, os.path.isfile(source_video_path))
        if not source_video_path or not os.path.isfile(source_video_path):
            await set_progress(
                "error", 0,
                "Video file not found. Please re-upload.",
                status="error",
            )
            return

        # ----------------------------------------------------------------
        # Retry path: if short_clips already populated, only process pending
        # ----------------------------------------------------------------
        existing_clips: list[dict] = doc.get("short_clips", [])
        is_retry = bool(existing_clips)
        pending_clip_ids = {
            c["id"] for c in existing_clips if c.get("status") in ("pending", "error", "")
        }

        words: list[Word] = []

        if is_retry and pending_clip_ids:
            # Restore words from stored transcript_words
            stored_words = doc.get("transcript_words", [])
            words = [
                Word(
                    text=w["text"],
                    start=float(w["start"]),
                    end=float(w["end"]),
                    precise=w.get("precise", True),
                )
                for w in stored_words
            ]
            logger.info(
                "Retry mode: %d pending clips, %d words from cache",
                len(pending_clip_ids), len(words),
            )
        else:
            # ----------------------------------------------------------------
            # Step 1: Transcribe
            # ----------------------------------------------------------------
            logger.info("Step 1: starting audio extraction")
            await set_progress("transcribing", 5, "Transcription en cours…")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
                audio_path = af.name
            logger.info("Temp wav path: %s", audio_path)
            try:
                logger.info("Calling extract_audio...")
                await asyncio.get_event_loop().run_in_executor(
                    None, extract_audio, source_video_path, audio_path
                )
                logger.info("Audio extracted OK, starting transcription")
                words = await asyncio.get_event_loop().run_in_executor(
                    None, transcribe_audio, audio_path
                )
                logger.info(
                    "Transcription complete: %d words, last word ends at %.1fs",
                    len(words), words[-1].end if words else 0,
                )
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

            # Persist transcript to DB for retry reuse
            logger.info("Persisting transcript to DB (%d words)…", len(words))
            word_dicts = [
                {"text": w.text, "start": w.start, "end": w.end, "precise": w.precise}
                for w in words
            ]
            flat_transcript = " ".join(w.text for w in words)
            await projects.update_one(
                {"id": project_id},
                {"$set": {"transcript_words": word_dicts, "transcript": flat_transcript}},
            )
            logger.info("Transcript persisted to DB OK")

            # ----------------------------------------------------------------
            # Step 2: Thematic segmentation
            # ----------------------------------------------------------------
            await set_progress("segmenting_themes", 12, "Segmentation thématique…")
            themes = await asyncio.get_event_loop().run_in_executor(
                None, segment_themes_with_claude, words
            )
            if not themes:
                await set_progress("error", 0, "Aucun thème identifié.", status="error")
                return

            # ----------------------------------------------------------------
            # Step 3: Clip boundaries — one best clip per theme
            # ----------------------------------------------------------------
            await set_progress("finding_boundaries", 20, "Recherche des moments clés…")
            raw_clips = await asyncio.get_event_loop().run_in_executor(
                None, find_clip_boundaries_with_claude, themes, words
            )
            logger.info(
                "Found %d themes, generated %d clips", len(themes), len(raw_clips)
            )
            if not raw_clips:
                await set_progress("error", 0, "Aucune limite de clip trouvée.", status="error")
                return

            # ----------------------------------------------------------------
            # Step 4: Internal silence trimming (no API call)
            # ----------------------------------------------------------------
            raw_clips = trim_clip_silences(raw_clips, words)
            logger.info("Pipeline complete — %d clips after silence trimming", len(raw_clips))
            if not raw_clips:
                await set_progress("error", 0, "Tous les clips supprimés au trimming.", status="error")
                return

            # ----------------------------------------------------------------
            # Step 3: Build ClipSpec stubs and write to DB
            # ----------------------------------------------------------------
            clip_specs_map: dict[str, ClipSpec] = {}
            clip_docs: list[dict] = []

            for raw in raw_clips:
                cid = str(uuid.uuid4())[:8]
                start = float(raw["start"])
                end = float(raw["end"])
                clip_words = [w for w in words if start <= w.start <= end]
                cards = build_subtitle_cards(clip_words, clip_start=start)

                spec = ClipSpec(
                    clip_id=cid,
                    project_id=project_id,
                    start=start,
                    end=end,
                    title=raw.get("title", ""),
                    hook=raw.get("hook", ""),
                    words=clip_words,
                    subtitle_cards=cards,
                )
                clip_specs_map[cid] = spec
                clip_docs.append({
                    "id": cid,
                    "caption": raw.get("title", ""),
                    "hook": raw.get("hook", ""),
                    "start": start,
                    "end": end,
                    "status": "pending",
                    "storage_path": "",
                    "error": "",
                })

            await projects.update_one(
                {"id": project_id},
                {"$set": {"short_clips": clip_docs}},
            )
            existing_clips = clip_docs
            pending_clip_ids = {c["id"] for c in clip_docs}

        # ----------------------------------------------------------------
        # Step 4: Face detection (shared across all clips)
        # ----------------------------------------------------------------
        await set_progress("detecting_speakers", 33, "Détection des visages…")

        source_w, source_h = await asyncio.get_event_loop().run_in_executor(
            None, get_video_dimensions, source_video_path
        )
        detections = await asyncio.get_event_loop().run_in_executor(
            None, detect_speakers, source_video_path
        )

        # ----------------------------------------------------------------
        # Step 5: Rebuild ClipSpec for pending clips (retry path needs this)
        # ----------------------------------------------------------------
        if is_retry:
            clip_specs_map: dict[str, ClipSpec] = {}
            for c in existing_clips:
                if c["id"] not in pending_clip_ids:
                    continue
                start = float(c["start"])
                end = float(c["end"])
                clip_words = [w for w in words if start <= w.start <= end]
                cards = build_subtitle_cards(clip_words, clip_start=start)
                clip_specs_map[c["id"]] = ClipSpec(
                    clip_id=c["id"],
                    project_id=project_id,
                    start=start,
                    end=end,
                    title=c.get("caption", ""),
                    hook=c.get("hook", ""),
                    words=clip_words,
                    subtitle_cards=cards,
                )

        # ----------------------------------------------------------------
        # Step 6: Render and upload in parallel
        # ----------------------------------------------------------------
        await set_progress("rendering", 35, f"Rendu de {len(clip_specs_map)} clips…")

        output_dir = tempfile.mkdtemp(prefix=f"project_{project_id}_")
        semaphore = asyncio.Semaphore(MAX_PARALLEL_CLIPS)
        total = len(clip_specs_map)
        done_count = 0

        async def render_and_upload(spec: ClipSpec, idx: int) -> None:
            nonlocal done_count
            async with semaphore:
                await update_clip(spec.clip_id, status="rendering", error="")
                try:
                    local_path = await render_clip(
                        spec, source_video_path, source_w, source_h,
                        detections, output_dir,
                    )

                    # Verify before upload
                    if not os.path.exists(local_path):
                        raise FileNotFoundError(f"Rendered file missing: {local_path}")
                    if os.path.getsize(local_path) < 100_000:
                        raise ValueError(
                            f"Rendered file too small: {os.path.getsize(local_path)} bytes"
                        )

                    # Save to /workspace/clips/ on the RunPod volume
                    storage_path = f"{project_id}/{spec.clip_id}.mp4"
                    await asyncio.get_event_loop().run_in_executor(
                        None, _put_file, local_path, storage_path
                    )

                    done_count += 1
                    progress = 35 + int(done_count / total * 60)
                    await update_clip(spec.clip_id, status="done", storage_path=storage_path)
                    await set_progress(
                        "rendering", progress,
                        f"{done_count}/{total} clips prêts",
                    )
                    logger.info("Clip %d/%d done: %s", idx + 1, total, spec.clip_id)

                except Exception as exc:
                    logger.exception("Clip %s failed: %s", spec.clip_id, exc)
                    await update_clip(
                        spec.clip_id, status="error", error=str(exc)[:500]
                    )
                finally:
                    local = os.path.join(output_dir, f"{spec.clip_id}.mp4")
                    if os.path.exists(local):
                        os.unlink(local)

        await asyncio.gather(*[
            render_and_upload(spec, i)
            for i, spec in enumerate(clip_specs_map.values())
        ])

        # ----------------------------------------------------------------
        # Step 7: Final status
        # ----------------------------------------------------------------
        final_doc = await projects.find_one({"id": project_id})
        all_clips = final_doc.get("short_clips", []) if final_doc else []
        n_done = sum(1 for c in all_clips if c.get("status") == "done")
        n_err = sum(1 for c in all_clips if c.get("status") == "error")

        final_status = "done" if n_err == 0 else ("partial" if n_done > 0 else "error")
        await set_progress(
            "done" if final_status != "error" else "error",
            100,
            f"{n_done} clips prêts" + (f", {n_err} erreurs" if n_err else ""),
            status=final_status,
        )
        logger.info(
            "Project %s complete — %d done, %d errors", project_id, n_done, n_err
        )

    except Exception as exc:
        import traceback as _tb
        _debug_path = Path(__file__).parent / "pipeline_error.txt"
        with open(_debug_path, "a", encoding="utf-8") as _f:
            _f.write(f"\n=== {project_id} ===\n")
            _f.write(_tb.format_exc())
        logger.exception("Fatal error in process_video_pipeline for %s: %s", project_id, exc)
        await set_progress("error", 0, str(exc)[:300], status="error")

    finally:
        if output_dir and os.path.isdir(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
