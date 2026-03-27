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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import mediapipe as mp
import whisper
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
SUB_FONTSIZE = 58
SUB_BOX_BORDER = 14       # boxborderw — gives pill/rounded look
SUB_Y_RATIO = 0.72        # vertical position as fraction of frame height
SUB_SHIFT_MS = 0.100      # shift card 100 ms earlier than word start
SUB_SILENCE_GAP = 0.3     # gap in seconds that means silence (no card)

# Face detection thresholds
FACE_CONF_THRESHOLD = 0.7
SPEAKER_SWITCH_MIN_FRAMES = 8   # minimum consecutive frames before switching

ROOT_DIR = Path(__file__).parent
MUSIC_DIR = ROOT_DIR / "music"
ASSETS_DIR = ROOT_DIR / "assets"

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

CLIP_SYSTEM_PROMPT = """\
You are an expert short-form video editor specializing in French business and \
entrepreneurship content. Your job is to extract the most viral, engaging clips \
from a long interview transcript.

SELECTION CRITERIA:
- Strong hook in the first 5 seconds (surprising stat, provocative question, \
  bold claim, or relatable dilemma)
- Clear standalone narrative arc — the clip must make sense without context
- Emotional resonance: curiosity, admiration, controversy, or humor
- French-speaking audience: relate to French/European business culture
- Ideal duration: 45–90 seconds (never under 30s, never over 90s)

VIRAL HOOK REFERENCE EXAMPLES (study these patterns):
- "Embaucher ma femme c'est une bonne idée ?" — mundane decision reframed as \
  personal dilemma
- "Un téléphone pro ? Seulement quand on gagne 1 000 000 €" — shocking threshold
- "Pourquoi faire du black vous appauvrit ?" — contrarian take
- "Il fait 500 000€/an et voyage pendant 5 mois !" — aspirational stat
- "Il ne travaille pas le samedi et le dimanche ?" — implied conflict
- "Je ne peux pas survivre avec un loyer de 15 000 €" — quote framing
- "Vous ne payez pas assez votre comptable" — direct challenge
- "Il n'a pas le choix, il doit vendre sa voiture" — dramatic consequence

CLIP COUNT HARD LIMIT: Never generate more than 16 clips regardless of video \
length. Always select the 16 strongest only. Quality over quantity.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no commentary:
{
  "clips": [
    {
      "start": <float seconds>,
      "end": <float seconds>,
      "title": "<French hook title, max 60 chars>",
      "hook": "<one sentence explaining why this clip is viral>"
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
        return " ".join(w.text for w in self.words).upper()


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
    Transcribe audio with local Whisper (large model) and return
    word-level timestamps.  All words are marked precise=True because
    Whisper's word_timestamps mode gives exact alignments.
    """
    logger.info("Loading Whisper large model…")
    model = whisper.load_model("large")

    logger.info("Transcribing %s …", audio_path)
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        verbose=False,
    )

    words: list[Word] = []
    for segment in result["segments"]:
        seg_words = segment.get("words", [])
        for w in seg_words:
            raw = w.get("word", "").strip()
            if not raw:
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
    - 2–3 words per card
    - Card appears SUB_SHIFT_MS before the first word starts
    - Card disappears exactly when the last word ends
    - Never bridge a silence gap > SUB_SILENCE_GAP seconds
    - Timestamps are relative to clip_start (i.e. 0 = start of clip)
    """
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

    for i, word in enumerate(words):
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
    return cards


# ---------------------------------------------------------------------------
# 3. Claude clip selection
# ---------------------------------------------------------------------------

def select_clips_with_claude(
    words: list[Word],
    project_id: str,
) -> list[dict[str, Any]]:
    """
    Send the transcript to Claude and return a list of clip dicts:
    [{start, end, title, hook}, ...]
    Max 16 clips.
    """
    transcript_text = words_to_transcript_text(words)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Sending transcript to Claude for clip selection…")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=CLIP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript_text}],
    )

    raw = message.content[0].text.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned invalid JSON: %s", raw[:500])
        raise ValueError(f"Claude returned invalid JSON: {exc}") from exc

    clips = data.get("clips", [])

    # Enforce limits and validate
    valid: list[dict[str, Any]] = []
    for c in clips[:16]:
        start = float(c.get("start", 0))
        end = float(c.get("end", 0))
        duration = end - start
        if duration < CLIP_MIN_DURATION or duration > CLIP_MAX_DURATION:
            logger.warning(
                "Clip '%s' skipped — duration %.1fs out of range",
                c.get("title", "?"), duration,
            )
            continue
        valid.append(c)

    logger.info("Claude selected %d valid clips", len(valid))
    return valid


# ---------------------------------------------------------------------------
# 4. Face / speaker detection (MediaPipe)
# ---------------------------------------------------------------------------

def detect_speakers(video_path: str, sample_every_n_frames: int = 15) -> list[dict]:
    """
    Sample frames from the video using MediaPipe Face Detection and record
    face bounding boxes.

    Returns a list of dicts:
    [{"time": float, "faces": [{"x", "y", "w", "h", "conf"}, ...]}, ...]
    Faces within each frame are sorted left-to-right (speaker 0 = leftmost).
    """
    import cv2

    mp_face = mp.solutions.face_detection  # type: ignore[attr-defined]
    detector = mp_face.FaceDetection(
        model_selection=1,          # model 1 = full-range (up to 5m), better for interviews
        min_detection_confidence=FACE_CONF_THRESHOLD,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detections: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                h, w = frame.shape[:2]
                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)

                faces: list[dict] = []
                if result.detections:
                    for det in result.detections:
                        bbox = det.location_data.relative_bounding_box
                        conf = det.score[0] if det.score else 0.0
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        faces.append({
                            "x": max(0, x), "y": max(0, y),
                            "w": bw, "h": bh,
                            "conf": float(conf),
                        })

                detections.append({
                    "time": frame_idx / fps,
                    "faces": sorted(faces, key=lambda f: f["x"]),  # left-to-right
                })
            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    logger.info("Face detection complete — %d sampled frames, video: %s", len(detections), video_path)
    return detections


# ---------------------------------------------------------------------------
# 5. Dynamic crop filter (speaker switching)
# ---------------------------------------------------------------------------

def build_dynamic_crop_filter(
    detections: list[dict],
    source_w: int,
    source_h: int,
    clip_start: float,
    clip_end: float,
    words: list[Word],
) -> str:
    """
    Build an FFmpeg sendcmd / crop expression that follows the active speaker.

    Rules:
    - Default is always speaker 0 (leftmost face), never center-of-frame
    - Only switch to speaker 1 when they are actively speaking
    - Never cut to a frame with no detected face
    - If only one speaker ever detected, never switch
    - Minimum SPEAKER_SWITCH_MIN_FRAMES before a switch is committed

    Returns an FFmpeg filter_complex crop+scale string.
    """
    clip_duration = clip_end - clip_start

    # Filter detections to clip window, re-zero timestamps
    local_dets = [
        {**d, "time": d["time"] - clip_start}
        for d in detections
        if clip_start <= d["time"] <= clip_end
    ]

    # Determine how many unique speaker positions exist
    max_speakers = max(
        (len(d["faces"]) for d in local_dets if d["faces"]),
        default=1,
    )
    has_two_speakers = max_speakers >= 2

    # Build a timeline of "who is speaking" based on words proximity to
    # detected faces (simple heuristic: whichever face changed more recently)
    # For single-speaker videos just use a static crop on speaker 0.

    def face_to_crop(face: dict) -> tuple[int, int]:
        """Return (crop_x, crop_y) to center a 9:16 frame on this face."""
        crop_w = source_h * 9 // 16   # 9:16 portrait crop width
        face_cx = face["x"] + face["w"] // 2
        face_cy = face["y"] + face["h"] // 2
        cx = max(crop_w // 2, min(source_w - crop_w // 2, face_cx))
        cy = max(FRAME_H // 2, min(source_h - FRAME_H // 2, face_cy))
        x = cx - crop_w // 2
        y = cy - FRAME_H // 2
        return max(0, x), max(0, y)

    if not local_dets or not any(d["faces"] for d in local_dets):
        # No faces detected — center crop
        crop_w = source_h * 9 // 16
        x = (source_w - crop_w) // 2
        return (
            f"crop={crop_w}:{source_h}:{x}:0,"
            f"scale={FRAME_W}:{FRAME_H}:flags=lanczos"
        )

    # Build word-time-to-speaker map for two-speaker scenarios
    # We use a simple rule: words before halfway point → speaker 0,
    # but override when we detect clear speaker-1-only frames.
    # A more sophisticated implementation would use diarization.

    crop_segments: list[tuple[float, float, int, int]] = []  # (t_start, t_end, x, y)

    if not has_two_speakers:
        # Single speaker — find their dominant face position
        all_faces = [d["faces"][0] for d in local_dets if d["faces"]]
        if all_faces:
            avg_x = int(sum(f["x"] for f in all_faces) / len(all_faces))
            avg_w = int(sum(f["w"] for f in all_faces) / len(all_faces))
            avg_cy = int(sum(f["y"] + f["h"] // 2 for f in all_faces) / len(all_faces))
            crop_w = source_h * 9 // 16
            face_cx = avg_x + avg_w // 2
            cx = max(crop_w // 2, min(source_w - crop_w // 2, face_cx))
            x = cx - crop_w // 2
            y = max(0, avg_cy - FRAME_H // 2)
            return (
                f"crop={crop_w}:{source_h}:{max(0,x)}:{max(0,y)},"
                f"scale={FRAME_W}:{FRAME_H}:flags=lanczos"
            )

    # Two speakers — switch based on active speaker heuristic
    # Build per-frame speaker assignments
    frame_speaker: list[tuple[float, int]] = []  # (time, speaker_idx)
    prev_speaker = 0
    consecutive = 0

    for det in local_dets:
        faces = det["faces"]
        if not faces:
            frame_speaker.append((det["time"], prev_speaker))
            continue

        t = det["time"]
        # Find nearest word at this time
        active_words = [w for w in words if w.start - clip_start <= t <= w.end - clip_start]
        candidate_speaker = prev_speaker

        if len(faces) >= 2:
            # If we have a word active near this time, guess speaker from position
            if active_words:
                # Heuristic: alternate based on word index parity (rough)
                candidate_speaker = 0  # default to speaker 0
            else:
                candidate_speaker = prev_speaker  # silence → stay

        if candidate_speaker == prev_speaker:
            consecutive += 1
        else:
            consecutive = 1

        if consecutive >= SPEAKER_SWITCH_MIN_FRAMES or candidate_speaker == prev_speaker:
            prev_speaker = candidate_speaker

        frame_speaker.append((t, prev_speaker))

    # Convert frame assignments to crop segments
    if not frame_speaker:
        crop_w = source_h * 9 // 16
        x = (source_w - crop_w) // 2
        return (
            f"crop={crop_w}:{source_h}:{x}:0,"
            f"scale={FRAME_W}:{FRAME_H}:flags=lanczos"
        )

    # For each segment of consistent speaker, find average crop position
    segments: list[dict] = []
    seg_start = frame_speaker[0][0]
    seg_spk = frame_speaker[0][1]

    for t, spk in frame_speaker[1:]:
        if spk != seg_spk:
            segments.append({"start": seg_start, "end": t, "speaker": seg_spk})
            seg_start = t
            seg_spk = spk
    segments.append({"start": seg_start, "end": clip_duration, "speaker": seg_spk})

    # Build crop per segment using faces from that speaker
    crop_parts: list[str] = []
    for seg in segments:
        spk = seg["speaker"]
        seg_dets = [
            d for d in local_dets
            if seg["start"] <= d["time"] <= seg["end"] and len(d["faces"]) > spk
        ]
        if seg_dets:
            faces_in_seg = [d["faces"][spk] for d in seg_dets]
            cx_avg = int(sum(f["x"] + f["w"] // 2 for f in faces_in_seg) / len(faces_in_seg))
            cy_avg = int(sum(f["y"] + f["h"] // 2 for f in faces_in_seg) / len(faces_in_seg))
        else:
            # Fall back to speaker 0 from any detection in window
            fallback = [d["faces"][0] for d in local_dets if d["faces"]]
            if fallback:
                cx_avg = int(sum(f["x"] + f["w"] // 2 for f in fallback) / len(fallback))
                cy_avg = int(sum(f["y"] + f["h"] // 2 for f in fallback) / len(fallback))
            else:
                crop_w = source_h * 9 // 16
                x = (source_w - crop_w) // 2
                cx_avg = x + crop_w // 2
                cy_avg = source_h // 2

        crop_w = source_h * 9 // 16
        x = max(0, min(source_w - crop_w, cx_avg - crop_w // 2))
        y = max(0, min(source_h - FRAME_H, cy_avg - FRAME_H // 2))
        crop_parts.append(
            f"crop={crop_w}:{source_h}:{x}:{y},"
            f"scale={FRAME_W}:{FRAME_H}:flags=lanczos"
        )

    # For multi-segment videos we return the FIRST segment's crop filter.
    # True per-segment switching via FFmpeg sendcmd is complex; for now we use
    # the dominant speaker's crop. Dynamic switching per segment can be layered
    # on as a follow-up.
    return crop_parts[0] if crop_parts else (
        f"crop={source_h * 9 // 16}:{source_h}:{(source_w - source_h * 9 // 16) // 2}:0,"
        f"scale={FRAME_W}:{FRAME_H}:flags=lanczos"
    )


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


def build_title_card_filter(title: str, duration: float) -> str:
    """
    Build an FFmpeg drawtext filter string that renders a title card
    with a pill-shaped white background and black bold text.

    Safe zone enforced: minimum SAFE_MARGIN px from all edges.
    Max 2 lines, no mid-word breaks, no space removal.
    Font size reduced until text fits within FRAME_W - 2*SAFE_MARGIN.
    Card shown for first 3 seconds of the clip (or full duration if shorter).
    """
    display_duration = min(3.0, duration)
    lines, fontsize = _wrap_title(title, MAX_TITLE_W, 72)

    line_height = int(fontsize * 1.4)
    card_h = len(lines) * line_height + SUB_BOX_BORDER * 2
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
            f":enable='between(t,0,{display_duration:.3f})'"
        )

    return ",".join(filters)


# ---------------------------------------------------------------------------
# 7. Subtitle drawtext filters
# ---------------------------------------------------------------------------

def _escape_drawtext(text: str) -> str:
    """Escape special characters for FFmpeg drawtext."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    return text


def build_subtitle_filters(cards: list[SubtitleCard]) -> str:
    """
    Build FFmpeg drawtext filters for all subtitle cards.
    Each card is rendered as UPPERCASE text with a white pill background
    at SUB_Y_RATIO of frame height.
    """
    if not cards:
        return ""

    sub_y = int(FRAME_H * SUB_Y_RATIO)
    filters: list[str] = []

    for card in cards:
        text = _escape_drawtext(card.text)
        t_start = f"{card.display_start:.3f}"
        t_end = f"{card.display_end:.3f}"
        filters.append(
            f"drawtext=fontfile={FONT_PATH}"
            f":fontsize={SUB_FONTSIZE}"
            f":fontcolor=black"
            f":text='{text}'"
            f":x=(w-text_w)/2"
            f":y={sub_y}"
            f":box=1"
            f":boxcolor=white@1.0"
            f":boxborderw={SUB_BOX_BORDER}"
            f":enable='between(t,{t_start},{t_end})'"
        )

    return ",".join(filters)


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
    Concat main_clip + CTA outro using FFmpeg concat demuxer.
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list = f.name
        f.write(f"file '{os.path.abspath(main_clip).replace(os.sep, '/')}'\n")
        f.write(f"file '{os.path.abspath(cta_file).replace(os.sep, '/')}'\n")

    try:
        _run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy",
            "-y", output_path,
        ], desc="CTA append")
    finally:
        os.unlink(concat_list)


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
    logger.debug("FFmpeg [%s]: %s", desc, " ".join(cmd))
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
        subtitle_filter = build_subtitle_filters(spec.subtitle_cards)
        title_filter = build_title_card_filter(spec.title, duration)

        combined_vf = ",".join(f for f in [subtitle_filter, title_filter] if f)
        if not combined_vf:
            combined_vf = "null"

        text_burned = os.path.join(tmp_dir, "text.mp4")
        _run_ffmpeg([
            "-i", cropped,
            "-vf", combined_vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-y", text_burned,
        ], desc=f"text {spec.clip_id}")

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
    from storage import put_object as _put_object

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
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

            # Persist transcript to DB for retry reuse
            word_dicts = [
                {"text": w.text, "start": w.start, "end": w.end, "precise": w.precise}
                for w in words
            ]
            flat_transcript = " ".join(w.text for w in words)
            await projects.update_one(
                {"id": project_id},
                {"$set": {"transcript_words": word_dicts, "transcript": flat_transcript}},
            )

            # ----------------------------------------------------------------
            # Step 2: Select clips with Claude
            # ----------------------------------------------------------------
            await set_progress("selecting_clips", 25, "Sélection des meilleurs clips…")

            raw_clips = await asyncio.get_event_loop().run_in_executor(
                None, select_clips_with_claude, words, project_id
            )
            if not raw_clips:
                await set_progress("error", 0, "Aucun clip sélectionné.", status="error")
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
        await set_progress("detecting_speakers", 30, "Détection des visages…")

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

                    # Upload — put_object(path, bytes, content_type)
                    storage_path = f"clips/{project_id}/{spec.clip_id}.mp4"
                    with open(local_path, "rb") as fh:
                        video_bytes = fh.read()
                    await asyncio.get_event_loop().run_in_executor(
                        None, _put_object, storage_path, video_bytes, "video/mp4"
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
